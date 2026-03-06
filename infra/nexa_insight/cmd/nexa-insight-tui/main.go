package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

type options struct {
	refresh  time.Duration
	runsRoot string
}

type gpuRow struct {
	Index   string
	Name    string
	UtilGPU int
	MemUsed int
	MemTot  int
	TempC   int
	PowerW  int
}

type hostRow struct {
	Hostname     string
	Kernel       string
	Uptime       string
	Load1        string
	Load5        string
	Load15       string
	MemUsedPct   float64
	MemUsedMB    int
	MemTotalMB   int
	CPUUtilPct   float64
	CPUCores     int
	NetRxMBps    float64
	NetTxMBps    float64
	ProcessCount int
}

type mlRow struct {
	RunID              string
	SamplesPerSec      float64
	StepsPerSec        float64
	TokensPerSec       float64
	LossFinal          float64
	GPUUtilMean        float64
	IdleGapMSMean      float64
	TrainRuntimeSec    float64
	LivePercent        int
	LiveItPerSec       float64
	LiveExamplesPerSec float64
	ProgressLine       string
	HasFinal           bool
}

type snapshot struct {
	timestamp time.Time
	host      hostRow
	gpus      []gpuRow
	ml        mlRow
	err       string
}

type tickMsg struct {
	s snapshot
}

type model struct {
	opts   options
	width  int
	height int
	data   snapshot

	smoothCPU    float64
	smoothGPU    float64
	smoothSample float64
	smoothToken  float64
}

var (
	rePct      = regexp.MustCompile(`(\d{1,3})%`)
	reItRate   = regexp.MustCompile(`([0-9]+(?:\.[0-9]+)?)\s*it/s`)
	reExRate   = regexp.MustCompile(`([0-9]+(?:\.[0-9]+)?)\s*examples/s`)
	reProcStep = regexp.MustCompile(`(\d+)\/(\d+)`)
)

var (
	prevCPUTotal uint64
	prevCPUIdle  uint64
	haveCPUPrev  bool

	prevNetRx uint64
	prevNetTx uint64
	prevNetTS time.Time
	haveNet   bool
)

func run(name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", err
	}
	return string(out), nil
}

func parseInt(s string) int {
	s = strings.TrimSpace(s)
	v, _ := strconv.Atoi(s)
	return v
}

func parseFloat(s string) float64 {
	s = strings.TrimSpace(s)
	v, _ := strconv.ParseFloat(s, 64)
	return v
}

func collectHost() (hostRow, error) {
	hostname, _ := os.Hostname()
	kernel, _ := run("uname", "-r")
	upRaw, _ := os.ReadFile("/proc/uptime")
	loadRaw, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		return hostRow{}, err
	}
	loadFields := strings.Fields(string(loadRaw))
	if len(loadFields) < 3 {
		return hostRow{}, errors.New("invalid /proc/loadavg")
	}

	memRaw, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return hostRow{}, err
	}
	totalKB, availKB := 0, 0
	for _, ln := range strings.Split(string(memRaw), "\n") {
		if strings.HasPrefix(ln, "MemTotal:") {
			f := strings.Fields(ln)
			if len(f) >= 2 {
				totalKB = parseInt(f[1])
			}
		}
		if strings.HasPrefix(ln, "MemAvailable:") {
			f := strings.Fields(ln)
			if len(f) >= 2 {
				availKB = parseInt(f[1])
			}
		}
	}
	if totalKB == 0 {
		return hostRow{}, errors.New("MemTotal missing")
	}
	usedKB := totalKB - availKB

	cpuPct := readCPUUtil()
	rxMBps, txMBps := readNetRateMBps()
	procCount := readProcessCount()

	return hostRow{
		Hostname:     hostname,
		Kernel:       strings.TrimSpace(kernel),
		Uptime:       humanUptime(upRaw),
		Load1:        loadFields[0],
		Load5:        loadFields[1],
		Load15:       loadFields[2],
		MemUsedPct:   (float64(usedKB) / float64(totalKB)) * 100.0,
		MemUsedMB:    usedKB / 1024,
		MemTotalMB:   totalKB / 1024,
		CPUUtilPct:   cpuPct,
		CPUCores:     runtime.NumCPU(),
		NetRxMBps:    rxMBps,
		NetTxMBps:    txMBps,
		ProcessCount: procCount,
	}, nil
}

func humanUptime(raw []byte) string {
	fields := strings.Fields(string(raw))
	if len(fields) == 0 {
		return "unknown"
	}
	sec := parseFloat(fields[0])
	if sec <= 0 {
		return "unknown"
	}
	total := int(sec)
	d := total / 86400
	h := (total % 86400) / 3600
	m := (total % 3600) / 60
	if d > 0 {
		return fmt.Sprintf("%dd %02dh %02dm", d, h, m)
	}
	return fmt.Sprintf("%02dh %02dm", h, m)
}

func readCPUUtil() float64 {
	b, err := os.ReadFile("/proc/stat")
	if err != nil {
		return 0
	}
	lines := strings.Split(string(b), "\n")
	if len(lines) == 0 {
		return 0
	}
	fields := strings.Fields(lines[0])
	if len(fields) < 8 || fields[0] != "cpu" {
		return 0
	}
	vals := make([]uint64, 0, 8)
	for i := 1; i <= 8; i++ {
		v, err := strconv.ParseUint(fields[i], 10, 64)
		if err != nil {
			return 0
		}
		vals = append(vals, v)
	}
	total := uint64(0)
	for _, v := range vals {
		total += v
	}
	idle := vals[3] + vals[4]
	if !haveCPUPrev {
		prevCPUTotal = total
		prevCPUIdle = idle
		haveCPUPrev = true
		return 0
	}
	deltaTotal := total - prevCPUTotal
	deltaIdle := idle - prevCPUIdle
	prevCPUTotal = total
	prevCPUIdle = idle
	if deltaTotal == 0 {
		return 0
	}
	busy := float64(deltaTotal-deltaIdle) / float64(deltaTotal)
	if busy < 0 {
		busy = 0
	}
	if busy > 1 {
		busy = 1
	}
	return busy * 100.0
}

func readNetRateMBps() (float64, float64) {
	b, err := os.ReadFile("/proc/net/dev")
	if err != nil {
		return 0, 0
	}
	var rx, tx uint64
	for _, ln := range strings.Split(string(b), "\n") {
		ln = strings.TrimSpace(ln)
		if ln == "" || !strings.Contains(ln, ":") {
			continue
		}
		parts := strings.SplitN(ln, ":", 2)
		iface := strings.TrimSpace(parts[0])
		if iface == "lo" {
			continue
		}
		f := strings.Fields(strings.TrimSpace(parts[1]))
		if len(f) < 16 {
			continue
		}
		rx += parseUint64(f[0])
		tx += parseUint64(f[8])
	}
	now := time.Now()
	if !haveNet {
		prevNetRx, prevNetTx, prevNetTS = rx, tx, now
		haveNet = true
		return 0, 0
	}
	dt := now.Sub(prevNetTS).Seconds()
	if dt <= 0 {
		return 0, 0
	}
	drx := float64(rx - prevNetRx)
	dtx := float64(tx - prevNetTx)
	prevNetRx, prevNetTx, prevNetTS = rx, tx, now
	return drx / (1024.0 * 1024.0) / dt, dtx / (1024.0 * 1024.0) / dt
}

func parseUint64(s string) uint64 {
	v, _ := strconv.ParseUint(strings.TrimSpace(s), 10, 64)
	return v
}

func readProcessCount() int {
	ents, err := os.ReadDir("/proc")
	if err != nil {
		return 0
	}
	count := 0
	for _, e := range ents {
		if e.IsDir() && isDigits(e.Name()) {
			count++
		}
	}
	return count
}

func isDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

func collectGPUs() ([]gpuRow, error) {
	out, err := run("nvidia-smi",
		"--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
		"--format=csv,noheader,nounits")
	if err != nil {
		return nil, err
	}
	rows := []gpuRow{}
	for _, ln := range strings.Split(strings.TrimSpace(out), "\n") {
		if strings.TrimSpace(ln) == "" {
			continue
		}
		p := strings.Split(ln, ",")
		if len(p) < 7 {
			continue
		}
		for i := range p {
			p[i] = strings.TrimSpace(p[i])
		}
		rows = append(rows, gpuRow{
			Index:   p[0],
			Name:    p[1],
			UtilGPU: parseInt(p[2]),
			MemUsed: parseInt(p[3]),
			MemTot:  parseInt(p[4]),
			TempC:   parseInt(p[5]),
			PowerW:  parseInt(p[6]),
		})
	}
	return rows, nil
}

func readLatestTrainMetrics(runsRoot string) (mlRow, error) {
	runID, metricPath, liveLogPath, err := findLatestRunArtifacts(runsRoot)
	if err != nil {
		return mlRow{}, nil
	}

	ml := mlRow{RunID: runID}
	active := false
	if liveLogPath != "" {
		if st, err := os.Stat(liveLogPath); err == nil {
			active = time.Since(st.ModTime()) <= 20*time.Second
		}
	}
	if metricPath != "" {
		b, err := os.ReadFile(metricPath)
		if err == nil {
			var raw map[string]any
			if json.Unmarshal(b, &raw) == nil {
				ml.SamplesPerSec = castFloat(raw["samples_per_sec"])
				ml.StepsPerSec = castFloat(raw["steps_per_sec"])
				ml.TokensPerSec = castFloat(raw["tokens_per_sec"])
				ml.LossFinal = castFloat(raw["loss_final"])
				ml.GPUUtilMean = castFloat(raw["gpu_util_mean"])
				ml.IdleGapMSMean = castFloat(raw["idle_gap_ms_mean"])
				ml.TrainRuntimeSec = castFloat(raw["train_runtime_sec"])
				ml.HasFinal = true
			}
		}
	}

	if liveLogPath != "" {
		line, pct, itRate, exRate := latestProgress(liveLogPath)
		ml.ProgressLine = line
		ml.LivePercent = pct
		ml.LiveItPerSec = itRate
		ml.LiveExamplesPerSec = exRate
		if exRate > 0 {
			ml.SamplesPerSec = exRate
		}
		if itRate > 0 {
			ml.StepsPerSec = itRate
		}
	}

	if ml.ProgressLine == "" {
		ml.ProgressLine = "progress line unavailable"
	}
	if active {
		ml.HasFinal = false
		return ml, nil
	}
	// If no live activity recently, show an idle zeroed state.
	return mlRow{
		RunID:        runID,
		ProgressLine: "idle",
		LivePercent:  0,
	}, nil
}

func findLatestRunArtifacts(runsRoot string) (string, string, string, error) {
	entries, err := os.ReadDir(runsRoot)
	if err != nil {
		return "", "", "", err
	}
	type runStamp struct {
		runID string
		mod   time.Time
	}
	latestByRun := map[string]time.Time{}
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() {
			p := filepath.Join(runsRoot, name, "train_metrics.json")
			st, err := os.Stat(p)
			if err == nil {
				latestByRun[name] = maxTime(latestByRun[name], st.ModTime())
			}
			continue
		}
		if strings.HasSuffix(name, ".live.log") {
			runID := strings.TrimSuffix(name, ".live.log")
			st, err := os.Stat(filepath.Join(runsRoot, name))
			if err == nil {
				latestByRun[runID] = maxTime(latestByRun[runID], st.ModTime())
			}
		}
	}
	if len(latestByRun) == 0 {
		return "", "", "", errors.New("no run artifacts found")
	}
	cands := make([]runStamp, 0, len(latestByRun))
	for k, v := range latestByRun {
		cands = append(cands, runStamp{runID: k, mod: v})
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].mod.After(cands[j].mod) })
	runID := cands[0].runID

	metricPath := filepath.Join(runsRoot, runID, "train_metrics.json")
	if _, err := os.Stat(metricPath); err != nil {
		metricPath = ""
	}
	liveLogPath := filepath.Join(runsRoot, runID+".live.log")
	if _, err := os.Stat(liveLogPath); err != nil {
		liveLogPath = ""
	}
	return runID, metricPath, liveLogPath, nil
}

func maxTime(a, b time.Time) time.Time {
	if b.After(a) {
		return b
	}
	return a
}

func castFloat(v any) float64 {
	switch t := v.(type) {
	case float64:
		return t
	case string:
		return parseFloat(t)
	case int:
		return float64(t)
	default:
		return 0
	}
}

func latestProgress(logPath string) (string, int, float64, float64) {
	b, err := readLogTail(logPath, 96*1024)
	if err != nil {
		return "no live log", 0, 0, 0
	}
	lines := strings.FieldsFunc(string(b), func(r rune) bool { return r == '\n' || r == '\r' })
	for i := len(lines) - 1; i >= 0; i-- {
		l := strings.TrimSpace(lines[i])
		if l == "" {
			continue
		}
		if strings.Contains(l, "train:") || strings.Contains(l, "Map:") || strings.Contains(l, "%|") || reProcStep.MatchString(l) {
			pct := extractInt(rePct, l)
			itRate := extractFloat(reItRate, l)
			exRate := extractFloat(reExRate, l)
			return truncate(l, 140), pct, itRate, exRate
		}
	}
	return "progress line unavailable", 0, 0, 0
}

func readLogTail(path string, maxBytes int64) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := st.Size()
	start := int64(0)
	if size > maxBytes {
		start = size - maxBytes
	}
	if _, err := f.Seek(start, io.SeekStart); err != nil {
		return nil, err
	}
	return io.ReadAll(f)
}

func extractInt(re *regexp.Regexp, s string) int {
	m := re.FindStringSubmatch(s)
	if len(m) < 2 {
		return 0
	}
	return parseInt(m[1])
}

func extractFloat(re *regexp.Regexp, s string) float64 {
	m := re.FindStringSubmatch(s)
	if len(m) < 2 {
		return 0
	}
	return parseFloat(m[1])
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	if n < 3 {
		return s[:n]
	}
	return s[:n-3] + "..."
}

func collect(opts options) snapshot {
	s := snapshot{timestamp: time.Now().UTC()}
	h, err := collectHost()
	if err == nil {
		s.host = h
	} else {
		s.err = err.Error()
	}
	g, err := collectGPUs()
	if err == nil {
		s.gpus = g
	} else if s.err == "" {
		s.err = "nvidia-smi unavailable"
	}
	ml, err := readLatestTrainMetrics(opts.runsRoot)
	if err == nil {
		s.ml = ml
	} else if s.err == "" {
		s.err = "ml metrics unavailable"
	}
	return s
}

func tickCmd(opts options) tea.Cmd {
	return tea.Tick(opts.refresh, func(time.Time) tea.Msg {
		return tickMsg{s: collect(opts)}
	})
}

func (m model) Init() tea.Cmd {
	return tickCmd(m.opts)
}

func avgGPUUtil(gpus []gpuRow) float64 {
	if len(gpus) == 0 {
		return 0
	}
	t := 0
	for _, g := range gpus {
		t += g.UtilGPU
	}
	return float64(t) / float64(len(gpus))
}

func ema(prev, cur float64) float64 {
	if prev == 0 {
		return cur
	}
	return (prev * 0.82) + (cur * 0.18)
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "q", "ctrl+c":
			return m, tea.Quit
		case "r":
			m.data = collect(m.opts)
			return m, tickCmd(m.opts)
		}
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
	case tickMsg:
		m.data = msg.s
		if msg.s.ml.ProgressLine == "idle" {
			m.smoothCPU = 0
			m.smoothGPU = 0
			m.smoothSample = 0
			m.smoothToken = 0
		} else {
			m.smoothCPU = ema(m.smoothCPU, msg.s.host.CPUUtilPct)
			m.smoothGPU = ema(m.smoothGPU, avgGPUUtil(msg.s.gpus))
			m.smoothSample = ema(m.smoothSample, msg.s.ml.SamplesPerSec)
			m.smoothToken = ema(m.smoothToken, msg.s.ml.TokensPerSec)
		}
		return m, tickCmd(m.opts)
	}
	return m, nil
}

func cyberStyles() (lipgloss.Style, lipgloss.Style, lipgloss.Style, lipgloss.Style, lipgloss.Style, lipgloss.Style) {
	bg := lipgloss.Color("#080912")
	neonPink := lipgloss.Color("#FF2DA6")
	neonCyan := lipgloss.Color("#00F0FF")
	neonGreen := lipgloss.Color("#39FF88")
	neonAmber := lipgloss.Color("#FFC857")
	text := lipgloss.Color("#D6DEEB")

	title := lipgloss.NewStyle().Foreground(neonCyan).Background(bg).Bold(true).Padding(0, 1)
	card := lipgloss.NewStyle().Foreground(text).Background(bg).Border(lipgloss.RoundedBorder()).BorderForeground(neonPink).Padding(0, 1)
	label := lipgloss.NewStyle().Foreground(neonGreen).Bold(true)
	warn := lipgloss.NewStyle().Foreground(neonAmber).Bold(true)
	accent := lipgloss.NewStyle().Foreground(neonCyan).Bold(true)
	foot := lipgloss.NewStyle().Foreground(neonPink)
	return title, card, label, warn, accent, foot
}

func clampPct(v float64) int {
	if v < 0 {
		return 0
	}
	if v > 100 {
		return 100
	}
	return int(v)
}

func utilBar(pct int, width int) string {
	if pct < 0 {
		pct = 0
	}
	if pct > 100 {
		pct = 100
	}
	if width < 8 {
		width = 8
	}
	fill := int((float64(pct) / 100.0) * float64(width))
	if fill > width {
		fill = width
	}
	return strings.Repeat("█", fill) + strings.Repeat("░", width-fill)
}

func kv(key string, value string) string {
	return fmt.Sprintf("%-16s : %s", key, value)
}

func renderGPULine(g gpuRow, innerW int) string {
	memPct := 0
	if g.MemTot > 0 {
		memPct = int((float64(g.MemUsed) / float64(g.MemTot)) * 100)
	}
	if innerW < 90 {
		return fmt.Sprintf("GPU%-2s util=%3d%% mem=%d/%dMB", g.Index, g.UtilGPU, g.MemUsed, g.MemTot)
	}
	nameW := 22
	barW := 18
	if innerW > 130 {
		barW = min(36, innerW-108)
		nameW = min(30, innerW-96-barW)
	}
	if nameW < 10 {
		nameW = 10
	}
	if barW < 8 {
		barW = 8
	}
	return fmt.Sprintf("GPU%-2s %-*s util=%3d%% %s mem=%6d/%6dMB (%3d%%) temp=%3dC pwr=%4dW",
		g.Index, nameW, truncate(g.Name, nameW), g.UtilGPU, utilBar(g.UtilGPU, barW), g.MemUsed, g.MemTot, memPct, g.TempC, g.PowerW)
}

func runProgressPct(ml mlRow) int {
	if ml.LivePercent > 0 {
		return ml.LivePercent
	}
	if ml.HasFinal {
		return 100
	}
	return 0
}

func (m model) View() string {
	title, card, label, warn, accent, foot := cyberStyles()
	s := m.data
	cardW := max(96, m.width-2)
	midW := max(100, m.width-2)
	innerW := max(40, cardW-6)
	midInnerW := max(44, midW-6)

	head := title.Render(" NEXA INSIGHT :: CYBERPUNK TRAIN OPS ")
	sub := fmt.Sprintf("UTC %s | refresh %s | q quit | r refresh", s.timestamp.Format(time.RFC3339), m.opts.refresh)

	telemetry := []string{
		label.Render("Telemetry"),
		kv("node", s.host.Hostname),
		kv("kernel", s.host.Kernel),
		kv("os/arch", runtime.GOOS+"/"+runtime.GOARCH),
		kv("uptime", s.host.Uptime),
		kv("processes", fmt.Sprintf("%d", s.host.ProcessCount)),
		kv("cpu cores", fmt.Sprintf("%d", s.host.CPUCores)),
		kv("cpu util", fmt.Sprintf("%.1f%%", s.host.CPUUtilPct)),
		kv("load", fmt.Sprintf("%s  %s  %s", s.host.Load1, s.host.Load5, s.host.Load15)),
		kv("memory", fmt.Sprintf("%d/%d MB (%.1f%%)", s.host.MemUsedMB, s.host.MemTotalMB, s.host.MemUsedPct)),
		kv("network", fmt.Sprintf("rx %.2f MB/s  tx %.2f MB/s", s.host.NetRxMBps, s.host.NetTxMBps)),
	}

	wideBar := max(28, midInnerW-8)
	cpuGPU := []string{
		label.Render("CPU + GPU Util"),
		fmt.Sprintf("%-16s : %3d%%", "CPU", clampPct(m.smoothCPU)),
		utilBar(clampPct(m.smoothCPU), wideBar),
		"",
		fmt.Sprintf("%-16s : %3d%%", "GPU(avg)", clampPct(m.smoothGPU)),
		utilBar(clampPct(m.smoothGPU), wideBar),
		"",
	}
	if len(s.gpus) == 0 {
		cpuGPU = append(cpuGPU, warn.Render("no GPU telemetry"))
	} else {
		for _, g := range s.gpus {
			cpuGPU = append(cpuGPU, renderGPULine(g, midInnerW))
		}
	}

	ml := []string{
		label.Render("ML Metrics"),
		kv("run", s.ml.RunID),
		kv("samples/s", fmt.Sprintf("%.1f (smoothed %.1f)", s.ml.SamplesPerSec, m.smoothSample)),
		kv("tokens/s", fmt.Sprintf("%.1f (smoothed %.1f)", s.ml.TokensPerSec, m.smoothToken)),
		kv("steps/s", fmt.Sprintf("%.2f", s.ml.StepsPerSec)),
		kv("live it/s", fmt.Sprintf("%.2f", s.ml.LiveItPerSec)),
		kv("live ex/s", fmt.Sprintf("%.2f", s.ml.LiveExamplesPerSec)),
		kv("loss", fmt.Sprintf("%.4f", s.ml.LossFinal)),
		kv("gpu util mean", fmt.Sprintf("%.1f%%", s.ml.GPUUtilMean)),
		kv("idle gap mean", fmt.Sprintf("%.2fms", s.ml.IdleGapMSMean)),
		kv("runtime", fmt.Sprintf("%.2fs", s.ml.TrainRuntimeSec)),
	}

	progressPct := runProgressPct(s.ml)
	progressWidth := max(24, innerW-14)
	progress := fmt.Sprintf("%3d%% [%s]", progressPct, utilBar(progressPct, progressWidth))
	progressDetail := truncate(s.ml.ProgressLine, cardW-4)
	bottom := accent.Render("tqdm") + " " + progress + "\n" + progressDetail

	body := lipgloss.JoinVertical(lipgloss.Left,
		card.Width(cardW).Render(strings.Join(telemetry, "\n")),
		card.Width(midW).Render(strings.Join(cpuGPU, "\n")),
		card.Width(cardW).Render(strings.Join(ml, "\n")),
		card.Width(cardW).Render(bottom),
	)

	errLine := ""
	if s.err != "" {
		errLine = warn.Render("warning: " + s.err)
	}

	return lipgloss.JoinVertical(lipgloss.Left, head, sub, errLine, body, foot.Render("Nexa Insight TUI: Bubble Tea + Lip Gloss")) + "\n"
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func main() {
	refresh := flag.Duration("refresh", 2*time.Second, "refresh interval")
	runsRoot := flag.String("runs-root", "benchmark/remote_results/runpod_h100_8x/campaign_v5", "root with run directories and live logs")
	flag.Parse()

	m := model{
		opts: options{
			refresh:  *refresh,
			runsRoot: *runsRoot,
		},
		data: snapshot{
			timestamp: time.Now().UTC(),
			ml:        mlRow{ProgressLine: "idle", LivePercent: 0},
		},
	}

	p := tea.NewProgram(m, tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
