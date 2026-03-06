package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"
)

type gpuRow struct {
	Index   string `json:"index"`
	Name    string `json:"name"`
	UtilGPU string `json:"util_gpu_pct"`
	UtilMem string `json:"util_mem_pct"`
	MemUsed string `json:"mem_used_mb"`
	MemTot  string `json:"mem_total_mb"`
	TempC   string `json:"temp_c"`
	PowerW  string `json:"power_w"`
	PowerL  string `json:"power_limit_w"`
}

type gpuProcRow struct {
	PID      string `json:"pid"`
	Name     string `json:"name"`
	GPUUUID  string `json:"gpu_uuid"`
	MemUsed  string `json:"mem_used_mb"`
	DeviceID string `json:"device_index,omitempty"`
}

type procRow struct {
	PID  string `json:"pid"`
	PPID string `json:"ppid"`
	CPU  string `json:"cpu_pct"`
	MEM  string `json:"mem_pct"`
	RSS  string `json:"rss_kb"`
	CMD  string `json:"cmd"`
}

type hostStats struct {
	Hostname      string  `json:"hostname"`
	Kernel        string  `json:"kernel"`
	NowUTC        string  `json:"now_utc"`
	UptimeSec     float64 `json:"uptime_sec"`
	Load1         string  `json:"load1"`
	Load5         string  `json:"load5"`
	Load15        string  `json:"load15"`
	MemTotalMB    string  `json:"mem_total_mb"`
	MemAvailMB    string  `json:"mem_available_mb"`
	MemUsedPct    string  `json:"mem_used_pct"`
	ProcessCount  int     `json:"process_count"`
	NetRxMBps     string  `json:"net_rx_mb_s"`
	NetTxMBps     string  `json:"net_tx_mb_s"`
	CollectorNote string  `json:"collector_note,omitempty"`
}

type snapshot struct {
	Host      hostStats    `json:"host"`
	GPUs      []gpuRow     `json:"gpus"`
	GPUProcs  []gpuProcRow `json:"gpu_processes"`
	TopProcs  []procRow    `json:"top_processes"`
	Tick      int64        `json:"tick"`
	Refreshed string       `json:"refreshed_utc"`
}

type netCounters struct {
	Rx uint64
	Tx uint64
}

func runCommand(name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("%s %v failed: %w", name, args, err)
	}
	return string(out), nil
}

func hasCommand(name string) bool {
	_, err := exec.LookPath(name)
	return err == nil
}

func parseProcNetDev() (netCounters, error) {
	f, err := os.Open("/proc/net/dev")
	if err != nil {
		return netCounters{}, err
	}
	defer f.Close()

	var totalRx, totalTx uint64
	sc := bufio.NewScanner(f)
	lineNo := 0
	for sc.Scan() {
		lineNo++
		if lineNo <= 2 {
			continue
		}
		line := strings.TrimSpace(sc.Text())
		if line == "" || !strings.Contains(line, ":") {
			continue
		}
		parts := strings.SplitN(line, ":", 2)
		if len(parts) != 2 {
			continue
		}
		iface := strings.TrimSpace(parts[0])
		if iface == "lo" {
			continue
		}
		fields := strings.Fields(parts[1])
		if len(fields) < 16 {
			continue
		}
		rx, err1 := strconv.ParseUint(fields[0], 10, 64)
		tx, err2 := strconv.ParseUint(fields[8], 10, 64)
		if err1 == nil {
			totalRx += rx
		}
		if err2 == nil {
			totalTx += tx
		}
	}
	if err := sc.Err(); err != nil {
		return netCounters{}, err
	}
	return netCounters{Rx: totalRx, Tx: totalTx}, nil
}

func parseMemInfo() (totalMB, availMB float64, err error) {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0, 0, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	var memTotalKB, memAvailKB float64
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "MemTotal:") {
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				memTotalKB, _ = strconv.ParseFloat(fields[1], 64)
			}
		}
		if strings.HasPrefix(line, "MemAvailable:") {
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				memAvailKB, _ = strconv.ParseFloat(fields[1], 64)
			}
		}
	}
	if err := sc.Err(); err != nil {
		return 0, 0, err
	}
	if memTotalKB == 0 {
		return 0, 0, errors.New("MemTotal not found")
	}
	return memTotalKB / 1024.0, memAvailKB / 1024.0, nil
}

func parseLoadAvg() (l1, l5, l15 string, err error) {
	b, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		return "", "", "", err
	}
	fields := strings.Fields(strings.TrimSpace(string(b)))
	if len(fields) < 3 {
		return "", "", "", errors.New("invalid /proc/loadavg")
	}
	return fields[0], fields[1], fields[2], nil
}

func parseUptime() (float64, error) {
	b, err := os.ReadFile("/proc/uptime")
	if err != nil {
		return 0, err
	}
	fields := strings.Fields(strings.TrimSpace(string(b)))
	if len(fields) < 1 {
		return 0, errors.New("invalid /proc/uptime")
	}
	return strconv.ParseFloat(fields[0], 64)
}

func processCount() int {
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return 0
	}
	count := 0
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		if _, err := strconv.Atoi(e.Name()); err == nil {
			count++
		}
	}
	return count
}

func parseGPUDeviceMap() map[string]string {
	m := map[string]string{}
	if !hasCommand("nvidia-smi") {
		return m
	}
	out, err := runCommand("nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits")
	if err != nil {
		return m
	}
	for _, ln := range strings.Split(strings.TrimSpace(out), "\n") {
		if strings.TrimSpace(ln) == "" {
			continue
		}
		p := strings.Split(ln, ",")
		if len(p) < 2 {
			continue
		}
		idx := strings.TrimSpace(p[0])
		uuid := strings.TrimSpace(p[1])
		m[uuid] = idx
	}
	return m
}

func collectGPUs() ([]gpuRow, []gpuProcRow) {
	if !hasCommand("nvidia-smi") {
		return nil, nil
	}
	gpuOut, err := runCommand("nvidia-smi",
		"--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
		"--format=csv,noheader,nounits")
	if err != nil {
		return nil, nil
	}

	rows := []gpuRow{}
	for _, ln := range strings.Split(strings.TrimSpace(gpuOut), "\n") {
		if strings.TrimSpace(ln) == "" {
			continue
		}
		p := strings.Split(ln, ",")
		if len(p) < 9 {
			continue
		}
		for i := range p {
			p[i] = strings.TrimSpace(p[i])
		}
		rows = append(rows, gpuRow{
			Index:   p[0],
			Name:    p[1],
			UtilGPU: p[2],
			UtilMem: p[3],
			MemUsed: p[4],
			MemTot:  p[5],
			TempC:   p[6],
			PowerW:  p[7],
			PowerL:  p[8],
		})
	}

	uuidToIdx := parseGPUDeviceMap()
	procRows := []gpuProcRow{}
	procOut, err := runCommand("nvidia-smi",
		"--query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory",
		"--format=csv,noheader,nounits")
	if err == nil {
		for _, ln := range strings.Split(strings.TrimSpace(procOut), "\n") {
			if strings.TrimSpace(ln) == "" {
				continue
			}
			p := strings.Split(ln, ",")
			if len(p) < 4 {
				continue
			}
			for i := range p {
				p[i] = strings.TrimSpace(p[i])
			}
			pr := gpuProcRow{
				PID:     p[0],
				Name:    p[1],
				GPUUUID: p[2],
				MemUsed: p[3],
			}
			if idx, ok := uuidToIdx[pr.GPUUUID]; ok {
				pr.DeviceID = idx
			}
			procRows = append(procRows, pr)
		}
	}
	sort.Slice(procRows, func(i, j int) bool {
		mi, _ := strconv.Atoi(procRows[i].MemUsed)
		mj, _ := strconv.Atoi(procRows[j].MemUsed)
		return mi > mj
	})
	return rows, procRows
}

func collectTopProcesses(limit int) []procRow {
	if !hasCommand("ps") {
		return nil
	}
	out, err := runCommand("ps", "-eo", "pid,ppid,pcpu,pmem,rss,comm", "--sort=-pcpu")
	if err != nil {
		return nil
	}
	rows := []procRow{}
	lines := strings.Split(strings.TrimSpace(out), "\n")
	for i, ln := range lines {
		if i == 0 {
			continue
		}
		fields := strings.Fields(ln)
		if len(fields) < 6 {
			continue
		}
		cmd := strings.Join(fields[5:], " ")
		rows = append(rows, procRow{
			PID:  fields[0],
			PPID: fields[1],
			CPU:  fields[2],
			MEM:  fields[3],
			RSS:  fields[4],
			CMD:  cmd,
		})
		if len(rows) >= limit {
			break
		}
	}
	return rows
}

func collectHost(prev netCounters, prevTS time.Time) (hostStats, netCounters) {
	host, _ := os.Hostname()
	kernelOut, _ := runCommand("uname", "-r")
	uptime, upErr := parseUptime()
	l1, l5, l15, loadErr := parseLoadAvg()
	memTotal, memAvail, memErr := parseMemInfo()
	pcount := processCount()
	now := time.Now().UTC()

	note := ""
	if upErr != nil || loadErr != nil || memErr != nil {
		note = "some host counters unavailable (non-Linux or restricted /proc)"
	}

	curNet, netErr := parseProcNetDev()
	rxRate, txRate := 0.0, 0.0
	if netErr == nil && !prevTS.IsZero() {
		sec := now.Sub(prevTS).Seconds()
		if sec > 0 {
			rxRate = float64(curNet.Rx-prev.Rx) / sec / (1024.0 * 1024.0)
			txRate = float64(curNet.Tx-prev.Tx) / sec / (1024.0 * 1024.0)
		}
	}
	if netErr != nil && note == "" {
		note = "network counters unavailable"
	}

	memUsedPct := 0.0
	if memTotal > 0 && memAvail >= 0 {
		memUsedPct = ((memTotal - memAvail) / memTotal) * 100.0
	}

	return hostStats{
		Hostname:      host,
		Kernel:        strings.TrimSpace(kernelOut),
		NowUTC:        now.Format(time.RFC3339),
		UptimeSec:     uptime,
		Load1:         l1,
		Load5:         l5,
		Load15:        l15,
		MemTotalMB:    fmt.Sprintf("%.1f", memTotal),
		MemAvailMB:    fmt.Sprintf("%.1f", memAvail),
		MemUsedPct:    fmt.Sprintf("%.1f", memUsedPct),
		ProcessCount:  pcount,
		NetRxMBps:     fmt.Sprintf("%.2f", rxRate),
		NetTxMBps:     fmt.Sprintf("%.2f", txRate),
		CollectorNote: note,
	}, curNet
}

func humanUptime(sec float64) string {
	if sec <= 0 {
		return "n/a"
	}
	s := int64(sec)
	d := s / 86400
	h := (s % 86400) / 3600
	m := (s % 3600) / 60
	rem := s % 60
	if d > 0 {
		return fmt.Sprintf("%dd %02dh %02dm %02ds", d, h, m, rem)
	}
	return fmt.Sprintf("%02dh %02dm %02ds", h, m, rem)
}

func clearScreen() {
	fmt.Print("\033[2J\033[H")
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	if n <= 3 {
		return s[:n]
	}
	return s[:n-3] + "..."
}

func render(s snapshot, refresh time.Duration) {
	clearScreen()
	fmt.Printf("Nexa Insight | single-node telemetry | refresh=%s | tick=%d\n", refresh, s.Tick)
	fmt.Printf("UTC: %s | host: %s | kernel: %s | goos=%s\n", s.Host.NowUTC, s.Host.Hostname, s.Host.Kernel, runtime.GOOS)
	fmt.Println(strings.Repeat("=", 120))
	fmt.Printf("Host  uptime=%s  load=%s %s %s  mem=%s/%s MB (%s%% used)  procs=%d  net rx/tx=%s/%s MB/s\n",
		humanUptime(s.Host.UptimeSec), s.Host.Load1, s.Host.Load5, s.Host.Load15,
		fmt.Sprintf("%.1f", mustF(s.Host.MemTotalMB)-mustF(s.Host.MemAvailMB)), s.Host.MemTotalMB, s.Host.MemUsedPct,
		s.Host.ProcessCount, s.Host.NetRxMBps, s.Host.NetTxMBps)
	if s.Host.CollectorNote != "" {
		fmt.Printf("Note: %s\n", s.Host.CollectorNote)
	}
	fmt.Println(strings.Repeat("-", 120))
	fmt.Println("GPU Summary")
	fmt.Printf("%-4s %-30s %-6s %-6s %-15s %-7s %-12s\n", "ID", "Name", "GPU%", "MEM%", "Mem(MB)", "TempC", "Power(W)")
	if len(s.GPUs) == 0 {
		fmt.Println("(no GPUs detected or nvidia-smi unavailable)")
	}
	for _, g := range s.GPUs {
		mem := fmt.Sprintf("%s/%s", g.MemUsed, g.MemTot)
		pw := fmt.Sprintf("%s/%s", g.PowerW, g.PowerL)
		fmt.Printf("%-4s %-30s %-6s %-6s %-15s %-7s %-12s\n", g.Index, truncate(g.Name, 30), g.UtilGPU, g.UtilMem, mem, g.TempC, pw)
	}
	fmt.Println(strings.Repeat("-", 120))
	fmt.Println("Top GPU Processes")
	fmt.Printf("%-5s %-4s %-10s %-40s %-30s\n", "PID", "GPU", "Mem(MB)", "Process", "UUID")
	if len(s.GPUProcs) == 0 {
		fmt.Println("(no active compute processes)")
	}
	maxGPUProcs := min(12, len(s.GPUProcs))
	for i := 0; i < maxGPUProcs; i++ {
		p := s.GPUProcs[i]
		gid := p.DeviceID
		if gid == "" {
			gid = "?"
		}
		fmt.Printf("%-5s %-4s %-10s %-40s %-30s\n", p.PID, gid, p.MemUsed, truncate(p.Name, 40), truncate(p.GPUUUID, 30))
	}
	fmt.Println(strings.Repeat("-", 120))
	fmt.Println("Top CPU Processes")
	fmt.Printf("%-6s %-6s %-6s %-6s %-10s %-40s\n", "PID", "PPID", "CPU%", "MEM%", "RSS(KB)", "CMD")
	if len(s.TopProcs) == 0 {
		fmt.Println("(process table unavailable)")
	}
	for _, p := range s.TopProcs {
		fmt.Printf("%-6s %-6s %-6s %-6s %-10s %-40s\n", p.PID, p.PPID, p.CPU, p.MEM, p.RSS, truncate(p.CMD, 40))
	}
	fmt.Println(strings.Repeat("=", 120))
	fmt.Println("Press Ctrl-C to exit.")
}

func mustF(s string) float64 {
	v, _ := strconv.ParseFloat(strings.TrimSpace(s), 64)
	return v
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func ensureDir(path string) error {
	if path == "" {
		return nil
	}
	dir := filepath.Dir(path)
	return os.MkdirAll(dir, 0o755)
}

func main() {
	refresh := flag.Duration("refresh", 2*time.Second, "refresh interval")
	topN := flag.Int("top", 12, "number of top CPU processes")
	jsonOut := flag.String("json-out", "", "optional ndjson output file path")
	noClear := flag.Bool("no-clear", false, "do not clear terminal between frames")
	flag.Parse()

	if *refresh < 250*time.Millisecond {
		*refresh = 250 * time.Millisecond
	}
	var outFile *os.File
	if *jsonOut != "" {
		if err := ensureDir(*jsonOut); err != nil {
			fmt.Fprintf(os.Stderr, "failed to create json output dir: %v\n", err)
			os.Exit(1)
		}
		f, err := os.OpenFile(*jsonOut, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to open json output file: %v\n", err)
			os.Exit(1)
		}
		outFile = f
		defer outFile.Close()
	}

	prevNet := netCounters{}
	prevTS := time.Time{}
	tick := int64(0)

	for {
		host, curNet := collectHost(prevNet, prevTS)
		gpus, gpuProcs := collectGPUs()
		top := collectTopProcesses(*topN)
		snap := snapshot{
			Host:      host,
			GPUs:      gpus,
			GPUProcs:  gpuProcs,
			TopProcs:  top,
			Tick:      tick,
			Refreshed: time.Now().UTC().Format(time.RFC3339Nano),
		}
		if !*noClear {
			render(snap, *refresh)
		} else {
			fmt.Printf("[%s] tick=%d gpu=%d procs=%d load=%s/%s/%s\n",
				snap.Refreshed, tick, len(gpus), host.ProcessCount, host.Load1, host.Load5, host.Load15)
		}
		if outFile != nil {
			b, err := json.Marshal(snap)
			if err == nil {
				_, _ = outFile.Write(append(b, '\n'))
			}
		}
		prevNet = curNet
		prevTS = time.Now()
		tick++
		time.Sleep(*refresh)
	}
}
