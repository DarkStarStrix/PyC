# Nexa Insight

`Nexa Insight` contains two single-node telemetry apps for GPU training boxes:

- `nexa-insight`: lightweight classic terminal dashboard (stdlib).
- `nexa-insight-tui`: Bubble Tea cyberpunk TUI with ML run metrics.

## What It Shows

- Host stats: uptime, load averages, memory usage, process count.
- Network throughput: aggregate RX/TX MB/s from `/proc/net/dev`.
- GPU summary: utilization, memory, temperature, and power from `nvidia-smi`.
- Active GPU compute processes: PID, process, GPU UUID, VRAM usage.
- Top CPU processes: PID/PPID/CPU%/MEM%/RSS.

## Build

```bash
cd infra/nexa_insight
go build -o nexa-insight ./cmd/nexa-insight
go build -o nexa-insight-tui ./cmd/nexa-insight-tui
```

## Run

```bash
cd infra/nexa_insight
./nexa-insight
```

Bubble Tea TUI:

```bash
./nexa-insight-tui --refresh 1s --runs-root benchmark/remote_results/runpod_h100_8x/campaign_v4
```

Or from repo root:

```bash
bash infra/run_nexa_insight.sh
bash infra/run_nexa_insight_tui.sh --refresh 1s
```

## Useful Flags

- `--refresh 1s`: update interval.
- `--top 20`: number of top CPU processes.
- `--json-out benchmark/remote_results/runpod_h100_8x/insight/live.ndjson`: append snapshots for later analysis.
- `--no-clear`: emit heartbeat lines instead of full-screen redraw.

## Notes

- Primary target is Linux GPU hosts (RunPod/Ubuntu).
- If `nvidia-smi` is unavailable, GPU sections are shown as unavailable.
- Exit with `Ctrl-C`.
