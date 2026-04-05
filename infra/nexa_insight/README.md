# Nexa Insight

`Nexa Insight` is now a single local SSH-backed Textual observer that watches the remote box from outside tmux.

## Core Identity

- Always-on telemetry dashboard with 1 Hz sparklines for SM utilization, memory-controller utilization, power, and temperature.
- Active GPU process table filtered to processes with an active GPU context.
- Structured progress view sourced from `latest_ada_fp32_gemm.progress.json`.
- Active tmux window table and live pane tail for execution visibility.
- Recent benchmark completions for quick judgment.

## Requirements

- Python 3.9+
- `textual`
- SSH access to the remote GPU box
- `nvidia-smi` and `tmux` on the remote host
- Benchmark progress state written to `latest_ada_fp32_gemm.progress.json`

## Run

```bash
bash infra/run_nexa_insight_local_tui.sh
```

The classic `run_nexa_insight.sh` and `run_nexa_insight_tui.sh` entrypoints now launch the same local observer.

## Keybindings

- `q`: quit
- `r`: refresh immediately

## Notes

- The local observer polls the remote box over SSH and is the primary judgment surface.
- `tmux` remains the execution surface only.
- The snapshot profiler groups `ncu` CSV output by kernel name and kernel ID.
- Progress state is read from `latest_ada_fp32_gemm.progress.json`, not inferred from terminal redraw behavior.
