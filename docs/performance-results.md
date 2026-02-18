# PyC Performance Results

## Current Baseline

Primary comparative baseline comes from run `20260218T023355Z_phase4_final` (Linux host with RTX 4090, batch `64`, hidden `2048`, 40 iters, 10 warmup).

### CPU (mean latency)

- PyC CUDA (CPU native path): `24.0459 ms`
- PyTorch Compile: `26.7237 ms`
- PyTorch Eager: `36.9971 ms`

### GPU (mean latency)

- PyTorch Eager: `0.1154 ms`
- PyTorch Compile: `0.1551 ms`
- PyC CUDA: `25.5228 ms` (proxy/fallback mode in this run)

## Interpretation

- CPU-path execution is competitive in this benchmark shape.
- GPU-path execution remains bottlenecked until PyC native CUDA dispatch stays active with zero fallback.

## Canonical Result Sources

- `docs/results.md`
- `docs/perf-report.md`
- `website/results/latest-summary.json`
- `website/results/manifest.json`
