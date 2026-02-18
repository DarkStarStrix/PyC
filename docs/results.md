# Results

## Scope

This document is the canonical summary of benchmark and validation outputs currently published by PyC.

## Published Artifact Contract

Website-published results are centralized under:

- `website/results/artifacts/` (all benchmark SVGs + `*.metadata.json`)
- `website/results/manifest.json` (full index)
- `website/results/latest-summary.json` (latest CPU/GPU summary)

Generation command:

```bash
python3 scripts/publish_site_results.py
```

This makes site content deterministic and reproducible from repository data.

## Latest Run Snapshot

Latest paired run ID: `20260218T023355Z_phase4_final`

- CPU: `device=cpu`, batch `64`, hidden `2048`, iters `40`
- GPU: `device=cuda`, batch `64`, hidden `2048`, iters `40`
- Host: `90c892d22de7` (Linux)

### CPU (mean latency, lower is better)

- PyC CUDA [native on CPU path]: `24.0459 ms`
- PyTorch Compile [native]: `26.7237 ms`
- PyTorch Eager [native]: `36.9971 ms`

### GPU (mean latency, lower is better)

- PyTorch Eager [native]: `0.1154 ms`
- PyTorch Compile [native]: `0.1551 ms`
- PyC CUDA [proxy fallback]: `25.5228 ms`

Interpretation: current PyC GPU path is not yet native-fast in this run (fallback/proxy mode), while CPU path is competitive in this benchmark shape.

## Validation Status

- CMake configure/build path is deterministic for canonical targets.
- Compiler-next tests and runtime contracts are encoded in `tests/compiler_next/`.
- Benchmark results now have centralized publication rails for docs/site consumption.

## Next Measurement Priorities

1. Enforce native-mode requirement for PyC CUDA runs in strict benchmark gates.
2. Add richer kernel-level profiling counters to explain dispatch vs execution deltas.
3. Expand competitor adapters from proxy mode to native integrations where available.
