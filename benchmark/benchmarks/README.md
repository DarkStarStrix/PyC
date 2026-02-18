# Benchmark Results Layout

All benchmark outputs are written under one root:

- `benchmark/benchmarks/results/images/` for SVG charts
- `benchmark/benchmarks/results/json/` for raw JSON + metadata stamps
- `benchmark/benchmarks/results/reports/` for Markdown reports

Each run is versioned by `run_id` and `tag` in filename form:

- `<run_id>__cpu.json`
- `<run_id>__gpu.json`
- `<run_id>__cpu.metadata.json`
- `<run_id>__cpu.svg`
- `<run_id>__cpu.md`

Run end-to-end suite:

```bash
bash benchmark/benchmarks/run_suite.sh
```

Run with explicit output root and run id:

```bash
bash benchmark/benchmarks/run_suite.sh benchmark/benchmarks/results 20260218T020000Z
```

Require true native CUDA mode for PyC (fail run if PyC reports proxy/fallback):

```bash
PYC_REQUIRE_NATIVE_CUDA=1 bash benchmark/benchmarks/run_suite.sh
```
