# Benchmark Results Layout

All benchmark outputs are written under one root:

- `benchmark/benchmarks/results/images/` for SVG charts
- `benchmark/benchmarks/results/json/` for raw JSON + metadata stamps
- `benchmark/benchmarks/results/reports/` for Markdown reports
- `benchmark/benchmarks/results/runs/<run_id>/<tag>/` canonical per-run bundle
- `benchmark/benchmarks/results/latest/` canonical latest CPU/GPU aliases
- `benchmark/benchmarks/results/manifest/results_index.json` machine index
- `benchmark/benchmarks/results/remote_results/hosts/<host>/runs/<run_id>/<tag>/` canonical remote snapshots
- `benchmark/benchmarks/results/remote_results/archive/root_legacy/` preserved legacy remote layout

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

By default, `run_suite.sh` now keeps flat roots (`json/`, `reports/`, `images/`) lean by retaining only the latest CPU/GPU pair there; historical runs are preserved in `results/runs/`.
Set `BENCH_PRUNE_FLAT_HISTORY=0` to keep historical flat files.

Normalize existing artifacts (imports remote snapshots into canonical roots and re-renders latest charts):

```bash
python3 scripts/standardize_benchmark_results.py
```

Normalize remote snapshot layout:

```bash
python3 scripts/standardize_remote_results.py
```

Run in strict native mode (fails if required adapters are not native):

```bash
BENCH_STRICT_NATIVE=1 \
STRICT_NATIVE_REQUIRED_GPU=torch_eager,torch_compile,pyc,tvm \
bash benchmark/benchmarks/run_suite.sh
```

Run with explicit output root and run id:

```bash
bash benchmark/benchmarks/run_suite.sh benchmark/benchmarks/results 20260218T020000Z
```

Ada FP32 GEMM shape-matrix sweep:

```bash
python3 benchmark/benchmarks/gpu/run_gemm_suite.py \
  --matrix-file benchmark/benchmarks/gpu/configs/ada_fp32_gemm_shapes.json \
  --dry-run
```

Drop `--dry-run` on the GPU VM to emit per-shape JSON/Markdown/SVG bundles plus the aggregate `latest_ada_fp32_gemm.*` aliases under `benchmark/benchmarks/results/`.

Require true native CUDA mode for PyC only:

```bash
BENCH_STRICT_NATIVE=1 \
STRICT_NATIVE_REQUIRED_GPU=pyc \
bash benchmark/benchmarks/run_suite.sh
```

Environment lock:

- Use `scripts/setup_benchmark_env_locked.sh` to pin benchmark dependencies.
- For native TVM CUDA, set `ENABLE_TVM_CUDA_BUILD=1` when running the setup script.

Key metrics now emitted per adapter (`results/json/*.json` and `results/reports/*.md`):

- `throughput_tokens_per_sec`
- Latency stats: `mean`, `p50`, `p95`, `min`, `max`
- Stability: jitter (`p95 - p50`), sample CV, repeat CV
- Startup/compile estimates: `startup_ms.*`, compile-overhead estimate
- Resource efficiency: peak memory, throughput per GiB
- Compute estimate: `estimated_flops_per_iter`, `estimated_tflops_per_sec`
