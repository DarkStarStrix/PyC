# Benchmark Results Layout

All benchmark outputs are written under one root:

- `benchmark/benchmarks/results/analysis/` for post-run judgment bundles (`raw/`, `graphs/`, `sheets/`, `rankings/`)
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
  --progress \
  --dry-run
```

Drop `--dry-run` on the GPU VM to emit per-shape JSON/Markdown/SVG bundles plus the aggregate `latest_ada_fp32_gemm.*` aliases under `benchmark/benchmarks/results/`.

For a single direct PyC run that stays readable in tmux while still writing the full JSON artifact, use:

```bash
bash scripts/run_pyc_bench_pretty.sh cuda 64 1024 5 2
```

Mixed-shape PyC phantom-graph run in one long-lived process:

```bash
BENCH_TASK=gemm \
BENCH_SEQUENCE='512x512x512;1024x1024x1024;2048x2048x2048;1024x1024x1024;512x512x512' \
PYC_BENCH_ENABLE_SPECULATIVE_PLANS=1 \
PYC_BENCH_MAX_SPECULATIVE_PLANS=3 \
PYC_BENCH_ENABLE_PHANTOM_GRAPH=1 \
PYC_BENCH_PHANTOM_HORIZON_STEPS=1 \
./build/pyc_compiler_next_bench cuda 512 512 20 4
```

That mode keeps one compiled model alive, reuses buffers, walks a realistic shape sequence, and emits per-step phantom drift/reshape telemetry under `sequence.steps[*].phantom_graph`.
The Ada analysis bundle now also summarizes `sequence.steps[*].reliability` so rematerialization shows up next to the phantom graph rather than only in raw JSON.

Build the local Ada judgment bundle after syncing the remote run back:

```bash
python3 benchmark/tools/analyze_ada_gemm_results.py
```

Or use the wrapper that pulls the latest remote Ada run, stages the kernel-lab JSON, and generates the bundle:

```bash
bash scripts/pull_and_analyze_ada_artifacts.sh
```

That writes one analysis directory with:

- `raw/` copied source JSON artifacts
- `graphs/` SVG comparison charts
- `sheets/analysis.md` for the narrative readout
- `rankings/rankings.{json,md}` for machine + human ranking views

The canonical operator flow for judgment is: run the remote Ada sweep or kernel-lab task, pull it with `scripts/pull_and_analyze_ada_artifacts.sh`, then inspect `raw/`, `graphs/`, `sheets/`, and `rankings/` together.

Require true native CUDA mode for PyC only:

```bash
BENCH_STRICT_NATIVE=1 \
STRICT_NATIVE_REQUIRED_GPU=pyc \
bash benchmark/benchmarks/run_suite.sh
```

The `pyc` adapter now defaults to the repo-local `benchmark/benchmarks/gpu/external/bench_pyc_cmd.py` helper when `PYC_GPU_BENCH_CMD` is unset, so a fresh GPU host can benchmark PyC without manually exporting that command first.

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
- PyC compile/runtime controls under `compile_options`, including speculative-plan and phantom-graph flags
- PyC runtime reliability under `reliability`
- PyC phantom-graph telemetry under `phantom_graph`
