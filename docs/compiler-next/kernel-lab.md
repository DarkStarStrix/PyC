# Kernel Lab CLI

`kernels/lab/kernel_lab.py` is the kernel prototyping utility for compile/run/benchmark experiments under deterministic contracts.

## Manifest and Command Model

Kernel definitions are declared in `kernels/lab/manifests/kernels.json` with:

- `name`, `source`, `description`, `tags`
- `compile_cmd`, `run_cmd`

Template placeholders:

- `{name}`, `{source}`, `{root}`, `{build_dir}`, `{nvcc}`

This keeps kernel workflows declarative and reproducible across machines.

## Core Commands

```bash
python3 kernels/lab/kernel_lab.py doctor
python3 kernels/lab/kernel_lab.py list
python3 kernels/lab/kernel_lab.py show ada_gemm
python3 kernels/lab/kernel_lab.py task-hardware
python3 kernels/lab/kernel_lab.py task-create ada-sm89-gemm --task-kind gemm --candidate-tag ada
python3 kernels/lab/kernel_lab.py task-show ada-sm89-gemm
python3 kernels/lab/kernel_lab.py task-run ada-sm89-gemm --progress
python3 kernels/lab/kernel_lab.py compile ada_gemm
python3 kernels/lab/kernel_lab.py run ada_gemm
python3 kernels/lab/kernel_lab.py bench ada_gemm --phase run --repeats 10 --warmup 2 --progress
python3 kernels/lab/kernel_lab.py bench-suite --tag ada --dry-run
python3 kernels/lab/kernel_lab.py bench-suite --tag ada --phase run --repeats 10 --warmup 2 --progress
python3 kernels/lab/kernel_lab.py bench-cmd noop "python3 -c 'print(1)'" --repeats 10 --warmup 2 --progress
python3 kernels/lab/kernel_lab.py compare kernels/lab/results/a.json kernels/lab/results/b.json
python3 kernels/lab/kernel_lab.py task-complete ada-sm89-gemm --winner ada_tensor_core_fp16 --result-json kernels/lab/results/winner.json
```

## Task Builder

Task records live under `kernels/lab/tasks/`, and baseline state lives in `kernels/lab/manifests/task_baselines.json`.

Each task captures:

- hardware profile and capacity tier
- selected baseline kernel for the current backend and arch
- benchmark/profile commands to run
- task-run execution records under `kernels/lab/tasks/runs/`
- fixed-shape or mixed-shape `gemm_sequence` results for long-lived hot-path runs
- winning kernel and promoted baseline entry when the task is closed

`task-complete` promotes the winning kernel into the baseline manifest for the same `task_kind` + backend + arch. That makes the last accepted winner the next kernel to beat.
`task-run` executes the saved `profile_protocol` commands in order, stores a timestamped run record, and captures `stdout_tail`, `stderr_tail`, parsed `observed_metrics`, and artifact paths announced via `wrote ...`.

## Feature-Profile Tasks

`task-create` also supports named PyC runtime feature profiles through `--pyc-feature-profile`.

Current built-in profiles are:

- `pyc-fp32-baseline`
- `pyc-fp32-speculative`
- `pyc-fp32-speculative-memory`
- `pyc-fp32-speculative-util`
- `pyc-fp32-phantom-shadow`
- `pyc-fp32-phantom-shadow-util`
- `pyc-fp32-no-graph-replay`

These profiles are emitted into the saved `profile_protocol` as explicit environment-variable prefixes, so the task record fully describes the actual runtime mode being benchmarked. The PyC benchmark helper now honors compile-option controls such as speculative plans, phantom graph, objective mode, memory budget, utilization floor, compile budget, and cache mode.

That makes kernel-lab suitable for comparing:

- bare CUDA execution-path tuning
- speculative-plan execution
- phantom-graph shadow and reshape behavior
- A2 rematerialization pressure behavior
- graph replay on/off
- policy-mode changes
- runtime-vs-prototype kernel experiments in the same task record

## Benchmark Protocol

Use this protocol when validating new kernels for promotion:

1. Run `doctor` and fail fast on missing toolchains.
2. Run warmup + timed repeats with fixed args and stable environment.
3. Persist timestamped JSON in `kernels/lab/results/`.
4. Compare candidate vs baseline JSON before adopting kernel changes.
5. Record fallback or toolchain gaps explicitly (never silent skip).
6. Normalize kernel-lab results into the benchmark reporting surface when the run is meaningful enough to keep.
7. Use `--progress` on long runs so warmup/repeat loops and suite execution stay observable in tmux.

Mixed-shape task runs should prefer `BENCH_SEQUENCE` over one-off compile/run loops when you want to judge runtime behavior under shape drift. Those runs should keep the per-step `phantom_graph` and `rematerialized_*` fields in the saved JSON so later analysis can compare shape transitions, pressure handling, and throughput in one record.

The reporting bridge now produces benchmark-style JSON, Markdown, and SVG artifacts from kernel-lab output, so the kernel lab and the standardized GPU suite stay aligned even before the VM GPU is available.
Kernel-lab benchmark JSON now also preserves `stdout_tail`, `stderr_tail`, and parsed `observed_metrics` from successful runs, which is important for kernels that print internal timing such as `best_ms` and `gflops`.

## PyC GPU Adapter

`benchmark/benchmarks/gpu/adapters/adapter_pyc.py` now defaults to the bundled `benchmark/benchmarks/gpu/external/bench_pyc_cmd.py` helper when `PYC_GPU_BENCH_CMD` is unset. That keeps fresh GPU boxes usable without a manual env bootstrap step.

## Analysis Bundles

After copying a remote Ada run back into `benchmark/benchmarks/results/remote_results/...`, generate the local analysis bundle with:

```bash
python3 benchmark/tools/analyze_ada_gemm_results.py
```

or pull + analyze the latest remote run in one step with:

```bash
bash scripts/pull_and_analyze_ada_artifacts.sh
```

That one-step script is the canonical operator flow for this repo: it discovers the latest Ada run, pulls the matching kernel-lab artifact, and writes the full local bundle before you decide whether the result is a kernel candidate, a runtime-path tuning result, or a regression to discard.

That creates:

- `raw/` for copied source JSON
- `graphs/` for SVG comparison views
- `sheets/analysis.md` for the narrative judgment sheet
- `rankings/` for machine-readable and Markdown ranking summaries

## Deterministic Contracts

Stable exit codes:

- `0`: success
- `2`: user input error
- `3`: manifest/schema error
- `4`: required toolchain missing
- `5`: command execution failed

Expected failures are returned as explicit actionable messages, not ambiguous tracebacks.

## CUDA Toolchain Notes

Linux/Windows with NVIDIA GPU:

```bash
nvcc --version
python3 kernels/lab/kernel_lab.py doctor
```

macOS Apple Silicon:

- `nvcc` is typically unavailable.
- Use `bench-cmd` locally for orchestration checks.
- Run compile/run CUDA benchmarks on a remote Linux GPU host and sync results back.

## Ada Staging

Ada-specific kernel work should use:

- `kernels/prototypes/ada/gemm/kernel.cu` for FP32 GEMM baseline work.
- `kernels/prototypes/ada/tensor_core/kernel.cu` for FP16/BF16 Tensor Core prep.
- `benchmark/benchmarks/gpu/run_gemm_suite.py` for the shape-matrix Ada benchmark sweep.
