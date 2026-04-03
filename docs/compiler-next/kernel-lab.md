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
python3 kernels/lab/kernel_lab.py compile ada_gemm
python3 kernels/lab/kernel_lab.py run ada_gemm
python3 kernels/lab/kernel_lab.py bench-suite --tag ada --dry-run
python3 kernels/lab/kernel_lab.py bench-cmd noop "python3 -c 'print(1)'" --repeats 10 --warmup 2
python3 kernels/lab/kernel_lab.py compare kernels/lab/results/a.json kernels/lab/results/b.json
```

## Benchmark Protocol

Use this protocol when validating new kernels for promotion:

1. Run `doctor` and fail fast on missing toolchains.
2. Run warmup + timed repeats with fixed args and stable environment.
3. Persist timestamped JSON in `kernels/lab/results/`.
4. Compare candidate vs baseline JSON before adopting kernel changes.
5. Record fallback or toolchain gaps explicitly (never silent skip).
6. Normalize kernel-lab results into the benchmark reporting surface when the run is meaningful enough to keep.

The reporting bridge now produces benchmark-style JSON, Markdown, and SVG artifacts from kernel-lab output, so the kernel lab and the standardized GPU suite stay aligned even before the VM GPU is available.

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
