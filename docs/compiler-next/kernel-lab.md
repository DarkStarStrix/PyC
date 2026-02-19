# Kernel Lab CLI

`Kernel/kernel_lab.py` is the kernel prototyping utility for compile/run/benchmark experiments under deterministic contracts.

## Manifest and Command Model

Kernel definitions are declared in `Kernel/lab/kernels.json` with:

- `name`, `source`, `description`, `tags`
- `compile_cmd`, `run_cmd`

Template placeholders:

- `{name}`, `{source}`, `{root}`, `{build_dir}`, `{nvcc}`

This keeps kernel workflows declarative and reproducible across machines.

## Core Commands

```bash
python3 Kernel/kernel_lab.py doctor
python3 Kernel/kernel_lab.py list
python3 Kernel/kernel_lab.py show matrix_mult
python3 Kernel/kernel_lab.py compile matrix_mult
python3 Kernel/kernel_lab.py run matrix_mult
python3 Kernel/kernel_lab.py bench matrix_mult --phase both --repeats 20 --warmup 5
python3 Kernel/kernel_lab.py bench-cmd noop "python3 -c 'print(1)'" --repeats 10 --warmup 2
python3 Kernel/kernel_lab.py compare Kernel/lab/results/a.json Kernel/lab/results/b.json
```

## Phase 5 Benchmark Protocol

Use this protocol when validating new kernels for promotion:

1. Run `doctor` and fail fast on missing toolchains.
2. Run warmup + timed repeats with fixed args and stable environment.
3. Persist timestamped JSON in `Kernel/lab/results/`.
4. Compare candidate vs baseline JSON before adopting kernel changes.
5. Record fallback or toolchain gaps explicitly (never silent skip).

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
python3 Kernel/kernel_lab.py doctor
```

macOS Apple Silicon:

- `nvcc` is typically unavailable.
- Use `bench-cmd` locally for orchestration checks.
- Run compile/run CUDA benchmarks on a remote Linux GPU host and sync results back.
