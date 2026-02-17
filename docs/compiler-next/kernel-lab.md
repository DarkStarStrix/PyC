# Kernel Lab CLI

`Kernel/kernel_lab.py` is a mini lab CLI for prototyping, testing, and benchmarking kernel workflows.

## Manifest-Driven Design

Kernel definitions live in:

- `Kernel/lab/kernels.json`

Each entry supports:

- `name`
- `source`
- `description`
- `tags`
- `compile_cmd`
- `run_cmd`

Template variables supported in commands:

- `{name}`
- `{source}`
- `{root}`
- `{build_dir}`
- `{nvcc}`

## Commands

Health/preflight check:

```bash
python3 Kernel/kernel_lab.py doctor
```

List kernels:

```bash
python3 Kernel/kernel_lab.py list
```

Show one kernel definition:

```bash
python3 Kernel/kernel_lab.py show matrix_mult
```

Compile a kernel using manifest command:

```bash
python3 Kernel/kernel_lab.py compile matrix_mult
```

Run a kernel using manifest `run_cmd`:

```bash
python3 Kernel/kernel_lab.py run matrix_mult
```

Benchmark compile phase:

```bash
python3 Kernel/kernel_lab.py bench matrix_mult --phase compile --repeats 10 --warmup 2
```

Benchmark arbitrary prototype command:

```bash
python3 Kernel/kernel_lab.py bench-cmd sleep_test "python3 -c 'import time; time.sleep(0.01)'" --repeats 20 --warmup 3
```

Compare two benchmark JSON files:

```bash
python3 Kernel/kernel_lab.py compare Kernel/lab/results/a.json Kernel/lab/results/b.json
```

## Outputs

Benchmark outputs are written to:

- `Kernel/lab/results/`

Each run produces timestamped JSON suitable for comparison and tracking.

## Notes

- `nvcc` is required for CUDA `compile` and CUDA `bench --phase compile|both` operations.
- Current default kernel entries are compile-focused (`run_cmd` empty).
- Add `run_cmd` entries in `kernels.json` for executable kernel harnesses.
- Use `--nvcc /path/to/nvcc` if `nvcc` is not in your `PATH`.

## Deterministic Error Handling

The CLI uses stable exit codes:

- `0`: success
- `2`: user input error
- `3`: manifest/schema error
- `4`: required toolchain missing (for example `nvcc`)
- `5`: command execution failed

Expected operational failures are reported as explicit errors (no traceback noise).

## CUDA Toolkit / nvcc

### Linux/Windows

Install NVIDIA CUDA Toolkit and ensure `nvcc` is on `PATH`, then verify:

```bash
nvcc --version
python3 Kernel/kernel_lab.py doctor
```

### macOS

Modern macOS does not support NVIDIA CUDA toolchain on Apple Silicon, so `nvcc` is typically unavailable.
On macOS hosts, use:

- `bench-cmd` for command-prototyping and workflow validation.
- remote Linux/Windows CUDA machine for real CUDA kernel compile/run benchmarks.
