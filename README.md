# PyC

Lightweight compiler/toolchain project with a canonical cross-platform CMake build and stable core targets for deterministic CI.

[![CI](https://github.com/DarkStarStrix/PyC/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/DarkStarStrix/PyC/actions/workflows/cmake-multi-platform.yml)
![Release](https://img.shields.io/badge/Release-Alpha-orange)
![Tests](https://img.shields.io/badge/Tests-12%2F12%20Passing-brightgreen)
![Stability](https://img.shields.io/badge/Stability-Stable%20Core%20Targets-blue)
![Compiler Scope](https://img.shields.io/badge/Compiler%20Pipeline-Experimental-yellow)
![Platforms](https://img.shields.io/badge/Platforms-Linux%20%7C%20macOS%20%7C%20Windows-5C7CFA)
![Build System](https://img.shields.io/badge/Build-CMake-0C8E76)
![Smoke Test](https://img.shields.io/badge/Smoke%20Driver-pyc-success)
![Benchmarking](https://img.shields.io/badge/Benchmark%20Harness-Enabled-success)
![Docs](https://img.shields.io/badge/Docs-Expanded-1E90FF)
![License](https://img.shields.io/badge/License-Apache--2.0-green)

## Overview

PyC currently focuses on stable build/link contracts and deterministic CI behavior:

- `pyc_core_obj`: object library for stable core objects.
- `pyc_core`: canonical static library from `pyc_core_obj`.
- `pyc_foundation`: compatibility static library from the same objects.
- `pyc`: minimal deterministic executable used by smoke tests.

The next-generation compiler scaffolding is available behind `PYC_BUILD_COMPILER_NEXT=ON`.

## Repository Layout

- `Core/C_Files/`: C sources.
- `Core/Header_Files/`: C headers.
- `.github/workflows/cmake-multi-platform.yml`: canonical CI workflow.
- `benchmark/`: benchmark harness and workloads.
- `AI/`: linked AI bridge layer that applies optimization-policy contracts to compiler-next options.
- `docs/`: project docs, benchmarking, build/CI, performance reports.

## Build

### Prerequisites

- CMake `>= 3.10`
- C compiler with C11 support
- Python 3 (for benchmark harness)

### Configure + Build Stable Targets

```bash
cmake -S . -B build
cmake --build build --parallel --target pyc pyc_core pyc_foundation
```

### Smoke Test

Linux/macOS:

```bash
./build/pyc
```

Windows (multi-config generators):

```powershell
.\build\Release\pyc.exe
```

Expected output:

```text
PyC CI driver: core targets configured successfully.
```

## CI

The project uses one canonical workflow: `CI` in `.github/workflows/cmake-multi-platform.yml`.

It runs on Ubuntu, macOS, and Windows, and performs:

1. CMake configure
2. Explicit build of `pyc`, `pyc_core`, `pyc_foundation`
3. OS-specific smoke test for `pyc`
4. Non-fatal `ctest`

CI also enforces source coverage for active C sources (`Core/C_Files`, `compiler`, `AI`, `tests/compiler_next`): if a `.c` file is not referenced by `CMakeLists.txt`, the suite fails.

## Build Efficiency

You do not need to rebuild everything from scratch locally on every push.

1. Reuse the same `build/` directory between edits.
2. Re-run `cmake --build build --parallel` for incremental builds.
3. CI runners are ephemeral, but Linux/macOS jobs now use `ccache` to reduce repeated compile time across runs.

## Benchmarking

PyC includes a deterministic benchmark harness for stable core targets.

Run:

```bash
python3 benchmark/harness.py --repeats 7 --micro-rounds 4000
```

Outputs:

- `benchmark/benchmarks/results/json/latest_core.json`
- `benchmark/benchmarks/results/reports/latest_core.md`
- `docs/performance-results.md`

Publish website-ready benchmark artifacts:

```bash
python3 scripts/publish_site_results.py
```

Published output:

- `website/results/manifest.json`
- `website/results/latest-summary.json`
- `website/results/artifacts/**` (SVG + metadata JSON)

## How To Use PyC

### Stable CLI Entrypoint (Recommended)

Build and run the deterministic CI driver:

```bash
cmake -S . -B build
cmake --build build --parallel --target pyc pyc_core pyc_foundation
./build/pyc
```

Expected output:

```text
PyC CI driver: core targets configured successfully.
```

### Link `pyc_core` in Your Own Project

1. Build `pyc_core`:

```bash
cmake -S . -B build
cmake --build build --parallel --target pyc_core
```

2. In your C/C++ project:
- Add include path: `Core/Header_Files/`
- Link static library: `build/libpyc_core.a` (or platform-equivalent)

Use `pyc_foundation` only when downstream compatibility requires it.

## Compiler-Next (Experimental)

Build and run the compiler-next smoke test:

```bash
cmake -S . -B build -D PYC_BUILD_COMPILER_NEXT=ON -D PYC_BUILD_COMPILER_NEXT_TESTS=ON
cmake --build build --parallel --target pyc_compiler_next pyc_compiler_next_smoke
./build/pyc_compiler_next_smoke
```

Public interfaces:

- `include/pyc/compiler_api.h`
- `include/pyc/ir.h`
- `include/pyc/pass_manager.h`
- `include/pyc/runtime_allocator.h`
- `include/pyc/kernel_registry.h`
- `include/pyc/runtime_control.h`
- `include/pyc/ai_bridge.h`

## Binary Distribution

Release binaries are packaged and published by:

- `.github/workflows/release-binaries.yml`

Assets are published per OS:

- `pyc-linux-x86_64.tar.gz`
- `pyc-macos-arm64.tar.gz`
- `pyc-windows-x86_64.zip`

Static download page for end users:

- `index.html` (uses `styles.css` and `app.js` at repo root)

When published with GitHub Pages, this page auto-detects OS and links the latest release asset.

## GPU Benchmarking (Remote CUDA)

For real GPU testing on rented Linux machines:

1. Provision Ubuntu + NVIDIA GPU host.
2. Run setup script:
   ```bash
   bash scripts/setup_cuda_remote_ubuntu.sh
   source .venv/bin/activate
   ```
3. Run standardized suite:
   ```bash
   python3 benchmark/benchmarks/gpu/run_gpu_suite.py --device cuda --tag gpu_baseline
   ```

Detailed guide:

- `docs/compiler-next/gpu-testing-playbook.md`

Adapter comparison includes:

- `torch_eager`, `torch_compile`, `pyc`, `tvm`, `xla`, `tensorrt`, `glow`

For non-PyTorch backends, set adapter command env vars (for example `TVM_BENCH_CMD`, `XLA_BENCH_CMD`, `TENSORRT_BENCH_CMD`, `PYC_GPU_BENCH_CMD`, `GLOW_BENCH_CMD`) to your standardized benchmark command that emits JSON.

## Status

- Stable CI/link targets are in place and cross-platform oriented.
- Experimental compiler pipeline is not yet part of stable CI guarantees.
- Benchmark harness is active for measuring build and runtime behavior of stable targets.

## Documentation

Start at `docs/README.md`.

Key docs:

- `docs/project-status.md`
- `docs/build-and-ci.md`
- `docs/benchmarking.md`
- `docs/results.md`
- `docs/perf-report.md`
- `docs/performance-results.md`
- `REPO_RULES.md`
- `docs/compiler-next/runtime-integration-spec.md`

## Community

- `CODE_OF_CONDUCT.md`
- `CONTRIBUTING.md`
- `SECURITY.md`
- `SUPPORT.md`
- `.github/ISSUE_TEMPLATE/`
- `.github/pull_request_template.md`

## License

Licensed under Apache 2.0. See `LICENSE`.
