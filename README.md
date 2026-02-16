# PyC

Lightweight compiler/toolchain project with a canonical cross-platform CMake build and stable core targets for deterministic CI.

[![CI](https://github.com/DarkStarStrix/PyC/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/DarkStarStrix/PyC/actions/workflows/cmake-multi-platform.yml)
![Release](https://img.shields.io/badge/Release-Alpha-orange)
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

Experimental compiler code remains available behind `PYC_BUILD_EXPERIMENTAL=ON`.

## Repository Layout

- `Core/C_Files/`: C sources.
- `Core/Header_Files/`: C headers.
- `.github/workflows/cmake-multi-platform.yml`: canonical CI workflow.
- `benchmark/`: benchmark harness and workloads.
- `docs/`: project docs, benchmarking, build/CI, performance reports.

## Build

### Prerequisites

- CMake `>= 3.10`
- C compiler with C11 support
- Python 3 (for benchmark harness)

### Configure + Build Stable Targets

```bash
cmake -S . -B build -D PYC_BUILD_EXPERIMENTAL=OFF
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

1. CMake configure with `PYC_BUILD_EXPERIMENTAL=OFF`
2. Explicit build of `pyc`, `pyc_core`, `pyc_foundation`
3. OS-specific smoke test for `pyc`
4. Non-fatal `ctest`

## Benchmarking

PyC includes a deterministic benchmark harness for stable core targets.

Run:

```bash
python3 benchmark/harness.py --repeats 7 --micro-rounds 4000
```

Outputs:

- `benchmark/results/latest.json`
- `benchmark/results/latest.md`
- `docs/performance-results.md`

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
- `docs/performance-results.md`

## License

Licensed under Apache 2.0. See `LICENSE`.
