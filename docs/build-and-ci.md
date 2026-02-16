# Build and CI Guide

## Canonical CI Workflow

PyC uses one workflow as the canonical build/test definition:

- File: `.github/workflows/cmake-multi-platform.yml`
- Name: `CI`

### OS Matrix

- `ubuntu-latest`
- `macos-latest`
- `windows-latest`

### CI Steps

1. Checkout repository (`actions/checkout@v4`).
2. Setup CMake (`jwlawson/actions-setup-cmake@v2`).
3. Print toolchain versions.
4. Configure: `cmake -S . -B build -D PYC_BUILD_EXPERIMENTAL=OFF`.
5. Build: `pyc pyc_core pyc_foundation`.
6. Smoke-run `pyc` (OS-specific path).
7. Run `ctest` non-fatally.

## Local Build

### Stable Build

```bash
cmake -S . -B build -D PYC_BUILD_EXPERIMENTAL=OFF
cmake --build build --parallel --target pyc pyc_core pyc_foundation
```

Linux/macOS smoke run:

```bash
./build/pyc
```

Windows smoke run:

```powershell
.\build\Release\pyc.exe
```

### Experimental Build

```bash
cmake -S . -B build -D PYC_BUILD_EXPERIMENTAL=ON
cmake --build build --parallel --target PyC_Core
```

### Benchmark Target Build

```bash
cmake -S . -B build -D PYC_BUILD_EXPERIMENTAL=OFF -D PYC_BUILD_BENCHMARKS=ON
cmake --build build --parallel --target pyc_core_microbench
```

## Build Options

- `PYC_BUILD_EXPERIMENTAL`
  - `OFF` by default.
  - Enables experimental executable target `PyC_Core` when `ON`.
- `PYC_BUILD_BENCHMARKS`
  - `OFF` by default.
  - Enables `pyc_core_microbench` when `ON`.

## Troubleshooting

### `./configure: No such file or directory`

Cause:

- Attempting legacy autotools flow in a CMake-only project.

Fix:

- Use the canonical CMake configure/build commands.

### MSVC parse failures around `__attribute__`

Cause:

- GCC/Clang extension used directly in portable headers or source.

Fix:

- Wrap compiler-specific attributes behind portability macros.

### `sys/wait.h` missing on Windows

Cause:

- POSIX-only include path.

Fix:

- Use `_WIN32` guards and alternate process APIs for Windows.

## Recommended Local Validation Before Push

1. Stable configure/build.
2. `pyc` smoke-run.
3. Optional benchmark run if core code changed.
4. Review generated docs artifacts if benchmark was executed.
