# Repository Guidelines

## Project Structure & Module Organization
- `Core/`: stable C core (`Core/C_Files`, `Core/Header_Files`) used for canonical CI targets.
- `compiler/` + `include/pyc/`: experimental compiler-next stack (IR, passes, runtime, public APIs).
- `tests/`: all tests; compiler-next tests live in `tests/compiler_next/`.
- `benchmark/`: deterministic harness, baselines, and regression guardrails.
- `docs/`: technical documentation and roadmap material.
- `.github/workflows/cmake-multi-platform.yml`: canonical cross-platform CI workflow.

## Build, Test, and Development Commands
- Configure stable build:
  ```bash
  cmake -S . -B build -D PYC_BUILD_EXPERIMENTAL=OFF
  ```
- Build stable CI targets:
  ```bash
  cmake --build build --parallel --target pyc pyc_core pyc_foundation
  ```
- Run stable smoke binary:
  ```bash
  ./build/pyc
  ```
- Build and run compiler-next tests:
  ```bash
  cmake -S . -B build -D PYC_BUILD_COMPILER_NEXT=ON -D PYC_BUILD_COMPILER_NEXT_TESTS=ON -D PYC_BUILD_EXPERIMENTAL=OFF
  cmake --build build --parallel
  ctest --test-dir build -C Release --output-on-failure
  ```
- Run performance guardrails:
  ```bash
  python3 benchmark/harness.py --repeats 5 --micro-rounds 2000
  python3 benchmark/tools/check_regression.py --baseline benchmark/baselines/ubuntu-latest.json --result benchmark/results/latest.json
  ```

## Coding Style & Naming Conventions
- Language baseline: C11 (`target_compile_features(... c_std_11)`).
- Use 4-space indentation, braces on same line, and keep functions small and deterministic.
- Naming: `snake_case` for functions/variables, `PYC_*` for enums/macros/constants, descriptive target names (`pyc_compiler_next_test_*`).
- Prefer explicit includes and portable code paths (MSVC + Unix-compatible).

## Testing Guidelines
- Every feature change must add/extend tests in `tests/`.
- Do not weaken or rewrite tests to pass broken code; fix implementation first.
- Add deterministic coverage for compiler stages when behavior changes (serialization, pass output, allocator/kernel selection).

## Commit & Pull Request Guidelines
- Commit messages should be short, imperative, and specific (e.g., `Fix Windows MSVC portability`, `Polish docs and add benchmark visualization`).
- PRs should include: summary, rationale, validation commands run, and risk notes.
- For performance-sensitive changes, attach benchmark deltas and guardrail results.
- CI determinism and cross-platform build stability are merge-blocking requirements.
