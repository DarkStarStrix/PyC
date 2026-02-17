# Repository Rules (Enforced)

These four rules are mandatory.

## Rule 1: Deterministic CI/CD Builds

- CI/CD builds must be deterministic and reproducible on supported distributions/toolchains.
- Canonical workflow and explicit target builds are required.

Enforcement:

- `.github/workflows/cmake-multi-platform.yml`
- explicit CMake configure/build target list
- mandatory smoke execution

## Rule 2: Tests Required for Features; Fix Code, Not Tests

- Every feature change must include tests.
- Never modify tests to make failing code pass.
- Fix implementation code, rerun validation.
- Tests live under `tests/`.

Enforcement:

- test suite organized in `tests/`
- contributor/test policy docs
- CI runs tests as required checks

## Rule 3: Performance Regressions Are Gated

- Performance changes must stay within guardrail thresholds.
- Regressions beyond thresholds fail CI unless baselines/thresholds are explicitly updated.

Enforcement:

- `benchmark/harness.py`
- `benchmark/tools/check_regression.py`
- `benchmark/baselines/ubuntu-latest.json`
- CI benchmark guardrail step

## Rule 4: Deterministic Compiler Stage Outputs

- Identical inputs/config must produce stable compiler outputs/behavior.
- No hidden randomness in stage behavior.

Enforcement:

- deterministic tests in `tests/compiler_next/`
- allocator/kernel selection consistency checks
- compile/run determinism test coverage
