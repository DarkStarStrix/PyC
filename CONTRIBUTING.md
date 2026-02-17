# Contributing

## Repository Rules (Required)

1. CI/CD must remain deterministic.
2. Every feature must include tests.
3. Never modify tests to make broken code pass.
4. Fix code first, then rerun validation.
5. Performance regressions are blocked by guardrail checks.
6. Compiler stage outputs must be deterministic for identical inputs/config.

See `docs/REPO_RULES.md` for full rule definitions and enforcement map.

## Test Layout

- All tests live under `tests/`
- Compiler-next tests live under `tests/compiler_next/`

## Required Validation Flow

1. Implement code changes.
2. Build targets.
3. Run tests.
4. If tests fail, fix code only.
5. Re-run until green.

## Compiler-Next Validation Command

```bash
cmake -S . -B build -D PYC_BUILD_COMPILER_NEXT=ON -D PYC_BUILD_COMPILER_NEXT_TESTS=ON
cmake --build build --parallel --target pyc_compiler_next pyc_ai_bridge pyc_compiler_next_smoke pyc_compiler_next_test_ir pyc_compiler_next_test_pass_manager pyc_compiler_next_test_runtime_allocator pyc_compiler_next_test_kernel_registry pyc_compiler_next_test_compiler_api pyc_compiler_next_test_determinism pyc_compiler_next_test_pass_golden pyc_compiler_next_test_policy_modes pyc_compiler_next_test_ai_bridge
ctest --test-dir build -C Release --output-on-failure
```

## Benchmark Guardrail Validation

```bash
python3 benchmark/harness.py --repeats 5 --micro-rounds 2000
python3 benchmark/tools/check_regression.py --baseline benchmark/baselines/ubuntu-latest.json --result benchmark/results/latest.json
```
