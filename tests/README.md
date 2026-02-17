# Test Policy

This repository treats tests and CI determinism as hard requirements.

## Enforced Rules

1. CI/CD builds must be deterministic.
2. Every feature change must include tests.
3. Never edit tests to make failing code pass.
4. Fix code first, then rerun validation.
5. Performance regressions are gated by benchmark guardrails.
6. Compiler stage behavior must be deterministic for identical inputs/config.

## Test Layout

- Compiler-next tests: `tests/compiler_next/`

## Iteration Workflow

1. Implement feature/fix.
2. Run test suite.
3. If failures occur, fix code only.
4. Re-run tests until green.

## Compiler-Next Validation Command

```bash
cmake -S . -B build -D PYC_BUILD_COMPILER_NEXT=ON -D PYC_BUILD_COMPILER_NEXT_TESTS=ON -D PYC_BUILD_EXPERIMENTAL=OFF
cmake --build build --parallel --target pyc_compiler_next pyc_compiler_next_smoke pyc_compiler_next_test_ir pyc_compiler_next_test_pass_manager pyc_compiler_next_test_runtime_allocator pyc_compiler_next_test_kernel_registry pyc_compiler_next_test_compiler_api pyc_compiler_next_test_determinism pyc_compiler_next_test_pass_golden
ctest --test-dir build -C Release --output-on-failure
```
