# Contributing

## Repository Rules (Required)

1. CI/CD must remain deterministic.
2. Every feature must include tests.
3. Never modify tests to make broken code pass.
4. Fix code first, then rerun validation.
5. Performance regressions are blocked by guardrail checks.
6. Compiler stage outputs must be deterministic for identical inputs/config.
7. Kernel staging work should stay in `kernels/lab/` and `kernels/prototypes/` until it is benchmark-backed.

See `docs/reference/repository-rules.md` for full rule definitions and enforcement map.

## Layout Rules

1. Keep root-level files minimal and intentional.
2. Put planning docs under `docs/plans/`, reports under `docs/reports/`, and reference material under `docs/reference/`.
3. Do not add free-floating `.cu` files under shared buckets such as `kernels/`, `src/compiler/cutlass_kernels/`, or `benchmark/workloads/`.
4. Put each kernel family in its own nested folder with colocated notes/manifests when needed.
5. Keep website source and published site data under `web/site/`.

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
cmake --build build --parallel --target pyc_compiler_next pyc_ai_bridge pyc_compiler_next_smoke pyc_compiler_next_test_ir pyc_compiler_next_test_pass_manager pyc_compiler_next_test_runtime_allocator pyc_compiler_next_test_kernel_registry pyc_compiler_next_test_compiler_api pyc_compiler_next_test_determinism pyc_compiler_next_test_pass_golden pyc_compiler_next_test_policy_modes pyc_compiler_next_test_runtime_control pyc_compiler_next_test_ai_bridge
ctest --test-dir build -C Release --output-on-failure
```

If your change touches kernel staging or remote GPU prep, also verify the staging path:

```bash
python3 kernels/lab/kernel_lab.py doctor
python3 kernels/lab/kernel_lab.py bench-suite --dry-run --tag ada
python3 benchmark/benchmarks/gpu/run_gemm_suite.py --matrix-file benchmark/benchmarks/gpu/configs/ada_fp32_gemm_shapes.json --dry-run
```

## Repository Guardrails

```bash
python3 tests/validate_repo_layout.py .
python3 tests/validate_cmake_sources.py CMakeLists.txt .
```

## Benchmark Guardrail Validation

```bash
python3 benchmark/harness.py --repeats 5 --micro-rounds 2000
python3 benchmark/tools/check_regression.py --baseline benchmark/baselines/ubuntu-latest.json --result benchmark/benchmarks/results/json/latest_core.json
```
