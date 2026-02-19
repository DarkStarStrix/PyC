# Compiler-Next Production Readiness (Phase 6)

This document defines how PyC moves from experimental performance work to deployable behavior in real application flows.

## Scope

Phase 6 is focused on deployability, not broad feature invention:

1. Stable API/status behavior under bad inputs and runtime failures.
2. Deterministic fallback and rollback behavior.
3. Actionable observability for incident response.
4. Repeatable promotion criteria before releases.

## Production Gate Suites

Required tests:

1. `pyc_compiler_next_test_prod_status_errors`
2. `pyc_compiler_next_test_prod_decision_log`
3. `pyc_compiler_next_test_prod_cuda_contracts`
4. `pyc_compiler_next_test_prod_runtime_rollback`

Run only production gates:

```bash
ctest --test-dir build --output-on-failure -R pyc_compiler_next_test_prod_
```

Run full compiler-next suite:

```bash
ctest --test-dir build --output-on-failure
```

## Commercial Readiness Checklist

A release candidate is blocked unless all items pass:

1. Contract safety:
- invalid arguments return explicit `PYC_STATUS_INVALID_ARGUMENT`
- invalid IR returns `PYC_STATUS_VERIFY_FAILED`
2. Runtime determinism:
- no silent success on known runtime failure paths
- fallback and error reasons are explicit and stable
3. Runtime control:
- rollback rails trigger on sustained runtime-error breaches
- rollback reason is externally visible in stats/logs
4. Observability:
- decision log includes mode, fallback, contract, graph-break, and fingerprint fields
- `pyc_run_stats` fields are parseable and populated for monitoring ingestion
5. Platform matrix:
- stable targets green on Linux/macOS/Windows
- CUDA path either native-valid or deterministic fallback-valid

## Metrics Required for Promotion

1. Correctness:
- differential pass rate against reference runtime on target model set
2. Reliability:
- fallback-rate threshold and guard-miss threshold per workload
3. Performance:
- p50/p95 latency and throughput tracked against baseline runs
4. Reproducibility:
- benchmark artifacts stamped with run metadata and committed/published

## Next Milestones After Initial Phase 6

1. ONNX/Torch FX ingestion path for non-handwritten IR workflows.
2. Versioned compatibility policy for operator support.
3. Service integration guide (embedding PyC under runtime launch path).
4. Release bundle structure and upgrade/deprecation policy.
