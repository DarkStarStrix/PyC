# Phase 2 Test Matrix (Compiler-Next)

## Pass Transformations

- `test_pass_manager.c`
  - pass control flow and report accounting.
- `test_pass_golden.c`
  - deterministic canonicalize/shape-infer/fusion output against golden snapshot.

## Memory Planner v1 Diagnostics

- `test_runtime_allocator.c`
  - verifies peak, reuse, allocation events, overlap metrics, and largest allocation size.

## Determinism Guardrails

- `test_determinism.c`
  - compile/run determinism for identical input/config.

## Interface and Integration Regression Safety

- `test_ir.c`
- `test_kernel_registry.c`
- `test_compiler_api.c`
- `compiler_next_smoke.c`

These remain required to ensure Phase 2 behavior does not regress Phase 1 contracts.
