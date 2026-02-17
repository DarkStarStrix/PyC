# Phase 1 Test Matrix (Compiler-Next)

This matrix defines what "comprehensively tested" means for Phase 1.

## Interface and Validation

- `test_ir.c`
  - Valid module verification path.
  - Empty module rejection.
  - Invalid input edge rejection.

## Pass Control Plane

- `test_pass_manager.c`
  - Default pipeline flags.
  - Failure on empty module.
  - Success path and pass-count accounting.

## Runtime Memory Planner

- `test_runtime_allocator.c`
  - Request validation.
  - Reuse behavior on non-overlapping lifetimes.
  - Stats integrity (peak/reuse/total).

## Kernel Selection Surface

- `test_kernel_registry.c`
  - Registration.
  - Priority-based selection.
  - Benchmark update/read aggregation.

## End-to-End API

- `test_compiler_api.c`
  - Invalid argument handling.
  - Verify-fail behavior.
  - Compile + run success path with copy correctness and telemetry.

## Integration Smoke

- `compiler_next_smoke.c`
  - Top-level integration guard for compile/run path with kernel selection.
