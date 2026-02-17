# Compiler-Next Pass Pipeline

The pass pipeline is controlled by `pyc_pass_pipeline` in `include/pyc/pass_manager.h`.

## Stages

1. Canonicalization
2. Shape inference
3. Layout propagation
4. Fusion
5. Liveness analysis
6. Allocation planning
7. Lowering

## Current Behavior

Current implementation tracks stage execution and validates pipeline-level success. It is intentionally lightweight and acts as the stable control plane before full pass implementations land.

## Near-Term Work

- Implement concrete IR rewrites for fusion and canonicalization.
- Persist pass diagnostics and transformation stats.
- Add pass-level golden tests.
