# Compiler-Next Feature Buildout Plan

## Purpose

This plan turns the remaining compiler-next backlog into an execution order that can be implemented without reopening repo-structure debt.

## Execution Order

1. `B1` Kernel + allocator co-selection
   - Route kernel choice through allocator outcomes, not pressure score alone.
   - Expose joint-selection diagnostics in decision logs and stats.
   - Status: in progress in this change.

2. `A3` Shape-clustered multi-plan execution
   - Extend the current speculative-plan layer into true shape-family variants.
   - Reuse compile artifacts by bucketed dynamic-shape signatures.

3. `A2` Dynamic reuse + rematerialization policy
   - Replace coarse rematerialization estimates with cost-based decisions.
   - Add pressure-aware remat tests and benchmark gates.

4. `B2` Online feedback planner
   - Let bounded runtime telemetry influence later plan choice.
   - Keep deterministic shadow-mode logs before enabling adaptation.

5. `B3` Deterministic what-if simulator
   - Predict likely winners before paying the cost of live benchmarking.
   - Measure prediction agreement against actual winners.

6. `C1` Optimization contracts and `C2` policy plugin interfaces
   - Turn planner behavior into auditable contracts.
   - Decouple future policy experimentation from core compiler code.

7. Product hardening
   - importer paths
   - production-readiness cleanup
   - SLO-backed rollout criteria

## Current Slice

The implementation work for this pass is focused on `B1`.

Definition of done for this slice:

- kernel selection can consume allocator outcomes beyond raw pressure score,
- compile and run decision logs explain the joint choice,
- deterministic tests cover the new selector behavior.
