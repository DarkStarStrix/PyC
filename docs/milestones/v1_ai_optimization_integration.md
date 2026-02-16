# Milestone: v1 AI optimization integration

Acceptance criteria:

1. Graph extraction payload from core IR is produced for all supported tensor ops.
2. Memory planner consumes `MemoryPlanningInput[]` and returns valid `MemoryPlanningOutput[]` without overlaps for active lifetimes.
3. Optimizer pass registry supports deterministic pass ordering and failure propagation.
4. `optimize` command feature flag can be enabled and executes registered passes through the new interface contract.
5. `visualize` and `kernel` contracts are fully specified and validated with integration tests before de-flagging.
