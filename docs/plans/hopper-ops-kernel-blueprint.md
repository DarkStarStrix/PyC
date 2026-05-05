# Hopper Ops + Kernel Blueprint

## Intent

This is the Hopper bring-up order for PyC:

1. make the ops/runtime layer cheap and stable enough that the GPU stays fed,
2. then specialize kernels so the busy time is productive,
3. keep both layers tied together through the kernel registry, controller, and telemetry.

The point is not to chase isolated TFLOPS wins. The point is to keep orchestration tax from burying kernel wins, then let Hopper-native paths pay off on top of a stable runtime.

## Layer 1: Ops / Runtime First

The first Hopper work should live in the reusable runtime path.

Priority items:

- eliminate hot-path `cudaMalloc` churn
- cache `cublasLt` descriptors, layouts, preferences, and heuristics for repeated shapes
- keep persistent device buffers alive across runs
- default repeated executions toward graph replay and replay-safe execution paths
- classify workloads by shape family so small, large-square, tall-skinny, and wide-skinny paths do not all share one selection surface
- make kernel selection hardware-aware so Ada-specialized paths do not accidentally become the Hopper default

The runtime win condition is:

- less setup time
- less allocator churn
- fewer dead gaps between launches
- deterministic selection behavior
- telemetry that explains why a path was chosen

## Layer 2: Kernel Work After Runtime Stability

Once the ops path is stable, kernel work should proceed in layers:

1. vectorized movement and cheaper memory-path instruction mix
2. shared-memory bank-conflict validation and cleanup
3. arithmetic-intensity improvement
4. shape-specialized tile families instead of one universal tile
5. Hopper-native extensions:
   - TMA for bulk async movement
   - WGMMA / tensor-core-native paths where the math mode supports it
   - deeper producer-consumer overlap
   - deliberate epilogue design so output staging does not become the next bottleneck

The kernel win condition is:

- more useful math per unit of movement
- more useful math per scheduling slot
- specialization by workload family, not one-size-fits-all tuning

## Architecture Rule

PyC should treat Hopper optimization as a coordinated system:

- compiler-next passes produce the right execution context
- runtime planning stabilizes memory and launch behavior
- kernel registry selects by workload family and hardware family
- promoted kernels plug into a runtime that already knows when they should win
- telemetry makes every selection explainable

So the system contract is:

- optimize ops so the machine stays busy
- optimize kernels so the busy time is productive
- design PyC so those layers reinforce each other instead of competing

## First Experimental Slice

The first Hopper-oriented implementation slice should do two things:

1. exact-shape `cublasLt` plan caching in the CUDA workspace
2. workload-family + hardware-family aware kernel co-selection

That gives Hopper a useful control surface immediately, even before a Hopper-native `.cu` kernel lands.

## Promotion Bar

Before moving deeper into Hopper kernel work, the runtime layer should show:

- repeated-shape runs avoid descriptor/heuristic rebuilds
- decision logs show workload family and hardware family
- kernel selection can prefer Hopper-specialized kernels when they exist
- no deterministic regressions in compile-cache, speculative-plan, phantom-graph, or controller tests
