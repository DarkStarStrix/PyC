# Runtime Memory Planner

## Purpose

The runtime allocator in `src/compiler/runtime/runtime_allocator.c` builds a reusable buffer plan from tensor lifetime intervals.

## Inputs

`pyc_alloc_request`:

- tensor id
- size in bytes
- alignment
- start step
- end step

## Outputs

`pyc_alloc_plan`:

- per-request offsets
- peak bytes
- reuse count

## Algorithm

Current strategy is:

- first-fit reuse based on non-overlapping lifetime intervals and capacity checks,
- followed by a cost-based rematerialization estimate when the memory budget is exceeded.

The rematerialization estimate is still experimental, but it is no longer only a `largest_allocation / 2` proxy. It now ranks candidate tensors by approximate relief per lifetime span and applies:

- more aggressive relief in `memory_first`,
- one conservative relief step in `balanced`,
- no rematerialization in `utilization_first`.

## Metrics

`pyc_alloc_stats` exposes:

- `peak_bytes`
- `total_requested_bytes`
- `reused_allocations`
- `rematerialized_tensors`
- `rematerialized_bytes`
- `pressure_score`

## Planned Extensions

- cost-based rematerialization
- size-class pools
- async stream-aware lifetimes
