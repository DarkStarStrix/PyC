# Runtime Memory Planner

## Purpose

The runtime allocator in `compiler/runtime/runtime_allocator.c` builds a reusable buffer plan from tensor lifetime intervals.

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

Current strategy is first-fit reuse based on non-overlapping lifetime intervals and capacity checks.

## Metrics

`pyc_alloc_stats` exposes:

- `peak_bytes`
- `total_requested_bytes`
- `reused_allocations`

## Planned Extensions

- cost-based rematerialization
- size-class pools
- async stream-aware lifetimes
