# Innovation Backlog (Shortlist)

This backlog is designed to improve one or more of:

- performance,
- flexibility,
- uniqueness.

Each item has measurable outcomes and a go/no-go rule.

## Tracker

| ID | Phase | Status | Primary KPI Target | Current Result | Owner |
|---|---|---|---|---|---|
| A1 Adaptive Planner Modes | 3-4 | Planned | `memory_first` >=20% peak-memory reduction; `latency_first` <=5% regression | Pending | Unassigned |
| A2 Reuse + Rematerialization | 3-4 | Planned | >=25% peak-memory reduction with <10% latency penalty | Pending | Unassigned |
| A3 Shape-Clustered Multi-Plan | 4 | Planned | >=1.2x p95 latency improvement on dynamic-shape suite | Pending | Unassigned |
| B1 Kernel + Allocator Co-Selection | 4-5 | Planned | >=10% end-to-end latency gain vs kernel-only selection | Pending | Unassigned |
| B2 Online Feedback Planner | 5 | Planned | >=15% tail-latency drift reduction post warm-up | Pending | Unassigned |
| B3 Deterministic What-If Simulator | 5 | Planned | >=80% predicted-winner agreement vs measured winner | Pending | Unassigned |
| C1 Optimization Contracts | 5 | Planned | 100% contract compliance + >=90% explainable decision logs | Pending | Unassigned |
| C2 Policy Plugin Interfaces | 5 | Planned | New policy integration <200 LOC adapter with no CI determinism regressions | Pending | Unassigned |
| D1 Compile-Runtime Reliability Rails | 4-5 | Planned | 0 silent mismatch incidents; >=95% runs with explainable guard/fallback logs | Pending | Unassigned |

R&D basis and sentiment context for this shortlist:

- `docs/compiler-next/rd-landscape.md`

## Priority A (Start Here)

## A1. Adaptive Planner Modes

### Idea

Support runtime-selectable planning profiles:

- `latency_first`
- `memory_first`
- `balanced`

### Why unique

Most systems expose limited planner policy control; this makes tradeoffs explicit and tunable per workload.

### KPI

- `memory_first`: at least 20% peak memory reduction vs `balanced` on memory stress suite.
- `latency_first`: no more than 5% latency regression vs best static baseline on latency suite.

### Go/No-Go

- Go if both KPI targets hold on at least 2 representative workloads.

### Roadmap Link

- Phase 3 implementation, Phase 4 tuning.

## A2. Dynamic Reuse + Rematerialization Policy

### Idea

Add rematerialization for cheap ops when memory pressure exceeds threshold.

### Why unique

Explicit memory/compute exchange at runtime enables better behavior under constrained GPU memory.

### KPI

- at least 25% peak memory reduction with less than 10% latency penalty.

### Go/No-Go

- Go if memory reduction target is met with latency within budget.

### Roadmap Link

- Phase 3 planner extension, Phase 4 CUDA validation.

## A3. Shape-Clustered Multi-Plan Execution

### Idea

Maintain multiple optimized plans by shape cluster instead of one global plan.

### Why unique

Improves flexibility for dynamic-shape workloads where one plan is suboptimal.

### KPI

- at least 1.2x p95 latency improvement on dynamic-shape benchmark set vs single-plan execution.

### Go/No-Go

- Go if p95 target achieved and determinism tests remain green.

### Roadmap Link

- Phase 4.

## Priority B (High Upside)

## B1. Kernel + Allocator Co-Selection

### Idea

Select kernels using allocator pressure signals (not latency-only ranking).

### Why unique

Joint optimization can avoid memory-thrashing kernels that look fast in isolation.

### KPI

- at least 10% end-to-end latency improvement vs kernel-only selection on memory-pressure workloads.

### Go/No-Go

- Go if end-to-end gains are stable across at least 3 benchmark runs per workload.

### Roadmap Link

- Phase 4 and Phase 5.

## B2. Online Feedback Planner

### Idea

Update planner decisions from runtime telemetry over a bounded window.

### Why unique

Supports adaptation to real production behavior without full recompilation.

### KPI

- at least 15% reduction in tail latency drift after warm-up window.

### Go/No-Go

- Go if variance decreases and determinism mode still reproducible when telemetry learning disabled.

### Roadmap Link

- Phase 5.

## B3. Deterministic What-If Simulator

### Idea

Simulate planner/kernel choices before committing runtime plan.

### Why unique

Lets you evaluate many policies fast while preserving deterministic execution path selection.

### KPI

- at least 80% agreement between predicted winner and measured winner across scenario set.

### Go/No-Go

- Go if prediction reliability target is met.

### Roadmap Link

- Phase 5.

## Priority C (Differentiation)

## C1. User-Facing Optimization Contracts

### Idea

Expose explicit optimization contracts in CLI/API (e.g. memory budget, max compile time, deterministic mode strictness).

### Why unique

Turns compiler behavior into auditable intent rather than opaque heuristics.

### KPI

- 100% contract compliance in contract test suite.
- at least 90% of runs produce explainable decision logs.

### Go/No-Go

- Go if compliance is complete and logs remain deterministic.

### Roadmap Link

- Phase 5.

## D1. Compile-Runtime Reliability Rails

### Idea

Treat correctness, guard handling, compile-budget limits, and fallback explainability as first-class runtime contracts.

### Why unique

Directly targets observed `torch.compile` reliability pain points while preserving optimization velocity.

### KPI

- 0 silent mismatch incidents in correctness suite.
- at least 95% runs include deterministic guard/fallback decision logs.

### Go/No-Go

- Go if guard/fallback behavior is deterministic and correctness parity holds under forced miss scenarios.

### Roadmap Link

- `docs/compiler-next/compile-runtime-reliability-spec.md`

## C2. Policy Plugins for Pass/Planner Decisions

### Idea

Pluggable decision policies with a stable interface for experimentation.

### Why unique

Accelerates research velocity without destabilizing core runtime contracts.

### KPI

- New policy integration in under 200 LOC per policy adapter.
- No regression in core deterministic CI suite.

### Go/No-Go

- Go if plugin integration remains lightweight and safe.

### Roadmap Link

- Phase 5.

## Execution Discipline

1. Every innovation must ship with tests and benchmark scenarios.
2. Bench claims must include median and variance across fixed repeats.
3. No promotion to default behavior without passing KPI thresholds.
4. If KPI misses, keep feature behind flag and document findings.
