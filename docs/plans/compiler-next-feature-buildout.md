# Compiler-Next Feature Buildout Plan

## Purpose

This plan turns the remaining compiler-next backlog into an execution order that can be implemented and validated without reopening repo-structure debt.

The target state for each feature is:

1. implemented behind experimental compiler-next paths,
2. covered by deterministic tests,
3. exercised in a realistic hot-path workflow when possible,
4. judged before promotion.

This repo is an ideas lab, so the bar is not "production default." The bar is "experimentally real, measurable, and explainable."

## Current State

The live compiler-next stack already includes:

- deterministic run-boundary guards,
- in-memory compile cache,
- speculative plan variants,
- shadow-first phantom graph tracking,
- budget-sensitive rematerialization telemetry,
- runtime controller rails,
- kernel + allocator co-selection,
- CUDA fast-path execution with cuBLASLt-first FP32 GEMM.

The next work is to turn those pieces into a broader experimental system instead of a set of isolated slices.

## Execution Board

### B1. Kernel + Allocator Co-Selection

Status: experimental, implemented

What exists now:

- kernel choice already consumes allocator pressure context,
- decision logs and stats expose joint-selection fields,
- policy-mode tests cover memory-first vs utilization-first behavior.

Files:

- `src/compiler/compiler_api.c`
- `src/compiler/runtime/kernel_registry.c`
- `tests/compiler_next/test_policy_modes.c`
- `tests/compiler_next/test_production_decision_log.c`

Promotion bar:

- stable end-to-end improvement on pressure-heavy workloads,
- no deterministic regression in selection logs.

### A3. Shape-Clustered Multi-Plan Execution

Status: experimental, hot-path validated

What exists now:

- speculative plans maintain multiple shape-family variants,
- runtime chooses variants by shape signature and bucket,
- mixed-shape single-process benchmark mode validates realistic drift/reshape behavior in the hot path,
- sequence reporting now keeps per-step phantom and rematerialization telemetry visible.

Files:

- `src/compiler/compiler_api.c`
- `benchmark/benchmarks/gpu/workloads/pyc_compiler_next_bench.c`
- `tests/compiler_next/test_speculative_plans.c`
- `tests/test_pyc_bench_sequence.py`

What is still missing:

- broader cluster generation than the current bounded scale-family set,
- plan families for non-trivial op skeleton drift,
- tighter reporting and analysis around multi-step dynamic-shape runs.

Promotion bar:

- at least 1.2x p95 improvement on dynamic-shape benchmark set vs single-plan execution,
- deterministic mixed-shape tests remain green,
- no guard-miss regressions in the realistic sequence path.

### A2. Dynamic Reuse + Rematerialization Policy

Status: experimental, hot-path validated under budget

Goal:

- replace the current coarse rematerialization estimate with a more defensible cost-based policy,
- preserve determinism,
- make pressure handling explainable enough to use in controller and kernel-selection experiments.

Files:

- `src/compiler/runtime/runtime_allocator.c`
- `include/pyc/runtime_allocator.h`
- `tests/compiler_next/test_runtime_allocator.c`
- `tests/compiler_next/test_policy_modes.c`

Experimental target for this pass:

- choose rematerialization relief using request cost proxies instead of only `largest_allocation / 2`,
- keep `memory_first` more aggressive than `balanced`,
- expose enough stats to judge whether the planner is behaving coherently.

Promotion bar:

- measurable peak-memory reduction on constrained runs,
- less than 10% latency penalty on selected workloads,
- deterministic allocator stats under repeated runs.

### B2. Online Feedback Planner

Status: experimental, implemented locally and ready for hot-path validation

What exists now:

- runtime controller observes bounded metrics and classifies pressure, latency, throughput, and runtime-error signals,
- `recommended_mode` and `recommendation_reason` are tracked independently from the applied mode,
- auto-switch, dwell, cooldown, and hard rollback rules bound adaptation deterministically,
- later plan selection can consume the applied mode while the decision log still shows the controller's shadow recommendation.

What is still missing:

- hot-path CUDA validation of the controller on a repeated-run workload,
- benchmarkable drift-reduction criteria beyond deterministic unit coverage.

Files:

- `src/compiler/compiler_api.c`
- `tests/compiler_next/test_runtime_control.c`

Promotion bar:

- reduced tail-latency drift after warm-up,
- deterministic behavior when learning/adaptation is disabled.

### B3. Deterministic What-If Simulator

Status: planned

Goal:

- predict likely winners before live benchmarking,
- reuse planner/kernel/allocator signals without making runtime choice non-deterministic.

Likely edit points:

- `src/compiler/compiler_api.c`
- `src/compiler/runtime/kernel_registry.c`
- benchmark analysis/reporting scripts

Promotion bar:

- at least 80% predicted-winner agreement vs measured winner on a fixed scenario suite.

### C1. Optimization Contracts

Status: planned

Goal:

- make planner behavior auditable via explicit contracts like memory budget, compile budget, deterministic strictness, and utilization floor.

Base surface already exists:

- `include/pyc/compiler_api.h`
- `include/pyc/optimizer_policy.h`

Promotion bar:

- 100% contract compliance in tests,
- explainable logs on at least 90% of runs.

### C2. Policy Plugin Interfaces

Status: planned

Goal:

- decouple future policy experimentation from core compiler code,
- let new planning logic land behind adapters instead of invasive rewrites.

Promotion bar:

- lightweight policy integration with no deterministic CI regressions.

### Product Hardening

Status: deferred until the experimental board is deeper

Includes:

- importer paths,
- production-readiness cleanup,
- rollout/SLO criteria,
- wider compatibility and promotion gates.

## Implementation Order

1. Finish `A3` as a genuine experimental dynamic-shape surface.
   - broaden sequence analysis and hot-path reporting.
2. Build `A2` into a more coherent rematerialization policy.
   - allocator first, then pressure-aware GPU checks.
3. Elevate `B2` from controller scaffolding to real feedback-planner experiments.
4. Add `B3` simulator once planner inputs are rich enough to predict from.
5. Formalize `C1` contracts, then `C2` plugins.
6. Only then do broader product hardening.

## Hot-Path Validation Matrix

Each experimental feature should be validated in the most realistic path it can support.

### Shape / Phantom / Speculative Features

Use:

- long-lived mixed-shape `pyc_compiler_next_bench` sequences,
- bounded shape families that stay inside speculative-plan coverage,
- hot-path CUDA runs on the Ada VM.

Minimum evidence:

- mismatch and reshape counters move when the shape family drifts,
- guard misses stay at zero on valid family transitions,
- fallback count stays at zero in the intended path.

### Allocator / Rematerialization Features

Use:

- deterministic allocator tests locally,
- budget-constrained compile/run paths,
- GPU checks with explicit low memory budgets where pressure is intentionally induced.

Minimum evidence:

- pressure events are explainable,
- rematerialization differs by policy mode,
- latency cost remains bounded.

## Canonical Operator Flow

Use one of two paths:

1. Fixed-shape Ada sweeps for throughput comparison.
2. Mixed-shape `gemm_sequence` runs for A3/A2 judgment.

After the remote run lands on the Ada box, pull and judge it with:

```bash
bash scripts/pull_and_analyze_ada_artifacts.sh
```

The script stages the latest remote sweep, copies the matching kernel-lab result, and emits the local analysis bundle with raw JSON, graphs, sheets, and rankings in one pass.

### Planner / Feedback Features

Use:

- repeated-run windows on a fixed workload,
- shadow-mode telemetry before any adaptive behavior becomes authoritative.

Minimum evidence:

- decision logs explain the shift,
- variance or fallback behavior improves,
- determinism remains intact when adaptation is disabled.

## Current Pass

This pass should accomplish two things:

1. turn this document into an accurate execution board,
2. push `A2` forward so the next compiler-next slice is experimentally real rather than only planned.

Definition of done for this pass:

- this buildout doc reflects current implemented state,
- `A2` moves from coarse heuristic toward cost-based rematerialization,
- deterministic tests cover the new behavior,
- at least one hot-path run exercises the feature under real CUDA execution conditions.
