# Compile Runtime Reliability Spec (PyC vs `torch.compile` Pain Points)

## Purpose

Define a PyC implementation path that addresses common `torch.compile` pain points:

1. silent correctness risk,
2. high and unpredictable compile latency,
3. graph breaks/guard friction,
4. platform/toolchain fragility,
5. inconsistent hardware gains.

This spec is the execution contract for reliability-first compiler-next behavior.

## Observed Failure Modes (Research-Derived)

## F1: Silent Correctness Drift

### Typical Symptoms

1. Compiled outputs differ materially from eager outputs.
2. In severe cases, eager behavior appears contaminated after compiled execution.

### Repro Patterns Seen in Public Reports

1. Alias/view + non-trivial storage/offset alignment interactions.
2. Transformer attention/path-specific numerical divergence under compiled mode.
3. Fused-kernel read/write state interactions causing miscompilation.

### Risk

Silent model-quality degradation in production, which is the highest-severity compiler failure class.

## F2: Recompilation Storms (Guard Churn)

### Typical Symptoms

1. Frequent recompiles triggered by guard failures (shape/stride/etc.).
2. Latency spikes and high warmup instability.
3. Eventual fallback to eager after recompile limits.

### Root Mechanism

`torch.compile` guard model is soundness-oriented; changing assumptions can trigger re-trace/recompile.

## F3: Graph-Break Fragmentation

### Typical Symptoms

1. Partial graph capture with many breaks.
2. Lost fusion and optimization opportunities.
3. Throughput below expectations despite compilation enabled.

### Common Trigger Classes

1. Data-dependent branching and loops.
2. Scalar extraction patterns (`.item`) and unsupported Python/C paths.

## F4: Cold-Start Compile Latency

### Typical Symptoms

1. First invocation latency is seconds/minutes; large models may be much longer.
2. Distributed jobs can timeout while waiting for compile warmup.

### Context

JIT compile overhead is expected; caching reduces it, but cache misses/regressions remain operationally significant.

## F5: Platform/Toolchain Fragility

### Typical Symptoms

1. Missing/incorrect compiler toolchain setup (especially on Windows).
2. Container/runtime linkage failures (`g++` missing, `-lcuda` unresolved, runtime mismatch).

### Operational Impact

High setup friction and non-deterministic rollout behavior across developer and CI environments.

## Design Requirements

1. Correctness before speed:
   - no silent wrong outputs;
   - guarded fallback to known-correct execution path on invariant miss.
2. Bounded compile overhead:
   - explicit compile budget modes (`fast_dev`, `balanced`, `max_perf`);
   - shape-bucket artifact cache to avoid repeated cold compile penalties.
3. Explainability:
   - deterministic decision log with guard hits/misses, graph-break reasons, fallback counts.
4. Portable deterministic setup:
   - fail-fast toolchain preflight for Linux/macOS/Windows;
   - backend availability matrix emitted at startup.
5. Stable performance claims:
    - benchmark gating requires repeated runs and variance bounds, not single-run wins.

## High-Resolution Target Pain Point

## Target: "Silent Wrong Output Under Dynamic/Aliasing Workloads"

This is the top pain point to solve first because it destroys trust in compiler deployment.

### Problem Definition

For workloads with dynamic shapes, aliasing/view-heavy tensors, and stateful/fused paths, compiled execution must never produce silent mismatches versus trusted eager reference.

### Reliability Contract

1. PyC must perform invariant checks at compile/run boundaries (shape, dtype, layout/alias metadata).
2. Any invariant uncertainty must trigger explicit fallback, never silent continuation.
3. Every fallback must emit deterministic machine-readable reason codes.

### Acceptance Criteria

1. Zero silent mismatches on targeted stress suite:
   - alias/view/storage-offset cases,
   - dynamic-shape sequences,
   - state read/write fused-path cases.
2. 100% of mismatches become explicit failures or explicit fallbacks with reason code.
3. Re-run determinism: same input trace -> same guard/fallback event sequence.

### Baseline Metrics to Track

1. `silent_mismatch_count` (must remain 0).
2. `guard_miss_count`, `fallback_count`, and top fallback reasons.
3. `compile_ms_cold`, `compile_ms_warm`, `run_p50/p95`.
4. `recompile_count_per_1k_calls`.

## API and Runtime Additions

Planned additions under `include/pyc/` and runtime internals:

1. `pyc_compile_options` extensions:
   - `compile_budget_ms`
   - `cache_mode`
   - `guard_strictness`
   - `fallback_policy`
2. Runtime stats extensions:
   - `compile_cache_hit`
   - `guard_miss_count`
   - `graph_break_count`
   - `fallback_count`
   - `compile_budget_exceeded`
3. Diagnostic interface:
   - `pyc_model_last_decision_log(...)` must include structured guard/fallback events.

## Implementation Plan

## R1: Guarded Correctness Rails

1. Implement shape/dtype/layout guard checks at run boundary.
2. On guard miss, execute fallback path and mark reason.
3. Add deterministic guard/fallback logging.

Exit: correctness tests pass with forced guard misses.

## R2: Compile Budget + Cache

1. Add compile budget enforcement and mode-specific optimization depth.
2. Add shape-bucket compile cache keying and artifact reuse.
3. Emit cache hit/miss and budget-overrun metrics.

Exit: second run shows cache-hit behavior and lower compile latency.

## R3: Graph-Break Visibility

1. Emit graph-break report by op and reason.
2. Add preflight “compilability score” and break counters.
3. Keep execution correct via fallback, never silent mismatch.

Exit: break reasons are reproducible in golden diagnostics.

## R4: Cross-Platform Determinism

1. Add unified toolchain preflight script and CI check.
2. Normalize platform diagnostics in benchmark artifacts.
3. Fail fast with actionable error messages when backend prerequisites are missing.

Exit: deterministic pass/fail behavior across Linux/macOS/Windows CI.

## Test and Benchmark Gates

1. Correctness:
   - golden output parity between optimized and fallback paths;
   - forced guard-miss tests in `tests/compiler_next/`.
2. Reliability:
   - no silent result mismatch tests;
   - compile budget overrun behavior tests.
3. Performance:
   - p50/p95/variance across repeated runs;
   - compile-latency and cache-hit metrics captured per run.
4. Promotion:
   - features remain behind flags until correctness + reliability + performance gates pass.

## Near-Term Deliverables

1. Add reliability counters into `pyc_run_stats`. (implemented)
2. Add guard/fallback deterministic tests. (implemented)
3. Add compile-cache smoke benchmark (`cold` vs `warm`). (implemented via `tests/compiler_next/test_compile_cache.c`)
4. Add graph-break diagnostic report format and golden test. (implemented foundation via pass report + `tests/compiler_next/test_graph_break_reporting.c`)

## Evidence References

1. PyTorch `torch.compile` troubleshooting (compile times, graph breaks, guards/recompiles):  
   https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_troubleshooting.html
2. PyTorch common graph breaks (data-dependent control flow and `.item` class):  
   https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.common_graph_breaks.html
3. PyTorch recompilation guide (guard failures and mitigation context):  
   https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.recompilation.html
4. PyTorch compile API behavior (guard failures, recompile limit fallback):  
   https://docs.pytorch.org/docs/stable/generated/torch.compile.html
5. PyTorch compile-time caching guides (cold-start mitigation mechanics):  
   https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html
6. Windows `torch.compile` setup guidance (MSVC/toolchain requirements):  
   https://docs.pytorch.org/tutorials/unstable/inductor_windows.html
7. Representative correctness issue class examples from PyTorch tracker:  
   https://github.com/pytorch/pytorch/issues/155690  
   https://github.com/pytorch/pytorch/issues/162722  
   https://github.com/pytorch/pytorch/issues/164701
