# Runtime Integration Spec (Practical AI Workflow)

This document explains how to use PyC in real AI workloads where optimization must happen under the runtime during job execution.

## 1) Where PyC Sits

PyC should be embedded in the job runtime path:

1. Job launcher chooses constraints (memory budget, utilization floor, backend).
2. Runtime calls PyC compile API once per model/shape cluster.
3. Runtime calls PyC run API per step/batch.
4. Runtime reads telemetry and may adjust policy mode between phases.

Use PyC as an in-process optimizer/runtime component, not a one-off offline compiler.

## 2) Integration Modes

### A) Embedded runtime (recommended)

- Link `pyc_compiler_next` + `pyc_ai_bridge` directly in trainer/serving worker.
- Lowest latency and easiest policy switching.

### B) Sidecar planner service

- Separate process returns policy/plan decisions.
- Useful for multi-tenant schedulers.
- Higher complexity and IPC overhead.

## 3) Contract Inputs

From launcher/scheduler, pass:

- `mode`: `PYC_MODE_BALANCED`, `PYC_MODE_MEMORY_FIRST`, `PYC_MODE_UTILIZATION_FIRST`
- `memory_budget_bytes`
- `target_utilization_floor`
- `deterministic_strict`

Apply using `pyc_ai_apply_policy_contract(...)` into `pyc_compile_options`.

## 4) Runtime Lifecycle

```c
pyc_policy_contract c;
pyc_compile_options o = {0};
pyc_compiled_model* model = NULL;

pyc_ai_default_policy_contract(&c);
c.mode = PYC_MODE_BALANCED;
c.memory_budget_bytes = budget;
c.target_utilization_floor = 0.80;
pyc_ai_apply_policy_contract(&o, &c);

pyc_compile_model(&desc, &o, &model);
for (step = 0; step < n; ++step) {
  pyc_run_model(model, inputs, in_count, outputs, out_count, &stats);
  // consume stats + decision log for control loop
}
pyc_destroy_model(model);
```

## 5) Dynamic Policy Control Loop

Recommended switching rules:

1. Warmup phase: `PYC_MODE_UTILIZATION_FIRST` for device saturation.
2. Steady phase: `PYC_MODE_BALANCED` for throughput + memory stability.
3. Pressure spike (high `pressure_score` or repeated `pressure_events`): switch to `PYC_MODE_MEMORY_FIRST`.
4. Recovery window: move back to `BALANCED` after pressure clears.

## 6) Telemetry to Wire Into Orchestrator

Collect per run from `pyc_run_stats`:

- `peak_bytes`, `total_requested_bytes`, `reused_allocations`
- `rematerialized_tensors`, `pressure_events`, `pressure_score`
- `estimated_utilization`, `selected_kernel_score`, `selected_kernel_symbol`
- `compile_ms`, `run_ms`

Also record `pyc_model_last_decision_log(model)` for deterministic audit.

## 7) Rollout Plan for Production

1. Shadow mode: run PyC decisions + telemetry without enforcing policy changes.
2. Canary: enable policy switching for 5-10% workloads.
3. Guardrails: auto-fallback to previous mode on latency/throughput regressions.
4. Full rollout only after stable KPI trend across fixed benchmark + live canary windows.

## 8) Failure and Fallback Policy

- Compile failure: keep last-known-good compiled model.
- Runtime pressure breach: force `PYC_MODE_MEMORY_FIRST` for next window.
- Determinism required workflows: keep `deterministic_strict=1` and block non-deterministic policy changes.

## 9) KPI Targets (Operational)

Track and enforce:

1. Peak memory reduction: >=25% in memory-pressure suites.
2. Utilization gain: >=10% in utilization-first workloads.
3. End-to-end improvement: >=1.2x throughput or equivalent p95 latency gain on reference workloads.
4. Determinism: identical inputs/config produce identical decision logs.
