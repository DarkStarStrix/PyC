# Core Compiler-Next ↔ AI Bridge Interfaces

This document defines stable contracts between compiler-next and the AI policy bridge.

## 1) Policy contract mapping

Canonical C types live in:

- `include/pyc/optimizer_policy.h` (`pyc_policy_contract`, `pyc_objective_mode`)
- `include/pyc/ai_bridge.h` (`pyc_ai_default_policy_contract`, `pyc_ai_apply_policy_contract`)

Contract:

- `pyc_ai_default_policy_contract(...)` must return deterministic defaults.
- `pyc_ai_apply_policy_contract(...)` maps policy fields into `pyc_compile_options`.
- Non-zero return indicates invalid mapping input.

## 2) Memory and utilization objectives

Contract:

- Policy mode controls planner/kernel selection behavior:
  - `PYC_MODE_MEMORY_FIRST`
  - `PYC_MODE_BALANCED`
  - `PYC_MODE_UTILIZATION_FIRST`
- `memory_budget_bytes` is treated as an explicit pressure budget when non-zero.
- `target_utilization_floor` expresses preferred GPU utilization target for backend-aware selection.

## 3) Determinism contract

Contract:

- `deterministic_strict` must preserve stable behavior for identical inputs/configuration.
- Decision logs from compile path must be reproducible under deterministic mode.
