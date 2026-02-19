# Compiler-Next Roadmap by Phase

## Summary

This roadmap defines how PyC moves from a stable CI core to a performance-first compiler/runtime trajectory aligned with TVM/XLA-class goals:

- end-to-end latency wins,
- dynamic memory optimization,
- backend-aware kernel optimization,
- and reproducible benchmarking + promotion gates.

Selected strategy:

1. Goal: performance-first compiler/runtime.
2. Backend order: CPU first, then CUDA.
3. Kernel order: GEMM + fused epilogues, then reductions/layernorm, then conv2d.
4. KPI priority: end-to-end latency (with kernel and memory metrics tracked as required secondary KPIs).
5. IR strategy: custom IR now, MLIR bridge later.

## Success Criteria

1. End-to-end latency: at least 1.5x speedup on 2 reference workloads by end of Phase 3.
2. Peak memory: at least 25% reduction on dynamic-memory benchmarks.
3. Kernel performance: at least 1.3x for GEMM family and at least 1.2x for reduction/layernorm family.
4. Stability: keep stable CI (`pyc`, `pyc_core`, `pyc_foundation`) green across Linux/macOS/Windows.

## Scope Boundaries

1. Stable CI contract remains minimal and must not regress.
2. Compiler-next evolves under explicit experimental flags.
3. Promotion to stronger guarantees occurs only after correctness + portability + performance gates.
4. Full Python language compatibility is not a near-term goal.

## Target Architecture

1. High-level IR layer (`compiler/ir`) with verifier and typed tensor semantics.
2. Pass pipeline (`compiler/passes`) for canonicalization, shape/layout, fusion, liveness, and lowering prep.
3. Lowering + codegen path (CPU first, CUDA second).
4. Runtime (`compiler/runtime`) with allocator planning, kernel selection, and execution telemetry.
5. Autotune control plane for candidate search and best-kernel persistence.

## Public Interfaces (Current + Planned)

Defined or planned under `include/pyc/`:

- `compiler_api.h`
  - `pyc_compile_model(...)`
  - `pyc_run_model(...)`
  - `pyc_destroy_model(...)`
- `ir.h`
  - IR module/op/shape/dtype and `pyc_ir_verify(...)`
- `pass_manager.h`
  - pass pipeline config, run entrypoint, and report
- `runtime_allocator.h`
  - allocation request/plan/stats APIs
- `kernel_registry.h`
  - register/select/benchmark update-read APIs

## Data Flow

1. Import/create model graph in IR.
2. Verify IR invariants.
3. Run passes in fixed order:
  - canonicalization
  - shape inference
  - layout propagation
  - fusion
  - liveness
  - allocation planning
  - lowering handoff
4. Build runtime execution plan.
5. Select kernels by op key + backend.
6. Execute and emit run telemetry.

## Dynamic Memory Optimization Plan

1. Model tensor liveness intervals.
2. Build overlap/interference reasoning for reuse.
3. Allocate with alignment-safe reuse policy.
4. Track planner metrics:
  - peak bytes
  - total requested bytes
  - reuse count
5. Extend later with rematerialization and stream-aware scheduling.

## Kernel Optimization Plan

### Wave 1: GEMM + fused epilogues

- focus ops: matmul + bias + relu/gelu
- knobs: tile, vector width, unroll, staging, launch geometry

### Wave 2: Reductions + LayerNorm

- focus memory-bound behavior and stability

### Wave 3: Conv2D families

- broaden to vision-oriented workloads

For each wave:

1. Candidate generation
2. Correctness validation
3. Benchmark (p50/p95, variance)
4. Persist best candidate
5. Runtime dispatch by shape signature + backend

## Phase Plan

## Phase 1: Foundations (Implemented)

### Completed

- compiler-next module skeleton
- public interface headers
- CMake target integration (`pyc_compiler_next`, test targets)
- executable smoke and subsystem tests
- docs set for compiler-next architecture and protocol

### Exit Status

- complete

## Phase 2: Real Passes + Memory Planner v1 (Implemented)

### Implemented

1. Real pass behavior in `compiler/passes/pass_manager.c`:
  - canonicalization of unnamed ops
  - shape inference for output/add/activation/layernorm/matmul
  - deterministic fusion (matmul + add/relu/gelu)
  - liveness peak analysis
2. IR deterministic serialization for golden tests in `compiler/ir/ir.c` (`pyc_ir_serialize`).
3. Memory planner diagnostics v1 in `compiler/runtime/runtime_allocator.c`:
  - allocation events
  - overlap pair count
  - largest allocation size
4. Golden and deterministic test coverage:
  - `tests/compiler_next/test_pass_golden.c`
  - `tests/compiler_next/golden/simple_pipeline_after.txt`
  - `tests/compiler_next/test_runtime_allocator.c` (expanded diagnostics checks)

### Exit Status

- complete

### Validation Status

- full compiler-next suite passes (`ctest`: 8/8 passing).

## Phase 3: CPU Performance Path (Implemented)

### Implemented

1. Real CPU execution in `compiler/compiler_api.c`:
  - graph execution for `input`, `matmul`, `add`, `relu`, `output`
  - explicit runtime validation for shapes, ids, tensor sizes, and dtype
2. Deterministic CPU correctness tests:
  - `tests/compiler_next/test_cpu_execution.c`
  - validates matmul numerics and add+relu numerics end-to-end
3. Benchmark rails and reproducibility assets:
  - deterministic benchmark harness and regression checker under `benchmark/`
  - GPU-comparison adapter rails under `benchmark/benchmarks/gpu/` for follow-on backend work

### Exit Status

- complete

### Validation Status

- full compiler-next suite passes (`ctest`: 13/13 passing).

## Phase 4: CUDA + Autotuning (Implemented)

### Objectives

1. Add CUDA backend integration.
2. Add autotune candidate search and selection persistence.
3. Implement reliability rails for compile/runtime behavior under dynamic workloads.

### Deliverables

1. Backend-aware CUDA lowering path.
2. Kernel candidate benchmark loop and stored best choices.
3. Stable fallback when tuned kernels are missing.
4. Reliability rails v1 (R1/R2 from `docs/compiler-next/compile-runtime-reliability-spec.md`):
  - explicit guard checks + deterministic fallback reason codes
  - compile budget modes + cache hit/miss instrumentation

### Implemented

1. CUDA runtime dispatch rail with deterministic fallback/error reasoning:
  - `include/pyc/cuda_backend.h`
  - `compiler/runtime/cuda_backend.c`
  - `tests/compiler_next/test_cuda_backend.c`
  - native CUDA execution path for supported ops (`input`, `matmul`, `add`, `relu`, `output`) with guarded fallback
2. Deterministic contract checks + guard counters in runtime stats:
  - `guard_miss_count`, `fallback_count`
3. Compile-budget and in-memory compile-cache instrumentation:
  - `compile_budget_ms` and `cache_mode` in `pyc_compile_options`
  - `compile_cache_hit` and `compile_budget_exceeded` in `pyc_run_stats`
  - cache/budget validation test: `tests/compiler_next/test_compile_cache.c`
4. Autotune candidate persistence (v1):
  - per-kernel benchmark updates by symbol
  - deterministic DB load/save for kernel timings
  - coverage in `tests/compiler_next/test_autotune_persistence.c`
5. Graph-break visibility (R3 foundation):
  - pass-level `graph_break_count`, `compilability_score`, and summary reason
  - surfaced in `pyc_run_stats`
  - coverage in `tests/compiler_next/test_graph_break_reporting.c`
6. Native CUDA execution tuning for repeated runs:
  - persistent CUDA workspace and cuBLAS handle reuse in `compiler/runtime/cuda_backend.c`
  - deterministic fallback on runtime errors
7. Richer autotune search space + compaction:
  - deterministic candidate enumeration (`pyc_kernel_collect`)
  - candidate-aware timing updates and persisted best-choice compaction
  - coverage in `tests/compiler_next/test_autotune_compaction.c`
8. Expanded graph-break taxonomy and per-op diagnostics:
  - per-op-type break counters (`const`, `gelu`, `reduce_sum`, `layernorm`, `unknown`)
  - first graph-break op id/name surfaced in pass and runtime reports
  - extended coverage in `tests/compiler_next/test_graph_break_reporting.c`

### Exit Status

- complete

### Validation Status

- full compiler-next suite passes (`ctest`: 19/19 passing).

### Exit Criteria

1. Correct CUDA execution on reference kernels.
2. Verified speedups over baseline kernels.
3. Deterministic runtime fallback behavior.
4. `silent_mismatch_count == 0` on targeted dynamic/aliasing stress suite.

## Phase 5: Scale + Promotion (Implemented, Hardening Active)

### Objectives

1. Expand kernel/operator coverage.
2. Promote hardened components into stricter CI guarantees.

### Deliverables

1. Extended operator families.
2. Broader test/perf matrix and regression thresholds.
3. Promotion checklist and compatibility matrix.
4. Reliability rails v2 (R3/R4 from `docs/compiler-next/compile-runtime-reliability-spec.md`):
  - graph-break visibility/reporting and compilability scoring
  - cross-platform deterministic toolchain preflight and diagnostics

### Implemented

1. Expanded CUDA execution path and chained op coverage:
  - deterministic matmul pipeline with optional `add` + `relu` epilogues
  - CUDA graph capture/replay path for repeated stable signatures
  - guarded reuse toggles for stable RHS workloads (`PYC_CUDA_ASSUME_STATIC_RHS`)
2. Reliability rails v2 surfaced through compiler-next stats and pass reports:
  - expanded graph-break taxonomy by op family
  - first break op metadata in diagnostics
  - explicit fallback/error reasoning (no silent fallback ambiguity)
3. Promotion and compatibility rails encoded as code+tests+benchmarks:
  - deterministic fallback on unsupported platforms/toolchains
  - benchmark adapters and manifests aligned to reproducible run stamping
  - promotion remains gated by deterministic tests and published artifacts
4. Extended operator and runtime validation coverage:
  - CUDA backend tests include matmul/add/relu path correctness
  - graph-break and autotune persistence tests enforce failure-surface visibility

### Promotion Checklist (Phase 5)

1. Correctness:
  - all compiler-next tests pass with `PYC_BUILD_COMPILER_NEXT_TESTS=ON`
  - no silent mismatch in fallback paths
2. Determinism:
  - run-to-run outputs stable for golden and backend tests
  - fallback reasons and graph-break diagnostics stable and explicit
3. Compatibility:
  - Linux/macOS/Windows build contract remains green for stable targets
  - compiler-next gracefully degrades where CUDA runtime/toolchain is unavailable
4. Performance evidence:
  - benchmark runs emit stamped JSON/SVG/metadata
  - regression checks run against versioned baselines before promotion

### Compatibility Matrix (Current)


| Environment           | Stable targets (`pyc`, `pyc_core`, `pyc_foundation`) | Compiler-next CPU path | Compiler-next CUDA path                               |
| --------------------- | ---------------------------------------------------- | ---------------------- | ----------------------------------------------------- |
| Linux (x86_64)        | required                                             | required               | required when CUDA toolkit + GPU are present          |
| macOS (Apple Silicon) | required                                             | required               | deterministic fallback/proxy path only                |
| Windows (MSVC)        | required                                             | required               | deterministic fallback unless CUDA runtime configured |


### Validation Status

- Local compiler-next suite: `ctest --test-dir build --output-on-failure` -> **19/19 passing**.
- CUDA-path tests pass on non-CUDA hosts through deterministic fallback contracts.

### Exit Criteria

1. Cross-platform reliability at expanded scope.
2. KPI targets met.
3. Promotion policy enforced in CI.
4. High-resolution pain point closed:
   - no silent wrong-output incidents in reliability suite
   - 100% mismatch scenarios converted to explicit fallback/failure with reason codes

## Phase 6: Productionization and Commercial Readiness (In Progress)

### Objectives

1. Turn compiler-next into a deployable, operator-safe runtime surface for real workflows.
2. Convert reliability expectations into enforceable production gates.
3. Define commercial rollout rails: compatibility, observability, and release criteria.

### Deliverables

1. Production contract test suite (required in CI):
   - `tests/compiler_next/test_production_status_errors.c`
   - `tests/compiler_next/test_production_decision_log.c`
   - `tests/compiler_next/test_production_cuda_contracts.c`
   - `tests/compiler_next/test_production_runtime_rollback.c`
2. Release-gate policy:
   - deterministic status/error contracts
   - reason-coded fallback/error behavior
   - runtime rollback behavior under failure pressure
   - decision-log observability contract
3. Integration readiness rails:
   - importer path requirements for real model ingestion (Torch FX/ONNX bridge target)
   - packaging and binary-distribution matrix by OS/toolchain
   - benchmark + correctness differential gating before promotion
4. Commercial operations guardrails:
   - baseline SLOs (p50/p95 latency + fallback-rate thresholds)
   - incident triage signals from `pyc_run_stats`
   - versioned compatibility and deprecation policy
5. Production readiness reference:
   - `docs/compiler-next/production-readiness.md`

### Test Matrix (Phase 6 Initial)

1. API and status safety:
   - invalid input paths return explicit `PYC_STATUS_*` codes
2. Observability:
   - decision log includes mode/fallback/contract/graph-break signatures
3. CUDA runtime behavior:
   - forced fallback and forced error paths are deterministic and reason-coded
4. Runtime control:
   - runtime-error breaches trigger rollback according to rails policy

### Exit Criteria

1. Production contract suite passes across Linux/macOS/Windows.
2. No silent mismatch/fallback ambiguity in required deployment modes.
3. Decision-log and `pyc_run_stats` fields are stable enough for external monitoring ingestion.
4. Differential correctness suite (PyC vs reference runtime) meets tolerance policy on target workloads.
5. CI enforces production gates before release artifacts are published.

## Innovation Backlog Integration

The innovation shortlist is tracked in:

- `docs/compiler-next/innovation-backlog.md`
- `docs/compiler-next/rd-landscape.md` (evidence and ecosystem sentiment inputs for prioritization)

Phase mapping:

1. Phase 3:
  - Adaptive planner modes
  - Dynamic reuse + rematerialization policy
2. Phase 4:
  - Shape-clustered multi-plan execution
  - Kernel + allocator co-selection
3. Phase 5:
   - Online feedback planner
   - Deterministic what-if simulator
   - User-facing optimization contracts
   - Policy plugin interfaces
   - Compile-runtime reliability rails (`docs/compiler-next/compile-runtime-reliability-spec.md`)
4. Phase 6:
   - Production contract suite and commercial release gates
   - Integration/importer hardening and compatibility policy
   - SLO-backed promotion and rollout criteria

Reliability-first mapping (new):

1. Phase 4:
  - R1 Guarded correctness rails
  - R2 Compile budget + cache
  - R3 Graph-break visibility/reporting and compilability scoring
2. Phase 5:
  - R4 Cross-platform deterministic preflight

Primary reliability target:

- `docs/compiler-next/compile-runtime-reliability-spec.md`  
section: **High-Resolution Target Pain Point**.

Promotion rule:

- Innovation items are promoted only after KPI pass + deterministic test pass.

## Test Strategy

1. Subsystem tests (IR, pass manager, allocator, kernel registry).
2. API integration tests through compile/run lifecycle.
3. End-to-end performance suites with fixed protocol.
4. Regression gates for correctness and performance.

## Rollout and Flags

1. Keep stable core targets as primary CI contract.
2. Keep compiler-next behind explicit build flags.
3. Promote by checklist only, never by ad hoc merge.

## Risk Controls

1. Architecture sprawl: enforce module boundaries and API ownership.
2. Noisy performance claims: fixed protocol, repeat counts, and variance reporting.
3. Backend drift: shared IR/passes, backend-specific codegen only.
4. Environment mismatch for CUDA: require explicit toolchain preflight and remote GPU workflow where needed.

## Assumptions

1. Development remains C/C++ with CMake.
2. CUDA validation often runs on rented/remote Linux GPU hosts.
3. MLIR bridge is deferred until custom IR path is stable and benchmarked.
4. End-to-end latency remains the primary KPI.
