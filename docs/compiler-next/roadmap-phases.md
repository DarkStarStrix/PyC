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

## Phase 4: CUDA + Autotuning (In Progress)

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

### Implemented So Far

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

### Remaining

1. Native CUDA path performance tuning beyond correctness path.
2. Richer autotune search space and stronger artifact compaction.
3. Expanded graph-break taxonomy and per-op diagnostics output.

### Exit Criteria

1. Correct CUDA execution on reference kernels.
2. Verified speedups over baseline kernels.
3. Deterministic runtime fallback behavior.
4. `silent_mismatch_count == 0` on targeted dynamic/aliasing stress suite.

## Phase 5: Scale + Promotion

### Objectives

1. Expand kernel/operator coverage.
2. Promote hardened components into stricter CI guarantees.

### Deliverables

1. Extended operator families.
2. broader test/perf matrix and regression thresholds.
3. Promotion checklist and compatibility matrix.
4. Reliability rails v2 (R3/R4 from `docs/compiler-next/compile-runtime-reliability-spec.md`):
   - graph-break visibility/reporting and compilability scoring
   - cross-platform deterministic toolchain preflight and diagnostics

### Exit Criteria

1. Cross-platform reliability at expanded scope.
2. KPI targets met.
3. Promotion policy enforced in CI.
4. High-resolution pain point closed:
   - no silent wrong-output incidents in reliability suite
   - 100% mismatch scenarios converted to explicit fallback/failure with reason codes

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

Reliability-first mapping (new):

1. Phase 4:
   - R1 Guarded correctness rails
   - R2 Compile budget + cache
2. Phase 5:
   - R3 Graph-break visibility
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
