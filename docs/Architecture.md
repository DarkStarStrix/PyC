# PyC Architecture

## 1) System Intent

PyC is split into a deterministic stable core and an experimental compiler-next stack. The stable core guarantees reproducible CMake targets for CI and downstream linking. Compiler-next is where graph IR, pass execution, runtime policy control, and kernel dispatch evolve quickly.

## 2) Repository-Level Component Map

- `Core/C_Files` + `Core/Header_Files`: stable C primitives and compatibility surface.
- `compiler/`: compiler-next implementation (IR, pass manager, runtime execution path, kernel registry, CUDA backend bridge).
- `include/pyc/`: public headers for compiler-next/runtime APIs.
- `tests/compiler_next/`: deterministic tests for correctness, policy behavior, cache/reliability contracts, and source-coverage enforcement.
- `benchmark/`: benchmark harness, adapters, regression checks, and rendered artifacts.
- `website/results/`: published benchmark charts + metadata for static site consumption.

## 3) Build Graph and Binary Contracts

Canonical targets:

- `pyc_core_obj` (OBJECT): shared object source set for stable core.
- `pyc_core` (STATIC): canonical stable static library.
- `pyc_foundation` (STATIC): compatibility alias over same object set.
- `pyc` (EXE): deterministic smoke entrypoint for CI validation.

Optional compiler-next targets (guarded by CMake options):

- `pyc_compiler_next` (STATIC)
- `pyc_compiler_next_smoke` / `pyc_compiler_next_test_*`

Design rule: stable artifacts must remain buildable across Linux/macOS/Windows even when experimental modules change.

## 4) Compiler-Next Execution Pipeline

### 4.1 Frontdoor API

`compiler_api` accepts workload descriptors and policy mode requests. Inputs are normalized into a compact internal representation with deterministic defaults.

### 4.2 IR and Pass Layer

The pass manager runs ordered, explicit passes. Current behavior focuses on deterministic transformations and predictable naming/serialization to support golden tests and regression checks.

### 4.3 Runtime Controller

Runtime orchestration resolves:

- policy mode (`balanced`, `memory_first`, `utilization_first`),
- kernel/backend selection,
- fallback path when a backend is unavailable,
- reliability counters (graph breaks, fallback count, guard misses).

This gives predictable behavior under pressure while preserving observability.

### 4.4 Kernel Registry

Kernel capabilities are resolved through a registry surface instead of direct hardcoding in call sites. This keeps backend routing auditable and testable.

### 4.5 CUDA Backend Boundary

`cuda_backend` is compiled in a portability-safe form:

- if CUDA toolkit is present, runtime can attempt native CUDA execution,
- if CUDA toolkit is absent, deterministic CPU/proxy fallback remains valid,
- availability can be controlled through explicit environment toggles for testability.

This prevents nondeterministic crashes on non-CUDA hosts while still enabling native execution on provisioned GPU machines.

## 5) Determinism and Reliability Contracts

Core contracts encoded in code/tests/CI:

- deterministic build entrypoints and explicit target lists,
- non-flaky smoke validation (`pyc`),
- feature tests must be added for behavior changes,
- no weakening tests to pass broken implementations,
- benchmark runs produce machine-readable metadata and reproducible artifacts.

Runtime reliability counters (fallbacks, guard misses, cache/autotune states) are treated as first-class signals for promotion decisions.

## 6) Benchmarking and Published Evidence

Benchmarking is adapter-based and produces JSON/MD/SVG. A publish step centralizes website-facing assets:

```bash
python3 scripts/publish_site_results.py
```

Outputs:

- `website/results/artifacts/**` (all SVG + `*.metadata.json`),
- `website/results/manifest.json` (full artifact index),
- `website/results/latest-summary.json` (latest CPU/GPU adapter summary).

The site reads these files directly, so published performance is traceable to versioned artifacts.

## 7) Known Constraints

- Some competitor adapters may run in proxy mode depending on environment/toolchain availability.
- TVM/XLA/TensorRT/Glow parity depends on installed runtimes and adapter wiring.
- Native PyC CUDA performance requires full kernel path maturation; fallback/proxy mode is intentionally conservative.

## 8) Promotion Path

A compiler-next component is promoted only when it satisfies all of:

1. Cross-platform build stability (Linux/macOS/Windows).
2. Deterministic behavior under repeated runs.
3. Dedicated correctness coverage in `tests/`.
4. Clear observability and failure handling.
5. Benchmark evidence with published metadata.
