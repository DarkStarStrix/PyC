# Compiler-Next Overview

`compiler-next` is the new performance-first stack for PyC. It is currently experimental and designed to coexist with the stable CI core targets.
The live flow now includes deterministic guards, speculative plans, a shadow-first phantom graph, budget-sensitive rematerialization telemetry, an in-memory compile cache, an online feedback controller with shadow recommendations, kernel + allocator co-selection, and the Ada FP32 CUDA fast path. The current buildout focus is to validate these features together on the Ada hot path before returning to deeper kernel promotion work.

## Goals

- Deliver measurable end-to-end model latency wins.
- Add dynamic memory planning for lower peak memory.
- Build an extensible kernel optimization/autotuning path.

## Current Components

- `src/compiler/ir/ir.c`: IR model + verifier.
- `src/compiler/passes/pass_manager.c`: pipeline orchestration.
- `src/compiler/runtime/kernel_registry.c`: backend-aware kernel registration/selection.
- `src/compiler/runtime/runtime_allocator.c`: dynamic allocation planning with reuse and rematerialization policy.
- `src/compiler/runtime/runtime_control.c`: bounded controller and shadow recommendation rails.
- `src/compiler/compiler_api.c`: public compile/run API with speculative plans, phantom-graph tracking, compile-cache reuse, deterministic guards, controller rails, and runtime kernel reselection.
- `include/pyc/optimizer_policy.h`: policy contract for memory-first, balanced, and utilization-first objective modes.
- `kernels/lab/kernel_lab.py`: task builder and Ada kernel-lab workflow.
- `benchmark/tools/analyze_ada_gemm_results.py`: local Ada result bundle synthesis.
- `tests/compiler_next/compiler_next_smoke.c`: integration smoke test.

- `tests/compiler_next/test_pass_golden.c`: golden pass-output determinism test.
- `tests/compiler_next/test_roadmap_phase2.md`: Phase 2 test matrix.

## Build Flags

- `PYC_BUILD_COMPILER_NEXT=ON|OFF`
- `PYC_BUILD_COMPILER_NEXT_TESTS=ON|OFF`

## Primary Target

- `pyc_compiler_next` (static library)

## Smoke Target

- `pyc_compiler_next_smoke`

## Roadmap and Test Policy

- `docs/compiler-next/roadmap-phases.md`
- `docs/compiler-next/innovation-backlog.md`
- `docs/compiler-next/rd-landscape.md` (classical vs AI compiler R&D, ecosystem sentiment, and priority experiments)
- `docs/compiler-next/runtime-integration-spec.md` (how to embed PyC under real job runtimes with dynamic policy control)
- `docs/compiler-next/cuda-gemm-fast-path.md` (mechanical description of the current FP32 CUDA GEMM fast path, including cuBLASLt preference and runtime controls)
- `docs/compiler-next/gpu-testing-playbook.md` (how to run standardized CUDA benchmarks on rented Linux GPU machines)
- `docs/compiler-next/compile-runtime-reliability-spec.md` (reliability-first plan to solve `torch.compile`-class pain points: correctness, compile latency, graph breaks, and deterministic fallback)
- `tests/compiler_next/test_roadmap_phase1.md`
- `docs/compiler-next/kernel-lab.md` (mini CLI lab for kernel prototyping/testing/benchmarking)
