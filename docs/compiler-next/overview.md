# Compiler-Next Overview

`compiler-next` is the new performance-first stack for PyC. It is currently experimental and designed to coexist with the stable CI core targets.

## Goals

- Deliver measurable end-to-end model latency wins.
- Add dynamic memory planning for lower peak memory.
- Build an extensible kernel optimization/autotuning path.

## Current Components

- `compiler/ir/ir.c`: IR model + verifier.
- `compiler/passes/pass_manager.c`: pipeline orchestration.
- `compiler/runtime/runtime_allocator.c`: dynamic allocation planning with reuse.
- `compiler/runtime/kernel_registry.c`: backend-aware kernel registration/selection.
- `compiler/compiler_api.c`: public compile/run API.
- `include/pyc/optimizer_policy.h`: policy contract for memory-first, balanced, and utilization-first objective modes.
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
- `docs/compiler-next/gpu-testing-playbook.md` (how to run standardized CUDA benchmarks on rented Linux GPU machines)
- `docs/compiler-next/compile-runtime-reliability-spec.md` (reliability-first plan to solve `torch.compile`-class pain points: correctness, compile latency, graph breaks, and deterministic fallback)
- `tests/compiler_next/test_roadmap_phase1.md`
- `docs/compiler-next/kernel-lab.md` (mini CLI lab for kernel prototyping/testing/benchmarking)
