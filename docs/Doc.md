# PyC: Product and Technical Overview

## Purpose

PyC is an experimental compiler-oriented project that currently prioritizes:

- deterministic cross-platform build behavior,
- stable library/linking contracts,
- a minimal executable for CI smoke validation,
- and a repeatable benchmarking workflow.

The project still contains broader compiler and AI-oriented modules, but the stable CI contract intentionally covers a smaller core surface area.

## Current Scope

### Stable Scope

The stable targets are:

- `pyc_core_obj`: object library for stable source objects.
- `pyc_core`: canonical static library.
- `pyc_foundation`: compatibility static library from the same objects.
- `pyc`: minimal deterministic executable used in CI smoke tests.

Current stable source set:

- `Core/C_Files/adapter.c`
- `Core/C_Files/semantic.c`
- `Core/C_Files/stack.c`
- `Core/C_Files/symbol_table.c`

### Experimental Scope

The broader compiler-next implementation is considered experimental and gated by:

- `PYC_BUILD_COMPILER_NEXT=ON`
- `PYC_BUILD_COMPILER_NEXT_TESTS=ON`

This includes IR/pass/runtime modules that are still under phased promotion into stricter CI guarantees.

## Why the Split Exists

A previous CI path relied on legacy autotools-style `./configure` behavior that does not exist in this project’s canonical build system. The stable core split exists to guarantee reproducible CMake-based behavior across Linux, macOS, and Windows while experimental compiler work continues.

## How the Stable Flow Works

1. Configure with CMake.
2. Build explicit stable targets (`pyc`, `pyc_core`, `pyc_foundation`).
3. Run `pyc` for deterministic smoke verification.
4. Optionally run benchmark harness for drift detection and trend tracking.

## Terminology

- Stable target:
  - A target guaranteed to build in canonical CI.
- Experimental target:
  - A target that may change quickly and is not required for stable CI.
- Smoke test:
  - Minimal executable run proving core target wiring and runtime viability.
- Canonical workflow:
  - The single supported CI definition under `.github/workflows/cmake-multi-platform.yml`.

## Practical Use Cases Today

- Validate cross-platform toolchain integration quickly.
- Link against `pyc_core` from downstream experiments.
- Monitor build/runtime regressions using the benchmark harness.
- Maintain an incremental path toward a larger production compiler pipeline.

## Non-Goals (Current Phase)

At current maturity, the project does not claim:

- full Python language compatibility,
- complete production-grade optimizer/backend behavior,
- or stable guarantees for all experimental modules.

## Next Evolution

The expected progression is:

1. Grow `pyc_core` source coverage from stable, portable modules.
2. Add targeted correctness tests for each newly promoted subsystem.
3. Extend benchmark workloads beyond microbench-level core operations.
4. Promote selected experimental components into stable CI guarantees.
