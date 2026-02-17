# PyC Architecture

## Architectural Goals

The architecture is designed around two competing needs:

- preserve deterministic, cross-platform integration behavior,
- allow fast iteration in experimental compiler modules.

The result is a layered model with clear stability boundaries.

## Layered View

### Layer 1: Build and Integration Contract

- Build system: CMake.
- Canonical CI workflow: `.github/workflows/cmake-multi-platform.yml`.
- Stable targets: `pyc`, `pyc_core`, `pyc_foundation`.

This layer guarantees that downstream consumers can reliably build and link core artifacts.

### Layer 2: Stable Core Library Surface

Stable source modules currently include:

- `adapter.c`
- `semantic.c`
- `stack.c`
- `symbol_table.c`

These modules are selected for portability and deterministic behavior.

### Layer 3: Experimental Compiler-Next Surface

Experimental modules include compiler-next IR/passes/runtime and the AI bridge layer. They are available for development but intentionally isolated from stable-core guarantees until hardened.

## Target Graph

```text
pyc_core_obj (OBJECT)
  ├─> pyc_core (STATIC)
  └─> pyc_foundation (STATIC, compatibility)

pyc (EXE, stable smoke driver)
  └─links─> pyc_core

pyc_compiler_next (STATIC, experimental)
  └─> pyc_ai_bridge (STATIC, policy bridge)

pyc_core_microbench (EXE, optional)
  └─links─> pyc_core
```

## Runtime Boundaries

### `pyc` Smoke Driver

`pyc` is intentionally minimal. Its job is to prove target resolution, linker correctness, and runtime invocation path across platforms.

### Adapter Boundary

`adapter.c` acts as an OS-facing boundary. Platform-specific behavior (for example command spawning differences) is isolated there, reducing portability risk in the rest of the stable core.

### Utility Data Structures

`stack.c` and `symbol_table.c` provide baseline utility behavior that is easy to benchmark and validate.

## Cross-Platform Strategy

- Avoid compiler-specific extensions in stable headers/source unless wrapped.
- Isolate POSIX-specific behavior behind platform guards.
- Keep stable core intentionally small until each additional module is validated under MSVC + Clang/GCC toolchains.

## Architectural Risks

- Experimental module drift can diverge from stable integration constraints.
- Tight coupling between unstable headers may make promotion harder.
- Benchmark coverage is currently core-focused, not full pipeline-focused.

## Promotion Criteria for Experimental Modules

A module should only move into stable core when it meets all criteria:

1. Builds cleanly on Linux, macOS, and Windows/MSVC.
2. Has deterministic behavior under repeated runs.
3. Has at least one focused correctness test.
4. Does not introduce nondeterministic CI dependencies.
5. Has measurable benchmark impact tracked over time.
