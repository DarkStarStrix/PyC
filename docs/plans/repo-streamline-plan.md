# Repo Streamline And Feature Delivery Plan

## Purpose

This plan defines how to:

1. reduce root-directory clutter,
2. enforce clearer ownership boundaries by folder,
3. move CUDA and documentation assets into dedicated nested homes,
4. deliver the highest-value missing compiler/runtime features without creating more structural debt.

The repo already has strong subsystem boundaries in the compiler-next stack, but the top-level layout has drifted and now mixes product code, website assets, prototypes, generated artifacts, and planning material too loosely.

## Problems To Fix

### Root-level clutter

The repository root currently mixes:

- build/config files,
- community files,
- product code,
- benchmark/report assets,
- website assets,
- prototype CUDA files,
- generated/demo artifacts.

That makes ownership, discoverability, and migration safety worse than it should be.

### File-type sprawl

Two patterns need to stop:

1. standalone `.md` files scattered without topic folders,
2. standalone `.cu` files sitting directly in a shared folder without per-kernel ownership boundaries.

### Mixed intent inside shared folders

The kernel area previously mixed ad hoc prototypes and lab tooling under one shared bucket. That split has now started, but the remaining source-tree normalization work still needs the same discipline elsewhere.

## Non-Negotiable Layout Rules

1. Root stays minimal.
2. Every major concern gets its own folder.
3. No new standalone `.cu` files at the top of a shared directory.
4. No new planning/spec `.md` files unless they live under an explicit docs subtree.
5. Benchmarks, prototypes, runtime code, docs, and website assets stay physically separated.
6. Any reorg must preserve deterministic CI and build/test commands until replacement paths are fully wired.

## Allowed Root-Level Exceptions

These should remain at the root unless there is a hard reason otherwise:

- `README.md`
- `LICENSE`
- `CMakeLists.txt`
- `pyproject.toml`
- `.github/`
- GitHub/community health files that GitHub expects at root or under `.github/`

That means files such as `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `SECURITY.md`, and `SUPPORT.md` can stay where GitHub expects them. Everything else should be pushed downward into dedicated folders over time.

## Target Top-Level Shape

The desired steady state is:

```text
/  
  .github/
  build/
  include/
  src/
    core/
    compiler/
    runtime/
  
    apps/
    artifacts/
    benchmark/
    docs/
      architecture/
      compiler-next/
      contracts/
      plans/
      reference/
      reports/
    examples/
    infra/
    kernels/
      lab/
      prototypes/
    python/
    scripts/
    tests/
    tools/
    web/
      site/
```

## Important Mapping Notes

This does not need to happen in one step.

Recommended mapping:

- `Core/` -> `src/core/`
- `compiler/` -> `src/compiler/`
- `runtime/` -> `src/runtime/`
- `kernels/` -> `kernels/`
- `Examples/` -> `examples/`
- `hello/`, `Hello.py`, `hello.spec` -> `artifacts/generated/hello_pyinstaller/`
- all static website assets -> `web/site/`

## Examples And Utility Folder Policy

Examples and local developer tools need separate ownership from vendored code.

### Rules

1. `examples/` is for runnable examples, grouped by language or feature area.
2. `tools/` is for first-party utilities, local CLIs, and helper binaries.
3. `third_party/` is reserved for actual vendored external code only.
4. First-party code must not be parked under `third_party/`.

### Immediate migration target

- `Examples/matrix_mult.py` -> `examples/python/matrix_mult.py`
- `Examples/simple_graph.py` -> `examples/python/simple_graph.py`
- `third_party/main.cpp` -> `tools/legacy_cli/main.cpp`

## CUDA And Kernel Folder Policy

This area needs immediate cleanup discipline.

### New rule

No new `.cu` file should live directly under a shared bucket such as `kernels/` or `src/compiler/cutlass_kernels/` unless that folder exists specifically for a single kernel family.

### Target kernel structure

```text
kernels/
  lab/
    kernel_lab.py
    manifests/
    results/
    build/
  prototypes/
    ada/
      gemm/
        kernel.cu
        manifest.json
        notes.md
    hopper/
    experimental/
```

### Current kernel layout

- `kernels/prototypes/ada/gemm/kernel.cu`
- `kernels/prototypes/baseline/matmul/kernel.cu`
- `kernels/prototypes/experimental/tokenizer_matmul/kernel.cu`
- `kernels/lab/manifests/kernels.json`
- `kernels/lab/results/`
- `kernels/lab/build/`

## Documentation Folder Policy

All new Markdown must live under topic folders.

### Target docs structure

```text
docs/
  architecture/
  compiler-next/
  contracts/
  milestones/
  plans/
  reference/
  reports/
  roadmap/
```

### Migration guidance

- `docs/Architecture.md` -> `docs/architecture/system-architecture.md`
- `docs/Doc.md` -> `docs/reference/project-overview.md`
- `docs/reports/results-legacy.md` -> delete after redirecting references
- `docs/results.md` -> `docs/reports/results.md`
- `docs/perf-report.md` -> `docs/reports/perf-report.md`
- `docs/performance-results.md` -> `docs/reports/performance-results.md`
- roadmap/planning docs stay grouped under `docs/compiler-next/` or `docs/plans/`

## Website And Static Asset Policy

The root should not contain user-facing static site assets.

### Immediate migration target

- `index.html` -> `web/site/index.html`
- `results-insights.html` -> `web/site/results-insights.html`
- `inference-site/` -> `web/site/inference/`
- website JS/CSS assets -> `web/site/assets/`
- published result payloads -> `web/site/results/`

### Rules

1. Website source files live under `web/site/`.
2. Published/generated site-facing result artifacts also live under `web/site/`.
3. Root should not be used as a static site asset bucket.

## Generated And Demo Artifact Policy

The following do not belong in the long-term clean root:

- `build/`
- `hello/`
- local generated bundles and transient result snapshots

If they must remain versioned, they should live under a dedicated subtree such as:

```text
artifacts/
  demos/
  generated/
```

If they do not need to be versioned, they should be ignored and kept out of the repo.

Current action already taken:

- PyInstaller-style `hello` assets were moved under `artifacts/generated/hello_pyinstaller/`.

## Feature Delivery Order

Structural cleanup should support feature work, not block it.

### Priority 1: Dynamic execution intelligence

1. Shape-clustered multi-plan execution
2. Speculative or phantom graph planning with guard-validated reuse
3. Deterministic what-if simulator

Reason:
This is the highest leverage path for dynamic workloads and aligns with current compiler-next gaps.

### Priority 2: Memory/runtime quality

1. Real rematerialization policy
2. allocator pressure-aware kernel co-selection
3. online feedback planner

Reason:
The allocator already exposes pressure signals, but the behavior is still heuristic and shallow.

### Priority 3: Operator and backend expansion

1. `reduce_sum`
2. `layernorm`
3. `gelu`
4. broader conv coverage
5. tighter production CUDA kernel promotion path

### Priority 4: Productization

1. user-facing optimization contracts
2. policy plugin interfaces
3. ONNX/Torch FX ingestion
4. compatibility policy and release bundle hardening

### Priority 5: Distributed roadmap

1. harden collective backend loading
2. establish true distributed execution seams
3. add staged data-parallel then ZeRO/tensor-parallel support

This should stay behind the core compiler/runtime cleanup and dynamic-plan work unless there is an external deadline.

## Execution Phases

## Phase A: Stop Further Layout Drift

1. Add and adopt this structure policy.
2. Reject new root-level stray assets.
3. Reject new free-floating `.cu` and planning `.md` files.
4. Update contributor docs with layout rules.

## Phase B: Root Cleanup

1. Move root static web assets into `web/download-site/`.
2. Move report-style docs into `docs/reports/`.
3. Move one-off planning docs into `docs/plans/`.
4. Audit generated/demo artifacts and either relocate or ignore them.

## Phase C: Kernel Area Rebuild

1. Keep `kernels/` as the single kernel root.
2. Preserve the split between lab tooling and prototypes.
3. Put each new kernel family in its own nested folder.
4. Keep manifests, docs, and benchmark commands aligned with that structure.

## Phase D: Source Tree Normalization

1. Keep `src/` as the long-term home for implementation code.
2. Keep `src/core/`, `src/compiler/`, and `src/runtime/` stable while remaining references and helper scripts are normalized.
3. Keep `include/pyc/` stable during the migration.

## Phase E: Feature Buildout

1. shape-clustered multi-plan execution
2. speculative graph plan cache
3. rematerialization
4. kernel + allocator co-selection
5. what-if simulator

## Phase F: Product Surface Hardening

1. importer paths
2. optimization contracts
3. plugin interfaces
4. production/readiness cleanup

## Guardrails During Reorg

1. Keep existing build commands working until replacement docs and CMake paths land.
2. Move one subsystem at a time.
3. Update tests and docs in the same change as each move.
4. Do not combine mass moves with major feature logic in the same PR.
5. Every migration step must keep deterministic CI intact.

## First Concrete Refactor Batch

This should be the first actual cleanup batch:

1. create `web/site/` and move all website assets there,
2. create `docs/plans/` and `docs/reports/` if missing,
3. create `kernels/lab/` and `kernels/prototypes/`,
4. move current Ada and baseline kernel prototypes into dedicated nested folders,
5. update README, docs index, and kernel lab paths,
6. remove stale or duplicate report pointers.

## Success Criteria

The cleanup effort is successful when:

1. the root contains only true entrypoint files and top-level subsystems,
2. no new stray `.md` or `.cu` files are introduced,
3. every kernel family has its own folder,
4. docs are grouped by function rather than historical accident,
5. feature work becomes easier because ownership boundaries are obvious,
6. CI and benchmark protocols remain deterministic throughout the transition.
