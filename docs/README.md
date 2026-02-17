# PyC Docs

This folder is the source of truth for project technical documentation. GitHub community health files (for example `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `SECURITY.md`, `SUPPORT.md`) live at the repository root or under `.github/`.

## Start Here

If you are new to the project, read these in order:

1. `docs/project-status.md`
2. `docs/Doc.md`
3. `docs/Architecture.md`
4. `docs/build-and-ci.md`
5. `docs/benchmarking.md`

## Documentation Map

- `docs/compiler-next/`
  - Next-generation compiler stack docs (IR, passes, runtime memory planner, kernels, benchmark protocol).
  - Includes phased roadmap in `docs/compiler-next/roadmap-phases.md`.
  - Includes innovation shortlist in `docs/compiler-next/innovation-backlog.md`.
  - Includes R&D landscape and source-backed direction in `docs/compiler-next/rd-landscape.md`.
  - Includes practical runtime usage spec in `docs/compiler-next/runtime-integration-spec.md`.
  - Includes Phase 1 test matrix in `tests/compiler_next/test_roadmap_phase1.md`.
  - Includes kernel mini-lab CLI guide in `docs/compiler-next/kernel-lab.md`.
- `docs/Doc.md`
  - Product and technical overview, scope, terminology, and current constraints.
- `docs/REPO_RULES.md`
  - Hard repository rules and the exact enforcement mechanisms.
- `docs/Architecture.md`
  - Component-level architecture, data flow, and module boundaries.
- `docs/project-status.md`
  - What is stable now vs what is experimental.
- `docs/build-and-ci.md`
  - Canonical build commands, CI behavior, and troubleshooting.
- `docs/benchmarking.md`
  - Deterministic benchmark harness usage and methodology.
- `docs/performance-results.md`
  - Latest measured benchmark outputs.
- `docs/downloads/`
  - Static user-facing download page that links latest release binaries by OS.
- `docs/Result.md`
  - Current outcomes, progress summary, and near-term roadmap.
- `docs/contracts/`
  - Interface contracts for cross-module behavior.
- `docs/milestones/`
  - Milestone planning and acceptance criteria.
- `CODE_OF_CONDUCT.md`
  - Community behavior and enforcement policy.
- `CONTRIBUTING.md`
  - Contribution workflow, validation checklist, and PR expectations.
- `SECURITY.md`
  - Security reporting policy and supported versions.
- `SUPPORT.md`
  - Support channels and issue-reporting expectations.

## Quick Commands

Build stable targets:

```bash
cmake -S . -B build
cmake --build build --parallel --target pyc pyc_core pyc_foundation
./build/pyc
```

Run benchmark harness:

```bash
python3 benchmark/harness.py --repeats 7 --micro-rounds 4000
```

Build compiler-next smoke target:

```bash
cmake -S . -B build -D PYC_BUILD_COMPILER_NEXT=ON -D PYC_BUILD_COMPILER_NEXT_TESTS=ON
cmake --build build --parallel --target pyc_compiler_next pyc_compiler_next_smoke
./build/pyc_compiler_next_smoke
```
