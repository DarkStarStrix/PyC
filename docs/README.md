# PyC Docs

This folder is the source of truth for project technical documentation. GitHub community health files (for example `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `SECURITY.md`, `SUPPORT.md`) live at the repository root or under `.github/`.

## Start Here

If you are new to the project, read these in order:

1. `docs/reports/project-status.md`
2. `docs/reference/project-overview.md`
3. `docs/architecture/system-architecture.md`
4. `docs/reference/build-and-ci.md`
5. `docs/reference/benchmarking.md`
6. `docs/reports/results.md`

## Documentation Map

- `docs/compiler-next/`
  - Next-generation compiler stack docs (IR, passes, runtime memory planner, kernels, benchmark protocol).
  - Includes phased roadmap in `docs/compiler-next/roadmap-phases.md`.
  - Includes innovation shortlist in `docs/compiler-next/innovation-backlog.md`.
  - Includes R&D landscape and source-backed direction in `docs/compiler-next/rd-landscape.md`.
  - Includes practical runtime usage spec in `docs/compiler-next/runtime-integration-spec.md`.
  - Includes Phase 1 test matrix in `tests/compiler_next/test_roadmap_phase1.md`.
  - Includes kernel mini-lab CLI guide in `docs/compiler-next/kernel-lab.md`.
- `kernels/lab/` and `kernels/prototypes/ada/`
  - Kernel staging area for prototyping, benchmark-surface preparation, and Ada-specific experiments before promotion.
- `docs/plans/`
  - Repo-level cleanup and migration plans, including structure and feature-delivery sequencing.
  - Includes the active compiler-next execution order in `docs/plans/compiler-next-feature-buildout.md`.
- `docs/reference/project-overview.md`
  - Product and technical overview, scope, terminology, and current constraints.
- `docs/reference/build-and-ci.md`
  - Canonical build commands, CI behavior, and troubleshooting.
- `docs/reference/benchmarking.md`
  - Deterministic benchmark harness usage and methodology.
- `infra/README.md`
  - GPU-host bootstrap flow, including the reusable Docker image path for faster VM bring-up.
- `docs/reference/repository-rules.md`
  - Non-negotiable repository rules plus layout guardrails and enforcement map.
- `docs/architecture/system-architecture.md`
  - Component-level architecture, data flow, and module boundaries.
- `docs/reports/results.md`
  - Canonical benchmark outcomes and published artifact references.
- `docs/reports/perf-report.md`
  - Short current-state performance assessment.
- `docs/reports/project-status.md`
  - What is stable now vs what is experimental.
- `docs/reports/performance-results.md`
  - Stable-core local benchmark snapshot.
- `docs/roadmap/`
  - Longer-horizon roadmap and distributed-system planning docs.
- `examples/`
  - Standalone example programs and exploratory snippets, grouped by language or topic.
- `tools/`
  - Internal developer utilities and legacy first-party entrypoints that do not belong under `third_party/`.
- `web/site/`
  - Static user-facing site bundle, including the main page, results page, inference portal, and published site-facing results data.
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

Publish website benchmark artifacts:

```bash
python3 scripts/publish_site_results.py
```

Build compiler-next smoke target:

```bash
cmake -S . -B build -D PYC_BUILD_COMPILER_NEXT=ON -D PYC_BUILD_COMPILER_NEXT_TESTS=ON
cmake --build build --parallel --target pyc_compiler_next pyc_compiler_next_smoke
./build/pyc_compiler_next_smoke
```

Bootstrap a GPU VM with the reusable image:

```bash
bash infra/build_bootstrap_image.sh
INSTALL_SYSTEM_DEPS=0 bash infra/run_bootstrap_image.sh
```
