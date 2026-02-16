# PyC Docs

This folder is the source of truth for project documentation. The repository root intentionally keeps only `README.md`; all other markdown lives under `docs/`.

## Start Here

If you are new to the project, read these in order:

1. `docs/project-status.md`
2. `docs/Doc.md`
3. `docs/Architecture.md`
4. `docs/build-and-ci.md`
5. `docs/benchmarking.md`

## Documentation Map

- `docs/Doc.md`
  - Product and technical overview, scope, terminology, and current constraints.
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
- `docs/Result.md`
  - Current outcomes, progress summary, and near-term roadmap.
- `docs/contracts/`
  - Interface contracts for cross-module behavior.
- `docs/milestones/`
  - Milestone planning and acceptance criteria.
- `docs/CODE_OF_CONDUCT.md`
  - Community behavior and enforcement policy.
- `docs/CONTRIBUTING.md`
  - Contribution workflow, validation checklist, and PR expectations.

## Quick Commands

Build stable targets:

```bash
cmake -S . -B build -D PYC_BUILD_EXPERIMENTAL=OFF
cmake --build build --parallel --target pyc pyc_core pyc_foundation
./build/pyc
```

Run benchmark harness:

```bash
python3 benchmark/harness.py --repeats 7 --micro-rounds 4000
```
