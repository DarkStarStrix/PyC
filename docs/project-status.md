# Project Status

## Snapshot

- Stage: `Alpha`
- Stable build contract: `Enabled`
- Canonical CI workflow: `Enabled`
- Experimental compiler pipeline: `Present, not fully stabilized`

## Stability Matrix

| Area | Status | Notes |
|---|---|---|
| CMake configure/build for stable targets | Stable | Built in canonical CI across OS matrix. |
| `pyc` smoke execution | Stable | Deterministic output used for smoke testing. |
| `pyc_core` and `pyc_foundation` linking contract | Stable | Explicitly built and intended for downstream consumers. |
| Benchmark harness (`benchmark/harness.py`) | Stable | Deterministic core-focused metrics emitted as JSON/Markdown. |
| Full compiler pipeline modules | Experimental | Not yet part of stable CI guarantees. |

## Stable Targets

- `pyc_core_obj`
- `pyc_core`
- `pyc_foundation`
- `pyc`

## Known Constraints

- Stable source set is intentionally narrow to preserve portability and reproducibility.
- Performance results are environment-dependent and should be compared only across similar machine classes.
- `ctest` currently runs non-fatally in CI and may report no tests depending on configured suites.

## Near-Term Priorities

1. Expand stable coverage module-by-module with explicit portability checks.
2. Increase correctness test depth for promoted modules.
3. Add benchmark trend comparison and optional threshold guardrails.
4. Reduce ambiguity in experimental module integration contracts.
