# Benchmarking Guide

## Objective

Benchmarking in this project is currently designed to answer one question reliably:

- Are stable core targets getting slower or larger over time?

It is intentionally scoped to stable targets, not full experimental compiler behavior.

## Harness Design

Harness: `benchmark/harness.py`

Inputs:

- CMake configure/build commands.
- Stable executable `pyc`.
- Benchmark workload executable `pyc_core_microbench`.

Outputs:

- `benchmark/results/latest.json`
- `benchmark/results/latest.md`
- `docs/performance-results.md`

## Metrics Collected

- Configure latency (ms)
- Build latency (ms)
- Smoke run latency for `pyc` (ms, sampled)
- Microbenchmark latency for `pyc_core_microbench` (ms, sampled)
- Artifact sizes for `pyc` and `pyc_core` (bytes)

## Run Commands

Default run:

```bash
python3 benchmark/harness.py
```

Tuned run:

```bash
python3 benchmark/harness.py --build-dir build --config Release --repeats 7 --micro-rounds 4000
```

## Methodology Notes

- First sample often includes warm-up overhead.
- Compare runs on equivalent machine types.
- Keep repeat count fixed when tracking trends.
- Treat large outliers as signals to investigate toolchain/host noise.

## Suggested Interpretation Rules

1. Runtime mean increase >10% with similar stdev: investigate regression.
2. Runtime stdev increase >2x with similar mean: investigate instability/noise.
3. Artifact size growth >15% without planned change: investigate binary bloat.

## Extending the Harness

When adding new benchmarks:

1. Prefer deterministic workloads with explicit checksums.
2. Keep dependency surface minimal.
3. Emit both machine-readable and human-readable results.
4. Document why the new metric is decision-relevant.

## Current Limitations

- No historical time-series store in-repo.
- No automated threshold gating in CI yet.
- Full experimental pipeline is not benchmarked here.
