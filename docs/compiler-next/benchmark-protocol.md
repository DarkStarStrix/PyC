# Compiler-Next Benchmark Protocol

## Objective

Produce reproducible, decision-grade performance signals for compiler-next components.

## Measurement Rules

1. Use fixed repeats and warm-up policy.
2. Record machine metadata with every run.
3. Compare only like-for-like hardware classes.
4. Report median and spread, not single-run bests.

## Required Outputs

- JSON artifact for automation
- Markdown summary for human review
- Visualization artifact for quick inspection

## Acceptance Gates

1. Correctness must pass before perf comparison.
2. End-to-end model latency is the primary KPI.
3. Memory reduction and kernel wins are tracked as secondary KPIs.

## CI Integration Path

- Early phases: non-blocking perf jobs.
- Later phases: threshold-based regression gates.
