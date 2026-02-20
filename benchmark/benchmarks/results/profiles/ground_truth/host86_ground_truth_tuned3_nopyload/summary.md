# Tuned3 Profile Summary (Reduced Python/Setup Overhead)

- Run ID: `host86_ground_truth_tuned3_nopyload`
- Profile mode: single shape (`seq=256`), no batch scaling, one model build per process.
- Key changes that reduced setup overhead:
- `graph_build_calls`: `4 -> 1` in `nsys_single_pass.json`.
- Nsight profile capture no longer sweeps extra buckets in the profiled process.

## Single-pass seq256 comparison
- Previous (`host86_ground_truth_tuned2`): `15.8706 ms`
- Current (`host86_ground_truth_tuned3_nopyload`): `15.1260 ms`
- Delta: `-4.69%` latency improvement.

## New flamegraph
- `nsys_kernel_flame.svg` generated at:
- `benchmark/benchmarks/results/profiles/ground_truth/host86_ground_truth_tuned3_nopyload/nsys_kernel_flame.svg`
