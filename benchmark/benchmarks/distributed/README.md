# Distributed Comm Benchmark

This suite benchmarks `pyc_collective_comm` backends through the distributed init path.

## Build

```bash
cmake -S . -B build-distributed \
  -D PYC_BUILD_EXPERIMENTAL=OFF \
  -D PYC_BUILD_COMPILER_NEXT=ON \
  -D PYC_BUILD_COMPILER_NEXT_TESTS=ON \
  -D PYC_BUILD_DISTRIBUTED_SCAFFOLD=ON \
  -D PYC_BUILD_BENCHMARKS=ON
cmake --build build-distributed --parallel
```

## Run

```bash
python3 benchmark/benchmarks/distributed/run_distributed_comm_suite.py \
  --build-dir build-distributed \
  --iters 4000 \
  --count 1024 \
  --repeats 3 \
  --tag distributed_comm_smoke
```

Outputs:

- `benchmark/benchmarks/results/json/<run_id>__<tag>.json`
- `benchmark/benchmarks/results/reports/<run_id>__<tag>.md`
- `benchmark/benchmarks/results/images/<run_id>__<tag>.svg`
- latest aliases:
  - `benchmark/benchmarks/results/json/latest_distributed_comm.json`
  - `benchmark/benchmarks/results/reports/latest_distributed_comm.md`
  - `benchmark/benchmarks/results/images/latest_distributed_comm.svg`
