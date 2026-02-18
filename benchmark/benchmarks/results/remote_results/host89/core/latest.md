# PyC Benchmark Results

- Timestamp (UTC): 2026-02-18T00:46:15.125448+00:00
- Platform: Linux-6.8.0-85-generic-x86_64-with-glibc2.35
- CPU: x86_64
- Python: 3.11.11
- Build directory: `/root/PyC/build`

## Build

- Configure: 157.729 ms
- Build (`pyc pyc_core pyc_foundation pyc_core_microbench`): 491.809 ms

## Runtime

- `pyc` smoke: mean 1.214 ms (min 1.141, max 1.311, stdev 0.065)
- `pyc_core_microbench`: mean 267.208 ms (min 260.243, max 274.62, stdev 5.459)

## Artifact Sizes

- `pyc`: 15960 bytes
- `pyc_core`: 11172 bytes

## Notes

- These numbers benchmark the stable core targets currently built in CI.
- They do not benchmark the experimental full compiler pipeline.
