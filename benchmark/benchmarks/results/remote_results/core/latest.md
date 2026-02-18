# PyC Benchmark Results

- Timestamp (UTC): 2026-02-17T01:25:10.517752+00:00
- Platform: macOS-26.3-arm64-arm-64bit-Mach-O
- CPU: arm
- Python: 3.14.0
- Build directory: `/Users/allanmurimiwandia/PyC/PyC/build`

## Build

- Configure: 73.822 ms
- Build (`pyc pyc_core pyc_foundation pyc_core_microbench`): 1582.912 ms

## Runtime

- `pyc` smoke: mean 63.085 ms (min 2.142, max 303.614, stdev 120.266)
- `pyc_core_microbench`: mean 218.771 ms (min 161.692, max 384.332, stdev 85.458)

## Artifact Sizes

- `pyc`: 33424 bytes
- `pyc_core`: 7856 bytes

## Notes

- These numbers benchmark the stable core targets currently built in CI.
- They do not benchmark the experimental full compiler pipeline.
