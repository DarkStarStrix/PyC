# PyC Benchmark Results

- Timestamp (UTC): 2026-02-20T02:14:04.806034+00:00
- Platform: macOS-26.3-arm64-arm-64bit-Mach-O
- CPU: arm
- Python: 3.14.0
- Build directory: `/Users/allanmurimiwandia/PyC/PyC/build`

## Build

- Configure: 347.739 ms
- Build (`pyc pyc_core pyc_foundation pyc_core_microbench`): 489.262 ms

## Runtime

- `pyc` smoke: mean 38.09 ms (min 2.256, max 180.431, stdev 71.171)
- `pyc_core_microbench`: mean 251.19 ms (min 151.581, max 636.968, stdev 192.901)

## Artifact Sizes

- `pyc`: 33424 bytes
- `pyc_core`: 6880 bytes

## Notes

- These numbers benchmark the stable core targets currently built in CI.
- They do not benchmark the experimental full compiler pipeline.
