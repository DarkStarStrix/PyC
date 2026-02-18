# PyC Benchmark Results

- Timestamp (UTC): 2026-02-18T02:30:14.913990+00:00
- Platform: macOS-26.3-arm64-arm-64bit-Mach-O
- CPU: arm
- Python: 3.14.0
- Build directory: `/Users/allanmurimiwandia/PyC/PyC/build`

## Build

- Configure: 295.994 ms
- Build (`pyc pyc_core pyc_foundation pyc_core_microbench`): 247.806 ms

## Runtime

- `pyc` smoke: mean 1.91 ms (min 1.595, max 2.414, stdev 0.36)
- `pyc_core_microbench`: mean 9.488 ms (min 8.438, max 11.082, stdev 1.146)

## Artifact Sizes

- `pyc`: 33424 bytes
- `pyc_core`: 6880 bytes

## Notes

- These numbers benchmark the stable core targets currently built in CI.
- They do not benchmark the experimental full compiler pipeline.
