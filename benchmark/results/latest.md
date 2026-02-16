# PyC Benchmark Results

- Timestamp (UTC): 2026-02-16T23:33:08.057997+00:00
- Platform: macOS-26.3-arm64-arm-64bit-Mach-O
- CPU: arm
- Python: 3.14.0
- Build directory: `/Users/allanmurimiwandia/PyC/PyC/build`

## Build

- Configure: 59.674 ms
- Build (`pyc pyc_core pyc_foundation pyc_core_microbench`): 257.738 ms

## Runtime

- `pyc` smoke: mean 1.783 ms (min 1.614, max 2.107, stdev 0.161)
- `pyc_core_microbench`: mean 361.203 ms (min 323.778, max 456.805, stdev 40.596)

## Artifact Sizes

- `pyc`: 33424 bytes
- `pyc_core`: 7856 bytes

## Notes

- These numbers benchmark the stable core targets currently built in CI.
- They do not benchmark the experimental full compiler pipeline.
