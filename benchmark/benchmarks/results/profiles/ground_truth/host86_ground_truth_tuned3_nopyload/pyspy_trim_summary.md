# Import/Load Overhead Trim Summary

## Applied trimming
- Removed `statistics` import path in `encoder_ground_truth.py` (avoids `statistics -> fractions -> decimal`).
- Single-shape/no-batch-scaling profiling mode to avoid setup sweep noise.
- Long-duration py-spy captures to amortize startup/teardown overhead.

## Classic flamegraph outputs
- `pyspy_flame_clean.svg` (startup-heavy)
- `pyspy_flame_steadyish.svg` (20s, steady-state biased)
- `pyspy_flame_steadyish_trim.svg` (20s, trimmed imports)
- `pyspy_flame_steadyish_long.svg` (60s)
- `pyspy_flame_steadyish_xlong.svg` (180s, most lean)

## `_find_and_load` peak progression
- Startup-heavy: `33.17%`
- 20s steady-state: `9.96%`
- 20s trimmed: `7.25%`
- 60s long: `3.47%`
- 180s xlong: `1.45%`

Current recommended flamegraph for hot-path analysis:
- `pyspy_flame_steadyish_xlong.svg`

## Environment hard limit
- True attach-mode (`py-spy --pid` after warmup) still blocked in this container due missing `CAP_SYS_PTRACE` in bounding set.
- To fully eliminate process-start noise, rerun container with `--cap-add SYS_PTRACE --security-opt seccomp=unconfined`.
