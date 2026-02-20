# Classic Flamegraphs (py-spy)

Generated classic stacked flamegraphs (peaks/blocks):

- `pyspy_flame.svg` (high-rate run, includes startup)
- `pyspy_flame_clean.svg` (lower-rate run, includes startup)
- `pyspy_flame_steadyish.svg` (20s capture on long run to bias toward steady-state)

Observed import/bootstrap prominence reduction:
- Startup-biased (`pyspy_flame_clean.svg`): top import stack ~`33.17%`
- Steady-state-biased (`pyspy_flame_steadyish.svg`): top import stack ~`9.96%`

The steady-state-biased flamegraph is the one to use for hot-path analysis.
