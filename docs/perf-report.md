# Performance Report (Current State)

## Executive Summary

PyC is currently strongest on deterministic CPU-path execution in the standardized benchmark shape and weakest on GPU-native execution when CUDA falls back to proxy mode.

## Current Readout (Run `20260218T023355Z_phase4_final`)

- CPU: PyC mean latency `24.0459 ms` (best among available adapters in this run).
- GPU: PyC mean latency `25.5228 ms` vs sub-millisecond native torch baselines.
- Reliability signal: PyC GPU recorded fallback events, indicating non-native dispatch in this run.

## What Is Working

- Stable build/CI artifacts and deterministic target graph.
- Reproducible benchmark metadata and chart generation.
- Runtime reliability telemetry is present and actionable.

## Main Bottleneck

GPU-native kernel path is not yet consistently active in benchmark environments. Until native CUDA kernels are selected without fallback, PyC GPU throughput/latency will remain non-competitive.

## Immediate Focus

1. Lock strict native CUDA benchmark gates for PyC in GPU runs.
2. Profile dispatch and graph execution split for native path verification.
3. Re-run adapter suite only after native mode is confirmed by reliability counters.
