# Kernel Optimization Playbook

## Priority Order

1. GEMM + fused epilogues
2. Reductions + LayerNorm
3. Conv2D family

## Registry Interface

Defined in `include/pyc/kernel_registry.h`:

- register kernels by op key + backend
- select highest-priority compatible kernel
- store benchmark signals per op/backend

## Experiment Loop

1. Generate candidate configuration.
2. Validate numerical correctness.
3. Benchmark with fixed protocol.
4. Persist best candidate.
5. Route runtime selection by op key + backend.

## Current Ada FP32 Baseline

- Current custom-kernel winner: `ada_gemm_k64_warp32_async`
- Source: `kernels/prototypes/ada/gemm_k64_warp32_async/kernel.cu`
- Shape: `1024 x 1024 x 1024`
- Geometry: `64 x 64 x 64` tile, `32 x 8` threads, `8 x 2` per-thread output
- Standalone event-timed result on RTX 6000 Ada: `30.394 TFLOPS`, `best_ms=0.071`, `max_abs_diff=0.000000`
- Standalone host-wall timing on the same run: `mean_wall_ms=0.076`, `best_wall_ms=0.076`

## Runtime Promotion Status

- Lean promoted runtime path:
  - `execution_path = cuda_promoted_gemm_graph:ada_gemm_k64_warp32_async_f32`
  - `mean wall ms = 0.0754`
  - `28.4903 TFLOPS`
- Important comparison rule:
  - do not compare standalone CUDA-event timing directly against full runtime wall time
  - compare standalone wall time against promoted runtime wall time
- Current read:
  - the promoted path now matches and slightly edges the standalone wall-clock surface on `1024^3`
  - the large earlier gap was mostly event-timing vs wall-timing mismatch, plus small runtime selection overhead that is now shaved down

## Recent Measured Delta

- Prior winner: `ada_gemm_k64_warp32_store2`
- Prior kernel time (`nsys`): `77.72 us`
- Current async winner kernel time (`nsys`): `68.41 us`
- Improvement: about `12.0%` lower kernel time and about `24.2%` higher throughput

## Current Decision

- The async/double-buffered shared-memory path is now the active prototype baseline.
- Cheap flag and small mapping tweaks are no longer the priority path.
- Next tuning should focus on either:
  - profiling other shapes against this async winner, or
  - preparing promotion into the runtime/kernel-registry path.

## Benchmark Signals

- latency (p50/p95)
- throughput
- memory footprint
- stability (variance)

## Guardrail

No candidate is promoted unless it is both correct and faster than baseline on target workload.
