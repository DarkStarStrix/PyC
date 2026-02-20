# Host86 Identity + Arena Summary

## 1) Compiled path GEMM identity verification
- Eager capture: `host86_identity_v1_eager`
- Compiled capture (Inductor + `INDUCTOR_GEMM_BACKENDS=ATEN`): `host86_identity_v1_compiled`
- Identity report: `host86_identity_v1_identity_report.json`

Result:
- `exact_top_k_match = true`
- Top-2 GEMM kernels are identical between eager and compiled paths:
- `cutlass_80_tensorop_f16...128x256...`
- `cutlass_80_tensorop_f16...256x128...`

This confirms compiled lowering is routing into the same dominant tensor-core GEMM algorithm family as eager for this workload.

## 2) Buffer arena (CUDA Graph) stability pass
- Arena run: `host86_arena_v1`
- Arena vs eager identity report: `host86_arena_v1_identity_vs_eager.json`

Result:
- `allocation_event_delta = 0`
- `segment_alloc_event_delta = 0`
- `memory_stable = true`

Latency/throughput (bucket seq=256):
- Eager: `15.7974 ms`, `1,037,132.98 tok/s`, alloc delta `1200`
- Compiled ATEN: `15.8256 ms`, `1,035,286.33 tok/s`, alloc delta `720`
- Arena: `15.9618 ms`, `1,026,451.95 tok/s`, alloc delta `0`

## 3) Interpretation
- Math path is matched (eager vs compiled ATEN) at kernel-identity level.
- Arena removes allocation churn completely with modest throughput cost (~1%).
- For kernel identity, use non-graph captures; graph replay captures may only show `cudaGraphLaunch` API (no kernel rows), which is expected.
