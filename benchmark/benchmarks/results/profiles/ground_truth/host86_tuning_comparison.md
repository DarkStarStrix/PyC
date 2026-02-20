# Host86 Ground-Truth Tuning Comparison (RTX 3090)

Runs compared:
- Baseline: `host86_ground_truth_run2`
- Tuned 1 (compiled): `host86_ground_truth_tuned1` (`torch.compile=max-autotune`, TF32, inference-mode, in-place residual)
- Tuned 2 (eager tuned): `host86_ground_truth_tuned2` (TF32, inference-mode, in-place residual)

## Key outcomes
- Tensor-core path already active in baseline and tuned eager runs (`sm80_xmma...`, `cutlass_80_tensorop...`).
- `torch.compile` path achieved allocator flatline (`allocation_event_delta=0`, `memory_stable=true`) across buckets.
- `torch.compile` path did **not** improve steady-state latency for this synthetic encoder case.

## Bucket latency / throughput
| Run | Seq 128 mean ms | Seq 256 mean ms | Seq 512 mean ms | Seq 256 throughput tokens/s |
|---|---:|---:|---:|---:|
| Baseline | 8.3668 | 16.0240 | 32.6037 | 1,022,464.92 |
| Tuned 1 (compiled) | 8.3640 | 16.5389 | 33.0223 | 990,631.48 |
| Tuned 2 (eager tuned) | 8.4224 | 16.1310 | 32.5815 | 1,015,683.57 |

## Delta vs baseline (seq 256)
- Tuned 1 (compiled): latency `+3.21%`, throughput `-3.11%`
- Tuned 2 (eager tuned): latency `+0.67%`, throughput `-0.66%`

## Memory behavior
- Baseline: allocation-event deltas positive (e.g., `+1440` in bucket checks).
- Tuned 1 (compiled): allocation-event deltas `0`, segment deltas `0`.
- Tuned 2 (eager tuned): still positive allocation-event deltas (improved but non-zero).

## Profiler caveat
- Nsight Compute remains blocked by `ERR_NVGPUCTRPERM` on this host.
- Compiled run Nsight Systems trace emphasizes `cudaGraphLaunch`; kernel-sum tables are not directly comparable to eager kernel-sum composition.
