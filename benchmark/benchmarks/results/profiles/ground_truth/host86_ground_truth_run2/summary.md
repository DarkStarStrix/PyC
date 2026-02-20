# Ground Truth Runtime Summary (host86, RTX 3090)

## Run Scope
- Device: CUDA (`float16`), hidden=2048, base batch=64
- Buckets: `128,256,512`
- Batch sweep: `16,32,64,96,128`
- Timing policy: strict sync around timing boundaries (`sync -> start -> run -> sync -> stop -> sync`)

## Ground Truth Checks
- Compile calls in steady-state: `0`
- Graph-build calls in steady-state: `0`
- Segment allocation deltas: `0` in all measured phases
- Allocation event deltas: positive in all measured phases (allocator activity still happens per iteration)

## Latency/Throughput Snapshot
- Bucket 128: mean `8.3668 ms`, throughput `979,104.92 tokens/s`
- Bucket 256: mean `16.0240 ms`, throughput `1,022,464.92 tokens/s`
- Bucket 512: mean `32.6037 ms`, throughput `1,005,040.53 tokens/s`
- Batch 16 -> 128 throughput: `939k -> 1.00M tokens/s` (near saturation by batch 64)

## Nsight Systems Findings
- Captured kernels in profiled pass: `9`
- Top kernels dominate time:
- `cutlass_80_tensorop_f16...relu...`: `47.7%`
- `sm80_xmma_gemm_f16...`: `44.8%`
- Remaining kernels are elementwise/reduction/layernorm and are much smaller.
- Inter-kernel gaps from timeline: ~`0.0006-0.0007 ms` (very small)
- CUDA API summary is sync-heavy by design (`cudaDeviceSynchronize` dominates due deterministic timing enforcement)

## Nsight Compute Status
- `ncu` capture attempted and handled deterministically.
- Result: `ERR_NVGPUCTRPERM` (GPU performance counters not enabled for this host).
- Action to unlock detailed occupancy/bandwidth counters: enable non-admin perf access on host driver (`NVreg_RestrictProfilingToAdminUsers=0`) and retry.
