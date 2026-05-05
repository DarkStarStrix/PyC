# Hopper Gap-Close Path

## Basis

- `benchmark/remote_results/hopper_box_20260421/kernels_lab/hopper-sm90-80-120-async-candidates-20260421T201129Z.json`
- `benchmark/remote_results/hopper_box_20260421/profiles/hopper_tensor_core_bf16_async_wide_4096_stats_cuda_gpu_kern_sum.csv`
- `benchmark/remote_results/hopper_box_20260421/profiles/hopper_cublaslt_bf16_4096_stats.csv`
- `kernels/prototypes/hopper/tensor_core_async/kernel.cu`

## Current State

On `4096x4096x4096` BF16 on H100:

- `hopper_tensor_core_bf16_async_wide`: about `1.004 ms`, `136.826 TFLOPS`
- `hopper_cublaslt_bf16`: about `0.161 ms`, `852.514 TFLOPS`

So the remaining gap is about:

- `6.2x` in kernel time
- about `715.7 TFLOPS` in throughput

The important constraint boundary has changed:

- The old problem was exposed pipeline bubbles and underfed WMMA work.
- The new problem is the device-side execution model itself.
- Host overhead is no longer the dominant explanation for the gap.

## Ranked Path

### 1. Short Async WMMA Shave Pass

Do one cheap structural pass on the promoted async lane. The goal is not to reach the control lane. The goal is to remove the last obvious low-cost taxes before committing to a Hopper-native lane.

Candidates:

- `hopper_tensor_core_bf16_async_wide`
- `hopper_tensor_core_bf16_async_wide_k64`
- `hopper_tensor_core_bf16_async_square`
- `hopper_tensor_core_bf16_async_square_k64`

Hypotheses:

- `k64` may amortize loop and stage handoff overhead better than `k32`
- a square `128x128` CTA may improve work-per-CTA on the `4096^3` square regime

Success bar:

- move the owned lane into roughly the `150-220 TFLOPS` band
- if no candidate clears that band with a real margin, stop spending time on WMMA polish

### 2. Lock The Best WMMA Guardrail

The best result from the shave pass becomes the guardrail kernel for the next phase.

Purpose:

- preserve a stable, known-good owned lane
- stop re-litigating earlier WMMA variants
- keep a clean comparison point for WGMMA/TMA work

### 3. Build The Hopper-Native Lane

This is the real gap-closing path.

Minimum bar for the first WGMMA/TMA prototype:

- BF16 `4096^3` on H100
- same standalone harness style as current kernels
- correctness on `512^3`
- one stable performance run at `4096^3`

Execution model goals:

- warpgroup MMA instead of per-warp WMMA
- TMA-backed bulk movement instead of manually issued shared-memory feed as the long-term path
- producer/consumer overlap as the default kernel structure
- explicit fast epilogue assumptions for the benchmarked lane

## Why This Split Matters

Do not mix these two questions:

1. can WMMA still give us a cheap improvement?
2. what closes the big remaining gap?

The first question is about saving easy engineering time.
The second question is about changing the kernel model.

If they are mixed together, the repo risks spending another loop polishing a lane that has already yielded most of its practical value.

## Operational Rule

Use `hopper-sm90-gap-close` as the short polish task.

If the winner is still materially above the promoted baseline, keep it.
If the winner is only marginally better, keep `hopper_tensor_core_bf16_async_wide` and move on.

After that, the main workstream should be:

- `tensor_core_async` as the owned WMMA guardrail
- `tensor_core_wgmma` as the real gap-closing implementation lane

## Execution Readout

Artifact:

- `benchmark/remote_results/hopper_box_20260421/kernels_lab/hopper-sm90-gap-close-20260421T202304Z.json`

Steady-state ranking on `4096x4096x4096` BF16:

- `hopper_cublaslt_bf16`: `0.161 ms`, `853.022 TFLOPS`
- `hopper_tensor_core_bf16_async_square_k64`: `0.926 ms`, `148.394 TFLOPS`
- `hopper_tensor_core_bf16_async_square`: `0.967 ms`, `142.119 TFLOPS`
- `hopper_tensor_core_bf16_async_wide`: `1.004 ms`, `136.839 TFLOPS`
- `hopper_tensor_core_bf16_async_wide_k64`: `1.016 ms`, `135.326 TFLOPS`

Correctness:

- `hopper_tensor_core_bf16_async_square_k64` passed `512^3` with `max_abs_diff=0.0`

Outcome:

- The best cheap shave was the square `128x128x64` async WMMA lane.
- It improved the owned baseline from about `1.004 ms` to about `0.926 ms`.
- That is about a `7.8%` kernel-time improvement and about an `8.4%` TFLOPS gain over the prior owned baseline.
- It still missed the `150 TFLOPS` lower bound by a small margin, which matters: the cheap WMMA pass helped, but it did not change the basic shape of the remaining gap.

Promotion:

- `hopper_tensor_core_bf16_async_square_k64` is now the SM90 owned GEMM baseline in `task_baselines.json`.
- Treat it as the locked WMMA guardrail for future comparisons.

Decision:

- Keep the promoted square `k64` lane.
- Do not spend another broad loop on WMMA-only polishing.
- The remaining work should move to `tensor_core_wgmma` and the Hopper-native feed path.
