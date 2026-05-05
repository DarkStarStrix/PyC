# Hopper 45 TFLOPS Next Loop

## Basis

- `benchmark/remote_results/hopper_box_20260421/kernels_lab/hopper-tensorcore-bringup-20260421T191817Z.json`
- `benchmark/remote_results/hopper_box_20260421/profiles/hopper_tensor_core_bf16_4096_stats.csv`
- `benchmark/remote_results/hopper_box_20260421/profiles/pyc_runtime_cuda_32x1024_nograph_stats.csv`
- `benchmark/remote_results/hopper_box_20260421/hopper_json/20260421T192145Z__hopper_pyc_nsys_nograph.json`
- `docs/plans/hopper-ops-kernel-blueprint.md`

## What The Current Evidence Rules Out

- Hopper alone is not the win. `hopper_tensor_core_fp16` stayed effectively flat against the Ada FP16 baseline, so architecture naming without lane or pipeline changes is not enough.
- The current PyC runtime is not kernel-bound on the profiled small-shape path. The `nsys` API summary is dominated by `cudaMallocHost`, `cudaGetDeviceProperties`, and `cuLibraryLoadData`, while the cuBLAS kernels are only a few microseconds each.
- A one-size-fits-all next step is not justified. The standalone BF16 kernel and the PyC runtime are blocked by different mechanisms.

## Bottlenecks Most Likely Blocking 45 TFLOPS

1. Too little work per warp in the Hopper prototype.
   The current kernel maps one 16x16 accumulator fragment per warp, which leaves too much scheduling and synchronization overhead relative to math.

2. Too many K-stage barriers for the amount of compute being done.
   The current prototype uses a `64x64x16` stage, so every 16-wide K slice pays a block-wide synchronization cost.

3. Shared-memory layout is still generic rather than tuned for the WMMA path.
   The kernel has no stride padding or stronger staging structure, so bank behavior and fragment feed efficiency are still likely leaving throughput on the floor.

## Ranked Experiments

1. `hopper_tensor_core_bf16_perf`
   Hypothesis: the existing BF16 lane should be re-measured on a larger steady-state shape to set a clean baseline for the new loop.
   Code area: `kernels/lab/manifests/kernels.json`
   Expected signal: repeatable `4096^3` baseline near the prior ~27 TFLOPS profile point.
   Promote if: best run is stable within ~5 percent across repeats.
   Reject if: variance is too large to compare future kernel changes honestly.

2. `hopper_tensor_core_bf16_warp2n`
   Hypothesis: giving each warp two N fragments will raise arithmetic work per warp and reduce scheduler bubbles.
   Code area: `kernels/prototypes/hopper/tensor_core/kernel.cu`
   Expected signal: higher TFLOPS than the steady-state BF16 baseline at the same `4096^3` shape.
   Promote if: clear win over baseline without correctness regressions on aligned shapes.
   Reject if: throughput is flat or worse despite fewer threads per CTA.

3. `hopper_tensor_core_bf16_k32`
   Hypothesis: doubling K-stage depth from 16 to 32 will amortize synchronization and shared-memory feed overhead.
   Code area: `kernels/prototypes/hopper/tensor_core/kernel.cu`
   Expected signal: lower barrier tax and better TFLOPS than the baseline kernel.
   Promote if: the result beats baseline and does not crater occupancy enough to lose the gain.
   Reject if: larger stages only add pressure without improving throughput.

4. `hopper_tensor_core_bf16_warp2n_k32`
   Hypothesis: the two strongest structural levers will compose: more work per warp plus fewer K-stage handoffs.
   Code area: `kernels/prototypes/hopper/tensor_core/kernel.cu`
   Expected signal: this should be the highest ceiling variant in the loop and the most likely path toward the mid-30s or better.
   Promote if: it is the loop winner and remains numerically stable on a smaller correctness pass.
   Reject if: register pressure or feed stalls erase the theoretical gain.

## Focus

This loop should prioritize standalone kernel work first, then use one PyC runtime profile as a control.

- Standalone kernel work is the only path in this evidence set that can plausibly move toward `45 TFLOPS`.
- Runtime ops work still matters, but it addresses end-to-end latency and utilization, not the standalone throughput ceiling.

## Research Merge

- Peirce's synthesis matched the local evidence: the remaining gap is pipeline/feed efficiency, not simple occupancy or architecture retargeting.
- The current prototype is still a WMMA-era shared-memory-fed loop, so the lightweight experiments in this doc should be treated as the last cheap structural pass before switching to a Hopper-native producer/consumer design.

## Internet-Backed Cutoff

- The official NVIDIA Hopper Tuning Guide calls out Tensor Memory Accelerator as the architecture feature that reduces register and SM overhead for data movement and enables warp-specialized producer/consumer execution on Hopper.
- CUTLASS Hopper guidance points in the same direction: warp-specialized persistent and ping-pong GEMM designs overlap producer, consumer, and epilogue work instead of paying repeated load-wait-compute cycles.
- That changes the stop condition for this loop:
  if the `warp2n` and `k32` variants do not move the ceiling decisively, stop shuffling WMMA tiles and build a real Hopper-native TMA/WGMMA path next.

## First Execution Readout

Artifact:
- `benchmark/remote_results/hopper_box_20260421/kernels_lab/hopper-bf16-nextloop-wmma-20260421T193831Z.json`

Ranking on `4096x4096x4096` BF16 steady-state:
- `hopper_tensor_core_bf16_perf`: `26.959 TFLOPS`, `best_ms=5.098`
- `hopper_tensor_core_bf16_warp2n`: `24.672 TFLOPS`, `best_ms=5.571`
- `hopper_tensor_core_bf16_warp2n_k32`: `22.486 TFLOPS`, `best_ms=6.112`
- `hopper_tensor_core_bf16_k32`: `21.125 TFLOPS`, `best_ms=6.506`

Conclusion:
- The baseline `64x64x16` BF16 WMMA lane is still the winner.
- Increasing per-warp N work without a Hopper-native feed path regressed.
- Doubling K-stage depth also regressed, which means the extra staging cost outweighed any barrier amortization in this kernel shape.
- This confirms the cutoff above: the next serious move is no longer WMMA tile shuffling. It is a real Hopper-native TMA/WGMMA pipeline.

## Hopper Control Ceiling

Artifacts:
- `benchmark/remote_results/hopper_box_20260421/kernels_lab/hopper_cublaslt_bf16_check-20260421T194551Z.json`
- `benchmark/remote_results/hopper_box_20260421/kernels_lab/hopper_cublaslt_bf16-20260421T194639Z.json`
- `benchmark/remote_results/hopper_box_20260421/profiles/hopper_cublaslt_bf16_4096_stats.csv`

Hardware:
- `NVIDIA H100 80GB HBM3`

Readout:
- `hopper_cublaslt_bf16_check` (`512^3`, reference on): `30.615 TFLOPS`, `best_ms=0.009`, `max_abs_diff=0.0`
- `hopper_cublaslt_bf16` (`4096^3`, reference off): `846.466 TFLOPS`, `best_ms=0.162`
- `nsys` ground truth on the `4096^3` run shows the dominant GPU kernel `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT` averaging about `158371 ns`, which matches the internal timer and confirms the ceiling is real on this box.

Implication:
- The current best custom Hopper WMMA kernel (`26.959 TFLOPS`) is about `31.4x` behind the hardware-native library lane on this problem size.
- That gap is too large to close with incremental WMMA tuning. The next owned-kernel loop should be judged against the `cuBLASLt` control, but it should optimize toward Hopper-native overlap mechanics:
  producer/consumer scheduling, larger useful pipeline stages, and eventually TMA/WGMMA rather than more `16x16x16` WMMA shuffling.

Acceptance framing for the next loop:
- Keep `hopper_tensor_core_bf16_perf` as the owned-kernel baseline.
- Keep `hopper_cublaslt_bf16` as the control ceiling.
- Do not spend more time on WMMA retuning unless a change can be justified as enabling the Hopper-native path.

## New Owned-Kernel Band

The owned-kernel target for the next Hopper phase is no longer `45 TFLOPS`.
It is the `80-120 TFLOPS` band on `4096x4096x4096` BF16 GEMM on H100.

Interpretation:
- `< 80 TFLOPS`: not enough structural progress to justify the complexity of a new Hopper-native path
- `80-120 TFLOPS`: credible first Hopper-native owned-kernel band
- `> 120 TFLOPS`: strong result and a good sign that the implementation is on the right side of Hopper's execution model

Execution contract:
- baseline: `hopper_tensor_core_bf16_perf`
- control ceiling: `hopper_cublaslt_bf16`
- only promote a new owned kernel if it beats the current owned baseline and lands inside or above the `80-120 TFLOPS` band without correctness regressions

What changes from here:
- Stop spending engineering time on WMMA-only retuning.
- Use the task system to track the owned baseline, the control ceiling, and the target band together.
- Treat the next implementation slice as Hopper-native pipeline work:
  wider useful stages, overlapped producer/consumer structure, and eventually TMA/WGMMA.

## Async Overlap Readout

Artifacts:
- `benchmark/remote_results/hopper_box_20260421/kernels_lab/hopper-sm90-80-120-async-candidates-20260421T201129Z.json`
- `benchmark/remote_results/hopper_box_20260421/profiles/hopper_tensor_core_bf16_async_wide_4096_stats_cuda_gpu_kern_sum.csv`
- `benchmark/remote_results/hopper_box_20260421/profiles/hopper_tensor_core_bf16_async_wide_4096_stats_cuda_api_sum.csv`

New candidates:
- `hopper_tensor_core_bf16_async`
- `hopper_tensor_core_bf16_async_wide`

Design change:
- Keep the same standalone harness and BF16 WMMA math path.
- Replace the serial shared-memory refill loop with `cp.async` double-buffered staging.
- Increase useful work per warp from one `16x16` accumulator fragment to four fragments per warp.
- Widen the promoted CTA to `64x128x32` with `8` warps so each stage amortizes more scheduling and barrier cost before switching stages.
- Store the B tile row-major in shared memory so the async copies are contiguous and legal, then load B fragments as row-major WMMA fragments.

Correctness:
- `hopper_tensor_core_bf16_async` passed `512^3` with `max_abs_diff=0.0`
- `hopper_tensor_core_bf16_async_wide` passed `512^3` with `max_abs_diff=0.0`

Steady-state ranking on `4096x4096x4096` BF16:
- `hopper_cublaslt_bf16`: `852.514 TFLOPS`, `best_ms=0.161`
- `hopper_tensor_core_bf16_async_wide`: `136.826 TFLOPS`, `best_ms=1.004`
- `hopper_tensor_core_bf16_async`: `129.824 TFLOPS`, `best_ms=1.059`
- `hopper_tensor_core_bf16_perf`: `26.855 TFLOPS`, `best_ms=5.118`

Outcome:
- The owned Hopper baseline moved from about `26.9 TFLOPS` to `136.8 TFLOPS`.
- That is about a `5.1x` speedup over the prior owned baseline.
- The promoted owned kernel now sits at about `16.1%` of the `cuBLASLt` control lane on this shape.
- The `80-120 TFLOPS` target band was cleared; `hopper_tensor_core_bf16_async_wide` is the new SM90 owned baseline.

## What Worked

The key win was not a new math primitive. It was feeding the existing math path properly.

- The old kernel forced a full-CTA `load -> sync -> compute -> sync` cadence on every K slice.
- The new kernel overlaps next-stage global-to-shared movement with current-stage WMMA compute.
- The widened CTA gives each stage more useful tensor work before the next handoff.
- The shared-memory layout for B now matches contiguous global reads, so the async staging path is actually efficient instead of structurally mismatched.

This is the first strong local proof of the Hopper worklog thesis inside this repo:
pipeline structure mattered much more than another round of tile shuffling.

## Remaining Gap

The profile confirms that the owned path is now kernel-dominated instead of orchestration-dominated.

- `nsys` reports the promoted kernel at about `997759 ns` for the profiled `4096^3` run.
- The prior owned baseline was about `5168862.8 ns`.
- The `cuBLASLt` control kernel is still only about `158371 ns`.

Implication:
- The async WMMA path removed a large amount of exposed bubble cost.
- The remaining gap is still about `6.2x` to the control ceiling.
- That remaining gap is too large to explain with host overhead; it is now inside the device pipeline itself.

Most likely remaining bottlenecks:
- WMMA fragment issue/consumption is still less efficient than Hopper-native WGMMA warpgroup execution.
- Shared-memory staging still consumes scheduler and instruction budget that TMA is designed to remove.
- Epilogue/store behavior is still generic relative to the highly tuned library path.

## Promotion And Next Move

Promotion:
- `task_baselines.json` now promotes `hopper_tensor_core_bf16_async_wide` as the SM90 owned GEMM baseline.
- `hopper-sm90-80-120.json` is marked completed with `hopper_tensor_core_bf16_async_wide` as the winner.

Next move:
- Keep this async-wide lane as the owned baseline and guardrail.
- Do not go back to serial WMMA retuning.
- Build the next Hopper lane around the same overlap idea but with Hopper-native primitives:
  TMA for staged movement, WGMMA for the mainloop, and a deliberate epilogue path.
