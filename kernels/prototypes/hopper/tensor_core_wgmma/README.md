# Hopper Tensor Core WGMMA Lane

This directory is reserved for the Hopper-native gap-closing kernel lane.

Current state:

- `tensor_core_async` is the owned WMMA guardrail lane.
- `cuBLASLt` is the control ceiling lane.
- `tensor_core_wgmma` is the next implementation lane intended to close the remaining device-side gap.

Minimum contract for the first implementation:

- target shape: `4096x4096x4096`
- dtype: BF16 input, FP32 accumulation
- architecture: `sm90`
- correctness lane: `512x512x512` with reference enabled
- performance lane: `4096x4096x4096` with reference disabled

Design goals:

- use warpgroup MMA rather than per-warp WMMA
- move toward TMA-backed staging for the long-term feed path
- preserve a standalone harness so the lane can be benchmarked by `kernel_lab`
- keep the first fast path simple: no generalized epilogue work unless it is proven necessary

Promotion rule:

- only promote a `tensor_core_wgmma` variant if it beats the current owned SM90 baseline without correctness regressions
