# CUDA GEMM Fast Path

This document formalizes the current PyC FP32 CUDA GEMM fast path on Ada-class GPUs.

## Intent

The goal of this path is simple:

1. Keep compiler/runtime overhead negligible.
2. Reuse persistent CUDA state across repeated runs.
3. Route FP32-compatible GEMM onto the best available CUDA library path.
4. Fall back deterministically when a faster path is unavailable.

## Mechanical Flow

For a supported `matmul` graph in [`src/compiler/runtime/cuda_backend.c`](/Users/allanmurimiwandia/PyC/PyC/src/compiler/runtime/cuda_backend.c):

1. `pyc_cuda_dispatch()` verifies CUDA availability and checks that the IR is in the supported native subset.
2. `execute_native_cuda_graph_cuda()` extracts the `M x K` lhs, `K x N` rhs, and `M x N` output contract from the IR.
3. `pyc_cuda_workspace_ensure()` prepares persistent device buffers, a CUDA stream, a cuBLAS handle, a cuBLASLt handle, and optional cuBLASLt workspace.
4. The runtime decides whether lhs/rhs copies can be skipped for stable repeated inputs.
5. If a promoted GEMM symbol is enabled, PyC tries the promoted CUTLASS path first.
6. Otherwise PyC takes the native CUDA GEMM path:
   - try CUDA graph replay/capture for stable signatures
   - inside that path, call `run_best_fp32_gemm()`
7. `run_best_fp32_gemm()` prefers `cublasLtMatmul` and falls back to the older cuBLAS GEMM path if Lt heuristic selection fails.
8. Post-matmul epilogues such as `add` and `relu` are applied if the graph requires them.
9. Runtime stats are emitted back through `pyc_run_stats`.

## Why It Got Faster

The earlier gap was not in compiler control logic. It was in the GEMM path itself.

The current fast path improves throughput through three concrete mechanisms:

1. Persistent workspace reuse.
   - CUDA stream, cuBLAS handle, cuBLASLt handle, device buffers, and optional Lt workspace are reused across runs.

2. Better library path selection.
   - FP32-compatible GEMM now prefers `cublasLtMatmul` with heuristic selection and workspace, instead of relying only on the simpler cuBLAS path.

3. Tensor Core-friendly FP32 policy.
   - When enabled, PyC allows TF32-compatible execution for the FP32 benchmark/runtime path.

## Runtime Decision Order

The effective order is:

1. promoted CUTLASS GEMM, if explicitly enabled and promoted
2. cuBLASLt FP32/TF32 GEMM
3. cuBLAS FP32 GEMM fallback
4. deterministic CPU fallback, if native CUDA execution cannot proceed

This keeps the path aggressive but still deterministic.

## Key Controls

Useful environment variables for this path:

- `PYC_CUDA_ENABLE_CUBLASLT=1`
  - Prefer the cuBLASLt GEMM path.
- `PYC_CUDA_LT_WORKSPACE_BYTES=<bytes>`
  - Size of the reusable cuBLASLt workspace. Current tuned value on Ada testing was `33554432`.
- `PYC_CUDA_ALLOW_TF32=1`
  - Allow TF32-compatible FP32 GEMM behavior on supported hardware.
- `PYC_CUDA_ENABLE_GRAPH_REPLAY=1`
  - Allow CUDA graph capture/replay for stable signatures.
- `PYC_CUDA_ASSUME_STATIC_RHS=1`
  - Reuse uploaded RHS when the pointer/size contract is stable.
- `PYC_CUDA_ASSUME_STATIC_LHS=1`
  - Same idea for lhs. This is mainly useful in controlled repeated benchmark loops.
- `PYC_CUDA_SKIP_HOST_OUTPUT_COPY=1`
  - Benchmark-oriented switch to avoid timing a device-to-host copy when comparing against frameworks that keep results on device.

## Benchmark vs Runtime Rules

Two switches are benchmark-only conveniences and should not be treated as production defaults:

- `PYC_CUDA_ASSUME_STATIC_LHS`
- `PYC_CUDA_SKIP_HOST_OUTPUT_COPY`

They are valid in tightly controlled loops where input/output semantics are known in advance, but they are not safe assumptions for arbitrary runtime integration.

The more generally useful runtime optimizations are:

- persistent workspace reuse
- cuBLASLt preference
- TF32 policy when the precision contract allows it
- graph replay for stable signatures
- static RHS reuse when the weight tensor is actually stable

## Current Outcome

On the Ada sweep that motivated this path, the direct `1024 x 1024 x 1024` PyC FP32-compatible GEMM moved from roughly `5 TFLOPS` to roughly `28 TFLOPS`, and the full FP32 comparison sweep showed PyC leading every tested shape under the tuned cuBLASLt path.

This should be treated as the current reference fast path for PyC FP32 CUDA GEMM on Ada-class machines.
