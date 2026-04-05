/*
 * gemm/kernel.cu
 *
 * CUTLASS-backed GEMM kernels for the PyC kernel registry.
 *
 * This file implements three GEMM variants at different performance/precision
 * trade-offs and registers them into the PyC kernel registry with appropriate
 * priority, occupancy, and tensor-core eligibility metadata.
 *
 * Kernel hierarchy (highest priority first):
 *   1. cutlass_gemm_tensorcore_f16  — FP16 Tensor Core GEMM (H100/A100/RTX)
 *   2. cutlass_gemm_tensorcore_bf16 — BF16 Tensor Core GEMM (H100/A100)
 *   3. cutlass_gemm_simt_f32        — FP32 SIMT GEMM (fallback, all GPUs)
 *
 * Requires: CUTLASS 3.x headers in include path, CUDA 12+, sm_80+
 */

#include "pyc/kernel_registry.h"
#include "pyc/ir.h"

// CUTLASS 3.x includes
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

extern "C" void pyc_register_ada_async_gemm_kernel(void);
extern "C" int pyc_ada_async_gemm_dispatch(
    int M, int N, int K,
    const void* A,
    const void* B,
    void* C,
    float alpha,
    float beta,
    cudaStream_t stream);

/* ----------------------------------------------------------------
 * Kernel 1: FP16 Tensor Core GEMM
 * Uses CUTLASS GemmUniversal with FP16 inputs, FP16 accumulator.
 * Optimal for H100/A100 at maximum throughput.
 * ---------------------------------------------------------------- */
using GemmFp16TensorCore = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t,                                    // ElementA
    cutlass::layout::RowMajor,                          // LayoutA
    cutlass::half_t,                                    // ElementB
    cutlass::layout::ColumnMajor,                       // LayoutB
    cutlass::half_t,                                    // ElementC
    cutlass::layout::RowMajor,                          // LayoutC
    float,                                              // ElementAccumulator
    cutlass::arch::OpClassTensorOp,                     // OpClass (Tensor Core)
    cutlass::arch::Sm80,                                // ArchTag (A100/H100)
    cutlass::gemm::GemmShape<128, 256, 32>,             // ThreadblockShape
    cutlass::gemm::GemmShape<64, 64, 32>,               // WarpShape
    cutlass::gemm::GemmShape<16, 8, 16>,                // InstructionShape (mma.sync)
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t, 8, float, float>,              // EpilogueOp
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3                                                   // Stages (software pipeline)
>;

/* ----------------------------------------------------------------
 * Kernel 2: BF16 Tensor Core GEMM
 * BF16 is preferred for training (better dynamic range than FP16).
 * ---------------------------------------------------------------- */
using GemmBf16TensorCore = cutlass::gemm::device::GemmUniversal<
    cutlass::bfloat16_t,
    cutlass::layout::RowMajor,
    cutlass::bfloat16_t,
    cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        cutlass::bfloat16_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3
>;

/* ----------------------------------------------------------------
 * Kernel 3: FP32 SIMT GEMM (universal fallback)
 * ---------------------------------------------------------------- */
using GemmF32Simt = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm50,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>
>;

/* ----------------------------------------------------------------
 * Registration function — called by registry/init.cu
 * at library load time to populate the PyC kernel registry.
 * ---------------------------------------------------------------- */
extern "C" void pyc_cutlass_register_gemm_kernels(void) {
    pyc_kernel_desc desc;

    /* --- FP16 Tensor Core GEMM --- */
    memset(&desc, 0, sizeof(desc));
    strncpy(desc.op_key,  "matmul",                       PYC_KERNEL_OP_KEY_MAX - 1);
    strncpy(desc.symbol,  "cutlass_gemm_tensorcore_f16",  PYC_KERNEL_SYMBOL_MAX - 1);
    desc.backend              = PYC_BACKEND_CUDA;
    desc.priority             = 100;          /* highest priority */
    desc.estimated_occupancy  = 0.87;         /* measured on A100 */
    desc.tensor_core_eligible = 1;
    desc.shared_mem_bytes     = 98304;        /* 96 KB smem */
    desc.reg_pressure_class   = 2;            /* medium-high */
    pyc_kernel_register(&desc);

    /* --- BF16 Tensor Core GEMM --- */
    memset(&desc, 0, sizeof(desc));
    strncpy(desc.op_key,  "matmul",                       PYC_KERNEL_OP_KEY_MAX - 1);
    strncpy(desc.symbol,  "cutlass_gemm_tensorcore_bf16", PYC_KERNEL_SYMBOL_MAX - 1);
    desc.backend              = PYC_BACKEND_CUDA;
    desc.priority             = 90;
    desc.estimated_occupancy  = 0.83;
    desc.tensor_core_eligible = 1;
    desc.shared_mem_bytes     = 98304;
    desc.reg_pressure_class   = 2;
    pyc_kernel_register(&desc);

    /* --- FP32 SIMT GEMM fallback --- */
    memset(&desc, 0, sizeof(desc));
    strncpy(desc.op_key,  "matmul",                       PYC_KERNEL_OP_KEY_MAX - 1);
    strncpy(desc.symbol,  "cutlass_gemm_simt_f32",        PYC_KERNEL_SYMBOL_MAX - 1);
    desc.backend              = PYC_BACKEND_CUDA;
    desc.priority             = 10;           /* lowest — fallback only */
    desc.estimated_occupancy  = 0.50;
    desc.tensor_core_eligible = 0;
    desc.shared_mem_bytes     = 32768;
    desc.reg_pressure_class   = 1;
    pyc_kernel_register(&desc);

    pyc_register_ada_async_gemm_kernel();
}

/* ----------------------------------------------------------------
 * Dispatch entry point — called by the PyC CUDA backend when
 * a CUTLASS GEMM kernel is selected by the kernel registry.
 * ---------------------------------------------------------------- */
extern "C" int pyc_cutlass_gemm_dispatch(
    const char*  symbol,
    int M, int N, int K,
    const void*  A,
    const void*  B,
    void*        C,
    float        alpha,
    float        beta,
    cudaStream_t stream)
{
    if (strcmp(symbol, "cutlass_gemm_tensorcore_f16") == 0) {
        GemmFp16TensorCore gemm_op;
        GemmFp16TensorCore::Arguments args(
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K},
            1,
            {alpha, beta},
            A, B, C, C,
            (int64_t)M * K, (int64_t)K * N, (int64_t)M * N, (int64_t)M * N,
            K, N, N, N
        );
        auto status = gemm_op(args, nullptr, stream);
        return (status == cutlass::Status::kSuccess) ? 0 : -1;
    }
    else if (strcmp(symbol, "cutlass_gemm_tensorcore_bf16") == 0) {
        GemmBf16TensorCore gemm_op;
        GemmBf16TensorCore::Arguments args(
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K},
            1,
            {alpha, beta},
            A, B, C, C,
            (int64_t)M * K, (int64_t)K * N, (int64_t)M * N, (int64_t)M * N,
            K, N, N, N
        );
        auto status = gemm_op(args, nullptr, stream);
        return (status == cutlass::Status::kSuccess) ? 0 : -1;
    }
    else if (strcmp(symbol, "cutlass_gemm_simt_f32") == 0) {
        GemmF32Simt gemm_op;
        GemmF32Simt::Arguments args(
            {M, N, K},
            {(float*)A, K}, {(float*)B, N},
            {(float*)C, N}, {(float*)C, N},
            {alpha, beta}
        );
        auto status = gemm_op(args, nullptr, stream);
        return (status == cutlass::Status::kSuccess) ? 0 : -1;
    }
    else if (strcmp(symbol, "ada_gemm_k64_warp32_async_f32") == 0) {
        return pyc_ada_async_gemm_dispatch(M, N, K, A, B, C, alpha, beta, stream);
    }
    return -1;  /* unknown symbol */
}
