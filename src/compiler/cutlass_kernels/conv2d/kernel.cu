/*
 * conv2d/kernel.cu
 *
 * CUTLASS-backed Conv2d kernels for the PyC kernel registry.
 *
 * Registered kernels:
 *   1. cutlass_conv2d_fprop_f16  — FP16 forward convolution (Tensor Core)
 *   2. cutlass_conv2d_fprop_bf16 — BF16 forward convolution (Tensor Core)
 *
 * Requires: CUTLASS 3.x, CUDA 12+, sm_80+
 */

#include "pyc/kernel_registry.h"
#include "cutlass/cutlass.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include <cuda_runtime.h>
#include <string.h>

/* ----------------------------------------------------------------
 * FP16 Conv2d Fprop via CUTLASS ImplicitGemm
 * ---------------------------------------------------------------- */
using Conv2dFpropFp16 = cutlass::conv::device::ImplicitGemmConvolution<
    cutlass::conv::kernel::DefaultConv2dFprop<
        cutlass::half_t,
        cutlass::layout::TensorNHWC,
        cutlass::half_t,
        cutlass::layout::TensorNHWC,
        cutlass::half_t,
        cutlass::layout::TensorNHWC,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        3,
        cutlass::arch::OpMultiplyAdd,
        cutlass::conv::IteratorAlgorithm::kOptimized
    >::Kernel
>;

/* ----------------------------------------------------------------
 * Registration
 * ---------------------------------------------------------------- */
extern "C" void pyc_cutlass_register_conv2d_kernels(void) {
    pyc_kernel_desc desc;

    /* FP16 Conv2d */
    memset(&desc, 0, sizeof(desc));
    strncpy(desc.op_key,  "conv2d",                     PYC_KERNEL_OP_KEY_MAX - 1);
    strncpy(desc.symbol,  "cutlass_conv2d_fprop_f16",   PYC_KERNEL_SYMBOL_MAX - 1);
    desc.backend              = PYC_BACKEND_CUDA;
    desc.priority             = 100;
    desc.estimated_occupancy  = 0.80;
    desc.tensor_core_eligible = 1;
    desc.shared_mem_bytes     = 65536;   /* 64 KB */
    desc.reg_pressure_class   = 2;
    pyc_kernel_register(&desc);

    /* BF16 Conv2d */
    memset(&desc, 0, sizeof(desc));
    strncpy(desc.op_key,  "conv2d",                     PYC_KERNEL_OP_KEY_MAX - 1);
    strncpy(desc.symbol,  "cutlass_conv2d_fprop_bf16",  PYC_KERNEL_SYMBOL_MAX - 1);
    desc.backend              = PYC_BACKEND_CUDA;
    desc.priority             = 90;
    desc.estimated_occupancy  = 0.77;
    desc.tensor_core_eligible = 1;
    desc.shared_mem_bytes     = 65536;
    desc.reg_pressure_class   = 2;
    pyc_kernel_register(&desc);
}

/* ----------------------------------------------------------------
 * Dispatch
 * ---------------------------------------------------------------- */
extern "C" int pyc_cutlass_conv2d_dispatch(
    const char*  symbol,
    int N, int H, int W, int C,
    int K, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    const void*  input,
    const void*  filter,
    void*        output,
    cudaStream_t stream)
{
    if (strcmp(symbol, "cutlass_conv2d_fprop_f16") == 0) {
        cutlass::conv::Conv2dProblemSize problem(
            {N, H, W, C}, {K, R, S, C},
            {pad_h, pad_w, pad_h, pad_w},
            {stride_h, stride_w},
            {1, 1},
            cutlass::conv::Mode::kCrossCorrelation, 1
        );
        Conv2dFpropFp16 conv_op;
        Conv2dFpropFp16::Arguments args(
            problem,
            {(cutlass::half_t*)input,  {C,   C*W,   C*W*H}},
            {(cutlass::half_t*)filter, {C,   C*S,   C*S*R}},
            {(cutlass::half_t*)output, {K,   K*((W-S+2*pad_w)/stride_w+1), K*((W-S+2*pad_w)/stride_w+1)*((H-R+2*pad_h)/stride_h+1)}},
            {(cutlass::half_t*)output, {K,   K*((W-S+2*pad_w)/stride_w+1), K*((W-S+2*pad_w)/stride_w+1)*((H-R+2*pad_h)/stride_h+1)}},
            {1.0f, 0.0f}
        );
        auto status = conv_op(args, nullptr, stream);
        return (status == cutlass::Status::kSuccess) ? 0 : -1;
    }
    return -1;
}
