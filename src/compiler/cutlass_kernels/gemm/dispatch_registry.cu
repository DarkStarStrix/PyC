#include "pyc/kernel_registry.h"

#include <cuda_runtime.h>
#include <string.h>

extern "C" void pyc_register_ada_async_gemm_kernel(void);
extern "C" int pyc_ada_async_gemm_dispatch(
    int M,
    int N,
    int K,
    const void* A,
    const void* B,
    void* C,
    float alpha,
    float beta,
    cudaStream_t stream);

extern "C" void pyc_cutlass_register_gemm_kernels(void) {
    pyc_register_ada_async_gemm_kernel();
}

extern "C" int pyc_cutlass_gemm_dispatch(
    const char* symbol,
    int M,
    int N,
    int K,
    const void* A,
    const void* B,
    void* C,
    float alpha,
    float beta,
    cudaStream_t stream) {
    if (strcmp(symbol, "ada_gemm_k64_warp32_async_f32") == 0) {
        return pyc_ada_async_gemm_dispatch(M, N, K, A, B, C, alpha, beta, stream);
    }
    return -1;
}
