#ifndef PYC_CUDA_BACKEND_H
#define PYC_CUDA_BACKEND_H

/*
 * pyc/cuda_backend.h
 *
 * Public C-ABI for the PyC CUDA backend.
 *
 * This header is consumed by:
 *   - C callers (compiler_api.c)
 *   - Rust FFI (build.rs / bindgen → ffi/mod.rs)
 *   - Python via ctypes (python/pyc/runtime/cuda.py)
 *
 * The CUDA backend dispatches to CUTLASS kernels (selected by the
 * kernel registry) or falls back to a user-supplied CPU executor.
 */

#include "pyc/ir.h"
#include "pyc/kernel_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------------------------
 * Dispatch status codes
 * ---------------------------------------------------------------- */
typedef enum {
    PYC_CUDA_DISPATCH_OK       = 0,   /* ran successfully on GPU */
    PYC_CUDA_DISPATCH_FALLBACK = 1,   /* fell back to CPU executor */
    PYC_CUDA_DISPATCH_ERROR    = -1   /* unrecoverable error */
} pyc_cuda_dispatch_status;

/* ----------------------------------------------------------------
 * Execution trace (returned to caller for telemetry)
 * ---------------------------------------------------------------- */
typedef struct {
    char     kernel_symbol[128];   /* selected kernel symbol name */
    int      used_tensor_cores;    /* 1 if Tensor Core path was taken */
    double   kernel_time_ms;       /* GPU kernel wall time */
    double   h2d_transfer_ms;      /* host-to-device transfer time */
    double   d2h_transfer_ms;      /* device-to-host transfer time */
    size_t   peak_device_bytes;    /* peak device memory during dispatch */
    int      fallback_reason;      /* 0 = no fallback; >0 = reason code */
} pyc_cuda_dispatch_trace;

/* ----------------------------------------------------------------
 * CPU fallback function pointer type
 * Called when CUDA dispatch fails or is unavailable.
 * ---------------------------------------------------------------- */
typedef int (*pyc_cpu_executor_fn)(
    const pyc_ir_module* module,
    const pyc_tensor*    inputs,
    size_t               n_inputs,
    pyc_tensor*          outputs,
    size_t               n_outputs,
    void*                ctx
);

/* ----------------------------------------------------------------
 * Primary dispatch entry point
 *
 * Selects the best CUTLASS kernel via the kernel registry, executes
 * it on the GPU, and falls back to `cpu_executor` on failure.
 *
 * Returns: pyc_cuda_dispatch_status
 * ---------------------------------------------------------------- */
int pyc_cuda_dispatch(
    const pyc_ir_module*    module,
    const pyc_tensor*       inputs,
    size_t                  n_inputs,
    pyc_tensor*             outputs,
    size_t                  n_outputs,
    pyc_cpu_executor_fn     cpu_executor,   /* may be NULL */
    void*                   cpu_ctx,        /* passed to cpu_executor */
    pyc_cuda_dispatch_trace* trace          /* may be NULL */
);

/* ----------------------------------------------------------------
 * CUTLASS kernel dispatch entry points
 * Called internally by pyc_cuda_dispatch; exposed for direct use.
 * ---------------------------------------------------------------- */
int pyc_cutlass_gemm_dispatch(
    const char*  symbol,
    int M, int N, int K,
    const void*  A,
    const void*  B,
    void*        C,
    float        alpha,
    float        beta,
    void*        stream   /* cudaStream_t */
);

int pyc_cutlass_conv2d_dispatch(
    const char*  symbol,
    int N, int H, int W, int C,
    int K, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    const void*  input,
    const void*  filter,
    void*        output,
    void*        stream
);

int pyc_cutlass_attention_dispatch(
    const char*  symbol,
    int seq_len,
    int d_head,
    const void*  Q,
    const void*  K,
    const void*  V,
    void*        Out,
    float        scale,
    void*        stream
);

/* ----------------------------------------------------------------
 * CUTLASS kernel count query (for diagnostics)
 * ---------------------------------------------------------------- */
int pyc_cutlass_kernel_count(const char* op_key);

#ifdef __cplusplus
}
#endif
#endif /* PYC_CUDA_BACKEND_H */
