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

#include <stddef.h>
#include "pyc/compiler_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------------------------
 * Dispatch status codes
 * ---------------------------------------------------------------- */
typedef enum {
    PYC_CUDA_DISPATCH_OK       = 0,   /* ran successfully on GPU */
    PYC_CUDA_DISPATCH_FALLBACK = 1,   /* fell back to CPU executor */
    PYC_CUDA_DISPATCH_ERROR    = 2    /* unrecoverable error */
} pyc_cuda_dispatch_status;

/* ----------------------------------------------------------------
 * Constants
 * ---------------------------------------------------------------- */
#define PYC_CUDA_REASON_MAX 128

/* ----------------------------------------------------------------
 * Execution trace (returned to caller for telemetry).
 *
 * Field names match exactly what cuda_backend.c and compiler_api.c
 * write into this struct — do not rename without updating both .c files.
 * ---------------------------------------------------------------- */
typedef struct {
    int  cuda_requested;                /* 1 if CUDA dispatch was attempted */
    int  cuda_available;                /* 1 if a CUDA device was found */
    int  fallback_to_cpu;               /* 1 if execution fell back to CPU */
    int  graph_replayed;                /* 1 if CUDA graph replay handled the run */
    double copy_in_ms;                  /* host->device copy wall time */
    double kernel_ms;                   /* kernel launch/body wall time */
    double copy_out_ms;                 /* device->host copy wall time */
    double sync_ms;                     /* stream sync / graph launch wall time */
    char reason[PYC_CUDA_REASON_MAX];   /* human-readable status string */
} pyc_cuda_dispatch_trace;

/* ----------------------------------------------------------------
 * Initialise a trace struct to safe defaults.
 * Must be called before passing a trace to pyc_cuda_dispatch().
 * ---------------------------------------------------------------- */
void pyc_cuda_dispatch_trace_init(pyc_cuda_dispatch_trace* trace);

/* ----------------------------------------------------------------
 * CPU fallback function pointer type.
 * Called when CUDA dispatch fails or is unavailable.
 * ---------------------------------------------------------------- */
typedef int (*pyc_cpu_executor_fn)(
    const pyc_ir_module* module,
    const pyc_tensor*    inputs,
    size_t               input_count,
    pyc_tensor*          outputs,
    size_t               output_count,
    void*                executor_ctx
);

/* ----------------------------------------------------------------
 * Primary dispatch entry point.
 *
 * Selects the best available CUDA kernel, executes it, and falls
 * back to cpu_executor on failure.  cpu_executor may be NULL.
 *
 * Returns: pyc_cuda_dispatch_status
 * ---------------------------------------------------------------- */
pyc_cuda_dispatch_status pyc_cuda_dispatch(
    const pyc_ir_module*     module,
    const pyc_tensor*        inputs,
    size_t                   input_count,
    pyc_tensor*              outputs,
    size_t                   output_count,
    pyc_cpu_executor_fn      cpu_executor,
    void*                    executor_ctx,
    pyc_cuda_dispatch_trace* trace
);

/* ----------------------------------------------------------------
 * CUTLASS kernel dispatch entry points.
 *
 * Called internally by pyc_cuda_dispatch(); exposed here so that
 * the Rust vortex_core runtime can invoke them directly via FFI.
 *
 * The stream argument is typed as void* to avoid pulling in
 * cuda_runtime.h in this header; callers cast to cudaStream_t.
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
    int N, int H, int W, int C_in,
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
