#ifndef PYC_CUDA_BACKEND_H
#define PYC_CUDA_BACKEND_H

#include <stddef.h>

#include "pyc/compiler_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PYC_CUDA_REASON_MAX 128

typedef enum {
    PYC_CUDA_DISPATCH_OK = 0,
    PYC_CUDA_DISPATCH_FALLBACK = 1,
    PYC_CUDA_DISPATCH_ERROR = 2
} pyc_cuda_dispatch_status;

typedef struct {
    int cuda_requested;
    int cuda_available;
    int fallback_to_cpu;
    char reason[PYC_CUDA_REASON_MAX];
} pyc_cuda_dispatch_trace;

typedef int (*pyc_cpu_executor_fn)(
    const pyc_ir_module* module,
    const pyc_tensor* inputs,
    size_t input_count,
    pyc_tensor* outputs,
    size_t output_count,
    void* executor_ctx);

void pyc_cuda_dispatch_trace_init(pyc_cuda_dispatch_trace* trace);

pyc_cuda_dispatch_status pyc_cuda_dispatch(
    const pyc_ir_module* module,
    const pyc_tensor* inputs,
    size_t input_count,
    pyc_tensor* outputs,
    size_t output_count,
    pyc_cpu_executor_fn cpu_executor,
    void* executor_ctx,
    pyc_cuda_dispatch_trace* trace);

#ifdef __cplusplus
}
#endif

#endif
