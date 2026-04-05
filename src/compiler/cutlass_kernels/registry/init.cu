/*
 * registry/init.cu
 *
 * Auto-registration of all CUTLASS kernels into the PyC kernel registry.
 *
 * This translation unit uses a CUDA/C++ constructor attribute to call
 * all registration functions at shared library load time, so the kernels
 * are available in the registry before any user code runs.
 *
 * The PyC kernel registry is then consulted by:
 *   - The Rust vortex_core runtime (via FFI) during pipeline execution
 *   - The Python SDK via the pyc.compiler.policy module
 */

#include "pyc/kernel_registry.h"
#include <cstdlib>
#include <stdio.h>

/* Forward declarations of per-file registration functions */
extern "C" void pyc_cutlass_register_gemm_kernels(void);
#if defined(PYC_CUTLASS_FULL_KERNELS)
extern "C" void pyc_cutlass_register_conv2d_kernels(void);
extern "C" void pyc_cutlass_register_attention_kernels(void);
#endif

/*
 * __attribute__((constructor)) ensures this runs when libpyc_cutlass_kernels.so
 * is loaded, before main() or any Python import.
 *
 * On Windows, use DllMain with DLL_PROCESS_ATTACH instead.
 */
extern "C" void pyc_cutlass_registry_init(void) {
    static int initialized = 0;

    if (initialized) {
        return;
    }
    initialized = 1;

    /* Initialize the registry if not already done */
    pyc_kernel_registry_reset();

    /* Register all CUTLASS kernel families */
    pyc_cutlass_register_gemm_kernels();
#if defined(PYC_CUTLASS_FULL_KERNELS)
    pyc_cutlass_register_conv2d_kernels();
    pyc_cutlass_register_attention_kernels();
#endif

    {
        const char* promoted_gemm = std::getenv("PYC_PROMOTED_GEMM_SYMBOL");
        if (promoted_gemm && promoted_gemm[0] != '\0') {
            if (pyc_kernel_promote_symbol("matmul", PYC_BACKEND_CUDA, promoted_gemm) == 0) {
#ifdef PYC_CUTLASS_VERBOSE_INIT
                fprintf(stderr, "[PyC CUTLASS] Promoted GEMM symbol: %s\n", promoted_gemm);
#endif
            } else {
#ifdef PYC_CUTLASS_VERBOSE_INIT
                fprintf(stderr, "[PyC CUTLASS] Ignored unknown promoted GEMM symbol: %s\n", promoted_gemm);
#endif
            }
        }
    }

#ifdef PYC_CUTLASS_VERBOSE_INIT
#if defined(PYC_CUTLASS_FULL_KERNELS)
    fprintf(stderr,
        "[PyC CUTLASS] Registered GEMM (FP16/BF16/FP32), "
        "Conv2d (FP16/BF16), Attention (FP16/BF16) kernels.\n");
#else
    fprintf(stderr,
        "[PyC CUTLASS] Registered minimal CUDA GEMM kernel registry.\n");
#endif
#endif
}

__attribute__((constructor))
static void pyc_cutlass_auto_register(void) {
    pyc_cutlass_registry_init();
}

/*
 * Public query: how many CUTLASS kernels are registered for a given op?
 * Useful for diagnostics and tests.
 */
extern "C" int pyc_cutlass_kernel_count(const char* op_key) {
    pyc_kernel_desc descs[32];
    return (int)pyc_kernel_collect(op_key, PYC_BACKEND_CUDA, descs, 32);
}
