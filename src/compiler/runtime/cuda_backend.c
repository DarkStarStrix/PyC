#include "pyc/cuda_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "pyc/pyc_mutex.h"

/* Serializes pyc_cuda_dispatch(): the engine keeps a single process-global GPU
 * workspace (device buffers, cuBLAS handles, captured graph), so concurrent
 * dispatches must not run at once. */
static pyc_mutex g_cuda_dispatch_mutex = PYC_MUTEX_INIT;

#if defined(PYC_HAVE_CUDA_RUNTIME)
#include <cuda_runtime_api.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#endif

#if !defined(PYC_HAVE_CUTLASS_KERNELS)
void pyc_cutlass_registry_init(void) {
}

int pyc_cutlass_gemm_dispatch(
    const char* symbol,
    int M,
    int N,
    int K,
    const void* A,
    const void* B,
    void* C,
    float alpha,
    float beta,
    void* stream) {
    (void)symbol;
    (void)M;
    (void)N;
    (void)K;
    (void)A;
    (void)B;
    (void)C;
    (void)alpha;
    (void)beta;
    (void)stream;
    return -1;
}

int pyc_cutlass_conv2d_dispatch(
    const char* symbol,
    int N,
    int H,
    int W,
    int C_in,
    int K,
    int R,
    int S,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    const void* input,
    const void* filter,
    void* output,
    void* stream) {
    (void)symbol;
    (void)N;
    (void)H;
    (void)W;
    (void)C_in;
    (void)K;
    (void)R;
    (void)S;
    (void)pad_h;
    (void)pad_w;
    (void)stride_h;
    (void)stride_w;
    (void)input;
    (void)filter;
    (void)output;
    (void)stream;
    return -1;
}

int pyc_cutlass_attention_dispatch(
    const char* symbol,
    int seq_len,
    int d_head,
    const void* Q,
    const void* K,
    const void* V,
    void* Out,
    float scale,
    void* stream) {
    (void)symbol;
    (void)seq_len;
    (void)d_head;
    (void)Q;
    (void)K;
    (void)V;
    (void)Out;
    (void)scale;
    (void)stream;
    return -1;
}

int pyc_cutlass_kernel_count(const char* op_key) {
    (void)op_key;
    return 0;
}
#else
void pyc_cutlass_registry_init(void);
#endif

#if defined(PYC_HAVE_CUDA_RUNTIME)
typedef struct {
    int valid;
    int tf32_enabled;
    int m;
    int n;
    int k;
    size_t workspace_bytes;
    cublasLtMatmulDesc_t op_desc;
    cublasLtMatrixLayout_t a_layout;
    cublasLtMatrixLayout_t b_layout;
    cublasLtMatrixLayout_t c_layout;
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulHeuristicResult_t heuristic;
} pyc_cublaslt_fp32_plan;

typedef struct {
    int valid;
    int device;
    pyc_kernel_hardware_family family;
} pyc_cuda_hardware_family_cache_entry;

typedef struct {
    void* dev_a;
    void* dev_b;
    void* dev_c;
    void* graph_host_a_stage;
    void* graph_host_b_stage;
    void* graph_host_out_stage;
    size_t a_bytes;
    size_t b_bytes;
    size_t c_bytes;
    size_t graph_host_a_stage_bytes;
    size_t graph_host_b_stage_bytes;
    size_t graph_host_out_stage_bytes;
    cublasHandle_t handle;
    cublasLtHandle_t lt_handle;
    cudaStream_t stream;
    void* lt_workspace;
    size_t lt_workspace_bytes;
    pyc_cublaslt_fp32_plan lt_fp32_plan;
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    int graph_valid;
    int graph_uses_promoted_gemm;
    int graph_lhs_copy_required;
    size_t graph_m;
    size_t graph_k;
    size_t graph_n;
    pyc_dtype graph_dtype;
    const void* graph_host_a_last_ptr;
    const void* graph_host_a_ptr;
    const void* graph_host_b_ptr;
    const void* graph_host_out_ptr;
    int graph_rhs_copy_required;
    char graph_promoted_symbol[PYC_KERNEL_SYMBOL_MAX];
    const void* host_a_last_ptr;
    size_t host_a_last_bytes;
    const void* host_b_last_ptr;
    size_t host_b_last_bytes;
    int lhs_uploaded;
    int rhs_uploaded;
    pyc_dtype active_dtype;
} pyc_cuda_workspace;

static pyc_cuda_workspace g_cuda_workspace;
static pyc_cuda_hardware_family_cache_entry g_cuda_hardware_family_cache[16];
/* Guards the hardware-family memoization cache. */
static pyc_mutex g_cuda_hw_family_mutex = PYC_MUTEX_INIT;

static int env_true(const char* name);
static int env_default_true(const char* name);
static size_t pyc_cuda_dtype_size_bytes(pyc_dtype dtype);
static int native_cuda_supports_dtype(pyc_dtype dtype, pyc_ir_op_kind kind);

static double wall_ms_now(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec * 1000.0) + ((double)ts.tv_nsec / 1000000.0);
}
static cublasStatus_t configure_cublas_math_mode(cublasHandle_t handle);
static const char* cublas_status_name(pyc_dtype dtype, cublasStatus_t status);
static void pyc_cuda_lt_plan_reset(pyc_cublaslt_fp32_plan* plan);
static cublasStatus_t run_cublas_fp32_gemm(
    cublasHandle_t handle,
    int m,
    int n,
    int k,
    const float* dev_a,
    const float* dev_b,
    float* dev_c);
static cublasStatus_t run_cublas_half_precision_gemm(
    cublasHandle_t handle,
    pyc_dtype dtype,
    int m,
    int n,
    int k,
    const void* dev_a,
    const void* dev_b,
    void* dev_c);
static cublasStatus_t run_cublaslt_fp32_gemm(
    cublasLtHandle_t lt_handle,
    cudaStream_t stream,
    void* workspace,
    size_t workspace_bytes,
    int m,
    int n,
    int k,
    const float* dev_a,
    const float* dev_b,
    float* dev_c);
static cublasStatus_t run_best_gemm(
    pyc_dtype dtype,
    int m,
    int n,
    int k,
    const void* dev_a,
    const void* dev_b,
    void* dev_c);

pyc_kernel_hardware_family pyc_cuda_current_hardware_family(void) {
    const char* override = getenv("PYC_CUDA_FORCE_HARDWARE_FAMILY");
    if (override && override[0] != '\0') {
        if (strcmp(override, "hopper") == 0) {
            return PYC_KERNEL_HARDWARE_HOPPER;
        }
        if (strcmp(override, "ada") == 0) {
            return PYC_KERNEL_HARDWARE_ADA;
        }
        if (strcmp(override, "generic") == 0) {
            return PYC_KERNEL_HARDWARE_GENERIC;
        }
    }
#if defined(PYC_HAVE_CUDA_RUNTIME)
    {
        size_t i;
        const size_t cache_len =
            sizeof(g_cuda_hardware_family_cache) / sizeof(g_cuda_hardware_family_cache[0]);
        int device = 0;
        struct cudaDeviceProp prop;
        pyc_kernel_hardware_family family = PYC_KERNEL_HARDWARE_GENERIC;
        int resolved = 0;

        if (cudaGetDevice(&device) != cudaSuccess) {
            return PYC_KERNEL_HARDWARE_GENERIC;
        }

        pyc_mutex_lock(&g_cuda_hw_family_mutex);
        for (i = 0; i < cache_len; ++i) {
            if (g_cuda_hardware_family_cache[i].valid &&
                g_cuda_hardware_family_cache[i].device == device) {
                family = g_cuda_hardware_family_cache[i].family;
                resolved = 1;
                break;
            }
        }
        pyc_mutex_unlock(&g_cuda_hw_family_mutex);
        if (resolved) {
            return family;
        }

        if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
            return PYC_KERNEL_HARDWARE_GENERIC;
        }
        if (prop.major >= 9) {
            family = PYC_KERNEL_HARDWARE_HOPPER;
        } else if (prop.major == 8 && prop.minor >= 9) {
            family = PYC_KERNEL_HARDWARE_ADA;
        } else {
            family = PYC_KERNEL_HARDWARE_GENERIC;
        }

        pyc_mutex_lock(&g_cuda_hw_family_mutex);
        for (i = 0; i < cache_len; ++i) {
            if (!g_cuda_hardware_family_cache[i].valid) {
                g_cuda_hardware_family_cache[i].valid = 1;
                g_cuda_hardware_family_cache[i].device = device;
                g_cuda_hardware_family_cache[i].family = family;
                break;
            }
        }
        pyc_mutex_unlock(&g_cuda_hw_family_mutex);
        return family;
    }
#endif
    return PYC_KERNEL_HARDWARE_GENERIC;
}

static void pyc_cuda_lt_plan_reset(pyc_cublaslt_fp32_plan* plan) {
    if (!plan) {
        return;
    }
    if (plan->pref) {
        (void)cublasLtMatmulPreferenceDestroy(plan->pref);
        plan->pref = NULL;
    }
    if (plan->c_layout) {
        (void)cublasLtMatrixLayoutDestroy(plan->c_layout);
        plan->c_layout = NULL;
    }
    if (plan->b_layout) {
        (void)cublasLtMatrixLayoutDestroy(plan->b_layout);
        plan->b_layout = NULL;
    }
    if (plan->a_layout) {
        (void)cublasLtMatrixLayoutDestroy(plan->a_layout);
        plan->a_layout = NULL;
    }
    if (plan->op_desc) {
        (void)cublasLtMatmulDescDestroy(plan->op_desc);
        plan->op_desc = NULL;
    }
    memset(plan, 0, sizeof(*plan));
}

static void pyc_cuda_graph_reset(void) {
    if (g_cuda_workspace.graph_exec) {
        (void)cudaGraphExecDestroy(g_cuda_workspace.graph_exec);
        g_cuda_workspace.graph_exec = NULL;
    }
    if (g_cuda_workspace.graph) {
        (void)cudaGraphDestroy(g_cuda_workspace.graph);
        g_cuda_workspace.graph = NULL;
    }
    g_cuda_workspace.graph_valid = 0;
    g_cuda_workspace.graph_uses_promoted_gemm = 0;
    g_cuda_workspace.graph_lhs_copy_required = 0;
    g_cuda_workspace.graph_m = 0;
    g_cuda_workspace.graph_k = 0;
    g_cuda_workspace.graph_n = 0;
    g_cuda_workspace.graph_dtype = PYC_DTYPE_UNKNOWN;
    g_cuda_workspace.graph_host_a_last_ptr = NULL;
    g_cuda_workspace.graph_host_a_ptr = NULL;
    g_cuda_workspace.graph_host_b_ptr = NULL;
    g_cuda_workspace.graph_host_out_ptr = NULL;
    g_cuda_workspace.graph_rhs_copy_required = 0;
    g_cuda_workspace.graph_promoted_symbol[0] = '\0';
}

static void pyc_cuda_workspace_release(void) {
    pyc_cuda_graph_reset();
    pyc_cuda_lt_plan_reset(&g_cuda_workspace.lt_fp32_plan);
    if (g_cuda_workspace.dev_a) {
        (void)cudaFree(g_cuda_workspace.dev_a);
        g_cuda_workspace.dev_a = NULL;
    }
    if (g_cuda_workspace.dev_b) {
        (void)cudaFree(g_cuda_workspace.dev_b);
        g_cuda_workspace.dev_b = NULL;
    }
    if (g_cuda_workspace.dev_c) {
        (void)cudaFree(g_cuda_workspace.dev_c);
        g_cuda_workspace.dev_c = NULL;
    }
    if (g_cuda_workspace.graph_host_a_stage) {
        (void)cudaFreeHost(g_cuda_workspace.graph_host_a_stage);
        g_cuda_workspace.graph_host_a_stage = NULL;
    }
    if (g_cuda_workspace.graph_host_b_stage) {
        (void)cudaFreeHost(g_cuda_workspace.graph_host_b_stage);
        g_cuda_workspace.graph_host_b_stage = NULL;
    }
    if (g_cuda_workspace.graph_host_out_stage) {
        (void)cudaFreeHost(g_cuda_workspace.graph_host_out_stage);
        g_cuda_workspace.graph_host_out_stage = NULL;
    }
    if (g_cuda_workspace.handle) {
        (void)cublasDestroy(g_cuda_workspace.handle);
        g_cuda_workspace.handle = NULL;
    }
    if (g_cuda_workspace.lt_handle) {
        (void)cublasLtDestroy(g_cuda_workspace.lt_handle);
        g_cuda_workspace.lt_handle = NULL;
    }
    if (g_cuda_workspace.stream) {
        (void)cudaStreamDestroy(g_cuda_workspace.stream);
        g_cuda_workspace.stream = NULL;
    }
    if (g_cuda_workspace.lt_workspace) {
        (void)cudaFree(g_cuda_workspace.lt_workspace);
        g_cuda_workspace.lt_workspace = NULL;
    }
    g_cuda_workspace.a_bytes = 0;
    g_cuda_workspace.b_bytes = 0;
    g_cuda_workspace.c_bytes = 0;
    g_cuda_workspace.graph_host_a_stage_bytes = 0;
    g_cuda_workspace.graph_host_b_stage_bytes = 0;
    g_cuda_workspace.graph_host_out_stage_bytes = 0;
    g_cuda_workspace.lt_workspace_bytes = 0;
    g_cuda_workspace.host_a_last_ptr = NULL;
    g_cuda_workspace.host_a_last_bytes = 0;
    g_cuda_workspace.host_b_last_ptr = NULL;
    g_cuda_workspace.host_b_last_bytes = 0;
    g_cuda_workspace.lhs_uploaded = 0;
    g_cuda_workspace.rhs_uploaded = 0;
    g_cuda_workspace.active_dtype = PYC_DTYPE_UNKNOWN;
}

static int pyc_cuda_workspace_ensure(
    size_t a_bytes,
    size_t b_bytes,
    size_t c_bytes,
    pyc_dtype dtype) {
    cudaError_t cuda_status;
    cublasStatus_t cublas_status;
    int b_reallocated = 0;
    int graph_invalidate = 0;
    size_t requested_lt_workspace = 0;

    if (env_default_true("PYC_CUDA_ENABLE_CUBLASLT") && dtype == PYC_DTYPE_F32) {
        const char* value = getenv("PYC_CUDA_LT_WORKSPACE_BYTES");
        requested_lt_workspace = value && value[0] != '\0' ? (size_t)strtoull(value, NULL, 10) : (size_t)(32u * 1024u * 1024u);
    }

    if (g_cuda_workspace.active_dtype != PYC_DTYPE_UNKNOWN &&
        g_cuda_workspace.active_dtype != dtype) {
        pyc_cuda_graph_reset();
        pyc_cuda_lt_plan_reset(&g_cuda_workspace.lt_fp32_plan);
        g_cuda_workspace.host_a_last_ptr = NULL;
        g_cuda_workspace.host_a_last_bytes = 0;
        g_cuda_workspace.host_b_last_ptr = NULL;
        g_cuda_workspace.host_b_last_bytes = 0;
        g_cuda_workspace.lhs_uploaded = 0;
        g_cuda_workspace.rhs_uploaded = 0;
    }
    g_cuda_workspace.active_dtype = dtype;

    if (!g_cuda_workspace.stream) {
        cuda_status = cudaStreamCreateWithFlags(&g_cuda_workspace.stream, cudaStreamNonBlocking);
        if (cuda_status != cudaSuccess) {
            pyc_cuda_workspace_release();
            return -1;
        }
    }

    if (!g_cuda_workspace.handle) {
        cublas_status = cublasCreate(&g_cuda_workspace.handle);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            pyc_cuda_workspace_release();
            return -1;
        }
    }
    if (!g_cuda_workspace.lt_handle) {
        cublasStatus_t lt_status = cublasLtCreate(&g_cuda_workspace.lt_handle);
        if (lt_status != CUBLAS_STATUS_SUCCESS) {
            pyc_cuda_workspace_release();
            return -1;
        }
    }
    cublas_status = cublasSetStream(g_cuda_workspace.handle, g_cuda_workspace.stream);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        pyc_cuda_workspace_release();
        return -1;
    }
    cublas_status = configure_cublas_math_mode(g_cuda_workspace.handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        pyc_cuda_workspace_release();
        return -1;
    }

    if (a_bytes > g_cuda_workspace.a_bytes) {
        if (g_cuda_workspace.dev_a) {
            (void)cudaFree(g_cuda_workspace.dev_a);
            g_cuda_workspace.dev_a = NULL;
            g_cuda_workspace.a_bytes = 0;
        }
        cuda_status = cudaMalloc((void**)&g_cuda_workspace.dev_a, a_bytes);
        if (cuda_status != cudaSuccess) {
            pyc_cuda_workspace_release();
            return -1;
        }
        g_cuda_workspace.a_bytes = a_bytes;
        g_cuda_workspace.host_a_last_ptr = NULL;
        g_cuda_workspace.host_a_last_bytes = 0;
        g_cuda_workspace.lhs_uploaded = 0;
        graph_invalidate = 1;
    }
    if (b_bytes > g_cuda_workspace.b_bytes) {
        if (g_cuda_workspace.dev_b) {
            (void)cudaFree(g_cuda_workspace.dev_b);
            g_cuda_workspace.dev_b = NULL;
            g_cuda_workspace.b_bytes = 0;
        }
        cuda_status = cudaMalloc((void**)&g_cuda_workspace.dev_b, b_bytes);
        if (cuda_status != cudaSuccess) {
            pyc_cuda_workspace_release();
            return -1;
        }
        g_cuda_workspace.b_bytes = b_bytes;
        b_reallocated = 1;
        graph_invalidate = 1;
    }
    if (c_bytes > g_cuda_workspace.c_bytes) {
        if (g_cuda_workspace.dev_c) {
            (void)cudaFree(g_cuda_workspace.dev_c);
            g_cuda_workspace.dev_c = NULL;
            g_cuda_workspace.c_bytes = 0;
        }
        cuda_status = cudaMalloc((void**)&g_cuda_workspace.dev_c, c_bytes);
        if (cuda_status != cudaSuccess) {
            pyc_cuda_workspace_release();
            return -1;
        }
        g_cuda_workspace.c_bytes = c_bytes;
        graph_invalidate = 1;
    }
    if (b_reallocated) {
        g_cuda_workspace.host_b_last_ptr = NULL;
        g_cuda_workspace.host_b_last_bytes = 0;
        g_cuda_workspace.rhs_uploaded = 0;
    }
    if (graph_invalidate) {
        pyc_cuda_graph_reset();
        pyc_cuda_lt_plan_reset(&g_cuda_workspace.lt_fp32_plan);
    }
    if (requested_lt_workspace > g_cuda_workspace.lt_workspace_bytes) {
        if (g_cuda_workspace.lt_workspace) {
            (void)cudaFree(g_cuda_workspace.lt_workspace);
            g_cuda_workspace.lt_workspace = NULL;
            g_cuda_workspace.lt_workspace_bytes = 0;
        }
        pyc_cuda_lt_plan_reset(&g_cuda_workspace.lt_fp32_plan);
        if (requested_lt_workspace > 0) {
            cuda_status = cudaMalloc(&g_cuda_workspace.lt_workspace, requested_lt_workspace);
            if (cuda_status != cudaSuccess) {
                g_cuda_workspace.lt_workspace = NULL;
                g_cuda_workspace.lt_workspace_bytes = 0;
            } else {
                g_cuda_workspace.lt_workspace_bytes = requested_lt_workspace;
            }
        }
    }

    return 0;
}

static int pyc_cuda_workspace_ensure_host_stage(
    void** stage_ptr,
    size_t* stage_bytes,
    size_t required_bytes) {
    cudaError_t cuda_status;

    if (!stage_ptr || !stage_bytes) {
        return -1;
    }
    if (required_bytes <= *stage_bytes) {
        return 0;
    }
    if (*stage_ptr) {
        (void)cudaFreeHost(*stage_ptr);
        *stage_ptr = NULL;
        *stage_bytes = 0;
    }
    if (required_bytes == 0) {
        return 0;
    }
    cuda_status = cudaMallocHost((void**)stage_ptr, required_bytes);
    if (cuda_status != cudaSuccess) {
        return -1;
    }
    *stage_bytes = required_bytes;
    return 0;
}

static int pyc_cuda_workspace_prepare_graph_stages(
    size_t a_bytes,
    size_t b_bytes,
    size_t c_bytes,
    int lhs_copy_required,
    int rhs_copy_required,
    int output_copy_required) {
    if (lhs_copy_required &&
        pyc_cuda_workspace_ensure_host_stage(
            &g_cuda_workspace.graph_host_a_stage,
            &g_cuda_workspace.graph_host_a_stage_bytes,
            a_bytes) != 0) {
        return -1;
    }
    if (rhs_copy_required &&
        pyc_cuda_workspace_ensure_host_stage(
            &g_cuda_workspace.graph_host_b_stage,
            &g_cuda_workspace.graph_host_b_stage_bytes,
            b_bytes) != 0) {
        return -1;
    }
    if (output_copy_required &&
        pyc_cuda_workspace_ensure_host_stage(
            &g_cuda_workspace.graph_host_out_stage,
            &g_cuda_workspace.graph_host_out_stage_bytes,
            c_bytes) != 0) {
        return -1;
    }
    return 0;
}

static void pyc_cuda_stage_graph_inputs(
    const void* host_a,
    const void* host_b,
    size_t a_bytes,
    size_t b_bytes,
    int lhs_copy_required,
    int rhs_copy_required,
    pyc_cuda_dispatch_trace* trace) {
    double t0;

    if (lhs_copy_required && host_a && g_cuda_workspace.graph_host_a_stage && a_bytes > 0) {
        t0 = wall_ms_now();
        memcpy(g_cuda_workspace.graph_host_a_stage, host_a, a_bytes);
        if (trace) {
            trace->copy_in_ms += wall_ms_now() - t0;
        }
    }
    if (rhs_copy_required && host_b && g_cuda_workspace.graph_host_b_stage && b_bytes > 0) {
        t0 = wall_ms_now();
        memcpy(g_cuda_workspace.graph_host_b_stage, host_b, b_bytes);
        if (trace) {
            trace->copy_in_ms += wall_ms_now() - t0;
        }
    }
}

static void pyc_cuda_flush_graph_output(
    void* host_out,
    size_t c_bytes,
    pyc_cuda_dispatch_trace* trace) {
    double t0;

    if (!host_out || !g_cuda_workspace.graph_host_out_stage || c_bytes == 0) {
        return;
    }
    t0 = wall_ms_now();
    memcpy(host_out, g_cuda_workspace.graph_host_out_stage, c_bytes);
    if (trace) {
        trace->copy_out_ms += wall_ms_now() - t0;
    }
}
#endif

static int env_true(const char* name) {
    const char* value = getenv(name);
    if (!value) {
        return 0;
    }
    return strcmp(value, "1") == 0 || strcmp(value, "true") == 0 || strcmp(value, "TRUE") == 0;
}

static int env_default_true(const char* name) {
    const char* value = getenv(name);
    if (!value || value[0] == '\0') {
        return 1;
    }
    return !(strcmp(value, "0") == 0 ||
             strcmp(value, "false") == 0 ||
             strcmp(value, "FALSE") == 0);
}

static int select_cutlass_matmul_symbol(
    pyc_dtype dtype,
    char* out_symbol,
    size_t out_symbol_size) {
    const char* env_symbol = NULL;
    const char* default_symbol = NULL;

    if (!out_symbol || out_symbol_size == 0) {
        return -1;
    }
    out_symbol[0] = '\0';

    switch (dtype) {
        case PYC_DTYPE_BF16:
            env_symbol = getenv("PYC_CUDA_BF16_GEMM_SYMBOL");
            default_symbol = "cutlass_gemm_tensorcore_bf16";
            break;
        default:
            return -1;
    }

    if (env_symbol && env_symbol[0] != '\0') {
        strncpy(out_symbol, env_symbol, out_symbol_size - 1);
    } else if (default_symbol) {
        strncpy(out_symbol, default_symbol, out_symbol_size - 1);
    }
    out_symbol[out_symbol_size - 1] = '\0';
    return out_symbol[0] != '\0' ? 0 : -1;
}

static size_t pyc_cuda_dtype_size_bytes(pyc_dtype dtype) {
    switch (dtype) {
        case PYC_DTYPE_F16:
        case PYC_DTYPE_BF16:
            return 2;
        case PYC_DTYPE_F32:
            return 4;
        default:
            return 0;
    }
}

static int native_cuda_supports_dtype(pyc_dtype dtype, pyc_ir_op_kind kind) {
    switch (dtype) {
        case PYC_DTYPE_F32:
            return 1;
        case PYC_DTYPE_F16:
        case PYC_DTYPE_BF16:
            return kind == PYC_IR_OP_INPUT ||
                   kind == PYC_IR_OP_MATMUL ||
                   kind == PYC_IR_OP_OUTPUT;
        default:
            return 0;
    }
}

#if defined(PYC_HAVE_CUDA_RUNTIME)
static cublasStatus_t configure_cublas_math_mode(cublasHandle_t handle) {
#if defined(CUBLAS_TF32_TENSOR_OP_MATH)
    cublasMath_t math_mode = env_default_true("PYC_CUDA_ALLOW_TF32") ?
        CUBLAS_TF32_TENSOR_OP_MATH :
        CUBLAS_DEFAULT_MATH;
    return cublasSetMathMode(handle, math_mode);
#else
    (void)handle;
    return CUBLAS_STATUS_SUCCESS;
#endif
}

static const char* cublas_status_name(pyc_dtype dtype, cublasStatus_t status) {
    (void)dtype;
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "success";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "not_initialized";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "alloc_failed";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "invalid_value";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "arch_mismatch";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "mapping_error";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "execution_failed";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "internal_error";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "not_supported";
#if defined(CUBLAS_STATUS_LICENSE_ERROR)
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "license_error";
#endif
        default:
            return "unknown";
    }
}

static cublasStatus_t run_cublas_fp32_gemm(
    cublasHandle_t handle,
    int m,
    int n,
    int k,
    const float* dev_a,
    const float* dev_b,
    float* dev_c) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
#if defined(CUBLAS_COMPUTE_32F_FAST_TF32)
    cublasComputeType_t compute_type = env_default_true("PYC_CUDA_ALLOW_TF32") ?
        CUBLAS_COMPUTE_32F_FAST_TF32 :
        CUBLAS_COMPUTE_32F;
    return cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        dev_b,
        CUDA_R_32F,
        n,
        dev_a,
        CUDA_R_32F,
        k,
        &beta,
        dev_c,
        CUDA_R_32F,
        n,
        compute_type,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#else
    return cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        dev_b,
        n,
        dev_a,
        k,
        &beta,
        dev_c,
        n);
#endif
}

static cublasStatus_t run_best_fp32_gemm(
    int m,
    int n,
    int k,
    const float* dev_a,
    const float* dev_b,
    float* dev_c) {
    if (env_default_true("PYC_CUDA_ENABLE_CUBLASLT")) {
        cublasStatus_t lt_status = run_cublaslt_fp32_gemm(
            g_cuda_workspace.lt_handle,
            g_cuda_workspace.stream,
            g_cuda_workspace.lt_workspace,
            g_cuda_workspace.lt_workspace_bytes,
            m,
            n,
            k,
            dev_a,
            dev_b,
            dev_c);
        if (lt_status == CUBLAS_STATUS_SUCCESS) {
            return lt_status;
        }
    }
    return run_cublas_fp32_gemm(g_cuda_workspace.handle, m, n, k, dev_a, dev_b, dev_c);
}

static cublasStatus_t run_cublas_half_precision_gemm(
    cublasHandle_t handle,
    pyc_dtype dtype,
    int m,
    int n,
    int k,
    const void* dev_a,
    const void* dev_b,
    void* dev_c) {
    cudaDataType_t cuda_dtype;

    if (dtype == PYC_DTYPE_F16) {
        cuda_dtype = CUDA_R_16F;
    } else if (dtype == PYC_DTYPE_BF16) {
#if defined(CUDA_R_16BF)
        cuda_dtype = CUDA_R_16BF;
#else
        return CUBLAS_STATUS_NOT_SUPPORTED;
#endif
    } else {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }

    {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        return cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n,
            m,
            k,
            &alpha,
            dev_b,
            cuda_dtype,
            n,
            dev_a,
            cuda_dtype,
            k,
            &beta,
            dev_c,
            cuda_dtype,
            n,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
}

static cublasStatus_t run_best_gemm(
    pyc_dtype dtype,
    int m,
    int n,
    int k,
    const void* dev_a,
    const void* dev_b,
    void* dev_c) {
    if (dtype == PYC_DTYPE_F32) {
        return run_best_fp32_gemm(
            m,
            n,
            k,
            (const float*)dev_a,
            (const float*)dev_b,
            (float*)dev_c);
    }
    return run_cublas_half_precision_gemm(
        g_cuda_workspace.handle,
        dtype,
        m,
        n,
        k,
        dev_a,
        dev_b,
        dev_c);
}

static cublasStatus_t ensure_cublaslt_fp32_plan(
    cublasLtHandle_t lt_handle,
    size_t workspace_bytes,
    int m,
    int n,
    int k) {
    cublasStatus_t status;
    pyc_cublaslt_fp32_plan* plan = &g_cuda_workspace.lt_fp32_plan;
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    int returned_results = 0;
    int tf32_enabled = env_default_true("PYC_CUDA_ALLOW_TF32");
#if defined(CUBLAS_COMPUTE_32F_FAST_TF32)
    cublasComputeType_t compute_type = tf32_enabled ?
        CUBLAS_COMPUTE_32F_FAST_TF32 :
        CUBLAS_COMPUTE_32F;
#else
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
#endif

    if (!lt_handle || m <= 0 || n <= 0 || k <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    if (plan->valid &&
        plan->m == m &&
        plan->n == n &&
        plan->k == k &&
        plan->workspace_bytes == workspace_bytes &&
        plan->tf32_enabled == tf32_enabled) {
        return CUBLAS_STATUS_SUCCESS;
    }

    pyc_cuda_lt_plan_reset(plan);

    status = cublasLtMatmulDescCreate(&plan->op_desc, compute_type, CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto fail;
    }
    status = cublasLtMatmulDescSetAttribute(plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto fail;
    }
    status = cublasLtMatmulDescSetAttribute(plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto fail;
    }

    status = cublasLtMatrixLayoutCreate(&plan->a_layout, CUDA_R_32F, m, k, k);
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto fail;
    }
    status = cublasLtMatrixLayoutCreate(&plan->b_layout, CUDA_R_32F, k, n, n);
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto fail;
    }
    status = cublasLtMatrixLayoutCreate(&plan->c_layout, CUDA_R_32F, m, n, n);
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto fail;
    }

#if defined(CUBLASLT_ORDER_ROW)
    {
        cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
        (void)cublasLtMatrixLayoutSetAttribute(plan->a_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
        (void)cublasLtMatrixLayoutSetAttribute(plan->b_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
        (void)cublasLtMatrixLayoutSetAttribute(plan->c_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    }
#endif

    status = cublasLtMatmulPreferenceCreate(&plan->pref);
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto fail;
    }
    status = cublasLtMatmulPreferenceSetAttribute(
        plan->pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_bytes,
        sizeof(workspace_bytes));
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto fail;
    }

    status = cublasLtMatmulAlgoGetHeuristic(
        lt_handle,
        plan->op_desc,
        plan->a_layout,
        plan->b_layout,
        plan->c_layout,
        plan->c_layout,
        plan->pref,
        1,
        &plan->heuristic,
        &returned_results);
    if (status != CUBLAS_STATUS_SUCCESS || returned_results <= 0) {
        status = CUBLAS_STATUS_NOT_SUPPORTED;
        goto fail;
    }

    plan->valid = 1;
    plan->tf32_enabled = tf32_enabled;
    plan->m = m;
    plan->n = n;
    plan->k = k;
    plan->workspace_bytes = workspace_bytes;
    return CUBLAS_STATUS_SUCCESS;

fail:
    pyc_cuda_lt_plan_reset(plan);
    return status;
}

static cublasStatus_t run_cublaslt_fp32_gemm(
    cublasLtHandle_t lt_handle,
    cudaStream_t stream,
    void* workspace,
    size_t workspace_bytes,
    int m,
    int n,
    int k,
    const float* dev_a,
    const float* dev_b,
    float* dev_c) {
    cublasStatus_t status;
    pyc_cublaslt_fp32_plan* plan = &g_cuda_workspace.lt_fp32_plan;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    if (!lt_handle || !dev_a || !dev_b || !dev_c) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    status = ensure_cublaslt_fp32_plan(lt_handle, workspace_bytes, m, n, k);
    if (status != CUBLAS_STATUS_SUCCESS || !plan->valid) {
        return status;
    }

    status = cublasLtMatmul(
        lt_handle,
        plan->op_desc,
        &alpha,
        dev_a,
        plan->a_layout,
        dev_b,
        plan->b_layout,
        &beta,
        dev_c,
        plan->c_layout,
        dev_c,
        plan->c_layout,
        &plan->heuristic.algo,
        workspace,
        workspace_bytes,
        stream);
    return status;
}
#endif

void pyc_cuda_dispatch_trace_init(pyc_cuda_dispatch_trace* trace) {
    if (!trace) {
        return;
    }
    memset(trace, 0, sizeof(*trace));
    trace->graph_replayed = 0;
    strcpy(trace->reason, "not_requested");
}

#if !defined(PYC_HAVE_CUDA_RUNTIME)
pyc_kernel_hardware_family pyc_cuda_current_hardware_family(void) {
    const char* override = getenv("PYC_CUDA_FORCE_HARDWARE_FAMILY");
    if (override && override[0] != '\0') {
        if (strcmp(override, "hopper") == 0) {
            return PYC_KERNEL_HARDWARE_HOPPER;
        }
        if (strcmp(override, "ada") == 0) {
            return PYC_KERNEL_HARDWARE_ADA;
        }
    }
    return PYC_KERNEL_HARDWARE_GENERIC;
}
#endif

static int detect_cuda_available(char* reason, size_t reason_size) {
    if (env_true("PYC_CUDA_DISABLE")) {
        strncpy(reason, "disabled_by_env", reason_size - 1);
        return 0;
    }
    if (env_true("PYC_CUDA_SIMULATE_AVAILABLE")) {
        strncpy(reason, "cuda_simulated_available", reason_size - 1);
        return 1;
    }
#if defined(PYC_HAVE_CUDA_RUNTIME)
    {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err == cudaSuccess && device_count > 0) {
            strncpy(reason, "cuda_runtime_detected", reason_size - 1);
            return 1;
        }
        if (err != cudaSuccess) {
            snprintf(reason, reason_size, "cuda_runtime_error_%d", (int)err);
        } else {
            strncpy(reason, "cuda_runtime_no_device", reason_size - 1);
        }
        return 0;
    }
#else
    strncpy(reason, "cuda_runtime_not_built", reason_size - 1);
    return 0;
#endif
}

static size_t shape_elements(const pyc_shape* shape) {
    size_t i;
    size_t total = 1;
    if (!shape || shape->rank == 0) {
        return 0;
    }
    for (i = 0; i < shape->rank; ++i) {
        if (shape->dims[i] <= 0) {
            return 0;
        }
        total *= (size_t)shape->dims[i];
    }
    return total;
}

static int native_cuda_supported_kind(pyc_ir_op_kind kind) {
    return kind == PYC_IR_OP_INPUT ||
           kind == PYC_IR_OP_MATMUL ||
           kind == PYC_IR_OP_ADD ||
           kind == PYC_IR_OP_RELU ||
           kind == PYC_IR_OP_OUTPUT;
}

static int native_cuda_can_execute(const pyc_ir_module* module) {
    size_t i;
    if (!module || module->op_count == 0) {
        return 0;
    }
    for (i = 0; i < module->op_count; ++i) {
        const pyc_ir_op* op = &module->ops[i];
        if (!native_cuda_supported_kind(op->kind)) {
            return 0;
        }
        if (!native_cuda_supports_dtype(op->dtype, op->kind)) {
            return 0;
        }
    }
    return 1;
}

typedef enum {
    PYC_CUDA_ADD_NONE = 0,
    PYC_CUDA_ADD_MATRIX = 1,
    PYC_CUDA_ADD_ROW_BIAS = 2,
    PYC_CUDA_ADD_SCALAR = 3
} pyc_cuda_add_mode;

typedef struct {
    int lhs_id;
    int rhs_id;
    int matmul_id;
    int add_id;
    int relu_id;
    int add_operand_id;
    int output_id;
    int final_id;
    size_t m;
    size_t k;
    size_t n;
    pyc_cuda_add_mode add_mode;
    size_t add_operand_elements;
} pyc_cuda_graph_spec;

static int classify_add_operand(
    const pyc_ir_op* add_operand,
    size_t m,
    size_t n,
    pyc_cuda_add_mode* out_mode,
    size_t* out_elements) {
    if (!add_operand || !out_mode || !out_elements) {
        return -1;
    }
    if (add_operand->shape.rank == 2 &&
        (size_t)add_operand->shape.dims[0] == m &&
        (size_t)add_operand->shape.dims[1] == n) {
        *out_mode = PYC_CUDA_ADD_MATRIX;
        *out_elements = m * n;
        return 0;
    }
    if (add_operand->shape.rank == 1 &&
        (size_t)add_operand->shape.dims[0] == n) {
        *out_mode = PYC_CUDA_ADD_ROW_BIAS;
        *out_elements = n;
        return 0;
    }
    if (add_operand->shape.rank == 1 &&
        add_operand->shape.dims[0] == 1) {
        *out_mode = PYC_CUDA_ADD_SCALAR;
        *out_elements = 1;
        return 0;
    }
    return -1;
}

static int parse_matmul_chain_graph(
    const pyc_ir_module* module,
    pyc_cuda_graph_spec* out_spec) {
    int matmul_id = -1;
    int add_id = -1;
    int relu_id = -1;
    int output_id = -1;
    size_t i;

    if (!module || !out_spec) {
        return -1;
    }
    memset(out_spec, 0, sizeof(*out_spec));
    out_spec->lhs_id = -1;
    out_spec->rhs_id = -1;
    out_spec->matmul_id = -1;
    out_spec->add_id = -1;
    out_spec->relu_id = -1;
    out_spec->add_operand_id = -1;
    out_spec->output_id = -1;
    out_spec->final_id = -1;
    out_spec->add_mode = PYC_CUDA_ADD_NONE;

    for (i = 0; i < module->op_count; ++i) {
        const pyc_ir_op* op = &module->ops[i];
        switch (op->kind) {
            case PYC_IR_OP_INPUT:
            case PYC_IR_OP_MATMUL:
            case PYC_IR_OP_ADD:
            case PYC_IR_OP_RELU:
            case PYC_IR_OP_OUTPUT:
                break;
            default:
                return -1;
        }
        if (op->kind == PYC_IR_OP_MATMUL) {
            if (matmul_id >= 0) {
                return -1;
            }
            matmul_id = (int)i;
        }
        if (op->kind == PYC_IR_OP_ADD) {
            if (add_id >= 0) {
                return -1;
            }
            add_id = (int)i;
        }
        if (op->kind == PYC_IR_OP_RELU) {
            if (relu_id >= 0) {
                return -1;
            }
            relu_id = (int)i;
        }
        if (op->kind == PYC_IR_OP_OUTPUT) {
            if (output_id >= 0) {
                return -1;
            }
            output_id = (int)i;
        }
    }

    if (matmul_id < 0 || output_id < 0) {
        return -1;
    }

    {
        const pyc_ir_op* matmul = &module->ops[(size_t)matmul_id];
        int lhs_id;
        int rhs_id;
        const pyc_ir_op* lhs;
        const pyc_ir_op* rhs;

        if (matmul->input_count < 2) {
            return -1;
        }
        lhs_id = matmul->input_ids[0];
        rhs_id = matmul->input_ids[1];
        if (lhs_id < 0 || rhs_id < 0 ||
            (size_t)lhs_id >= module->op_count ||
            (size_t)rhs_id >= module->op_count) {
            return -1;
        }
        lhs = &module->ops[(size_t)lhs_id];
        rhs = &module->ops[(size_t)rhs_id];
        if (lhs->kind != PYC_IR_OP_INPUT || rhs->kind != PYC_IR_OP_INPUT) {
            return -1;
        }
        if (lhs->shape.rank != 2 || rhs->shape.rank != 2) {
            return -1;
        }
        if (lhs->shape.dims[1] != rhs->shape.dims[0]) {
            return -1;
        }

        out_spec->lhs_id = lhs_id;
        out_spec->rhs_id = rhs_id;
        out_spec->matmul_id = matmul_id;
        out_spec->m = (size_t)lhs->shape.dims[0];
        out_spec->k = (size_t)lhs->shape.dims[1];
        out_spec->n = (size_t)rhs->shape.dims[1];
        out_spec->final_id = matmul_id;
    }

    if (add_id >= 0) {
        const pyc_ir_op* add_op = &module->ops[(size_t)add_id];
        int add_other_id = -1;
        const pyc_ir_op* add_other;
        if (add_op->input_count < 2) {
            return -1;
        }
        if (add_op->input_ids[0] == out_spec->final_id) {
            add_other_id = add_op->input_ids[1];
        } else if (add_op->input_ids[1] == out_spec->final_id) {
            add_other_id = add_op->input_ids[0];
        } else {
            return -1;
        }
        if (add_other_id < 0 || (size_t)add_other_id >= module->op_count) {
            return -1;
        }
        add_other = &module->ops[(size_t)add_other_id];
        if (add_other->kind != PYC_IR_OP_INPUT) {
            return -1;
        }
        if (classify_add_operand(
                add_other,
                out_spec->m,
                out_spec->n,
                &out_spec->add_mode,
                &out_spec->add_operand_elements) != 0) {
            return -1;
        }
        out_spec->add_id = add_id;
        out_spec->add_operand_id = add_other_id;
        out_spec->final_id = add_id;
    }

    if (relu_id >= 0) {
        const pyc_ir_op* relu_op = &module->ops[(size_t)relu_id];
        if (relu_op->input_count < 1 || relu_op->input_ids[0] != out_spec->final_id) {
            return -1;
        }
        out_spec->relu_id = relu_id;
        out_spec->final_id = relu_id;
    }

    {
        const pyc_ir_op* output = &module->ops[(size_t)output_id];
        if (output->input_count < 1) {
            return -1;
        }
        if (output->input_ids[0] != out_spec->final_id) {
            return -1;
        }
        out_spec->output_id = output_id;
    }

    return 0;
}

static void matmul_f32_naive(
    const float* lhs,
    const float* rhs,
    float* out,
    size_t m,
    size_t k,
    size_t n) {
    size_t r;
    for (r = 0; r < m; ++r) {
        size_t c;
        for (c = 0; c < n; ++c) {
            size_t t;
            float sum = 0.0f;
            for (t = 0; t < k; ++t) {
                sum += lhs[r * k + t] * rhs[t * n + c];
            }
            out[r * n + c] = sum;
        }
    }
}

static int execute_native_cuda_graph_simulated(
    const pyc_ir_module* module,
    const pyc_tensor* inputs,
    size_t input_count,
    pyc_tensor* outputs,
    size_t output_count) {
    const float* read_ptrs[PYC_IR_MAX_OPS];
    float* owned_ptrs[PYC_IR_MAX_OPS];
    size_t elem_count[PYC_IR_MAX_OPS];
    size_t input_index = 0;
    size_t output_index = 0;
    size_t i;
    int failed = 0;

    memset(read_ptrs, 0, sizeof(read_ptrs));
    memset(owned_ptrs, 0, sizeof(owned_ptrs));
    memset(elem_count, 0, sizeof(elem_count));

    for (i = 0; i < module->op_count; ++i) {
        const pyc_ir_op* op = &module->ops[i];
        elem_count[i] = shape_elements(&op->shape);

        switch (op->kind) {
            case PYC_IR_OP_INPUT: {
                size_t required = elem_count[i] * pyc_cuda_dtype_size_bytes(op->dtype);
                if (input_index >= input_count || !inputs[input_index].data) {
                    failed = 1;
                    goto cleanup;
                }
                if (inputs[input_index].dtype != op->dtype) {
                    failed = 1;
                    goto cleanup;
                }
                if (required == 0 || inputs[input_index].size_bytes < required) {
                    failed = 1;
                    goto cleanup;
                }
                read_ptrs[i] = (const float*)inputs[input_index].data;
                input_index++;
                break;
            }
            case PYC_IR_OP_MATMUL: {
                int lhs_id;
                int rhs_id;
                const pyc_ir_op* lhs_op;
                const pyc_ir_op* rhs_op;
                size_t m;
                size_t k;
                size_t n;
                float* out;
                if (op->input_count < 2) {
                    failed = 1;
                    goto cleanup;
                }
                if (op->dtype != PYC_DTYPE_F32) {
                    failed = 1;
                    goto cleanup;
                }
                lhs_id = op->input_ids[0];
                rhs_id = op->input_ids[1];
                if (lhs_id < 0 || rhs_id < 0 ||
                    (size_t)lhs_id >= module->op_count ||
                    (size_t)rhs_id >= module->op_count) {
                    failed = 1;
                    goto cleanup;
                }
                lhs_op = &module->ops[(size_t)lhs_id];
                rhs_op = &module->ops[(size_t)rhs_id];
                if (lhs_op->shape.rank != 2 || rhs_op->shape.rank != 2) {
                    failed = 1;
                    goto cleanup;
                }
                m = (size_t)lhs_op->shape.dims[0];
                k = (size_t)lhs_op->shape.dims[1];
                n = (size_t)rhs_op->shape.dims[1];
                if (k != (size_t)rhs_op->shape.dims[0]) {
                    failed = 1;
                    goto cleanup;
                }
                out = (float*)malloc(m * n * sizeof(float));
                if (!out) {
                    failed = 1;
                    goto cleanup;
                }
                matmul_f32_naive(
                    read_ptrs[(size_t)lhs_id],
                    read_ptrs[(size_t)rhs_id],
                    out,
                    m,
                    k,
                    n);
                owned_ptrs[i] = out;
                read_ptrs[i] = out;
                break;
            }
            case PYC_IR_OP_ADD: {
                int a_id;
                int b_id;
                const pyc_ir_op* a_op;
                const pyc_ir_op* b_op;
                const float* a_ptr;
                const float* b_ptr;
                float* out;
                size_t j;
                size_t out_elems;
                size_t a_elems;
                size_t b_elems;
                if (op->input_count < 2 || elem_count[i] == 0) {
                    failed = 1;
                    goto cleanup;
                }
                if (op->dtype != PYC_DTYPE_F32) {
                    failed = 1;
                    goto cleanup;
                }
                a_id = op->input_ids[0];
                b_id = op->input_ids[1];
                if (a_id < 0 || b_id < 0 ||
                    (size_t)a_id >= module->op_count ||
                    (size_t)b_id >= module->op_count) {
                    failed = 1;
                    goto cleanup;
                }
                a_op = &module->ops[(size_t)a_id];
                b_op = &module->ops[(size_t)b_id];
                a_ptr = read_ptrs[(size_t)a_id];
                b_ptr = read_ptrs[(size_t)b_id];
                if (!a_ptr || !b_ptr) {
                    failed = 1;
                    goto cleanup;
                }
                out = (float*)malloc(elem_count[i] * sizeof(float));
                if (!out) {
                    failed = 1;
                    goto cleanup;
                }
                out_elems = elem_count[i];
                a_elems = shape_elements(&a_op->shape);
                b_elems = shape_elements(&b_op->shape);
                if (a_elems == out_elems && b_elems == out_elems) {
                    for (j = 0; j < out_elems; ++j) {
                        out[j] = a_ptr[j] + b_ptr[j];
                    }
                } else if (a_op->shape.rank == 2 &&
                           b_op->shape.rank == 1 &&
                           (size_t)b_op->shape.dims[0] == (size_t)a_op->shape.dims[1] &&
                           out_elems == a_elems) {
                    size_t rows = (size_t)a_op->shape.dims[0];
                    size_t cols = (size_t)a_op->shape.dims[1];
                    size_t r;
                    for (r = 0; r < rows; ++r) {
                        size_t c;
                        for (c = 0; c < cols; ++c) {
                            out[r * cols + c] = a_ptr[r * cols + c] + b_ptr[c];
                        }
                    }
                } else if (b_op->shape.rank == 2 &&
                           a_op->shape.rank == 1 &&
                           (size_t)a_op->shape.dims[0] == (size_t)b_op->shape.dims[1] &&
                           out_elems == b_elems) {
                    size_t rows = (size_t)b_op->shape.dims[0];
                    size_t cols = (size_t)b_op->shape.dims[1];
                    size_t r;
                    for (r = 0; r < rows; ++r) {
                        size_t c;
                        for (c = 0; c < cols; ++c) {
                            out[r * cols + c] = a_ptr[c] + b_ptr[r * cols + c];
                        }
                    }
                } else if (a_elems == out_elems && b_elems == 1) {
                    float bias = b_ptr[0];
                    for (j = 0; j < out_elems; ++j) {
                        out[j] = a_ptr[j] + bias;
                    }
                } else if (b_elems == out_elems && a_elems == 1) {
                    float bias = a_ptr[0];
                    for (j = 0; j < out_elems; ++j) {
                        out[j] = b_ptr[j] + bias;
                    }
                } else {
                    free(out);
                    failed = 1;
                    goto cleanup;
                }
                owned_ptrs[i] = out;
                read_ptrs[i] = out;
                break;
            }
            case PYC_IR_OP_RELU: {
                int src_id;
                float* out;
                size_t j;
                if (op->input_count < 1 || elem_count[i] == 0) {
                    failed = 1;
                    goto cleanup;
                }
                if (op->dtype != PYC_DTYPE_F32) {
                    failed = 1;
                    goto cleanup;
                }
                src_id = op->input_ids[0];
                if (src_id < 0 || (size_t)src_id >= module->op_count) {
                    failed = 1;
                    goto cleanup;
                }
                out = (float*)malloc(elem_count[i] * sizeof(float));
                if (!out) {
                    failed = 1;
                    goto cleanup;
                }
                for (j = 0; j < elem_count[i]; ++j) {
                    float v = read_ptrs[(size_t)src_id][j];
                    out[j] = v > 0.0f ? v : 0.0f;
                }
                owned_ptrs[i] = out;
                read_ptrs[i] = out;
                break;
            }
            case PYC_IR_OP_OUTPUT: {
                int src_id;
                size_t required;
                if (output_index >= output_count || !outputs[output_index].data) {
                    failed = 1;
                    goto cleanup;
                }
                if (op->input_count < 1) {
                    failed = 1;
                    goto cleanup;
                }
                src_id = op->input_ids[0];
                if (src_id < 0 || (size_t)src_id >= module->op_count) {
                    failed = 1;
                    goto cleanup;
                }
                if (outputs[output_index].dtype != module->ops[(size_t)src_id].dtype) {
                    failed = 1;
                    goto cleanup;
                }
                required = elem_count[(size_t)src_id] * pyc_cuda_dtype_size_bytes(module->ops[(size_t)src_id].dtype);
                if (required == 0 || outputs[output_index].size_bytes < required) {
                    failed = 1;
                    goto cleanup;
                }
                memcpy(outputs[output_index].data, read_ptrs[(size_t)src_id], required);
                output_index++;
                break;
            }
            default:
                failed = 1;
                goto cleanup;
        }
    }

cleanup:
    for (i = 0; i < module->op_count; ++i) {
        free(owned_ptrs[i]);
    }
    return failed ? -1 : 0;
}

#if defined(PYC_HAVE_CUDA_RUNTIME)
static int find_input_tensor_index(
    const pyc_ir_module* module,
    int target_op_id,
    int* out_input_index) {
    size_t op_idx;
    size_t input_seen = 0;
    if (!module || !out_input_index || target_op_id < 0) {
        return -1;
    }
    for (op_idx = 0; op_idx < module->op_count; ++op_idx) {
        const pyc_ir_op* op = &module->ops[op_idx];
        if (op->kind != PYC_IR_OP_INPUT) {
            continue;
        }
        if ((int)op_idx == target_op_id) {
            *out_input_index = (int)input_seen;
            return 0;
        }
        input_seen++;
    }
    return -1;
}

static void apply_add_epilogue(
    float* out,
    const float* add_src,
    size_t m,
    size_t n,
    pyc_cuda_add_mode mode) {
    size_t i;
    if (!out || !add_src || mode == PYC_CUDA_ADD_NONE) {
        return;
    }
    if (mode == PYC_CUDA_ADD_MATRIX) {
        size_t total = m * n;
        for (i = 0; i < total; ++i) {
            out[i] += add_src[i];
        }
        return;
    }
    if (mode == PYC_CUDA_ADD_ROW_BIAS) {
        size_t r;
        for (r = 0; r < m; ++r) {
            size_t c;
            for (c = 0; c < n; ++c) {
                out[r * n + c] += add_src[c];
            }
        }
        return;
    }
    if (mode == PYC_CUDA_ADD_SCALAR) {
        float v = add_src[0];
        size_t total = m * n;
        for (i = 0; i < total; ++i) {
            out[i] += v;
        }
    }
}

static void apply_relu_epilogue(float* out, size_t total) {
    size_t i;
    if (!out) {
        return;
    }
    for (i = 0; i < total; ++i) {
        float v = out[i];
        out[i] = v > 0.0f ? v : 0.0f;
    }
}

static int run_cuda_matmul_pipeline(
    const void* host_a,
    const void* host_b,
    void* host_out,
    size_t a_bytes,
    size_t b_bytes,
    size_t c_bytes,
    size_t m,
    size_t k,
    size_t n,
    pyc_dtype dtype,
    int lhs_copy_required,
    int rhs_copy_required,
    pyc_cuda_dispatch_trace* trace,
    char* failure_reason,
    size_t failure_reason_size) {
    cublasStatus_t cublas_status;
    cudaError_t cuda_status;
    int skip_output_copy = env_true("PYC_CUDA_SKIP_HOST_OUTPUT_COPY");
    double t0;
    double t1;

    if (lhs_copy_required) {
        t0 = wall_ms_now();
        cuda_status = cudaMemcpyAsync(
            g_cuda_workspace.dev_a,
            host_a,
            a_bytes,
            cudaMemcpyHostToDevice,
            g_cuda_workspace.stream);
        if (cuda_status != cudaSuccess) {
            if (failure_reason && failure_reason_size > 0) {
                snprintf(failure_reason, failure_reason_size, "cuda_copy_in_lhs_failed:%s", cudaGetErrorString(cuda_status));
            }
            return -1;
        }
        if (trace) {
            trace->copy_in_ms += wall_ms_now() - t0;
        }
    }
    if (rhs_copy_required) {
        t0 = wall_ms_now();
        cuda_status = cudaMemcpyAsync(
            g_cuda_workspace.dev_b,
            host_b,
            b_bytes,
            cudaMemcpyHostToDevice,
            g_cuda_workspace.stream);
        if (cuda_status != cudaSuccess) {
            if (failure_reason && failure_reason_size > 0) {
                snprintf(failure_reason, failure_reason_size, "cuda_copy_in_rhs_failed:%s", cudaGetErrorString(cuda_status));
            }
            return -1;
        }
        if (trace) {
            trace->copy_in_ms += wall_ms_now() - t0;
        }
    }

    t0 = wall_ms_now();
    cublas_status = run_best_gemm(
        dtype,
        (int)m,
        (int)n,
        (int)k,
        g_cuda_workspace.dev_a,
        g_cuda_workspace.dev_b,
        g_cuda_workspace.dev_c);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        if (failure_reason && failure_reason_size > 0) {
            snprintf(
                failure_reason,
                failure_reason_size,
                "cuda_gemm_failed:%s",
                cublas_status_name(dtype, cublas_status));
        }
        return -1;
    }
    t1 = wall_ms_now();
    if (trace) {
        trace->kernel_ms += t1 - t0;
    }

    if (!skip_output_copy) {
        t0 = wall_ms_now();
        cuda_status = cudaMemcpyAsync(
            host_out,
            g_cuda_workspace.dev_c,
            c_bytes,
            cudaMemcpyDeviceToHost,
            g_cuda_workspace.stream);
        if (cuda_status != cudaSuccess) {
            if (failure_reason && failure_reason_size > 0) {
                snprintf(failure_reason, failure_reason_size, "cuda_copy_out_failed:%s", cudaGetErrorString(cuda_status));
            }
            return -1;
        }
        if (trace) {
            trace->copy_out_ms += wall_ms_now() - t0;
        }
    }
    t0 = wall_ms_now();
    cuda_status = cudaStreamSynchronize(g_cuda_workspace.stream);
    if (trace) {
        trace->sync_ms += wall_ms_now() - t0;
    }
    if (cuda_status != cudaSuccess && failure_reason && failure_reason_size > 0) {
        snprintf(failure_reason, failure_reason_size, "cuda_sync_failed:%s", cudaGetErrorString(cuda_status));
    }
    return cuda_status == cudaSuccess ? 0 : -1;
}

static int run_promoted_cutlass_gemm_pipeline(
    const char* promoted_symbol,
    const void* host_a,
    const void* host_b,
    void* host_out,
    size_t a_bytes,
    size_t b_bytes,
    size_t c_bytes,
    size_t m,
    size_t k,
    size_t n,
    int lhs_copy_required,
    int rhs_copy_required,
    int* out_used,
    pyc_cuda_dispatch_trace* trace,
    char* failure_reason,
    size_t failure_reason_size) {
    cudaError_t cuda_status;
    int skip_output_copy = env_true("PYC_CUDA_SKIP_HOST_OUTPUT_COPY");
    double t0;
    double t1;

    if (out_used) {
        *out_used = 0;
    }
    if (!promoted_symbol || promoted_symbol[0] == '\0') {
        return 1;
    }

    if (lhs_copy_required) {
        t0 = wall_ms_now();
        cuda_status = cudaMemcpyAsync(
            g_cuda_workspace.dev_a,
            host_a,
            a_bytes,
            cudaMemcpyHostToDevice,
            g_cuda_workspace.stream);
        if (cuda_status != cudaSuccess) {
            if (failure_reason && failure_reason_size > 0) {
                snprintf(failure_reason, failure_reason_size, "cutlass_copy_in_lhs_failed:%s", cudaGetErrorString(cuda_status));
            }
            return -1;
        }
        if (trace) {
            trace->copy_in_ms += wall_ms_now() - t0;
        }
    }
    if (rhs_copy_required) {
        t0 = wall_ms_now();
        cuda_status = cudaMemcpyAsync(
            g_cuda_workspace.dev_b,
            host_b,
            b_bytes,
            cudaMemcpyHostToDevice,
            g_cuda_workspace.stream);
        if (cuda_status != cudaSuccess) {
            if (failure_reason && failure_reason_size > 0) {
                snprintf(failure_reason, failure_reason_size, "cutlass_copy_in_rhs_failed:%s", cudaGetErrorString(cuda_status));
            }
            return -1;
        }
        if (trace) {
            trace->copy_in_ms += wall_ms_now() - t0;
        }
    }

    t0 = wall_ms_now();
    if (pyc_cutlass_gemm_dispatch(
            promoted_symbol,
            (int)m,
            (int)n,
            (int)k,
            g_cuda_workspace.dev_a,
            g_cuda_workspace.dev_b,
            g_cuda_workspace.dev_c,
            1.0f,
            0.0f,
            g_cuda_workspace.stream) != 0) {
        if (failure_reason && failure_reason_size > 0) {
            snprintf(failure_reason, failure_reason_size, "cutlass_gemm_failed:%s", promoted_symbol);
        }
        return -1;
    }
    t1 = wall_ms_now();
    if (trace) {
        trace->kernel_ms += t1 - t0;
    }

    if (!skip_output_copy) {
        t0 = wall_ms_now();
        cuda_status = cudaMemcpyAsync(
            host_out,
            g_cuda_workspace.dev_c,
            c_bytes,
            cudaMemcpyDeviceToHost,
            g_cuda_workspace.stream);
        if (cuda_status != cudaSuccess) {
            if (failure_reason && failure_reason_size > 0) {
                snprintf(failure_reason, failure_reason_size, "cutlass_copy_out_failed:%s", cudaGetErrorString(cuda_status));
            }
            return -1;
        }
        if (trace) {
            trace->copy_out_ms += wall_ms_now() - t0;
        }
    }
    t0 = wall_ms_now();
    cuda_status = cudaStreamSynchronize(g_cuda_workspace.stream);
    if (trace) {
        trace->sync_ms += wall_ms_now() - t0;
    }
    if (cuda_status != cudaSuccess) {
        if (failure_reason && failure_reason_size > 0) {
            snprintf(failure_reason, failure_reason_size, "cutlass_sync_failed:%s", cudaGetErrorString(cuda_status));
        }
        return -1;
    }
    if (out_used) {
        *out_used = 1;
    }
    return 0;
}

static int run_promoted_cutlass_gemm_pipeline_graph(
    const char* promoted_symbol,
    const void* host_a,
    const void* host_b,
    void* host_out,
    size_t a_bytes,
    size_t b_bytes,
    size_t c_bytes,
    size_t m,
    size_t k,
    size_t n,
    pyc_dtype dtype,
    int lhs_copy_required,
    int rhs_copy_required,
    int* out_replayed,
    pyc_cuda_dispatch_trace* trace,
    char* failure_reason,
    size_t failure_reason_size) {
    cudaError_t cuda_status;
    cudaError_t capture_status;
    int skip_output_copy = env_true("PYC_CUDA_SKIP_HOST_OUTPUT_COPY");
    double t0;
    double t1;

    if (!out_replayed) {
        return -1;
    }
    *out_replayed = 0;

    if (!promoted_symbol || promoted_symbol[0] == '\0') {
        return 1;
    }
    if (!env_default_true("PYC_CUDA_ENABLE_GRAPH_REPLAY")) {
        return 1;
    }
    if (pyc_cuda_workspace_prepare_graph_stages(
            a_bytes,
            b_bytes,
            c_bytes,
            lhs_copy_required,
            rhs_copy_required,
            !skip_output_copy) != 0) {
        if (failure_reason && failure_reason_size > 0) {
            snprintf(failure_reason, failure_reason_size, "cutlass_graph_stage_alloc_failed");
        }
        return -1;
    }
    pyc_cuda_stage_graph_inputs(
        host_a,
        host_b,
        a_bytes,
        b_bytes,
        lhs_copy_required,
        rhs_copy_required,
        trace);

    if (g_cuda_workspace.graph_valid &&
        g_cuda_workspace.graph_exec != NULL &&
        g_cuda_workspace.graph_uses_promoted_gemm &&
        g_cuda_workspace.graph_m == m &&
        g_cuda_workspace.graph_k == k &&
        g_cuda_workspace.graph_n == n &&
        g_cuda_workspace.graph_dtype == dtype &&
        g_cuda_workspace.graph_lhs_copy_required == lhs_copy_required &&
        g_cuda_workspace.graph_rhs_copy_required == rhs_copy_required &&
        strcmp(g_cuda_workspace.graph_promoted_symbol, promoted_symbol) == 0) {
        t0 = wall_ms_now();
        cuda_status = cudaGraphLaunch(g_cuda_workspace.graph_exec, g_cuda_workspace.stream);
        if (cuda_status != cudaSuccess) {
            pyc_cuda_graph_reset();
            if (failure_reason && failure_reason_size > 0) {
                snprintf(failure_reason, failure_reason_size, "cutlass_graph_launch_failed:%s", cudaGetErrorString(cuda_status));
            }
            return -1;
        }
        t1 = wall_ms_now();
        if (trace) {
            trace->graph_replayed = 1;
            trace->kernel_ms += t1 - t0;
        }
        t0 = wall_ms_now();
        cuda_status = cudaStreamSynchronize(g_cuda_workspace.stream);
        if (trace) {
            trace->sync_ms += wall_ms_now() - t0;
        }
        if (cuda_status != cudaSuccess) {
            pyc_cuda_graph_reset();
            if (failure_reason && failure_reason_size > 0) {
                snprintf(failure_reason, failure_reason_size, "cutlass_graph_sync_failed:%s", cudaGetErrorString(cuda_status));
            }
            return -1;
        }
        if (!skip_output_copy) {
            pyc_cuda_flush_graph_output(host_out, c_bytes, trace);
        }
        *out_replayed = 1;
        return 0;
    }

    pyc_cuda_graph_reset();
    capture_status = cudaStreamBeginCapture(g_cuda_workspace.stream, cudaStreamCaptureModeGlobal);
    if (capture_status != cudaSuccess) {
        if (failure_reason && failure_reason_size > 0) {
            snprintf(failure_reason, failure_reason_size, "cutlass_graph_begin_capture_failed:%s", cudaGetErrorString(capture_status));
        }
        return -1;
    }

    if (lhs_copy_required) {
        cuda_status = cudaMemcpyAsync(
            g_cuda_workspace.dev_a,
            g_cuda_workspace.graph_host_a_stage,
            a_bytes,
            cudaMemcpyHostToDevice,
            g_cuda_workspace.stream);
        if (cuda_status != cudaSuccess) {
            (void)cudaStreamEndCapture(g_cuda_workspace.stream, &g_cuda_workspace.graph);
            pyc_cuda_graph_reset();
            if (failure_reason && failure_reason_size > 0) {
                snprintf(failure_reason, failure_reason_size, "cutlass_graph_copy_in_lhs_failed:%s", cudaGetErrorString(cuda_status));
            }
            return -1;
        }
    }
    if (rhs_copy_required) {
        cuda_status = cudaMemcpyAsync(
            g_cuda_workspace.dev_b,
            g_cuda_workspace.graph_host_b_stage,
            b_bytes,
            cudaMemcpyHostToDevice,
            g_cuda_workspace.stream);
        if (cuda_status != cudaSuccess) {
            (void)cudaStreamEndCapture(g_cuda_workspace.stream, &g_cuda_workspace.graph);
            pyc_cuda_graph_reset();
            if (failure_reason && failure_reason_size > 0) {
                snprintf(failure_reason, failure_reason_size, "cutlass_graph_copy_in_rhs_failed:%s", cudaGetErrorString(cuda_status));
            }
            return -1;
        }
    }

    t0 = wall_ms_now();
    if (pyc_cutlass_gemm_dispatch(
            promoted_symbol,
            (int)m,
            (int)n,
            (int)k,
            g_cuda_workspace.dev_a,
            g_cuda_workspace.dev_b,
            g_cuda_workspace.dev_c,
            1.0f,
            0.0f,
            g_cuda_workspace.stream) != 0) {
        (void)cudaStreamEndCapture(g_cuda_workspace.stream, &g_cuda_workspace.graph);
        pyc_cuda_graph_reset();
        if (failure_reason && failure_reason_size > 0) {
            snprintf(failure_reason, failure_reason_size, "cutlass_graph_gemm_failed:%s", promoted_symbol);
        }
        return -1;
    }
    t1 = wall_ms_now();
    if (trace) {
        trace->kernel_ms += t1 - t0;
    }

    if (!skip_output_copy) {
        cuda_status = cudaMemcpyAsync(
            g_cuda_workspace.graph_host_out_stage,
            g_cuda_workspace.dev_c,
            c_bytes,
            cudaMemcpyDeviceToHost,
            g_cuda_workspace.stream);
        if (cuda_status != cudaSuccess) {
            (void)cudaStreamEndCapture(g_cuda_workspace.stream, &g_cuda_workspace.graph);
            pyc_cuda_graph_reset();
            if (failure_reason && failure_reason_size > 0) {
                snprintf(failure_reason, failure_reason_size, "cutlass_graph_copy_out_failed:%s", cudaGetErrorString(cuda_status));
            }
            return -1;
        }
    }

    capture_status = cudaStreamEndCapture(g_cuda_workspace.stream, &g_cuda_workspace.graph);
    if (capture_status != cudaSuccess) {
        pyc_cuda_graph_reset();
        if (failure_reason && failure_reason_size > 0) {
            snprintf(failure_reason, failure_reason_size, "cutlass_graph_end_capture_failed:%s", cudaGetErrorString(capture_status));
        }
        return -1;
    }
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
    cuda_status = cudaGraphInstantiate(&g_cuda_workspace.graph_exec, g_cuda_workspace.graph, 0);
#else
    cuda_status = cudaGraphInstantiate(
        &g_cuda_workspace.graph_exec,
        g_cuda_workspace.graph,
        NULL,
        NULL,
        0);
#endif
    if (cuda_status != cudaSuccess) {
        pyc_cuda_graph_reset();
        if (failure_reason && failure_reason_size > 0) {
            snprintf(failure_reason, failure_reason_size, "cutlass_graph_instantiate_failed:%s", cudaGetErrorString(cuda_status));
        }
        return -1;
    }

    g_cuda_workspace.graph_valid = 1;
    g_cuda_workspace.graph_uses_promoted_gemm = 1;
    g_cuda_workspace.graph_m = m;
    g_cuda_workspace.graph_k = k;
    g_cuda_workspace.graph_n = n;
    g_cuda_workspace.graph_dtype = dtype;
    g_cuda_workspace.graph_lhs_copy_required = lhs_copy_required;
    g_cuda_workspace.graph_rhs_copy_required = rhs_copy_required;
    g_cuda_workspace.graph_host_a_ptr = (const void*)g_cuda_workspace.graph_host_a_stage;
    g_cuda_workspace.graph_host_b_ptr = (const void*)g_cuda_workspace.graph_host_b_stage;
    g_cuda_workspace.graph_host_out_ptr = (const void*)g_cuda_workspace.graph_host_out_stage;
    strncpy(g_cuda_workspace.graph_promoted_symbol, promoted_symbol, sizeof(g_cuda_workspace.graph_promoted_symbol) - 1);
    g_cuda_workspace.graph_promoted_symbol[sizeof(g_cuda_workspace.graph_promoted_symbol) - 1] = '\0';

    t0 = wall_ms_now();
    cuda_status = cudaGraphLaunch(g_cuda_workspace.graph_exec, g_cuda_workspace.stream);
    if (cuda_status != cudaSuccess) {
        pyc_cuda_graph_reset();
        if (failure_reason && failure_reason_size > 0) {
            snprintf(failure_reason, failure_reason_size, "cutlass_graph_launch_failed:%s", cudaGetErrorString(cuda_status));
        }
        return -1;
    }
    t1 = wall_ms_now();
    if (trace) {
        trace->kernel_ms += t1 - t0;
    }
    t0 = wall_ms_now();
    cuda_status = cudaStreamSynchronize(g_cuda_workspace.stream);
    if (trace) {
        trace->sync_ms += wall_ms_now() - t0;
    }
    if (cuda_status != cudaSuccess) {
        pyc_cuda_graph_reset();
        if (failure_reason && failure_reason_size > 0) {
            snprintf(failure_reason, failure_reason_size, "cutlass_graph_sync_failed:%s", cudaGetErrorString(cuda_status));
        }
        return -1;
    }
    if (!skip_output_copy) {
        pyc_cuda_flush_graph_output(host_out, c_bytes, trace);
    }
    return 0;
}

static int graph_signature_matches(
    const void* host_a,
    const void* host_b,
    void* host_out,
    size_t m,
    size_t k,
    size_t n,
    pyc_dtype dtype,
    int lhs_copy_required,
    int rhs_copy_required) {
    (void)host_a;
    (void)host_b;
    (void)host_out;
    return g_cuda_workspace.graph_valid &&
           g_cuda_workspace.graph_exec != NULL &&
           g_cuda_workspace.graph_m == m &&
           g_cuda_workspace.graph_k == k &&
           g_cuda_workspace.graph_n == n &&
           g_cuda_workspace.graph_dtype == dtype &&
           g_cuda_workspace.graph_lhs_copy_required == lhs_copy_required &&
           g_cuda_workspace.graph_rhs_copy_required == rhs_copy_required;
}

static int run_cuda_matmul_pipeline_graph(
    const void* host_a,
    const void* host_b,
    void* host_out,
    size_t a_bytes,
    size_t b_bytes,
    size_t c_bytes,
    size_t m,
    size_t k,
    size_t n,
    pyc_dtype dtype,
    int lhs_copy_required,
    int rhs_copy_required,
    int* out_replayed,
    pyc_cuda_dispatch_trace* trace,
    char* failure_reason,
    size_t failure_reason_size) {
    cublasStatus_t cublas_status;
    cudaError_t cuda_status;
    cudaError_t capture_status;
    int skip_output_copy = env_true("PYC_CUDA_SKIP_HOST_OUTPUT_COPY");
    double t0;
    double t1;

    if (!out_replayed) {
        return -1;
    }
    *out_replayed = 0;

    if (!env_default_true("PYC_CUDA_ENABLE_GRAPH_REPLAY")) {
        return 1;
    }
    if (pyc_cuda_workspace_prepare_graph_stages(
            a_bytes,
            b_bytes,
            c_bytes,
            lhs_copy_required,
            rhs_copy_required,
            !skip_output_copy) != 0) {
        if (failure_reason && failure_reason_size > 0) {
            snprintf(failure_reason, failure_reason_size, "cuda_graph_stage_alloc_failed");
        }
        return -1;
    }
    pyc_cuda_stage_graph_inputs(
        host_a,
        host_b,
        a_bytes,
        b_bytes,
        lhs_copy_required,
        rhs_copy_required,
        trace);

    if (graph_signature_matches(host_a, host_b, host_out, m, k, n, dtype, lhs_copy_required, rhs_copy_required)) {
        t0 = wall_ms_now();
        cuda_status = cudaGraphLaunch(g_cuda_workspace.graph_exec, g_cuda_workspace.stream);
        if (cuda_status != cudaSuccess) {
            pyc_cuda_graph_reset();
            if (failure_reason && failure_reason_size > 0) {
                snprintf(failure_reason, failure_reason_size, "cuda_graph_launch_failed:%s", cudaGetErrorString(cuda_status));
            }
            return -1;
        }
        t1 = wall_ms_now();
        if (trace) {
            trace->graph_replayed = 1;
            trace->kernel_ms += t1 - t0;
        }
        t0 = wall_ms_now();
        cuda_status = cudaStreamSynchronize(g_cuda_workspace.stream);
        if (trace) {
            trace->sync_ms += wall_ms_now() - t0;
        }
        if (cuda_status != cudaSuccess) {
            pyc_cuda_graph_reset();
            if (failure_reason && failure_reason_size > 0) {
                snprintf(failure_reason, failure_reason_size, "cuda_graph_sync_failed:%s", cudaGetErrorString(cuda_status));
            }
            return -1;
        }
        if (!skip_output_copy) {
            pyc_cuda_flush_graph_output(host_out, c_bytes, trace);
        }
        *out_replayed = 1;
        return 0;
    }

    pyc_cuda_graph_reset();
    capture_status = cudaStreamBeginCapture(g_cuda_workspace.stream, cudaStreamCaptureModeGlobal);
    if (capture_status != cudaSuccess) {
        if (failure_reason && failure_reason_size > 0) {
            snprintf(failure_reason, failure_reason_size, "cuda_graph_begin_capture_failed:%s", cudaGetErrorString(capture_status));
        }
        return -1;
    }

    if (lhs_copy_required) {
        cuda_status = cudaMemcpyAsync(
            g_cuda_workspace.dev_a,
            g_cuda_workspace.graph_host_a_stage,
            a_bytes,
            cudaMemcpyHostToDevice,
            g_cuda_workspace.stream);
        if (cuda_status != cudaSuccess) {
            (void)cudaStreamEndCapture(g_cuda_workspace.stream, &g_cuda_workspace.graph);
            pyc_cuda_graph_reset();
            if (failure_reason && failure_reason_size > 0) {
                snprintf(failure_reason, failure_reason_size, "cuda_graph_copy_in_lhs_failed:%s", cudaGetErrorString(cuda_status));
            }
            return -1;
        }
    }
    if (rhs_copy_required) {
        cuda_status = cudaMemcpyAsync(
            g_cuda_workspace.dev_b,
            g_cuda_workspace.graph_host_b_stage,
            b_bytes,
            cudaMemcpyHostToDevice,
            g_cuda_workspace.stream);
        if (cuda_status != cudaSuccess) {
            (void)cudaStreamEndCapture(g_cuda_workspace.stream, &g_cuda_workspace.graph);
            pyc_cuda_graph_reset();
            if (failure_reason && failure_reason_size > 0) {
                snprintf(failure_reason, failure_reason_size, "cuda_graph_copy_in_rhs_failed:%s", cudaGetErrorString(cuda_status));
            }
            return -1;
        }
    }

    t0 = wall_ms_now();
    cublas_status = run_best_gemm(
        dtype,
        (int)m,
        (int)n,
        (int)k,
        g_cuda_workspace.dev_a,
        g_cuda_workspace.dev_b,
        g_cuda_workspace.dev_c);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        (void)cudaStreamEndCapture(g_cuda_workspace.stream, &g_cuda_workspace.graph);
        pyc_cuda_graph_reset();
        if (failure_reason && failure_reason_size > 0) {
            snprintf(
                failure_reason,
                failure_reason_size,
                "cuda_graph_gemm_failed:%s",
                cublas_status_name(dtype, cublas_status));
        }
        return -1;
    }
    t1 = wall_ms_now();
    if (trace) {
        trace->kernel_ms += t1 - t0;
    }

    if (!skip_output_copy) {
        cuda_status = cudaMemcpyAsync(
            g_cuda_workspace.graph_host_out_stage,
            g_cuda_workspace.dev_c,
            c_bytes,
            cudaMemcpyDeviceToHost,
            g_cuda_workspace.stream);
        if (cuda_status != cudaSuccess) {
            (void)cudaStreamEndCapture(g_cuda_workspace.stream, &g_cuda_workspace.graph);
            pyc_cuda_graph_reset();
            if (failure_reason && failure_reason_size > 0) {
                snprintf(failure_reason, failure_reason_size, "cuda_graph_copy_out_failed:%s", cudaGetErrorString(cuda_status));
            }
            return -1;
        }
    }

    capture_status = cudaStreamEndCapture(g_cuda_workspace.stream, &g_cuda_workspace.graph);
    if (capture_status != cudaSuccess) {
        pyc_cuda_graph_reset();
        if (failure_reason && failure_reason_size > 0) {
            snprintf(failure_reason, failure_reason_size, "cuda_graph_end_capture_failed:%s", cudaGetErrorString(capture_status));
        }
        return -1;
    }
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
    cuda_status = cudaGraphInstantiate(&g_cuda_workspace.graph_exec, g_cuda_workspace.graph, 0);
#else
    cuda_status = cudaGraphInstantiate(
        &g_cuda_workspace.graph_exec,
        g_cuda_workspace.graph,
        NULL,
        NULL,
        0);
#endif
    if (cuda_status != cudaSuccess) {
        pyc_cuda_graph_reset();
        if (failure_reason && failure_reason_size > 0) {
            snprintf(failure_reason, failure_reason_size, "cuda_graph_instantiate_failed:%s", cudaGetErrorString(cuda_status));
        }
        return -1;
    }

    g_cuda_workspace.graph_valid = 1;
    g_cuda_workspace.graph_m = m;
    g_cuda_workspace.graph_k = k;
    g_cuda_workspace.graph_n = n;
    g_cuda_workspace.graph_dtype = dtype;
    g_cuda_workspace.graph_lhs_copy_required = lhs_copy_required;
    g_cuda_workspace.graph_host_a_ptr = (const void*)g_cuda_workspace.graph_host_a_stage;
    g_cuda_workspace.graph_host_b_ptr = (const void*)g_cuda_workspace.graph_host_b_stage;
    g_cuda_workspace.graph_host_out_ptr = (const void*)g_cuda_workspace.graph_host_out_stage;
    g_cuda_workspace.graph_rhs_copy_required = rhs_copy_required;

    t0 = wall_ms_now();
    cuda_status = cudaGraphLaunch(g_cuda_workspace.graph_exec, g_cuda_workspace.stream);
    if (cuda_status != cudaSuccess) {
        pyc_cuda_graph_reset();
        if (failure_reason && failure_reason_size > 0) {
            snprintf(failure_reason, failure_reason_size, "cuda_graph_launch_failed:%s", cudaGetErrorString(cuda_status));
        }
        return -1;
    }
    t1 = wall_ms_now();
    if (trace) {
        trace->kernel_ms += t1 - t0;
    }
    t0 = wall_ms_now();
    cuda_status = cudaStreamSynchronize(g_cuda_workspace.stream);
    if (trace) {
        trace->sync_ms += wall_ms_now() - t0;
    }
    if (cuda_status != cudaSuccess) {
        pyc_cuda_graph_reset();
        if (failure_reason && failure_reason_size > 0) {
            snprintf(failure_reason, failure_reason_size, "cuda_graph_sync_failed:%s", cudaGetErrorString(cuda_status));
        }
        return -1;
    }
    if (!skip_output_copy) {
        pyc_cuda_flush_graph_output(host_out, c_bytes, trace);
    }
    return 0;
}

static int execute_native_cuda_graph_cuda(
    const pyc_ir_module* module,
    const pyc_tensor* inputs,
    size_t input_count,
    pyc_tensor* outputs,
    size_t output_count,
    pyc_cuda_dispatch_trace* trace,
    char* native_reason,
    size_t native_reason_size) {
    pyc_cuda_graph_spec spec;
    int lhs_input_index = -1;
    int rhs_input_index = -1;
    int add_input_index = -1;
    int target_output_index = -1;
    const void* host_a;
    const void* host_b;
    const void* host_add = NULL;
    void* host_out;
    pyc_dtype data_dtype;
    size_t elem_bytes;
    size_t a_bytes;
    size_t b_bytes;
    size_t c_bytes;
    size_t add_bytes = 0;
    int lhs_copy_required = 1;
    int rhs_copy_required = 1;
    int assume_static_lhs = env_true("PYC_CUDA_ASSUME_STATIC_LHS");
    int assume_static_rhs = env_true("PYC_CUDA_ASSUME_STATIC_RHS");
    int skip_output_copy = env_true("PYC_CUDA_SKIP_HOST_OUTPUT_COPY");
    int promoted_gemm_used = 0;
    int promoted_graph_replayed = 0;
    char promoted_symbol[PYC_KERNEL_SYMBOL_MAX];
    char promoted_failure_reason[PYC_CUDA_REASON_MAX];
    int graph_replayed = 0;
    int run_status;
    int ok = 0;

    if (!native_reason || native_reason_size == 0) {
        return -1;
    }
    native_reason[0] = '\0';

    if (parse_matmul_chain_graph(module, &spec) != 0) {
        return -1;
    }

    if (find_input_tensor_index(module, spec.lhs_id, &lhs_input_index) != 0 ||
        find_input_tensor_index(module, spec.rhs_id, &rhs_input_index) != 0) {
        return -1;
    }
    if (spec.add_id >= 0 &&
        find_input_tensor_index(module, spec.add_operand_id, &add_input_index) != 0) {
        return -1;
    }
    target_output_index = 0;
    {
        size_t op_idx;
        size_t output_seen = 0;
        for (op_idx = 0; op_idx < module->op_count; ++op_idx) {
            const pyc_ir_op* op = &module->ops[op_idx];
            if (op->kind != PYC_IR_OP_OUTPUT) {
                continue;
            }
            if ((int)op_idx == spec.output_id) {
                target_output_index = (int)output_seen;
                break;
            }
            output_seen++;
        }
    }

    if (lhs_input_index < 0 || rhs_input_index < 0 || target_output_index < 0) {
        return -1;
    }
    if ((size_t)lhs_input_index >= input_count ||
        (size_t)rhs_input_index >= input_count ||
        (size_t)target_output_index >= output_count) {
        return -1;
    }

    data_dtype = module->ops[(size_t)spec.matmul_id].dtype;
    elem_bytes = pyc_cuda_dtype_size_bytes(data_dtype);
    if (elem_bytes == 0) {
        return -1;
    }

    host_a = inputs[(size_t)lhs_input_index].data;
    host_b = inputs[(size_t)rhs_input_index].data;
    host_out = outputs[(size_t)target_output_index].data;
    if (!host_a || !host_b || (!host_out && !skip_output_copy)) {
        return -1;
    }
    if (inputs[(size_t)lhs_input_index].dtype != data_dtype ||
        inputs[(size_t)rhs_input_index].dtype != data_dtype ||
        outputs[(size_t)target_output_index].dtype != data_dtype) {
        return -1;
    }
    if (spec.add_id >= 0) {
        if (add_input_index < 0 || (size_t)add_input_index >= input_count) {
            return -1;
        }
        if (data_dtype != PYC_DTYPE_F32) {
            return -1;
        }
        host_add = inputs[(size_t)add_input_index].data;
        if (!host_add) {
            return -1;
        }
    }

    a_bytes = spec.m * spec.k * elem_bytes;
    b_bytes = spec.k * spec.n * elem_bytes;
    c_bytes = spec.m * spec.n * elem_bytes;
    add_bytes = spec.add_operand_elements * elem_bytes;
    if (inputs[(size_t)lhs_input_index].size_bytes < a_bytes ||
        inputs[(size_t)rhs_input_index].size_bytes < b_bytes ||
        outputs[(size_t)target_output_index].size_bytes < c_bytes) {
        return -1;
    }
    if (spec.add_id >= 0 &&
        inputs[(size_t)add_input_index].size_bytes < add_bytes) {
        return -1;
    }

    if (pyc_cuda_workspace_ensure(a_bytes, b_bytes, c_bytes, data_dtype) != 0) {
        return -1;
    }

    pyc_cutlass_registry_init();

    if (assume_static_lhs &&
        g_cuda_workspace.lhs_uploaded &&
        g_cuda_workspace.host_a_last_ptr == (const void*)host_a &&
        g_cuda_workspace.host_a_last_bytes == a_bytes) {
        lhs_copy_required = 0;
    }
    if (assume_static_rhs &&
        g_cuda_workspace.rhs_uploaded &&
        g_cuda_workspace.host_b_last_ptr == (const void*)host_b &&
        g_cuda_workspace.host_b_last_bytes == b_bytes) {
        rhs_copy_required = 0;
    }

    promoted_symbol[0] = '\0';
    promoted_failure_reason[0] = '\0';
    if (spec.add_id < 0 && spec.relu_id < 0) {
        if (data_dtype == PYC_DTYPE_BF16) {
            (void)select_cutlass_matmul_symbol(data_dtype, promoted_symbol, sizeof(promoted_symbol));
        } else if (data_dtype == PYC_DTYPE_F32 && env_true("PYC_CUDA_ENABLE_PROMOTED_GEMM")) {
            (void)pyc_kernel_promoted_symbol("matmul", PYC_BACKEND_CUDA, promoted_symbol, sizeof(promoted_symbol));
        }
        if (promoted_symbol[0] != '\0') {
            run_status = run_promoted_cutlass_gemm_pipeline_graph(
                promoted_symbol,
                host_a,
                host_b,
                host_out,
                a_bytes,
                b_bytes,
                c_bytes,
                spec.m,
                spec.k,
                spec.n,
                data_dtype,
                lhs_copy_required,
                rhs_copy_required,
                &promoted_graph_replayed,
                trace,
                promoted_failure_reason,
                sizeof(promoted_failure_reason));
            if (run_status != 0) {
                promoted_graph_replayed = 0;
                run_status = run_promoted_cutlass_gemm_pipeline(
                    promoted_symbol,
                    host_a,
                    host_b,
                    host_out,
                    a_bytes,
                    b_bytes,
                    c_bytes,
                    spec.m,
                    spec.k,
                    spec.n,
                    lhs_copy_required,
                    rhs_copy_required,
                    &promoted_gemm_used,
                    trace,
                    promoted_failure_reason,
                    sizeof(promoted_failure_reason));
            } else {
                promoted_gemm_used = 1;
            }
            if (run_status == 0) {
                if (lhs_copy_required) {
                    g_cuda_workspace.lhs_uploaded = 1;
                    g_cuda_workspace.host_a_last_ptr = (const void*)host_a;
                    g_cuda_workspace.host_a_last_bytes = a_bytes;
                }
                if (promoted_gemm_used && promoted_symbol[0] != '\0') {
                    const char* reason_prefix =
                        data_dtype == PYC_DTYPE_BF16 ?
                            (promoted_graph_replayed ? "cuda_cutlass_bf16_graph" : "cuda_cutlass_bf16") :
                            (promoted_graph_replayed ? "cuda_promoted_gemm_graph" : "cuda_promoted_gemm");
                    snprintf(
                        native_reason,
                        native_reason_size,
                        "%s:%s",
                        reason_prefix,
                        promoted_symbol);
                } else {
                    strncpy(
                        native_reason,
                        data_dtype == PYC_DTYPE_BF16 ?
                            (promoted_graph_replayed ? "cuda_cutlass_bf16_graph" : "cuda_cutlass_bf16") :
                            (promoted_graph_replayed ? "cuda_promoted_gemm_graph" : "cuda_promoted_gemm"),
                        native_reason_size - 1);
                    native_reason[native_reason_size - 1] = '\0';
                }
                if (rhs_copy_required) {
                    g_cuda_workspace.rhs_uploaded = 1;
                    g_cuda_workspace.host_b_last_ptr = (const void*)host_b;
                    g_cuda_workspace.host_b_last_bytes = b_bytes;
                }
                ok = 1;
                goto cleanup;
            }
        }
    }

    run_status = run_cuda_matmul_pipeline_graph(
        host_a,
        host_b,
        host_out,
        a_bytes,
        b_bytes,
        c_bytes,
        spec.m,
        spec.k,
        spec.n,
        data_dtype,
        lhs_copy_required,
        rhs_copy_required,
        &graph_replayed,
        trace,
        native_reason,
        native_reason_size);
    if (run_status != 0) {
        graph_replayed = 0;
        run_status = run_cuda_matmul_pipeline(
            host_a,
            host_b,
            host_out,
            a_bytes,
            b_bytes,
            c_bytes,
            spec.m,
            spec.k,
            spec.n,
            data_dtype,
            lhs_copy_required,
            rhs_copy_required,
            trace,
            native_reason,
            native_reason_size);
    }
    if (run_status != 0) {
        if (promoted_failure_reason[0] != '\0' && native_reason[0] != '\0') {
            char combined_reason[PYC_CUDA_REASON_MAX];
            snprintf(combined_reason, sizeof(combined_reason), "%s;%s", promoted_failure_reason, native_reason);
            strncpy(native_reason, combined_reason, native_reason_size - 1);
            native_reason[native_reason_size - 1] = '\0';
        } else if (promoted_failure_reason[0] != '\0' && native_reason[0] == '\0') {
            strncpy(native_reason, promoted_failure_reason, native_reason_size - 1);
            native_reason[native_reason_size - 1] = '\0';
        }
        goto cleanup;
    }
    if (lhs_copy_required) {
        g_cuda_workspace.lhs_uploaded = 1;
        g_cuda_workspace.host_a_last_ptr = (const void*)host_a;
        g_cuda_workspace.host_a_last_bytes = a_bytes;
    }
    if (rhs_copy_required) {
        g_cuda_workspace.rhs_uploaded = 1;
        g_cuda_workspace.host_b_last_ptr = (const void*)host_b;
        g_cuda_workspace.host_b_last_bytes = b_bytes;
    }

    if (!skip_output_copy && spec.add_id >= 0) {
        apply_add_epilogue((float*)host_out, (const float*)host_add, spec.m, spec.n, spec.add_mode);
    }
    if (!skip_output_copy && spec.relu_id >= 0) {
        apply_relu_epilogue((float*)host_out, spec.m * spec.n);
    }

    if (graph_replayed) {
        strncpy(native_reason, data_dtype == PYC_DTYPE_F32 ? "cuda_native_cublas_graph_replay" : "cuda_native_half_graph_replay", native_reason_size - 1);
    } else if (env_default_true("PYC_CUDA_ENABLE_GRAPH_REPLAY")) {
        strncpy(native_reason, data_dtype == PYC_DTYPE_F32 ? "cuda_native_cublas_graph_capture" : "cuda_native_half_graph_capture", native_reason_size - 1);
    } else {
        strncpy(native_reason, data_dtype == PYC_DTYPE_F32 ? "cuda_native_cublas" : "cuda_native_half", native_reason_size - 1);
    }
    native_reason[native_reason_size - 1] = '\0';
    ok = 1;

cleanup:
    if (!ok) {
        pyc_cuda_workspace_release();
    }
    return ok ? 0 : -1;
}
#endif

static pyc_cuda_dispatch_status pyc_cuda_dispatch_impl(
    const pyc_ir_module* module,
    const pyc_tensor* inputs,
    size_t input_count,
    pyc_tensor* outputs,
    size_t output_count,
    pyc_cpu_executor_fn cpu_executor,
    void* executor_ctx,
    pyc_cuda_dispatch_trace* trace) {
    int available;
    char reason[PYC_CUDA_REASON_MAX];
    char native_reason[PYC_CUDA_REASON_MAX];
    int run_status;

    if (!module || !inputs || !outputs || !cpu_executor) {
        return PYC_CUDA_DISPATCH_ERROR;
    }

    pyc_cuda_dispatch_trace_init(trace);
    if (trace) {
        trace->cuda_requested = 1;
    }

    if (env_true("PYC_CUDA_FORCE_ERROR")) {
        if (trace) {
            trace->cuda_available = 0;
            trace->fallback_to_cpu = 0;
            strcpy(trace->reason, "forced_error");
        }
        return PYC_CUDA_DISPATCH_ERROR;
    }

    memset(reason, 0, sizeof(reason));
    memset(native_reason, 0, sizeof(native_reason));
    available = detect_cuda_available(reason, sizeof(reason));
    if (trace) {
        trace->cuda_available = available;
    }

    if (available && !env_true("PYC_CUDA_FORCE_FALLBACK") &&
        native_cuda_can_execute(module)) {
#if defined(PYC_HAVE_CUDA_RUNTIME)
        run_status = execute_native_cuda_graph_cuda(
            module,
            inputs,
            input_count,
            outputs,
            output_count,
            trace,
            native_reason,
            sizeof(native_reason));
        if (run_status == 0) {
            if (trace) {
                trace->fallback_to_cpu = 0;
                if (native_reason[0] != '\0') {
                    strncpy(trace->reason, native_reason, sizeof(trace->reason) - 1);
                } else {
                    strncpy(trace->reason, "cuda_native_cublas", sizeof(trace->reason) - 1);
                }
            }
            return PYC_CUDA_DISPATCH_OK;
        }
#endif
        if (env_true("PYC_CUDA_SIMULATE_AVAILABLE")) {
            run_status = execute_native_cuda_graph_simulated(module, inputs, input_count, outputs, output_count);
            if (run_status == 0) {
                if (trace) {
                    trace->fallback_to_cpu = 0;
                    strncpy(trace->reason, "cuda_native_simulated", sizeof(trace->reason) - 1);
                }
                return PYC_CUDA_DISPATCH_OK;
            }
        }
    }

    run_status = cpu_executor(module, inputs, input_count, outputs, output_count, executor_ctx);
    if (run_status != 0) {
        if (trace) {
            trace->fallback_to_cpu = 0;
            if (native_reason[0] != '\0') {
                snprintf(trace->reason, sizeof(trace->reason), "%s;cpu_fallback_failed", native_reason);
            } else {
                strncpy(trace->reason, "cpu_fallback_failed", sizeof(trace->reason) - 1);
            }
        }
        return PYC_CUDA_DISPATCH_ERROR;
    }

    if (trace) {
        trace->fallback_to_cpu = 1;
        if (env_true("PYC_CUDA_FORCE_FALLBACK")) {
            strncpy(trace->reason, "cuda_forced_fallback", sizeof(trace->reason) - 1);
        } else if (available && !native_cuda_can_execute(module)) {
            strncpy(trace->reason, "cuda_native_unsupported_fallback", sizeof(trace->reason) - 1);
        } else if (available) {
            strncpy(trace->reason, "cuda_native_failed_fallback", sizeof(trace->reason) - 1);
        } else {
            strncpy(trace->reason, reason, sizeof(trace->reason) - 1);
        }
    }
    return PYC_CUDA_DISPATCH_FALLBACK;
}

pyc_cuda_dispatch_status pyc_cuda_dispatch(
    const pyc_ir_module* module,
    const pyc_tensor* inputs,
    size_t input_count,
    pyc_tensor* outputs,
    size_t output_count,
    pyc_cpu_executor_fn cpu_executor,
    void* executor_ctx,
    pyc_cuda_dispatch_trace* trace) {
    pyc_cuda_dispatch_status status;
    /* Serialize access to the shared GPU workspace across threads. */
    pyc_mutex_lock(&g_cuda_dispatch_mutex);
    status = pyc_cuda_dispatch_impl(
        module, inputs, input_count, outputs, output_count,
        cpu_executor, executor_ctx, trace);
    pyc_mutex_unlock(&g_cuda_dispatch_mutex);
    return status;
}
