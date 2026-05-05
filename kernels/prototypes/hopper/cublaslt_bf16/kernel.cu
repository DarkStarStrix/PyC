#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int m;
    int n;
    int k;
    int warmup;
    int iters;
    int skip_reference;
} pyc_hopper_cublaslt_bf16_config;

static int parse_int_arg(const char* text, int* out_value) {
    char* end = NULL;
    long parsed;
    if (!text || !out_value) {
        return -1;
    }
    parsed = strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < 0 || parsed > INT32_MAX) {
        return -1;
    }
    *out_value = (int)parsed;
    return 0;
}

static int env_flag(const char* name, int default_value) {
    const char* raw = getenv(name);
    if (!raw || raw[0] == '\0') {
        return default_value;
    }
    if (
        strcmp(raw, "1") == 0 || strcmp(raw, "true") == 0 || strcmp(raw, "TRUE") == 0
        || strcmp(raw, "yes") == 0 || strcmp(raw, "on") == 0) {
        return 1;
    }
    if (
        strcmp(raw, "0") == 0 || strcmp(raw, "false") == 0 || strcmp(raw, "FALSE") == 0
        || strcmp(raw, "no") == 0 || strcmp(raw, "off") == 0) {
        return 0;
    }
    return default_value;
}

static int check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
        return -1;
    }
    return 0;
}

static int check_cublas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s failed: cublas status %d\n", what, (int)status);
        return -1;
    }
    return 0;
}

static void fill_matrix(__nv_bfloat16* data, int rows, int cols, float scale) {
    int i;
    for (i = 0; i < rows * cols; ++i) {
        int pattern = (i * 23 + rows * 13 + cols * 5) % 31;
        data[i] = __float2bfloat16(((float)pattern - 15.0f) * scale);
    }
}

static void reference_gemm(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* d,
    int m,
    int n,
    int k) {
    int row;
    for (row = 0; row < m; ++row) {
        int col;
        for (col = 0; col < n; ++col) {
            float acc = 0.0f;
            int kk;
            for (kk = 0; kk < k; ++kk) {
                acc += __bfloat162float(a[row * k + kk]) * __bfloat162float(b[kk * n + col]);
            }
            d[row * n + col] = __float2bfloat16(acc);
        }
    }
}

static int parse_config(int argc, char** argv, pyc_hopper_cublaslt_bf16_config* cfg) {
    if (!cfg) {
        return -1;
    }

    cfg->m = 4096;
    cfg->n = 4096;
    cfg->k = 4096;
    cfg->warmup = 3;
    cfg->iters = 30;
    cfg->skip_reference = env_flag("PYC_HOPPER_CUBLASLT_SKIP_REFERENCE", 1);

    if (argc > 1 && parse_int_arg(argv[1], &cfg->m) != 0) return -1;
    if (argc > 2 && parse_int_arg(argv[2], &cfg->n) != 0) return -1;
    if (argc > 3 && parse_int_arg(argv[3], &cfg->k) != 0) return -1;
    if (argc > 4 && parse_int_arg(argv[4], &cfg->warmup) != 0) return -1;
    if (argc > 5 && parse_int_arg(argv[5], &cfg->iters) != 0) return -1;
    if (argc > 6 && parse_int_arg(argv[6], &cfg->skip_reference) != 0) return -1;

    return 0;
}

int main(int argc, char** argv) {
    pyc_hopper_cublaslt_bf16_config cfg;
    struct cudaDeviceProp props;
    cublasLtHandle_t lt_handle = NULL;
    cublasLtMatmulDesc_t op_desc = NULL;
    cublasLtMatrixLayout_t a_layout = NULL;
    cublasLtMatrixLayout_t b_layout = NULL;
    cublasLtMatrixLayout_t c_layout = NULL;
    cublasLtMatmulPreference_t pref = NULL;
    cublasLtMatmulHeuristicResult_t heuristic;
    int returned_results = 0;
    cudaEvent_t start = NULL;
    cudaEvent_t stop = NULL;
    cudaStream_t stream = NULL;
    __nv_bfloat16* host_a = NULL;
    __nv_bfloat16* host_b = NULL;
    __nv_bfloat16* host_d = NULL;
    __nv_bfloat16* ref_d = NULL;
    __nv_bfloat16* dev_a = NULL;
    __nv_bfloat16* dev_b = NULL;
    __nv_bfloat16* dev_d = NULL;
    void* workspace = NULL;
    size_t workspace_bytes = 64u * 1024u * 1024u;
    size_t a_bytes;
    size_t b_bytes;
    size_t d_bytes;
    float alpha = 1.0f;
    float beta = 0.0f;
    float elapsed_ms = 0.0f;
    double best_ms = 0.0;
    double max_abs_diff = 0.0;
    int iter;

    if (parse_config(argc, argv, &cfg) != 0) {
        fprintf(stderr, "usage: %s [m] [n] [k] [warmup] [iters] [skip_reference]\n", argv[0]);
        return 2;
    }

    if (check_cuda(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties") != 0) {
        return 1;
    }
    if (props.major < 9) {
        fprintf(stderr, "Hopper cuBLASLt BF16 prototype requires sm_90-class hardware\n");
        return 1;
    }

    a_bytes = (size_t)cfg.m * (size_t)cfg.k * sizeof(__nv_bfloat16);
    b_bytes = (size_t)cfg.k * (size_t)cfg.n * sizeof(__nv_bfloat16);
    d_bytes = (size_t)cfg.m * (size_t)cfg.n * sizeof(__nv_bfloat16);

    host_a = (__nv_bfloat16*)malloc(a_bytes);
    host_b = (__nv_bfloat16*)malloc(b_bytes);
    if (!host_a || !host_b) {
        fprintf(stderr, "host allocation failed\n");
        return 1;
    }
    if (!cfg.skip_reference) {
        host_d = (__nv_bfloat16*)malloc(d_bytes);
        ref_d = (__nv_bfloat16*)malloc(d_bytes);
        if (!host_d || !ref_d) {
            fprintf(stderr, "host validation allocation failed\n");
            return 1;
        }
    }

    fill_matrix(host_a, cfg.m, cfg.k, 0.03125f);
    fill_matrix(host_b, cfg.k, cfg.n, 0.0625f);
    if (!cfg.skip_reference) {
        reference_gemm(host_a, host_b, ref_d, cfg.m, cfg.n, cfg.k);
    }

    if (check_cuda(cudaMalloc((void**)&dev_a, a_bytes), "cudaMalloc(a)") != 0) return 1;
    if (check_cuda(cudaMalloc((void**)&dev_b, b_bytes), "cudaMalloc(b)") != 0) return 1;
    if (check_cuda(cudaMalloc((void**)&dev_d, d_bytes), "cudaMalloc(d)") != 0) return 1;
    if (check_cuda(cudaMalloc(&workspace, workspace_bytes), "cudaMalloc(workspace)") != 0) return 1;

    if (check_cuda(cudaMemcpy(dev_a, host_a, a_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(a)") != 0) return 1;
    if (check_cuda(cudaMemcpy(dev_b, host_b, b_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(b)") != 0) return 1;
    if (check_cuda(cudaMemset(dev_d, 0, d_bytes), "cudaMemset(d)") != 0) return 1;

    if (check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags") != 0) return 1;
    if (check_cublas(cublasLtCreate(&lt_handle), "cublasLtCreate") != 0) return 1;

    if (check_cublas(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "cublasLtMatmulDescCreate") != 0) return 1;
    {
        cublasOperation_t transa = CUBLAS_OP_N;
        cublasOperation_t transb = CUBLAS_OP_N;
        if (check_cublas(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)), "cublasLtMatmulDescSetAttribute(TRANSA)") != 0) return 1;
        if (check_cublas(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)), "cublasLtMatmulDescSetAttribute(TRANSB)") != 0) return 1;
    }

    if (check_cublas(cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_16BF, cfg.m, cfg.k, cfg.k), "cublasLtMatrixLayoutCreate(A)") != 0) return 1;
    if (check_cublas(cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_16BF, cfg.k, cfg.n, cfg.n), "cublasLtMatrixLayoutCreate(B)") != 0) return 1;
    if (check_cublas(cublasLtMatrixLayoutCreate(&c_layout, CUDA_R_16BF, cfg.m, cfg.n, cfg.n), "cublasLtMatrixLayoutCreate(D)") != 0) return 1;
    {
        cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
        if (check_cublas(cublasLtMatrixLayoutSetAttribute(a_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)), "cublasLtMatrixLayoutSetAttribute(A order)") != 0) return 1;
        if (check_cublas(cublasLtMatrixLayoutSetAttribute(b_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)), "cublasLtMatrixLayoutSetAttribute(B order)") != 0) return 1;
        if (check_cublas(cublasLtMatrixLayoutSetAttribute(c_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)), "cublasLtMatrixLayoutSetAttribute(D order)") != 0) return 1;
    }

    if (check_cublas(cublasLtMatmulPreferenceCreate(&pref), "cublasLtMatmulPreferenceCreate") != 0) return 1;
    if (check_cublas(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)), "cublasLtMatmulPreferenceSetAttribute") != 0) return 1;
    if (check_cublas(cublasLtMatmulAlgoGetHeuristic(lt_handle, op_desc, a_layout, b_layout, c_layout, c_layout, pref, 1, &heuristic, &returned_results), "cublasLtMatmulAlgoGetHeuristic") != 0) return 1;
    if (returned_results <= 0) {
        fprintf(stderr, "cublasLtMatmulAlgoGetHeuristic returned no algorithms\n");
        return 1;
    }

    if (check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)") != 0) return 1;
    if (check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)") != 0) return 1;

    for (iter = 0; iter < cfg.warmup; ++iter) {
        if (check_cublas(
                cublasLtMatmul(
                    lt_handle,
                    op_desc,
                    &alpha,
                    dev_a,
                    a_layout,
                    dev_b,
                    b_layout,
                    &beta,
                    dev_d,
                    c_layout,
                    dev_d,
                    c_layout,
                    &heuristic.algo,
                    workspace,
                    workspace_bytes,
                    stream),
                "cublasLtMatmul(warmup)")
            != 0) return 1;
    }
    if (check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(warmup)") != 0) return 1;

    for (iter = 0; iter < cfg.iters; ++iter) {
        if (check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(start)") != 0) return 1;
        if (check_cublas(
                cublasLtMatmul(
                    lt_handle,
                    op_desc,
                    &alpha,
                    dev_a,
                    a_layout,
                    dev_b,
                    b_layout,
                    &beta,
                    dev_d,
                    c_layout,
                    dev_d,
                    c_layout,
                    &heuristic.algo,
                    workspace,
                    workspace_bytes,
                    stream),
                "cublasLtMatmul(bench)")
            != 0) return 1;
        if (check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop)") != 0) return 1;
        if (check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)") != 0) return 1;
        if (check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime") != 0) return 1;
        if (iter == 0 || elapsed_ms < (float)best_ms) {
            best_ms = elapsed_ms;
        }
    }

    if (!cfg.skip_reference) {
        int idx;
        if (check_cuda(cudaMemcpy(host_d, dev_d, d_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(d)") != 0) return 1;
        for (idx = 0; idx < cfg.m * cfg.n; ++idx) {
            double diff = fabs((double)__bfloat162float(host_d[idx]) - (double)__bfloat162float(ref_d[idx]));
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
            }
        }
    }

    printf("kernel=hopper_cublaslt_bf16\n");
    printf("arch=sm%d%d\n", props.major, props.minor);
    printf("shape=%dx%dx%d\n", cfg.m, cfg.n, cfg.k);
    printf("workspace_bytes=%zu\n", workspace_bytes);
    printf("heuristic_workspace_bytes=%zu\n", heuristic.workspaceSize);
    printf("skip_reference=%d\n", cfg.skip_reference);
    printf("best_ms=%.3f\n", best_ms);
    if (!cfg.skip_reference) {
        printf("max_abs_diff=%.6f\n", max_abs_diff);
    }
    if (best_ms > 0.0) {
        double flops = 2.0 * (double)cfg.m * (double)cfg.n * (double)cfg.k;
        double gflops = flops / (best_ms * 1.0e6);
        printf("gflops=%.3f\n", gflops);
        printf("tflops=%.3f\n", gflops / 1000.0);
    }

    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
    if (stream) cudaStreamDestroy(stream);
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (c_layout) cublasLtMatrixLayoutDestroy(c_layout);
    if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
    if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
    if (op_desc) cublasLtMatmulDescDestroy(op_desc);
    if (lt_handle) cublasLtDestroy(lt_handle);
    if (workspace) cudaFree(workspace);
    if (dev_d) cudaFree(dev_d);
    if (dev_b) cudaFree(dev_b);
    if (dev_a) cudaFree(dev_a);
    free(ref_d);
    free(host_d);
    free(host_b);
    free(host_a);
    return (cfg.skip_reference || max_abs_diff <= 0.25) ? 0 : 1;
}
