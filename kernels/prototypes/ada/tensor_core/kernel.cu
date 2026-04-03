#include <cuda_runtime.h>
#include <mma.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(PYC_ADA_TENSOR_CORE_USE_BF16) && PYC_ADA_TENSOR_CORE_USE_BF16
#include <cuda_bf16.h>
typedef __nv_bfloat16 pyc_tc_scalar_t;
#define PYC_TC_LANE_NAME "bf16"
static __host__ __device__ inline pyc_tc_scalar_t pyc_tc_make_scalar(float value) {
    return __float2bfloat16(value);
}
static __host__ __device__ inline float pyc_tc_scalar_to_float(pyc_tc_scalar_t value) {
    return __bfloat162float(value);
}
#else
#include <cuda_fp16.h>
typedef half pyc_tc_scalar_t;
#define PYC_TC_LANE_NAME "fp16"
static __host__ __device__ inline pyc_tc_scalar_t pyc_tc_make_scalar(float value) {
    return __float2half(value);
}
static __host__ __device__ inline float pyc_tc_scalar_to_float(pyc_tc_scalar_t value) {
    return __half2float(value);
}
#endif

namespace wmma = nvcuda::wmma;

#define PYC_TC_CTA_M 32
#define PYC_TC_CTA_N 64
#define PYC_TC_CTA_K 16
#define PYC_TC_WARPS_PER_BLOCK 8
#define PYC_TC_THREADS_PER_BLOCK 256

typedef struct {
    int m;
    int n;
    int k;
    int warmup;
    int iters;
} pyc_tc_config;

static int check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
        return -1;
    }
    return 0;
}

static int parse_int_arg(const char* text, int* out_value) {
    char* end = NULL;
    long parsed;
    if (!text || !out_value) {
        return -1;
    }
    parsed = strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed <= 0 || parsed > INT32_MAX) {
        return -1;
    }
    *out_value = (int)parsed;
    return 0;
}

static void fill_matrix(pyc_tc_scalar_t* data, int rows, int cols, float scale) {
    int i;
    for (i = 0; i < rows * cols; ++i) {
        int pattern = (i * 19 + rows * 11 + cols * 7) % 29;
        data[i] = pyc_tc_make_scalar(((float)pattern - 14.0f) * scale);
    }
}

static void reference_gemm(
    const pyc_tc_scalar_t* a,
    const pyc_tc_scalar_t* b,
    float* c,
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
                acc += pyc_tc_scalar_to_float(a[row * k + kk]) * pyc_tc_scalar_to_float(b[kk * n + col]);
            }
            c[row * n + col] = acc;
        }
    }
}

__launch_bounds__(PYC_TC_THREADS_PER_BLOCK, 2)
__global__ void pyc_tc_gemm_kernel(
    const pyc_tc_scalar_t* __restrict__ a,
    const pyc_tc_scalar_t* __restrict__ b,
    float* __restrict__ c,
    int m,
    int n,
    int k) {
    __shared__ pyc_tc_scalar_t shared_a[PYC_TC_CTA_M][PYC_TC_CTA_K];
    __shared__ pyc_tc_scalar_t shared_b[PYC_TC_CTA_N][PYC_TC_CTA_K];

    const int warp_id = threadIdx.x / 32;
    const int block_row = blockIdx.y * PYC_TC_CTA_M;
    const int block_col = blockIdx.x * PYC_TC_CTA_N;
    const int warp_row = (warp_id / 4) * 16;
    const int warp_col = (warp_id % 4) * 16;
    const int c_row = block_row + warp_row;
    const int c_col = block_col + warp_col;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    if (warp_id >= PYC_TC_WARPS_PER_BLOCK) {
        return;
    }

    for (int kk = 0; kk < k; kk += PYC_TC_CTA_K) {
        int idx;

        for (idx = threadIdx.x; idx < PYC_TC_CTA_M * PYC_TC_CTA_K; idx += blockDim.x) {
            const int row = idx / PYC_TC_CTA_K;
            const int col = idx % PYC_TC_CTA_K;
            const int g_row = block_row + row;
            const int g_col = kk + col;
            if (g_row < m && g_col < k) {
                shared_a[row][col] = a[g_row * k + g_col];
            } else {
                shared_a[row][col] = pyc_tc_make_scalar(0.0f);
            }
        }

        for (idx = threadIdx.x; idx < PYC_TC_CTA_N * PYC_TC_CTA_K; idx += blockDim.x) {
            const int row = idx / PYC_TC_CTA_K;
            const int col = idx % PYC_TC_CTA_K;
            const int g_row = kk + col;
            const int g_col = block_col + row;
            if (g_row < k && g_col < n) {
                shared_b[row][col] = b[g_row * n + g_col];
            } else {
                shared_b[row][col] = pyc_tc_make_scalar(0.0f);
            }
        }

        __syncthreads();

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, pyc_tc_scalar_t, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, pyc_tc_scalar_t, wmma::col_major> b_frag;
            wmma::load_matrix_sync(a_frag, &shared_a[warp_row][0], PYC_TC_CTA_K);
            wmma::load_matrix_sync(b_frag, &shared_b[warp_col][0], PYC_TC_CTA_K);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }

        __syncthreads();
    }

    if (c_row < m && c_col < n) {
        wmma::store_matrix_sync(&c[c_row * n + c_col], acc, n, wmma::mem_row_major);
    }
}

static int set_kernel_attributes(void) {
    cudaError_t status;

    status = cudaFuncSetAttribute(
        pyc_tc_gemm_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);
    if (status != cudaSuccess && status != cudaErrorNotSupported) {
        fprintf(stderr, "cudaFuncSetAttribute failed: %s\n", cudaGetErrorString(status));
        return -1;
    }

    return 0;
}

static int parse_config(int argc, char** argv, pyc_tc_config* cfg) {
    if (!cfg) {
        return -1;
    }

    cfg->m = 1024;
    cfg->n = 1024;
    cfg->k = 1024;
    cfg->warmup = 10;
    cfg->iters = 50;

    if (argc > 1 && parse_int_arg(argv[1], &cfg->m) != 0) return -1;
    if (argc > 2 && parse_int_arg(argv[2], &cfg->n) != 0) return -1;
    if (argc > 3 && parse_int_arg(argv[3], &cfg->k) != 0) return -1;
    if (argc > 4 && parse_int_arg(argv[4], &cfg->warmup) != 0) return -1;
    if (argc > 5 && parse_int_arg(argv[5], &cfg->iters) != 0) return -1;

    return 0;
}

int main(int argc, char** argv) {
    pyc_tc_config cfg;
    cudaDeviceProp props;
    pyc_tc_scalar_t* host_a = NULL;
    pyc_tc_scalar_t* host_b = NULL;
    float* host_c = NULL;
    float* ref_c = NULL;
    pyc_tc_scalar_t* dev_a = NULL;
    pyc_tc_scalar_t* dev_b = NULL;
    float* dev_c = NULL;
    cudaEvent_t start = NULL;
    cudaEvent_t stop = NULL;
    size_t a_bytes;
    size_t b_bytes;
    size_t c_bytes;
    dim3 block;
    dim3 grid;
    float elapsed_ms = 0.0f;
    double best_ms = 0.0;
    int iter;
    double max_abs_diff = 0.0;

    if (parse_config(argc, argv, &cfg) != 0) {
        fprintf(stderr, "usage: %s [m] [n] [k] [warmup] [iters]\n", argv[0]);
        return 2;
    }

    if ((cfg.m % PYC_TC_CTA_M) != 0 || (cfg.n % PYC_TC_CTA_N) != 0 || (cfg.k % PYC_TC_CTA_K) != 0) {
        fprintf(stderr, "Tensor Core lane requires %dx%dx%d-aligned shapes\n", PYC_TC_CTA_M, PYC_TC_CTA_N, PYC_TC_CTA_K);
        return 2;
    }

    if (check_cuda(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties") != 0) {
        return 1;
    }
    if (props.major < 8 || (props.major == 8 && props.minor < 9)) {
        fprintf(stderr, "Ada Tensor Core prototype requires sm_89-class hardware\n");
        return 1;
    }

    a_bytes = (size_t)cfg.m * (size_t)cfg.k * sizeof(pyc_tc_scalar_t);
    b_bytes = (size_t)cfg.k * (size_t)cfg.n * sizeof(pyc_tc_scalar_t);
    c_bytes = (size_t)cfg.m * (size_t)cfg.n * sizeof(float);

    host_a = (pyc_tc_scalar_t*)malloc(a_bytes);
    host_b = (pyc_tc_scalar_t*)malloc(b_bytes);
    host_c = (float*)malloc(c_bytes);
    ref_c = (float*)malloc(c_bytes);
    if (!host_a || !host_b || !host_c || !ref_c) {
        fprintf(stderr, "host allocation failed\n");
        return 1;
    }

    fill_matrix(host_a, cfg.m, cfg.k, 0.03125f);
    fill_matrix(host_b, cfg.k, cfg.n, 0.0625f);
    reference_gemm(host_a, host_b, ref_c, cfg.m, cfg.n, cfg.k);

    if (check_cuda(cudaMalloc((void**)&dev_a, a_bytes), "cudaMalloc(a)") != 0) return 1;
    if (check_cuda(cudaMalloc((void**)&dev_b, b_bytes), "cudaMalloc(b)") != 0) return 1;
    if (check_cuda(cudaMalloc((void**)&dev_c, c_bytes), "cudaMalloc(c)") != 0) return 1;

    if (check_cuda(cudaMemcpy(dev_a, host_a, a_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(a)") != 0) return 1;
    if (check_cuda(cudaMemcpy(dev_b, host_b, b_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(b)") != 0) return 1;

    if (set_kernel_attributes() != 0) return 1;

    block = dim3(PYC_TC_THREADS_PER_BLOCK, 1, 1);
    grid = dim3(
        (unsigned int)((cfg.n + PYC_TC_CTA_N - 1) / PYC_TC_CTA_N),
        (unsigned int)((cfg.m + PYC_TC_CTA_M - 1) / PYC_TC_CTA_M),
        1);

    if (check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)") != 0) return 1;
    if (check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)") != 0) return 1;

    for (iter = 0; iter < cfg.warmup; ++iter) {
        pyc_tc_gemm_kernel<<<grid, block>>>(dev_a, dev_b, dev_c, cfg.m, cfg.n, cfg.k);
    }
    if (check_cuda(cudaGetLastError(), "kernel launch warmup") != 0) return 1;
    if (check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup") != 0) return 1;

    best_ms = 0.0;
    for (iter = 0; iter < cfg.iters; ++iter) {
        if (check_cuda(cudaEventRecord(start), "cudaEventRecord(start)") != 0) return 1;
        pyc_tc_gemm_kernel<<<grid, block>>>(dev_a, dev_b, dev_c, cfg.m, cfg.n, cfg.k);
        if (check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)") != 0) return 1;
        if (check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)") != 0) return 1;
        if (check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime") != 0) return 1;
        if (iter == 0 || elapsed_ms < (float)best_ms) {
            best_ms = elapsed_ms;
        }
    }

    if (check_cuda(cudaMemcpy(host_c, dev_c, c_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(c)") != 0) return 1;

    for (iter = 0; iter < cfg.m * cfg.n; ++iter) {
        double diff = fabs((double)host_c[iter] - (double)ref_c[iter]);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
        }
    }

    printf("lane=%s\n", PYC_TC_LANE_NAME);
    printf("shape=%dx%dx%d\n", cfg.m, cfg.n, cfg.k);
    printf("tile=%dx%dx%d threads=%d\n", PYC_TC_CTA_M, PYC_TC_CTA_N, PYC_TC_CTA_K, PYC_TC_THREADS_PER_BLOCK);
    printf("best_ms=%.3f\n", best_ms);
    printf("max_abs_diff=%.6f\n", max_abs_diff);
    if (best_ms > 0.0) {
        double flops = 2.0 * (double)cfg.m * (double)cfg.n * (double)cfg.k;
        double gflops = flops / (best_ms * 1.0e6);
        printf("gflops=%.3f\n", gflops);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(host_a);
    free(host_b);
    free(host_c);
    free(ref_c);
    return max_abs_diff <= 0.2 ? 0 : 1;
}
