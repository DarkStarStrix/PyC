#include <cuda_runtime.h>
#include <mma.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(PYC_HOPPER_TENSOR_CORE_USE_BF16) && PYC_HOPPER_TENSOR_CORE_USE_BF16
#include <cuda_bf16.h>
typedef __nv_bfloat16 pyc_hopper_tc_scalar_t;
#define PYC_HOPPER_TC_LANE_NAME "bf16"
static __host__ __device__ inline pyc_hopper_tc_scalar_t pyc_hopper_tc_make_scalar(float value) {
    return __float2bfloat16(value);
}
static __host__ __device__ inline float pyc_hopper_tc_scalar_to_float(pyc_hopper_tc_scalar_t value) {
    return __bfloat162float(value);
}
#else
#include <cuda_fp16.h>
typedef half pyc_hopper_tc_scalar_t;
#define PYC_HOPPER_TC_LANE_NAME "fp16"
static __host__ __device__ inline pyc_hopper_tc_scalar_t pyc_hopper_tc_make_scalar(float value) {
    return __float2half(value);
}
static __host__ __device__ inline float pyc_hopper_tc_scalar_to_float(pyc_hopper_tc_scalar_t value) {
    return __half2float(value);
}
#endif

namespace wmma = nvcuda::wmma;

#ifndef PYC_HOPPER_TC_MMA_TILE_M
#define PYC_HOPPER_TC_MMA_TILE_M 16
#endif

#ifndef PYC_HOPPER_TC_MMA_TILE_N
#define PYC_HOPPER_TC_MMA_TILE_N 16
#endif

#ifndef PYC_HOPPER_TC_MMA_TILE_K
#define PYC_HOPPER_TC_MMA_TILE_K 16
#endif

#ifndef PYC_HOPPER_TC_WARP_ROW_TILES
#define PYC_HOPPER_TC_WARP_ROW_TILES 1
#endif

#ifndef PYC_HOPPER_TC_WARP_COL_TILES
#define PYC_HOPPER_TC_WARP_COL_TILES 1
#endif

#ifndef PYC_HOPPER_TC_WARP_ROW_GROUPS
#define PYC_HOPPER_TC_WARP_ROW_GROUPS 4
#endif

#ifndef PYC_HOPPER_TC_WARP_COL_GROUPS
#define PYC_HOPPER_TC_WARP_COL_GROUPS 4
#endif

#ifndef PYC_HOPPER_TC_TILE_K
#define PYC_HOPPER_TC_TILE_K 16
#endif

#ifndef PYC_HOPPER_TC_SHARED_PAD_A
#define PYC_HOPPER_TC_SHARED_PAD_A 0
#endif

#ifndef PYC_HOPPER_TC_SHARED_PAD_B
#define PYC_HOPPER_TC_SHARED_PAD_B 0
#endif

#define PYC_HOPPER_TC_WARP_TILE_M (PYC_HOPPER_TC_MMA_TILE_M * PYC_HOPPER_TC_WARP_ROW_TILES)
#define PYC_HOPPER_TC_WARP_TILE_N (PYC_HOPPER_TC_MMA_TILE_N * PYC_HOPPER_TC_WARP_COL_TILES)
#define PYC_HOPPER_TC_TILE_M (PYC_HOPPER_TC_WARP_TILE_M * PYC_HOPPER_TC_WARP_ROW_GROUPS)
#define PYC_HOPPER_TC_TILE_N (PYC_HOPPER_TC_WARP_TILE_N * PYC_HOPPER_TC_WARP_COL_GROUPS)
#define PYC_HOPPER_TC_WARPS_PER_BLOCK (PYC_HOPPER_TC_WARP_ROW_GROUPS * PYC_HOPPER_TC_WARP_COL_GROUPS)
#define PYC_HOPPER_TC_THREADS_PER_BLOCK (PYC_HOPPER_TC_WARPS_PER_BLOCK * 32)
#define PYC_HOPPER_TC_SHARED_STRIDE_A (PYC_HOPPER_TC_TILE_K + PYC_HOPPER_TC_SHARED_PAD_A)
#define PYC_HOPPER_TC_SHARED_STRIDE_B (PYC_HOPPER_TC_TILE_K + PYC_HOPPER_TC_SHARED_PAD_B)

typedef struct {
    int m;
    int n;
    int k;
    int warmup;
    int iters;
    int skip_reference;
} pyc_hopper_tc_config;

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

static void fill_matrix(pyc_hopper_tc_scalar_t* data, int rows, int cols, float scale) {
    int i;
    for (i = 0; i < rows * cols; ++i) {
        int pattern = (i * 23 + rows * 13 + cols * 5) % 31;
        data[i] = pyc_hopper_tc_make_scalar(((float)pattern - 15.0f) * scale);
    }
}

static void reference_gemm(
    const pyc_hopper_tc_scalar_t* a,
    const pyc_hopper_tc_scalar_t* b,
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
                acc += pyc_hopper_tc_scalar_to_float(a[row * k + kk]) * pyc_hopper_tc_scalar_to_float(b[kk * n + col]);
            }
            c[row * n + col] = acc;
        }
    }
}

__launch_bounds__(PYC_HOPPER_TC_THREADS_PER_BLOCK, 1)
__global__ void pyc_hopper_tc_gemm_kernel(
    const pyc_hopper_tc_scalar_t* __restrict__ a,
    const pyc_hopper_tc_scalar_t* __restrict__ b,
    float* __restrict__ c,
    int m,
    int n,
    int k) {
    __shared__ __align__(16) pyc_hopper_tc_scalar_t shared_a[PYC_HOPPER_TC_TILE_M][PYC_HOPPER_TC_SHARED_STRIDE_A];
    __shared__ __align__(16) pyc_hopper_tc_scalar_t shared_b[PYC_HOPPER_TC_TILE_N][PYC_HOPPER_TC_SHARED_STRIDE_B];

    const int warp_id = threadIdx.x / 32;
    const int block_row = blockIdx.y * PYC_HOPPER_TC_TILE_M;
    const int block_col = blockIdx.x * PYC_HOPPER_TC_TILE_N;
    const int warp_row_group = warp_id / PYC_HOPPER_TC_WARP_COL_GROUPS;
    const int warp_col_group = warp_id % PYC_HOPPER_TC_WARP_COL_GROUPS;
    const int warp_row = warp_row_group * PYC_HOPPER_TC_WARP_TILE_M;
    const int warp_col = warp_col_group * PYC_HOPPER_TC_WARP_TILE_N;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[PYC_HOPPER_TC_WARP_ROW_TILES][PYC_HOPPER_TC_WARP_COL_TILES];

    if (warp_id >= PYC_HOPPER_TC_WARPS_PER_BLOCK) {
        return;
    }

    for (int row_tile = 0; row_tile < PYC_HOPPER_TC_WARP_ROW_TILES; ++row_tile) {
        for (int col_tile = 0; col_tile < PYC_HOPPER_TC_WARP_COL_TILES; ++col_tile) {
            wmma::fill_fragment(acc[row_tile][col_tile], 0.0f);
        }
    }

    for (int kk = 0; kk < k; kk += PYC_HOPPER_TC_TILE_K) {
        int idx;

        for (idx = threadIdx.x; idx < PYC_HOPPER_TC_TILE_M * PYC_HOPPER_TC_TILE_K; idx += blockDim.x) {
            const int row = idx / PYC_HOPPER_TC_TILE_K;
            const int col = idx % PYC_HOPPER_TC_TILE_K;
            const int g_row = block_row + row;
            const int g_col = kk + col;
            if (g_row < m && g_col < k) {
                shared_a[row][col] = a[g_row * k + g_col];
            } else {
                shared_a[row][col] = pyc_hopper_tc_make_scalar(0.0f);
            }
        }

        for (idx = threadIdx.x; idx < PYC_HOPPER_TC_TILE_N * PYC_HOPPER_TC_TILE_K; idx += blockDim.x) {
            const int row = idx / PYC_HOPPER_TC_TILE_K;
            const int col = idx % PYC_HOPPER_TC_TILE_K;
            const int g_row = kk + col;
            const int g_col = block_col + row;
            if (g_row < k && g_col < n) {
                shared_b[row][col] = b[g_row * n + g_col];
            } else {
                shared_b[row][col] = pyc_hopper_tc_make_scalar(0.0f);
            }
        }

        __syncthreads();

        for (int k_frag = 0; k_frag < PYC_HOPPER_TC_TILE_K; k_frag += PYC_HOPPER_TC_MMA_TILE_K) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, pyc_hopper_tc_scalar_t, wmma::row_major> a_frag[PYC_HOPPER_TC_WARP_ROW_TILES];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, pyc_hopper_tc_scalar_t, wmma::col_major> b_frag[PYC_HOPPER_TC_WARP_COL_TILES];

            for (int row_tile = 0; row_tile < PYC_HOPPER_TC_WARP_ROW_TILES; ++row_tile) {
                const int a_row = warp_row + row_tile * PYC_HOPPER_TC_MMA_TILE_M;
                wmma::load_matrix_sync(a_frag[row_tile], &shared_a[a_row][k_frag], PYC_HOPPER_TC_SHARED_STRIDE_A);
            }

            for (int col_tile = 0; col_tile < PYC_HOPPER_TC_WARP_COL_TILES; ++col_tile) {
                const int b_row = warp_col + col_tile * PYC_HOPPER_TC_MMA_TILE_N;
                wmma::load_matrix_sync(b_frag[col_tile], &shared_b[b_row][k_frag], PYC_HOPPER_TC_SHARED_STRIDE_B);
            }

            for (int row_tile = 0; row_tile < PYC_HOPPER_TC_WARP_ROW_TILES; ++row_tile) {
                for (int col_tile = 0; col_tile < PYC_HOPPER_TC_WARP_COL_TILES; ++col_tile) {
                    wmma::mma_sync(acc[row_tile][col_tile], a_frag[row_tile], b_frag[col_tile], acc[row_tile][col_tile]);
                }
            }
        }

        __syncthreads();
    }

    for (int row_tile = 0; row_tile < PYC_HOPPER_TC_WARP_ROW_TILES; ++row_tile) {
        const int c_row = block_row + warp_row + row_tile * PYC_HOPPER_TC_MMA_TILE_M;
        for (int col_tile = 0; col_tile < PYC_HOPPER_TC_WARP_COL_TILES; ++col_tile) {
            const int c_col = block_col + warp_col + col_tile * PYC_HOPPER_TC_MMA_TILE_N;
            if (c_row < m && c_col < n) {
                wmma::store_matrix_sync(&c[c_row * n + c_col], acc[row_tile][col_tile], n, wmma::mem_row_major);
            }
        }
    }
}

static int set_kernel_attributes(void) {
    cudaError_t status;

    status = cudaFuncSetAttribute(
        pyc_hopper_tc_gemm_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);
    if (status != cudaSuccess && status != cudaErrorNotSupported) {
        fprintf(stderr, "cudaFuncSetAttribute failed: %s\n", cudaGetErrorString(status));
        return -1;
    }

    return 0;
}

static int parse_config(int argc, char** argv, pyc_hopper_tc_config* cfg) {
    if (!cfg) {
        return -1;
    }

    cfg->m = 1024;
    cfg->n = 1024;
    cfg->k = 1024;
    cfg->warmup = 5;
    cfg->iters = 20;
    cfg->skip_reference = env_flag("PYC_HOPPER_TC_SKIP_REFERENCE", 0);

    if (argc > 1 && parse_int_arg(argv[1], &cfg->m) != 0) return -1;
    if (argc > 2 && parse_int_arg(argv[2], &cfg->n) != 0) return -1;
    if (argc > 3 && parse_int_arg(argv[3], &cfg->k) != 0) return -1;
    if (argc > 4 && parse_int_arg(argv[4], &cfg->warmup) != 0) return -1;
    if (argc > 5 && parse_int_arg(argv[5], &cfg->iters) != 0) return -1;
    if (argc > 6 && parse_int_arg(argv[6], &cfg->skip_reference) != 0) return -1;

    return 0;
}

int main(int argc, char** argv) {
    pyc_hopper_tc_config cfg;
    struct cudaDeviceProp props;
    pyc_hopper_tc_scalar_t* host_a = NULL;
    pyc_hopper_tc_scalar_t* host_b = NULL;
    float* host_c = NULL;
    float* ref_c = NULL;
    pyc_hopper_tc_scalar_t* dev_a = NULL;
    pyc_hopper_tc_scalar_t* dev_b = NULL;
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
    double max_abs_diff = 0.0;
    int iter;

    if (parse_config(argc, argv, &cfg) != 0) {
        fprintf(stderr, "usage: %s [m] [n] [k] [warmup] [iters] [skip_reference]\n", argv[0]);
        return 2;
    }

    if (
        (cfg.m % PYC_HOPPER_TC_TILE_M) != 0 || (cfg.n % PYC_HOPPER_TC_TILE_N) != 0
        || (cfg.k % PYC_HOPPER_TC_TILE_K) != 0) {
        fprintf(
            stderr,
            "Hopper Tensor Core lane requires %dx%dx%d-aligned shapes\n",
            PYC_HOPPER_TC_TILE_M,
            PYC_HOPPER_TC_TILE_N,
            PYC_HOPPER_TC_TILE_K);
        return 2;
    }

    if (check_cuda(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties") != 0) {
        return 1;
    }
    if (props.major < 9) {
        fprintf(stderr, "Hopper Tensor Core prototype requires sm_90-class hardware\n");
        return 1;
    }

    a_bytes = (size_t)cfg.m * (size_t)cfg.k * sizeof(pyc_hopper_tc_scalar_t);
    b_bytes = (size_t)cfg.k * (size_t)cfg.n * sizeof(pyc_hopper_tc_scalar_t);
    c_bytes = (size_t)cfg.m * (size_t)cfg.n * sizeof(float);

    host_a = (pyc_hopper_tc_scalar_t*)malloc(a_bytes);
    host_b = (pyc_hopper_tc_scalar_t*)malloc(b_bytes);
    if (!host_a || !host_b) {
        fprintf(stderr, "host allocation failed\n");
        return 1;
    }
    if (!cfg.skip_reference) {
        host_c = (float*)malloc(c_bytes);
        ref_c = (float*)malloc(c_bytes);
        if (!host_c || !ref_c) {
            fprintf(stderr, "host validation allocation failed\n");
            return 1;
        }
    }

    fill_matrix(host_a, cfg.m, cfg.k, 0.03125f);
    fill_matrix(host_b, cfg.k, cfg.n, 0.0625f);
    if (!cfg.skip_reference) {
        reference_gemm(host_a, host_b, ref_c, cfg.m, cfg.n, cfg.k);
    }

    if (check_cuda(cudaMalloc((void**)&dev_a, a_bytes), "cudaMalloc(a)") != 0) return 1;
    if (check_cuda(cudaMalloc((void**)&dev_b, b_bytes), "cudaMalloc(b)") != 0) return 1;
    if (check_cuda(cudaMalloc((void**)&dev_c, c_bytes), "cudaMalloc(c)") != 0) return 1;

    if (check_cuda(cudaMemcpy(dev_a, host_a, a_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(a)") != 0) return 1;
    if (check_cuda(cudaMemcpy(dev_b, host_b, b_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(b)") != 0) return 1;

    if (set_kernel_attributes() != 0) return 1;

    block = dim3(PYC_HOPPER_TC_THREADS_PER_BLOCK, 1, 1);
    grid = dim3(
        (unsigned int)((cfg.n + PYC_HOPPER_TC_TILE_N - 1) / PYC_HOPPER_TC_TILE_N),
        (unsigned int)((cfg.m + PYC_HOPPER_TC_TILE_M - 1) / PYC_HOPPER_TC_TILE_M),
        1);

    if (check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)") != 0) return 1;
    if (check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)") != 0) return 1;

    for (iter = 0; iter < cfg.warmup; ++iter) {
        pyc_hopper_tc_gemm_kernel<<<grid, block>>>(dev_a, dev_b, dev_c, cfg.m, cfg.n, cfg.k);
    }
    if (check_cuda(cudaGetLastError(), "kernel launch warmup") != 0) return 1;
    if (check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup") != 0) return 1;

    best_ms = 0.0;
    for (iter = 0; iter < cfg.iters; ++iter) {
        if (check_cuda(cudaEventRecord(start), "cudaEventRecord(start)") != 0) return 1;
        pyc_hopper_tc_gemm_kernel<<<grid, block>>>(dev_a, dev_b, dev_c, cfg.m, cfg.n, cfg.k);
        if (check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)") != 0) return 1;
        if (check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)") != 0) return 1;
        if (check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime") != 0) return 1;
        if (iter == 0 || elapsed_ms < (float)best_ms) {
            best_ms = elapsed_ms;
        }
    }

    if (!cfg.skip_reference) {
        if (check_cuda(cudaMemcpy(host_c, dev_c, c_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(c)") != 0) return 1;
        for (iter = 0; iter < cfg.m * cfg.n; ++iter) {
            double diff = fabs((double)host_c[iter] - (double)ref_c[iter]);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
            }
        }
    }

    printf("kernel=hopper_tensor_core\n");
    printf("lane=%s\n", PYC_HOPPER_TC_LANE_NAME);
    printf("arch=sm%d%d\n", props.major, props.minor);
    printf("shape=%dx%dx%d\n", cfg.m, cfg.n, cfg.k);
    printf(
        "tile=%dx%dx%d warp_tile=%dx%d warp_groups=%dx%d pads=%dx%d warps=%d threads=%d\n",
        PYC_HOPPER_TC_TILE_M,
        PYC_HOPPER_TC_TILE_N,
        PYC_HOPPER_TC_TILE_K,
        PYC_HOPPER_TC_WARP_TILE_M,
        PYC_HOPPER_TC_WARP_TILE_N,
        PYC_HOPPER_TC_WARP_ROW_GROUPS,
        PYC_HOPPER_TC_WARP_COL_GROUPS,
        PYC_HOPPER_TC_SHARED_PAD_A,
        PYC_HOPPER_TC_SHARED_PAD_B,
        PYC_HOPPER_TC_WARPS_PER_BLOCK,
        PYC_HOPPER_TC_THREADS_PER_BLOCK);
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

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(host_a);
    free(host_b);
    free(host_c);
    free(ref_c);
    return (cfg.skip_reference || max_abs_diff <= 0.2) ? 0 : 1;
}
