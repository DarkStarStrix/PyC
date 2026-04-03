#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define PYC_ADA_BLOCK_M 64
#define PYC_ADA_BLOCK_N 64
#define PYC_ADA_BLOCK_K 16
#define PYC_ADA_THREADS_X 16
#define PYC_ADA_THREADS_Y 16
#define PYC_ADA_THREAD_TILE_M 4
#define PYC_ADA_THREAD_TILE_N 4

typedef struct {
    int m;
    int n;
    int k;
    int warmup;
    int iters;
} ada_gemm_config;

static int check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
        return -1;
    }
    return 0;
}

static void fill_matrix(float* data, int rows, int cols, float scale) {
    int i;
    for (i = 0; i < rows * cols; ++i) {
        int pattern = (i * 17 + rows * 13 + cols * 7) % 31;
        data[i] = ((float)pattern - 15.0f) * scale;
    }
}

static void reference_gemm(
    const float* a,
    const float* b,
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
                acc += a[row * k + kk] * b[kk * n + col];
            }
            c[row * n + col] = acc;
        }
    }
}

__launch_bounds__(PYC_ADA_THREADS_X * PYC_ADA_THREADS_Y, 2)
__global__ void ada_fp32_tiled_gemm(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m,
    int n,
    int k) {
    __shared__ float a_tile[PYC_ADA_BLOCK_M][PYC_ADA_BLOCK_K];
    __shared__ float b_tile[PYC_ADA_BLOCK_K][PYC_ADA_BLOCK_N];

    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;
    const int block_row = blockIdx.y * PYC_ADA_BLOCK_M;
    const int block_col = blockIdx.x * PYC_ADA_BLOCK_N;
    const int lane_linear = thread_row * blockDim.x + thread_col;
    const int a_loads_per_thread = (PYC_ADA_BLOCK_M * PYC_ADA_BLOCK_K) / (PYC_ADA_THREADS_X * PYC_ADA_THREADS_Y);
    const int b_loads_per_thread = (PYC_ADA_BLOCK_K * PYC_ADA_BLOCK_N) / (PYC_ADA_THREADS_X * PYC_ADA_THREADS_Y);
    float accum[PYC_ADA_THREAD_TILE_M][PYC_ADA_THREAD_TILE_N];
    int row_fragment = thread_row * PYC_ADA_THREAD_TILE_M;
    int col_fragment = thread_col * PYC_ADA_THREAD_TILE_N;
    int kk_base;
    int i;
    int j;

    for (i = 0; i < PYC_ADA_THREAD_TILE_M; ++i) {
        for (j = 0; j < PYC_ADA_THREAD_TILE_N; ++j) {
            accum[i][j] = 0.0f;
        }
    }

    for (kk_base = 0; kk_base < k; kk_base += PYC_ADA_BLOCK_K) {
        for (i = 0; i < a_loads_per_thread; ++i) {
            int linear = lane_linear * a_loads_per_thread + i;
            int tile_row = linear / PYC_ADA_BLOCK_K;
            int tile_col = linear % PYC_ADA_BLOCK_K;
            int global_row = block_row + tile_row;
            int global_col = kk_base + tile_col;
            float value = 0.0f;
            if (global_row < m && global_col < k) {
                value = a[global_row * k + global_col];
            }
            a_tile[tile_row][tile_col] = value;
        }

        for (i = 0; i < b_loads_per_thread; ++i) {
            int linear = lane_linear * b_loads_per_thread + i;
            int tile_row = linear / PYC_ADA_BLOCK_N;
            int tile_col = linear % PYC_ADA_BLOCK_N;
            int global_row = kk_base + tile_row;
            int global_col = block_col + tile_col;
            float value = 0.0f;
            if (global_row < k && global_col < n) {
                value = b[global_row * n + global_col];
            }
            b_tile[tile_row][tile_col] = value;
        }

        __syncthreads();

        #pragma unroll
        for (i = 0; i < PYC_ADA_BLOCK_K; ++i) {
            float a_frag[PYC_ADA_THREAD_TILE_M];
            float b_frag[PYC_ADA_THREAD_TILE_N];

            #pragma unroll
            for (j = 0; j < PYC_ADA_THREAD_TILE_M; ++j) {
                a_frag[j] = a_tile[row_fragment + j][i];
            }
            #pragma unroll
            for (j = 0; j < PYC_ADA_THREAD_TILE_N; ++j) {
                b_frag[j] = b_tile[i][col_fragment + j];
            }
            #pragma unroll
            for (j = 0; j < PYC_ADA_THREAD_TILE_M; ++j) {
                int jj;
                #pragma unroll
                for (jj = 0; jj < PYC_ADA_THREAD_TILE_N; ++jj) {
                    accum[j][jj] += a_frag[j] * b_frag[jj];
                }
            }
        }

        __syncthreads();
    }

    for (i = 0; i < PYC_ADA_THREAD_TILE_M; ++i) {
        int out_row = block_row + row_fragment + i;
        if (out_row >= m) {
            continue;
        }
        for (j = 0; j < PYC_ADA_THREAD_TILE_N; ++j) {
            int out_col = block_col + col_fragment + j;
            if (out_col < n) {
                c[out_row * n + out_col] = accum[i][j];
            }
        }
    }
}

static int configure_ada_kernel(void) {
    cudaError_t status;

    status = cudaFuncSetAttribute(
        ada_fp32_tiled_gemm,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);
    if (status != cudaSuccess && status != cudaErrorNotSupported) {
        fprintf(stderr, "cudaFuncSetAttribute failed: %s\n", cudaGetErrorString(status));
        return -1;
    }

    return 0;
}

static int parse_int_arg(const char* text, int* out) {
    char* end = NULL;
    long value;

    if (!text || !out) {
        return -1;
    }

    value = strtol(text, &end, 10);
    if (*text == '\0' || !end || *end != '\0' || value <= 0 || value > 1 << 20) {
        return -1;
    }

    *out = (int)value;
    return 0;
}

static int parse_config(int argc, char** argv, ada_gemm_config* cfg) {
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
    ada_gemm_config cfg;
    cudaDeviceProp props;
    float* host_a = NULL;
    float* host_b = NULL;
    float* host_c = NULL;
    float* ref_c = NULL;
    float* dev_a = NULL;
    float* dev_b = NULL;
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
    int device = 0;

    if (parse_config(argc, argv, &cfg) != 0) {
        fprintf(stderr, "usage: %s [m] [n] [k] [warmup] [iters]\n", argv[0]);
        return 2;
    }

    if (check_cuda(cudaGetDevice(&device), "cudaGetDevice") != 0) return 1;
    if (check_cuda(cudaGetDeviceProperties(&props, device), "cudaGetDeviceProperties") != 0) return 1;

    printf("device=%s cc=%d.%d\n", props.name, props.major, props.minor);
    if (!(props.major == 8 && props.minor == 9)) {
        printf("note=prototype tuned for Ada (sm_89); running on a different architecture\n");
    }

    a_bytes = (size_t)cfg.m * (size_t)cfg.k * sizeof(float);
    b_bytes = (size_t)cfg.k * (size_t)cfg.n * sizeof(float);
    c_bytes = (size_t)cfg.m * (size_t)cfg.n * sizeof(float);

    host_a = (float*)malloc(a_bytes);
    host_b = (float*)malloc(b_bytes);
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

    if (configure_ada_kernel() != 0) return 1;

    block = dim3(PYC_ADA_THREADS_X, PYC_ADA_THREADS_Y, 1);
    grid = dim3(
        (unsigned int)((cfg.n + PYC_ADA_BLOCK_N - 1) / PYC_ADA_BLOCK_N),
        (unsigned int)((cfg.m + PYC_ADA_BLOCK_M - 1) / PYC_ADA_BLOCK_M),
        1);

    if (check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)") != 0) return 1;
    if (check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)") != 0) return 1;

    for (iter = 0; iter < cfg.warmup; ++iter) {
        ada_fp32_tiled_gemm<<<grid, block>>>(dev_a, dev_b, dev_c, cfg.m, cfg.n, cfg.k);
    }
    if (check_cuda(cudaGetLastError(), "kernel launch warmup") != 0) return 1;
    if (check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup") != 0) return 1;

    best_ms = 0.0;
    for (iter = 0; iter < cfg.iters; ++iter) {
        if (check_cuda(cudaEventRecord(start), "cudaEventRecord(start)") != 0) return 1;
        ada_fp32_tiled_gemm<<<grid, block>>>(dev_a, dev_b, dev_c, cfg.m, cfg.n, cfg.k);
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

    printf("shape=%dx%dx%d\n", cfg.m, cfg.n, cfg.k);
    printf("tile=%dx%dx%d threads=%dx%d\n",
           PYC_ADA_BLOCK_M,
           PYC_ADA_BLOCK_N,
           PYC_ADA_BLOCK_K,
           PYC_ADA_THREADS_X,
           PYC_ADA_THREADS_Y);
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
    return max_abs_diff <= 1e-2 ? 0 : 1;
}
