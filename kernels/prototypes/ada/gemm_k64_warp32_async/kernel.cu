#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PYC_ADA_K64ASYNC_BLOCK_M 64
#define PYC_ADA_K64ASYNC_BLOCK_N 64
#define PYC_ADA_K64ASYNC_BLOCK_K 64
#define PYC_ADA_K64ASYNC_THREADS_X 32
#define PYC_ADA_K64ASYNC_THREADS_Y 8
#define PYC_ADA_K64ASYNC_THREAD_TILE_M 8
#define PYC_ADA_K64ASYNC_THREAD_TILE_N 2
#define PYC_ADA_K64ASYNC_VEC 4
#define PYC_ADA_K64ASYNC_STAGES 2
#define PYC_ADA_K64ASYNC_SHARED_STRIDE_A (PYC_ADA_K64ASYNC_BLOCK_K + 4)
#define PYC_ADA_K64ASYNC_SHARED_STRIDE_B (PYC_ADA_K64ASYNC_BLOCK_N + 4)
#define PYC_ADA_K64ASYNC_STAGE_A_ELEMS (PYC_ADA_K64ASYNC_BLOCK_M * PYC_ADA_K64ASYNC_SHARED_STRIDE_A)
#define PYC_ADA_K64ASYNC_STAGE_B_ELEMS (PYC_ADA_K64ASYNC_BLOCK_K * PYC_ADA_K64ASYNC_SHARED_STRIDE_B)
#define PYC_ADA_K64ASYNC_SHARED_ELEMS (PYC_ADA_K64ASYNC_STAGES * (PYC_ADA_K64ASYNC_STAGE_A_ELEMS + PYC_ADA_K64ASYNC_STAGE_B_ELEMS))
#define PYC_ADA_K64ASYNC_SHARED_BYTES (PYC_ADA_K64ASYNC_SHARED_ELEMS * (int)sizeof(float))

typedef struct {
    int m;
    int n;
    int k;
    int warmup;
    int iters;
} ada_gemm_k64async_config;

static int check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
        return -1;
    }
    return 0;
}

static double wall_ms_now(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec * 1000.0) + ((double)ts.tv_nsec / 1000000.0);
}

static void fill_matrix(float* data, int rows, int cols, float scale) {
    int i;
    for (i = 0; i < rows * cols; ++i) {
        int pattern = (i * 17 + rows * 13 + cols * 7) % 31;
        data[i] = ((float)pattern - 15.0f) * scale;
    }
}

static void reference_gemm(const float* a, const float* b, float* c, int m, int n, int k) {
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

__device__ static __forceinline__ void async_copy_16(void* dst, const void* src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    unsigned int smem_addr = (unsigned int)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(src));
#else
    *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(src);
#endif
}

__device__ static __forceinline__ void async_commit(void) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

__device__ static __forceinline__ void async_wait(void) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}

__device__ static __forceinline__ float* shared_stage_a(float* shared_mem, int stage) {
    return shared_mem + stage * (PYC_ADA_K64ASYNC_STAGE_A_ELEMS + PYC_ADA_K64ASYNC_STAGE_B_ELEMS);
}

__device__ static __forceinline__ float* shared_stage_b(float* shared_mem, int stage) {
    return shared_stage_a(shared_mem, stage) + PYC_ADA_K64ASYNC_STAGE_A_ELEMS;
}

__device__ static __forceinline__ float shared_a_load(const float* shared_a, int row, int col) {
    return shared_a[row * PYC_ADA_K64ASYNC_SHARED_STRIDE_A + col];
}

__device__ static __forceinline__ float shared_b_load(const float* shared_b, int row, int col) {
    return shared_b[row * PYC_ADA_K64ASYNC_SHARED_STRIDE_B + col];
}

__device__ static __forceinline__ void shared_a_store(float* shared_a, int row, int col, float value) {
    shared_a[row * PYC_ADA_K64ASYNC_SHARED_STRIDE_A + col] = value;
}

__device__ static __forceinline__ void shared_b_store(float* shared_b, int row, int col, float value) {
    shared_b[row * PYC_ADA_K64ASYNC_SHARED_STRIDE_B + col] = value;
}

__device__ static void load_a_stage(
    const float* __restrict__ a,
    float* shared_a,
    int stage,
    int lane_linear,
    int block_row,
    int kk_base,
    int m,
    int k) {
    const int block_threads = PYC_ADA_K64ASYNC_THREADS_X * PYC_ADA_K64ASYNC_THREADS_Y;
    const int vecs_per_row = PYC_ADA_K64ASYNC_BLOCK_K / PYC_ADA_K64ASYNC_VEC;
    const int total_vecs = (PYC_ADA_K64ASYNC_BLOCK_M * PYC_ADA_K64ASYNC_BLOCK_K) / PYC_ADA_K64ASYNC_VEC;
    const int full_tile = (block_row + PYC_ADA_K64ASYNC_BLOCK_M <= m) &&
                          (kk_base + PYC_ADA_K64ASYNC_BLOCK_K <= k) &&
                          ((k & (PYC_ADA_K64ASYNC_VEC - 1)) == 0);
    int phase;

    for (phase = 0; phase < total_vecs / block_threads; ++phase) {
        const int linear = lane_linear + phase * block_threads;
        const int tile_row = linear / vecs_per_row;
        const int tile_col = (linear % vecs_per_row) * PYC_ADA_K64ASYNC_VEC;
        const int global_row = block_row + tile_row;
        const int global_col = kk_base + tile_col;
        int i;

        if (full_tile) {
            async_copy_16(
                &shared_a[tile_row * PYC_ADA_K64ASYNC_SHARED_STRIDE_A + tile_col],
                &a[global_row * k + global_col]);
            continue;
        }

        for (i = 0; i < PYC_ADA_K64ASYNC_VEC; ++i) {
            float value = 0.0f;
            if (global_row < m && global_col + i < k) {
                value = a[global_row * k + global_col + i];
            }
            shared_a_store(shared_a, tile_row, tile_col + i, value);
        }
    }
}

__device__ static void load_b_stage(
    const float* __restrict__ b,
    float* shared_b,
    int stage,
    int lane_linear,
    int block_col,
    int kk_base,
    int k,
    int n) {
    const int block_threads = PYC_ADA_K64ASYNC_THREADS_X * PYC_ADA_K64ASYNC_THREADS_Y;
    const int vecs_per_row = PYC_ADA_K64ASYNC_BLOCK_N / PYC_ADA_K64ASYNC_VEC;
    const int total_vecs = (PYC_ADA_K64ASYNC_BLOCK_K * PYC_ADA_K64ASYNC_BLOCK_N) / PYC_ADA_K64ASYNC_VEC;
    const int full_tile = (block_col + PYC_ADA_K64ASYNC_BLOCK_N <= n) &&
                          (kk_base + PYC_ADA_K64ASYNC_BLOCK_K <= k) &&
                          ((n & (PYC_ADA_K64ASYNC_VEC - 1)) == 0);
    int phase;

    for (phase = 0; phase < total_vecs / block_threads; ++phase) {
        const int linear = lane_linear + phase * block_threads;
        const int tile_row = linear / vecs_per_row;
        const int tile_col = (linear % vecs_per_row) * PYC_ADA_K64ASYNC_VEC;
        const int global_row = kk_base + tile_row;
        const int global_col = block_col + tile_col;
        int i;

        if (full_tile) {
            async_copy_16(
                &shared_b[tile_row * PYC_ADA_K64ASYNC_SHARED_STRIDE_B + tile_col],
                &b[global_row * n + global_col]);
            continue;
        }

        for (i = 0; i < PYC_ADA_K64ASYNC_VEC; ++i) {
            float value = 0.0f;
            if (global_row < k && global_col + i < n) {
                value = b[global_row * n + global_col + i];
            }
            shared_b_store(shared_b, tile_row, tile_col + i, value);
        }
    }
}

__launch_bounds__(PYC_ADA_K64ASYNC_THREADS_X * PYC_ADA_K64ASYNC_THREADS_Y, 2)
__global__ void ada_fp32_k64async_gemm(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m,
    int n,
    int k) {
    extern __shared__ __align__(16) float shared_mem[];

    const int lane_linear = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_row = blockIdx.y * PYC_ADA_K64ASYNC_BLOCK_M;
    const int block_col = blockIdx.x * PYC_ADA_K64ASYNC_BLOCK_N;
    const int row_fragment = threadIdx.y * PYC_ADA_K64ASYNC_THREAD_TILE_M;
    const int col_fragment = threadIdx.x * PYC_ADA_K64ASYNC_THREAD_TILE_N;
    float accum[PYC_ADA_K64ASYNC_THREAD_TILE_M][PYC_ADA_K64ASYNC_THREAD_TILE_N];
    int kk_base;
    int stage;
    int next_stage;
    int i;
    int j;

    for (i = 0; i < PYC_ADA_K64ASYNC_THREAD_TILE_M; ++i) {
        for (j = 0; j < PYC_ADA_K64ASYNC_THREAD_TILE_N; ++j) {
            accum[i][j] = 0.0f;
        }
    }

    stage = 0;
    load_a_stage(a, shared_stage_a(shared_mem, stage), stage, lane_linear, block_row, 0, m, k);
    load_b_stage(b, shared_stage_b(shared_mem, stage), stage, lane_linear, block_col, 0, k, n);
    async_commit();
    async_wait();
    __syncthreads();

    for (kk_base = 0; kk_base < k; kk_base += PYC_ADA_K64ASYNC_BLOCK_K) {
        const int next_kk = kk_base + PYC_ADA_K64ASYNC_BLOCK_K;

        next_stage = stage ^ 1;
        if (next_kk < k) {
            load_a_stage(a, shared_stage_a(shared_mem, next_stage), next_stage, lane_linear, block_row, next_kk, m, k);
            load_b_stage(b, shared_stage_b(shared_mem, next_stage), next_stage, lane_linear, block_col, next_kk, k, n);
            async_commit();
        }

        #pragma unroll
        for (i = 0; i < PYC_ADA_K64ASYNC_BLOCK_K; ++i) {
            float a_frag[PYC_ADA_K64ASYNC_THREAD_TILE_M];
            float b_frag[PYC_ADA_K64ASYNC_THREAD_TILE_N];
            int ii;

            #pragma unroll
            for (ii = 0; ii < PYC_ADA_K64ASYNC_THREAD_TILE_M; ++ii) {
                a_frag[ii] = shared_a_load(shared_stage_a(shared_mem, stage), row_fragment + ii, i);
            }

            #pragma unroll
            for (ii = 0; ii < PYC_ADA_K64ASYNC_THREAD_TILE_N; ++ii) {
                b_frag[ii] = shared_b_load(shared_stage_b(shared_mem, stage), i, col_fragment + ii);
            }

            #pragma unroll
            for (ii = 0; ii < PYC_ADA_K64ASYNC_THREAD_TILE_M; ++ii) {
                int jj;
                #pragma unroll
                for (jj = 0; jj < PYC_ADA_K64ASYNC_THREAD_TILE_N; ++jj) {
                    accum[ii][jj] = fmaf(a_frag[ii], b_frag[jj], accum[ii][jj]);
                }
            }
        }

        __syncthreads();
        if (next_kk < k) {
            async_wait();
            __syncthreads();
            stage = next_stage;
        }
    }

    for (i = 0; i < PYC_ADA_K64ASYNC_THREAD_TILE_M; ++i) {
        int out_row = block_row + row_fragment + i;
        if (out_row >= m) {
            continue;
        }
        if (col_fragment + 1 < PYC_ADA_K64ASYNC_BLOCK_N) {
            int out_col = block_col + col_fragment;
            if (out_col + 1 < n) {
                float2 value;
                value.x = accum[i][0];
                value.y = accum[i][1];
                *reinterpret_cast<float2*>(&c[out_row * n + out_col]) = value;
                continue;
            }
        }
        for (j = 0; j < PYC_ADA_K64ASYNC_THREAD_TILE_N; ++j) {
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
        ada_fp32_k64async_gemm,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        PYC_ADA_K64ASYNC_SHARED_BYTES);
    if (status != cudaSuccess && status != cudaErrorNotSupported) {
        fprintf(stderr, "cudaFuncSetAttribute(max_dynamic_shared) failed: %s\n", cudaGetErrorString(status));
        return -1;
    }

    status = cudaFuncSetAttribute(
        ada_fp32_k64async_gemm,
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

static int parse_config(int argc, char** argv, ada_gemm_k64async_config* cfg) {
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
    ada_gemm_k64async_config cfg;
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
    double best_wall_ms = 0.0;
    double wall_sum_ms = 0.0;
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

    block = dim3(PYC_ADA_K64ASYNC_THREADS_X, PYC_ADA_K64ASYNC_THREADS_Y, 1);
    grid = dim3(
        (unsigned int)((cfg.n + PYC_ADA_K64ASYNC_BLOCK_N - 1) / PYC_ADA_K64ASYNC_BLOCK_N),
        (unsigned int)((cfg.m + PYC_ADA_K64ASYNC_BLOCK_M - 1) / PYC_ADA_K64ASYNC_BLOCK_M),
        1);

    if (check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)") != 0) return 1;
    if (check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)") != 0) return 1;

    for (iter = 0; iter < cfg.warmup; ++iter) {
        ada_fp32_k64async_gemm<<<grid, block, PYC_ADA_K64ASYNC_SHARED_BYTES>>>(dev_a, dev_b, dev_c, cfg.m, cfg.n, cfg.k);
    }
    if (check_cuda(cudaGetLastError(), "kernel launch warmup") != 0) return 1;
    if (check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup") != 0) return 1;

    best_ms = 0.0;
    best_wall_ms = 0.0;
    wall_sum_ms = 0.0;
    for (iter = 0; iter < cfg.iters; ++iter) {
        double wall_start_ms = wall_ms_now();
        if (check_cuda(cudaEventRecord(start), "cudaEventRecord(start)") != 0) return 1;
        ada_fp32_k64async_gemm<<<grid, block, PYC_ADA_K64ASYNC_SHARED_BYTES>>>(dev_a, dev_b, dev_c, cfg.m, cfg.n, cfg.k);
        if (check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)") != 0) return 1;
        if (check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)") != 0) return 1;
        if (check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime") != 0) return 1;
        {
            double wall_elapsed_ms = wall_ms_now() - wall_start_ms;
            wall_sum_ms += wall_elapsed_ms;
            if (iter == 0 || wall_elapsed_ms < best_wall_ms) {
                best_wall_ms = wall_elapsed_ms;
            }
        }
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
    printf("tile=%dx%dx%d threads=%dx%d thread_tile=%dx%d vector_width=%d\n",
           PYC_ADA_K64ASYNC_BLOCK_M,
           PYC_ADA_K64ASYNC_BLOCK_N,
           PYC_ADA_K64ASYNC_BLOCK_K,
           PYC_ADA_K64ASYNC_THREADS_X,
           PYC_ADA_K64ASYNC_THREADS_Y,
           PYC_ADA_K64ASYNC_THREAD_TILE_M,
           PYC_ADA_K64ASYNC_THREAD_TILE_N,
           PYC_ADA_K64ASYNC_VEC);
    printf("best_ms=%.3f\n", best_ms);
    if (cfg.iters > 0) {
        printf("best_wall_ms=%.3f\n", best_wall_ms);
        printf("mean_wall_ms=%.3f\n", wall_sum_ms / (double)cfg.iters);
    }
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
