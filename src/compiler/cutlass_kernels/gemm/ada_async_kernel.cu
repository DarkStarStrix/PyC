#include "pyc/kernel_registry.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#define PYC_ADA_REG_BLOCK_M 64
#define PYC_ADA_REG_BLOCK_N 64
#define PYC_ADA_REG_BLOCK_K 64
#define PYC_ADA_REG_THREADS_X 32
#define PYC_ADA_REG_THREADS_Y 8
#define PYC_ADA_REG_THREAD_TILE_M 8
#define PYC_ADA_REG_THREAD_TILE_N 2
#define PYC_ADA_REG_VEC 4
#define PYC_ADA_REG_STAGES 2
#define PYC_ADA_REG_SHARED_STRIDE_A (PYC_ADA_REG_BLOCK_K + 4)
#define PYC_ADA_REG_SHARED_STRIDE_B (PYC_ADA_REG_BLOCK_N + 4)
#define PYC_ADA_REG_STAGE_A_ELEMS (PYC_ADA_REG_BLOCK_M * PYC_ADA_REG_SHARED_STRIDE_A)
#define PYC_ADA_REG_STAGE_B_ELEMS (PYC_ADA_REG_BLOCK_K * PYC_ADA_REG_SHARED_STRIDE_B)
#define PYC_ADA_REG_SHARED_ELEMS (PYC_ADA_REG_STAGES * (PYC_ADA_REG_STAGE_A_ELEMS + PYC_ADA_REG_STAGE_B_ELEMS))
#define PYC_ADA_REG_SHARED_BYTES (PYC_ADA_REG_SHARED_ELEMS * (int)sizeof(float))

__device__ static __forceinline__ void pyc_ada_async_copy_16(void* dst, const void* src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    unsigned int smem_addr = (unsigned int)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(src));
#else
    *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(src);
#endif
}

__device__ static __forceinline__ void pyc_ada_async_commit(void) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

__device__ static __forceinline__ void pyc_ada_async_wait(void) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}

__device__ static __forceinline__ float* pyc_ada_shared_stage_a(float* shared_mem, int stage) {
    return shared_mem + stage * (PYC_ADA_REG_STAGE_A_ELEMS + PYC_ADA_REG_STAGE_B_ELEMS);
}

__device__ static __forceinline__ float* pyc_ada_shared_stage_b(float* shared_mem, int stage) {
    return pyc_ada_shared_stage_a(shared_mem, stage) + PYC_ADA_REG_STAGE_A_ELEMS;
}

__device__ static __forceinline__ float pyc_ada_shared_a_load(const float* shared_a, int row, int col) {
    return shared_a[row * PYC_ADA_REG_SHARED_STRIDE_A + col];
}

__device__ static __forceinline__ float pyc_ada_shared_b_load(const float* shared_b, int row, int col) {
    return shared_b[row * PYC_ADA_REG_SHARED_STRIDE_B + col];
}

__device__ static __forceinline__ void pyc_ada_shared_a_store(float* shared_a, int row, int col, float value) {
    shared_a[row * PYC_ADA_REG_SHARED_STRIDE_A + col] = value;
}

__device__ static __forceinline__ void pyc_ada_shared_b_store(float* shared_b, int row, int col, float value) {
    shared_b[row * PYC_ADA_REG_SHARED_STRIDE_B + col] = value;
}

__device__ static void pyc_ada_load_a_stage(
    const float* __restrict__ a,
    float* shared_a,
    int lane_linear,
    int block_row,
    int kk_base,
    int m,
    int k) {
    const int block_threads = PYC_ADA_REG_THREADS_X * PYC_ADA_REG_THREADS_Y;
    const int vecs_per_row = PYC_ADA_REG_BLOCK_K / PYC_ADA_REG_VEC;
    const int total_vecs = (PYC_ADA_REG_BLOCK_M * PYC_ADA_REG_BLOCK_K) / PYC_ADA_REG_VEC;
    const int full_tile = (block_row + PYC_ADA_REG_BLOCK_M <= m) &&
                          (kk_base + PYC_ADA_REG_BLOCK_K <= k) &&
                          ((k & (PYC_ADA_REG_VEC - 1)) == 0);
    int phase;

    for (phase = 0; phase < total_vecs / block_threads; ++phase) {
        const int linear = lane_linear + phase * block_threads;
        const int tile_row = linear / vecs_per_row;
        const int tile_col = (linear % vecs_per_row) * PYC_ADA_REG_VEC;
        const int global_row = block_row + tile_row;
        const int global_col = kk_base + tile_col;
        int i;

        if (full_tile) {
            pyc_ada_async_copy_16(
                &shared_a[tile_row * PYC_ADA_REG_SHARED_STRIDE_A + tile_col],
                &a[global_row * k + global_col]);
            continue;
        }

        for (i = 0; i < PYC_ADA_REG_VEC; ++i) {
            float value = 0.0f;
            if (global_row < m && global_col + i < k) {
                value = a[global_row * k + global_col + i];
            }
            pyc_ada_shared_a_store(shared_a, tile_row, tile_col + i, value);
        }
    }
}

__device__ static void pyc_ada_load_b_stage(
    const float* __restrict__ b,
    float* shared_b,
    int lane_linear,
    int block_col,
    int kk_base,
    int k,
    int n) {
    const int block_threads = PYC_ADA_REG_THREADS_X * PYC_ADA_REG_THREADS_Y;
    const int vecs_per_row = PYC_ADA_REG_BLOCK_N / PYC_ADA_REG_VEC;
    const int total_vecs = (PYC_ADA_REG_BLOCK_K * PYC_ADA_REG_BLOCK_N) / PYC_ADA_REG_VEC;
    const int full_tile = (block_col + PYC_ADA_REG_BLOCK_N <= n) &&
                          (kk_base + PYC_ADA_REG_BLOCK_K <= k) &&
                          ((n & (PYC_ADA_REG_VEC - 1)) == 0);
    int phase;

    for (phase = 0; phase < total_vecs / block_threads; ++phase) {
        const int linear = lane_linear + phase * block_threads;
        const int tile_row = linear / vecs_per_row;
        const int tile_col = (linear % vecs_per_row) * PYC_ADA_REG_VEC;
        const int global_row = kk_base + tile_row;
        const int global_col = block_col + tile_col;
        int i;

        if (full_tile) {
            pyc_ada_async_copy_16(
                &shared_b[tile_row * PYC_ADA_REG_SHARED_STRIDE_B + tile_col],
                &b[global_row * n + global_col]);
            continue;
        }

        for (i = 0; i < PYC_ADA_REG_VEC; ++i) {
            float value = 0.0f;
            if (global_row < k && global_col + i < n) {
                value = b[global_row * n + global_col + i];
            }
            pyc_ada_shared_b_store(shared_b, tile_row, tile_col + i, value);
        }
    }
}

__launch_bounds__(PYC_ADA_REG_THREADS_X * PYC_ADA_REG_THREADS_Y, 2)
__global__ void pyc_ada_async_gemm_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m,
    int n,
    int k) {
    extern __shared__ __align__(16) float shared_mem[];

    const int lane_linear = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_row = blockIdx.y * PYC_ADA_REG_BLOCK_M;
    const int block_col = blockIdx.x * PYC_ADA_REG_BLOCK_N;
    const int row_fragment = threadIdx.y * PYC_ADA_REG_THREAD_TILE_M;
    const int col_fragment = threadIdx.x * PYC_ADA_REG_THREAD_TILE_N;
    float accum[PYC_ADA_REG_THREAD_TILE_M][PYC_ADA_REG_THREAD_TILE_N];
    int kk_base;
    int stage;
    int i;
    int j;

    for (i = 0; i < PYC_ADA_REG_THREAD_TILE_M; ++i) {
        for (j = 0; j < PYC_ADA_REG_THREAD_TILE_N; ++j) {
            accum[i][j] = 0.0f;
        }
    }

    stage = 0;
    pyc_ada_load_a_stage(a, pyc_ada_shared_stage_a(shared_mem, stage), lane_linear, block_row, 0, m, k);
    pyc_ada_load_b_stage(b, pyc_ada_shared_stage_b(shared_mem, stage), lane_linear, block_col, 0, k, n);
    pyc_ada_async_commit();
    pyc_ada_async_wait();
    __syncthreads();

    for (kk_base = 0; kk_base < k; kk_base += PYC_ADA_REG_BLOCK_K) {
        const int next_kk = kk_base + PYC_ADA_REG_BLOCK_K;
        const int next_stage = stage ^ 1;

        if (next_kk < k) {
            pyc_ada_load_a_stage(a, pyc_ada_shared_stage_a(shared_mem, next_stage), lane_linear, block_row, next_kk, m, k);
            pyc_ada_load_b_stage(b, pyc_ada_shared_stage_b(shared_mem, next_stage), lane_linear, block_col, next_kk, k, n);
            pyc_ada_async_commit();
        }

        #pragma unroll
        for (i = 0; i < PYC_ADA_REG_BLOCK_K; ++i) {
            float a_frag[PYC_ADA_REG_THREAD_TILE_M];
            float b_frag[PYC_ADA_REG_THREAD_TILE_N];
            int ii;

            #pragma unroll
            for (ii = 0; ii < PYC_ADA_REG_THREAD_TILE_M; ++ii) {
                a_frag[ii] = pyc_ada_shared_a_load(pyc_ada_shared_stage_a(shared_mem, stage), row_fragment + ii, i);
            }

            #pragma unroll
            for (ii = 0; ii < PYC_ADA_REG_THREAD_TILE_N; ++ii) {
                b_frag[ii] = pyc_ada_shared_b_load(pyc_ada_shared_stage_b(shared_mem, stage), i, col_fragment + ii);
            }

            #pragma unroll
            for (ii = 0; ii < PYC_ADA_REG_THREAD_TILE_M; ++ii) {
                int jj;
                #pragma unroll
                for (jj = 0; jj < PYC_ADA_REG_THREAD_TILE_N; ++jj) {
                    accum[ii][jj] = fmaf(a_frag[ii], b_frag[jj], accum[ii][jj]);
                }
            }
        }

        __syncthreads();
        if (next_kk < k) {
            pyc_ada_async_wait();
            __syncthreads();
            stage = next_stage;
        }
    }

    for (i = 0; i < PYC_ADA_REG_THREAD_TILE_M; ++i) {
        int out_row = block_row + row_fragment + i;
        if (out_row >= m) {
            continue;
        }
        if (col_fragment + 1 < PYC_ADA_REG_BLOCK_N) {
            int out_col = block_col + col_fragment;
            if (out_col + 1 < n) {
                float2 value;
                value.x = accum[i][0];
                value.y = accum[i][1];
                *reinterpret_cast<float2*>(&c[out_row * n + out_col]) = value;
                continue;
            }
        }
        for (j = 0; j < PYC_ADA_REG_THREAD_TILE_N; ++j) {
            int out_col = block_col + col_fragment + j;
            if (out_col < n) {
                c[out_row * n + out_col] = accum[i][j];
            }
        }
    }
}

extern "C" void pyc_register_ada_async_gemm_kernel(void) {
    pyc_kernel_desc desc;
    memset(&desc, 0, sizeof(desc));
    strncpy(desc.op_key, "matmul", PYC_KERNEL_OP_KEY_MAX - 1);
    strncpy(desc.symbol, "ada_gemm_k64_warp32_async_f32", PYC_KERNEL_SYMBOL_MAX - 1);
    desc.backend = PYC_BACKEND_CUDA;
    desc.priority = 35;
    desc.estimated_occupancy = 0.78;
    desc.tensor_core_eligible = 0;
    desc.shared_mem_bytes = (size_t)PYC_ADA_REG_SHARED_BYTES;
    desc.reg_pressure_class = 3;
    pyc_kernel_register(&desc);
}

extern "C" int pyc_ada_async_gemm_dispatch(
    int M,
    int N,
    int K,
    const void* A,
    const void* B,
    void* C,
    float alpha,
    float beta,
    cudaStream_t stream) {
    cudaError_t status;
    dim3 block;
    dim3 grid;

    if (!A || !B || !C) {
        return -1;
    }
    if (alpha != 1.0f || beta != 0.0f) {
        return -1;
    }

    status = cudaFuncSetAttribute(
        pyc_ada_async_gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        PYC_ADA_REG_SHARED_BYTES);
    if (status != cudaSuccess && status != cudaErrorNotSupported) {
        return -1;
    }

    status = cudaFuncSetAttribute(
        pyc_ada_async_gemm_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);
    if (status != cudaSuccess && status != cudaErrorNotSupported) {
        return -1;
    }

    block = dim3(PYC_ADA_REG_THREADS_X, PYC_ADA_REG_THREADS_Y, 1);
    grid = dim3(
        (unsigned int)((N + PYC_ADA_REG_BLOCK_N - 1) / PYC_ADA_REG_BLOCK_N),
        (unsigned int)((M + PYC_ADA_REG_BLOCK_M - 1) / PYC_ADA_REG_BLOCK_M),
        1);

    pyc_ada_async_gemm_kernel<<<grid, block, PYC_ADA_REG_SHARED_BYTES, stream>>>(
        reinterpret_cast<const float*>(A),
        reinterpret_cast<const float*>(B),
        reinterpret_cast<float*>(C),
        M,
        N,
        K);
    status = cudaGetLastError();
    return status == cudaSuccess ? 0 : -1;
}
