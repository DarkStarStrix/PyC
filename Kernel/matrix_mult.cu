// /kernel/matrix_mult.cu
__global__ void matrix_mult_kernel(float* a, float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0;
        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

void matrix_multiply(float* a, float* b, float* c, int m, int n, int k) {
    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (m + 15) / 16);
    matrix_mult_kernel<<<grid, block>>>(a, b, c, m, n, k);
    cudaDeviceSynchronize();
}
