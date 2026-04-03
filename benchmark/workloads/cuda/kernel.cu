#include <stdio.h>

__global__ void matrix_mult_kernel(float* a, float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main() {
    const int m = 512, n = 512, k = 512;
    float *a = (float*)malloc(m * k * sizeof(float));
    float *b = (float*)malloc(k * n * sizeof(float));
    float *c = (float*)malloc(m * n * sizeof(float));

    // Initialize matrices
    for (int i = 0; i < m * k; i++) a[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < k * n; i++) b[i] = (float)rand() / RAND_MAX;

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, m * k * sizeof(float));
    cudaMalloc(&d_b, k * n * sizeof(float));
    cudaMalloc(&d_c, m * n * sizeof(float));

    cudaMemcpy(d_a, a, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, k * n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);

    for (int i = 0; i < 100; i++) {
        matrix_mult_kernel<<<blocks, threads>>>(d_a, d_b, d_c, m, n, k);
    }

    cudaMemcpy(c, d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    FILE *fp = fopen("output.txt", "w");
    for (int i = 0; i < m * n; i++) fprintf(fp, "%f\n", c[i]);
    fclose(fp);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(a); free(b); free(c);
    return 0;
}
