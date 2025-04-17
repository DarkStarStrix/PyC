#include "parser.cuh"
#include "error_handler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm> // For std::sort
#include <cuda_fp16.h> // For FP16 support
#include <cuda_runtime.h>

// CUDA kernel for tokenizing input in parallel
__global__ void tokenize_kernel(const char* input, size_t input_length, TokenGPU* tokens, int* token_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < input_length) {
        // Start from this thread's position
        int pos = tid;
        
        // Skip if this position isn't the start of a token
        if (pos > 0) {
            char prev = input[pos - 1];
            // Skip if we're in the middle of a token
            if ((isalnum(prev) && isalnum(input[pos])) || 
                (isspace(prev) && isspace(input[pos]))) {
                return;
            }
        }
        
        // Process potential token at this position
        if (isalpha(input[pos])) {
            // Found an identifier token
            int token_idx = atomicAdd(token_count, 1);
            if (token_idx < MAX_TOKENS) {
                tokens[token_idx].type = TOKEN_IDENTIFIER;
                tokens[token_idx].start_pos = pos;
                
                // Find the end position
                int end_pos = pos;
                while (end_pos < input_length && isalnum(input[end_pos])) {
                    end_pos++;
                }
                tokens[token_idx].end_pos = end_pos - 1;
                tokens[token_idx].length = end_pos - pos;
            }
        }
        else if (isdigit(input[pos])) {
            // Found a number token
            int token_idx = atomicAdd(token_count, 1);
            if (token_idx < MAX_TOKENS) {
                tokens[token_idx].type = TOKEN_NUMBER;
                tokens[token_idx].start_pos = pos;
                
                // Find the end position
                int end_pos = pos;
                while (end_pos < input_length && isdigit(input[end_pos])) {
                    end_pos++;
                }
                tokens[token_idx].end_pos = end_pos - 1;
                tokens[token_idx].length = end_pos - pos;
            }
        }
        else if (input[pos] == '+' || input[pos] == '-' || 
                 input[pos] == '*' || input[pos] == '/') {
            // Found an operator token
            int token_idx = atomicAdd(token_count, 1);
            if (token_idx < MAX_TOKENS) {
                tokens[token_idx].type = TOKEN_OPERATOR;
                tokens[token_idx].start_pos = pos;
                tokens[token_idx].end_pos = pos;
                tokens[token_idx].length = 1;
            }
        }
        // Add other token types as needed
    }
}

// Function to parse with CUDA acceleration
ASTNode* cuda_parse(const char* source_code) {
    // Initialize error handler
    init_error_handler("input.py", source_code);
    
    // Calculate input length
    size_t input_length = strlen(source_code);
    
    // Allocate device memory for input
    char* d_input;
    cudaMalloc((void**)&d_input, input_length + 1);
    cudaMemcpy(d_input, source_code, input_length + 1, cudaMemcpyHostToDevice);
    
    // Allocate memory for tokens
    TokenGPU* d_tokens;
    cudaMalloc((void**)&d_tokens, MAX_TOKENS * sizeof(TokenGPU));
    
    // Token count (to be updated by the kernel)
    int* d_token_count;
    cudaMalloc((void**)&d_token_count, sizeof(int));
    cudaMemset(d_token_count, 0, sizeof(int));
    
    // Launch kernel
    int block_size = 256;
    int grid_size = (input_length + block_size - 1) / block_size;
    tokenize_kernel<<<grid_size, block_size>>>(d_input, input_length, d_tokens, d_token_count);
    
    // Get token count
    int token_count = 0;
    cudaMemcpy(&token_count, d_token_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("CUDA Tokenization found %d tokens\n", token_count);
    
    // Get tokens
    TokenGPU* tokens = (TokenGPU*)malloc(token_count * sizeof(TokenGPU));
    cudaMemcpy(tokens, d_tokens, token_count * sizeof(TokenGPU), cudaMemcpyDeviceToHost);
    
    // Sort tokens based on their start positions to preserve source code order
    std::sort(tokens, tokens + token_count, [](const TokenGPU& a, const TokenGPU& b) {
        return a.start_pos < b.start_pos;
    });
    
    // Convert GPU tokens to regular tokens
    Token* cpu_tokens = (Token*)malloc(token_count * sizeof(Token));
    for (int i = 0; i < token_count; i++) {
        cpu_tokens[i].type = tokens[i].type;
        
        // Extract token value from source code
        int length = tokens[i].length;
        if (length > 255) length = 255; // Ensure we don't overflow
        strncpy(cpu_tokens[i].value, &source_code[tokens[i].start_pos], length);
        cpu_tokens[i].value[length] = '\0';
    }
    
    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_tokens);
    cudaFree(d_token_count);
    
    // Free temporary CPU memory for GPU tokens
    free(tokens);
    
    // Parse the tokens to build the AST using existing parsing logic
    ASTNode* ast = parse_tokens(cpu_tokens, token_count);
    
    // Free CPU tokens (assuming parse_tokens copies necessary data into the AST)
    free(cpu_tokens);
    
    // Return the root of the AST
    return ast;
}

// CUDA kernel for FP16 matrix multiplication (simple inference kernel)
__global__ void matmul_fp16(__half* A, __half* B, __half* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        __half sum = __float2half(0.0f);
        for (int k = 0; k < K; k++) {
            sum = __hadd(sum, __hmul(A[row * K + k], B[k * N + col]));
        }
        C[row * N + col] = sum;
    }
}

// Host function to launch FP16 matrix multiplication kernel
void launch_matmul_fp16(__half* A, __half* B, __half* C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_fp16<<<grid, block>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
