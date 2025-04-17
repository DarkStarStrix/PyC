#include <stdlib.h>
#include <string.h>
#include "graph_compiler.h"

// Structure for tensor node in memory graph
typedef struct TensorNode {
    char* name;
    size_t size; // Size in bytes
    int start_time; // When tensor is allocated
    int end_time; // When tensor is deallocated
    struct TensorNode* next;
} TensorNode;

// Structure for memory pool
typedef struct MemoryPool {
    void* base;
    size_t total_size;
    size_t used;
} MemoryPool;

// Global memory graph and pool
static TensorNode* memory_graph = NULL;
static MemoryPool cpu_pool = {0};
static MemoryPool gpu_pool = {0};

// Initialize memory pool
void init_memory_pool(MemoryPool* pool, size_t size, int is_gpu) {
    pool->total_size = size;
    pool->used = 0;
    pool->base = is_gpu ? cudaMalloc(&pool->base, size) : malloc(size);
    if (!pool->base) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
}

// Add tensor to memory graph
void add_tensor_to_graph(const char* name, size_t size, int start, int end) {
    TensorNode* node = (TensorNode*)malloc(sizeof(TensorNode));
    node->name = strdup(name);
    node->size = size;
    node->start_time = start;
    node->end_time = end;
    node->next = memory_graph;
    memory_graph = node;
}

// Analyze lifetimes and allocate memory
void* allocate_tensor(const char* name, size_t size, int is_gpu) {
    TensorNode* node = memory_graph;
    while (node) {
        if (strcmp(node->name, name) == 0) {
            MemoryPool* pool = is_gpu ? &gpu_pool : &cpu_pool;
            if (pool->used + size <= pool->total_size) {
                void* ptr = (char*)pool->base + pool->used;
                pool->used += size;
                return ptr;
            } else {
                fprintf(stderr, "Out of memory in %s pool\n", is_gpu ? "GPU" : "CPU");
                exit(1);
            }
        }
        node = node->next;
    }
    return NULL;
}

// Visualize memory graph (outputs DOT format)
void visualize_memory_graph(FILE* output) {
    fprintf(output, "digraph MemoryGraph {\n");
    TensorNode* node = memory_graph;
    while (node) {
        fprintf(output, "  %s [label=\"%s\\nSize: %zu\\nLifetime: [%d, %d]\"];\n",
                node->name, node->name, node->size, node->start_time, node->end_time);
        if (node->next) {
            fprintf(output, "  %s -> %s;\n", node->name, node->next->name);
        }
        node = node->next;
    }
    fprintf(output, "}\n");
}

// Cleanup memory graph and pools
void cleanup_memory_planner() {
    TensorNode* node = memory_graph;
    while (node) {
        TensorNode* next = node->next;
        free(node->name);
        free(node);
        node = next;
    }
    memory_graph = NULL;
    if (cpu_pool.base) free(cpu_pool.base);
    if (gpu_pool.base) cudaFree(gpu_pool.base);
}
