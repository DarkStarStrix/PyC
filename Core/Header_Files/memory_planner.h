#ifndef MEMORY_PLANNER_H
#define MEMORY_PLANNER_H

typedef struct TensorNode {
    char* name;            // Tensor identifier
    size_t size;           // Size in bytes
    int start_time;        // Lifetime start
    int end_time;          // Lifetime end
    struct TensorNode* next; // Next tensor in list
} TensorNode;

typedef struct MemoryBlock {
    size_t offset;         // Offset in memory pool
    size_t size;           // Size of the block
    TensorNode** tensors;  // Array of tensors allocated to this block
    int num_tensors;       // Number of tensors in the block
} MemoryBlock;

extern MemoryBlock* memory_blocks; // Global array of memory blocks
extern int num_blocks;            // Number of memory blocks

#endif // MEMORY_PLANNER_H
