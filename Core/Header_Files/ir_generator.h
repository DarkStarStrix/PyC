#ifndef IR_GENERATOR_H
#define IR_GENERATOR_H

typedef struct IRNode {
    char* op;              // Operation (e.g., "add", "conv2d")
    int* inputs;           // Indices of input tensors
    int num_inputs;        // Number of inputs
    char* output_tensor;   // Name of output tensor
    int start_time;        // Execution start time
    int end_time;          // Execution end time
    size_t mem_offset;     // Memory offset for output tensor
    size_t mem_size;       // Memory size for output tensor
    struct IRNode* next;   // Next IR instruction
} IRNode;

extern IRNode* ir; // Global IR linked list

// Function to generate IR from computational graph and memory planner
void generate_ir();

#endif // IR_GENERATOR_H
