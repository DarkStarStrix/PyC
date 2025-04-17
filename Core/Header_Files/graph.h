#ifndef GRAPH_COMPILER_H
#define GRAPH_COMPILER_H

typedef struct GraphNode {
    char* op;              // Operation name (e.g., "add", "conv2d")
    int* inputs;           // Array of input node indices
    int num_inputs;        // Number of inputs
    int output_shape[4];   // Shape of the output tensor
    struct GraphNode* next; // Next node in the linked list
} GraphNode;

extern GraphNode* comp_graph; // Global computational graph

#endif // GRAPH_COMPILER_H
