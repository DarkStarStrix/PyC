#include "graph_compiler.h"
#include "error_handler.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Structure for computational graph node (from graph_compiler.h)
typedef struct GraphNode {
    char* op; // Operation (e.g., "matmul", "add")
    int* inputs; // Indices of input nodes
    int num_inputs;
    int output_shape[4]; // Tensor shape
    struct GraphNode* next;
} GraphNode;

// Structure for subgraph
typedef struct Subgraph {
    GraphNode* nodes;
    int num_nodes;
} Subgraph;

// External global computational graph from graph_compiler.c
extern GraphNode* comp_graph;

// Helper function to check if two nodes can be fused
static int can_fuse_nodes(GraphNode* node1, GraphNode* node2) {
    // Example: Fuse matmul + add
    if (strcmp(node1->op, "matmul") == 0 && strcmp(node2->op, "add") == 0) {
        return 1;
    }
    // Add more fusion rules as needed
    return 0;
}

// Decompose graph into subgraphs (simple clustering based on connectivity)
Subgraph* decompose_graph(int* num_subgraphs) {
    if (!comp_graph) {
        report_error(0, 0, "No computational graph to decompose");
        return NULL;
    }

    // Simple clustering: Group nodes by operation type for now
    Subgraph* subgraphs = (Subgraph*)malloc(10 * sizeof(Subgraph));
    if (!subgraphs) {
        report_error(0, 0, "Memory allocation failed for subgraphs");
        return NULL;
    }

    *num_subgraphs = 1;
    subgraphs[0].nodes = NULL;
    subgraphs[0].num_nodes = 0;

    // Copy all nodes into one subgraph (placeholder for more sophisticated partitioning)
    GraphNode* current = comp_graph;
    GraphNode* last = NULL;
    while (current) {
        GraphNode* new_node = (GraphNode*)malloc(sizeof(GraphNode));
        if (!new_node) {
            report_error(0, 0, "Memory allocation failed for graph node");
            return NULL;
        }
        new_node->op = strdup(current->op);
        new_node->inputs = (int*)malloc(current->num_inputs * sizeof(int));
        memcpy(new_node->inputs, current->inputs, current->num_inputs * sizeof(int));
        new_node->num_inputs = current->num_inputs;
        memcpy(new_node->output_shape, current->output_shape, 4 * sizeof(int));
        new_node->next = NULL;

        if (!subgraphs[0].nodes) {
            subgraphs[0].nodes = new_node;
        } else {
            last->next = new_node;
        }
        last = new_node;
        subgraphs[0].num_nodes++;
        current = current->next;
    }

    return subgraphs;
}

// Simplify subgraph (operator fusion, constant folding, dead node removal)
void simplify_subgraph(Subgraph* subgraph) {
    if (!subgraph || !subgraph->nodes) return;

    GraphNode* node = subgraph->nodes;
    GraphNode* prev = NULL;

    while (node && node->next) {
        // Operator fusion
        if (can_fuse_nodes(node, node->next)) {
            char* new_op = (char*)malloc(256);
            snprintf(new_op, 256, "fused_%s_%s", node->op, node->next->op);
            free(node->op);
            node->op = new_op;

            // Update inputs and shape
            node->num_inputs = node->num_inputs + node->next->num_inputs - 1; // Adjust for shared input
            int* new_inputs = (int*)malloc(node->num_inputs * sizeof(int));
            memcpy(new_inputs, node->inputs, node->num_inputs * sizeof(int));
            free(node->inputs);
            node->inputs = new_inputs;

            // Remove the fused node
            GraphNode* to_remove = node->next;
            node->next = to_remove->next;
            free(to_remove->op);
            free(to_remove->inputs);
            free(to_remove);
            subgraph->num_nodes--;
            continue;
        }

        // Constant folding (placeholder)
        if (strcmp(node->op, "add") == 0 && node->num_inputs == 2) {
            // Check if inputs are constants (simplified check)
            // If so, fold into a single constant node (not implemented here)
        }

        // Dead node removal (placeholder)
        // Check if node contributes to output (not implemented here)

        prev = node;
        node = node->next;
    }
}

// Select optimal configuration for subgraph
void select_optimal_config(Subgraph* subgraph) {
    if (!subgraph || !subgraph->nodes) return;

    GraphNode* node = subgraph->nodes;
    while (node) {
        // Example: Choose GPU kernel for matmul or fused operations
        if (strncmp(node->op, "matmul", 6) == 0 || strncmp(node->op, "fused_matmul", 12) == 0) {
            // Mark for GPU execution (metadata could be added to node)
            printf("Selected GPU kernel for %s\n", node->op);
        } else {
            // Default to CPU execution
            printf("Selected CPU kernel for %s\n", node->op);
        }
        node = node->next;
    }
}

// Perform shape inference on subgraph
void infer_shapes(Subgraph* subgraph) {
    if (!subgraph || !subgraph->nodes) return;

    GraphNode* node = subgraph->nodes;
    while (node) {
        if (strcmp(node->op, "matmul") == 0 && node->num_inputs == 2) {
            // Assume inputs are matrices: [m, k] and [k, n]
            GraphNode* input1 = comp_graph; // Simplified lookup
            GraphNode* input2 = comp_graph;
            for (int i = 0; input1 && i < node->inputs[0]; i++) input1 = input1->next;
            for (int i = 0; input2 && i < node->inputs[1]; i++) input2 = input2->next;

            if (input1 && input2) {
                node->output_shape[0] = input1->output_shape[0]; // m
                node->output_shape[1] = input2->output_shape[1]; // n
                node->output_shape[2] = 0;
                node->output_shape[3] = 0;
            } else {
                report_warning(0, 0, "Invalid inputs for matmul shape inference");
            }
        } else if (strncmp(node->op, "fused_matmul", 12) == 0) {
            // Similar logic, adjusting for fused operation
            node->output_shape[0] = comp_graph[node->inputs[0]].output_shape[0];
            node->output_shape[1] = comp_graph[node->inputs[1]].output_shape[1];
            node->output_shape[2] = 0;
            node->output_shape[3] = 0;
        }
        node = node->next;
    }
}

// Optimize computational graph (main entry point)
void optimize_graph() {
    int num_subgraphs = 0;
    Subgraph* subgraphs = decompose_graph(&num_subgraphs);
    if (!subgraphs) return;

    for (int i = 0; i < num_subgraphs; i++) {
        simplify_subgraph(&subgraphs[i]);
        infer_shapes(&subgraphs[i]);
        select_optimal_config(&subgraphs[i]);
    }

    // Reconstruct global graph (simplified: assume one subgraph for now)
    comp_graph = subgraphs[0].nodes;
    subgraphs[0].nodes = NULL; // Prevent double-free
    free(subgraphs);
}

// Cleanup optimizer (called by graph_compiler.c)
void cleanup_optimizer() {
    // Cleanup handled by graph_compiler.c
}
