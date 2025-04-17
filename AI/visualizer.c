#include "graph_compiler.h"
#include "memory_planner.h"
#include <stdio.h>

// Visualize the computational graph in DOT format
void visualize_comp_graph(FILE* output) {
    if (!comp_graph) {
        fprintf(output, "digraph CompGraph { }\n");
        return;
    }
    
    fprintf(output, "digraph CompGraph {\n");
    
    GraphNode* node = comp_graph;
    int id = 0;
    while (node) {
        fprintf(output, "  node%d [label=\"%s\"];\n", id, node->op);
        for (int i = 0; i < node->num_inputs; i++) {
            int input_index = node->inputs[i];
            fprintf(output, "  node%d -> node%d;\n", input_index, id);
        }
        node = node->next;
        id++;
    }
    
    fprintf(output, "}\n");
}

// Visualize the memory allocation graph in DOT format
void visualize_memory_graph(FILE* output) {
    if (!memory_blocks || num_blocks == 0) {
        fprintf(output, "digraph MemoryGraph { }\n");
        return;
    }
    
    fprintf(output, "digraph MemoryGraph {\n");
    
    for (int block_id = 0; block_id < num_blocks; block_id++) {
        MemoryBlock* block = &memory_blocks[block_id];
        fprintf(output, "  subgraph cluster_block%d {\n", block_id);
        fprintf(output, "    label = \"Memory Block %d (offset: %zu, size: %zu)\";\n", block_id, block->offset, block->size);
        
        for (int i = 0; i < block->num_tensors; i++) {
            TensorNode* tensor = block->tensors[i];
            fprintf(output, "    %s [label=\"%s\\n[%d, %d]\"];\n", tensor->name, tensor->name, tensor->start_time, tensor->end_time);
            if (i > 0) {
                TensorNode* prev_tensor = block->tensors[i-1];
                fprintf(output, "    %s -> %s;\n", prev_tensor->name, tensor->name);
            }
        }
        
        fprintf(output, "  }\n");
    }
    
    fprintf(output, "}\n");
}
