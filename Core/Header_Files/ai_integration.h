#ifndef PYC_AI_INTEGRATION_H
#define PYC_AI_INTEGRATION_H

#include <stddef.h>

/* Graph extraction format passed from core IR to AI modules. */
typedef struct {
    const char* op_name;
    const char** input_ids;
    size_t input_count;
    const char* output_id;
} GraphExtractionNode;

typedef struct {
    const GraphExtractionNode* nodes;
    size_t node_count;
} GraphExtractionPayload;

/* Memory planning contract. */
typedef struct {
    const char* tensor_id;
    size_t bytes;
    int first_use_index;
    int last_use_index;
} MemoryPlanningInput;

typedef struct {
    const char* tensor_id;
    size_t offset;
    size_t bytes;
} MemoryPlanningOutput;

/* Optimizer pass registration contract. */
typedef int (*AIPassFn)(const GraphExtractionPayload* graph, char* err, size_t err_size);

typedef struct {
    const char* pass_name;
    AIPassFn run;
} AIPassRegistration;

#endif
