#ifndef PYC_PASS_MANAGER_H
#define PYC_PASS_MANAGER_H

#include "pyc/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int canonicalize;
    int shape_inference;
    int layout_propagation;
    int fusion;
    int liveness;
    int allocation;
    int lowering;
} pyc_pass_pipeline_config;

typedef struct {
    int passes_run;
    int warnings;
    int errors;
    int transformed_ops;
    int fused_patterns;
    int inferred_shapes;
    size_t peak_live_values;
    size_t graph_break_count;
    size_t graph_break_const_count;
    size_t graph_break_gelu_count;
    size_t graph_break_reduce_sum_count;
    size_t graph_break_layernorm_count;
    size_t graph_break_unknown_count;
    int first_graph_break_op_id;
    char first_graph_break_op_name[PYC_IR_MAX_NAME];
    double compilability_score;
    char graph_break_summary[128];
} pyc_pass_report;

typedef struct {
    pyc_pass_pipeline_config config;
} pyc_pass_pipeline;

void pyc_pass_pipeline_default(pyc_pass_pipeline* pipeline);
int pyc_pass_pipeline_run(pyc_pass_pipeline* pipeline, pyc_ir_module* module, pyc_pass_report* report);

#ifdef __cplusplus
}
#endif

#endif
