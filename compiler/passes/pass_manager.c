#include "pyc/pass_manager.h"

#include <stdio.h>
#include <string.h>

static int is_activation(pyc_ir_op_kind kind) {
    return kind == PYC_IR_OP_RELU || kind == PYC_IR_OP_GELU;
}

static void canonicalize_module(pyc_ir_module* module, pyc_pass_report* report) {
    size_t i;
    for (i = 0; i < module->op_count; ++i) {
        pyc_ir_op* op = &module->ops[i];
        if (op->name[0] == '\0') {
            snprintf(op->name, sizeof(op->name), "op_%zu", i);
            report->transformed_ops++;
        }
    }
}

static int infer_shapes_module(pyc_ir_module* module, pyc_pass_report* report) {
    size_t i;
    for (i = 0; i < module->op_count; ++i) {
        pyc_ir_op* op = &module->ops[i];
        if (op->kind == PYC_IR_OP_OUTPUT || op->kind == PYC_IR_OP_ADD || is_activation(op->kind) || op->kind == PYC_IR_OP_LAYERNORM) {
            if (op->input_count >= 1) {
                const pyc_ir_op* in0 = &module->ops[(size_t)op->input_ids[0]];
                op->shape = in0->shape;
                op->dtype = in0->dtype;
                report->inferred_shapes++;
            }
        } else if (op->kind == PYC_IR_OP_MATMUL) {
            if (op->input_count < 2) {
                report->errors++;
                return -1;
            }
            {
                const pyc_ir_op* lhs = &module->ops[(size_t)op->input_ids[0]];
                const pyc_ir_op* rhs = &module->ops[(size_t)op->input_ids[1]];
                if (lhs->shape.rank != 2 || rhs->shape.rank != 2 || lhs->shape.dims[1] != rhs->shape.dims[0]) {
                    report->errors++;
                    return -1;
                }
                op->shape.rank = 2;
                op->shape.dims[0] = lhs->shape.dims[0];
                op->shape.dims[1] = rhs->shape.dims[1];
                op->dtype = lhs->dtype;
                report->inferred_shapes++;
            }
        }
    }
    return 0;
}

static void rewrite_input_ids_after_remove(pyc_ir_module* module, size_t removed) {
    size_t i;
    for (i = 0; i < module->op_count; ++i) {
        pyc_ir_op* op = &module->ops[i];
        size_t j;
        for (j = 0; j < op->input_count; ++j) {
            if ((size_t)op->input_ids[j] > removed) {
                op->input_ids[j]--;
            }
        }
    }
}

static void replace_consumers(pyc_ir_module* module, int from_id, int to_id) {
    size_t i;
    for (i = 0; i < module->op_count; ++i) {
        pyc_ir_op* op = &module->ops[i];
        size_t j;
        for (j = 0; j < op->input_count; ++j) {
            if (op->input_ids[j] == from_id) {
                op->input_ids[j] = to_id;
            }
        }
    }
}

static void remove_op_at(pyc_ir_module* module, size_t idx) {
    size_t i;
    for (i = idx; i + 1 < module->op_count; ++i) {
        module->ops[i] = module->ops[i + 1];
    }
    module->op_count--;
    rewrite_input_ids_after_remove(module, idx);
}

static void fuse_module(pyc_ir_module* module, pyc_pass_report* report) {
    size_t i = 0;
    while (i + 1 < module->op_count) {
        pyc_ir_op* cur = &module->ops[i];
        pyc_ir_op* nxt = &module->ops[i + 1];
        int can_fuse = 0;

        if (cur->kind == PYC_IR_OP_MATMUL && nxt->kind == PYC_IR_OP_ADD && nxt->input_count >= 1 && nxt->input_ids[0] == (int)i) {
            can_fuse = 1;
        } else if (cur->kind == PYC_IR_OP_MATMUL && is_activation(nxt->kind) && nxt->input_count >= 1 && nxt->input_ids[0] == (int)i) {
            can_fuse = 1;
        }

        if (can_fuse) {
            char fused_name[PYC_IR_MAX_NAME];
            snprintf(fused_name, sizeof(fused_name), "fused_%s_%s", cur->name, nxt->name);
            strncpy(cur->name, fused_name, sizeof(cur->name) - 1);
            cur->name[sizeof(cur->name) - 1] = '\0';
            replace_consumers(module, (int)(i + 1), (int)i);
            remove_op_at(module, i + 1);
            report->fused_patterns++;
            report->transformed_ops++;
            continue;
        }
        i++;
    }
}

static void analyze_liveness(const pyc_ir_module* module, pyc_pass_report* report) {
    size_t i;
    size_t live = 0;
    size_t peak = 0;
    int last_use[PYC_IR_MAX_OPS];

    for (i = 0; i < module->op_count; ++i) {
        last_use[i] = -1;
    }

    for (i = 0; i < module->op_count; ++i) {
        const pyc_ir_op* op = &module->ops[i];
        size_t j;
        for (j = 0; j < op->input_count; ++j) {
            int id = op->input_ids[j];
            if (id >= 0 && (size_t)id < module->op_count) {
                last_use[(size_t)id] = (int)i;
            }
        }
    }

    for (i = 0; i < module->op_count; ++i) {
        live++;
        if (live > peak) {
            peak = live;
        }
        if (last_use[i] == (int)i || last_use[i] == -1) {
            if (live > 0) {
                live--;
            }
        }
    }

    report->peak_live_values = peak;
}

static int is_compiler_next_supported_kind(pyc_ir_op_kind kind) {
    return kind == PYC_IR_OP_INPUT ||
           kind == PYC_IR_OP_MATMUL ||
           kind == PYC_IR_OP_ADD ||
           kind == PYC_IR_OP_RELU ||
           kind == PYC_IR_OP_OUTPUT;
}

static void analyze_graph_breaks(const pyc_ir_module* module, pyc_pass_report* report) {
    size_t i;
    size_t breaks = 0;
    const char* first_reason = "none";
    int first_op_id = -1;
    const char* first_op_name = "none";
    size_t const_count = 0;
    size_t gelu_count = 0;
    size_t reduce_sum_count = 0;
    size_t layernorm_count = 0;
    size_t unknown_count = 0;

    report->first_graph_break_op_id = -1;
    report->first_graph_break_op_name[0] = '\0';

    for (i = 0; i < module->op_count; ++i) {
        const pyc_ir_op* op = &module->ops[i];
        if (!is_compiler_next_supported_kind(op->kind)) {
            const char* reason = "unknown";
            breaks++;
            switch (op->kind) {
                case PYC_IR_OP_CONST:
                    const_count++;
                    reason = "const";
                    break;
                case PYC_IR_OP_GELU:
                    gelu_count++;
                    reason = "gelu";
                    break;
                case PYC_IR_OP_REDUCE_SUM:
                    reduce_sum_count++;
                    reason = "reduce_sum";
                    break;
                case PYC_IR_OP_LAYERNORM:
                    layernorm_count++;
                    reason = "layernorm";
                    break;
                default:
                    unknown_count++;
                    reason = "unknown";
                    break;
            }
            if (first_op_id < 0) {
                first_reason = reason;
                first_op_id = (int)i;
                first_op_name = op->name[0] ? op->name : "unnamed";
            }
        }
    }

    report->graph_break_count = breaks;
    report->graph_break_const_count = const_count;
    report->graph_break_gelu_count = gelu_count;
    report->graph_break_reduce_sum_count = reduce_sum_count;
    report->graph_break_layernorm_count = layernorm_count;
    report->graph_break_unknown_count = unknown_count;
    report->first_graph_break_op_id = first_op_id;
    strncpy(
        report->first_graph_break_op_name,
        first_op_name,
        sizeof(report->first_graph_break_op_name) - 1);
    report->first_graph_break_op_name[sizeof(report->first_graph_break_op_name) - 1] = '\0';
    if (module->op_count == 0) {
        report->compilability_score = 0.0;
    } else {
        report->compilability_score =
            ((double)(module->op_count - breaks) / (double)module->op_count);
    }
    snprintf(
        report->graph_break_summary,
        sizeof(report->graph_break_summary),
        "breaks=%zu first=%s@%d c=%zu g=%zu r=%zu l=%zu u=%zu",
        breaks,
        first_reason,
        first_op_id,
        const_count,
        gelu_count,
        reduce_sum_count,
        layernorm_count,
        unknown_count);
}

void pyc_pass_pipeline_default(pyc_pass_pipeline* pipeline) {
    if (!pipeline) {
        return;
    }
    memset(pipeline, 0, sizeof(*pipeline));
    pipeline->config.canonicalize = 1;
    pipeline->config.shape_inference = 1;
    pipeline->config.layout_propagation = 1;
    pipeline->config.fusion = 1;
    pipeline->config.liveness = 1;
    pipeline->config.allocation = 1;
    pipeline->config.lowering = 1;
}

int pyc_pass_pipeline_run(pyc_pass_pipeline* pipeline, pyc_ir_module* module, pyc_pass_report* report) {
    if (!pipeline || !module || !report) {
        return -1;
    }

    memset(report, 0, sizeof(*report));

    if (module->op_count == 0) {
        report->errors = 1;
        return -1;
    }

    if (pipeline->config.canonicalize) {
        canonicalize_module(module, report);
        report->passes_run++;
    }
    if (pipeline->config.shape_inference) {
        if (infer_shapes_module(module, report) != 0) {
            return -1;
        }
        report->passes_run++;
    }
    if (pipeline->config.layout_propagation) {
        report->passes_run++;
    }
    if (pipeline->config.fusion) {
        fuse_module(module, report);
        report->passes_run++;
    }
    if (pipeline->config.liveness) {
        analyze_liveness(module, report);
        report->passes_run++;
    }
    if (pipeline->config.allocation) {
        report->passes_run++;
    }
    if (pipeline->config.lowering) {
        report->passes_run++;
    }
    analyze_graph_breaks(module, report);

    return 0;
}
