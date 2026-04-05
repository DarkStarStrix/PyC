#include "pyc/pass_manager.h"

#include <stdio.h>
#include <string.h>

static size_t dtype_size_bytes(pyc_dtype dtype) {
    switch (dtype) {
        case PYC_DTYPE_F16:
            return 2;
        case PYC_DTYPE_I8:
            return 1;
        case PYC_DTYPE_F32:
        case PYC_DTYPE_I32:
            return 4;
        case PYC_DTYPE_UNKNOWN:
        default:
            return 0;
    }
}

static size_t shape_num_elements(const pyc_shape* shape) {
    size_t i;
    size_t elems = 1;
    if (!shape || shape->rank == 0) {
        return 0;
    }
    for (i = 0; i < shape->rank; ++i) {
        if (shape->dims[i] <= 0) {
            return 0;
        }
        elems *= (size_t)shape->dims[i];
    }
    return elems;
}

static void describe_speculative_shape_bucket(
    size_t total_input_bytes,
    size_t input_count,
    char* out_bucket,
    size_t out_bucket_size) {
    const char* klass = "unknown";
    if (!out_bucket || out_bucket_size == 0) {
        return;
    }
    if (total_input_bytes > 0 && total_input_bytes <= 256) {
        klass = "tiny";
    } else if (total_input_bytes <= 4096) {
        klass = "small";
    } else if (total_input_bytes <= 65536) {
        klass = "medium";
    } else if (total_input_bytes > 65536) {
        klass = "large";
    }
    snprintf(out_bucket, out_bucket_size, "%s:i%zu:b%zu", klass, input_count, total_input_bytes);
}

static void analyze_speculative_plans(const pyc_ir_module* module, pyc_pass_report* report) {
    size_t i;
    size_t input_count = 0;
    size_t total_input_bytes = 0;
    double confidence;

    if (!module || !report) {
        return;
    }

    for (i = 0; i < module->op_count; ++i) {
        const pyc_ir_op* op = &module->ops[i];
        size_t elems;
        size_t elem_bytes;
        if (op->kind != PYC_IR_OP_INPUT) {
            continue;
        }
        elems = shape_num_elements(&op->shape);
        elem_bytes = dtype_size_bytes(op->dtype);
        input_count++;
        if (elems > 0 && elem_bytes > 0) {
            total_input_bytes += elems * elem_bytes;
        }
    }

    describe_speculative_shape_bucket(
        total_input_bytes,
        input_count,
        report->speculative_shape_bucket,
        sizeof(report->speculative_shape_bucket));
    describe_speculative_shape_bucket(
        total_input_bytes,
        input_count,
        report->phantom_shape_bucket,
        sizeof(report->phantom_shape_bucket));

    report->phantom_shape_signature[0] = '\0';
    for (i = 0; i < module->op_count; ++i) {
        const pyc_ir_op* op = &module->ops[i];
        size_t used;
        int n;
        size_t d;
        if (op->kind != PYC_IR_OP_INPUT) {
            continue;
        }
        used = strlen(report->phantom_shape_signature);
        n = snprintf(
            report->phantom_shape_signature + used,
            sizeof(report->phantom_shape_signature) - used,
            "%s%d:r%zu",
            used == 0 ? "" : ";",
            (int)op->dtype,
            op->shape.rank);
        if (n < 0 || (size_t)n >= sizeof(report->phantom_shape_signature) - used) {
            report->phantom_shape_signature[0] = '\0';
            break;
        }
        used += (size_t)n;
        for (d = 0; d < op->shape.rank; ++d) {
            n = snprintf(
                report->phantom_shape_signature + used,
                sizeof(report->phantom_shape_signature) - used,
                "x%lld",
                (long long)op->shape.dims[d]);
            if (n < 0 || (size_t)n >= sizeof(report->phantom_shape_signature) - used) {
                report->phantom_shape_signature[0] = '\0';
                break;
            }
            used += (size_t)n;
        }
        if (report->phantom_shape_signature[0] == '\0') {
            break;
        }
    }

    confidence = report->compilability_score;
    if (report->graph_break_count > 0) {
        confidence *= 0.65;
    }
    if (report->peak_live_values > 8) {
        confidence *= 0.95;
    }
    if (confidence < 0.0) {
        confidence = 0.0;
    }
    if (confidence > 1.0) {
        confidence = 1.0;
    }
    report->speculative_confidence = confidence;
    report->phantom_confidence = confidence;
    if (report->graph_break_count == 0) {
        report->speculative_plan_count = 3;
    } else if (report->compilability_score >= 0.75) {
        report->speculative_plan_count = 2;
    } else {
        report->speculative_plan_count = 1;
    }
}

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
    analyze_speculative_plans(module, report);

    return 0;
}
