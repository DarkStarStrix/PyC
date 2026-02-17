#include "pyc/ir.h"

#include <stdio.h>
#include <string.h>

void pyc_ir_module_init(pyc_ir_module* module) {
    if (!module) {
        return;
    }
    memset(module, 0, sizeof(*module));
}

int pyc_ir_add_op(pyc_ir_module* module, const pyc_ir_op* op) {
    if (!module || !op) {
        return -1;
    }
    if (module->op_count >= PYC_IR_MAX_OPS) {
        return -1;
    }
    module->ops[module->op_count++] = *op;
    return 0;
}

static int validate_shape(const pyc_shape* shape) {
    size_t i;
    if (shape->rank > PYC_IR_MAX_DIMS) {
        return 0;
    }
    for (i = 0; i < shape->rank; ++i) {
        if (shape->dims[i] <= 0) {
            return 0;
        }
    }
    return 1;
}

int pyc_ir_verify(const pyc_ir_module* module, pyc_ir_diagnostic* diag) {
    size_t i;

    if (diag) {
        memset(diag, 0, sizeof(*diag));
    }

    if (!module) {
        if (diag) {
            diag->code = 1;
            snprintf(diag->message, sizeof(diag->message), "module is NULL");
        }
        return -1;
    }

    if (module->op_count == 0) {
        if (diag) {
            diag->code = 2;
            snprintf(diag->message, sizeof(diag->message), "module has no ops");
        }
        return -1;
    }

    for (i = 0; i < module->op_count; ++i) {
        const pyc_ir_op* op = &module->ops[i];
        size_t j;

        if (op->input_count > PYC_IR_MAX_INPUTS) {
            if (diag) {
                diag->code = 3;
                snprintf(diag->message, sizeof(diag->message), "op %zu has too many inputs", i);
            }
            return -1;
        }

        if (!validate_shape(&op->shape)) {
            if (diag) {
                diag->code = 4;
                snprintf(diag->message, sizeof(diag->message), "op %zu has invalid shape", i);
            }
            return -1;
        }

        for (j = 0; j < op->input_count; ++j) {
            int input_id = op->input_ids[j];
            if (input_id < 0 || (size_t)input_id >= i) {
                if (diag) {
                    diag->code = 5;
                    snprintf(diag->message, sizeof(diag->message), "op %zu has invalid input id %d", i, input_id);
                }
                return -1;
            }
        }
    }

    return 0;
}

static const char* kind_name(pyc_ir_op_kind kind) {
    switch (kind) {
        case PYC_IR_OP_INPUT: return "input";
        case PYC_IR_OP_CONST: return "const";
        case PYC_IR_OP_MATMUL: return "matmul";
        case PYC_IR_OP_ADD: return "add";
        case PYC_IR_OP_RELU: return "relu";
        case PYC_IR_OP_GELU: return "gelu";
        case PYC_IR_OP_REDUCE_SUM: return "reduce_sum";
        case PYC_IR_OP_LAYERNORM: return "layernorm";
        case PYC_IR_OP_OUTPUT: return "output";
        default: return "unknown";
    }
}

int pyc_ir_serialize(const pyc_ir_module* module, char* buffer, size_t buffer_size) {
    size_t i;
    size_t used = 0;

    if (!module || !buffer || buffer_size == 0) {
        return -1;
    }

    buffer[0] = '\0';

    for (i = 0; i < module->op_count; ++i) {
        const pyc_ir_op* op = &module->ops[i];
        int n;
        size_t d;
        size_t in_idx;

        n = snprintf(
            buffer + used,
            buffer_size - used,
            "%zu:%s:%s:dtype=%d:shape=",
            i,
            kind_name(op->kind),
            op->name,
            (int)op->dtype
        );
        if (n < 0 || (size_t)n >= buffer_size - used) {
            return -1;
        }
        used += (size_t)n;

        for (d = 0; d < op->shape.rank; ++d) {
            n = snprintf(buffer + used, buffer_size - used, "%s%lld", d == 0 ? "" : "x", (long long)op->shape.dims[d]);
            if (n < 0 || (size_t)n >= buffer_size - used) {
                return -1;
            }
            used += (size_t)n;
        }

        n = snprintf(buffer + used, buffer_size - used, ":inputs=");
        if (n < 0 || (size_t)n >= buffer_size - used) {
            return -1;
        }
        used += (size_t)n;

        for (in_idx = 0; in_idx < op->input_count; ++in_idx) {
            n = snprintf(buffer + used, buffer_size - used, "%s%d", in_idx == 0 ? "" : ",", op->input_ids[in_idx]);
            if (n < 0 || (size_t)n >= buffer_size - used) {
                return -1;
            }
            used += (size_t)n;
        }

        n = snprintf(buffer + used, buffer_size - used, "\n");
        if (n < 0 || (size_t)n >= buffer_size - used) {
            return -1;
        }
        used += (size_t)n;
    }

    return 0;
}
