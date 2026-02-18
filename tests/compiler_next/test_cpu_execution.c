#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyc/compiler_api.h"

static int approx_eq(float a, float b) {
    float diff = fabsf(a - b);
    return diff < 1e-6f;
}

static void add_input_op(pyc_ir_module* m, const char* name, int64_t d0, int64_t d1) {
    pyc_ir_op op;
    memset(&op, 0, sizeof(op));
    op.kind = PYC_IR_OP_INPUT;
    strncpy(op.name, name, sizeof(op.name) - 1);
    op.dtype = PYC_DTYPE_F32;
    op.shape.rank = (d1 > 0) ? 2u : 1u;
    op.shape.dims[0] = d0;
    if (d1 > 0) {
        op.shape.dims[1] = d1;
    }
    pyc_ir_add_op(m, &op);
}

static pyc_status run_case(
    pyc_ir_module* module,
    pyc_compile_options* options,
    const pyc_tensor* inputs,
    size_t input_count,
    pyc_tensor* outputs,
    size_t output_count) {
    pyc_compiled_model* model = NULL;
    pyc_model_desc desc;
    pyc_status st;

    memset(&desc, 0, sizeof(desc));
    desc.module = module;
    desc.backend = PYC_BACKEND_CPU;

    st = pyc_compile_model(&desc, options, &model);
    if (st != PYC_STATUS_OK) {
        return st;
    }

    st = pyc_run_model(model, inputs, input_count, outputs, output_count, NULL);
    pyc_destroy_model(model);
    return st;
}

static int test_matmul(void) {
    pyc_ir_module m;
    pyc_compile_options opts;
    pyc_ir_op matmul;
    pyc_ir_op output;
    float lhs[6] = {1, 2, 3, 4, 5, 6};
    float rhs[6] = {1, 2, 3, 4, 5, 6};
    float out[4] = {0, 0, 0, 0};
    float expected[4] = {22, 28, 49, 64};
    pyc_tensor inputs[2];
    pyc_tensor outputs[1];
    size_t i;
    pyc_status st;

    pyc_ir_module_init(&m);
    add_input_op(&m, "lhs", 2, 3);
    add_input_op(&m, "rhs", 3, 2);

    memset(&matmul, 0, sizeof(matmul));
    matmul.kind = PYC_IR_OP_MATMUL;
    strcpy(matmul.name, "matmul0");
    matmul.dtype = PYC_DTYPE_F32;
    matmul.input_ids[0] = 0;
    matmul.input_ids[1] = 1;
    matmul.input_count = 2;
    pyc_ir_add_op(&m, &matmul);

    memset(&output, 0, sizeof(output));
    output.kind = PYC_IR_OP_OUTPUT;
    strcpy(output.name, "out0");
    output.dtype = PYC_DTYPE_F32;
    output.input_ids[0] = 2;
    output.input_count = 1;
    pyc_ir_add_op(&m, &output);

    memset(&opts, 0, sizeof(opts));
    opts.enable_fusion = 0;
    opts.enable_memory_reuse = 1;
    opts.enable_autotune = 0;
    opts.objective_mode = PYC_MODE_BALANCED;
    opts.target_utilization_floor = 0.70;
    opts.deterministic_strict = 1;
    pyc_runtime_rails_default(&opts.rails);

    memset(inputs, 0, sizeof(inputs));
    inputs[0].data = lhs;
    inputs[0].size_bytes = sizeof(lhs);
    inputs[0].dtype = PYC_DTYPE_F32;
    inputs[0].shape.rank = 2;
    inputs[0].shape.dims[0] = 2;
    inputs[0].shape.dims[1] = 3;
    inputs[1].data = rhs;
    inputs[1].size_bytes = sizeof(rhs);
    inputs[1].dtype = PYC_DTYPE_F32;
    inputs[1].shape.rank = 2;
    inputs[1].shape.dims[0] = 3;
    inputs[1].shape.dims[1] = 2;

    memset(outputs, 0, sizeof(outputs));
    outputs[0].data = out;
    outputs[0].size_bytes = sizeof(out);
    outputs[0].dtype = PYC_DTYPE_F32;
    outputs[0].shape.rank = 2;
    outputs[0].shape.dims[0] = 2;
    outputs[0].shape.dims[1] = 2;

    st = run_case(&m, &opts, inputs, 2, outputs, 1);
    if (st != PYC_STATUS_OK) {
        return 1;
    }
    for (i = 0; i < 4; ++i) {
        if (!approx_eq(out[i], expected[i])) {
            return 2;
        }
    }
    return 0;
}

static int test_add_relu(void) {
    pyc_ir_module m;
    pyc_compile_options opts;
    pyc_ir_op add;
    pyc_ir_op relu;
    pyc_ir_op output;
    float a[4] = {-1, 2, -3, 4};
    float b[4] = {2, -5, 4, -1};
    float out[4] = {0, 0, 0, 0};
    float expected[4] = {1, 0, 1, 3};
    pyc_tensor inputs[2];
    pyc_tensor outputs[1];
    size_t i;
    pyc_status st;

    pyc_ir_module_init(&m);
    add_input_op(&m, "a", 4, 0);
    add_input_op(&m, "b", 4, 0);

    memset(&add, 0, sizeof(add));
    add.kind = PYC_IR_OP_ADD;
    strcpy(add.name, "add0");
    add.dtype = PYC_DTYPE_F32;
    add.input_ids[0] = 0;
    add.input_ids[1] = 1;
    add.input_count = 2;
    pyc_ir_add_op(&m, &add);

    memset(&relu, 0, sizeof(relu));
    relu.kind = PYC_IR_OP_RELU;
    strcpy(relu.name, "relu0");
    relu.dtype = PYC_DTYPE_F32;
    relu.input_ids[0] = 2;
    relu.input_count = 1;
    pyc_ir_add_op(&m, &relu);

    memset(&output, 0, sizeof(output));
    output.kind = PYC_IR_OP_OUTPUT;
    strcpy(output.name, "out0");
    output.dtype = PYC_DTYPE_F32;
    output.input_ids[0] = 3;
    output.input_count = 1;
    pyc_ir_add_op(&m, &output);

    memset(&opts, 0, sizeof(opts));
    opts.enable_fusion = 0;
    opts.enable_memory_reuse = 1;
    opts.enable_autotune = 0;
    opts.objective_mode = PYC_MODE_BALANCED;
    opts.target_utilization_floor = 0.70;
    opts.deterministic_strict = 1;
    pyc_runtime_rails_default(&opts.rails);

    memset(inputs, 0, sizeof(inputs));
    inputs[0].data = a;
    inputs[0].size_bytes = sizeof(a);
    inputs[0].dtype = PYC_DTYPE_F32;
    inputs[0].shape.rank = 1;
    inputs[0].shape.dims[0] = 4;
    inputs[1].data = b;
    inputs[1].size_bytes = sizeof(b);
    inputs[1].dtype = PYC_DTYPE_F32;
    inputs[1].shape.rank = 1;
    inputs[1].shape.dims[0] = 4;

    memset(outputs, 0, sizeof(outputs));
    outputs[0].data = out;
    outputs[0].size_bytes = sizeof(out);
    outputs[0].dtype = PYC_DTYPE_F32;
    outputs[0].shape.rank = 1;
    outputs[0].shape.dims[0] = 4;

    st = run_case(&m, &opts, inputs, 2, outputs, 1);
    if (st != PYC_STATUS_OK) {
        return 1;
    }
    for (i = 0; i < 4; ++i) {
        if (!approx_eq(out[i], expected[i])) {
            return 2;
        }
    }
    return 0;
}

int main(void) {
    if (test_matmul() != 0) {
        return 1;
    }
    if (test_add_relu() != 0) {
        return 2;
    }
    printf("test_cpu_execution: ok\n");
    return 0;
}
