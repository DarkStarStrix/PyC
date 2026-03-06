#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pyc/compiler_api.h"

static int approx_eq(float a, float b) {
    float diff = a - b;
    if (diff < 0.0f) {
        diff = -diff;
    }
    return diff < 1e-5f;
}

static void build_passthrough_module(pyc_ir_module* m) {
    pyc_ir_op in;
    pyc_ir_op out;

    pyc_ir_module_init(m);

    memset(&in, 0, sizeof(in));
    in.kind = PYC_IR_OP_INPUT;
    strcpy(in.name, "input0");
    in.dtype = PYC_DTYPE_F32;
    in.shape.rank = 1;
    in.shape.dims[0] = 4;
    pyc_ir_add_op(m, &in);

    memset(&out, 0, sizeof(out));
    out.kind = PYC_IR_OP_OUTPUT;
    strcpy(out.name, "output0");
    out.dtype = PYC_DTYPE_F32;
    out.shape.rank = 1;
    out.shape.dims[0] = 4;
    out.input_ids[0] = 0;
    out.input_count = 1;
    pyc_ir_add_op(m, &out);
}

static void set_env_flag(const char* key, const char* value) {
#if defined(_WIN32)
    _putenv_s(key, value);
#else
    setenv(key, value, 1);
#endif
}

static int run_matmul_add_relu_case(void) {
    pyc_ir_module module;
    pyc_ir_op op;
    pyc_model_desc desc;
    pyc_compile_options options;
    pyc_compiled_model* model = NULL;
    pyc_status st;
    pyc_tensor inputs[3];
    pyc_tensor outputs[1];
    pyc_run_stats stats;
    float lhs[4] = {1.0f, -2.0f, 3.0f, 4.0f};
    float rhs[4] = {2.0f, 1.0f, -1.0f, 3.0f};
    float bias[2] = {0.5f, -1.5f};
    float out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float expected[4] = {4.5f, 0.0f, 2.5f, 13.5f};
    size_t i;
    const char* reason;

    pyc_ir_module_init(&module);

    memset(&op, 0, sizeof(op));
    op.kind = PYC_IR_OP_INPUT;
    strcpy(op.name, "lhs");
    op.dtype = PYC_DTYPE_F32;
    op.shape.rank = 2;
    op.shape.dims[0] = 2;
    op.shape.dims[1] = 2;
    pyc_ir_add_op(&module, &op);

    memset(&op, 0, sizeof(op));
    op.kind = PYC_IR_OP_INPUT;
    strcpy(op.name, "rhs");
    op.dtype = PYC_DTYPE_F32;
    op.shape.rank = 2;
    op.shape.dims[0] = 2;
    op.shape.dims[1] = 2;
    pyc_ir_add_op(&module, &op);

    memset(&op, 0, sizeof(op));
    op.kind = PYC_IR_OP_INPUT;
    strcpy(op.name, "bias");
    op.dtype = PYC_DTYPE_F32;
    op.shape.rank = 1;
    op.shape.dims[0] = 2;
    pyc_ir_add_op(&module, &op);

    memset(&op, 0, sizeof(op));
    op.kind = PYC_IR_OP_MATMUL;
    strcpy(op.name, "matmul0");
    op.dtype = PYC_DTYPE_F32;
    op.input_ids[0] = 0;
    op.input_ids[1] = 1;
    op.input_count = 2;
    pyc_ir_add_op(&module, &op);

    memset(&op, 0, sizeof(op));
    op.kind = PYC_IR_OP_ADD;
    strcpy(op.name, "add0");
    op.dtype = PYC_DTYPE_F32;
    op.input_ids[0] = 3;
    op.input_ids[1] = 2;
    op.input_count = 2;
    pyc_ir_add_op(&module, &op);

    memset(&op, 0, sizeof(op));
    op.kind = PYC_IR_OP_RELU;
    strcpy(op.name, "relu0");
    op.dtype = PYC_DTYPE_F32;
    op.input_ids[0] = 4;
    op.input_count = 1;
    pyc_ir_add_op(&module, &op);

    memset(&op, 0, sizeof(op));
    op.kind = PYC_IR_OP_OUTPUT;
    strcpy(op.name, "out0");
    op.dtype = PYC_DTYPE_F32;
    op.shape.rank = 2;
    op.shape.dims[0] = 2;
    op.shape.dims[1] = 2;
    op.input_ids[0] = 5;
    op.input_count = 1;
    pyc_ir_add_op(&module, &op);

    memset(&desc, 0, sizeof(desc));
    desc.module = &module;
    desc.backend = PYC_BACKEND_CUDA;

    memset(&options, 0, sizeof(options));
    options.enable_fusion = 1;
    options.enable_memory_reuse = 1;
    options.enable_autotune = 0;
    options.objective_mode = PYC_MODE_BALANCED;
    options.target_utilization_floor = 0.70;
    options.deterministic_strict = 1;
    pyc_runtime_rails_default(&options.rails);

    st = pyc_compile_model(&desc, &options, &model);
    if (st != PYC_STATUS_OK || !model) return 21;

    memset(inputs, 0, sizeof(inputs));
    inputs[0].data = lhs;
    inputs[0].size_bytes = sizeof(lhs);
    inputs[0].dtype = PYC_DTYPE_F32;
    inputs[0].shape.rank = 2;
    inputs[0].shape.dims[0] = 2;
    inputs[0].shape.dims[1] = 2;

    inputs[1].data = rhs;
    inputs[1].size_bytes = sizeof(rhs);
    inputs[1].dtype = PYC_DTYPE_F32;
    inputs[1].shape.rank = 2;
    inputs[1].shape.dims[0] = 2;
    inputs[1].shape.dims[1] = 2;

    inputs[2].data = bias;
    inputs[2].size_bytes = sizeof(bias);
    inputs[2].dtype = PYC_DTYPE_F32;
    inputs[2].shape.rank = 1;
    inputs[2].shape.dims[0] = 2;

    memset(outputs, 0, sizeof(outputs));
    outputs[0].data = out;
    outputs[0].size_bytes = sizeof(out);
    outputs[0].dtype = PYC_DTYPE_F32;
    outputs[0].shape.rank = 2;
    outputs[0].shape.dims[0] = 2;
    outputs[0].shape.dims[1] = 2;

    set_env_flag("PYC_CUDA_SIMULATE_AVAILABLE", "1");
    set_env_flag("PYC_CUDA_DISABLE", "0");
    set_env_flag("PYC_CUDA_FORCE_FALLBACK", "0");
    set_env_flag("PYC_CUDA_FORCE_ERROR", "0");
    set_env_flag("PYC_CUDA_ENABLE_GRAPH_REPLAY", "1");
    set_env_flag("PYC_CUDA_ASSUME_STATIC_RHS", "1");

    st = pyc_run_model(model, inputs, 3, outputs, 1, &stats);
    if (st != PYC_STATUS_OK) {
        pyc_destroy_model(model);
        return 22;
    }
    if (stats.fallback_count != 0) {
        pyc_destroy_model(model);
        return 23;
    }
    reason = pyc_model_last_decision_log(model);
    if (strstr(reason, "cuda_fallback=0") == NULL) {
        pyc_destroy_model(model);
        return 24;
    }
    for (i = 0; i < 4; ++i) {
        if (!approx_eq(out[i], expected[i])) {
            pyc_destroy_model(model);
            return 25;
        }
    }

    pyc_destroy_model(model);
    return 0;
}

int main(void) {
    pyc_ir_module module;
    pyc_model_desc desc;
    pyc_compile_options options;
    pyc_compiled_model* model = NULL;
    pyc_run_stats stats;
    pyc_status st;
    pyc_tensor input;
    pyc_tensor output;
    float in_data[4] = {1.0f, -2.0f, 3.0f, -4.0f};
    float out_data[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    build_passthrough_module(&module);

    memset(&desc, 0, sizeof(desc));
    desc.module = &module;
    desc.backend = PYC_BACKEND_CUDA;

    memset(&options, 0, sizeof(options));
    options.enable_fusion = 1;
    options.enable_memory_reuse = 1;
    options.enable_autotune = 0;
    options.objective_mode = PYC_MODE_BALANCED;
    options.target_utilization_floor = 0.70;
    options.deterministic_strict = 1;
    pyc_runtime_rails_default(&options.rails);

    st = pyc_compile_model(&desc, &options, &model);
    if (st != PYC_STATUS_OK || !model) return 1;

    memset(&input, 0, sizeof(input));
    input.data = in_data;
    input.size_bytes = sizeof(in_data);
    input.dtype = PYC_DTYPE_F32;
    input.shape.rank = 1;
    input.shape.dims[0] = 4;

    memset(&output, 0, sizeof(output));
    output.data = out_data;
    output.size_bytes = sizeof(out_data);
    output.dtype = PYC_DTYPE_F32;
    output.shape.rank = 1;
    output.shape.dims[0] = 4;

    set_env_flag("PYC_CUDA_SIMULATE_AVAILABLE", "1");
    set_env_flag("PYC_CUDA_DISABLE", "0");
    set_env_flag("PYC_CUDA_FORCE_FALLBACK", "0");
    set_env_flag("PYC_CUDA_FORCE_ERROR", "0");
    st = pyc_run_model(model, &input, 1, &output, 1, &stats);
    if (st != PYC_STATUS_OK) {
        pyc_destroy_model(model);
        return 2;
    }
    if (memcmp(in_data, out_data, sizeof(in_data)) != 0) {
        pyc_destroy_model(model);
        return 3;
    }
    if (stats.fallback_count != 0) {
        pyc_destroy_model(model);
        return 4;
    }
    if (stats.guard_miss_count != 0) {
        pyc_destroy_model(model);
        return 5;
    }
    if (strstr(pyc_model_last_decision_log(model), "cuda_fallback=0") == NULL) {
        pyc_destroy_model(model);
        return 6;
    }
    if (run_matmul_add_relu_case() != 0) {
        pyc_destroy_model(model);
        return 10;
    }

    set_env_flag("PYC_CUDA_FORCE_FALLBACK", "1");
    st = pyc_run_model(model, &input, 1, &output, 1, &stats);
    if (st != PYC_STATUS_OK) {
        pyc_destroy_model(model);
        return 7;
    }
    if (stats.fallback_count == 0) {
        pyc_destroy_model(model);
        return 8;
    }

    set_env_flag("PYC_CUDA_FORCE_ERROR", "1");
    st = pyc_run_model(model, &input, 1, &output, 1, NULL);
    if (st != PYC_STATUS_RUNTIME_FAILED) {
        pyc_destroy_model(model);
        return 11;
    }

    set_env_flag("PYC_CUDA_SIMULATE_AVAILABLE", "0");
    set_env_flag("PYC_CUDA_DISABLE", "0");
    set_env_flag("PYC_CUDA_FORCE_FALLBACK", "0");
    set_env_flag("PYC_CUDA_FORCE_ERROR", "0");
    pyc_destroy_model(model);
    printf("test_cuda_backend: ok\n");
    return 0;
}
