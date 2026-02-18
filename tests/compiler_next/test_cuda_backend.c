#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pyc/compiler_api.h"

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
        return 9;
    }

    set_env_flag("PYC_CUDA_SIMULATE_AVAILABLE", "0");
    set_env_flag("PYC_CUDA_DISABLE", "0");
    set_env_flag("PYC_CUDA_FORCE_FALLBACK", "0");
    set_env_flag("PYC_CUDA_FORCE_ERROR", "0");
    pyc_destroy_model(model);
    printf("test_cuda_backend: ok\n");
    return 0;
}
