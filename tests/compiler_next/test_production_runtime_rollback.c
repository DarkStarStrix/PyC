#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pyc/compiler_api.h"

static void set_env_flag(const char* key, const char* value) {
#if defined(_WIN32)
    _putenv_s(key, value);
#else
    setenv(key, value, 1);
#endif
}

static void build_module(pyc_ir_module* module) {
    pyc_ir_op in;
    pyc_ir_op out;

    pyc_ir_module_init(module);

    memset(&in, 0, sizeof(in));
    in.kind = PYC_IR_OP_INPUT;
    strcpy(in.name, "input0");
    in.dtype = PYC_DTYPE_F32;
    in.shape.rank = 1;
    in.shape.dims[0] = 4;
    pyc_ir_add_op(module, &in);

    memset(&out, 0, sizeof(out));
    out.kind = PYC_IR_OP_OUTPUT;
    strcpy(out.name, "output0");
    out.dtype = PYC_DTYPE_F32;
    out.shape.rank = 1;
    out.shape.dims[0] = 4;
    out.input_ids[0] = 0;
    out.input_count = 1;
    pyc_ir_add_op(module, &out);
}

int main(void) {
    pyc_ir_module module;
    pyc_model_desc desc;
    pyc_compile_options opts;
    pyc_compiled_model* model = NULL;
    pyc_tensor in;
    pyc_tensor out;
    pyc_run_stats stats;
    pyc_status st;
    float in_data[4] = {1, 2, 3, 4};
    float out_data[4] = {0, 0, 0, 0};

    build_module(&module);
    memset(&desc, 0, sizeof(desc));
    desc.module = &module;
    desc.backend = PYC_BACKEND_CUDA;

    memset(&opts, 0, sizeof(opts));
    opts.enable_fusion = 1;
    opts.enable_memory_reuse = 1;
    opts.enable_autotune = 0;
    opts.objective_mode = PYC_MODE_UTILIZATION_FIRST;
    opts.deterministic_strict = 0;
    pyc_runtime_rails_default(&opts.rails);
    opts.rails.enable_hard_rollback = 1;
    opts.rails.enable_auto_switch = 1;
    opts.rails.consecutive_breach_windows = 1;
    opts.rails.cooldown_steps = 0;
    opts.rails.dwell_steps = 0;

    st = pyc_compile_model(&desc, &opts, &model);
    if (st != PYC_STATUS_OK || !model) return 1;

    memset(&in, 0, sizeof(in));
    in.data = in_data;
    in.size_bytes = sizeof(in_data);
    in.dtype = PYC_DTYPE_F32;
    in.shape.rank = 1;
    in.shape.dims[0] = 4;

    memset(&out, 0, sizeof(out));
    out.data = out_data;
    out.size_bytes = sizeof(out_data);
    out.dtype = PYC_DTYPE_F32;
    out.shape.rank = 1;
    out.shape.dims[0] = 4;

    set_env_flag("PYC_CUDA_DISABLE", "0");
    set_env_flag("PYC_CUDA_FORCE_ERROR", "1");
    set_env_flag("PYC_CUDA_FORCE_FALLBACK", "0");
    set_env_flag("PYC_CUDA_SIMULATE_AVAILABLE", "0");

    st = pyc_run_model(model, &in, 1, &out, 1, &stats);
    if (st != PYC_STATUS_RUNTIME_FAILED) {
        pyc_destroy_model(model);
        return 2;
    }
    if (stats.rollback_reason != PYC_ROLLBACK_RUNTIME_ERROR) {
        pyc_destroy_model(model);
        return 3;
    }
    if (stats.rollback_count < 1) {
        pyc_destroy_model(model);
        return 4;
    }
    if (stats.active_mode != PYC_MODE_BALANCED) {
        pyc_destroy_model(model);
        return 5;
    }

    set_env_flag("PYC_CUDA_FORCE_ERROR", "0");
    pyc_destroy_model(model);
    printf("test_production_runtime_rollback: ok\n");
    return 0;
}
