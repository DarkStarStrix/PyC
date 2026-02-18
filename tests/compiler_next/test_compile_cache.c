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
    in.shape.dims[0] = 8;
    pyc_ir_add_op(module, &in);

    memset(&out, 0, sizeof(out));
    out.kind = PYC_IR_OP_OUTPUT;
    strcpy(out.name, "output0");
    out.dtype = PYC_DTYPE_F32;
    out.shape.rank = 1;
    out.shape.dims[0] = 8;
    out.input_ids[0] = 0;
    out.input_count = 1;
    pyc_ir_add_op(module, &out);
}

static int compile_and_run(
    const pyc_ir_module* module,
    const pyc_compile_options* options,
    pyc_run_stats* stats_out) {
    pyc_model_desc desc;
    pyc_compiled_model* model = NULL;
    pyc_status st;
    pyc_tensor in;
    pyc_tensor out;
    float in_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float out_data[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    memset(&desc, 0, sizeof(desc));
    desc.module = module;
    desc.backend = PYC_BACKEND_CPU;

    st = pyc_compile_model(&desc, options, &model);
    if (st != PYC_STATUS_OK) {
        return 1;
    }

    memset(&in, 0, sizeof(in));
    in.data = in_data;
    in.size_bytes = sizeof(in_data);
    in.dtype = PYC_DTYPE_F32;
    in.shape.rank = 1;
    in.shape.dims[0] = 8;

    memset(&out, 0, sizeof(out));
    out.data = out_data;
    out.size_bytes = sizeof(out_data);
    out.dtype = PYC_DTYPE_F32;
    out.shape.rank = 1;
    out.shape.dims[0] = 8;

    st = pyc_run_model(model, &in, 1, &out, 1, stats_out);
    pyc_destroy_model(model);
    if (st != PYC_STATUS_OK) {
        return 2;
    }
    if (memcmp(in_data, out_data, sizeof(in_data)) != 0) {
        return 3;
    }
    return 0;
}

int main(void) {
    pyc_ir_module module;
    pyc_compile_options options;
    pyc_run_stats stats;
    int rc;

    build_module(&module);
    memset(&options, 0, sizeof(options));
    options.enable_fusion = 1;
    options.enable_memory_reuse = 1;
    options.enable_autotune = 0;
    options.objective_mode = PYC_MODE_BALANCED;
    options.target_utilization_floor = 0.7;
    options.deterministic_strict = 1;
    options.compile_budget_ms = 1.0;
    options.cache_mode = PYC_COMPILE_CACHE_IN_MEMORY;
    pyc_runtime_rails_default(&options.rails);

    set_env_flag("PYC_COMPILE_DELAY_MS", "20");
    rc = compile_and_run(&module, &options, &stats);
    if (rc != 0) return 1;
    if (stats.compile_cache_hit != 0) return 2;
    if (stats.compile_budget_exceeded != 1) return 3;

    set_env_flag("PYC_COMPILE_DELAY_MS", "0");
    rc = compile_and_run(&module, &options, &stats);
    if (rc != 0) return 4;
    if (stats.compile_cache_hit != 1) return 5;
    if (stats.compile_budget_exceeded != 0) return 6;

    set_env_flag("PYC_COMPILE_DELAY_MS", "0");
    printf(
        "test_compile_cache: ok (cache_hit=%d budget_exceeded=%d)\n",
        stats.compile_cache_hit,
        stats.compile_budget_exceeded);
    return 0;
}
