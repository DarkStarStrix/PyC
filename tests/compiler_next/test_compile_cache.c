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
    pyc_ir_op a;
    pyc_ir_op b;
    pyc_ir_op mm;
    pyc_ir_op out;

    pyc_ir_module_init(module);

    memset(&a, 0, sizeof(a));
    a.kind = PYC_IR_OP_INPUT;
    strcpy(a.name, "a");
    a.dtype = PYC_DTYPE_F32;
    a.shape.rank = 2;
    a.shape.dims[0] = 2;
    a.shape.dims[1] = 2;
    pyc_ir_add_op(module, &a);

    memset(&b, 0, sizeof(b));
    b.kind = PYC_IR_OP_INPUT;
    strcpy(b.name, "b");
    b.dtype = PYC_DTYPE_F32;
    b.shape.rank = 2;
    b.shape.dims[0] = 2;
    b.shape.dims[1] = 2;
    pyc_ir_add_op(module, &b);

    memset(&mm, 0, sizeof(mm));
    mm.kind = PYC_IR_OP_MATMUL;
    strcpy(mm.name, "mm");
    mm.dtype = PYC_DTYPE_F32;
    mm.input_ids[0] = 0;
    mm.input_ids[1] = 1;
    mm.input_count = 2;
    pyc_ir_add_op(module, &mm);

    memset(&out, 0, sizeof(out));
    out.kind = PYC_IR_OP_OUTPUT;
    strcpy(out.name, "out");
    out.dtype = PYC_DTYPE_F32;
    out.input_ids[0] = 2;
    out.input_count = 1;
    pyc_ir_add_op(module, &out);
}

static int compile_and_run(
    const pyc_ir_module* module,
    const pyc_compile_options* options,
    size_t runtime_dim,
    pyc_run_stats* stats_out) {
    pyc_model_desc desc;
    pyc_compiled_model* model = NULL;
    pyc_status st;
    pyc_tensor in;
    pyc_tensor rhs;
    pyc_tensor out;
    float in_data[256];
    float rhs_data[256];
    float out_data[256];
    size_t io_bytes = runtime_dim * runtime_dim * sizeof(float);
    size_t i;

    memset(&desc, 0, sizeof(desc));
    desc.module = module;
    desc.backend = PYC_BACKEND_CPU;

    st = pyc_compile_model(&desc, options, &model);
    if (st != PYC_STATUS_OK) {
        return 1;
    }

    for (i = 0; i < 256; ++i) {
        in_data[i] = (float)(i + 1);
        rhs_data[i] = 0.0f;
        out_data[i] = 0.0f;
    }
    for (i = 0; i < runtime_dim; ++i) {
        rhs_data[(i * runtime_dim) + i] = 1.0f;
    }

    memset(&in, 0, sizeof(in));
    in.data = in_data;
    in.size_bytes = io_bytes;
    in.dtype = PYC_DTYPE_F32;
    in.shape.rank = 2;
    in.shape.dims[0] = (int64_t)runtime_dim;
    in.shape.dims[1] = (int64_t)runtime_dim;

    memset(&rhs, 0, sizeof(rhs));
    rhs.data = rhs_data;
    rhs.size_bytes = io_bytes;
    rhs.dtype = PYC_DTYPE_F32;
    rhs.shape.rank = 2;
    rhs.shape.dims[0] = (int64_t)runtime_dim;
    rhs.shape.dims[1] = (int64_t)runtime_dim;

    memset(&out, 0, sizeof(out));
    out.data = out_data;
    out.size_bytes = io_bytes;
    out.dtype = PYC_DTYPE_F32;
    out.shape.rank = 2;
    out.shape.dims[0] = (int64_t)runtime_dim;
    out.shape.dims[1] = (int64_t)runtime_dim;

    st = pyc_run_model(model, (pyc_tensor[]){in, rhs}, 2, &out, 1, stats_out);
    pyc_destroy_model(model);
    if (st != PYC_STATUS_OK) {
        return 2;
    }
    if (memcmp(in_data, out_data, io_bytes) != 0) {
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
    options.enable_speculative_plans = 1;
    options.enable_phantom_graph = 1;
    options.max_speculative_plans = 3;
    options.phantom_horizon_steps = 1;
    options.objective_mode = PYC_MODE_BALANCED;
    options.target_utilization_floor = 0.7;
    options.deterministic_strict = 1;
    options.compile_budget_ms = 1.0;
    options.cache_mode = PYC_COMPILE_CACHE_IN_MEMORY;
    pyc_runtime_rails_default(&options.rails);

    set_env_flag("PYC_COMPILE_DELAY_MS", "20");
    rc = compile_and_run(&module, &options, 8, &stats);
    if (rc != 0) return 1;
    if (stats.compile_cache_hit != 0) return 2;
    if (stats.compile_budget_exceeded != 1) return 3;
    if (stats.speculative_plan_count != 3) return 7;
    if (!stats.phantom_graph_enabled) return 13;
    if (stats.phantom_graph_match) return 14;
    if (stats.phantom_graph_mismatch_count != 1) return 19;
    if (stats.phantom_graph_reshape_count != 1) return 20;
    if (strcmp(stats.phantom_graph_expected_signature, stats.phantom_graph_observed_signature) != 0) return 21;

    set_env_flag("PYC_COMPILE_DELAY_MS", "0");
    rc = compile_and_run(&module, &options, 2, &stats);
    if (rc != 0) return 4;
    if (stats.compile_cache_hit != 1) return 5;
    if (stats.compile_budget_exceeded != 0) return 6;
    if (!stats.speculative_plan_hit) return 8;
    if (stats.speculative_plan_count != 3) return 9;
    if (stats.selected_kernel_candidates == 0) return 10;
    if (stats.selected_kernel_allocator_penalty < 0.0) return 11;
    if (stats.selected_kernel_reuse_bonus < 0.0) return 12;
    if (!stats.phantom_graph_enabled) return 15;
    if (!stats.phantom_graph_match) return 16;
    if (stats.phantom_graph_match_count == 0) return 17;
    if (strcmp(stats.phantom_graph_expected_signature, stats.phantom_graph_observed_signature) != 0) return 18;

    set_env_flag("PYC_COMPILE_DELAY_MS", "0");
    printf(
        "test_compile_cache: ok (cache_hit=%d budget_exceeded=%d)\n",
        stats.compile_cache_hit,
        stats.compile_budget_exceeded);
    return 0;
}
