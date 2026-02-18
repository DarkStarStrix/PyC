#include <stdio.h>
#include <string.h>

#include "pyc/compiler_api.h"

static void build_module(pyc_ir_module* m) {
    pyc_ir_op in;
    pyc_ir_op out;
    pyc_ir_module_init(m);

    memset(&in, 0, sizeof(in));
    in.kind = PYC_IR_OP_INPUT;
    strcpy(in.name, "input0");
    in.dtype = PYC_DTYPE_F32;
    in.shape.rank = 1;
    in.shape.dims[0] = 8;
    pyc_ir_add_op(m, &in);

    memset(&out, 0, sizeof(out));
    out.kind = PYC_IR_OP_OUTPUT;
    strcpy(out.name, "output0");
    out.dtype = PYC_DTYPE_F32;
    out.shape.rank = 1;
    out.shape.dims[0] = 8;
    out.input_ids[0] = 0;
    out.input_count = 1;
    pyc_ir_add_op(m, &out);
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
    float in_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float out_data[8] = {0};

    build_module(&module);
    memset(&desc, 0, sizeof(desc));
    desc.module = &module;
    desc.backend = PYC_BACKEND_CPU;

    memset(&opts, 0, sizeof(opts));
    opts.enable_fusion = 1;
    opts.enable_memory_reuse = 1;
    opts.enable_autotune = 0;
    opts.objective_mode = PYC_MODE_BALANCED;
    opts.target_utilization_floor = 0.7;
    opts.deterministic_strict = 1;
    pyc_runtime_rails_default(&opts.rails);

    st = pyc_compile_model(&desc, &opts, &model);
    if (st != PYC_STATUS_OK) return 1;

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

    st = pyc_run_model(model, &in, 1, &out, 1, &stats);
    if (st != PYC_STATUS_OK) return 2;
    if (!stats.deterministic_contract_enforced) return 3;
    if (!stats.deterministic_contract_ok) return 4;
    if (strcmp(stats.deterministic_contract_reason, "ok") != 0) return 5;
    if (stats.model_fingerprint == 0) return 6;
    if (stats.dispatch_ms < 0.0 || stats.controller_ms < 0.0 || stats.kernel_select_ms < 0.0) return 7;
    if (stats.compile_cache_hit != 0) return 8;
    if (stats.compile_budget_exceeded != 0) return 9;
    if (stats.guard_miss_count != 0) return 10;
    if (stats.fallback_count != 0) return 11;
    if (stats.graph_break_count != 0) return 12;
    if (stats.compilability_score < 0.99) return 13;
    if (stats.autotune_loaded != 0 || stats.autotune_saved != 0) return 14;

    pyc_destroy_model(model);
    printf("test_deterministic_contracts: ok (fingerprint=%llu)\n", (unsigned long long)stats.model_fingerprint);
    return 0;
}
