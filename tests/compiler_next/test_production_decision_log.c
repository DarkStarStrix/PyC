#include <stdio.h>
#include <string.h>

#include "pyc/compiler_api.h"

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

static int must_contain(const char* haystack, const char* needle) {
    return haystack && needle && strstr(haystack, needle) != NULL;
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
    const char* log;
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
    opts.enable_speculative_plans = 1;
    opts.max_speculative_plans = 3;
    opts.objective_mode = PYC_MODE_BALANCED;
    opts.deterministic_strict = 1;
    opts.cache_mode = PYC_COMPILE_CACHE_IN_MEMORY;
    opts.compile_budget_ms = 1000.0;
    pyc_runtime_rails_default(&opts.rails);

    st = pyc_compile_model(&desc, &opts, &model);
    if (st != PYC_STATUS_OK || !model) return 1;

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
    if (st != PYC_STATUS_OK) {
        pyc_destroy_model(model);
        return 2;
    }

    log = pyc_model_last_decision_log(model);
    if (!must_contain(log, "mode=")) {
        pyc_destroy_model(model);
        return 3;
    }
    if (!must_contain(log, "pressure=")) {
        pyc_destroy_model(model);
        return 4;
    }
    if (!must_contain(log, "kernel=")) {
        pyc_destroy_model(model);
        return 5;
    }
    if (!must_contain(log, "alloc_penalty=") || !must_contain(log, "reuse_bonus=") || !must_contain(log, "kernel_candidates=")) {
        pyc_destroy_model(model);
        return 18;
    }
    if (!must_contain(log, "cuda_reason=")) {
        pyc_destroy_model(model);
        return 6;
    }
    if (!must_contain(log, "contract=") || !must_contain(log, "contract_reason=")) {
        pyc_destroy_model(model);
        return 7;
    }
    if (!must_contain(log, "guard_miss=") || !must_contain(log, "fallback=")) {
        pyc_destroy_model(model);
        return 8;
    }
    if (!must_contain(log, "spec_plans=") || !must_contain(log, "spec_hit=") || !must_contain(log, "spec_bucket=")) {
        pyc_destroy_model(model);
        return 16;
    }
    if (!must_contain(log, "spec_conf=")) {
        pyc_destroy_model(model);
        return 20;
    }
    if (!must_contain(log, "spec_plans=3")) {
        pyc_destroy_model(model);
        return 21;
    }
    if (!must_contain(log, "spec_hit=1")) {
        pyc_destroy_model(model);
        return 22;
    }
    if (!must_contain(log, "graph_breaks=") || !must_contain(log, "compilability=")) {
        pyc_destroy_model(model);
        return 9;
    }
    if (!must_contain(log, "fp=")) {
        pyc_destroy_model(model);
        return 10;
    }

    if (!stats.deterministic_contract_enforced || !stats.deterministic_contract_ok) {
        pyc_destroy_model(model);
        return 11;
    }
    if (strcmp(stats.deterministic_contract_reason, "ok") != 0) {
        pyc_destroy_model(model);
        return 12;
    }
    if (stats.model_fingerprint == 0) {
        pyc_destroy_model(model);
        return 13;
    }
    if (stats.graph_break_count != 0) {
        pyc_destroy_model(model);
        return 14;
    }
    if (stats.compilability_score < 0.99) {
        pyc_destroy_model(model);
        return 15;
    }
    if (stats.speculative_plan_count != 3) {
        pyc_destroy_model(model);
        return 17;
    }
    if (stats.selected_kernel_candidates == 0) {
        pyc_destroy_model(model);
        return 19;
    }
    if (stats.selected_kernel_symbol[0] == '\0') {
        pyc_destroy_model(model);
        return 23;
    }
    if (stats.speculative_plan_hit != 1) {
        pyc_destroy_model(model);
        return 24;
    }
    if (stats.speculative_shape_bucket[0] == '\0') {
        pyc_destroy_model(model);
        return 25;
    }

    pyc_destroy_model(model);
    printf("test_production_decision_log: ok\n");
    return 0;
}
