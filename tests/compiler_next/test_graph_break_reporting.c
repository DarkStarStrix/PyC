#include <stdio.h>
#include <string.h>

#include "pyc/compiler_api.h"

static void build_reduce_module(pyc_ir_module* m) {
    pyc_ir_op in;
    pyc_ir_op reduce;
    pyc_ir_op out;

    pyc_ir_module_init(m);

    memset(&in, 0, sizeof(in));
    in.kind = PYC_IR_OP_INPUT;
    strcpy(in.name, "input0");
    in.dtype = PYC_DTYPE_F32;
    in.shape.rank = 1;
    in.shape.dims[0] = 4;
    pyc_ir_add_op(m, &in);

    memset(&reduce, 0, sizeof(reduce));
    reduce.kind = PYC_IR_OP_REDUCE_SUM;
    strcpy(reduce.name, "reduce0");
    reduce.dtype = PYC_DTYPE_F32;
    reduce.shape.rank = 1;
    reduce.shape.dims[0] = 4;
    reduce.input_ids[0] = 0;
    reduce.input_count = 1;
    pyc_ir_add_op(m, &reduce);

    memset(&out, 0, sizeof(out));
    out.kind = PYC_IR_OP_OUTPUT;
    strcpy(out.name, "output0");
    out.dtype = PYC_DTYPE_F32;
    out.shape.rank = 1;
    out.shape.dims[0] = 4;
    out.input_ids[0] = 1;
    out.input_count = 1;
    pyc_ir_add_op(m, &out);
}

int main(void) {
    pyc_ir_module module;
    pyc_model_desc desc;
    pyc_compile_options options;
    pyc_compiled_model* model = NULL;
    pyc_tensor in;
    pyc_tensor out;
    pyc_run_stats stats;
    pyc_status st;
    float in_data[4] = {1, 2, 3, 4};
    float out_data[4] = {0, 0, 0, 0};

    build_reduce_module(&module);

    memset(&desc, 0, sizeof(desc));
    desc.module = &module;
    desc.backend = PYC_BACKEND_CPU;

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

    st = pyc_run_model(model, &in, 1, &out, 1, &stats);
    if (st != PYC_STATUS_RUNTIME_FAILED) {
        pyc_destroy_model(model);
        return 2;
    }
    if (stats.graph_break_count == 0) {
        pyc_destroy_model(model);
        return 3;
    }
    if (stats.compilability_score >= 1.0) {
        pyc_destroy_model(model);
        return 4;
    }
    if (strstr(stats.graph_break_summary, "reduce_sum") == NULL) {
        pyc_destroy_model(model);
        return 5;
    }

    pyc_destroy_model(model);
    printf(
        "test_graph_break_reporting: ok (breaks=%zu score=%.3f)\n",
        stats.graph_break_count,
        stats.compilability_score);
    return 0;
}
