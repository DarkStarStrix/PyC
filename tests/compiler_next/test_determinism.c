#include <stdio.h>
#include <string.h>

#include "pyc/compiler_api.h"
#include "pyc/ir.h"
#include "pyc/kernel_registry.h"

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
    pyc_compile_options options;
    pyc_compiled_model* model_a = NULL;
    pyc_compiled_model* model_b = NULL;
    pyc_run_stats stats_a;
    pyc_run_stats stats_b;
    pyc_tensor in;
    pyc_tensor out_a;
    pyc_tensor out_b;
    float in_data[8] = {1,2,3,4,5,6,7,8};
    float out_data_a[8] = {0};
    float out_data_b[8] = {0};
    pyc_status st;
    pyc_kernel_desc kd;

    build_module(&module);

    pyc_kernel_registry_reset();
    memset(&kd, 0, sizeof(kd));
    strcpy(kd.op_key, "matmul_fused");
    strcpy(kd.symbol, "kernel_cpu_v1");
    kd.backend = PYC_BACKEND_CPU;
    kd.priority = 3;
    pyc_kernel_register(&kd);

    desc.module = &module;
    desc.backend = PYC_BACKEND_CPU;

    memset(&options, 0, sizeof(options));
    options.enable_fusion = 1;
    options.enable_memory_reuse = 1;
    options.enable_autotune = 0;
    options.objective_mode = PYC_MODE_BALANCED;
    options.memory_budget_bytes = 0;
    options.target_utilization_floor = 0.70;
    options.deterministic_strict = 1;

    st = pyc_compile_model(&desc, &options, &model_a);
    if (st != PYC_STATUS_OK) return 1;
    st = pyc_compile_model(&desc, &options, &model_b);
    if (st != PYC_STATUS_OK) return 2;

    memset(&in, 0, sizeof(in));
    in.data = in_data;
    in.size_bytes = sizeof(in_data);
    in.dtype = PYC_DTYPE_F32;
    in.shape.rank = 1;
    in.shape.dims[0] = 8;

    memset(&out_a, 0, sizeof(out_a));
    out_a.data = out_data_a;
    out_a.size_bytes = sizeof(out_data_a);
    out_a.dtype = PYC_DTYPE_F32;
    out_a.shape.rank = 1;
    out_a.shape.dims[0] = 8;

    memset(&out_b, 0, sizeof(out_b));
    out_b.data = out_data_b;
    out_b.size_bytes = sizeof(out_data_b);
    out_b.dtype = PYC_DTYPE_F32;
    out_b.shape.rank = 1;
    out_b.shape.dims[0] = 8;

    st = pyc_run_model(model_a, &in, 1, &out_a, 1, &stats_a);
    if (st != PYC_STATUS_OK) return 3;
    st = pyc_run_model(model_b, &in, 1, &out_b, 1, &stats_b);
    if (st != PYC_STATUS_OK) return 4;

    if (memcmp(out_data_a, out_data_b, sizeof(out_data_a)) != 0) return 5;
    if (memcmp(out_data_a, in_data, sizeof(out_data_a)) != 0) return 6;

    if (stats_a.peak_bytes != stats_b.peak_bytes) return 7;
    if (stats_a.selected_kernel_count != stats_b.selected_kernel_count) return 8;
    if (strcmp(pyc_model_last_decision_log(model_a), pyc_model_last_decision_log(model_b)) != 0) return 9;
    if (stats_a.rollback_count != stats_b.rollback_count) return 10;
    if (stats_a.active_mode != stats_b.active_mode) return 11;

    pyc_destroy_model(model_a);
    pyc_destroy_model(model_b);

    printf("test_determinism: ok (peak=%zu kernel=%d)\n", stats_a.peak_bytes, stats_a.selected_kernel_count);
    return 0;
}
