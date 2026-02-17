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
    in.shape.rank = 2;
    in.shape.dims[0] = 16;
    in.shape.dims[1] = 16;
    pyc_ir_add_op(m, &in);

    memset(&out, 0, sizeof(out));
    out.kind = PYC_IR_OP_OUTPUT;
    strcpy(out.name, "output0");
    out.dtype = PYC_DTYPE_F32;
    out.shape.rank = 2;
    out.shape.dims[0] = 16;
    out.shape.dims[1] = 16;
    out.input_ids[0] = 0;
    out.input_count = 1;
    pyc_ir_add_op(m, &out);
}

int main(void) {
    pyc_ir_module module;
    pyc_model_desc desc;
    pyc_compile_options memory_opt;
    pyc_compile_options util_opt;
    pyc_compiled_model* mem_model = NULL;
    pyc_compiled_model* util_model = NULL;
    pyc_status st;
    pyc_kernel_desc kd;
    pyc_tensor in;
    pyc_tensor out_mem;
    pyc_tensor out_util;
    pyc_run_stats mem_stats;
    pyc_run_stats util_stats;
    float in_data[256];
    float out_data_mem[256] = {0};
    float out_data_util[256] = {0};
    size_t i;

    for (i = 0; i < 256; ++i) {
        in_data[i] = (float)i;
    }

    build_module(&module);

    pyc_kernel_registry_reset();
    memset(&kd, 0, sizeof(kd));
    strcpy(kd.op_key, "matmul_fused");
    strcpy(kd.symbol, "kernel_mem");
    kd.backend = PYC_BACKEND_CPU;
    kd.priority = 5;
    kd.estimated_occupancy = 0.55;
    kd.shared_mem_bytes = 2 * 1024;
    kd.reg_pressure_class = 1;
    pyc_kernel_register(&kd);

    memset(&kd, 0, sizeof(kd));
    strcpy(kd.op_key, "matmul_fused");
    strcpy(kd.symbol, "kernel_util");
    kd.backend = PYC_BACKEND_CPU;
    kd.priority = 5;
    kd.estimated_occupancy = 0.95;
    kd.shared_mem_bytes = 64 * 1024;
    kd.reg_pressure_class = 3;
    pyc_kernel_register(&kd);

    desc.module = &module;
    desc.backend = PYC_BACKEND_CPU;

    memset(&memory_opt, 0, sizeof(memory_opt));
    memory_opt.enable_fusion = 1;
    memory_opt.enable_memory_reuse = 1;
    memory_opt.objective_mode = PYC_MODE_MEMORY_FIRST;
    memory_opt.memory_budget_bytes = 512;
    memory_opt.target_utilization_floor = 0.70;
    memory_opt.deterministic_strict = 1;

    memset(&util_opt, 0, sizeof(util_opt));
    util_opt.enable_fusion = 1;
    util_opt.enable_memory_reuse = 1;
    util_opt.objective_mode = PYC_MODE_UTILIZATION_FIRST;
    util_opt.memory_budget_bytes = 512;
    util_opt.target_utilization_floor = 0.70;
    util_opt.deterministic_strict = 1;

    st = pyc_compile_model(&desc, &memory_opt, &mem_model);
    if (st != PYC_STATUS_OK) return 1;
    st = pyc_compile_model(&desc, &util_opt, &util_model);
    if (st != PYC_STATUS_OK) return 2;

    memset(&in, 0, sizeof(in));
    in.data = in_data;
    in.size_bytes = sizeof(in_data);
    in.dtype = PYC_DTYPE_F32;
    in.shape.rank = 2;
    in.shape.dims[0] = 16;
    in.shape.dims[1] = 16;

    memset(&out_mem, 0, sizeof(out_mem));
    out_mem.data = out_data_mem;
    out_mem.size_bytes = sizeof(out_data_mem);
    out_mem.dtype = PYC_DTYPE_F32;
    out_mem.shape = in.shape;

    memset(&out_util, 0, sizeof(out_util));
    out_util.data = out_data_util;
    out_util.size_bytes = sizeof(out_data_util);
    out_util.dtype = PYC_DTYPE_F32;
    out_util.shape = in.shape;

    st = pyc_run_model(mem_model, &in, 1, &out_mem, 1, &mem_stats);
    if (st != PYC_STATUS_OK) return 3;
    st = pyc_run_model(util_model, &in, 1, &out_util, 1, &util_stats);
    if (st != PYC_STATUS_OK) return 4;

    if (mem_stats.pressure_events == 0) return 5;
    if (mem_stats.rematerialized_tensors == 0) return 6;
    if (util_stats.rematerialized_tensors != 0) return 7;
    if (strcmp(pyc_model_last_decision_log(mem_model), "") == 0) return 8;
    if (strcmp(mem_stats.selected_kernel_symbol, util_stats.selected_kernel_symbol) == 0) return 9;

    pyc_destroy_model(mem_model);
    pyc_destroy_model(util_model);

    printf("test_policy_modes: ok (mem_peak=%zu util_kernel=%s)\n", mem_stats.peak_bytes, util_stats.selected_kernel_symbol);
    return 0;
}
