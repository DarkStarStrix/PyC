#include <stdio.h>
#include <string.h>

#include "pyc/compiler_api.h"
#include "pyc/ir.h"
#include "pyc/kernel_registry.h"

static void build_valid_module(pyc_ir_module* m) {
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

int main(void) {
    pyc_status st;
    pyc_compiled_model* model = NULL;
    pyc_model_desc desc;
    pyc_compile_options options;
    pyc_ir_module module;
    pyc_ir_module bad;
    pyc_tensor input;
    pyc_tensor output;
    pyc_run_stats stats;
    float in_data[4] = {1, 2, 3, 4};
    float out_data[4] = {0, 0, 0, 0};
    pyc_kernel_desc kd;

    st = pyc_compile_model(NULL, NULL, NULL);
    if (st != PYC_STATUS_INVALID_ARGUMENT) return 1;

    pyc_ir_module_init(&bad);
    desc.module = &bad;
    desc.backend = PYC_BACKEND_CPU;
    st = pyc_compile_model(&desc, NULL, &model);
    if (st != PYC_STATUS_VERIFY_FAILED) return 2;

    build_valid_module(&module);
    pyc_kernel_registry_reset();
    memset(&kd, 0, sizeof(kd));
    strcpy(kd.op_key, "matmul_fused");
    strcpy(kd.symbol, "kernel_cpu_v1");
    kd.backend = PYC_BACKEND_CPU;
    kd.priority = 1;
    pyc_kernel_register(&kd);

    desc.module = &module;
    desc.backend = PYC_BACKEND_CPU;
    options.enable_fusion = 1;
    options.enable_memory_reuse = 1;
    options.enable_autotune = 0;
    options.objective_mode = PYC_MODE_BALANCED;
    options.memory_budget_bytes = 0;
    options.target_utilization_floor = 0.70;
    options.deterministic_strict = 1;

    st = pyc_compile_model(&desc, &options, &model);
    if (st != PYC_STATUS_OK || !model) return 3;

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

    st = pyc_run_model(model, &input, 1, &output, 1, &stats);
    if (st != PYC_STATUS_OK) {
        pyc_destroy_model(model);
        return 4;
    }

    if (memcmp(in_data, out_data, sizeof(in_data)) != 0) {
        pyc_destroy_model(model);
        return 5;
    }

    if (stats.peak_bytes == 0) {
        pyc_destroy_model(model);
        return 6;
    }
    if (stats.selected_kernel_count != 1) {
        pyc_destroy_model(model);
        return 7;
    }
    if (stats.selected_kernel_symbol[0] == '\0') {
        pyc_destroy_model(model);
        return 8;
    }

    pyc_destroy_model(model);
    printf("test_compiler_api: ok (compile=%.3f run=%.3f peak=%zu)\n", stats.compile_ms, stats.run_ms, stats.peak_bytes);
    return 0;
}
