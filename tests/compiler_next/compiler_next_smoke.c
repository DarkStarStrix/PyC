#include <stdio.h>
#include <string.h>

#include "pyc/compiler_api.h"
#include "pyc/ir.h"
#include "pyc/kernel_registry.h"

int main(void) {
    pyc_ir_module module;
    pyc_ir_op in;
    pyc_ir_op out;
    pyc_model_desc desc;
    pyc_compile_options options;
    pyc_compiled_model* model = NULL;
    pyc_tensor input;
    pyc_tensor output;
    pyc_run_stats stats;
    float in_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out_data[4] = {0};
    pyc_status st;
    pyc_kernel_desc kd;

    pyc_ir_module_init(&module);

    memset(&in, 0, sizeof(in));
    in.kind = PYC_IR_OP_INPUT;
    strcpy(in.name, "input0");
    in.dtype = PYC_DTYPE_F32;
    in.shape.rank = 1;
    in.shape.dims[0] = 4;
    if (pyc_ir_add_op(&module, &in) != 0) {
        return 2;
    }

    memset(&out, 0, sizeof(out));
    out.kind = PYC_IR_OP_OUTPUT;
    strcpy(out.name, "output0");
    out.dtype = PYC_DTYPE_F32;
    out.shape.rank = 1;
    out.shape.dims[0] = 4;
    out.input_ids[0] = 0;
    out.input_count = 1;
    if (pyc_ir_add_op(&module, &out) != 0) {
        return 3;
    }

    pyc_kernel_registry_reset();
    memset(&kd, 0, sizeof(kd));
    strcpy(kd.op_key, "matmul_fused");
    kd.backend = PYC_BACKEND_CPU;
    strcpy(kd.symbol, "kernel_matmul_fused_cpu_v1");
    kd.priority = 10;
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

    st = pyc_compile_model(&desc, &options, &model);
    if (st != PYC_STATUS_OK) {
        fprintf(stderr, "compile failed: %s\n", pyc_status_string(st));
        return 4;
    }

    memset(&input, 0, sizeof(input));
    input.data = in_data;
    input.size_bytes = sizeof(in_data);
    input.dtype = PYC_DTYPE_F32;
    input.shape = in.shape;

    memset(&output, 0, sizeof(output));
    output.data = out_data;
    output.size_bytes = sizeof(out_data);
    output.dtype = PYC_DTYPE_F32;
    output.shape = out.shape;

    st = pyc_run_model(model, &input, 1, &output, 1, &stats);
    if (st != PYC_STATUS_OK) {
        fprintf(stderr, "run failed: %s\n", pyc_status_string(st));
        pyc_destroy_model(model);
        return 5;
    }

    if (memcmp(in_data, out_data, sizeof(in_data)) != 0) {
        fprintf(stderr, "output mismatch\n");
        pyc_destroy_model(model);
        return 6;
    }

    printf("compiler-next smoke ok (compile=%.3f ms run=%.3f ms peak=%zu kernel=%d)\n",
           stats.compile_ms,
           stats.run_ms,
           stats.peak_bytes,
           stats.selected_kernel_count);

    pyc_destroy_model(model);
    return 0;
}
