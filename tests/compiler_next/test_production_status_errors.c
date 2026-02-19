#include <stdio.h>
#include <string.h>

#include "pyc/compiler_api.h"

static void build_passthrough_module(pyc_ir_module* module) {
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
    pyc_ir_module valid_module;
    pyc_ir_module invalid_module;
    pyc_model_desc desc;
    pyc_compile_options opts;
    pyc_compiled_model* model = NULL;
    pyc_tensor in;
    pyc_tensor out;
    pyc_status st;
    float in_data[4] = {1, 2, 3, 4};
    float out_data[4] = {0, 0, 0, 0};

    if (strcmp(pyc_status_string(PYC_STATUS_OK), "OK") != 0) return 1;
    if (strcmp(pyc_status_string(PYC_STATUS_INVALID_ARGUMENT), "INVALID_ARGUMENT") != 0) return 2;
    if (strcmp(pyc_status_string(PYC_STATUS_VERIFY_FAILED), "VERIFY_FAILED") != 0) return 3;
    if (strcmp(pyc_status_string(PYC_STATUS_COMPILE_FAILED), "COMPILE_FAILED") != 0) return 4;
    if (strcmp(pyc_status_string(PYC_STATUS_RUNTIME_FAILED), "RUNTIME_FAILED") != 0) return 5;

    st = pyc_compile_model(NULL, NULL, &model);
    if (st != PYC_STATUS_INVALID_ARGUMENT) return 6;

    pyc_ir_module_init(&invalid_module);
    memset(&desc, 0, sizeof(desc));
    desc.module = &invalid_module;
    desc.backend = PYC_BACKEND_CPU;
    st = pyc_compile_model(&desc, NULL, &model);
    if (st != PYC_STATUS_VERIFY_FAILED) return 7;

    build_passthrough_module(&valid_module);
    memset(&desc, 0, sizeof(desc));
    desc.module = &valid_module;
    desc.backend = PYC_BACKEND_CPU;
    memset(&opts, 0, sizeof(opts));
    opts.enable_fusion = 1;
    opts.enable_memory_reuse = 1;
    opts.enable_autotune = 0;
    opts.objective_mode = PYC_MODE_BALANCED;
    opts.deterministic_strict = 1;
    pyc_runtime_rails_default(&opts.rails);

    st = pyc_compile_model(&desc, &opts, &model);
    if (st != PYC_STATUS_OK || !model) return 8;

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

    st = pyc_run_model(NULL, &in, 1, &out, 1, NULL);
    if (st != PYC_STATUS_INVALID_ARGUMENT) {
        pyc_destroy_model(model);
        return 9;
    }

    st = pyc_run_model(model, NULL, 1, &out, 1, NULL);
    if (st != PYC_STATUS_INVALID_ARGUMENT) {
        pyc_destroy_model(model);
        return 10;
    }

    st = pyc_run_model(model, &in, 0, &out, 1, NULL);
    if (st != PYC_STATUS_INVALID_ARGUMENT) {
        pyc_destroy_model(model);
        return 11;
    }

    st = pyc_run_model(model, &in, 1, NULL, 1, NULL);
    if (st != PYC_STATUS_INVALID_ARGUMENT) {
        pyc_destroy_model(model);
        return 12;
    }

    st = pyc_run_model(model, &in, 1, &out, 0, NULL);
    if (st != PYC_STATUS_INVALID_ARGUMENT) {
        pyc_destroy_model(model);
        return 13;
    }

    pyc_destroy_model(model);
    printf("test_production_status_errors: ok\n");
    return 0;
}
