#include <stdio.h>
#include <string.h>

#include "pyc/compiler_api.h"

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

static void init_square_io(
    pyc_tensor* a,
    pyc_tensor* b,
    pyc_tensor* out,
    float* a_data,
    float* b_data,
    float* out_data,
    size_t dim) {
    size_t elems = dim * dim;

    memset(a, 0, sizeof(*a));
    a->data = a_data;
    a->size_bytes = elems * sizeof(float);
    a->dtype = PYC_DTYPE_F32;
    a->shape.rank = 2;
    a->shape.dims[0] = (int64_t)dim;
    a->shape.dims[1] = (int64_t)dim;

    memset(b, 0, sizeof(*b));
    b->data = b_data;
    b->size_bytes = elems * sizeof(float);
    b->dtype = PYC_DTYPE_F32;
    b->shape.rank = 2;
    b->shape.dims[0] = (int64_t)dim;
    b->shape.dims[1] = (int64_t)dim;

    memset(out, 0, sizeof(*out));
    out->data = out_data;
    out->size_bytes = elems * sizeof(float);
    out->dtype = PYC_DTYPE_F32;
    out->shape.rank = 2;
    out->shape.dims[0] = (int64_t)dim;
    out->shape.dims[1] = (int64_t)dim;
}

int main(void) {
    pyc_ir_module module;
    pyc_model_desc desc;
    pyc_compile_options opts;
    pyc_compiled_model* model = NULL;
    pyc_tensor inputs[2];
    pyc_tensor output;
    pyc_run_stats stats;
    pyc_status st;
    const char* log;
    float a2_data[4] = {1, 2, 3, 4};
    float b2_data[4] = {1, 0, 0, 1};
    float out2_data[4] = {0, 0, 0, 0};
    float a4_data[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    float b4_data[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    float out4_data[16] = {0};
    float a3_data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float b3_data[9] = {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };
    float out3_data[9] = {0};

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
    pyc_runtime_rails_default(&opts.rails);

    st = pyc_compile_model(&desc, &opts, &model);
    if (st != PYC_STATUS_OK || !model) return 1;

    init_square_io(&inputs[0], &inputs[1], &output, a2_data, b2_data, out2_data, 2);
    st = pyc_run_model(model, inputs, 2, &output, 1, &stats);
    if (st != PYC_STATUS_OK) {
        pyc_destroy_model(model);
        return 2;
    }
    if (stats.speculative_plan_count != 3) {
        pyc_destroy_model(model);
        return 3;
    }
    if (!stats.speculative_plan_hit) {
        pyc_destroy_model(model);
        return 4;
    }
    if (stats.speculative_plan_miss_count != 0 || stats.speculative_guard_miss_count != 0) {
        pyc_destroy_model(model);
        return 5;
    }
    if (stats.speculative_shape_bucket[0] == '\0') {
        pyc_destroy_model(model);
        return 6;
    }
    if (((float*)output.data)[0] != 1.0f || ((float*)output.data)[3] != 4.0f) {
        pyc_destroy_model(model);
        return 7;
    }

    log = pyc_model_last_decision_log(model);
    if (!log || strstr(log, "spec_plans=") == NULL || strstr(log, "spec_hit=1") == NULL) {
        pyc_destroy_model(model);
        return 8;
    }

    init_square_io(&inputs[0], &inputs[1], &output, a4_data, b4_data, out4_data, 4);
    st = pyc_run_model(model, inputs, 2, &output, 1, &stats);
    if (st != PYC_STATUS_OK) {
        pyc_destroy_model(model);
        return 9;
    }
    if (!stats.speculative_plan_hit || stats.speculative_plan_count != 3) {
        pyc_destroy_model(model);
        return 10;
    }
    if (((float*)output.data)[0] != 1.0f || ((float*)output.data)[15] != 16.0f) {
        pyc_destroy_model(model);
        return 11;
    }
    if (stats.selected_kernel_symbol[0] == '\0') {
        pyc_destroy_model(model);
        return 15;
    }
    if (stats.selected_kernel_candidates == 0) {
        pyc_destroy_model(model);
        return 16;
    }
    if (stats.selected_kernel_allocator_penalty < 0.0 || stats.selected_kernel_reuse_bonus < 0.0) {
        pyc_destroy_model(model);
        return 17;
    }

    init_square_io(&inputs[0], &inputs[1], &output, a3_data, b3_data, out3_data, 3);
    st = pyc_run_model(model, inputs, 2, &output, 1, &stats);
    if (st != PYC_STATUS_RUNTIME_FAILED) {
        pyc_destroy_model(model);
        return 12;
    }
    if (stats.speculative_guard_miss_count == 0) {
        pyc_destroy_model(model);
        return 13;
    }
    if (strcmp(stats.deterministic_contract_reason, "input_shape_mismatch") != 0) {
        pyc_destroy_model(model);
        return 14;
    }

    pyc_destroy_model(model);
    printf("test_speculative_plans: ok\n");
    return 0;
}
