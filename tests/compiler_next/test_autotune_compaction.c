#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <process.h>
#define PYC_GETPID _getpid
#else
#include <unistd.h>
#define PYC_GETPID getpid
#endif

#include "pyc/compiler_api.h"

static void build_matmul_module(pyc_ir_module* m) {
    pyc_ir_op lhs;
    pyc_ir_op rhs;
    pyc_ir_op matmul;
    pyc_ir_op out;

    pyc_ir_module_init(m);

    memset(&lhs, 0, sizeof(lhs));
    lhs.kind = PYC_IR_OP_INPUT;
    strcpy(lhs.name, "lhs");
    lhs.dtype = PYC_DTYPE_F32;
    lhs.shape.rank = 2;
    lhs.shape.dims[0] = 2;
    lhs.shape.dims[1] = 2;
    pyc_ir_add_op(m, &lhs);

    memset(&rhs, 0, sizeof(rhs));
    rhs.kind = PYC_IR_OP_INPUT;
    strcpy(rhs.name, "rhs");
    rhs.dtype = PYC_DTYPE_F32;
    rhs.shape.rank = 2;
    rhs.shape.dims[0] = 2;
    rhs.shape.dims[1] = 2;
    pyc_ir_add_op(m, &rhs);

    memset(&matmul, 0, sizeof(matmul));
    matmul.kind = PYC_IR_OP_MATMUL;
    strcpy(matmul.name, "matmul0");
    matmul.dtype = PYC_DTYPE_F32;
    matmul.shape.rank = 2;
    matmul.shape.dims[0] = 2;
    matmul.shape.dims[1] = 2;
    matmul.input_ids[0] = 0;
    matmul.input_ids[1] = 1;
    matmul.input_count = 2;
    pyc_ir_add_op(m, &matmul);

    memset(&out, 0, sizeof(out));
    out.kind = PYC_IR_OP_OUTPUT;
    strcpy(out.name, "out");
    out.dtype = PYC_DTYPE_F32;
    out.shape.rank = 2;
    out.shape.dims[0] = 2;
    out.shape.dims[1] = 2;
    out.input_ids[0] = 2;
    out.input_count = 1;
    pyc_ir_add_op(m, &out);
}

static int run_once(const pyc_ir_module* module, const char* db_path) {
    pyc_model_desc desc;
    pyc_compile_options opts;
    pyc_compiled_model* model = NULL;
    pyc_status st;
    pyc_tensor inputs[2];
    pyc_tensor outputs[1];
    pyc_run_stats stats;
    float lhs_data[4] = {1, 2, 3, 4};
    float rhs_data[4] = {1, 0, 0, 1};
    float out_data[4] = {0, 0, 0, 0};

    memset(&desc, 0, sizeof(desc));
    desc.module = module;
    desc.backend = PYC_BACKEND_CPU;

    memset(&opts, 0, sizeof(opts));
    opts.enable_fusion = 1;
    opts.enable_memory_reuse = 1;
    opts.enable_autotune = 1;
    opts.objective_mode = PYC_MODE_BALANCED;
    opts.target_utilization_floor = 0.70;
    opts.deterministic_strict = 1;
    opts.cache_mode = PYC_COMPILE_CACHE_DISABLED;
    opts.autotune_db_path = db_path;
    pyc_runtime_rails_default(&opts.rails);

    st = pyc_compile_model(&desc, &opts, &model);
    if (st != PYC_STATUS_OK || !model) {
        return 1;
    }

    memset(inputs, 0, sizeof(inputs));
    inputs[0].data = lhs_data;
    inputs[0].size_bytes = sizeof(lhs_data);
    inputs[0].dtype = PYC_DTYPE_F32;
    inputs[0].shape.rank = 2;
    inputs[0].shape.dims[0] = 2;
    inputs[0].shape.dims[1] = 2;

    inputs[1].data = rhs_data;
    inputs[1].size_bytes = sizeof(rhs_data);
    inputs[1].dtype = PYC_DTYPE_F32;
    inputs[1].shape.rank = 2;
    inputs[1].shape.dims[0] = 2;
    inputs[1].shape.dims[1] = 2;

    memset(outputs, 0, sizeof(outputs));
    outputs[0].data = out_data;
    outputs[0].size_bytes = sizeof(out_data);
    outputs[0].dtype = PYC_DTYPE_F32;
    outputs[0].shape.rank = 2;
    outputs[0].shape.dims[0] = 2;
    outputs[0].shape.dims[1] = 2;

    st = pyc_run_model(model, inputs, 2, outputs, 1, &stats);
    pyc_destroy_model(model);

    if (st != PYC_STATUS_OK) {
        return 2;
    }
    if (!stats.autotune_saved) {
        return 3;
    }
    return 0;
}

int main(void) {
    pyc_ir_module module;
    char db_path[512];
    FILE* f;
    char line[512];
    char seen_symbols[64][PYC_KERNEL_SYMBOL_MAX];
    int seen_backend[64];
    char seen_ops[64][PYC_KERNEL_OP_KEY_MAX];
    size_t seen_count = 0;
    size_t line_count = 0;
    int i;

    snprintf(
        db_path,
        sizeof(db_path),
        "pyc_autotune_compaction_%d_%ld.db",
        (int)PYC_GETPID(),
        (long)time(NULL));

    remove(db_path);
    build_matmul_module(&module);

    for (i = 0; i < 8; ++i) {
        int rc = run_once(&module, db_path);
        if (rc != 0) {
            fprintf(stderr, "run_once failed rc=%d iteration=%d\n", rc, i);
            remove(db_path);
            return 1;
        }
    }

    f = fopen(db_path, "r");
    if (!f) {
        fprintf(stderr, "missing autotune db %s\n", db_path);
        return 2;
    }

    while (fgets(line, sizeof(line), f)) {
        char op[PYC_KERNEL_OP_KEY_MAX];
        char symbol[PYC_KERNEL_SYMBOL_MAX];
        int backend = -1;
        double best_ms = 0.0;
        size_t j;
        int duplicate = 0;

        if (sscanf(line, "%63[^|]|%d|%127[^|]|%lf", op, &backend, symbol, &best_ms) != 4) {
            continue;
        }
        if (best_ms <= 0.0) {
            fclose(f);
            remove(db_path);
            return 3;
        }
        line_count++;

        for (j = 0; j < seen_count; ++j) {
            if (seen_backend[j] == backend &&
                strcmp(seen_ops[j], op) == 0 &&
                strcmp(seen_symbols[j], symbol) == 0) {
                duplicate = 1;
                break;
            }
        }
        if (duplicate) {
            fclose(f);
            remove(db_path);
            return 4;
        }

        if (seen_count < 64) {
            strncpy(seen_ops[seen_count], op, sizeof(seen_ops[seen_count]) - 1);
            seen_ops[seen_count][sizeof(seen_ops[seen_count]) - 1] = '\0';
            strncpy(seen_symbols[seen_count], symbol, sizeof(seen_symbols[seen_count]) - 1);
            seen_symbols[seen_count][sizeof(seen_symbols[seen_count]) - 1] = '\0';
            seen_backend[seen_count] = backend;
            seen_count++;
        }
    }
    fclose(f);

    if (line_count == 0) {
        remove(db_path);
        return 5;
    }
    if (line_count > 16) {
        remove(db_path);
        return 6;
    }

    remove(db_path);
    printf("test_autotune_compaction: ok (entries=%zu)\n", line_count);
    return 0;
}
