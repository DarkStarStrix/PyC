#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

#if defined(_WIN32)
#include <process.h>
#define PYC_GETPID _getpid
#else
#include <unistd.h>
#define PYC_GETPID getpid
#endif

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

static int compile_run_once(
    const pyc_ir_module* module,
    const char* db_path,
    pyc_run_stats* stats_out) {
    pyc_model_desc desc;
    pyc_compile_options opts;
    pyc_compiled_model* model = NULL;
    pyc_status st;
    pyc_tensor in;
    pyc_tensor out;
    float in_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float out_data[8] = {0};

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

    st = pyc_run_model(model, &in, 1, &out, 1, stats_out);
    pyc_destroy_model(model);
    if (st != PYC_STATUS_OK) {
        return 2;
    }
    if (memcmp(in_data, out_data, sizeof(in_data)) != 0) {
        return 3;
    }
    return 0;
}

int main(void) {
    pyc_ir_module module;
    pyc_run_stats first_stats;
    pyc_run_stats second_stats;
    char db_path[512];
    const char* tmp_dir = NULL;
    int first_rc;
    int second_rc;

#if defined(_WIN32)
    tmp_dir = getenv("TEMP");
    if (!tmp_dir || tmp_dir[0] == '\0') {
        tmp_dir = getenv("TMP");
    }
    if (!tmp_dir || tmp_dir[0] == '\0') {
        tmp_dir = ".";
    }
    snprintf(
        db_path,
        sizeof(db_path),
        "%s\\pyc_autotune_test_%d_%ld.db",
        tmp_dir,
        (int)PYC_GETPID(),
        (long)time(NULL));
#else
    tmp_dir = getenv("TMPDIR");
    if (!tmp_dir || tmp_dir[0] == '\0') {
        tmp_dir = "/tmp";
    }
    snprintf(
        db_path,
        sizeof(db_path),
        "%s/pyc_autotune_test_%d_%ld.db",
        tmp_dir,
        (int)PYC_GETPID(),
        (long)time(NULL));
#endif

    remove(db_path);
    build_module(&module);

    first_rc = compile_run_once(&module, db_path, &first_stats);
    if (first_rc != 0) {
        fprintf(stderr, "autotune first run failed rc=%d db=%s\n", first_rc, db_path);
        remove(db_path);
        return 1;
    }
    if (!first_stats.autotune_saved) {
        fprintf(stderr, "autotune first run did not save db=%s\n", db_path);
        remove(db_path);
        return 2;
    }
    if (first_stats.autotune_loaded) {
        fprintf(stderr, "autotune first run unexpectedly loaded db=%s\n", db_path);
        remove(db_path);
        return 3;
    }

    second_rc = compile_run_once(&module, db_path, &second_stats);
    if (second_rc != 0) {
        fprintf(stderr, "autotune second run failed rc=%d db=%s\n", second_rc, db_path);
        remove(db_path);
        return 4;
    }
    if (!second_stats.autotune_saved) {
        fprintf(stderr, "autotune second run did not save db=%s\n", db_path);
        remove(db_path);
        return 5;
    }
    if (!second_stats.autotune_loaded) {
        fprintf(stderr, "autotune second run did not load db=%s\n", db_path);
        remove(db_path);
        return 6;
    }

    remove(db_path);
    printf(
        "test_autotune_persistence: ok (loaded=%d saved=%d)\n",
        second_stats.autotune_loaded,
        second_stats.autotune_saved);
    return 0;
}
