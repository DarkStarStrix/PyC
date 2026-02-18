#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "pyc/compiler_api.h"

static double elapsed_ms(clock_t a, clock_t b) {
    return ((double)(b - a) * 1000.0) / (double)CLOCKS_PER_SEC;
}

static int cmp_double(const void* a, const void* b) {
    const double da = *(const double*)a;
    const double db = *(const double*)b;
    return (da > db) - (da < db);
}

static double percentile(double* values, size_t n, double p) {
    size_t idx;
    qsort(values, n, sizeof(double), cmp_double);
    idx = (size_t)((p / 100.0) * (double)(n - 1));
    if (idx >= n) idx = n - 1;
    return values[idx];
}

static void build_matmul_module(pyc_ir_module* m, int batch, int hidden) {
    pyc_ir_op lhs;
    pyc_ir_op rhs;
    pyc_ir_op mm;
    pyc_ir_op out;
    pyc_ir_module_init(m);

    memset(&lhs, 0, sizeof(lhs));
    lhs.kind = PYC_IR_OP_INPUT;
    strcpy(lhs.name, "lhs");
    lhs.dtype = PYC_DTYPE_F32;
    lhs.shape.rank = 2;
    lhs.shape.dims[0] = batch;
    lhs.shape.dims[1] = hidden;
    pyc_ir_add_op(m, &lhs);

    memset(&rhs, 0, sizeof(rhs));
    rhs.kind = PYC_IR_OP_INPUT;
    strcpy(rhs.name, "rhs");
    rhs.dtype = PYC_DTYPE_F32;
    rhs.shape.rank = 2;
    rhs.shape.dims[0] = hidden;
    rhs.shape.dims[1] = hidden;
    pyc_ir_add_op(m, &rhs);

    memset(&mm, 0, sizeof(mm));
    mm.kind = PYC_IR_OP_MATMUL;
    strcpy(mm.name, "matmul0");
    mm.dtype = PYC_DTYPE_F32;
    mm.input_ids[0] = 0;
    mm.input_ids[1] = 1;
    mm.input_count = 2;
    pyc_ir_add_op(m, &mm);

    memset(&out, 0, sizeof(out));
    out.kind = PYC_IR_OP_OUTPUT;
    strcpy(out.name, "out0");
    out.dtype = PYC_DTYPE_F32;
    out.shape.rank = 2;
    out.shape.dims[0] = batch;
    out.shape.dims[1] = hidden;
    out.input_ids[0] = 2;
    out.input_count = 1;
    pyc_ir_add_op(m, &out);
}

int main(int argc, char** argv) {
    const char* device = "cpu";
    int batch = 64;
    int hidden = 1024;
    int iters = 40;
    int warmup = 10;
    int total;
    pyc_backend backend = PYC_BACKEND_CPU;
    pyc_ir_module module;
    pyc_model_desc desc;
    pyc_compile_options opts;
    pyc_compiled_model* model = NULL;
    pyc_tensor inputs[2];
    pyc_tensor outputs[1];
    float* lhs = NULL;
    float* rhs = NULL;
    float* out = NULL;
    int elements_lhs;
    int elements_rhs;
    int elements_out;
    int i;
    int s;
    pyc_run_stats rs;
    double* samples = NULL;
    size_t n = 0;
    double mean = 0.0;
    double p50;
    double p95;
    double min_v;
    double max_v;
    double sum_dispatch = 0.0;
    double sum_controller = 0.0;
    double sum_kernel = 0.0;
    double sum_graph = 0.0;
    clock_t run_a;
    clock_t run_b;

    if (argc >= 2) device = argv[1];
    if (argc >= 3) batch = atoi(argv[2]);
    if (argc >= 4) hidden = atoi(argv[3]);
    if (argc >= 5) iters = atoi(argv[4]);
    if (argc >= 6) warmup = atoi(argv[5]);
    if (batch <= 0 || hidden <= 0 || iters <= 0 || warmup < 0) {
        printf("{\"status\":\"error\",\"error\":\"invalid args\"}\n");
        return 1;
    }
    if (strcmp(device, "cuda") == 0) {
        backend = PYC_BACKEND_CUDA;
    }

    elements_lhs = batch * hidden;
    elements_rhs = hidden * hidden;
    elements_out = batch * hidden;
    lhs = (float*)malloc((size_t)elements_lhs * sizeof(float));
    rhs = (float*)malloc((size_t)elements_rhs * sizeof(float));
    out = (float*)malloc((size_t)elements_out * sizeof(float));
    samples = (double*)malloc((size_t)iters * sizeof(double));
    if (!lhs || !rhs || !out || !samples) {
        printf("{\"status\":\"error\",\"error\":\"alloc failed\"}\n");
        free(lhs); free(rhs); free(out); free(samples);
        return 1;
    }

    for (i = 0; i < elements_lhs; ++i) lhs[i] = (float)((i % 13) - 6) * 0.1f;
    for (i = 0; i < elements_rhs; ++i) rhs[i] = (float)((i % 17) - 8) * 0.05f;
    memset(out, 0, (size_t)elements_out * sizeof(float));

    build_matmul_module(&module, batch, hidden);
    memset(&desc, 0, sizeof(desc));
    desc.module = &module;
    desc.backend = backend;

    memset(&opts, 0, sizeof(opts));
    opts.enable_fusion = 1;
    opts.enable_memory_reuse = 1;
    opts.enable_autotune = 0;
    opts.objective_mode = PYC_MODE_BALANCED;
    opts.target_utilization_floor = 0.70;
    opts.deterministic_strict = 1;
    opts.compile_budget_ms = 0.0;
    opts.cache_mode = PYC_COMPILE_CACHE_IN_MEMORY;
    pyc_runtime_rails_default(&opts.rails);

    if (pyc_compile_model(&desc, &opts, &model) != PYC_STATUS_OK || !model) {
        printf("{\"status\":\"error\",\"error\":\"compile failed\"}\n");
        free(lhs); free(rhs); free(out); free(samples);
        return 1;
    }

    memset(inputs, 0, sizeof(inputs));
    inputs[0].data = lhs;
    inputs[0].size_bytes = (size_t)elements_lhs * sizeof(float);
    inputs[0].dtype = PYC_DTYPE_F32;
    inputs[0].shape.rank = 2;
    inputs[0].shape.dims[0] = batch;
    inputs[0].shape.dims[1] = hidden;
    inputs[1].data = rhs;
    inputs[1].size_bytes = (size_t)elements_rhs * sizeof(float);
    inputs[1].dtype = PYC_DTYPE_F32;
    inputs[1].shape.rank = 2;
    inputs[1].shape.dims[0] = hidden;
    inputs[1].shape.dims[1] = hidden;

    memset(outputs, 0, sizeof(outputs));
    outputs[0].data = out;
    outputs[0].size_bytes = (size_t)elements_out * sizeof(float);
    outputs[0].dtype = PYC_DTYPE_F32;
    outputs[0].shape.rank = 2;
    outputs[0].shape.dims[0] = batch;
    outputs[0].shape.dims[1] = hidden;

    total = warmup + iters;
    for (i = 0; i < total; ++i) {
        run_a = clock();
        s = pyc_run_model(model, inputs, 2, outputs, 1, &rs);
        run_b = clock();
        if (s != PYC_STATUS_OK) {
            printf("{\"status\":\"error\",\"error\":\"run failed\"}\n");
            pyc_destroy_model(model);
            free(lhs); free(rhs); free(out); free(samples);
            return 1;
        }
        if (i >= warmup) {
            double ms = elapsed_ms(run_a, run_b);
            samples[n++] = ms;
            sum_dispatch += rs.dispatch_ms;
            sum_controller += rs.controller_ms;
            sum_kernel += rs.kernel_select_ms;
            sum_graph += rs.graph_exec_ms;
        }
    }

    min_v = samples[0];
    max_v = samples[0];
    for (i = 0; i < (int)n; ++i) {
        mean += samples[i];
        if (samples[i] < min_v) min_v = samples[i];
        if (samples[i] > max_v) max_v = samples[i];
    }
    mean /= (double)n;
    p50 = percentile(samples, n, 50.0);
    p95 = percentile(samples, n, 95.0);

    printf(
        "{\"status\":\"ok\",\"backend\":\"pyc_compiler_next\",\"device\":\"%s\",\"batch\":%d,\"hidden\":%d,\"iters\":%d,\"warmup\":%d,"
        "\"latency_ms\":{\"mean\":%.4f,\"p50\":%.4f,\"p95\":%.4f,\"min\":%.4f,\"max\":%.4f},"
        "\"throughput_tokens_per_sec\":%.2f,\"peak_memory_bytes\":%zu,"
        "\"profile\":{\"dispatch_ms_mean\":%.4f,\"graph_exec_ms_mean\":%.4f,\"controller_ms_mean\":%.4f,\"kernel_select_ms_mean\":%.4f},"
        "\"reliability\":{\"compile_cache_hit\":%d,\"compile_budget_exceeded\":%d,\"guard_miss_count\":%zu,\"fallback_count\":%zu,\"graph_break_count\":%zu,\"compilability_score\":%.4f,\"autotune_loaded\":%d,\"autotune_saved\":%d}}"
        "\n",
        device, batch, hidden, iters, warmup,
        mean, p50, p95, min_v, max_v,
        ((double)batch * (double)hidden / mean) * 1000.0,
        rs.peak_bytes,
        sum_dispatch / (double)n,
        sum_graph / (double)n,
        sum_controller / (double)n,
        sum_kernel / (double)n,
        rs.compile_cache_hit,
        rs.compile_budget_exceeded,
        rs.guard_miss_count,
        rs.fallback_count,
        rs.graph_break_count,
        rs.compilability_score,
        rs.autotune_loaded,
        rs.autotune_saved
    );

    pyc_destroy_model(model);
    free(lhs); free(rhs); free(out); free(samples);
    return 0;
}
