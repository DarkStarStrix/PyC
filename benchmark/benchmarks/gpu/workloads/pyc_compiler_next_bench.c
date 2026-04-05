#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "pyc/compiler_api.h"

#if defined(PYC_HAVE_CUDA_RUNTIME)
#include <cuda_runtime_api.h>
#endif

#define BENCH_SIGNATURE_MAX 128

static double now_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec * 1000.0) + ((double)ts.tv_nsec / 1000000.0);
}

static int env_default_true(const char* name) {
    const char* value = getenv(name);
    if (!value || value[0] == '\0') {
        return 1;
    }
    return !(strcmp(value, "0") == 0 ||
             strcmp(value, "false") == 0 ||
             strcmp(value, "FALSE") == 0);
}

static int env_default_false(const char* name) {
    const char* value = getenv(name);
    if (!value || value[0] == '\0') {
        return 0;
    }
    return !(strcmp(value, "0") == 0 ||
             strcmp(value, "false") == 0 ||
             strcmp(value, "FALSE") == 0);
}

static int env_int_or(const char* name, int fallback) {
    const char* value = getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    return atoi(value);
}

static size_t env_size_or(const char* name, size_t fallback) {
    const char* value = getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    return (size_t)strtoull(value, NULL, 10);
}

static double env_double_or(const char* name, double fallback) {
    const char* value = getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    return strtod(value, NULL);
}

static const char* env_string_or(const char* name, const char* fallback) {
    const char* value = getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    return value;
}

static pyc_objective_mode env_objective_mode_or(const char* name, pyc_objective_mode fallback) {
    const char* value = env_string_or(name, "");
    if (strcmp(value, "balanced") == 0) {
        return PYC_MODE_BALANCED;
    }
    if (strcmp(value, "memory_first") == 0) {
        return PYC_MODE_MEMORY_FIRST;
    }
    if (strcmp(value, "utilization_first") == 0) {
        return PYC_MODE_UTILIZATION_FIRST;
    }
    return fallback;
}

static const char* objective_mode_name(pyc_objective_mode mode) {
    switch (mode) {
        case PYC_MODE_MEMORY_FIRST:
            return "memory_first";
        case PYC_MODE_UTILIZATION_FIRST:
            return "utilization_first";
        case PYC_MODE_BALANCED:
        default:
            return "balanced";
    }
}

static const char* rollback_reason_name(pyc_rollback_reason reason) {
    switch (reason) {
        case PYC_ROLLBACK_LATENCY:
            return "latency";
        case PYC_ROLLBACK_THROUGHPUT:
            return "throughput";
        case PYC_ROLLBACK_PRESSURE:
            return "pressure";
        case PYC_ROLLBACK_RUNTIME_ERROR:
            return "runtime_error";
        case PYC_ROLLBACK_NONE:
        default:
            return "none";
    }
}

static void set_default_env_if_missing(const char* name, const char* value) {
    if (!name || !value || getenv(name) != NULL) {
        return;
    }
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 0);
#endif
}

static float* alloc_host_f32(size_t count, int prefer_pinned, int* out_is_pinned) {
    if (out_is_pinned) {
        *out_is_pinned = 0;
    }
#if defined(PYC_HAVE_CUDA_RUNTIME)
    float* ptr = NULL;
    if (prefer_pinned) {
        if (cudaMallocHost((void**)&ptr, count * sizeof(float)) == cudaSuccess) {
            if (out_is_pinned) {
                *out_is_pinned = 1;
            }
            return ptr;
        }
    }
#else
    (void)prefer_pinned;
#endif
    return (float*)malloc(count * sizeof(float));
}

static void free_host_f32(float* ptr, int use_pinned) {
    if (!ptr) {
        return;
    }
#if defined(PYC_HAVE_CUDA_RUNTIME)
    if (use_pinned) {
        (void)cudaFreeHost(ptr);
        return;
    }
#else
    (void)use_pinned;
#endif
    free(ptr);
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

typedef struct {
    int m;
    int k;
    int n;
} gemm_shape_step;

typedef struct {
    int index;
    int m;
    int k;
    int n;
    double mean_ms;
    double p50_ms;
    double p95_ms;
    double min_ms;
    double max_ms;
    double tflops;
    double dispatch_ms_mean;
    double graph_exec_ms_mean;
    double controller_ms_mean;
    double kernel_select_ms_mean;
    double cuda_copy_in_ms_mean;
    double cuda_kernel_ms_mean;
    double cuda_copy_out_ms_mean;
    double cuda_sync_ms_mean;
    int selected_kernel_count;
    size_t selected_kernel_candidates;
    char selected_kernel_symbol[PYC_KERNEL_SYMBOL_MAX];
    int compile_cache_hit;
    int speculative_plan_hit;
    char controller_objective_mode[32];
    char controller_shadow_mode[32];
    char controller_shadow_reason[32];
    char controller_rollback_reason[32];
    size_t controller_rollback_count;
    char execution_path[96];
    size_t rematerialized_tensors;
    size_t rematerialized_bytes;
    int phantom_match;
    size_t phantom_match_delta;
    size_t phantom_mismatch_delta;
    size_t phantom_reshape_delta;
    double phantom_confidence;
    double phantom_match_score;
    char phantom_expected_signature[BENCH_SIGNATURE_MAX];
    char phantom_observed_signature[BENCH_SIGNATURE_MAX];
} gemm_sequence_result;

static void populate_controller_telemetry(
    const pyc_run_stats* rs,
    char* out_objective_mode,
    size_t out_objective_mode_size,
    char* out_shadow_mode,
    size_t out_shadow_mode_size,
    char* out_shadow_reason,
    size_t out_shadow_reason_size,
    char* out_rollback_reason,
    size_t out_rollback_reason_size,
    size_t* out_rollback_count) {
    if (out_objective_mode && out_objective_mode_size > 0) {
        snprintf(out_objective_mode, out_objective_mode_size, "%s", objective_mode_name(rs->active_mode));
    }
    if (out_shadow_mode && out_shadow_mode_size > 0) {
        snprintf(out_shadow_mode, out_shadow_mode_size, "%s", objective_mode_name(rs->shadow_mode));
    }
    if (out_shadow_reason && out_shadow_reason_size > 0) {
        snprintf(out_shadow_reason, out_shadow_reason_size, "%s", rollback_reason_name(rs->shadow_reason));
    }
    if (out_rollback_reason && out_rollback_reason_size > 0) {
        snprintf(out_rollback_reason, out_rollback_reason_size, "%s", rollback_reason_name(rs->rollback_reason));
    }
    if (out_rollback_count) {
        *out_rollback_count = rs->rollback_count;
    }
}

static void populate_execution_path(
    const pyc_run_stats* rs,
    char* out_execution_path,
    size_t out_execution_path_size) {
    if (!out_execution_path || out_execution_path_size == 0) {
        return;
    }
    snprintf(
        out_execution_path,
        out_execution_path_size,
        "%s",
        (rs && rs->execution_path[0] != '\0') ? rs->execution_path : "unknown");
}

static int parse_shape_token(const char* token, gemm_shape_step* out_step) {
    int m = 0;
    int k = 0;
    int n = 0;
    if (!token || !out_step) {
        return -1;
    }
    if (sscanf(token, "%dx%dx%d", &m, &k, &n) != 3) {
        return -1;
    }
    if (m <= 0 || k <= 0 || n <= 0) {
        return -1;
    }
    out_step->m = m;
    out_step->k = k;
    out_step->n = n;
    return 0;
}

static int parse_gemm_sequence(
    const char* raw,
    gemm_shape_step* out_steps,
    size_t max_steps,
    size_t* out_count) {
    char buffer[1024];
    char* cursor;
    char* token;
    size_t count = 0;

    if (out_count) {
        *out_count = 0;
    }
    if (!raw || !out_steps || max_steps == 0) {
        return -1;
    }
    if (strlen(raw) >= sizeof(buffer)) {
        return -1;
    }

    strncpy(buffer, raw, sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\0';
    cursor = buffer;
    while ((token = strsep(&cursor, ";")) != NULL) {
        while (*token == ' ' || *token == '\t' || *token == '\n') {
            token++;
        }
        if (*token == '\0') {
            continue;
        }
        if (count >= max_steps || parse_shape_token(token, &out_steps[count]) != 0) {
            return -1;
        }
        count++;
    }

    if (count == 0) {
        return -1;
    }
    if (out_count) {
        *out_count = count;
    }
    return 0;
}

static void fill_host_pattern(float* dst, size_t count, int seed, float scale) {
    size_t i;
    if (!dst) {
        return;
    }
    for (i = 0; i < count; ++i) {
        int value = (int)((i + (size_t)seed) % 23) - 11;
        dst[i] = (float)value * scale;
    }
}

static void print_json_string(const char* value) {
    const char* p = value ? value : "";
    putchar('"');
    for (; *p != '\0'; ++p) {
        switch (*p) {
            case '\\':
                fputs("\\\\", stdout);
                break;
            case '"':
                fputs("\\\"", stdout);
                break;
            case '\n':
                fputs("\\n", stdout);
                break;
            case '\r':
                fputs("\\r", stdout);
                break;
            case '\t':
                fputs("\\t", stdout);
                break;
            default:
                putchar(*p);
                break;
        }
    }
    putchar('"');
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

static void build_gemm_module(pyc_ir_module* m, int m_dim, int k_dim, int n_dim) {
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
    lhs.shape.dims[0] = m_dim;
    lhs.shape.dims[1] = k_dim;
    pyc_ir_add_op(m, &lhs);

    memset(&rhs, 0, sizeof(rhs));
    rhs.kind = PYC_IR_OP_INPUT;
    strcpy(rhs.name, "rhs");
    rhs.dtype = PYC_DTYPE_F32;
    rhs.shape.rank = 2;
    rhs.shape.dims[0] = k_dim;
    rhs.shape.dims[1] = n_dim;
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
    out.shape.dims[0] = m_dim;
    out.shape.dims[1] = n_dim;
    out.input_ids[0] = 2;
    out.input_count = 1;
    pyc_ir_add_op(m, &out);
}

int main(int argc, char** argv) {
    const char* device = "cpu";
    const char* task = getenv("BENCH_TASK");
    const char* sequence_raw = getenv("BENCH_SEQUENCE");
    int batch = 64;
    int hidden = 1024;
    int m_dim = 64;
    int k_dim = 1024;
    int n_dim = 1024;
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
    gemm_shape_step sequence_steps[16];
    gemm_sequence_result sequence_results[16];
    size_t sequence_count = 0;
    size_t sequence_index = 0;
    size_t max_elements_lhs = 0;
    size_t max_elements_rhs = 0;
    size_t max_elements_out = 0;
    int elements_lhs;
    int elements_rhs;
    int elements_out;
    int i;
    int s;
    pyc_run_stats rs;
    char controller_objective_mode[32];
    char controller_shadow_mode[32];
    char controller_shadow_reason[32];
    char controller_rollback_reason[32];
    size_t controller_rollback_count = 0;
    char execution_path[96];
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
    double sum_copy_in = 0.0;
    double sum_cuda_kernel = 0.0;
    double sum_copy_out = 0.0;
    double sum_sync = 0.0;
    double run_a_ms;
    double run_b_ms;
    int pinned_host_buffers = 0;
    int lhs_pinned = 0;
    int rhs_pinned = 0;
    int out_pinned = 0;
    int sequence_mode = 0;

    if (argc >= 2) device = argv[1];
    if (argc >= 3) batch = atoi(argv[2]);
    if (argc >= 4) hidden = atoi(argv[3]);
    if (argc >= 5) iters = atoi(argv[4]);
    if (argc >= 6) warmup = atoi(argv[5]);
    if (!task || task[0] == '\0') {
        task = "mlp";
    }
    if (strcmp(task, "gemm") == 0) {
        const char* env_m = getenv("BENCH_M");
        const char* env_k = getenv("BENCH_K");
        const char* env_n = getenv("BENCH_N");
        if (env_m) m_dim = atoi(env_m);
        if (env_k) k_dim = atoi(env_k);
        if (env_n) n_dim = atoi(env_n);
        if (m_dim <= 0) m_dim = batch;
        if (k_dim <= 0) k_dim = hidden;
        if (n_dim <= 0) n_dim = hidden;
    }
    if (strcmp(task, "gemm") == 0 && sequence_raw && sequence_raw[0] != '\0') {
        if (parse_gemm_sequence(sequence_raw, sequence_steps, sizeof(sequence_steps) / sizeof(sequence_steps[0]), &sequence_count) != 0) {
            printf("{\"status\":\"error\",\"error\":\"invalid BENCH_SEQUENCE\"}\n");
            return 1;
        }
        sequence_mode = sequence_count > 1 ? 1 : 0;
        m_dim = sequence_steps[0].m;
        k_dim = sequence_steps[0].k;
        n_dim = sequence_steps[0].n;
    }
    if (batch <= 0 || hidden <= 0 || iters <= 0 || warmup < 0) {
        printf("{\"status\":\"error\",\"error\":\"invalid args\"}\n");
        return 1;
    }
    if (strcmp(device, "cuda") == 0) {
        backend = PYC_BACKEND_CUDA;
        pinned_host_buffers = env_default_true("PYC_BENCH_USE_PINNED_HOST");
        set_default_env_if_missing("PYC_CUDA_ASSUME_STATIC_LHS", "1");
        set_default_env_if_missing("PYC_CUDA_ASSUME_STATIC_RHS", "1");
        set_default_env_if_missing("PYC_CUDA_SKIP_HOST_OUTPUT_COPY", "1");
        set_default_env_if_missing("PYC_ENABLE_RUNTIME_DECISION_LOG", "0");
    }

    if (strcmp(task, "gemm") == 0) {
        if (sequence_mode) {
            for (sequence_index = 0; sequence_index < sequence_count; ++sequence_index) {
                size_t lhs_count = (size_t)sequence_steps[sequence_index].m * (size_t)sequence_steps[sequence_index].k;
                size_t rhs_count = (size_t)sequence_steps[sequence_index].k * (size_t)sequence_steps[sequence_index].n;
                size_t out_count = (size_t)sequence_steps[sequence_index].m * (size_t)sequence_steps[sequence_index].n;
                if (lhs_count > max_elements_lhs) {
                    max_elements_lhs = lhs_count;
                }
                if (rhs_count > max_elements_rhs) {
                    max_elements_rhs = rhs_count;
                }
                if (out_count > max_elements_out) {
                    max_elements_out = out_count;
                }
            }
            elements_lhs = (int)max_elements_lhs;
            elements_rhs = (int)max_elements_rhs;
            elements_out = (int)max_elements_out;
        } else {
            elements_lhs = m_dim * k_dim;
            elements_rhs = k_dim * n_dim;
            elements_out = m_dim * n_dim;
        }
    } else {
        elements_lhs = batch * hidden;
        elements_rhs = hidden * hidden;
        elements_out = batch * hidden;
    }
    lhs = alloc_host_f32((size_t)elements_lhs, pinned_host_buffers, &lhs_pinned);
    rhs = alloc_host_f32((size_t)elements_rhs, pinned_host_buffers, &rhs_pinned);
    out = alloc_host_f32((size_t)elements_out, pinned_host_buffers, &out_pinned);
    samples = (double*)malloc((size_t)iters * sizeof(double));
    if (!lhs || !rhs || !out || !samples) {
        printf("{\"status\":\"error\",\"error\":\"alloc failed\"}\n");
        free_host_f32(lhs, lhs_pinned);
        free_host_f32(rhs, rhs_pinned);
        free_host_f32(out, out_pinned);
        free(samples);
        return 1;
    }

    fill_host_pattern(lhs, (size_t)elements_lhs, 0, 0.1f);
    fill_host_pattern(rhs, (size_t)elements_rhs, 7, 0.05f);
    memset(out, 0, (size_t)elements_out * sizeof(float));

    if (strcmp(task, "gemm") == 0) {
        build_gemm_module(&module, m_dim, k_dim, n_dim);
    } else {
        build_matmul_module(&module, batch, hidden);
    }
    memset(&desc, 0, sizeof(desc));
    desc.module = &module;
    desc.backend = backend;

    memset(&opts, 0, sizeof(opts));
    opts.enable_fusion = env_default_true("PYC_BENCH_ENABLE_FUSION");
    opts.enable_memory_reuse = env_default_true("PYC_BENCH_ENABLE_MEMORY_REUSE");
    opts.enable_autotune = env_default_false("PYC_BENCH_ENABLE_AUTOTUNE");
    opts.enable_speculative_plans = env_default_false("PYC_BENCH_ENABLE_SPECULATIVE_PLANS");
    opts.enable_phantom_graph = env_default_false("PYC_BENCH_ENABLE_PHANTOM_GRAPH");
    opts.max_speculative_plans = env_size_or("PYC_BENCH_MAX_SPECULATIVE_PLANS", opts.enable_speculative_plans ? 4 : 0);
    opts.phantom_horizon_steps = env_size_or("PYC_BENCH_PHANTOM_HORIZON_STEPS", opts.enable_phantom_graph ? 1 : 0);
    opts.objective_mode = env_objective_mode_or("PYC_BENCH_OBJECTIVE_MODE", PYC_MODE_BALANCED);
    opts.memory_budget_bytes = env_size_or("PYC_BENCH_MEMORY_BUDGET_BYTES", 0);
    opts.target_utilization_floor = env_double_or("PYC_BENCH_TARGET_UTILIZATION_FLOOR", 0.70);
    opts.deterministic_strict = env_default_true("PYC_BENCH_DETERMINISTIC_STRICT");
    opts.compile_budget_ms = env_double_or("PYC_BENCH_COMPILE_BUDGET_MS", 0.0);
    opts.cache_mode = env_default_true("PYC_BENCH_CACHE_IN_MEMORY") ? PYC_COMPILE_CACHE_IN_MEMORY : PYC_COMPILE_CACHE_DISABLED;
    pyc_runtime_rails_default(&opts.rails);

    if (pyc_compile_model(&desc, &opts, &model) != PYC_STATUS_OK || !model) {
        printf("{\"status\":\"error\",\"error\":\"compile failed\"}\n");
        free_host_f32(lhs, lhs_pinned);
        free_host_f32(rhs, rhs_pinned);
        free_host_f32(out, out_pinned);
        free(samples);
        return 1;
    }

    memset(inputs, 0, sizeof(inputs));
    inputs[0].data = lhs;
    inputs[0].size_bytes = (size_t)elements_lhs * sizeof(float);
    inputs[0].dtype = PYC_DTYPE_F32;
    inputs[0].shape.rank = 2;
    inputs[0].shape.dims[0] = strcmp(task, "gemm") == 0 ? m_dim : batch;
    inputs[0].shape.dims[1] = strcmp(task, "gemm") == 0 ? k_dim : hidden;
    inputs[1].data = rhs;
    inputs[1].size_bytes = (size_t)elements_rhs * sizeof(float);
    inputs[1].dtype = PYC_DTYPE_F32;
    inputs[1].shape.rank = 2;
    inputs[1].shape.dims[0] = strcmp(task, "gemm") == 0 ? k_dim : hidden;
    inputs[1].shape.dims[1] = strcmp(task, "gemm") == 0 ? n_dim : hidden;

    memset(outputs, 0, sizeof(outputs));
    outputs[0].data = out;
    outputs[0].size_bytes = (size_t)elements_out * sizeof(float);
    outputs[0].dtype = PYC_DTYPE_F32;
    outputs[0].shape.rank = 2;
    outputs[0].shape.dims[0] = strcmp(task, "gemm") == 0 ? m_dim : batch;
    outputs[0].shape.dims[1] = strcmp(task, "gemm") == 0 ? n_dim : hidden;

    if (sequence_mode) {
        size_t prior_match_count = 0;
        size_t prior_mismatch_count = 0;
        size_t prior_reshape_count = 0;
        for (sequence_index = 0; sequence_index < sequence_count; ++sequence_index) {
            const gemm_shape_step* step = &sequence_steps[sequence_index];
            gemm_sequence_result* result = &sequence_results[sequence_index];
            size_t step_elements_lhs = (size_t)step->m * (size_t)step->k;
            size_t step_elements_rhs = (size_t)step->k * (size_t)step->n;
            size_t step_elements_out = (size_t)step->m * (size_t)step->n;

            n = 0;
            mean = 0.0;
            sum_dispatch = 0.0;
            sum_controller = 0.0;
            sum_kernel = 0.0;
            sum_graph = 0.0;
            sum_copy_in = 0.0;
            sum_cuda_kernel = 0.0;
            sum_copy_out = 0.0;
            sum_sync = 0.0;
            memset(samples, 0, (size_t)iters * sizeof(double));
            memset(out, 0, step_elements_out * sizeof(float));
            fill_host_pattern(lhs, step_elements_lhs, (int)(sequence_index * 17), 0.1f);
            fill_host_pattern(rhs, step_elements_rhs, (int)(sequence_index * 29), 0.05f);

            inputs[0].size_bytes = step_elements_lhs * sizeof(float);
            inputs[0].shape.dims[0] = step->m;
            inputs[0].shape.dims[1] = step->k;
            inputs[1].size_bytes = step_elements_rhs * sizeof(float);
            inputs[1].shape.dims[0] = step->k;
            inputs[1].shape.dims[1] = step->n;
            outputs[0].size_bytes = step_elements_out * sizeof(float);
            outputs[0].shape.dims[0] = step->m;
            outputs[0].shape.dims[1] = step->n;

            total = warmup + iters;
            for (i = 0; i < total; ++i) {
                run_a_ms = now_ms();
                s = pyc_run_model(model, inputs, 2, outputs, 1, &rs);
                run_b_ms = now_ms();
                if (s != PYC_STATUS_OK) {
                    printf(
                        "{\"status\":\"error\",\"error\":\"run failed\",\"sequence_index\":%zu,\"m\":%d,\"k\":%d,\"n\":%d}\n",
                        sequence_index,
                        step->m,
                        step->k,
                        step->n);
                    pyc_destroy_model(model);
                    free_host_f32(lhs, lhs_pinned);
                    free_host_f32(rhs, rhs_pinned);
                    free_host_f32(out, out_pinned);
                    free(samples);
                    return 1;
                }
                if (i >= warmup) {
                    double ms = run_b_ms - run_a_ms;
                    samples[n++] = ms;
                    sum_dispatch += rs.dispatch_ms;
                    sum_controller += rs.controller_ms;
                    sum_kernel += rs.kernel_select_ms;
                    sum_graph += rs.graph_exec_ms;
                    sum_copy_in += rs.cuda_copy_in_ms;
                    sum_cuda_kernel += rs.cuda_kernel_ms;
                    sum_copy_out += rs.cuda_copy_out_ms;
                    sum_sync += rs.cuda_sync_ms;
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

            memset(result, 0, sizeof(*result));
            result->index = (int)sequence_index;
            result->m = step->m;
            result->k = step->k;
            result->n = step->n;
            result->mean_ms = mean;
            result->p50_ms = p50;
            result->p95_ms = p95;
            result->min_ms = min_v;
            result->max_ms = max_v;
            result->tflops = ((((double)2 * (double)step->m * (double)step->k * (double)step->n / mean) * 1000.0) / 1.0e12);
            result->dispatch_ms_mean = sum_dispatch / (double)n;
            result->graph_exec_ms_mean = sum_graph / (double)n;
            result->controller_ms_mean = sum_controller / (double)n;
            result->kernel_select_ms_mean = sum_kernel / (double)n;
            result->cuda_copy_in_ms_mean = sum_copy_in / (double)n;
            result->cuda_kernel_ms_mean = sum_cuda_kernel / (double)n;
            result->cuda_copy_out_ms_mean = sum_copy_out / (double)n;
            result->cuda_sync_ms_mean = sum_sync / (double)n;
            result->selected_kernel_count = rs.selected_kernel_count;
            result->selected_kernel_candidates = rs.selected_kernel_candidates;
            strncpy(
                result->selected_kernel_symbol,
                rs.selected_kernel_symbol,
                sizeof(result->selected_kernel_symbol) - 1);
            result->compile_cache_hit = rs.compile_cache_hit;
            result->speculative_plan_hit = rs.speculative_plan_hit;
            populate_controller_telemetry(
                &rs,
                result->controller_objective_mode,
                sizeof(result->controller_objective_mode),
                result->controller_shadow_mode,
                sizeof(result->controller_shadow_mode),
                result->controller_shadow_reason,
                sizeof(result->controller_shadow_reason),
                result->controller_rollback_reason,
                sizeof(result->controller_rollback_reason),
                &result->controller_rollback_count);
            populate_execution_path(&rs, result->execution_path, sizeof(result->execution_path));
            result->rematerialized_tensors = rs.rematerialized_tensors;
            result->rematerialized_bytes = rs.rematerialized_bytes;
            result->phantom_match = rs.phantom_graph_match;
            result->phantom_match_delta = rs.phantom_graph_match_count - prior_match_count;
            result->phantom_mismatch_delta = rs.phantom_graph_mismatch_count - prior_mismatch_count;
            result->phantom_reshape_delta = rs.phantom_graph_reshape_count - prior_reshape_count;
            result->phantom_confidence = rs.phantom_graph_confidence;
            result->phantom_match_score = rs.phantom_graph_match_score;
            strncpy(
                result->phantom_expected_signature,
                rs.phantom_graph_expected_signature,
                sizeof(result->phantom_expected_signature) - 1);
            strncpy(
                result->phantom_observed_signature,
                rs.phantom_graph_observed_signature,
                sizeof(result->phantom_observed_signature) - 1);

            prior_match_count = rs.phantom_graph_match_count;
            prior_mismatch_count = rs.phantom_graph_mismatch_count;
            prior_reshape_count = rs.phantom_graph_reshape_count;
        }

        populate_controller_telemetry(
            &rs,
            controller_objective_mode,
            sizeof(controller_objective_mode),
            controller_shadow_mode,
            sizeof(controller_shadow_mode),
            controller_shadow_reason,
            sizeof(controller_shadow_reason),
            controller_rollback_reason,
            sizeof(controller_rollback_reason),
            &controller_rollback_count);
        populate_execution_path(&rs, execution_path, sizeof(execution_path));
        printf(
            "{\"status\":\"ok\",\"backend\":\"pyc_compiler_next\",\"task\":\"gemm_sequence\",\"device\":\"%s\",\"iters\":%d,\"warmup\":%d,",
            device,
            iters,
            warmup);
        fputs("\"controller\":{\"objective_mode\":", stdout);
        print_json_string(controller_objective_mode);
        fputs(",\"shadow_mode\":", stdout);
        print_json_string(controller_shadow_mode);
        fputs(",\"shadow_reason\":", stdout);
        print_json_string(controller_shadow_reason);
        fputs(",\"rollback_reason\":", stdout);
        print_json_string(controller_rollback_reason);
        printf(",\"rollback_count\":%zu},", controller_rollback_count);
        printf(
            "\"sequence\":{\"count\":%zu,\"base\":{\"m\":%d,\"k\":%d,\"n\":%d},\"steps\":[",
            sequence_count,
            sequence_steps[0].m,
            sequence_steps[0].k,
            sequence_steps[0].n);
        for (sequence_index = 0; sequence_index < sequence_count; ++sequence_index) {
            const gemm_sequence_result* result = &sequence_results[sequence_index];
            if (sequence_index > 0) {
                putchar(',');
            }
            printf(
                "{\"index\":%d,\"m\":%d,\"k\":%d,\"n\":%d,"
                "\"latency_ms\":{\"mean\":%.4f,\"p50\":%.4f,\"p95\":%.4f,\"min\":%.4f,\"max\":%.4f},"
                "\"throughput_tflops_per_sec\":%.4f,"
                "\"profile\":{\"dispatch_ms_mean\":%.4f,\"graph_exec_ms_mean\":%.4f,\"controller_ms_mean\":%.4f,\"kernel_select_ms_mean\":%.4f,"
                "\"cuda_copy_in_ms\":%.4f,\"cuda_kernel_ms\":%.4f,\"cuda_copy_out_ms\":%.4f,\"cuda_sync_ms\":%.4f},"
                "\"execution_path\":",
                result->index,
                result->m,
                result->k,
                result->n,
                result->mean_ms,
                result->p50_ms,
                result->p95_ms,
                result->min_ms,
                result->max_ms,
                result->tflops,
                result->dispatch_ms_mean,
                result->graph_exec_ms_mean,
                result->controller_ms_mean,
                result->kernel_select_ms_mean,
                result->cuda_copy_in_ms_mean,
                result->cuda_kernel_ms_mean,
                result->cuda_copy_out_ms_mean,
                result->cuda_sync_ms_mean);
            print_json_string(result->execution_path);
            fputs(",\"kernel_selection\":{\"count\":", stdout);
            printf(
                "%d,\"candidates\":%zu,\"symbol\":",
                result->selected_kernel_count,
                result->selected_kernel_candidates);
            print_json_string(result->selected_kernel_symbol);
            fputs("},\"reliability\":{", stdout);
            printf(
                "\"compile_cache_hit\":%d,\"speculative_plan_hit\":%d,\"rematerialized_tensors\":%zu,\"rematerialized_bytes\":%zu},\"phantom_graph\":{\"match\":%s,\"match_delta\":%zu,\"mismatch_delta\":%zu,\"reshape_delta\":%zu,\"confidence\":%.4f,\"match_score\":%.4f,\"expected_signature\":",
                result->compile_cache_hit,
                result->speculative_plan_hit,
                result->rematerialized_tensors,
                result->rematerialized_bytes,
                result->phantom_match ? "true" : "false",
                result->phantom_match_delta,
                result->phantom_mismatch_delta,
                result->phantom_reshape_delta,
                result->phantom_confidence,
                result->phantom_match_score);
            print_json_string(result->phantom_expected_signature);
            fputs(",\"observed_signature\":", stdout);
            print_json_string(result->phantom_observed_signature);
            putchar('}');
            putchar('}');
        }
        printf(
            "],\"summary\":{\"phantom_match_count\":%zu,\"phantom_mismatch_count\":%zu,\"phantom_reshape_count\":%zu,\"final_confidence\":%.4f}}",
            rs.phantom_graph_match_count,
            rs.phantom_graph_mismatch_count,
            rs.phantom_graph_reshape_count,
            rs.phantom_graph_confidence);
        printf(
            ",\"compile_options\":{\"enable_fusion\":%s,\"enable_memory_reuse\":%s,\"enable_autotune\":%s,\"enable_speculative_plans\":%s,\"enable_phantom_graph\":%s,\"max_speculative_plans\":%zu,\"phantom_horizon_steps\":%zu,"
            "\"objective_mode\":\"%s\",\"memory_budget_bytes\":%zu,\"target_utilization_floor\":%.4f,\"deterministic_strict\":%s,"
            "\"compile_budget_ms\":%.4f,\"cache_mode\":\"%s\"},"
            "\"reliability\":{\"compile_cache_hit\":%d,\"compile_budget_exceeded\":%d,\"guard_miss_count\":%zu,\"fallback_count\":%zu,\"graph_break_count\":%zu,\"compilability_score\":%.4f,\"autotune_loaded\":%d,\"autotune_saved\":%d,\"rematerialized_tensors\":%zu,\"rematerialized_bytes\":%zu},"
            "\"phantom_graph\":{\"enabled\":%s,\"match\":%s,\"match_count\":%zu,\"mismatch_count\":%zu,\"reshape_count\":%zu,\"confidence\":%.4f,\"match_score\":%.4f,"
            "\"expected_bucket\":\"%s\",\"expected_signature\":\"%s\",\"observed_signature\":\"%s\"},"
            "\"precision_policy\":{\"allow_tf32\":%s,\"pinned_host_buffers\":%s,\"assume_static_lhs\":%s,\"assume_static_rhs\":%s,\"skip_host_output_copy\":%s}}",
            opts.enable_fusion ? "true" : "false",
            opts.enable_memory_reuse ? "true" : "false",
            opts.enable_autotune ? "true" : "false",
            opts.enable_speculative_plans ? "true" : "false",
            opts.enable_phantom_graph ? "true" : "false",
            opts.max_speculative_plans,
            opts.phantom_horizon_steps,
            objective_mode_name(opts.objective_mode),
            opts.memory_budget_bytes,
            opts.target_utilization_floor,
            opts.deterministic_strict ? "true" : "false",
            opts.compile_budget_ms,
            opts.cache_mode == PYC_COMPILE_CACHE_IN_MEMORY ? "in_memory" : "disabled",
            rs.compile_cache_hit,
            rs.compile_budget_exceeded,
            rs.guard_miss_count,
            rs.fallback_count,
            rs.graph_break_count,
            rs.compilability_score,
            rs.autotune_loaded,
            rs.autotune_saved,
            rs.rematerialized_tensors,
            rs.rematerialized_bytes,
            rs.phantom_graph_enabled ? "true" : "false",
            rs.phantom_graph_match ? "true" : "false",
            rs.phantom_graph_match_count,
            rs.phantom_graph_mismatch_count,
            rs.phantom_graph_reshape_count,
            rs.phantom_graph_confidence,
            rs.phantom_graph_match_score,
            rs.phantom_graph_expected_bucket,
            rs.phantom_graph_expected_signature,
            rs.phantom_graph_observed_signature,
            env_default_true("PYC_CUDA_ALLOW_TF32") ? "true" : "false",
            pinned_host_buffers ? "true" : "false",
            env_default_false("PYC_CUDA_ASSUME_STATIC_LHS") ? "true" : "false",
            env_default_false("PYC_CUDA_ASSUME_STATIC_RHS") ? "true" : "false",
            env_default_false("PYC_CUDA_SKIP_HOST_OUTPUT_COPY") ? "true" : "false"
        );

        putchar('\n');

        pyc_destroy_model(model);
        free_host_f32(lhs, lhs_pinned);
        free_host_f32(rhs, rhs_pinned);
        free_host_f32(out, out_pinned);
        free(samples);
        return 0;
    }

    total = warmup + iters;
    for (i = 0; i < total; ++i) {
        run_a_ms = now_ms();
        s = pyc_run_model(model, inputs, 2, outputs, 1, &rs);
        run_b_ms = now_ms();
        if (s != PYC_STATUS_OK) {
            printf("{\"status\":\"error\",\"error\":\"run failed\"}\n");
            pyc_destroy_model(model);
            free_host_f32(lhs, lhs_pinned);
            free_host_f32(rhs, rhs_pinned);
            free_host_f32(out, out_pinned);
            free(samples);
            return 1;
        }
        if (i >= warmup) {
            double ms = run_b_ms - run_a_ms;
            samples[n++] = ms;
            sum_dispatch += rs.dispatch_ms;
            sum_controller += rs.controller_ms;
            sum_kernel += rs.kernel_select_ms;
            sum_graph += rs.graph_exec_ms;
            sum_copy_in += rs.cuda_copy_in_ms;
            sum_cuda_kernel += rs.cuda_kernel_ms;
            sum_copy_out += rs.cuda_copy_out_ms;
            sum_sync += rs.cuda_sync_ms;
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
    populate_controller_telemetry(
        &rs,
        controller_objective_mode,
        sizeof(controller_objective_mode),
        controller_shadow_mode,
        sizeof(controller_shadow_mode),
        controller_shadow_reason,
        sizeof(controller_shadow_reason),
        controller_rollback_reason,
        sizeof(controller_rollback_reason),
        &controller_rollback_count);
    populate_execution_path(&rs, execution_path, sizeof(execution_path));

    printf(
        "{\"status\":\"ok\",\"backend\":\"pyc_compiler_next\",\"task\":\"%s\",\"device\":\"%s\",\"batch\":%d,\"hidden\":%d,\"m\":%d,\"k\":%d,\"n\":%d,\"iters\":%d,\"warmup\":%d,"
        "\"latency_ms\":{\"mean\":%.4f,\"p50\":%.4f,\"p95\":%.4f,\"min\":%.4f,\"max\":%.4f},"
        "\"throughput_tokens_per_sec\":%.2f,\"throughput_flops_per_sec\":%.2f,\"throughput_tflops_per_sec\":%.4f,\"peak_memory_bytes\":%zu,"
        "\"profile\":{\"dispatch_ms_mean\":%.4f,\"graph_exec_ms_mean\":%.4f,\"controller_ms_mean\":%.4f,\"kernel_select_ms_mean\":%.4f,"
        "\"cuda_copy_in_ms\":%.4f,\"cuda_kernel_ms\":%.4f,\"cuda_copy_out_ms\":%.4f,\"cuda_sync_ms\":%.4f},",
        task, device, batch, hidden, m_dim, k_dim, n_dim, iters, warmup,
        mean, p50, p95, min_v, max_v,
        strcmp(task, "gemm") == 0 ? 0.0 : ((double)batch * (double)hidden / mean) * 1000.0,
        strcmp(task, "gemm") == 0 ? (((double)2 * (double)m_dim * (double)k_dim * (double)n_dim / mean) * 1000.0) : 0.0,
        strcmp(task, "gemm") == 0 ? ((((double)2 * (double)m_dim * (double)k_dim * (double)n_dim / mean) * 1000.0) / 1.0e12) : 0.0,
        rs.peak_bytes,
        sum_dispatch / (double)n,
        sum_graph / (double)n,
        sum_controller / (double)n,
        sum_kernel / (double)n,
        sum_copy_in / (double)n,
        sum_cuda_kernel / (double)n,
        sum_copy_out / (double)n,
        sum_sync / (double)n);
    fputs("\"controller\":{\"objective_mode\":", stdout);
    print_json_string(controller_objective_mode);
    fputs(",\"shadow_mode\":", stdout);
    print_json_string(controller_shadow_mode);
    fputs(",\"shadow_reason\":", stdout);
    print_json_string(controller_shadow_reason);
    fputs(",\"rollback_reason\":", stdout);
    print_json_string(controller_rollback_reason);
    printf(",\"rollback_count\":%zu},", controller_rollback_count);
    fputs("\"execution_path\":", stdout);
    print_json_string(execution_path);
    fputs(",", stdout);
    fputs("\"kernel_selection\":{\"count\":", stdout);
    printf("%d,\"candidates\":%zu,\"symbol\":", rs.selected_kernel_count, rs.selected_kernel_candidates);
    print_json_string(rs.selected_kernel_symbol);
    fputs("},", stdout);
    printf(
        "\"compile_options\":{\"enable_fusion\":%s,\"enable_memory_reuse\":%s,\"enable_autotune\":%s,\"enable_speculative_plans\":%s,\"enable_phantom_graph\":%s,\"max_speculative_plans\":%zu,\"phantom_horizon_steps\":%zu,"
        "\"objective_mode\":\"%s\",\"memory_budget_bytes\":%zu,\"target_utilization_floor\":%.4f,\"deterministic_strict\":%s,"
        "\"compile_budget_ms\":%.4f,\"cache_mode\":\"%s\"},"
        "\"reliability\":{\"compile_cache_hit\":%d,\"compile_budget_exceeded\":%d,\"guard_miss_count\":%zu,\"fallback_count\":%zu,\"graph_break_count\":%zu,\"compilability_score\":%.4f,\"autotune_loaded\":%d,\"autotune_saved\":%d,\"rematerialized_tensors\":%zu,\"rematerialized_bytes\":%zu},"
        "\"phantom_graph\":{\"enabled\":%s,\"match\":%s,\"match_count\":%zu,\"mismatch_count\":%zu,\"reshape_count\":%zu,\"confidence\":%.4f,\"match_score\":%.4f,"
        "\"expected_bucket\":\"%s\",\"expected_signature\":\"%s\",\"observed_signature\":\"%s\"},"
        "\"precision_policy\":{\"allow_tf32\":%s,\"pinned_host_buffers\":%s,\"assume_static_lhs\":%s,\"assume_static_rhs\":%s,\"skip_host_output_copy\":%s}}"
        "\n",
        opts.enable_fusion ? "true" : "false",
        opts.enable_memory_reuse ? "true" : "false",
        opts.enable_autotune ? "true" : "false",
        opts.enable_speculative_plans ? "true" : "false",
        opts.enable_phantom_graph ? "true" : "false",
        opts.max_speculative_plans,
        opts.phantom_horizon_steps,
        objective_mode_name(opts.objective_mode),
        opts.memory_budget_bytes,
        opts.target_utilization_floor,
        opts.deterministic_strict ? "true" : "false",
        opts.compile_budget_ms,
        opts.cache_mode == PYC_COMPILE_CACHE_IN_MEMORY ? "in_memory" : "disabled",
        rs.compile_cache_hit,
        rs.compile_budget_exceeded,
        rs.guard_miss_count,
        rs.fallback_count,
        rs.graph_break_count,
        rs.compilability_score,
        rs.autotune_loaded,
        rs.autotune_saved,
        rs.rematerialized_tensors,
        rs.rematerialized_bytes,
        rs.phantom_graph_enabled ? "true" : "false",
        rs.phantom_graph_match ? "true" : "false",
        rs.phantom_graph_match_count,
        rs.phantom_graph_mismatch_count,
        rs.phantom_graph_reshape_count,
        rs.phantom_graph_confidence,
        rs.phantom_graph_match_score,
        rs.phantom_graph_expected_bucket,
        rs.phantom_graph_expected_signature,
        rs.phantom_graph_observed_signature,
        env_default_true("PYC_CUDA_ALLOW_TF32") ? "true" : "false",
        pinned_host_buffers ? "true" : "false",
        env_default_false("PYC_CUDA_ASSUME_STATIC_LHS") ? "true" : "false",
        env_default_false("PYC_CUDA_ASSUME_STATIC_RHS") ? "true" : "false",
        env_default_false("PYC_CUDA_SKIP_HOST_OUTPUT_COPY") ? "true" : "false"
    );

    pyc_destroy_model(model);
    free_host_f32(lhs, lhs_pinned);
    free_host_f32(rhs, rhs_pinned);
    free_host_f32(out, out_pinned);
    free(samples);
    return 0;
}
