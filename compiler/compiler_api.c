#include "pyc/compiler_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if !defined(_WIN32)
#include <pthread.h>
#endif

#include "pyc/cuda_backend.h"
#include "pyc/kernel_registry.h"
#include "pyc/pass_manager.h"
#include "pyc/runtime_allocator.h"
#include "pyc/runtime_control.h"

#define PYC_AUTOTUNE_CANDIDATE_MAX 16
#define PYC_AUTOTUNE_DB_ENTRY_MAX 512

typedef struct pyc_compiled_model {
    pyc_ir_module module;
    pyc_backend backend;
    pyc_compile_options options;
    pyc_alloc_plan alloc_plan;
    pyc_kernel_desc selected_kernel;
    pyc_kernel_selection_trace kernel_trace;
    pyc_runtime_controller controller;
    uint64_t module_fingerprint;
    int deterministic_contract_enforced;
    double baseline_p95_ms;
    double baseline_throughput;
    size_t run_count;
    int has_selected_kernel;
    pyc_cuda_dispatch_trace cuda_trace;
    float* cpu_workspace[PYC_IR_MAX_OPS];
    size_t cpu_workspace_bytes[PYC_IR_MAX_OPS];
    double compile_ms;
    int compile_cache_hit;
    int compile_budget_exceeded;
    size_t guard_miss_count;
    size_t fallback_count;
    size_t graph_break_count;
    size_t graph_break_const_count;
    size_t graph_break_gelu_count;
    size_t graph_break_reduce_sum_count;
    size_t graph_break_layernorm_count;
    size_t graph_break_unknown_count;
    int first_graph_break_op_id;
    char first_graph_break_op_name[PYC_IR_MAX_NAME];
    double compilability_score;
    int autotune_loaded;
    int autotune_saved;
    pyc_kernel_desc autotune_candidates[PYC_AUTOTUNE_CANDIDATE_MAX];
    size_t autotune_candidate_count;
    char graph_break_summary[128];
    char autotune_db_path[256];
    uint64_t compile_cache_key;
    char decision_log[512];
} pyc_compiled_model;

static uint64_t fnv1a64(const char* data, size_t len) {
    size_t i;
    uint64_t hash = 1469598103934665603ULL;
    for (i = 0; i < len; ++i) {
        hash ^= (unsigned char)data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static int module_fingerprint(const pyc_ir_module* module, uint64_t* out_hash) {
    char buffer[262144];
    size_t len;
    if (!module || !out_hash) {
        return -1;
    }
    if (pyc_ir_serialize(module, buffer, sizeof(buffer)) != 0) {
        return -1;
    }
    len = strlen(buffer);
    *out_hash = fnv1a64(buffer, len);
    return 0;
}

#define PYC_COMPILE_CACHE_MAX 32

typedef struct {
    int valid;
    uint64_t key;
    pyc_ir_module module;
    pyc_alloc_plan alloc_plan;
    pyc_kernel_desc selected_kernel;
    pyc_kernel_selection_trace kernel_trace;
    int has_selected_kernel;
    uint64_t module_fingerprint;
    size_t graph_break_count;
    size_t graph_break_const_count;
    size_t graph_break_gelu_count;
    size_t graph_break_reduce_sum_count;
    size_t graph_break_layernorm_count;
    size_t graph_break_unknown_count;
    int first_graph_break_op_id;
    char first_graph_break_op_name[PYC_IR_MAX_NAME];
    double compilability_score;
    char graph_break_summary[128];
} pyc_compile_cache_entry;

static pyc_compile_cache_entry g_compile_cache[PYC_COMPILE_CACHE_MAX];
static size_t g_compile_cache_next;

static uint64_t hash_u64(uint64_t seed, uint64_t v) {
    uint64_t combined = seed ^ v;
    return fnv1a64((const char*)&combined, sizeof(combined));
}

static uint64_t build_compile_cache_key(
    uint64_t source_fingerprint,
    pyc_backend backend,
    const pyc_compile_options* options) {
    uint64_t key = source_fingerprint;
    if (!options) {
        return key;
    }
    key = hash_u64(key, (uint64_t)backend);
    key = hash_u64(key, (uint64_t)options->enable_fusion);
    key = hash_u64(key, (uint64_t)options->enable_memory_reuse);
    key = hash_u64(key, (uint64_t)options->enable_autotune);
    key = hash_u64(key, (uint64_t)options->objective_mode);
    key = hash_u64(key, (uint64_t)options->memory_budget_bytes);
    key = hash_u64(key, (uint64_t)(options->target_utilization_floor * 1000.0));
    key = hash_u64(key, (uint64_t)options->deterministic_strict);
    key = hash_u64(key, (uint64_t)options->cache_mode);
    key = hash_u64(key, (uint64_t)(options->compile_budget_ms * 1000.0));
    if (options->autotune_db_path && options->autotune_db_path[0] != '\0') {
        key = fnv1a64(options->autotune_db_path, strlen(options->autotune_db_path)) ^ key;
    }
    return key;
}

static const pyc_compile_cache_entry* compile_cache_find(uint64_t key) {
    size_t i;
    for (i = 0; i < PYC_COMPILE_CACHE_MAX; ++i) {
        if (g_compile_cache[i].valid && g_compile_cache[i].key == key) {
            return &g_compile_cache[i];
        }
    }
    return NULL;
}

static void compile_cache_store(
    uint64_t key,
    const pyc_ir_module* module,
    const pyc_alloc_plan* plan,
    const pyc_kernel_desc* selected_kernel,
    const pyc_kernel_selection_trace* kernel_trace,
    int has_selected_kernel,
    uint64_t compiled_fingerprint,
    size_t graph_break_count,
    size_t graph_break_const_count,
    size_t graph_break_gelu_count,
    size_t graph_break_reduce_sum_count,
    size_t graph_break_layernorm_count,
    size_t graph_break_unknown_count,
    int first_graph_break_op_id,
    const char* first_graph_break_op_name,
    double compilability_score,
    const char* graph_break_summary) {
    pyc_compile_cache_entry* entry;
    if (!module || !plan || !kernel_trace) {
        return;
    }
    entry = &g_compile_cache[g_compile_cache_next % PYC_COMPILE_CACHE_MAX];
    memset(entry, 0, sizeof(*entry));
    entry->valid = 1;
    entry->key = key;
    entry->module = *module;
    entry->alloc_plan = *plan;
    if (selected_kernel) {
        entry->selected_kernel = *selected_kernel;
    }
    entry->kernel_trace = *kernel_trace;
    entry->has_selected_kernel = has_selected_kernel;
    entry->module_fingerprint = compiled_fingerprint;
    entry->graph_break_count = graph_break_count;
    entry->graph_break_const_count = graph_break_const_count;
    entry->graph_break_gelu_count = graph_break_gelu_count;
    entry->graph_break_reduce_sum_count = graph_break_reduce_sum_count;
    entry->graph_break_layernorm_count = graph_break_layernorm_count;
    entry->graph_break_unknown_count = graph_break_unknown_count;
    entry->first_graph_break_op_id = first_graph_break_op_id;
    if (first_graph_break_op_name) {
        strncpy(
            entry->first_graph_break_op_name,
            first_graph_break_op_name,
            sizeof(entry->first_graph_break_op_name) - 1);
        entry->first_graph_break_op_name[sizeof(entry->first_graph_break_op_name) - 1] = '\0';
    }
    entry->compilability_score = compilability_score;
    if (graph_break_summary) {
        strncpy(entry->graph_break_summary, graph_break_summary, sizeof(entry->graph_break_summary) - 1);
        entry->graph_break_summary[sizeof(entry->graph_break_summary) - 1] = '\0';
    }
    g_compile_cache_next++;
}

static double elapsed_ms(clock_t start, clock_t end) {
    return ((double)(end - start) * 1000.0) / (double)CLOCKS_PER_SEC;
}

static void spin_sleep_ms(int delay_ms) {
    clock_t start;
    if (delay_ms <= 0) {
        return;
    }
    start = clock();
    while (elapsed_ms(start, clock()) < (double)delay_ms) {
    }
}

static FILE* fopen_with_retries(const char* path, const char* mode) {
    int attempt;
    FILE* f = NULL;
    for (attempt = 0; attempt < 8; ++attempt) {
        f = fopen(path, mode);
        if (f) {
            return f;
        }
        spin_sleep_ms(2);
    }
    return NULL;
}

static void maybe_inject_compile_delay(void) {
    const char* env_delay = getenv("PYC_COMPILE_DELAY_MS");
    long delay_ms = 0;
    clock_t start;
    if (!env_delay || env_delay[0] == '\0') {
        return;
    }
    delay_ms = strtol(env_delay, NULL, 10);
    if (delay_ms <= 0) {
        return;
    }
    start = clock();
    while (elapsed_ms(start, clock()) < (double)delay_ms) {
    }
}

static double autotune_estimate_candidate_ms(
    double baseline_ms,
    const pyc_kernel_desc* candidate,
    pyc_objective_mode mode,
    double pressure_score) {
    double adjusted = baseline_ms;
    double occupancy_bonus;
    double pressure_penalty;
    double mode_bias;
    if (!candidate) {
        return baseline_ms;
    }
    occupancy_bonus = (1.0 - candidate->estimated_occupancy) * 0.08;
    pressure_penalty = (pressure_score > 0.0)
        ? (((double)candidate->reg_pressure_class * pressure_score) * 0.005)
        : 0.0;
    mode_bias = mode == PYC_MODE_MEMORY_FIRST
        ? ((double)candidate->shared_mem_bytes / (128.0 * 1024.0))
        : 0.0;
    adjusted = baseline_ms * (1.0 + occupancy_bonus + pressure_penalty + mode_bias);
    if (candidate->tensor_core_eligible) {
        adjusted *= 0.97;
    }
    if (adjusted <= 0.0) {
        adjusted = 0.001;
    }
    return adjusted;
}

static pyc_compile_options default_compile_options(void) {
    pyc_compile_options o;
    memset(&o, 0, sizeof(o));
    o.enable_fusion = 1;
    o.enable_memory_reuse = 1;
    o.enable_autotune = 0;
    o.objective_mode = PYC_MODE_BALANCED;
    o.memory_budget_bytes = 0;
    o.target_utilization_floor = 0.70;
    o.deterministic_strict = 1;
    o.compile_budget_ms = 0.0;
    o.cache_mode = PYC_COMPILE_CACHE_IN_MEMORY;
    o.autotune_db_path = NULL;
    pyc_runtime_rails_default(&o.rails);
    return o;
}

static void sanitize_runtime_rails(pyc_runtime_rails* rails) {
    pyc_runtime_rails defaults;
    if (!rails) {
        return;
    }
    pyc_runtime_rails_default(&defaults);

    if (!(rails->enable_auto_switch == 0 || rails->enable_auto_switch == 1)) {
        rails->enable_auto_switch = defaults.enable_auto_switch;
    }
    if (!(rails->enable_hard_rollback == 0 || rails->enable_hard_rollback == 1)) {
        rails->enable_hard_rollback = defaults.enable_hard_rollback;
    }
    if (rails->dwell_steps == 0 || rails->dwell_steps > 100000) {
        rails->dwell_steps = defaults.dwell_steps;
    }
    if (rails->cooldown_steps > 100000) {
        rails->cooldown_steps = defaults.cooldown_steps;
    }
    if (rails->consecutive_breach_windows == 0 || rails->consecutive_breach_windows > 1000) {
        rails->consecutive_breach_windows = defaults.consecutive_breach_windows;
    }
    if (rails->latency_regression_threshold <= 0.0 || rails->latency_regression_threshold > 1.0) {
        rails->latency_regression_threshold = defaults.latency_regression_threshold;
    }
    if (rails->throughput_regression_threshold <= 0.0 || rails->throughput_regression_threshold > 1.0) {
        rails->throughput_regression_threshold = defaults.throughput_regression_threshold;
    }
    if (rails->pressure_score_threshold <= 0.0 || rails->pressure_score_threshold > 10.0) {
        rails->pressure_score_threshold = defaults.pressure_score_threshold;
    }
    if (rails->pressure_events_threshold == 0 || rails->pressure_events_threshold > 100000) {
        rails->pressure_events_threshold = defaults.pressure_events_threshold;
    }
    if (rails->rematerialized_tensors_threshold == 0 || rails->rematerialized_tensors_threshold > 100000) {
        rails->rematerialized_tensors_threshold = defaults.rematerialized_tensors_threshold;
    }
}

static size_t shape_num_elements(const pyc_shape* shape) {
    size_t n = 1;
    size_t d;
    if (!shape || shape->rank == 0) {
        return 0;
    }
    for (d = 0; d < shape->rank; ++d) {
        if (shape->dims[d] <= 0) {
            return 0;
        }
        n *= (size_t)shape->dims[d];
    }
    return n;
}

static int op_uses_workspace(pyc_ir_op_kind kind) {
    return kind == PYC_IR_OP_MATMUL ||
           kind == PYC_IR_OP_ADD ||
           kind == PYC_IR_OP_RELU;
}

static void release_cpu_workspace(pyc_compiled_model* model) {
    size_t i;
    if (!model) {
        return;
    }
    for (i = 0; i < PYC_IR_MAX_OPS; ++i) {
        free(model->cpu_workspace[i]);
        model->cpu_workspace[i] = NULL;
        model->cpu_workspace_bytes[i] = 0;
    }
}

static int init_cpu_workspace(pyc_compiled_model* model) {
    size_t i;
    if (!model) {
        return -1;
    }
    for (i = 0; i < model->module.op_count; ++i) {
        const pyc_ir_op* op = &model->module.ops[i];
        size_t elems;
        size_t bytes;
        if (!op_uses_workspace(op->kind)) {
            continue;
        }
        elems = shape_num_elements(&op->shape);
        if (elems == 0) {
            continue;
        }
        bytes = elems * sizeof(float);
        model->cpu_workspace[i] = (float*)malloc(bytes);
        if (!model->cpu_workspace[i]) {
            release_cpu_workspace(model);
            return -1;
        }
        model->cpu_workspace_bytes[i] = bytes;
    }
    return 0;
}

typedef struct {
    const float* a;
    const float* b;
    float* c;
    size_t k;
    size_t n;
    size_t row_begin;
    size_t row_end;
} matmul_job;

static void matmul_compute_rows(const matmul_job* job) {
    const size_t k_block = 64;
    size_t r;
    size_t kk0;
    for (r = job->row_begin; r < job->row_end; ++r) {
        const float* a_row = job->a + (r * job->k);
        float* c_row = job->c + (r * job->n);
        memset(c_row, 0, job->n * sizeof(float));
        for (kk0 = 0; kk0 < job->k; kk0 += k_block) {
            size_t kk_end = kk0 + k_block;
            size_t kk;
            if (kk_end > job->k) {
                kk_end = job->k;
            }
            for (kk = kk0; kk < kk_end; ++kk) {
                float av = a_row[kk];
                const float* b_row = job->b + (kk * job->n);
                size_t ccol = 0;
                for (; ccol + 7 < job->n; ccol += 8) {
                    c_row[ccol] += av * b_row[ccol];
                    c_row[ccol + 1] += av * b_row[ccol + 1];
                    c_row[ccol + 2] += av * b_row[ccol + 2];
                    c_row[ccol + 3] += av * b_row[ccol + 3];
                    c_row[ccol + 4] += av * b_row[ccol + 4];
                    c_row[ccol + 5] += av * b_row[ccol + 5];
                    c_row[ccol + 6] += av * b_row[ccol + 6];
                    c_row[ccol + 7] += av * b_row[ccol + 7];
                }
                for (; ccol < job->n; ++ccol) {
                    c_row[ccol] += av * b_row[ccol];
                }
            }
        }
    }
}

#if !defined(_WIN32)
static void* matmul_worker(void* arg) {
    matmul_compute_rows((const matmul_job*)arg);
    return NULL;
}
#endif

static size_t matmul_thread_count(size_t m) {
    const size_t max_threads = 32;
    long configured_threads = 0;
    size_t thread_count = 1;
    const char* env_threads = getenv("PYC_NUM_THREADS");

    if (env_threads && env_threads[0] != '\0') {
        configured_threads = strtol(env_threads, NULL, 10);
        if (configured_threads > 0) {
            thread_count = (size_t)configured_threads;
        }
    }
    if (thread_count > max_threads) {
        thread_count = max_threads;
    }
    if (thread_count > m) {
        thread_count = m;
    }
    if (m < 8) {
        thread_count = 1;
    }
    return thread_count;
}

static void matmul_tiled_f32(
    const float* a,
    const float* b,
    float* c,
    size_t m,
    size_t k,
    size_t n) {
    size_t thread_count = matmul_thread_count(m);

#if !defined(_WIN32)
    if (thread_count > 1) {
        pthread_t threads[32];
        matmul_job jobs[32];
        size_t t;
        size_t rows_per_thread = (m + thread_count - 1) / thread_count;
        int spawn_failed = 0;

        for (t = 0; t < thread_count; ++t) {
            size_t begin = t * rows_per_thread;
            size_t end = begin + rows_per_thread;
            if (begin >= m) {
                jobs[t].row_begin = m;
                jobs[t].row_end = m;
                continue;
            }
            if (end > m) {
                end = m;
            }
            jobs[t].a = a;
            jobs[t].b = b;
            jobs[t].c = c;
            jobs[t].k = k;
            jobs[t].n = n;
            jobs[t].row_begin = begin;
            jobs[t].row_end = end;
            if (pthread_create(&threads[t], NULL, matmul_worker, &jobs[t]) != 0) {
                spawn_failed = 1;
                thread_count = t;
                break;
            }
        }
        for (t = 0; t < thread_count; ++t) {
            pthread_join(threads[t], NULL);
        }
        if (!spawn_failed) {
            return;
        }
    }
#endif

    {
        matmul_job single_job;
        single_job.a = a;
        single_job.b = b;
        single_job.c = c;
        single_job.k = k;
        single_job.n = n;
        single_job.row_begin = 0;
        single_job.row_end = m;
        matmul_compute_rows(&single_job);
    }
}

static int execute_cpu_graph(
    const pyc_ir_module* module,
    const pyc_tensor* inputs,
    size_t input_count,
    pyc_tensor* outputs,
    size_t output_count,
    void* executor_ctx) {
    pyc_compiled_model* owner = (pyc_compiled_model*)executor_ctx;
    const float* op_read[PYC_IR_MAX_OPS];
    float* op_owned[PYC_IR_MAX_OPS];
    size_t op_elems[PYC_IR_MAX_OPS];
    size_t input_index = 0;
    size_t output_index = 0;
    size_t i;
    int fail = 0;

    memset(op_read, 0, sizeof(op_read));
    memset(op_owned, 0, sizeof(op_owned));
    memset(op_elems, 0, sizeof(op_elems));

    for (i = 0; i < module->op_count; ++i) {
        const pyc_ir_op* op = &module->ops[i];
        op_elems[i] = shape_num_elements(&op->shape);

        switch (op->kind) {
            case PYC_IR_OP_INPUT: {
                size_t required;
                if (input_index >= input_count || !inputs[input_index].data) {
                    fail = 1;
                    goto cleanup;
                }
                if (inputs[input_index].dtype != PYC_DTYPE_F32) {
                    fail = 1;
                    goto cleanup;
                }
                required = op_elems[i] * sizeof(float);
                if (required == 0 || inputs[input_index].size_bytes < required) {
                    fail = 1;
                    goto cleanup;
                }
                op_read[i] = (const float*)inputs[input_index].data;
                input_index++;
                break;
            }
            case PYC_IR_OP_MATMUL: {
                const pyc_ir_op* a_op;
                const pyc_ir_op* b_op;
                const float* a;
                const float* b;
                float* out;
                size_t m, k, n;
                size_t required;
                if (op->input_count < 2) {
                    fail = 1;
                    goto cleanup;
                }
                if (op->input_ids[0] < 0 || op->input_ids[1] < 0 ||
                    (size_t)op->input_ids[0] >= module->op_count ||
                    (size_t)op->input_ids[1] >= module->op_count) {
                    fail = 1;
                    goto cleanup;
                }
                a_op = &module->ops[(size_t)op->input_ids[0]];
                b_op = &module->ops[(size_t)op->input_ids[1]];
                if (a_op->shape.rank != 2 || b_op->shape.rank != 2) {
                    fail = 1;
                    goto cleanup;
                }
                m = (size_t)a_op->shape.dims[0];
                k = (size_t)a_op->shape.dims[1];
                if (k != (size_t)b_op->shape.dims[0]) {
                    fail = 1;
                    goto cleanup;
                }
                n = (size_t)b_op->shape.dims[1];
                a = op_read[(size_t)op->input_ids[0]];
                b = op_read[(size_t)op->input_ids[1]];
                if (!a || !b) {
                    fail = 1;
                    goto cleanup;
                }
                required = m * n * sizeof(float);
                out = NULL;
                if (owner && owner->cpu_workspace[i] && owner->cpu_workspace_bytes[i] >= required) {
                    out = owner->cpu_workspace[i];
                } else {
                    out = (float*)malloc(required);
                    if (!out) {
                        fail = 1;
                        goto cleanup;
                    }
                    op_owned[i] = out;
                }
                matmul_tiled_f32(a, b, out, m, k, n);
                op_read[i] = out;
                break;
            }
            case PYC_IR_OP_ADD: {
                const float* a;
                const float* b;
                float* out;
                size_t n;
                size_t j;
                size_t required;
                if (op->input_count < 2) {
                    fail = 1;
                    goto cleanup;
                }
                if (op->input_ids[0] < 0 || op->input_ids[1] < 0 ||
                    (size_t)op->input_ids[0] >= module->op_count ||
                    (size_t)op->input_ids[1] >= module->op_count) {
                    fail = 1;
                    goto cleanup;
                }
                n = op_elems[i];
                a = op_read[(size_t)op->input_ids[0]];
                b = op_read[(size_t)op->input_ids[1]];
                if (!a || !b || n == 0) {
                    fail = 1;
                    goto cleanup;
                }
                required = n * sizeof(float);
                out = NULL;
                if (owner && owner->cpu_workspace[i] && owner->cpu_workspace_bytes[i] >= required) {
                    out = owner->cpu_workspace[i];
                } else {
                    out = (float*)malloc(required);
                    if (!out) {
                        fail = 1;
                        goto cleanup;
                    }
                    op_owned[i] = out;
                }
                for (j = 0; j < n; ++j) {
                    out[j] = a[j] + b[j];
                }
                op_read[i] = out;
                break;
            }
            case PYC_IR_OP_RELU: {
                const float* a;
                float* out;
                size_t n;
                size_t j;
                size_t required;
                if (op->input_count < 1) {
                    fail = 1;
                    goto cleanup;
                }
                if (op->input_ids[0] < 0 || (size_t)op->input_ids[0] >= module->op_count) {
                    fail = 1;
                    goto cleanup;
                }
                n = op_elems[i];
                a = op_read[(size_t)op->input_ids[0]];
                if (!a || n == 0) {
                    fail = 1;
                    goto cleanup;
                }
                required = n * sizeof(float);
                out = NULL;
                if (owner && owner->cpu_workspace[i] && owner->cpu_workspace_bytes[i] >= required) {
                    out = owner->cpu_workspace[i];
                } else {
                    out = (float*)malloc(required);
                    if (!out) {
                        fail = 1;
                        goto cleanup;
                    }
                    op_owned[i] = out;
                }
                for (j = 0; j < n; ++j) {
                    out[j] = a[j] > 0.0f ? a[j] : 0.0f;
                }
                op_read[i] = out;
                break;
            }
            case PYC_IR_OP_OUTPUT: {
                const float* src;
                size_t required;
                int src_id;
                if (output_index >= output_count || !outputs[output_index].data || op->input_count < 1) {
                    fail = 1;
                    goto cleanup;
                }
                src_id = op->input_ids[0];
                if (src_id < 0 || (size_t)src_id >= module->op_count) {
                    fail = 1;
                    goto cleanup;
                }
                src = op_read[(size_t)src_id];
                required = op_elems[(size_t)src_id] * sizeof(float);
                if (!src || required == 0 || outputs[output_index].size_bytes < required) {
                    fail = 1;
                    goto cleanup;
                }
                memcpy(outputs[output_index].data, src, required);
                output_index++;
                break;
            }
            case PYC_IR_OP_CONST:
            case PYC_IR_OP_GELU:
            case PYC_IR_OP_REDUCE_SUM:
            case PYC_IR_OP_LAYERNORM:
            default:
                fail = 1;
                goto cleanup;
        }
    }

cleanup:
    for (i = 0; i < module->op_count; ++i) {
        free(op_owned[i]);
    }
    return fail ? -1 : 0;
}

static void ensure_default_kernel_catalog(pyc_backend backend) {
    pyc_kernel_desc probe;
    pyc_kernel_desc k1;
    pyc_kernel_desc k2;
    if (pyc_kernel_select("matmul_fused", backend, &probe) == 0) {
        return;
    }
    memset(&k1, 0, sizeof(k1));
    strcpy(k1.op_key, "matmul_fused");
    k1.backend = backend;
    strcpy(k1.symbol, backend == PYC_BACKEND_CUDA ? "matmul_cuda_ref" : "matmul_cpu_ref");
    k1.priority = 1;
    k1.estimated_occupancy = 0.55;
    k1.tensor_core_eligible = backend == PYC_BACKEND_CUDA ? 1 : 0;
    k1.shared_mem_bytes = 8 * 1024;
    k1.reg_pressure_class = 2;
    (void)pyc_kernel_register(&k1);

    memset(&k2, 0, sizeof(k2));
    strcpy(k2.op_key, "matmul_fused");
    k2.backend = backend;
    strcpy(k2.symbol, backend == PYC_BACKEND_CUDA ? "matmul_cuda_tuned" : "matmul_cpu_tuned");
    k2.priority = 2;
    k2.estimated_occupancy = 0.75;
    k2.tensor_core_eligible = backend == PYC_BACKEND_CUDA ? 1 : 0;
    k2.shared_mem_bytes = 16 * 1024;
    k2.reg_pressure_class = 3;
    (void)pyc_kernel_register(&k2);
}

static void resolve_autotune_db_path(
    const pyc_compile_options* options,
    char* out_path,
    size_t out_path_size) {
    const char* env_path = getenv("PYC_AUTOTUNE_DB_PATH");
    const char* selected = NULL;
    if (!out_path || out_path_size == 0) {
        return;
    }
    out_path[0] = '\0';
    if (options && options->autotune_db_path && options->autotune_db_path[0] != '\0') {
        selected = options->autotune_db_path;
    } else if (env_path && env_path[0] != '\0') {
        selected = env_path;
    } else {
        selected = "benchmark/benchmarks/results/json/pyc_autotune.db";
    }
    strncpy(out_path, selected, out_path_size - 1);
    out_path[out_path_size - 1] = '\0';
}

static int autotune_load_into_registry(
    const char* path,
    const char* op_key,
    pyc_backend backend) {
    FILE* f;
    char line[512];
    int loaded = 0;
    if (!path || !op_key || path[0] == '\0') {
        return 0;
    }
    f = fopen(path, "r");
    if (!f) {
        return 0;
    }
    while (fgets(line, sizeof(line), f)) {
        char file_op[PYC_KERNEL_OP_KEY_MAX];
        char file_symbol[PYC_KERNEL_SYMBOL_MAX];
        int file_backend = -1;
        double best_ms = 0.0;
        if (sscanf(
                line,
                "%63[^|]|%d|%127[^|]|%lf",
                file_op,
                &file_backend,
                file_symbol,
                &best_ms) != 4) {
            continue;
        }
        if (strcmp(file_op, op_key) != 0 || file_backend != (int)backend) {
            continue;
        }
        if (best_ms <= 0.0) {
            continue;
        }
        if (pyc_kernel_benchmark_update_symbol(op_key, backend, file_symbol, best_ms) == 0) {
            loaded = 1;
        }
    }
    fclose(f);
    return loaded;
}

typedef struct {
    char op_key[PYC_KERNEL_OP_KEY_MAX];
    int backend;
    char symbol[PYC_KERNEL_SYMBOL_MAX];
    double best_ms;
} pyc_autotune_db_entry;

static int parse_autotune_line(const char* line, pyc_autotune_db_entry* out) {
    if (!line || !out) {
        return -1;
    }
    if (sscanf(
            line,
            "%63[^|]|%d|%127[^|]|%lf",
            out->op_key,
            &out->backend,
            out->symbol,
            &out->best_ms) != 4) {
        return -1;
    }
    if (out->best_ms <= 0.0) {
        return -1;
    }
    return 0;
}

static int autotune_compact_db(const char* path) {
    FILE* in;
    FILE* out;
    char line[512];
    pyc_autotune_db_entry entries[PYC_AUTOTUNE_DB_ENTRY_MAX];
    size_t entry_count = 0;
    size_t i;

    if (!path || path[0] == '\0') {
        return -1;
    }

    in = fopen_with_retries(path, "r");
    if (!in) {
        return -1;
    }

    while (fgets(line, sizeof(line), in)) {
        pyc_autotune_db_entry parsed;
        int found = 0;
        if (parse_autotune_line(line, &parsed) != 0) {
            continue;
        }
        for (i = 0; i < entry_count; ++i) {
            if (strcmp(entries[i].op_key, parsed.op_key) == 0 &&
                entries[i].backend == parsed.backend &&
                strcmp(entries[i].symbol, parsed.symbol) == 0) {
                if (parsed.best_ms < entries[i].best_ms) {
                    entries[i].best_ms = parsed.best_ms;
                }
                found = 1;
                break;
            }
        }
        if (!found && entry_count < PYC_AUTOTUNE_DB_ENTRY_MAX) {
            entries[entry_count++] = parsed;
        }
    }
    fclose(in);

    if (entry_count == 0) {
        return 0;
    }

    out = fopen_with_retries(path, "w");
    if (!out) {
        return -1;
    }

    for (i = 0; i < entry_count; ++i) {
        if (fprintf(
            out,
            "%s|%d|%s|%.6f\n",
            entries[i].op_key,
            entries[i].backend,
            entries[i].symbol,
            entries[i].best_ms) < 0) {
            fclose(out);
            return -1;
        }
    }
    if (fclose(out) != 0) {
        return -1;
    }
    return 0;
}

static int autotune_persist_result(
    const char* path,
    const char* op_key,
    pyc_backend backend,
    const char* symbol,
    double best_ms) {
    FILE* f;
    if (!path || !op_key || !symbol || best_ms <= 0.0) {
        return -1;
    }
    f = fopen_with_retries(path, "a");
    if (!f) {
        return -1;
    }
    if (fprintf(f, "%s|%d|%s|%.6f\n", op_key, (int)backend, symbol, best_ms) < 0) {
        fclose(f);
        return -1;
    }
    if (fclose(f) != 0) {
        return -1;
    }
    return autotune_compact_db(path);
}

const char* pyc_status_string(pyc_status status) {
    switch (status) {
        case PYC_STATUS_OK: return "OK";
        case PYC_STATUS_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case PYC_STATUS_VERIFY_FAILED: return "VERIFY_FAILED";
        case PYC_STATUS_COMPILE_FAILED: return "COMPILE_FAILED";
        case PYC_STATUS_RUNTIME_FAILED: return "RUNTIME_FAILED";
        default: return "UNKNOWN";
    }
}

pyc_status pyc_compile_model(const pyc_model_desc* desc, const pyc_compile_options* options, pyc_compiled_model** out_model) {
    pyc_ir_diagnostic diag;
    pyc_pass_pipeline pipeline;
    pyc_pass_report report;
    pyc_compiled_model* model;
    const pyc_compile_cache_entry* cache_entry = NULL;
    clock_t start;
    clock_t end;
    uint64_t source_fingerprint = 0;
    uint64_t cache_key = 0;
    int use_compile_cache = 0;

    if (!desc || !desc->module || !out_model) {
        return PYC_STATUS_INVALID_ARGUMENT;
    }

    if (pyc_ir_verify(desc->module, &diag) != 0) {
        (void)diag;
        return PYC_STATUS_VERIFY_FAILED;
    }

    model = (pyc_compiled_model*)malloc(sizeof(*model));
    if (!model) {
        return PYC_STATUS_COMPILE_FAILED;
    }
    memset(model, 0, sizeof(*model));

    model->module = *desc->module;
    model->backend = desc->backend;
    model->options = default_compile_options();
    if (options) {
        model->options.enable_fusion = options->enable_fusion;
        model->options.enable_memory_reuse = options->enable_memory_reuse;
        model->options.enable_autotune = options->enable_autotune;
        model->options.objective_mode = options->objective_mode;
        model->options.memory_budget_bytes = options->memory_budget_bytes;
        model->options.target_utilization_floor = options->target_utilization_floor;
        model->options.deterministic_strict = options->deterministic_strict;
        model->options.compile_budget_ms = options->compile_budget_ms;
        model->options.cache_mode = options->cache_mode;
        model->options.autotune_db_path = options->autotune_db_path;
        model->options.rails = options->rails;
    }
    if (model->options.cache_mode != PYC_COMPILE_CACHE_DISABLED &&
        model->options.cache_mode != PYC_COMPILE_CACHE_IN_MEMORY) {
        model->options.cache_mode = PYC_COMPILE_CACHE_IN_MEMORY;
    }
    if (model->options.compile_budget_ms < 0.0) {
        model->options.compile_budget_ms = 0.0;
    }
    if (model->options.deterministic_strict) {
        model->options.rails.enable_auto_switch = 0;
        model->options.rails.enable_hard_rollback = 0;
        model->deterministic_contract_enforced = 1;
    }
    sanitize_runtime_rails(&model->options.rails);
    if (model->options.objective_mode < PYC_MODE_BALANCED ||
        model->options.objective_mode > PYC_MODE_UTILIZATION_FIRST) {
        model->options.objective_mode = PYC_MODE_BALANCED;
    }
    resolve_autotune_db_path(&model->options, model->autotune_db_path, sizeof(model->autotune_db_path));

    pyc_alloc_plan_init(&model->alloc_plan);
    pyc_runtime_controller_init(&model->controller, model->options.objective_mode);
    pyc_cuda_dispatch_trace_init(&model->cuda_trace);
    if (module_fingerprint(desc->module, &source_fingerprint) != 0) {
        free(model);
        return PYC_STATUS_COMPILE_FAILED;
    }
    cache_key = build_compile_cache_key(source_fingerprint, model->backend, &model->options);
    model->compile_cache_key = cache_key;
    use_compile_cache = model->options.cache_mode == PYC_COMPILE_CACHE_IN_MEMORY;

    start = clock();

    if (use_compile_cache) {
        cache_entry = compile_cache_find(cache_key);
    }

    if (cache_entry) {
        model->module = cache_entry->module;
        model->alloc_plan = cache_entry->alloc_plan;
        model->selected_kernel = cache_entry->selected_kernel;
        model->kernel_trace = cache_entry->kernel_trace;
        model->has_selected_kernel = cache_entry->has_selected_kernel;
        model->module_fingerprint = cache_entry->module_fingerprint;
        model->graph_break_count = cache_entry->graph_break_count;
        model->graph_break_const_count = cache_entry->graph_break_const_count;
        model->graph_break_gelu_count = cache_entry->graph_break_gelu_count;
        model->graph_break_reduce_sum_count = cache_entry->graph_break_reduce_sum_count;
        model->graph_break_layernorm_count = cache_entry->graph_break_layernorm_count;
        model->graph_break_unknown_count = cache_entry->graph_break_unknown_count;
        model->first_graph_break_op_id = cache_entry->first_graph_break_op_id;
        strncpy(
            model->first_graph_break_op_name,
            cache_entry->first_graph_break_op_name,
            sizeof(model->first_graph_break_op_name) - 1);
        model->first_graph_break_op_name[sizeof(model->first_graph_break_op_name) - 1] = '\0';
        model->compilability_score = cache_entry->compilability_score;
        strncpy(model->graph_break_summary, cache_entry->graph_break_summary, sizeof(model->graph_break_summary) - 1);
        model->graph_break_summary[sizeof(model->graph_break_summary) - 1] = '\0';
        model->compile_cache_hit = 1;
    } else {
        maybe_inject_compile_delay();
        pyc_pass_pipeline_default(&pipeline);
        if (!model->options.enable_fusion || model->backend == PYC_BACKEND_CUDA) {
            /* CUDA runtime executes explicit op chains; disabling fusion here avoids
               dropping add/activation semantics until fused-op lowering is explicit. */
            pipeline.config.fusion = 0;
        }

        if (pyc_pass_pipeline_run(&pipeline, &model->module, &report) != 0 || report.errors) {
            free(model);
            return PYC_STATUS_COMPILE_FAILED;
        }
        model->graph_break_count = report.graph_break_count;
        model->graph_break_const_count = report.graph_break_const_count;
        model->graph_break_gelu_count = report.graph_break_gelu_count;
        model->graph_break_reduce_sum_count = report.graph_break_reduce_sum_count;
        model->graph_break_layernorm_count = report.graph_break_layernorm_count;
        model->graph_break_unknown_count = report.graph_break_unknown_count;
        model->first_graph_break_op_id = report.first_graph_break_op_id;
        strncpy(
            model->first_graph_break_op_name,
            report.first_graph_break_op_name,
            sizeof(model->first_graph_break_op_name) - 1);
        model->first_graph_break_op_name[sizeof(model->first_graph_break_op_name) - 1] = '\0';
        model->compilability_score = report.compilability_score;
        strncpy(model->graph_break_summary, report.graph_break_summary, sizeof(model->graph_break_summary) - 1);
        model->graph_break_summary[sizeof(model->graph_break_summary) - 1] = '\0';
        if (module_fingerprint(&model->module, &model->module_fingerprint) != 0) {
            free(model);
            return PYC_STATUS_COMPILE_FAILED;
        }

        {
            size_t i;
            for (i = 0; i < model->module.op_count; ++i) {
                pyc_alloc_request req;
                size_t bytes = 4;
                size_t d;
                const pyc_ir_op* op = &model->module.ops[i];

                if (op->shape.rank == 0) {
                    continue;
                }

                for (d = 0; d < op->shape.rank; ++d) {
                    bytes *= (size_t)op->shape.dims[d];
                }

                req.tensor_id = (int)i;
                req.size_bytes = bytes;
                req.alignment = 64;
                req.start_step = (int)i;
                req.end_step = (int)(i + 2);
                pyc_alloc_plan_add_request(&model->alloc_plan, req);
            }

            if (model->options.enable_memory_reuse) {
                if (pyc_alloc_plan_build_with_mode(
                        &model->alloc_plan,
                        model->options.objective_mode,
                        model->options.memory_budget_bytes) != 0) {
                    free(model);
                    return PYC_STATUS_COMPILE_FAILED;
                }
            }
        }

        {
            pyc_kernel_desc selected;
            double pressure_score = model->alloc_plan.pressure_score;
            ensure_default_kernel_catalog(model->backend);
            if (model->options.enable_autotune) {
                model->autotune_loaded = autotune_load_into_registry(
                    model->autotune_db_path,
                    "matmul_fused",
                    model->backend);
            }
            if (pyc_kernel_select_with_policy(
                    "matmul_fused",
                    model->backend,
                    model->options.objective_mode,
                    pressure_score,
                    &selected,
                    &model->kernel_trace) == 0) {
                model->selected_kernel = selected;
                model->has_selected_kernel = 1;
            }
        }

        if (use_compile_cache) {
            compile_cache_store(
                cache_key,
                &model->module,
                &model->alloc_plan,
                model->has_selected_kernel ? &model->selected_kernel : NULL,
                &model->kernel_trace,
                model->has_selected_kernel,
                model->module_fingerprint,
                model->graph_break_count,
                model->graph_break_const_count,
                model->graph_break_gelu_count,
                model->graph_break_reduce_sum_count,
                model->graph_break_layernorm_count,
                model->graph_break_unknown_count,
                model->first_graph_break_op_id,
                model->first_graph_break_op_name,
                model->compilability_score,
                model->graph_break_summary);
        }
    }

    model->autotune_candidate_count = pyc_kernel_collect(
        "matmul_fused",
        model->backend,
        model->autotune_candidates,
        PYC_AUTOTUNE_CANDIDATE_MAX);

    if (init_cpu_workspace(model) != 0) {
        free(model);
        return PYC_STATUS_COMPILE_FAILED;
    }

    end = clock();
    model->compile_ms = elapsed_ms(start, end);
    if (model->options.compile_budget_ms > 0.0 &&
        model->compile_ms > model->options.compile_budget_ms) {
        model->compile_budget_exceeded = 1;
    }

    snprintf(
        model->decision_log,
        sizeof(model->decision_log),
        "mode=%d budget=%zu pressure=%.6f kernel=%s score=%.3f util=%.3f det=%d cache_hit=%d compile_ms=%.3f budget_ms=%.3f budget_exceeded=%d graph_breaks=%zu break_first=%s@%d break_counts=%zu,%zu,%zu,%zu,%zu compilability=%.3f autotune_loaded=%d",
        (int)model->options.objective_mode,
        model->options.memory_budget_bytes,
        model->alloc_plan.pressure_score,
        model->has_selected_kernel ? model->selected_kernel.symbol : "none",
        model->kernel_trace.selected_score,
        model->kernel_trace.selected_estimated_utilization,
        model->options.deterministic_strict ? 1 : 0,
        model->compile_cache_hit,
        model->compile_ms,
        model->options.compile_budget_ms,
        model->compile_budget_exceeded,
        model->graph_break_count,
        model->first_graph_break_op_name,
        model->first_graph_break_op_id,
        model->graph_break_const_count,
        model->graph_break_gelu_count,
        model->graph_break_reduce_sum_count,
        model->graph_break_layernorm_count,
        model->graph_break_unknown_count,
        model->compilability_score,
        model->autotune_loaded);

    *out_model = model;
    return PYC_STATUS_OK;
}

pyc_status pyc_run_model(pyc_compiled_model* model, const pyc_tensor* inputs, size_t input_count, pyc_tensor* outputs, size_t output_count, pyc_run_stats* out_stats) {
    clock_t start;
    clock_t end;
    clock_t stage_start;
    clock_t stage_end;
    pyc_alloc_stats stats;
    pyc_runtime_window_metrics metrics;
    pyc_objective_mode active_mode;
    pyc_rollback_reason rollback_reason;
    double run_ms;
    double throughput;
    double dispatch_ms = 0.0;
    double graph_exec_ms = 0.0;
    double controller_ms = 0.0;
    double kernel_select_ms = 0.0;
    int contract_ok = 1;
    uint64_t run_fingerprint = 0;
    char contract_reason[64];
    int runtime_error = 0;
    size_t guard_miss_count_this_run = 0;
    size_t fallback_count_this_run = 0;
    double autotune_ms = 0.0;

    if (!model || !inputs || !outputs || input_count == 0 || output_count == 0) {
        return PYC_STATUS_INVALID_ARGUMENT;
    }

    start = clock();
    memset(contract_reason, 0, sizeof(contract_reason));
    strcpy(contract_reason, "ok");

    if (model->deterministic_contract_enforced) {
        if (module_fingerprint(&model->module, &run_fingerprint) != 0) {
            contract_ok = 0;
            runtime_error = 1;
            guard_miss_count_this_run++;
            strcpy(contract_reason, "fingerprint_unavailable");
        } else if (run_fingerprint != model->module_fingerprint) {
            contract_ok = 0;
            runtime_error = 1;
            guard_miss_count_this_run++;
            strcpy(contract_reason, "fingerprint_mismatch");
        }
    }

    pyc_cuda_dispatch_trace_init(&model->cuda_trace);
    stage_start = clock();
    if (!runtime_error) {
        if (model->backend == PYC_BACKEND_CPU) {
            if (execute_cpu_graph(&model->module, inputs, input_count, outputs, output_count, model) != 0) {
                runtime_error = 1;
            }
        } else if (model->backend == PYC_BACKEND_CUDA) {
            pyc_cuda_dispatch_status cuda_status = pyc_cuda_dispatch(
                &model->module,
                inputs,
                input_count,
                outputs,
                output_count,
                execute_cpu_graph,
                model,
                &model->cuda_trace);
            if (cuda_status == PYC_CUDA_DISPATCH_ERROR) {
                runtime_error = 1;
            } else if (cuda_status == PYC_CUDA_DISPATCH_FALLBACK) {
                fallback_count_this_run++;
            }
        } else {
            runtime_error = 1;
        }
    }
    stage_end = clock();
    dispatch_ms = elapsed_ms(stage_start, stage_end);
    graph_exec_ms = dispatch_ms;

    end = clock();
    run_ms = elapsed_ms(start, end);
    throughput = run_ms > 0.0 ? 1000.0 / run_ms : 0.0;
    model->run_count++;

    pyc_alloc_plan_stats(&model->alloc_plan, &stats);
    if (model->baseline_p95_ms == 0.0) {
        model->baseline_p95_ms = run_ms;
    }
    if (model->baseline_throughput == 0.0) {
        model->baseline_throughput = throughput;
    }

    memset(&metrics, 0, sizeof(metrics));
    metrics.baseline_p95_ms = model->baseline_p95_ms;
    metrics.observed_p95_ms = run_ms;
    metrics.baseline_throughput = model->baseline_throughput;
    metrics.observed_throughput = throughput;
    metrics.pressure_score = stats.pressure_score;
    metrics.pressure_events = stats.pressure_events;
    metrics.rematerialized_tensors = stats.rematerialized_tensors;
    metrics.runtime_error = runtime_error;

    stage_start = clock();
    if (pyc_runtime_controller_observe(
        &model->controller,
        &model->options.rails,
        &metrics,
        &active_mode,
        &rollback_reason) != 0) {
        return PYC_STATUS_RUNTIME_FAILED;
    }
    stage_end = clock();
    controller_ms = elapsed_ms(stage_start, stage_end);

    model->options.objective_mode = active_mode;
    model->kernel_trace.selected_score = 0.0;
    model->kernel_trace.selected_estimated_utilization = 0.0;
    model->has_selected_kernel = 0;
    stage_start = clock();
    {
        pyc_kernel_desc selected;
        if (pyc_kernel_select_with_policy(
                "matmul_fused",
                model->backend,
                model->options.objective_mode,
                stats.pressure_score,
                &selected,
                &model->kernel_trace) == 0) {
            model->selected_kernel = selected;
            model->has_selected_kernel = 1;
        }
    }
    stage_end = clock();
    kernel_select_ms = elapsed_ms(stage_start, stage_end);
    model->guard_miss_count += guard_miss_count_this_run;
    model->fallback_count += fallback_count_this_run;
    autotune_ms = run_ms;
    if (autotune_ms <= 0.0) {
        autotune_ms = dispatch_ms;
    }
    if (autotune_ms <= 0.0) {
        autotune_ms = 0.001;
    }
    if (!runtime_error && model->options.enable_autotune) {
        size_t candidate_index = 0;
        pyc_kernel_desc candidate;
        double candidate_ms = autotune_ms;

        model->autotune_candidate_count = pyc_kernel_collect(
            "matmul_fused",
            model->backend,
            model->autotune_candidates,
            PYC_AUTOTUNE_CANDIDATE_MAX);

        if (model->autotune_candidate_count > 0) {
            if (model->run_count == 0) {
                candidate_index = 0;
            } else {
                candidate_index = (model->run_count - 1) % model->autotune_candidate_count;
            }
            candidate = model->autotune_candidates[candidate_index];
            candidate_ms = autotune_estimate_candidate_ms(
                autotune_ms,
                &candidate,
                model->options.objective_mode,
                stats.pressure_score);
        } else if (model->has_selected_kernel) {
            candidate = model->selected_kernel;
            candidate_ms = autotune_ms;
        } else {
            memset(&candidate, 0, sizeof(candidate));
            strncpy(candidate.symbol, "matmul_cpu_ref", sizeof(candidate.symbol) - 1);
        }

        if (pyc_kernel_benchmark_update_symbol(
                "matmul_fused",
                model->backend,
                candidate.symbol,
                candidate_ms) == 0) {
            if (autotune_persist_result(
                    model->autotune_db_path,
                    "matmul_fused",
                    model->backend,
                    candidate.symbol,
                    candidate_ms) == 0) {
                model->autotune_saved = 1;
            }
        }
    }

    snprintf(
        model->decision_log,
        sizeof(model->decision_log),
        "mode=%d rollback=%d rollback_count=%zu pressure=%.6f kernel=%s score=%.3f util=%.3f cuda_fallback=%d cuda_reason=%s contract=%d contract_reason=%s guard_miss=%zu fallback=%zu cache_hit=%d budget_exceeded=%d graph_breaks=%zu break_first=%s@%d break_counts=%zu,%zu,%zu,%zu,%zu compilability=%.3f autotune_loaded=%d autotune_saved=%d autotune_candidates=%zu fp=%llu",
        (int)model->options.objective_mode,
        (int)model->controller.last_rollback_reason,
        model->controller.rollback_count,
        stats.pressure_score,
        model->has_selected_kernel ? model->selected_kernel.symbol : "none",
        model->kernel_trace.selected_score,
        model->kernel_trace.selected_estimated_utilization,
        model->cuda_trace.fallback_to_cpu,
        model->cuda_trace.reason,
        contract_ok,
        contract_reason,
        model->guard_miss_count,
        model->fallback_count,
        model->compile_cache_hit,
        model->compile_budget_exceeded,
        model->graph_break_count,
        model->first_graph_break_op_name,
        model->first_graph_break_op_id,
        model->graph_break_const_count,
        model->graph_break_gelu_count,
        model->graph_break_reduce_sum_count,
        model->graph_break_layernorm_count,
        model->graph_break_unknown_count,
        model->compilability_score,
        model->autotune_loaded,
        model->autotune_saved,
        model->autotune_candidate_count,
        (unsigned long long)model->module_fingerprint);

    if (out_stats) {
        memset(out_stats, 0, sizeof(*out_stats));
        out_stats->compile_ms = model->compile_ms;
        out_stats->run_ms = run_ms;
        out_stats->peak_bytes = stats.peak_bytes;
        out_stats->total_requested_bytes = stats.total_requested_bytes;
        out_stats->reused_allocations = stats.reused_allocations;
        out_stats->rematerialized_tensors = stats.rematerialized_tensors;
        out_stats->pressure_events = stats.pressure_events;
        out_stats->pressure_score = stats.pressure_score;
        out_stats->selected_kernel_count = model->has_selected_kernel ? 1 : 0;
        out_stats->selected_kernel_score = model->kernel_trace.selected_score;
        out_stats->estimated_utilization = model->kernel_trace.selected_estimated_utilization;
        out_stats->active_mode = model->options.objective_mode;
        out_stats->rollback_reason = rollback_reason;
        out_stats->rollback_count = model->controller.rollback_count;
        out_stats->dispatch_ms = dispatch_ms;
        out_stats->graph_exec_ms = graph_exec_ms;
        out_stats->controller_ms = controller_ms;
        out_stats->kernel_select_ms = kernel_select_ms;
        out_stats->deterministic_contract_enforced = model->deterministic_contract_enforced;
        out_stats->deterministic_contract_ok = contract_ok;
        out_stats->model_fingerprint = model->module_fingerprint;
        strncpy(out_stats->deterministic_contract_reason, contract_reason, sizeof(out_stats->deterministic_contract_reason) - 1);
        out_stats->compile_cache_hit = model->compile_cache_hit;
        out_stats->compile_budget_exceeded = model->compile_budget_exceeded;
        out_stats->guard_miss_count = model->guard_miss_count;
        out_stats->fallback_count = model->fallback_count;
        out_stats->graph_break_count = model->graph_break_count;
        out_stats->graph_break_const_count = model->graph_break_const_count;
        out_stats->graph_break_gelu_count = model->graph_break_gelu_count;
        out_stats->graph_break_reduce_sum_count = model->graph_break_reduce_sum_count;
        out_stats->graph_break_layernorm_count = model->graph_break_layernorm_count;
        out_stats->graph_break_unknown_count = model->graph_break_unknown_count;
        out_stats->first_graph_break_op_id = model->first_graph_break_op_id;
        strncpy(
            out_stats->first_graph_break_op_name,
            model->first_graph_break_op_name,
            sizeof(out_stats->first_graph_break_op_name) - 1);
        out_stats->first_graph_break_op_name[sizeof(out_stats->first_graph_break_op_name) - 1] = '\0';
        out_stats->compilability_score = model->compilability_score;
        out_stats->autotune_loaded = model->autotune_loaded;
        out_stats->autotune_saved = model->autotune_saved;
        strncpy(out_stats->graph_break_summary, model->graph_break_summary, sizeof(out_stats->graph_break_summary) - 1);
        if (model->has_selected_kernel) {
            strcpy(out_stats->selected_kernel_symbol, model->selected_kernel.symbol);
        }
    }

    return runtime_error ? PYC_STATUS_RUNTIME_FAILED : PYC_STATUS_OK;
}

const char* pyc_model_last_decision_log(const pyc_compiled_model* model) {
    if (!model) {
        return "";
    }
    return model->decision_log;
}

void pyc_destroy_model(pyc_compiled_model* model) {
    release_cpu_workspace(model);
    free(model);
}
