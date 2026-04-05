#include "pyc/compiler_api.h"

#include <stdarg.h>
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
#define PYC_SPECULATIVE_PLAN_MAX 3
#define PYC_SPECULATIVE_SIGNATURE_MAX 128

typedef struct {
    int valid;
    pyc_objective_mode mode;
    pyc_ir_module module;
    pyc_alloc_plan alloc_plan;
    pyc_kernel_desc selected_kernel;
    pyc_kernel_selection_trace kernel_trace;
    int has_selected_kernel;
    char shape_bucket[64];
    char shape_signature[PYC_SPECULATIVE_SIGNATURE_MAX];
    double confidence;
} pyc_speculative_plan;

typedef struct {
    int enabled;
    size_t horizon_steps;
    size_t match_count;
    size_t mismatch_count;
    size_t reshape_count;
    double confidence;
    double last_match_score;
    char expected_bucket[64];
    char expected_signature[PYC_SPECULATIVE_SIGNATURE_MAX];
    char observed_bucket[64];
    char observed_signature[PYC_SPECULATIVE_SIGNATURE_MAX];
} pyc_phantom_graph_state;

typedef struct pyc_compiled_model {
    pyc_ir_module module;
    pyc_backend backend;
    pyc_compile_options options;
    pyc_distributed_runtime* distributed_runtime;
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
    pyc_speculative_plan speculative_plans[PYC_SPECULATIVE_PLAN_MAX];
    size_t speculative_plan_count;
    size_t speculative_plan_miss_count;
    size_t speculative_guard_miss_count;
    double speculative_confidence;
    char speculative_shape_bucket[64];
    char speculative_shape_signature[PYC_SPECULATIVE_SIGNATURE_MAX];
    pyc_phantom_graph_state phantom_graph;
    char graph_break_summary[128];
    char autotune_db_path[256];
    uint64_t compile_cache_key;
    char decision_log[2048];
    int last_runtime_shape_valid;
    uint64_t last_runtime_shape_hash;
    char last_runtime_shape_bucket[64];
    char last_runtime_shape_signature[PYC_SPECULATIVE_SIGNATURE_MAX];
    size_t last_runtime_input_count;
    size_t last_runtime_output_count;
    pyc_tensor last_runtime_inputs[PYC_IR_MAX_INPUTS];
    pyc_tensor last_runtime_outputs[PYC_IR_MAX_INPUTS];
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

static int tensor_descriptor_equal(const pyc_tensor* a, const pyc_tensor* b) {
    size_t dim;
    if (!a || !b) {
        return 0;
    }
    if (a->data != b->data ||
        a->size_bytes != b->size_bytes ||
        a->dtype != b->dtype ||
        a->shape.rank != b->shape.rank) {
        return 0;
    }
    for (dim = 0; dim < a->shape.rank && dim < PYC_IR_MAX_DIMS; ++dim) {
        if (a->shape.dims[dim] != b->shape.dims[dim]) {
            return 0;
        }
    }
    return 1;
}

static int tensor_descriptor_array_equal(
    const pyc_tensor* current,
    size_t current_count,
    const pyc_tensor* cached,
    size_t cached_count) {
    size_t i;
    if (!current || !cached || current_count != cached_count) {
        return 0;
    }
    for (i = 0; i < current_count; ++i) {
        if (!tensor_descriptor_equal(&current[i], &cached[i])) {
            return 0;
        }
    }
    return 1;
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

static size_t decision_log_append_text(char* dst, size_t dst_size, size_t used, const char* text) {
    size_t copy;
    if (!dst || dst_size == 0) {
        return 0;
    }
    if (used >= dst_size) {
        dst[dst_size - 1] = '\0';
        return dst_size - 1;
    }
    if (!text) {
        text = "";
    }
    copy = strlen(text);
    if (copy > dst_size - used - 1) {
        copy = dst_size - used - 1;
    }
    if (copy > 0) {
        memcpy(dst + used, text, copy);
    }
    used += copy;
    dst[used] = '\0';
    return used;
}

static size_t decision_log_appendf(char* dst, size_t dst_size, size_t used, const char* fmt, ...) {
    va_list ap;
    int written;
    size_t remaining;

    if (!dst || dst_size == 0 || !fmt) {
        return 0;
    }
    if (used >= dst_size) {
        return dst_size - 1;
    }
    remaining = dst_size - used;
    va_start(ap, fmt);
    written = vsnprintf(dst + used, remaining, fmt, ap);
    va_end(ap);
    if (written < 0) {
        return used;
    }
    if ((size_t)written >= remaining) {
        return dst_size - 1;
    }
    return used + (size_t)written;
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
    pyc_speculative_plan speculative_plans[PYC_SPECULATIVE_PLAN_MAX];
    size_t speculative_plan_count;
    double speculative_confidence;
    char speculative_shape_bucket[64];
    char speculative_shape_signature[PYC_SPECULATIVE_SIGNATURE_MAX];
    pyc_phantom_graph_state phantom_graph;
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
    key = hash_u64(key, (uint64_t)options->enable_speculative_plans);
    key = hash_u64(key, (uint64_t)options->enable_phantom_graph);
    key = hash_u64(key, (uint64_t)options->max_speculative_plans);
    key = hash_u64(key, (uint64_t)options->phantom_horizon_steps);
    key = hash_u64(key, (uint64_t)options->objective_mode);
    key = hash_u64(key, (uint64_t)options->memory_budget_bytes);
    key = hash_u64(key, (uint64_t)(options->target_utilization_floor * 1000.0));
    key = hash_u64(key, (uint64_t)options->deterministic_strict);
    key = hash_u64(key, (uint64_t)options->cache_mode);
    key = hash_u64(key, (uint64_t)(options->compile_budget_ms * 1000.0));
    key = hash_u64(key, (uint64_t)options->distributed.enabled);
    key = hash_u64(key, (uint64_t)options->distributed.backend);
    key = hash_u64(key, (uint64_t)options->distributed.strategy);
    key = hash_u64(key, (uint64_t)options->distributed.world_size);
    key = hash_u64(key, (uint64_t)options->distributed.rank);
    key = hash_u64(key, (uint64_t)options->distributed.local_rank);
    if (options->autotune_db_path && options->autotune_db_path[0] != '\0') {
        key = fnv1a64(options->autotune_db_path, strlen(options->autotune_db_path)) ^ key;
    }
    if (options->distributed.backend_path && options->distributed.backend_path[0] != '\0') {
        key = fnv1a64(options->distributed.backend_path, strlen(options->distributed.backend_path)) ^ key;
    }
    if (options->distributed.config_json && options->distributed.config_json[0] != '\0') {
        key = fnv1a64(options->distributed.config_json, strlen(options->distributed.config_json)) ^ key;
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
    const char* graph_break_summary,
    const pyc_speculative_plan* speculative_plans,
    size_t speculative_plan_count,
    double speculative_confidence,
    const char* speculative_shape_bucket,
    const char* speculative_shape_signature,
    const pyc_phantom_graph_state* phantom_graph) {
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
    if (speculative_plans) {
        size_t i;
        if (speculative_plan_count > PYC_SPECULATIVE_PLAN_MAX) {
            speculative_plan_count = PYC_SPECULATIVE_PLAN_MAX;
        }
        entry->speculative_plan_count = speculative_plan_count;
        for (i = 0; i < speculative_plan_count; ++i) {
            entry->speculative_plans[i] = speculative_plans[i];
        }
    }
    entry->speculative_confidence = speculative_confidence;
    if (speculative_shape_bucket) {
        strncpy(
            entry->speculative_shape_bucket,
            speculative_shape_bucket,
            sizeof(entry->speculative_shape_bucket) - 1);
        entry->speculative_shape_bucket[sizeof(entry->speculative_shape_bucket) - 1] = '\0';
    }
    if (speculative_shape_signature) {
        strncpy(
            entry->speculative_shape_signature,
            speculative_shape_signature,
            sizeof(entry->speculative_shape_signature) - 1);
        entry->speculative_shape_signature[sizeof(entry->speculative_shape_signature) - 1] = '\0';
    }
    if (phantom_graph) {
        entry->phantom_graph = *phantom_graph;
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
    o.enable_speculative_plans = 0;
    o.enable_phantom_graph = 0;
    o.max_speculative_plans = 0;
    o.phantom_horizon_steps = 1;
    o.objective_mode = PYC_MODE_BALANCED;
    o.memory_budget_bytes = 0;
    o.target_utilization_floor = 0.70;
    o.deterministic_strict = 1;
    o.compile_budget_ms = 0.0;
    o.cache_mode = PYC_COMPILE_CACHE_IN_MEMORY;
    o.autotune_db_path = NULL;
    pyc_runtime_rails_default(&o.rails);
    o.distributed.enabled = 0;
    o.distributed.backend = PYC_DIST_BACKEND_NONE;
    o.distributed.strategy = PYC_DIST_STRATEGY_NONE;
    o.distributed.world_size = 1;
    o.distributed.rank = 0;
    o.distributed.local_rank = 0;
    o.distributed.backend_path = NULL;
    o.distributed.config_json = NULL;
    return o;
}

static void sanitize_distributed_options(pyc_distributed_options* dist) {
    if (!dist) {
        return;
    }
    if (!(dist->enabled == 0 || dist->enabled == 1)) {
        dist->enabled = 0;
    }
    if (dist->backend < PYC_DIST_BACKEND_NONE || dist->backend > PYC_DIST_BACKEND_CUSTOM) {
        dist->backend = PYC_DIST_BACKEND_NONE;
    }
    if (dist->strategy < PYC_DIST_STRATEGY_NONE || dist->strategy > PYC_DIST_STRATEGY_3D) {
        dist->strategy = PYC_DIST_STRATEGY_NONE;
    }
    if (dist->world_size <= 0) {
        dist->world_size = 1;
    }
    if (dist->rank < 0 || dist->rank >= dist->world_size) {
        dist->rank = 0;
    }
    if (dist->local_rank < 0) {
        dist->local_rank = 0;
    }
    if (!dist->enabled) {
        dist->backend = PYC_DIST_BACKEND_NONE;
        dist->strategy = PYC_DIST_STRATEGY_NONE;
        dist->world_size = 1;
        dist->rank = 0;
        dist->local_rank = 0;
        dist->backend_path = NULL;
        dist->config_json = NULL;
    }
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

static size_t dtype_size_bytes(pyc_dtype dtype) {
    switch (dtype) {
        case PYC_DTYPE_F16:
            return 2;
        case PYC_DTYPE_I8:
            return 1;
        case PYC_DTYPE_F32:
        case PYC_DTYPE_I32:
            return 4;
        case PYC_DTYPE_UNKNOWN:
        default:
            return 0;
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

static void describe_shape_bucket(
    size_t total_input_bytes,
    size_t input_count,
    char* out_bucket,
    size_t out_bucket_size) {
    const char* klass = "unknown";
    if (!out_bucket || out_bucket_size == 0) {
        return;
    }
    if (total_input_bytes > 0 && total_input_bytes <= 256) {
        klass = "tiny";
    } else if (total_input_bytes <= 4096) {
        klass = "small";
    } else if (total_input_bytes <= 65536) {
        klass = "medium";
    } else if (total_input_bytes > 65536) {
        klass = "large";
    }
    snprintf(out_bucket, out_bucket_size, "%s:i%zu:b%zu", klass, input_count, total_input_bytes);
}

static int append_shape_signature_token(
    char* out_signature,
    size_t out_signature_size,
    size_t* io_used,
    pyc_dtype dtype,
    const pyc_shape* shape) {
    int n;
    size_t d;
    if (!out_signature || !io_used || !shape || out_signature_size == 0) {
        return -1;
    }

    n = snprintf(
        out_signature + *io_used,
        out_signature_size - *io_used,
        "%s%d:r%zu",
        *io_used == 0 ? "" : ";",
        (int)dtype,
        shape->rank);
    if (n < 0 || (size_t)n >= out_signature_size - *io_used) {
        return -1;
    }
    *io_used += (size_t)n;

    for (d = 0; d < shape->rank; ++d) {
        n = snprintf(
            out_signature + *io_used,
            out_signature_size - *io_used,
            "x%lld",
            (long long)shape->dims[d]);
        if (n < 0 || (size_t)n >= out_signature_size - *io_used) {
            return -1;
        }
        *io_used += (size_t)n;
    }

    return 0;
}

static void module_input_shape_bucket(
    const pyc_ir_module* module,
    char* out_bucket,
    size_t out_bucket_size) {
    size_t i;
    size_t total_input_bytes = 0;
    size_t input_count = 0;

    if (!module) {
        if (out_bucket && out_bucket_size > 0) {
            out_bucket[0] = '\0';
        }
        return;
    }

    for (i = 0; i < module->op_count; ++i) {
        const pyc_ir_op* op = &module->ops[i];
        size_t elems;
        size_t elem_bytes;
        if (op->kind != PYC_IR_OP_INPUT) {
            continue;
        }
        elems = shape_num_elements(&op->shape);
        elem_bytes = dtype_size_bytes(op->dtype);
        input_count++;
        if (elems > 0 && elem_bytes > 0) {
            total_input_bytes += elems * elem_bytes;
        }
    }

    describe_shape_bucket(total_input_bytes, input_count, out_bucket, out_bucket_size);
}

static void module_input_shape_signature(
    const pyc_ir_module* module,
    char* out_signature,
    size_t out_signature_size) {
    size_t i;
    size_t used = 0;

    if (!out_signature || out_signature_size == 0) {
        return;
    }
    out_signature[0] = '\0';
    if (!module) {
        return;
    }

    for (i = 0; i < module->op_count; ++i) {
        const pyc_ir_op* op = &module->ops[i];
        if (op->kind != PYC_IR_OP_INPUT) {
            continue;
        }
        if (append_shape_signature_token(
                out_signature,
                out_signature_size,
                &used,
                op->dtype,
                &op->shape) != 0) {
            out_signature[0] = '\0';
            return;
        }
    }
}

static double phantom_graph_match_score(
    const char* expected_signature,
    const char* observed_signature,
    const char* expected_bucket,
    const char* observed_bucket) {
    if (expected_signature && observed_signature &&
        expected_signature[0] != '\0' &&
        observed_signature[0] != '\0' &&
        strcmp(expected_signature, observed_signature) == 0) {
        return 1.0;
    }
    if (expected_bucket && observed_bucket &&
        expected_bucket[0] != '\0' &&
        observed_bucket[0] != '\0' &&
        strcmp(expected_bucket, observed_bucket) == 0) {
        return 0.75;
    }
    if ((expected_signature && expected_signature[0] != '\0') ||
        (observed_signature && observed_signature[0] != '\0')) {
        return 0.25;
    }
    return 0.0;
}

static void phantom_graph_init(
    pyc_phantom_graph_state* phantom,
    const pyc_compile_options* options,
    const pyc_pass_report* report,
    const pyc_ir_module* module) {
    if (!phantom) {
        return;
    }
    memset(phantom, 0, sizeof(*phantom));
    if (!options || !options->enable_phantom_graph) {
        return;
    }
    phantom->enabled = 1;
    phantom->horizon_steps = options->phantom_horizon_steps == 0 ? 1 : options->phantom_horizon_steps;
    phantom->confidence = report ? report->phantom_confidence : 0.0;
    if (report && report->phantom_shape_bucket[0] != '\0') {
        strncpy(phantom->expected_bucket, report->phantom_shape_bucket, sizeof(phantom->expected_bucket) - 1);
    } else {
        module_input_shape_bucket(module, phantom->expected_bucket, sizeof(phantom->expected_bucket));
    }
    if (report && report->phantom_shape_signature[0] != '\0') {
        strncpy(phantom->expected_signature, report->phantom_shape_signature, sizeof(phantom->expected_signature) - 1);
    } else {
        module_input_shape_signature(module, phantom->expected_signature, sizeof(phantom->expected_signature));
    }
}

static int phantom_graph_compare(
    pyc_phantom_graph_state* phantom,
    const char* runtime_bucket,
    const char* runtime_signature,
    double* out_score) {
    double score;
    int matched;
    if (out_score) {
        *out_score = 0.0;
    }
    if (!phantom || !phantom->enabled) {
        return 0;
    }
    strncpy(phantom->observed_bucket, runtime_bucket ? runtime_bucket : "", sizeof(phantom->observed_bucket) - 1);
    phantom->observed_bucket[sizeof(phantom->observed_bucket) - 1] = '\0';
    strncpy(phantom->observed_signature, runtime_signature ? runtime_signature : "", sizeof(phantom->observed_signature) - 1);
    phantom->observed_signature[sizeof(phantom->observed_signature) - 1] = '\0';
    score = phantom_graph_match_score(
        phantom->expected_signature,
        phantom->observed_signature,
        phantom->expected_bucket,
        phantom->observed_bucket);
    phantom->last_match_score = score;
    matched = score >= 0.999;
    if (matched) {
        phantom->match_count++;
    } else {
        phantom->mismatch_count++;
    }
    if (out_score) {
        *out_score = score;
    }
    return matched;
}

static void phantom_graph_adapt(
    pyc_phantom_graph_state* phantom,
    const char* runtime_bucket,
    const char* runtime_signature,
    int runtime_error) {
    int should_reshape;
    if (!phantom || !phantom->enabled || runtime_error) {
        return;
    }
    should_reshape =
        ((runtime_signature && runtime_signature[0] != '\0' &&
          strcmp(phantom->expected_signature, runtime_signature) != 0) ||
         (runtime_bucket && runtime_bucket[0] != '\0' &&
          strcmp(phantom->expected_bucket, runtime_bucket) != 0));
    if (!should_reshape) {
        phantom->confidence = (phantom->confidence * 0.85) + (phantom->last_match_score * 0.15);
        return;
    }
    if (phantom->horizon_steps > 1 &&
        (phantom->mismatch_count % phantom->horizon_steps) != 0) {
        phantom->confidence = (phantom->confidence * 0.7) + (phantom->last_match_score * 0.3);
        return;
    }
    phantom->reshape_count++;
    if (runtime_bucket && runtime_bucket[0] != '\0') {
        strncpy(phantom->expected_bucket, runtime_bucket, sizeof(phantom->expected_bucket) - 1);
        phantom->expected_bucket[sizeof(phantom->expected_bucket) - 1] = '\0';
    }
    if (runtime_signature && runtime_signature[0] != '\0') {
        strncpy(phantom->expected_signature, runtime_signature, sizeof(phantom->expected_signature) - 1);
        phantom->expected_signature[sizeof(phantom->expected_signature) - 1] = '\0';
    }
    phantom->confidence = (phantom->confidence * 0.6) + (phantom->last_match_score * 0.4);
}

static void runtime_input_shape_bucket(
    const pyc_tensor* inputs,
    size_t input_count,
    char* out_bucket,
    size_t out_bucket_size) {
    size_t i;
    size_t total_input_bytes = 0;
    if (!inputs) {
        if (out_bucket && out_bucket_size > 0) {
            out_bucket[0] = '\0';
        }
        return;
    }
    for (i = 0; i < input_count; ++i) {
        size_t elems = shape_num_elements(&inputs[i].shape);
        size_t elem_bytes = dtype_size_bytes(inputs[i].dtype);
        if (elems > 0 && elem_bytes > 0) {
            total_input_bytes += elems * elem_bytes;
        } else {
            total_input_bytes += inputs[i].size_bytes;
        }
    }
    describe_shape_bucket(total_input_bytes, input_count, out_bucket, out_bucket_size);
}

static void runtime_input_shape_signature(
    const pyc_tensor* inputs,
    size_t input_count,
    char* out_signature,
    size_t out_signature_size) {
    size_t i;
    size_t used = 0;

    if (!out_signature || out_signature_size == 0) {
        return;
    }
    out_signature[0] = '\0';
    if (!inputs) {
        return;
    }

    for (i = 0; i < input_count; ++i) {
        if (append_shape_signature_token(
                out_signature,
                out_signature_size,
                &used,
                inputs[i].dtype,
                &inputs[i].shape) != 0) {
            out_signature[0] = '\0';
            return;
        }
    }
}

static uint64_t runtime_input_shape_hash(const pyc_tensor* inputs, size_t input_count) {
    uint64_t hash = 1469598103934665603ULL;
    size_t i;

    if (!inputs) {
        return 0;
    }
    for (i = 0; i < input_count; ++i) {
        size_t d;
        hash ^= (uint64_t)inputs[i].dtype;
        hash *= 1099511628211ULL;
        hash ^= (uint64_t)inputs[i].size_bytes;
        hash *= 1099511628211ULL;
        hash ^= (uint64_t)inputs[i].shape.rank;
        hash *= 1099511628211ULL;
        for (d = 0; d < inputs[i].shape.rank; ++d) {
            hash ^= (uint64_t)inputs[i].shape.dims[d];
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

static int scale_positive_dim(int64_t dim, size_t scale_factor, int64_t* out_dim) {
    size_t scaled;
    if (!out_dim || dim <= 0 || scale_factor == 0) {
        return -1;
    }
    if ((uint64_t)dim > (uint64_t)((int64_t)INT64_MAX / (int64_t)scale_factor)) {
        return -1;
    }
    scaled = (size_t)dim * scale_factor;
    if (scaled == 0 || scaled > (size_t)INT64_MAX) {
        return -1;
    }
    *out_dim = (int64_t)scaled;
    return 0;
}

static int propagate_specialized_shapes(pyc_ir_module* module) {
    size_t i;

    if (!module) {
        return -1;
    }

    for (i = 0; i < module->op_count; ++i) {
        pyc_ir_op* op = &module->ops[i];
        if (op->kind == PYC_IR_OP_INPUT || op->kind == PYC_IR_OP_CONST) {
            continue;
        }
        if (op->kind == PYC_IR_OP_MATMUL) {
            const pyc_ir_op* lhs;
            const pyc_ir_op* rhs;
            if (op->input_count < 2 ||
                op->input_ids[0] < 0 ||
                op->input_ids[1] < 0 ||
                (size_t)op->input_ids[0] >= module->op_count ||
                (size_t)op->input_ids[1] >= module->op_count) {
                return -1;
            }
            lhs = &module->ops[(size_t)op->input_ids[0]];
            rhs = &module->ops[(size_t)op->input_ids[1]];
            if (lhs->shape.rank != 2 || rhs->shape.rank != 2) {
                return -1;
            }
            if (lhs->shape.dims[1] != rhs->shape.dims[0]) {
                return -1;
            }
            op->dtype = lhs->dtype;
            op->shape.rank = 2;
            op->shape.dims[0] = lhs->shape.dims[0];
            op->shape.dims[1] = rhs->shape.dims[1];
        } else if (op->input_count >= 1 &&
                   op->input_ids[0] >= 0 &&
                   (size_t)op->input_ids[0] < module->op_count) {
            const pyc_ir_op* in0 = &module->ops[(size_t)op->input_ids[0]];
            op->shape = in0->shape;
            op->dtype = in0->dtype;
        }
    }

    return 0;
}

static int build_scaled_shape_variant(
    const pyc_ir_module* base_module,
    size_t scale_factor,
    pyc_ir_module* out_module) {
    size_t i;

    if (!base_module || !out_module || scale_factor == 0) {
        return -1;
    }

    *out_module = *base_module;
    if (scale_factor == 1) {
        return 0;
    }

    for (i = 0; i < out_module->op_count; ++i) {
        pyc_ir_op* op = &out_module->ops[i];
        size_t d;
        if (op->kind != PYC_IR_OP_INPUT) {
            continue;
        }
        for (d = 0; d < op->shape.rank; ++d) {
            int64_t scaled_dim;
            if (scale_positive_dim(op->shape.dims[d], scale_factor, &scaled_dim) != 0) {
                return -1;
            }
            op->shape.dims[d] = scaled_dim;
        }
    }

    return propagate_specialized_shapes(out_module);
}

static int inputs_match_module(
    const pyc_ir_module* module,
    const pyc_tensor* inputs,
    size_t input_count,
    char* out_reason,
    size_t out_reason_size,
    char* out_bucket,
    size_t out_bucket_size) {
    size_t i;
    size_t seen_inputs = 0;

    if (out_reason && out_reason_size > 0) {
        strncpy(out_reason, "ok", out_reason_size - 1);
        out_reason[out_reason_size - 1] = '\0';
    }
    runtime_input_shape_bucket(inputs, input_count, out_bucket, out_bucket_size);

    if (!module || !inputs) {
        if (out_reason && out_reason_size > 0) {
            strncpy(out_reason, "input_unavailable", out_reason_size - 1);
            out_reason[out_reason_size - 1] = '\0';
        }
        return 0;
    }

    for (i = 0; i < module->op_count; ++i) {
        const pyc_ir_op* op = &module->ops[i];
        size_t d;
        size_t required_bytes;
        if (op->kind != PYC_IR_OP_INPUT) {
            continue;
        }
        if (seen_inputs >= input_count) {
            if (out_reason && out_reason_size > 0) {
                strncpy(out_reason, "input_count_mismatch", out_reason_size - 1);
                out_reason[out_reason_size - 1] = '\0';
            }
            return 0;
        }
        if (inputs[seen_inputs].dtype != op->dtype) {
            if (out_reason && out_reason_size > 0) {
                strncpy(out_reason, "input_dtype_mismatch", out_reason_size - 1);
                out_reason[out_reason_size - 1] = '\0';
            }
            return 0;
        }
        if (inputs[seen_inputs].shape.rank != op->shape.rank) {
            if (out_reason && out_reason_size > 0) {
                strncpy(out_reason, "input_shape_mismatch", out_reason_size - 1);
                out_reason[out_reason_size - 1] = '\0';
            }
            return 0;
        }
        for (d = 0; d < op->shape.rank; ++d) {
            if (inputs[seen_inputs].shape.dims[d] != op->shape.dims[d]) {
                if (out_reason && out_reason_size > 0) {
                    strncpy(out_reason, "input_shape_mismatch", out_reason_size - 1);
                    out_reason[out_reason_size - 1] = '\0';
                }
                return 0;
            }
        }
        required_bytes = shape_num_elements(&op->shape) * dtype_size_bytes(op->dtype);
        if (required_bytes == 0 || inputs[seen_inputs].size_bytes < required_bytes) {
            if (out_reason && out_reason_size > 0) {
                strncpy(out_reason, "input_size_mismatch", out_reason_size - 1);
                out_reason[out_reason_size - 1] = '\0';
            }
            return 0;
        }
        seen_inputs++;
    }

    if (seen_inputs != input_count) {
        if (out_reason && out_reason_size > 0) {
            strncpy(out_reason, "input_count_mismatch", out_reason_size - 1);
            out_reason[out_reason_size - 1] = '\0';
        }
        return 0;
    }

    return 1;
}

static int populate_alloc_requests_for_module(const pyc_ir_module* module, pyc_alloc_plan* plan) {
    size_t i;
    if (!module || !plan) {
        return -1;
    }
    pyc_alloc_plan_init(plan);
    for (i = 0; i < module->op_count; ++i) {
        pyc_alloc_request req;
        size_t bytes = dtype_size_bytes(module->ops[i].dtype);
        size_t d;
        const pyc_ir_op* op = &module->ops[i];

        if (op->shape.rank == 0 || bytes == 0) {
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
        if (pyc_alloc_plan_add_request(plan, req) != 0) {
            return -1;
        }
    }
    return 0;
}

static void populate_coselect_context_from_plan(
    const pyc_alloc_plan* plan,
    size_t memory_budget_bytes,
    pyc_kernel_coselect_context* out_context) {
    pyc_alloc_stats stats;
    if (!out_context) {
        return;
    }
    memset(out_context, 0, sizeof(*out_context));
    if (!plan) {
        return;
    }
    pyc_alloc_plan_stats(plan, &stats);
    out_context->pressure_score = stats.pressure_score;
    out_context->pressure_events = stats.pressure_events;
    out_context->rematerialized_tensors = stats.rematerialized_tensors;
    out_context->reused_allocations = stats.reused_allocations;
    out_context->total_requested_bytes = stats.total_requested_bytes;
    out_context->peak_bytes = stats.peak_bytes;
    out_context->memory_budget_bytes = memory_budget_bytes;
}

static void populate_coselect_context_from_stats(
    const pyc_alloc_stats* stats,
    size_t memory_budget_bytes,
    pyc_kernel_coselect_context* out_context) {
    if (!out_context) {
        return;
    }
    memset(out_context, 0, sizeof(*out_context));
    if (!stats) {
        return;
    }
    out_context->pressure_score = stats->pressure_score;
    out_context->pressure_events = stats->pressure_events;
    out_context->rematerialized_tensors = stats->rematerialized_tensors;
    out_context->reused_allocations = stats->reused_allocations;
    out_context->total_requested_bytes = stats->total_requested_bytes;
    out_context->peak_bytes = stats->peak_bytes;
    out_context->memory_budget_bytes = memory_budget_bytes;
}

static int build_speculative_plan_variant(
    const pyc_ir_module* module,
    pyc_backend backend,
    const pyc_compile_options* options,
    pyc_objective_mode mode,
    const char* shape_bucket,
    const char* shape_signature,
    double confidence,
    pyc_speculative_plan* out_plan) {
    pyc_kernel_coselect_context context;
    pyc_kernel_desc selected;
    if (!module || !options || !out_plan) {
        return -1;
    }

    memset(out_plan, 0, sizeof(*out_plan));
    out_plan->valid = 1;
    out_plan->mode = mode;
    out_plan->module = *module;
    out_plan->confidence = confidence;
    if (shape_bucket) {
        strncpy(out_plan->shape_bucket, shape_bucket, sizeof(out_plan->shape_bucket) - 1);
        out_plan->shape_bucket[sizeof(out_plan->shape_bucket) - 1] = '\0';
    }
    if (shape_signature) {
        strncpy(out_plan->shape_signature, shape_signature, sizeof(out_plan->shape_signature) - 1);
        out_plan->shape_signature[sizeof(out_plan->shape_signature) - 1] = '\0';
    }
    if (populate_alloc_requests_for_module(&out_plan->module, &out_plan->alloc_plan) != 0) {
        return -1;
    }
    if (options->enable_memory_reuse) {
        if (pyc_alloc_plan_build_with_mode(
                &out_plan->alloc_plan,
                mode,
                options->memory_budget_bytes) != 0) {
            return -1;
        }
    }

    populate_coselect_context_from_plan(&out_plan->alloc_plan, options->memory_budget_bytes, &context);
    if (pyc_kernel_coselect_with_context(
            "matmul_fused",
            backend,
            mode,
            &context,
            &selected,
            &out_plan->kernel_trace) == 0) {
        out_plan->selected_kernel = selected;
        out_plan->has_selected_kernel = 1;
    }
    return 0;
}

static void apply_speculative_plan(pyc_compiled_model* model, const pyc_speculative_plan* plan) {
    if (!model || !plan || !plan->valid) {
        return;
    }
    model->alloc_plan = plan->alloc_plan;
    model->kernel_trace = plan->kernel_trace;
    model->has_selected_kernel = plan->has_selected_kernel;
    if (plan->has_selected_kernel) {
        model->selected_kernel = plan->selected_kernel;
    } else {
        memset(&model->selected_kernel, 0, sizeof(model->selected_kernel));
    }
    model->speculative_confidence = plan->confidence;
    strncpy(
        model->speculative_shape_bucket,
        plan->shape_bucket,
        sizeof(model->speculative_shape_bucket) - 1);
    model->speculative_shape_bucket[sizeof(model->speculative_shape_bucket) - 1] = '\0';
    strncpy(
        model->speculative_shape_signature,
        plan->shape_signature,
        sizeof(model->speculative_shape_signature) - 1);
    model->speculative_shape_signature[sizeof(model->speculative_shape_signature) - 1] = '\0';
}

static const pyc_speculative_plan* find_speculative_plan(
    const pyc_compiled_model* model,
    pyc_objective_mode mode,
    const char* shape_signature,
    const char* shape_bucket) {
    size_t i;
    if (!model) {
        return NULL;
    }
    for (i = 0; i < model->speculative_plan_count; ++i) {
        const pyc_speculative_plan* plan = &model->speculative_plans[i];
        if (!plan->valid) {
            continue;
        }
        if (shape_signature && shape_signature[0] != '\0' &&
            plan->mode == mode &&
            strcmp(plan->shape_signature, shape_signature) == 0) {
            return plan;
        }
    }
    for (i = 0; i < model->speculative_plan_count; ++i) {
        const pyc_speculative_plan* plan = &model->speculative_plans[i];
        if (!plan->valid) {
            continue;
        }
        if (shape_signature && shape_signature[0] != '\0' &&
            strcmp(plan->shape_signature, shape_signature) == 0) {
            return plan;
        }
    }
    if (!shape_bucket || shape_bucket[0] == '\0') {
        return NULL;
    }
    for (i = 0; i < model->speculative_plan_count; ++i) {
        const pyc_speculative_plan* plan = &model->speculative_plans[i];
        if (!plan->valid) {
            continue;
        }
        if (plan->mode == mode && strcmp(plan->shape_bucket, shape_bucket) == 0) {
            return plan;
        }
    }
    for (i = 0; i < model->speculative_plan_count; ++i) {
        const pyc_speculative_plan* plan = &model->speculative_plans[i];
        if (!plan->valid) {
            continue;
        }
        if (strcmp(plan->shape_bucket, shape_bucket) == 0) {
            return plan;
        }
    }
    return NULL;
}

static int initialize_speculative_plans(
    pyc_compiled_model* model,
    const pyc_pass_report* report) {
    static const size_t scale_factors[PYC_SPECULATIVE_PLAN_MAX] = {1, 2, 4};
    size_t requested_count;
    size_t i;

    if (!model) {
        return -1;
    }

    model->speculative_plan_count = 0;
    model->speculative_plan_miss_count = 0;
    model->speculative_guard_miss_count = 0;
    module_input_shape_bucket(
        &model->module,
        model->speculative_shape_bucket,
        sizeof(model->speculative_shape_bucket));
    module_input_shape_signature(
        &model->module,
        model->speculative_shape_signature,
        sizeof(model->speculative_shape_signature));
    model->speculative_confidence = report ? report->speculative_confidence : model->compilability_score;

    if (!model->options.enable_speculative_plans) {
        return 0;
    }

    requested_count = model->options.max_speculative_plans;
    if (requested_count == 0 && report) {
        requested_count = report->speculative_plan_count;
    }
    if (requested_count == 0) {
        requested_count = PYC_SPECULATIVE_PLAN_MAX;
    }
    if (requested_count > PYC_SPECULATIVE_PLAN_MAX) {
        requested_count = PYC_SPECULATIVE_PLAN_MAX;
    }

    for (i = 0; i < requested_count; ++i) {
        pyc_ir_module specialized_module;
        char specialized_bucket[64];
        char specialized_signature[PYC_SPECULATIVE_SIGNATURE_MAX];
        double scaled_confidence = model->speculative_confidence;
        if (build_scaled_shape_variant(&model->module, scale_factors[i], &specialized_module) != 0) {
            return -1;
        }
        module_input_shape_bucket(
            &specialized_module,
            specialized_bucket,
            sizeof(specialized_bucket));
        module_input_shape_signature(
            &specialized_module,
            specialized_signature,
            sizeof(specialized_signature));
        if (i == 1) {
            scaled_confidence *= 0.95;
        } else if (i >= 2) {
            scaled_confidence *= 0.85;
        }
        if (build_speculative_plan_variant(
                &specialized_module,
                model->backend,
                &model->options,
                model->options.objective_mode,
                specialized_bucket,
                specialized_signature,
                scaled_confidence,
                &model->speculative_plans[model->speculative_plan_count]) != 0) {
            return -1;
        }
        model->speculative_plan_count++;
    }

    if (model->speculative_plan_count > 0) {
        apply_speculative_plan(model, &model->speculative_plans[0]);
    }

    return 0;
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

static void release_model_resources(pyc_compiled_model* model) {
    if (!model) {
        return;
    }
    if (model->distributed_runtime) {
        pyc_distributed_runtime_destroy(model->distributed_runtime);
        model->distributed_runtime = NULL;
    }
    release_cpu_workspace(model);
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
        model->options.enable_speculative_plans = options->enable_speculative_plans;
        model->options.enable_phantom_graph = options->enable_phantom_graph;
        model->options.max_speculative_plans = options->max_speculative_plans;
        model->options.phantom_horizon_steps = options->phantom_horizon_steps;
        model->options.objective_mode = options->objective_mode;
        model->options.memory_budget_bytes = options->memory_budget_bytes;
        model->options.target_utilization_floor = options->target_utilization_floor;
        model->options.deterministic_strict = options->deterministic_strict;
        model->options.compile_budget_ms = options->compile_budget_ms;
        model->options.cache_mode = options->cache_mode;
        model->options.autotune_db_path = options->autotune_db_path;
        model->options.rails = options->rails;
        model->options.distributed = options->distributed;
    }
    if (model->options.cache_mode != PYC_COMPILE_CACHE_DISABLED &&
        model->options.cache_mode != PYC_COMPILE_CACHE_IN_MEMORY) {
        model->options.cache_mode = PYC_COMPILE_CACHE_IN_MEMORY;
    }
    if (!(model->options.enable_speculative_plans == 0 || model->options.enable_speculative_plans == 1)) {
        model->options.enable_speculative_plans = 0;
    }
    if (!(model->options.enable_phantom_graph == 0 || model->options.enable_phantom_graph == 1)) {
        model->options.enable_phantom_graph = 0;
    }
    if (model->options.max_speculative_plans > PYC_SPECULATIVE_PLAN_MAX) {
        model->options.max_speculative_plans = PYC_SPECULATIVE_PLAN_MAX;
    }
    if (model->options.phantom_horizon_steps == 0) {
        model->options.phantom_horizon_steps = 1;
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
    sanitize_distributed_options(&model->options.distributed);
    if (model->options.objective_mode < PYC_MODE_BALANCED ||
        model->options.objective_mode > PYC_MODE_UTILIZATION_FIRST) {
        model->options.objective_mode = PYC_MODE_BALANCED;
    }
    resolve_autotune_db_path(&model->options, model->autotune_db_path, sizeof(model->autotune_db_path));

    if (model->options.distributed.enabled) {
        if (!model->options.distributed.backend_path || model->options.distributed.backend_path[0] == '\0') {
            free(model);
            return PYC_STATUS_INVALID_ARGUMENT;
        }
        model->distributed_runtime = pyc_distributed_runtime_init(
            model->options.distributed.backend_path,
            model->options.distributed.config_json,
            model->options.distributed.world_size,
            model->options.distributed.rank,
            model->options.distributed.local_rank);
        if (!model->distributed_runtime) {
            free(model);
            return PYC_STATUS_COMPILE_FAILED;
        }
    }

    pyc_alloc_plan_init(&model->alloc_plan);
    pyc_runtime_controller_init(&model->controller, model->options.objective_mode);
    pyc_cuda_dispatch_trace_init(&model->cuda_trace);
    if (module_fingerprint(desc->module, &source_fingerprint) != 0) {
        release_model_resources(model);
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
        size_t i;
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
        model->speculative_plan_count = cache_entry->speculative_plan_count;
        model->speculative_confidence = cache_entry->speculative_confidence;
        strncpy(
            model->speculative_shape_bucket,
            cache_entry->speculative_shape_bucket,
            sizeof(model->speculative_shape_bucket) - 1);
        model->speculative_shape_bucket[sizeof(model->speculative_shape_bucket) - 1] = '\0';
        strncpy(
            model->speculative_shape_signature,
            cache_entry->speculative_shape_signature,
            sizeof(model->speculative_shape_signature) - 1);
        model->speculative_shape_signature[sizeof(model->speculative_shape_signature) - 1] = '\0';
        model->phantom_graph = cache_entry->phantom_graph;
        for (i = 0; i < model->speculative_plan_count && i < PYC_SPECULATIVE_PLAN_MAX; ++i) {
            model->speculative_plans[i] = cache_entry->speculative_plans[i];
        }
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
            release_model_resources(model);
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
            release_model_resources(model);
            free(model);
            return PYC_STATUS_COMPILE_FAILED;
        }

        ensure_default_kernel_catalog(model->backend);
        if (model->options.enable_autotune) {
            model->autotune_loaded = autotune_load_into_registry(
                model->autotune_db_path,
                "matmul_fused",
                model->backend);
        }
        if (initialize_speculative_plans(model, &report) != 0) {
            release_model_resources(model);
            free(model);
            return PYC_STATUS_COMPILE_FAILED;
        }
        phantom_graph_init(&model->phantom_graph, &model->options, &report, &model->module);
        if (model->speculative_plan_count == 0) {
            pyc_speculative_plan base_plan;
            module_input_shape_bucket(
                &model->module,
                model->speculative_shape_bucket,
                sizeof(model->speculative_shape_bucket));
            if (build_speculative_plan_variant(
                    &model->module,
                    model->backend,
                    &model->options,
                    model->options.objective_mode,
                    model->speculative_shape_bucket,
                    model->speculative_shape_signature,
                    report.speculative_confidence,
                    &base_plan) != 0) {
                release_model_resources(model);
                free(model);
                return PYC_STATUS_COMPILE_FAILED;
            }
            apply_speculative_plan(model, &base_plan);
            model->speculative_confidence = report.speculative_confidence;
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
                model->graph_break_summary,
                model->speculative_plans,
                model->speculative_plan_count,
                model->speculative_confidence,
                model->speculative_shape_bucket,
                model->speculative_shape_signature,
                &model->phantom_graph);
        }
    }

    model->autotune_candidate_count = pyc_kernel_collect(
        "matmul_fused",
        model->backend,
        model->autotune_candidates,
        PYC_AUTOTUNE_CANDIDATE_MAX);

    if (init_cpu_workspace(model) != 0) {
        release_model_resources(model);
        free(model);
        return PYC_STATUS_COMPILE_FAILED;
    }

    end = clock();
    model->compile_ms = elapsed_ms(start, end);
    if (model->options.compile_budget_ms > 0.0 &&
        model->compile_ms > model->options.compile_budget_ms) {
        model->compile_budget_exceeded = 1;
    }

    {
        size_t decision_log_used = 0;
        model->decision_log[0] = '\0';
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, "mode=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%d", (int)model->options.objective_mode);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " budget=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%zu", model->options.memory_budget_bytes);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " pressure=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%.6f", model->alloc_plan.pressure_score);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " kernel=");
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, model->has_selected_kernel ? model->selected_kernel.symbol : "none");
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " score=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%.3f", model->kernel_trace.selected_score);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " util=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%.3f", model->kernel_trace.selected_estimated_utilization);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " alloc_penalty=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%.3f", model->kernel_trace.allocator_penalty);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " reuse_bonus=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%.3f", model->kernel_trace.reuse_bonus);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " kernel_candidates=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%zu", model->kernel_trace.candidates_considered);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " det=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%d", model->options.deterministic_strict ? 1 : 0);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " cache_hit=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%d", model->compile_cache_hit);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " compile_ms=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%.3f", model->compile_ms);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " budget_ms=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%.3f", model->options.compile_budget_ms);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " budget_exceeded=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%d", model->compile_budget_exceeded);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " graph_breaks=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%zu", model->graph_break_count);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " break_first=");
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, model->first_graph_break_op_name);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, "@");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%d", model->first_graph_break_op_id);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " break_counts=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%zu,%zu,%zu,%zu,%zu",
            model->graph_break_const_count,
            model->graph_break_gelu_count,
            model->graph_break_reduce_sum_count,
            model->graph_break_layernorm_count,
            model->graph_break_unknown_count);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " compilability=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%.3f", model->compilability_score);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " autotune_loaded=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%d", model->autotune_loaded);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " spec_plans=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%zu", model->speculative_plan_count);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " spec_bucket=");
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, model->speculative_shape_bucket);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " spec_conf=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%.3f", model->speculative_confidence);
    }

    *out_model = model;
    return PYC_STATUS_OK;
}

pyc_status pyc_run_model(pyc_compiled_model* model, const pyc_tensor* inputs, size_t input_count, pyc_tensor* outputs, size_t output_count, pyc_run_stats* out_stats) {
    clock_t start;
    clock_t end;
    clock_t stage_start;
    clock_t stage_end;
    pyc_alloc_stats stats;
    pyc_kernel_coselect_context kernel_context;
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
    char runtime_shape_bucket[64];
    char runtime_shape_signature[PYC_SPECULATIVE_SIGNATURE_MAX];
    const pyc_ir_module* execution_module = &model->module;
    int runtime_error = 0;
    int speculative_plan_hit_this_run = 0;
    int phantom_graph_match_this_run = 0;
    size_t guard_miss_count_this_run = 0;
    size_t fallback_count_this_run = 0;
    double autotune_ms = 0.0;
    double phantom_graph_score_this_run = 0.0;
    uint64_t runtime_shape_hash = 0;
    int runtime_shape_cached = 0;
    int stable_repeat_fastpath = 0;

    if (!model || !inputs || !outputs || input_count == 0 || output_count == 0) {
        return PYC_STATUS_INVALID_ARGUMENT;
    }

    start = clock();
    memset(contract_reason, 0, sizeof(contract_reason));
    memset(runtime_shape_bucket, 0, sizeof(runtime_shape_bucket));
    memset(runtime_shape_signature, 0, sizeof(runtime_shape_signature));
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

    if (!model->options.enable_speculative_plans &&
        !model->options.enable_phantom_graph &&
        model->last_runtime_shape_valid) {
        if (input_count <= PYC_IR_MAX_INPUTS &&
            output_count <= PYC_IR_MAX_INPUTS &&
            tensor_descriptor_array_equal(
                inputs,
                input_count,
                model->last_runtime_inputs,
                model->last_runtime_input_count) &&
            tensor_descriptor_array_equal(
                outputs,
                output_count,
                model->last_runtime_outputs,
                model->last_runtime_output_count)) {
            strncpy(runtime_shape_bucket, model->last_runtime_shape_bucket, sizeof(runtime_shape_bucket) - 1);
            runtime_shape_bucket[sizeof(runtime_shape_bucket) - 1] = '\0';
            strncpy(runtime_shape_signature, model->last_runtime_shape_signature, sizeof(runtime_shape_signature) - 1);
            runtime_shape_signature[sizeof(runtime_shape_signature) - 1] = '\0';
            runtime_shape_cached = 1;
            runtime_shape_hash = model->last_runtime_shape_hash;
        } else {
            runtime_shape_hash = runtime_input_shape_hash(inputs, input_count);
            if (runtime_shape_hash != 0 &&
                runtime_shape_hash == model->last_runtime_shape_hash) {
                strncpy(runtime_shape_bucket, model->last_runtime_shape_bucket, sizeof(runtime_shape_bucket) - 1);
                runtime_shape_bucket[sizeof(runtime_shape_bucket) - 1] = '\0';
                strncpy(runtime_shape_signature, model->last_runtime_shape_signature, sizeof(runtime_shape_signature) - 1);
                runtime_shape_signature[sizeof(runtime_shape_signature) - 1] = '\0';
                runtime_shape_cached = 1;
            }
        }
    }
    if (!runtime_shape_cached) {
        runtime_input_shape_bucket(inputs, input_count, runtime_shape_bucket, sizeof(runtime_shape_bucket));
        runtime_input_shape_signature(inputs, input_count, runtime_shape_signature, sizeof(runtime_shape_signature));
        if (runtime_shape_hash == 0) {
            runtime_shape_hash = runtime_input_shape_hash(inputs, input_count);
        }
    }
    phantom_graph_match_this_run = phantom_graph_compare(
        &model->phantom_graph,
        runtime_shape_bucket,
        runtime_shape_signature,
        &phantom_graph_score_this_run);
    if (!runtime_error && model->speculative_plan_count > 0) {
        const pyc_speculative_plan* active_plan = find_speculative_plan(
            model,
            model->options.objective_mode,
            runtime_shape_signature,
            runtime_shape_bucket);
        if (active_plan) {
            execution_module = &active_plan->module;
            apply_speculative_plan(model, active_plan);
            speculative_plan_hit_this_run = 1;
        } else {
            model->speculative_plan_miss_count++;
            if (runtime_shape_bucket[0] != '\0' &&
                strcmp(runtime_shape_bucket, model->speculative_shape_bucket) == 0) {
                execution_module = &model->module;
            }
        }
    }

    if (!runtime_error &&
        !runtime_shape_cached &&
        !inputs_match_module(
            execution_module,
            inputs,
            input_count,
            contract_reason,
            sizeof(contract_reason),
            NULL,
            0)) {
        contract_ok = 0;
        runtime_error = 1;
        guard_miss_count_this_run++;
        model->speculative_guard_miss_count++;
    }

    pyc_cuda_dispatch_trace_init(&model->cuda_trace);
    stage_start = clock();
    if (!runtime_error) {
        if (model->backend == PYC_BACKEND_CPU) {
            if (execute_cpu_graph(execution_module, inputs, input_count, outputs, output_count, model) != 0) {
                runtime_error = 1;
            }
        } else if (model->backend == PYC_BACKEND_CUDA) {
            pyc_cuda_dispatch_status cuda_status = pyc_cuda_dispatch(
                execution_module,
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
    if (!runtime_error &&
        !model->options.enable_speculative_plans &&
        !model->options.enable_phantom_graph &&
        runtime_shape_hash != 0) {
        size_t i;
        model->last_runtime_shape_valid = 1;
        model->last_runtime_shape_hash = runtime_shape_hash;
        strncpy(model->last_runtime_shape_bucket, runtime_shape_bucket, sizeof(model->last_runtime_shape_bucket) - 1);
        model->last_runtime_shape_bucket[sizeof(model->last_runtime_shape_bucket) - 1] = '\0';
        strncpy(model->last_runtime_shape_signature, runtime_shape_signature, sizeof(model->last_runtime_shape_signature) - 1);
        model->last_runtime_shape_signature[sizeof(model->last_runtime_shape_signature) - 1] = '\0';
        model->last_runtime_input_count = input_count <= PYC_IR_MAX_INPUTS ? input_count : 0;
        model->last_runtime_output_count = output_count <= PYC_IR_MAX_INPUTS ? output_count : 0;
        for (i = 0; i < model->last_runtime_input_count; ++i) {
            model->last_runtime_inputs[i] = inputs[i];
        }
        for (i = 0; i < model->last_runtime_output_count; ++i) {
            model->last_runtime_outputs[i] = outputs[i];
        }
    } else if (runtime_error) {
        model->last_runtime_shape_valid = 0;
        model->last_runtime_shape_hash = 0;
        model->last_runtime_shape_bucket[0] = '\0';
        model->last_runtime_shape_signature[0] = '\0';
        model->last_runtime_input_count = 0;
        model->last_runtime_output_count = 0;
    }
    phantom_graph_adapt(
        &model->phantom_graph,
        runtime_shape_bucket,
        runtime_shape_signature,
        runtime_error);

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

    stable_repeat_fastpath =
        env_default_true("PYC_ENABLE_STABLE_REPEAT_FASTPATH") &&
        runtime_shape_cached &&
        !runtime_error &&
        !model->options.enable_speculative_plans &&
        !model->options.enable_phantom_graph &&
        !model->options.enable_autotune &&
        model->backend == PYC_BACKEND_CUDA;

    if (!stable_repeat_fastpath) {
        populate_coselect_context_from_stats(&stats, model->options.memory_budget_bytes, &kernel_context);
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
        if (model->speculative_plan_count > 0) {
            const pyc_speculative_plan* next_plan = find_speculative_plan(
                model,
                model->options.objective_mode,
                runtime_shape_signature,
                runtime_shape_bucket[0] != '\0' ? runtime_shape_bucket : model->speculative_shape_bucket);
            if (next_plan) {
                apply_speculative_plan(model, next_plan);
            } else {
                model->speculative_plan_miss_count++;
                if (pyc_kernel_coselect_with_context(
                        "matmul_fused",
                        model->backend,
                        model->options.objective_mode,
                        &kernel_context,
                        &model->selected_kernel,
                        &model->kernel_trace) == 0) {
                    model->has_selected_kernel = 1;
                }
            }
        } else {
            pyc_kernel_desc selected;
            if (pyc_kernel_coselect_with_context(
                    "matmul_fused",
                    model->backend,
                    model->options.objective_mode,
                    &kernel_context,
                    &selected,
                    &model->kernel_trace) == 0) {
                model->selected_kernel = selected;
                model->has_selected_kernel = 1;
            }
        }
        stage_end = clock();
        kernel_select_ms = elapsed_ms(stage_start, stage_end);
    } else {
        active_mode = model->options.objective_mode;
        rollback_reason = model->controller.last_rollback_reason;
        controller_ms = 0.0;
        kernel_select_ms = 0.0;
    }
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

    if (env_default_true("PYC_ENABLE_RUNTIME_DECISION_LOG")) {
        size_t decision_log_used = 0;
        const char* spec_bucket = runtime_shape_bucket[0] != '\0' ? runtime_shape_bucket : model->speculative_shape_bucket;
        model->decision_log[0] = '\0';
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, "mode=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%d", (int)model->options.objective_mode);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " shadow_mode=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%d", (int)model->controller.recommended_mode);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " shadow_reason=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%d", (int)model->controller.recommendation_reason);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " rollback=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%d", (int)model->controller.last_rollback_reason);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " rollback_count=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%zu", model->controller.rollback_count);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " pressure=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%.6f", stats.pressure_score);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " kernel=");
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, model->has_selected_kernel ? model->selected_kernel.symbol : "none");
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " score=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%.3f", model->kernel_trace.selected_score);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " util=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%.3f", model->kernel_trace.selected_estimated_utilization);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " alloc_penalty=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%.3f", model->kernel_trace.allocator_penalty);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " reuse_bonus=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%.3f", model->kernel_trace.reuse_bonus);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " kernel_candidates=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%zu", model->kernel_trace.candidates_considered);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " cuda_fallback=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%d", model->cuda_trace.fallback_to_cpu);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " cuda_reason=");
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, model->cuda_trace.reason);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " contract=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%d", contract_ok);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " contract_reason=");
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, contract_reason);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " guard_miss=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%zu", model->guard_miss_count);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " fallback=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%zu", model->fallback_count);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " cache_hit=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%d", model->compile_cache_hit);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " budget_exceeded=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%d", model->compile_budget_exceeded);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " graph_breaks=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%zu", model->graph_break_count);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " break_first=");
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, model->first_graph_break_op_name);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, "@");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%d", model->first_graph_break_op_id);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " break_counts=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%zu,%zu,%zu,%zu,%zu",
            model->graph_break_const_count,
            model->graph_break_gelu_count,
            model->graph_break_reduce_sum_count,
            model->graph_break_layernorm_count,
            model->graph_break_unknown_count);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " compilability=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%.3f", model->compilability_score);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " autotune_loaded=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%d", model->autotune_loaded);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " autotune_saved=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%d", model->autotune_saved);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " autotune_candidates=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%zu", model->autotune_candidate_count);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " spec_plans=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%zu", model->speculative_plan_count);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " spec_hit=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%d", speculative_plan_hit_this_run);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " spec_miss=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%zu", model->speculative_plan_miss_count);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " spec_guard=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%zu", model->speculative_guard_miss_count);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " spec_bucket=");
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, spec_bucket);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " spec_conf=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%.3f", model->speculative_confidence);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " phantom_enabled=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%d", model->phantom_graph.enabled);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " phantom_match=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%d", phantom_graph_match_this_run);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " phantom_score=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%.3f", model->phantom_graph.last_match_score);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " phantom_matches=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%zu", model->phantom_graph.match_count);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " phantom_misses=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%zu", model->phantom_graph.mismatch_count);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " phantom_reshapes=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%zu", model->phantom_graph.reshape_count);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " phantom_expect=");
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, model->phantom_graph.expected_signature);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " phantom_observed=");
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, model->phantom_graph.observed_signature);
        decision_log_used = decision_log_append_text(model->decision_log, sizeof(model->decision_log), decision_log_used, " fp=");
        decision_log_used = decision_log_appendf(model->decision_log, sizeof(model->decision_log), decision_log_used, "%llu", (unsigned long long)model->module_fingerprint);
    } else {
        model->decision_log[0] = '\0';
    }

    if (out_stats) {
        memset(out_stats, 0, sizeof(*out_stats));
        out_stats->compile_ms = model->compile_ms;
        out_stats->run_ms = run_ms;
        out_stats->peak_bytes = stats.peak_bytes;
        out_stats->total_requested_bytes = stats.total_requested_bytes;
        out_stats->reused_allocations = stats.reused_allocations;
        out_stats->rematerialized_tensors = stats.rematerialized_tensors;
        out_stats->rematerialized_bytes = stats.rematerialized_bytes;
        out_stats->pressure_events = stats.pressure_events;
        out_stats->pressure_score = stats.pressure_score;
        out_stats->selected_kernel_count = model->has_selected_kernel ? 1 : 0;
        out_stats->selected_kernel_candidates = model->kernel_trace.candidates_considered;
        out_stats->selected_kernel_score = model->kernel_trace.selected_score;
        out_stats->selected_kernel_allocator_penalty = model->kernel_trace.allocator_penalty;
        out_stats->selected_kernel_reuse_bonus = model->kernel_trace.reuse_bonus;
        out_stats->estimated_utilization = model->kernel_trace.selected_estimated_utilization;
        out_stats->active_mode = model->options.objective_mode;
        out_stats->rollback_reason = rollback_reason;
        out_stats->rollback_count = model->controller.rollback_count;
        out_stats->shadow_mode = model->controller.recommended_mode;
        out_stats->shadow_reason = model->controller.recommendation_reason;
        strncpy(out_stats->execution_path, model->cuda_trace.reason, sizeof(out_stats->execution_path) - 1);
        out_stats->execution_path[sizeof(out_stats->execution_path) - 1] = '\0';
        out_stats->dispatch_ms = dispatch_ms;
        out_stats->graph_exec_ms = graph_exec_ms;
        out_stats->controller_ms = controller_ms;
        out_stats->kernel_select_ms = kernel_select_ms;
        out_stats->cuda_copy_in_ms = model->cuda_trace.copy_in_ms;
        out_stats->cuda_kernel_ms = model->cuda_trace.kernel_ms;
        out_stats->cuda_copy_out_ms = model->cuda_trace.copy_out_ms;
        out_stats->cuda_sync_ms = model->cuda_trace.sync_ms;
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
        out_stats->speculative_plan_count = model->speculative_plan_count;
        out_stats->speculative_plan_hit = speculative_plan_hit_this_run;
        out_stats->speculative_plan_miss_count = model->speculative_plan_miss_count;
        out_stats->speculative_guard_miss_count = model->speculative_guard_miss_count;
        out_stats->speculative_confidence = model->speculative_confidence;
        strncpy(
            out_stats->speculative_shape_bucket,
            runtime_shape_bucket[0] != '\0' ? runtime_shape_bucket : model->speculative_shape_bucket,
            sizeof(out_stats->speculative_shape_bucket) - 1);
        out_stats->speculative_shape_bucket[sizeof(out_stats->speculative_shape_bucket) - 1] = '\0';
        out_stats->phantom_graph_enabled = model->phantom_graph.enabled;
        out_stats->phantom_graph_match = phantom_graph_match_this_run;
        out_stats->phantom_graph_match_count = model->phantom_graph.match_count;
        out_stats->phantom_graph_mismatch_count = model->phantom_graph.mismatch_count;
        out_stats->phantom_graph_reshape_count = model->phantom_graph.reshape_count;
        out_stats->phantom_graph_confidence = model->phantom_graph.confidence;
        out_stats->phantom_graph_match_score = phantom_graph_score_this_run;
        strncpy(
            out_stats->phantom_graph_expected_bucket,
            model->phantom_graph.expected_bucket,
            sizeof(out_stats->phantom_graph_expected_bucket) - 1);
        out_stats->phantom_graph_expected_bucket[sizeof(out_stats->phantom_graph_expected_bucket) - 1] = '\0';
        strncpy(
            out_stats->phantom_graph_expected_signature,
            model->phantom_graph.expected_signature,
            sizeof(out_stats->phantom_graph_expected_signature) - 1);
        out_stats->phantom_graph_expected_signature[sizeof(out_stats->phantom_graph_expected_signature) - 1] = '\0';
        strncpy(
            out_stats->phantom_graph_observed_signature,
            model->phantom_graph.observed_signature,
            sizeof(out_stats->phantom_graph_observed_signature) - 1);
        out_stats->phantom_graph_observed_signature[sizeof(out_stats->phantom_graph_observed_signature) - 1] = '\0';
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

const pyc_collective_comm* pyc_model_distributed_comm(const pyc_compiled_model* model) {
    if (!model || !model->distributed_runtime) {
        return NULL;
    }
    return pyc_distributed_runtime_comm_const(model->distributed_runtime);
}

pyc_comm_handle_t pyc_model_distributed_comm_handle(const pyc_compiled_model* model) {
    if (!model || !model->distributed_runtime) {
        return NULL;
    }
    return pyc_distributed_runtime_handle(model->distributed_runtime);
}

void pyc_destroy_model(pyc_compiled_model* model) {
    release_model_resources(model);
    free(model);
}
