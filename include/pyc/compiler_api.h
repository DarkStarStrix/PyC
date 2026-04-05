#ifndef PYC_COMPILER_API_H
#define PYC_COMPILER_API_H

#include <stddef.h>
#include <stdint.h>

#include "pyc/ir.h"
#include "pyc/collective_comm.h"
#include "pyc/distributed_runtime.h"
#include "pyc/kernel_registry.h"
#include "pyc/optimizer_policy.h"
#include "pyc/runtime_control.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PYC_STATUS_OK = 0,
    PYC_STATUS_INVALID_ARGUMENT = 1,
    PYC_STATUS_VERIFY_FAILED = 2,
    PYC_STATUS_COMPILE_FAILED = 3,
    PYC_STATUS_RUNTIME_FAILED = 4
} pyc_status;

typedef struct {
    const pyc_ir_module* module;
    pyc_backend backend;
} pyc_model_desc;

typedef enum {
    PYC_COMPILE_CACHE_DISABLED = 0,
    PYC_COMPILE_CACHE_IN_MEMORY = 1
} pyc_compile_cache_mode;

typedef enum {
    PYC_DIST_BACKEND_NONE = 0,
    PYC_DIST_BACKEND_NCCL = 1,
    PYC_DIST_BACKEND_RCCL = 2,
    PYC_DIST_BACKEND_MPI = 3,
    PYC_DIST_BACKEND_CUSTOM = 4
} pyc_distributed_backend;

typedef enum {
    PYC_DIST_STRATEGY_NONE = 0,
    PYC_DIST_STRATEGY_DATA_PARALLEL = 1,
    PYC_DIST_STRATEGY_TENSOR_PARALLEL = 2,
    PYC_DIST_STRATEGY_PIPELINE_PARALLEL = 3,
    PYC_DIST_STRATEGY_ZERO = 4,
    PYC_DIST_STRATEGY_3D = 5
} pyc_distributed_strategy;

typedef struct {
    int enabled;
    pyc_distributed_backend backend;
    pyc_distributed_strategy strategy;
    int world_size;
    int rank;
    int local_rank;
    const char* backend_path;
    const char* config_json;
} pyc_distributed_options;

typedef struct {
    int enable_fusion;
    int enable_memory_reuse;
    int enable_autotune;
    int enable_speculative_plans;
    int enable_phantom_graph;
    size_t max_speculative_plans;
    size_t phantom_horizon_steps;
    pyc_objective_mode objective_mode;
    size_t memory_budget_bytes;
    double target_utilization_floor;
    int deterministic_strict;
    double compile_budget_ms;
    pyc_compile_cache_mode cache_mode;
    const char* autotune_db_path;
    pyc_runtime_rails rails;
    pyc_distributed_options distributed;
} pyc_compile_options;

typedef struct {
    void* data;
    size_t size_bytes;
    pyc_dtype dtype;
    pyc_shape shape;
} pyc_tensor;

typedef struct {
    double compile_ms;
    double run_ms;
    size_t peak_bytes;
    size_t total_requested_bytes;
    size_t reused_allocations;
    size_t rematerialized_tensors;
    size_t rematerialized_bytes;
    size_t pressure_events;
    double pressure_score;
    double selected_kernel_score;
    double selected_kernel_allocator_penalty;
    double selected_kernel_reuse_bonus;
    double estimated_utilization;
    pyc_objective_mode active_mode;
    pyc_rollback_reason rollback_reason;
    size_t rollback_count;
    pyc_objective_mode shadow_mode;
    pyc_rollback_reason shadow_reason;
    int selected_kernel_count;
    size_t selected_kernel_candidates;
    char selected_kernel_symbol[PYC_KERNEL_SYMBOL_MAX];
    char execution_path[128];
    double dispatch_ms;
    double graph_exec_ms;
    double controller_ms;
    double kernel_select_ms;
    double cuda_copy_in_ms;
    double cuda_kernel_ms;
    double cuda_copy_out_ms;
    double cuda_sync_ms;
    int deterministic_contract_enforced;
    int deterministic_contract_ok;
    uint64_t model_fingerprint;
    char deterministic_contract_reason[64];
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
    char graph_break_summary[128];
    size_t speculative_plan_count;
    int speculative_plan_hit;
    size_t speculative_plan_miss_count;
    size_t speculative_guard_miss_count;
    double speculative_confidence;
    char speculative_shape_bucket[64];
    int phantom_graph_enabled;
    int phantom_graph_match;
    size_t phantom_graph_match_count;
    size_t phantom_graph_mismatch_count;
    size_t phantom_graph_reshape_count;
    double phantom_graph_confidence;
    double phantom_graph_match_score;
    char phantom_graph_expected_bucket[64];
    char phantom_graph_expected_signature[128];
    char phantom_graph_observed_signature[128];
} pyc_run_stats;

typedef struct pyc_compiled_model pyc_compiled_model;

const char* pyc_status_string(pyc_status status);
pyc_status pyc_compile_model(const pyc_model_desc* desc, const pyc_compile_options* options, pyc_compiled_model** out_model);
pyc_status pyc_run_model(pyc_compiled_model* model, const pyc_tensor* inputs, size_t input_count, pyc_tensor* outputs, size_t output_count, pyc_run_stats* out_stats);
const char* pyc_model_last_decision_log(const pyc_compiled_model* model);
const pyc_collective_comm* pyc_model_distributed_comm(const pyc_compiled_model* model);
pyc_comm_handle_t pyc_model_distributed_comm_handle(const pyc_compiled_model* model);
void pyc_destroy_model(pyc_compiled_model* model);

#ifdef __cplusplus
}
#endif

#endif
