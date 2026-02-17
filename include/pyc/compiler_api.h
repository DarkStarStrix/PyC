#ifndef PYC_COMPILER_API_H
#define PYC_COMPILER_API_H

#include <stddef.h>

#include "pyc/ir.h"
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

typedef struct {
    int enable_fusion;
    int enable_memory_reuse;
    int enable_autotune;
    pyc_objective_mode objective_mode;
    size_t memory_budget_bytes;
    double target_utilization_floor;
    int deterministic_strict;
    pyc_runtime_rails rails;
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
    size_t pressure_events;
    double pressure_score;
    double selected_kernel_score;
    double estimated_utilization;
    pyc_objective_mode active_mode;
    pyc_rollback_reason rollback_reason;
    size_t rollback_count;
    int selected_kernel_count;
    char selected_kernel_symbol[PYC_KERNEL_SYMBOL_MAX];
} pyc_run_stats;

typedef struct pyc_compiled_model pyc_compiled_model;

const char* pyc_status_string(pyc_status status);
pyc_status pyc_compile_model(const pyc_model_desc* desc, const pyc_compile_options* options, pyc_compiled_model** out_model);
pyc_status pyc_run_model(pyc_compiled_model* model, const pyc_tensor* inputs, size_t input_count, pyc_tensor* outputs, size_t output_count, pyc_run_stats* out_stats);
const char* pyc_model_last_decision_log(const pyc_compiled_model* model);
void pyc_destroy_model(pyc_compiled_model* model);

#ifdef __cplusplus
}
#endif

#endif
