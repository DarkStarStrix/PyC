#ifndef PYC_KERNEL_REGISTRY_H
#define PYC_KERNEL_REGISTRY_H

#include <stddef.h>

#include "pyc/optimizer_policy.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PYC_KERNEL_MAX 512
#define PYC_KERNEL_OP_KEY_MAX 64
#define PYC_KERNEL_SYMBOL_MAX 128

typedef enum {
    PYC_BACKEND_CPU = 0,
    PYC_BACKEND_CUDA = 1
} pyc_backend;

typedef struct {
    char op_key[PYC_KERNEL_OP_KEY_MAX];
    pyc_backend backend;
    char symbol[PYC_KERNEL_SYMBOL_MAX];
    int priority;
    double estimated_occupancy;
    int tensor_core_eligible;
    size_t shared_mem_bytes;
    int reg_pressure_class;
} pyc_kernel_desc;

typedef struct {
    int selected;
    int considered;
    double best_time_ms;
} pyc_kernel_benchmark;

typedef struct {
    double selected_score;
    double selected_estimated_utilization;
    double pressure_penalty;
    double allocator_penalty;
    double reuse_bonus;
    size_t candidates_considered;
} pyc_kernel_selection_trace;

typedef struct {
    double pressure_score;
    size_t pressure_events;
    size_t rematerialized_tensors;
    size_t reused_allocations;
    size_t total_requested_bytes;
    size_t peak_bytes;
    size_t memory_budget_bytes;
} pyc_kernel_coselect_context;

void pyc_kernel_registry_reset(void);
int pyc_kernel_register(const pyc_kernel_desc* desc);
int pyc_kernel_select(const char* op_key, pyc_backend backend, pyc_kernel_desc* out);
int pyc_kernel_promote_symbol(const char* op_key, pyc_backend backend, const char* symbol);
int pyc_kernel_promoted_symbol(
    const char* op_key,
    pyc_backend backend,
    char* out_symbol,
    size_t out_symbol_size);
int pyc_kernel_select_with_policy(
    const char* op_key,
    pyc_backend backend,
    pyc_objective_mode mode,
    double pressure_score,
    pyc_kernel_desc* out,
    pyc_kernel_selection_trace* trace);
int pyc_kernel_coselect_with_context(
    const char* op_key,
    pyc_backend backend,
    pyc_objective_mode mode,
    const pyc_kernel_coselect_context* context,
    pyc_kernel_desc* out,
    pyc_kernel_selection_trace* trace);
int pyc_kernel_benchmark_update(const char* op_key, pyc_backend backend, double time_ms);
int pyc_kernel_benchmark_update_symbol(const char* op_key, pyc_backend backend, const char* symbol, double time_ms);
void pyc_kernel_benchmark_read(const char* op_key, pyc_backend backend, pyc_kernel_benchmark* out);
size_t pyc_kernel_collect(
    const char* op_key,
    pyc_backend backend,
    pyc_kernel_desc* out_descs,
    size_t out_capacity);

#ifdef __cplusplus
}
#endif

#endif
