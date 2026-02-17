#ifndef PYC_RUNTIME_ALLOCATOR_H
#define PYC_RUNTIME_ALLOCATOR_H

#include <stddef.h>

#include "pyc/optimizer_policy.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PYC_ALLOC_MAX_REQUESTS 2048

typedef struct {
    int tensor_id;
    size_t size_bytes;
    size_t alignment;
    int start_step;
    int end_step;
} pyc_alloc_request;

typedef struct {
    pyc_alloc_request requests[PYC_ALLOC_MAX_REQUESTS];
    size_t request_count;
    size_t offsets[PYC_ALLOC_MAX_REQUESTS];
    size_t peak_bytes;
    size_t reused_allocations;
    size_t allocation_events;
    size_t overlap_pairs_observed;
    size_t largest_allocation_bytes;
    size_t rematerialized_tensors;
    size_t pressure_events;
    double pressure_score;
} pyc_alloc_plan;

typedef struct {
    size_t peak_bytes;
    size_t total_requested_bytes;
    size_t reused_allocations;
    size_t allocation_events;
    size_t overlap_pairs_observed;
    size_t largest_allocation_bytes;
    size_t rematerialized_tensors;
    size_t pressure_events;
    double pressure_score;
} pyc_alloc_stats;

void pyc_alloc_plan_init(pyc_alloc_plan* plan);
int pyc_alloc_plan_add_request(pyc_alloc_plan* plan, pyc_alloc_request req);
int pyc_alloc_plan_build(pyc_alloc_plan* plan);
int pyc_alloc_plan_build_with_mode(pyc_alloc_plan* plan, pyc_objective_mode mode, size_t memory_budget_bytes);
void pyc_alloc_plan_stats(const pyc_alloc_plan* plan, pyc_alloc_stats* stats);

#ifdef __cplusplus
}
#endif

#endif
