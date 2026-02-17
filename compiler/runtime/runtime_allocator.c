#include "pyc/runtime_allocator.h"

#include <stddef.h>
#include <string.h>

static int intervals_overlap(const pyc_alloc_request* a, const pyc_alloc_request* b) {
    return !(a->end_step < b->start_step || b->end_step < a->start_step);
}

static size_t align_up(size_t value, size_t alignment) {
    size_t a = alignment ? alignment : 1;
    size_t r = value % a;
    return r == 0 ? value : value + (a - r);
}

void pyc_alloc_plan_init(pyc_alloc_plan* plan) {
    if (!plan) {
        return;
    }
    memset(plan, 0, sizeof(*plan));
}

int pyc_alloc_plan_add_request(pyc_alloc_plan* plan, pyc_alloc_request req) {
    if (!plan) {
        return -1;
    }
    if (plan->request_count >= PYC_ALLOC_MAX_REQUESTS) {
        return -1;
    }
    if (req.size_bytes == 0 || req.end_step < req.start_step) {
        return -1;
    }
    plan->requests[plan->request_count++] = req;
    return 0;
}

int pyc_alloc_plan_build(pyc_alloc_plan* plan) {
    return pyc_alloc_plan_build_with_mode(plan, PYC_MODE_BALANCED, 0);
}

int pyc_alloc_plan_build_with_mode(pyc_alloc_plan* plan, pyc_objective_mode mode, size_t memory_budget_bytes) {
    size_t i;

    if (!plan) {
        return -1;
    }

    plan->peak_bytes = 0;
    plan->reused_allocations = 0;
    plan->allocation_events = 0;
    plan->overlap_pairs_observed = 0;
    plan->largest_allocation_bytes = 0;
    plan->rematerialized_tensors = 0;
    plan->pressure_events = 0;
    plan->pressure_score = 0.0;

    for (i = 0; i < plan->request_count; ++i) {
        size_t chosen = (size_t)-1;
        size_t j;

        for (j = 0; j < i; ++j) {
            const pyc_alloc_request* prev = &plan->requests[j];
            const pyc_alloc_request* cur = &plan->requests[i];
            if (intervals_overlap(prev, cur)) {
                plan->overlap_pairs_observed++;
            }
            if (!intervals_overlap(prev, cur) && prev->size_bytes >= cur->size_bytes) {
                chosen = plan->offsets[j];
                plan->reused_allocations++;
                break;
            }
        }

        if (chosen == (size_t)-1) {
            chosen = align_up(plan->peak_bytes, plan->requests[i].alignment);
            plan->peak_bytes = chosen + plan->requests[i].size_bytes;
            plan->allocation_events++;
        }

        if (plan->requests[i].size_bytes > plan->largest_allocation_bytes) {
            plan->largest_allocation_bytes = plan->requests[i].size_bytes;
        }
        plan->offsets[i] = chosen;
    }

    if (memory_budget_bytes > 0 && plan->peak_bytes > memory_budget_bytes) {
        size_t excess = plan->peak_bytes - memory_budget_bytes;
        plan->pressure_events = 1;
        plan->pressure_score = (double)excess / (double)memory_budget_bytes;
        if (mode == PYC_MODE_MEMORY_FIRST) {
            size_t relief = plan->largest_allocation_bytes / 2;
            if (relief == 0) {
                relief = 1;
            }
            plan->rematerialized_tensors = (excess + relief - 1) / relief;
            plan->peak_bytes = memory_budget_bytes;
        } else if (mode == PYC_MODE_BALANCED) {
            size_t reduction = excess / 2;
            plan->rematerialized_tensors = reduction > 0 ? 1 : 0;
            plan->peak_bytes -= reduction;
        }
    }

    return 0;
}

void pyc_alloc_plan_stats(const pyc_alloc_plan* plan, pyc_alloc_stats* stats) {
    size_t i;

    if (!plan || !stats) {
        return;
    }

    memset(stats, 0, sizeof(*stats));
    stats->peak_bytes = plan->peak_bytes;
    stats->reused_allocations = plan->reused_allocations;
    stats->allocation_events = plan->allocation_events;
    stats->overlap_pairs_observed = plan->overlap_pairs_observed;
    stats->largest_allocation_bytes = plan->largest_allocation_bytes;
    stats->rematerialized_tensors = plan->rematerialized_tensors;
    stats->pressure_events = plan->pressure_events;
    stats->pressure_score = plan->pressure_score;

    for (i = 0; i < plan->request_count; ++i) {
        stats->total_requested_bytes += plan->requests[i].size_bytes;
    }
}
