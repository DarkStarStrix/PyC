#include "pyc/runtime_allocator.h"

#include <stddef.h>
#include <string.h>

static int intervals_overlap(const pyc_alloc_request* a, const pyc_alloc_request* b) {
    return !(a->end_step < b->start_step || b->end_step < a->start_step);
}

typedef struct {
    size_t benefit_bytes;
    size_t lifetime_span;
} remat_candidate;

static size_t align_up(size_t value, size_t alignment) {
    size_t a = alignment ? alignment : 1;
    size_t r = value % a;
    return r == 0 ? value : value + (a - r);
}

static size_t request_lifetime_span(const pyc_alloc_request* req) {
    if (!req || req->end_step < req->start_step) {
        return 0;
    }
    return (size_t)(req->end_step - req->start_step + 1);
}

static size_t request_remat_benefit(const pyc_alloc_request* req) {
    size_t span;
    if (!req || req->size_bytes == 0) {
        return 0;
    }
    span = request_lifetime_span(req);
    if (span <= 1) {
        return req->size_bytes;
    }
    return req->size_bytes / span;
}

static int remat_candidate_better(const remat_candidate* lhs, const remat_candidate* rhs) {
    size_t lhs_den;
    size_t rhs_den;
    if (!lhs) {
        return 0;
    }
    if (!rhs) {
        return 1;
    }
    lhs_den = lhs->lifetime_span == 0 ? 1 : lhs->lifetime_span;
    rhs_den = rhs->lifetime_span == 0 ? 1 : rhs->lifetime_span;
    return (lhs->benefit_bytes * rhs_den) > (rhs->benefit_bytes * lhs_den);
}

static size_t estimate_rematerialization_relief(
    pyc_alloc_plan* plan,
    pyc_objective_mode mode,
    size_t target_relief_bytes) {
    remat_candidate candidates[PYC_ALLOC_MAX_REQUESTS];
    size_t candidate_count = 0;
    size_t relieved_bytes = 0;
    size_t i;

    if (!plan || target_relief_bytes == 0) {
        return 0;
    }

    for (i = 0; i < plan->request_count; ++i) {
        size_t benefit = request_remat_benefit(&plan->requests[i]);
        size_t span = request_lifetime_span(&plan->requests[i]);
        if (benefit == 0) {
            continue;
        }
        candidates[candidate_count].benefit_bytes = benefit;
        candidates[candidate_count].lifetime_span = span;
        candidate_count++;
    }

    while (candidate_count > 0 && relieved_bytes < target_relief_bytes) {
        size_t best = 0;
        for (i = 1; i < candidate_count; ++i) {
            if (remat_candidate_better(&candidates[i], &candidates[best])) {
                best = i;
            }
        }
        relieved_bytes += candidates[best].benefit_bytes;
        plan->rematerialized_tensors++;
        plan->rematerialized_bytes += candidates[best].benefit_bytes;
        candidates[best] = candidates[candidate_count - 1];
        candidate_count--;
        if (mode == PYC_MODE_BALANCED) {
            break;
        }
    }

    return relieved_bytes;
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
    plan->rematerialized_bytes = 0;
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
        size_t relief_target = 0;
        size_t relieved_bytes = 0;
        plan->pressure_events = 1;
        plan->pressure_score = (double)excess / (double)memory_budget_bytes;
        if (mode == PYC_MODE_MEMORY_FIRST) {
            relief_target = excess;
        } else if (mode == PYC_MODE_BALANCED) {
            relief_target = excess / 2;
            if (relief_target == 0) {
                relief_target = 1;
            }
        }
        relieved_bytes = estimate_rematerialization_relief(plan, mode, relief_target);
        if (relieved_bytes > 0) {
            if (relieved_bytes >= excess) {
                plan->peak_bytes = memory_budget_bytes;
            } else {
                plan->peak_bytes -= relieved_bytes;
            }
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
    stats->rematerialized_bytes = plan->rematerialized_bytes;
    stats->pressure_events = plan->pressure_events;
    stats->pressure_score = plan->pressure_score;

    for (i = 0; i < plan->request_count; ++i) {
        stats->total_requested_bytes += plan->requests[i].size_bytes;
    }
}
