#include "pyc/kernel_registry.h"

#include <float.h>
#include <string.h>

typedef struct {
    pyc_kernel_desc desc;
    pyc_kernel_benchmark bench;
    int used;
} kernel_slot;

static kernel_slot g_registry[PYC_KERNEL_MAX];

static double kernel_time_component(double best_time_ms) {
    if (best_time_ms <= 0.0 || best_time_ms == DBL_MAX) {
        return 0.0;
    }
    return -100.0 * best_time_ms;
}

static double kernel_pressure_penalty(const pyc_kernel_desc* desc, pyc_objective_mode mode, double pressure_score) {
    if (pressure_score <= 0.0 || mode == PYC_MODE_UTILIZATION_FIRST) {
        return 0.0;
    }
    return pressure_score * (double)(desc->shared_mem_bytes / 1024U + (size_t)(desc->reg_pressure_class * 8));
}

static double kernel_score(const kernel_slot* slot, pyc_objective_mode mode, double pressure_score, double* out_penalty) {
    double base = (double)slot->desc.priority * 100.0;
    double occ_weight = mode == PYC_MODE_UTILIZATION_FIRST ? 12.0 : 6.0;
    double util = occ_weight * slot->desc.estimated_occupancy;
    double tensor_core_bonus = slot->desc.tensor_core_eligible ? 25.0 : 0.0;
    double time = kernel_time_component(slot->bench.best_time_ms);
    double penalty = kernel_pressure_penalty(&slot->desc, mode, pressure_score);
    if (out_penalty) {
        *out_penalty = penalty;
    }
    return base + util + tensor_core_bonus + time - penalty;
}

void pyc_kernel_registry_reset(void) {
    memset(g_registry, 0, sizeof(g_registry));
}

int pyc_kernel_register(const pyc_kernel_desc* desc) {
    size_t i;
    if (!desc || desc->op_key[0] == '\0') {
        return -1;
    }

    for (i = 0; i < PYC_KERNEL_MAX; ++i) {
        if (!g_registry[i].used) {
            g_registry[i].used = 1;
            g_registry[i].desc = *desc;
            g_registry[i].bench.best_time_ms = DBL_MAX;
            return 0;
        }
    }

    return -1;
}

int pyc_kernel_select(const char* op_key, pyc_backend backend, pyc_kernel_desc* out) {
    return pyc_kernel_select_with_policy(op_key, backend, PYC_MODE_BALANCED, 0.0, out, NULL);
}

int pyc_kernel_select_with_policy(
    const char* op_key,
    pyc_backend backend,
    pyc_objective_mode mode,
    double pressure_score,
    pyc_kernel_desc* out,
    pyc_kernel_selection_trace* trace) {
    size_t i;
    int found = 0;
    pyc_kernel_desc best;
    double best_score = 0.0;
    double best_util = 0.0;
    double best_penalty = 0.0;

    if (!op_key || !out) {
        return -1;
    }

    memset(&best, 0, sizeof(best));

    for (i = 0; i < PYC_KERNEL_MAX; ++i) {
        double penalty;
        double score;
        int replace = 0;

        if (!g_registry[i].used) {
            continue;
        }
        if (g_registry[i].desc.backend != backend) {
            continue;
        }
        if (strcmp(g_registry[i].desc.op_key, op_key) != 0) {
            continue;
        }

        score = kernel_score(&g_registry[i], mode, pressure_score, &penalty);
        if (!found || score > best_score) {
            replace = 1;
        } else if (found && score == best_score && strcmp(g_registry[i].desc.symbol, best.symbol) < 0) {
            replace = 1;
        }

        if (replace) {
            best = g_registry[i].desc;
            best_score = score;
            best_util = g_registry[i].desc.estimated_occupancy;
            best_penalty = penalty;
            found = 1;
        }
    }

    if (!found) {
        return -1;
    }

    *out = best;
    if (trace) {
        trace->selected_score = best_score;
        trace->selected_estimated_utilization = best_util;
        trace->pressure_penalty = best_penalty;
    }
    return 0;
}

int pyc_kernel_benchmark_update(const char* op_key, pyc_backend backend, double time_ms) {
    size_t i;
    int updated = 0;

    for (i = 0; i < PYC_KERNEL_MAX; ++i) {
        if (!g_registry[i].used) {
            continue;
        }
        if (g_registry[i].desc.backend != backend) {
            continue;
        }
        if (strcmp(g_registry[i].desc.op_key, op_key) != 0) {
            continue;
        }

        g_registry[i].bench.considered++;
        if (time_ms < g_registry[i].bench.best_time_ms) {
            g_registry[i].bench.best_time_ms = time_ms;
            g_registry[i].bench.selected++;
        }
        updated = 1;
    }

    return updated ? 0 : -1;
}

void pyc_kernel_benchmark_read(const char* op_key, pyc_backend backend, pyc_kernel_benchmark* out) {
    size_t i;

    if (!out) {
        return;
    }

    memset(out, 0, sizeof(*out));

    for (i = 0; i < PYC_KERNEL_MAX; ++i) {
        if (!g_registry[i].used) {
            continue;
        }
        if (g_registry[i].desc.backend != backend) {
            continue;
        }
        if (strcmp(g_registry[i].desc.op_key, op_key) != 0) {
            continue;
        }

        out->considered += g_registry[i].bench.considered;
        out->selected += g_registry[i].bench.selected;
        if (out->best_time_ms == 0.0 || g_registry[i].bench.best_time_ms < out->best_time_ms) {
            out->best_time_ms = g_registry[i].bench.best_time_ms;
        }
    }
}
