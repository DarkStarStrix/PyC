#include "pyc/kernel_registry.h"

#include <float.h>
#include <string.h>

typedef struct {
    pyc_kernel_desc desc;
    pyc_kernel_benchmark bench;
    int used;
    int promoted;
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

static double kernel_allocator_penalty(
    const pyc_kernel_desc* desc,
    pyc_objective_mode mode,
    const pyc_kernel_coselect_context* context) {
    double penalty = 0.0;
    double shared_mem_kb;
    if (!desc || !context) {
        return 0.0;
    }
    if (mode == PYC_MODE_UTILIZATION_FIRST) {
        return 0.0;
    }
    shared_mem_kb = (double)desc->shared_mem_bytes / 1024.0;

    if (context->pressure_events > 0) {
        double pressure_weight = mode == PYC_MODE_MEMORY_FIRST ? 10.0 :
            (mode == PYC_MODE_BALANCED ? 7.0 : 3.0);
        penalty += (double)context->pressure_events *
            (pressure_weight + shared_mem_kb * 0.5 + (double)desc->reg_pressure_class * 4.0);
    }

    if (context->rematerialized_tensors > 0) {
        double remat_weight = mode == PYC_MODE_MEMORY_FIRST ? 16.0 :
            (mode == PYC_MODE_BALANCED ? 10.0 : 5.0);
        penalty += (double)context->rematerialized_tensors *
            (remat_weight + shared_mem_kb * 0.35 + (double)desc->reg_pressure_class * 3.0);
    }

    if (context->memory_budget_bytes > 0 && context->peak_bytes > context->memory_budget_bytes) {
        double excess_ratio =
            (double)(context->peak_bytes - context->memory_budget_bytes) /
            (double)context->memory_budget_bytes;
        double budget_weight = mode == PYC_MODE_MEMORY_FIRST ? 120.0 :
            (mode == PYC_MODE_BALANCED ? 90.0 : 45.0);
        penalty += excess_ratio *
            (budget_weight + shared_mem_kb * 1.5 + (double)desc->reg_pressure_class * 12.0);
    }

    return penalty;
}

static double kernel_reuse_bonus(
    const pyc_kernel_desc* desc,
    pyc_objective_mode mode,
    const pyc_kernel_coselect_context* context) {
    double reuse_ratio;
    double friendliness;
    double mode_weight;
    if (!desc || !context || context->total_requested_bytes == 0 || context->reused_allocations == 0) {
        return 0.0;
    }
    if (mode == PYC_MODE_UTILIZATION_FIRST) {
        return 0.0;
    }
    reuse_ratio =
        (double)context->reused_allocations /
        (double)(context->reused_allocations + context->pressure_events + 1U);
    friendliness =
        1.0 /
        (1.0 + (double)desc->reg_pressure_class + ((double)desc->shared_mem_bytes / (32.0 * 1024.0)));
    mode_weight = mode == PYC_MODE_MEMORY_FIRST ? 42.0 :
        (mode == PYC_MODE_BALANCED ? 24.0 : 10.0);
    return reuse_ratio * friendliness * mode_weight;
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

static double kernel_joint_score(
    const kernel_slot* slot,
    pyc_objective_mode mode,
    const pyc_kernel_coselect_context* context,
    double* out_pressure_penalty,
    double* out_allocator_penalty,
    double* out_reuse_bonus) {
    double pressure_penalty = 0.0;
    double base_score;
    double allocator_penalty = kernel_allocator_penalty(&slot->desc, mode, context);
    double reuse_bonus = kernel_reuse_bonus(&slot->desc, mode, context);
    double pressure_score = context ? context->pressure_score : 0.0;
    base_score = kernel_score(slot, mode, pressure_score, &pressure_penalty);
    if (out_pressure_penalty) {
        *out_pressure_penalty = pressure_penalty;
    }
    if (out_allocator_penalty) {
        *out_allocator_penalty = allocator_penalty;
    }
    if (out_reuse_bonus) {
        *out_reuse_bonus = reuse_bonus;
    }
    return base_score - allocator_penalty + reuse_bonus;
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
            g_registry[i].promoted = 0;
            return 0;
        }
    }

    return -1;
}

int pyc_kernel_promote_symbol(const char* op_key, pyc_backend backend, const char* symbol) {
    size_t i;
    int target_found = 0;

    if (!op_key) {
        return -1;
    }

    if (symbol && symbol[0] != '\0') {
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
            if (strcmp(g_registry[i].desc.symbol, symbol) == 0) {
                target_found = 1;
                break;
            }
        }
        if (!target_found) {
            return -1;
        }
    }

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
        g_registry[i].promoted = (symbol && symbol[0] != '\0' &&
            strcmp(g_registry[i].desc.symbol, symbol) == 0) ? 1 : 0;
    }

    return 0;
}

int pyc_kernel_promoted_symbol(
    const char* op_key,
    pyc_backend backend,
    char* out_symbol,
    size_t out_symbol_size) {
    size_t i;

    if (!op_key || !out_symbol || out_symbol_size == 0) {
        return -1;
    }

    out_symbol[0] = '\0';
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
        if (!g_registry[i].promoted) {
            continue;
        }
        strncpy(out_symbol, g_registry[i].desc.symbol, out_symbol_size - 1);
        out_symbol[out_symbol_size - 1] = '\0';
        return 0;
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
        trace->allocator_penalty = 0.0;
        trace->reuse_bonus = 0.0;
        trace->candidates_considered = 0;
    }
    return 0;
}

int pyc_kernel_coselect_with_context(
    const char* op_key,
    pyc_backend backend,
    pyc_objective_mode mode,
    const pyc_kernel_coselect_context* context,
    pyc_kernel_desc* out,
    pyc_kernel_selection_trace* trace) {
    size_t i;
    int found = 0;
    pyc_kernel_desc best;
    double best_score = 0.0;
    double best_util = 0.0;
    double best_pressure_penalty = 0.0;
    double best_allocator_penalty = 0.0;
    double best_reuse_bonus = 0.0;
    size_t considered = 0;

    if (!op_key || !out) {
        return -1;
    }

    memset(&best, 0, sizeof(best));

    for (i = 0; i < PYC_KERNEL_MAX; ++i) {
        double score;
        double pressure_penalty = 0.0;
        double allocator_penalty = 0.0;
        double reuse_bonus = 0.0;
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

        considered++;
        score = kernel_joint_score(
            &g_registry[i],
            mode,
            context,
            &pressure_penalty,
            &allocator_penalty,
            &reuse_bonus);
        if (!found || score > best_score) {
            replace = 1;
        } else if (found && score == best_score &&
                   strcmp(g_registry[i].desc.symbol, best.symbol) < 0) {
            replace = 1;
        }

        if (replace) {
            best = g_registry[i].desc;
            best_score = score;
            best_util = g_registry[i].desc.estimated_occupancy;
            best_pressure_penalty = pressure_penalty;
            best_allocator_penalty = allocator_penalty;
            best_reuse_bonus = reuse_bonus;
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
        trace->pressure_penalty = best_pressure_penalty;
        trace->allocator_penalty = best_allocator_penalty;
        trace->reuse_bonus = best_reuse_bonus;
        trace->candidates_considered = considered;
    }
    return 0;
}

int pyc_kernel_benchmark_update_symbol(
    const char* op_key,
    pyc_backend backend,
    const char* symbol,
    double time_ms) {
    size_t i;
    int updated = 0;

    if (!op_key) {
        return -1;
    }
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
        if (symbol && symbol[0] != '\0' &&
            strcmp(g_registry[i].desc.symbol, symbol) != 0) {
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

int pyc_kernel_benchmark_update(const char* op_key, pyc_backend backend, double time_ms) {
    return pyc_kernel_benchmark_update_symbol(op_key, backend, NULL, time_ms);
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

size_t pyc_kernel_collect(
    const char* op_key,
    pyc_backend backend,
    pyc_kernel_desc* out_descs,
    size_t out_capacity) {
    size_t i;
    size_t count = 0;
    if (!op_key || !out_descs || out_capacity == 0) {
        return 0;
    }

    for (i = 0; i < PYC_KERNEL_MAX; ++i) {
        size_t pos;
        if (!g_registry[i].used) {
            continue;
        }
        if (g_registry[i].desc.backend != backend) {
            continue;
        }
        if (strcmp(g_registry[i].desc.op_key, op_key) != 0) {
            continue;
        }
        if (count >= out_capacity) {
            break;
        }

        /* Keep deterministic lexical order by symbol. */
        pos = count;
        while (pos > 0 && strcmp(g_registry[i].desc.symbol, out_descs[pos - 1].symbol) < 0) {
            out_descs[pos] = out_descs[pos - 1];
            pos--;
        }
        out_descs[pos] = g_registry[i].desc;
        count++;
    }
    return count;
}
