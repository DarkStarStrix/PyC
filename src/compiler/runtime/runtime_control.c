#include "pyc/runtime_control.h"

#include <string.h>

void pyc_runtime_rails_default(pyc_runtime_rails* rails) {
    if (!rails) {
        return;
    }
    memset(rails, 0, sizeof(*rails));
    rails->enable_auto_switch = 1;
    rails->enable_hard_rollback = 1;
    rails->dwell_steps = 4;
    rails->cooldown_steps = 2;
    rails->consecutive_breach_windows = 2;
    rails->latency_regression_threshold = 0.08;
    rails->throughput_regression_threshold = 0.05;
    rails->pressure_score_threshold = 0.20;
    rails->pressure_events_threshold = 1;
    rails->rematerialized_tensors_threshold = 4;
}

void pyc_runtime_controller_init(pyc_runtime_controller* controller, pyc_objective_mode initial_mode) {
    if (!controller) {
        return;
    }
    memset(controller, 0, sizeof(*controller));
    controller->current_mode = initial_mode;
    controller->recommended_mode = initial_mode;
    controller->recommendation_reason = PYC_ROLLBACK_NONE;
    controller->last_stable_mode = initial_mode;
}

static int is_latency_breach(const pyc_runtime_rails* rails, const pyc_runtime_window_metrics* metrics) {
    if (metrics->baseline_p95_ms <= 0.0) {
        return 0;
    }
    return metrics->observed_p95_ms >
           metrics->baseline_p95_ms * (1.0 + rails->latency_regression_threshold);
}

static int is_throughput_breach(const pyc_runtime_rails* rails, const pyc_runtime_window_metrics* metrics) {
    if (metrics->baseline_throughput <= 0.0) {
        return 0;
    }
    return metrics->observed_throughput <
           metrics->baseline_throughput * (1.0 - rails->throughput_regression_threshold);
}

static int is_pressure_breach(const pyc_runtime_rails* rails, const pyc_runtime_window_metrics* metrics) {
    return metrics->pressure_score > rails->pressure_score_threshold ||
           metrics->pressure_events >= rails->pressure_events_threshold ||
           metrics->rematerialized_tensors >= rails->rematerialized_tensors_threshold;
}

static void pyc_runtime_controller_shadow_recommendation(
    const pyc_runtime_controller* controller,
    const pyc_runtime_rails* rails,
    const pyc_runtime_window_metrics* metrics,
    pyc_objective_mode* out_recommended_mode,
    pyc_rollback_reason* out_recommendation_reason,
    int* out_breached) {
    pyc_objective_mode recommended_mode;
    pyc_rollback_reason recommendation_reason;
    int breached;

    recommended_mode = controller->current_mode;
    recommendation_reason = PYC_ROLLBACK_NONE;
    breached = 0;

    if (metrics->runtime_error) {
        recommendation_reason = PYC_ROLLBACK_RUNTIME_ERROR;
        recommended_mode = PYC_MODE_BALANCED;
        breached = 1;
    } else if (is_pressure_breach(rails, metrics)) {
        recommendation_reason = PYC_ROLLBACK_PRESSURE;
        recommended_mode = PYC_MODE_MEMORY_FIRST;
        breached = 1;
    } else if (is_latency_breach(rails, metrics)) {
        recommendation_reason = PYC_ROLLBACK_LATENCY;
        recommended_mode = PYC_MODE_BALANCED;
        breached = 1;
    } else if (is_throughput_breach(rails, metrics)) {
        recommendation_reason = PYC_ROLLBACK_THROUGHPUT;
        recommended_mode = PYC_MODE_UTILIZATION_FIRST;
        breached = 1;
    } else if (controller->current_mode == PYC_MODE_MEMORY_FIRST ||
               controller->current_mode == PYC_MODE_UTILIZATION_FIRST) {
        recommended_mode = PYC_MODE_BALANCED;
    }

    *out_recommended_mode = recommended_mode;
    *out_recommendation_reason = recommendation_reason;
    *out_breached = breached;
}

int pyc_runtime_controller_observe(
    pyc_runtime_controller* controller,
    const pyc_runtime_rails* rails,
    const pyc_runtime_window_metrics* metrics,
    pyc_objective_mode* out_mode,
    pyc_rollback_reason* out_rollback_reason) {
    int breached = 0;
    pyc_rollback_reason reason = PYC_ROLLBACK_NONE;
    pyc_objective_mode recommended_mode = controller ? controller->current_mode : PYC_MODE_BALANCED;
    pyc_rollback_reason recommendation_reason = PYC_ROLLBACK_NONE;

    if (!controller || !rails || !metrics || !out_mode || !out_rollback_reason) {
        return -1;
    }

    controller->steps_in_mode++;
    if (controller->cooldown_remaining > 0) {
        controller->cooldown_remaining--;
    }

    pyc_runtime_controller_shadow_recommendation(
        controller,
        rails,
        metrics,
        &recommended_mode,
        &recommendation_reason,
        &breached);
    controller->recommended_mode = recommended_mode;
    controller->recommendation_reason = recommendation_reason;
    reason = recommendation_reason;

    if (breached) {
        controller->consecutive_breaches++;
    } else {
        controller->consecutive_breaches = 0;
    }

    if (rails->enable_hard_rollback &&
        controller->consecutive_breaches >= rails->consecutive_breach_windows) {
        controller->last_rollback_reason = reason;
        controller->rollback_count++;
        controller->current_mode = PYC_MODE_BALANCED;
        controller->last_stable_mode = PYC_MODE_BALANCED;
        controller->steps_in_mode = 0;
        controller->cooldown_remaining = rails->cooldown_steps;
        controller->consecutive_breaches = 0;
    } else if (rails->enable_auto_switch &&
               controller->cooldown_remaining == 0 &&
               controller->steps_in_mode >= rails->dwell_steps &&
               recommended_mode != controller->current_mode) {
        controller->current_mode = recommended_mode;
        controller->last_stable_mode = recommended_mode;
        controller->steps_in_mode = 0;
        controller->cooldown_remaining = rails->cooldown_steps;
        controller->consecutive_breaches = 0;
    }

    *out_mode = controller->current_mode;
    *out_rollback_reason = controller->last_rollback_reason;
    return 0;
}
