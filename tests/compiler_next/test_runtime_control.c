#include <stdio.h>
#include <string.h>

#include "pyc/runtime_control.h"

int main(void) {
    pyc_runtime_rails rails;
    pyc_runtime_controller controller;
    pyc_runtime_window_metrics m;
    pyc_objective_mode mode;
    pyc_rollback_reason reason;

    pyc_runtime_rails_default(&rails);
    pyc_runtime_controller_init(&controller, PYC_MODE_UTILIZATION_FIRST);

    memset(&m, 0, sizeof(m));
    m.baseline_p95_ms = 10.0;
    m.observed_p95_ms = 10.1;
    m.baseline_throughput = 1000.0;
    m.observed_throughput = 1005.0;

    /* Disabled adaptation should still report shadow recommendations deterministically. */
    rails.enable_auto_switch = 0;
    rails.enable_hard_rollback = 0;
    m.pressure_score = rails.pressure_score_threshold + 1.0;
    m.pressure_events = rails.pressure_events_threshold;
    m.rematerialized_tensors = rails.rematerialized_tensors_threshold;
    if (pyc_runtime_controller_observe(&controller, &rails, &m, &mode, &reason) != 0) return 1;
    if (mode != PYC_MODE_UTILIZATION_FIRST) return 2;
    if (controller.current_mode != PYC_MODE_UTILIZATION_FIRST) return 3;
    if (controller.recommended_mode != PYC_MODE_MEMORY_FIRST) return 4;
    if (controller.recommendation_reason != PYC_ROLLBACK_PRESSURE) return 5;
    if (controller.rollback_count != 0) return 6;

    /* Repeating the same observation should be stable when adaptation is disabled. */
    if (pyc_runtime_controller_observe(&controller, &rails, &m, &mode, &reason) != 0) return 7;
    if (mode != PYC_MODE_UTILIZATION_FIRST) return 8;
    if (controller.recommended_mode != PYC_MODE_MEMORY_FIRST) return 9;
    if (controller.rollback_count != 0) return 10;

    /* Auto-switch should honor dwell, so the shadow recommendation can lead the applied mode. */
    pyc_runtime_controller_init(&controller, PYC_MODE_UTILIZATION_FIRST);
    rails.enable_auto_switch = 1;
    rails.enable_hard_rollback = 0;
    controller.steps_in_mode = 0;
    controller.cooldown_remaining = 0;
    if (pyc_runtime_controller_observe(&controller, &rails, &m, &mode, &reason) != 0) return 11;
    if (controller.recommended_mode != PYC_MODE_MEMORY_FIRST) return 12;
    if (mode != PYC_MODE_UTILIZATION_FIRST) return 13;
    if (controller.current_mode != PYC_MODE_UTILIZATION_FIRST) return 14;
    if (controller.recommendation_reason != PYC_ROLLBACK_PRESSURE) return 15;

    controller.steps_in_mode = rails.dwell_steps;
    controller.cooldown_remaining = 0;
    if (pyc_runtime_controller_observe(&controller, &rails, &m, &mode, &reason) != 0) return 16;
    if (mode != PYC_MODE_MEMORY_FIRST) return 17;
    if (controller.current_mode != PYC_MODE_MEMORY_FIRST) return 18;
    if (controller.recommended_mode != PYC_MODE_MEMORY_FIRST) return 19;

    /* Consecutive latency breaches should hard rollback to balanced. */
    pyc_runtime_controller_init(&controller, PYC_MODE_UTILIZATION_FIRST);
    rails.enable_auto_switch = 0;
    rails.enable_hard_rollback = 1;
    m.pressure_score = 0.0;
    m.pressure_events = 0;
    m.rematerialized_tensors = 0;
    m.observed_p95_ms = m.baseline_p95_ms * (1.0 + rails.latency_regression_threshold + 0.2);
    controller.cooldown_remaining = 0;
    controller.steps_in_mode = rails.dwell_steps;
    for (size_t i = 0; i < rails.consecutive_breach_windows; ++i) {
        if (pyc_runtime_controller_observe(&controller, &rails, &m, &mode, &reason) != 0) return 20;
    }
    if (mode != PYC_MODE_BALANCED) return 21;
    if (controller.current_mode != PYC_MODE_BALANCED) return 22;
    if (reason != PYC_ROLLBACK_LATENCY) return 23;
    if (controller.recommended_mode != PYC_MODE_BALANCED) return 24;
    if (controller.recommendation_reason != PYC_ROLLBACK_LATENCY) return 25;
    if (controller.rollback_count == 0) return 26;

    printf(
        "test_runtime_control: ok (rollbacks=%zu shadow=%d applied=%d)\n",
        controller.rollback_count,
        controller.recommended_mode,
        controller.current_mode);
    return 0;
}
