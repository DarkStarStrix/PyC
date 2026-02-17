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

    /* Warmup should settle to balanced after dwell windows without breach. */
    for (size_t i = 0; i < rails.dwell_steps + rails.cooldown_steps + 1; ++i) {
        if (pyc_runtime_controller_observe(&controller, &rails, &m, &mode, &reason) != 0) return 1;
    }
    if (mode != PYC_MODE_BALANCED) return 2;

    /* Pressure breaches should trigger memory_first before hard rollback threshold. */
    controller.consecutive_breaches = 0;
    controller.steps_in_mode = rails.dwell_steps;
    controller.cooldown_remaining = 0;
    m.pressure_score = rails.pressure_score_threshold + 1.0;
    m.pressure_events = rails.pressure_events_threshold;
    m.rematerialized_tensors = rails.rematerialized_tensors_threshold;
    if (pyc_runtime_controller_observe(&controller, &rails, &m, &mode, &reason) != 0) return 3;
    if (mode != PYC_MODE_MEMORY_FIRST) return 4;

    /* Consecutive latency breaches should hard rollback to balanced. */
    m.pressure_score = 0.0;
    m.pressure_events = 0;
    m.rematerialized_tensors = 0;
    m.observed_p95_ms = m.baseline_p95_ms * (1.0 + rails.latency_regression_threshold + 0.2);
    controller.cooldown_remaining = 0;
    controller.steps_in_mode = rails.dwell_steps;
    for (size_t i = 0; i < rails.consecutive_breach_windows; ++i) {
        if (pyc_runtime_controller_observe(&controller, &rails, &m, &mode, &reason) != 0) return 5;
    }
    if (mode != PYC_MODE_BALANCED) return 6;
    if (reason != PYC_ROLLBACK_LATENCY) return 7;
    if (controller.rollback_count == 0) return 8;

    printf("test_runtime_control: ok (rollbacks=%zu)\n", controller.rollback_count);
    return 0;
}
