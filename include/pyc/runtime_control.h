#ifndef PYC_RUNTIME_CONTROL_H
#define PYC_RUNTIME_CONTROL_H

#include <stddef.h>

#include "pyc/optimizer_policy.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PYC_ROLLBACK_NONE = 0,
    PYC_ROLLBACK_LATENCY = 1,
    PYC_ROLLBACK_THROUGHPUT = 2,
    PYC_ROLLBACK_PRESSURE = 3,
    PYC_ROLLBACK_RUNTIME_ERROR = 4
} pyc_rollback_reason;

typedef struct {
    int enable_auto_switch;
    int enable_hard_rollback;
    size_t dwell_steps;
    size_t cooldown_steps;
    size_t consecutive_breach_windows;
    double latency_regression_threshold;
    double throughput_regression_threshold;
    double pressure_score_threshold;
    size_t pressure_events_threshold;
    size_t rematerialized_tensors_threshold;
} pyc_runtime_rails;

typedef struct {
    double baseline_p95_ms;
    double observed_p95_ms;
    double baseline_throughput;
    double observed_throughput;
    double pressure_score;
    size_t pressure_events;
    size_t rematerialized_tensors;
    int runtime_error;
} pyc_runtime_window_metrics;

typedef struct {
    pyc_objective_mode current_mode;
    pyc_objective_mode last_stable_mode;
    pyc_rollback_reason last_rollback_reason;
    size_t rollback_count;
    size_t steps_in_mode;
    size_t cooldown_remaining;
    size_t consecutive_breaches;
} pyc_runtime_controller;

void pyc_runtime_rails_default(pyc_runtime_rails* rails);
void pyc_runtime_controller_init(pyc_runtime_controller* controller, pyc_objective_mode initial_mode);
int pyc_runtime_controller_observe(
    pyc_runtime_controller* controller,
    const pyc_runtime_rails* rails,
    const pyc_runtime_window_metrics* metrics,
    pyc_objective_mode* out_mode,
    pyc_rollback_reason* out_rollback_reason);

#ifdef __cplusplus
}
#endif

#endif
