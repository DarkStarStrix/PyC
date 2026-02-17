#ifndef PYC_OPTIMIZER_POLICY_H
#define PYC_OPTIMIZER_POLICY_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PYC_MODE_BALANCED = 0,
    PYC_MODE_MEMORY_FIRST = 1,
    PYC_MODE_UTILIZATION_FIRST = 2
} pyc_objective_mode;

typedef struct {
    pyc_objective_mode mode;
    size_t memory_budget_bytes;
    double target_utilization_floor;
    int deterministic_strict;
} pyc_policy_contract;

#ifdef __cplusplus
}
#endif

#endif
