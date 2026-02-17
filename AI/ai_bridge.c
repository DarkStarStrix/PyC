#include "pyc/ai_bridge.h"

#include <string.h>

void pyc_ai_default_policy_contract(pyc_policy_contract* contract) {
    if (!contract) {
        return;
    }
    memset(contract, 0, sizeof(*contract));
    contract->mode = PYC_MODE_BALANCED;
    contract->memory_budget_bytes = 0;
    contract->target_utilization_floor = 0.70;
    contract->deterministic_strict = 1;
}

int pyc_ai_apply_policy_contract(pyc_compile_options* options, const pyc_policy_contract* contract) {
    if (!options || !contract) {
        return -1;
    }
    options->objective_mode = contract->mode;
    options->memory_budget_bytes = contract->memory_budget_bytes;
    options->target_utilization_floor = contract->target_utilization_floor;
    options->deterministic_strict = contract->deterministic_strict ? 1 : 0;
    return 0;
}
