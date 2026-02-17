#include <stdio.h>

#include "pyc/ai_bridge.h"

int main(void) {
    pyc_policy_contract contract;
    pyc_compile_options options;

    pyc_ai_default_policy_contract(&contract);
    if (contract.mode != PYC_MODE_BALANCED) return 1;
    if (contract.target_utilization_floor <= 0.0) return 2;
    if (contract.deterministic_strict != 1) return 3;

    options.objective_mode = PYC_MODE_BALANCED;
    options.memory_budget_bytes = 0;
    options.target_utilization_floor = 0.0;
    options.deterministic_strict = 0;

    contract.mode = PYC_MODE_MEMORY_FIRST;
    contract.memory_budget_bytes = 4096;
    contract.target_utilization_floor = 0.85;
    contract.deterministic_strict = 1;

    if (pyc_ai_apply_policy_contract(&options, &contract) != 0) return 4;
    if (options.objective_mode != PYC_MODE_MEMORY_FIRST) return 5;
    if (options.memory_budget_bytes != 4096) return 6;
    if (options.target_utilization_floor != 0.85) return 7;
    if (options.deterministic_strict != 1) return 8;

    printf("test_ai_bridge: ok\n");
    return 0;
}
