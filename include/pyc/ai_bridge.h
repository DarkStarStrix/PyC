#ifndef PYC_AI_BRIDGE_H
#define PYC_AI_BRIDGE_H

#include "pyc/compiler_api.h"
#include "pyc/optimizer_policy.h"

#ifdef __cplusplus
extern "C" {
#endif

void pyc_ai_default_policy_contract(pyc_policy_contract* contract);
int pyc_ai_apply_policy_contract(pyc_compile_options* options, const pyc_policy_contract* contract);

#ifdef __cplusplus
}
#endif

#endif
