#ifndef PYC_DISTRIBUTED_RUNTIME_H
#define PYC_DISTRIBUTED_RUNTIME_H

#include "pyc/collective_comm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pyc_distributed_runtime pyc_distributed_runtime;

pyc_distributed_runtime* pyc_distributed_runtime_init(
    const char* backend_path,
    const char* config_json,
    int world_size,
    int rank,
    int local_rank);

void pyc_distributed_runtime_destroy(pyc_distributed_runtime* runtime);

pyc_collective_comm* pyc_distributed_runtime_comm(pyc_distributed_runtime* runtime);
const pyc_collective_comm* pyc_distributed_runtime_comm_const(const pyc_distributed_runtime* runtime);
pyc_comm_handle_t pyc_distributed_runtime_handle(const pyc_distributed_runtime* runtime);

#ifdef __cplusplus
}
#endif

#endif
