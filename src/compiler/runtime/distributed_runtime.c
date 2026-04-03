#include "pyc/distributed_runtime.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct pyc_distributed_runtime {
    pyc_collective_comm* comm;
    pyc_comm_handle_t handle;
    int owns_handle_token;
    int world_size;
    int rank;
    int local_rank;
};

static pyc_comm_handle_t parse_comm_handle_from_json(const char* config_json) {
    const char* key;
    const char* colon;
    const char* start;
    char* end = NULL;
    unsigned long long value;
    if (!config_json) {
        return NULL;
    }
    key = strstr(config_json, "\"comm_handle\"");
    if (!key) {
        return NULL;
    }
    colon = strchr(key, ':');
    if (!colon) {
        return NULL;
    }
    start = colon + 1;
    while (*start == ' ' || *start == '\t' || *start == '"') {
        ++start;
    }
    value = strtoull(start, &end, 0);
    if (value == 0 || end == start) {
        return NULL;
    }
    return (pyc_comm_handle_t)(uintptr_t)value;
}

pyc_distributed_runtime* pyc_distributed_runtime_init(
    const char* backend_path,
    const char* config_json,
    int world_size,
    int rank,
    int local_rank) {
    pyc_distributed_runtime* runtime;
    if (!backend_path || backend_path[0] == '\0') {
        return NULL;
    }
    runtime = (pyc_distributed_runtime*)calloc(1, sizeof(*runtime));
    if (!runtime) {
        return NULL;
    }

    runtime->comm = pyc_load_comm_backend(backend_path, config_json);
    if (!runtime->comm) {
        free(runtime);
        return NULL;
    }

    runtime->world_size = world_size > 0 ? world_size : 1;
    runtime->rank = rank >= 0 ? rank : 0;
    runtime->local_rank = local_rank >= 0 ? local_rank : 0;
    if (runtime->rank >= runtime->world_size) {
        runtime->rank = 0;
    }

    runtime->handle = parse_comm_handle_from_json(config_json);
    if (!runtime->handle) {
        char* token = (char*)malloc(1);
        if (!token) {
            pyc_unload_comm_backend(runtime->comm);
            free(runtime);
            return NULL;
        }
        runtime->handle = (pyc_comm_handle_t)token;
        runtime->owns_handle_token = 1;
    }

    return runtime;
}

void pyc_distributed_runtime_destroy(pyc_distributed_runtime* runtime) {
    if (!runtime) {
        return;
    }
    if (runtime->comm) {
        pyc_unload_comm_backend(runtime->comm);
        runtime->comm = NULL;
    }
    if (runtime->owns_handle_token && runtime->handle) {
        free(runtime->handle);
        runtime->handle = NULL;
    }
    free(runtime);
}

pyc_collective_comm* pyc_distributed_runtime_comm(pyc_distributed_runtime* runtime) {
    if (!runtime) {
        return NULL;
    }
    return runtime->comm;
}

const pyc_collective_comm* pyc_distributed_runtime_comm_const(const pyc_distributed_runtime* runtime) {
    if (!runtime) {
        return NULL;
    }
    return runtime->comm;
}

pyc_comm_handle_t pyc_distributed_runtime_handle(const pyc_distributed_runtime* runtime) {
    if (!runtime) {
        return NULL;
    }
    return runtime->handle;
}
