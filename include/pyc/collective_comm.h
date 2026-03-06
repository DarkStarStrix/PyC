#ifndef PYC_COLLECTIVE_COMM_H
#define PYC_COLLECTIVE_COMM_H

#include <stddef.h>
#include <stdint.h>

#include "pyc/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PYC_REDUCE_SUM = 0,
    PYC_REDUCE_PROD = 1,
    PYC_REDUCE_MIN = 2,
    PYC_REDUCE_MAX = 3,
    PYC_REDUCE_AVG = 4
} pyc_reduce_op;

typedef enum {
    PYC_COMM_OK = 0,
    PYC_COMM_ERR_TIMEOUT = 1,
    PYC_COMM_ERR_HARDWARE = 2,
    PYC_COMM_ERR_INVALID = 3
} pyc_comm_status;

typedef void* pyc_comm_handle_t;

typedef struct pyc_collective_comm {
    void* backend_ctx;

    pyc_comm_status (*all_reduce)(
        void* backend_ctx,
        pyc_comm_handle_t comm,
        const void* send_buf,
        void* recv_buf,
        size_t count,
        pyc_dtype dtype,
        pyc_reduce_op op,
        void* stream
    );

    pyc_comm_status (*all_gather)(
        void* backend_ctx,
        pyc_comm_handle_t comm,
        const void* send_buf,
        void* recv_buf,
        size_t count,
        pyc_dtype dtype,
        void* stream
    );

    pyc_comm_status (*reduce_scatter)(
        void* backend_ctx,
        pyc_comm_handle_t comm,
        const void* send_buf,
        void* recv_buf,
        size_t count,
        pyc_dtype dtype,
        pyc_reduce_op op,
        void* stream
    );

    pyc_comm_status (*broadcast)(
        void* backend_ctx,
        pyc_comm_handle_t comm,
        const void* send_buf,
        void* recv_buf,
        size_t count,
        pyc_dtype dtype,
        int root_rank,
        void* stream
    );

    pyc_comm_status (*send)(
        void* backend_ctx,
        pyc_comm_handle_t comm,
        const void* send_buf,
        size_t count,
        pyc_dtype dtype,
        int peer_rank,
        void* stream
    );

    pyc_comm_status (*recv)(
        void* backend_ctx,
        pyc_comm_handle_t comm,
        void* recv_buf,
        size_t count,
        pyc_dtype dtype,
        int peer_rank,
        void* stream
    );

    pyc_comm_status (*barrier)(
        void* backend_ctx,
        pyc_comm_handle_t comm,
        void* stream
    );
} pyc_collective_comm;

typedef pyc_collective_comm* (*pyc_comm_backend_create_fn)(const char* config_json);
typedef void (*pyc_comm_backend_destroy_fn)(pyc_collective_comm* comm);

/* Backend-shared library exported symbol names. */
#define PYC_COMM_BACKEND_CREATE_SYMBOL "pyc_comm_backend_create"
#define PYC_COMM_BACKEND_DESTROY_SYMBOL "pyc_comm_backend_destroy"

/* Expected backend entrypoints for dynamically loaded comm backends. */
pyc_collective_comm* pyc_comm_backend_create(const char* config_json);
void pyc_comm_backend_destroy(pyc_collective_comm* comm);

/* Cross-platform shared-library loader helpers. */
pyc_collective_comm* pyc_load_comm_backend(const char* backend_path, const char* config_json);
void pyc_unload_comm_backend(pyc_collective_comm* comm);
const char* pyc_comm_loader_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
