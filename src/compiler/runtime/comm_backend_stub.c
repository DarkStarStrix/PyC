#include "pyc/collective_comm.h"

#include <stdlib.h>

typedef struct {
    int initialized;
} pyc_stub_backend_ctx;

static pyc_comm_status stub_validate_common(
    void* backend_ctx,
    pyc_comm_handle_t comm,
    size_t count,
    pyc_dtype dtype) {
    pyc_stub_backend_ctx* ctx = (pyc_stub_backend_ctx*)backend_ctx;
    if (!ctx || !ctx->initialized || !comm || count == 0) {
        return PYC_COMM_ERR_INVALID;
    }
    if (dtype < PYC_DTYPE_F32 || dtype > PYC_DTYPE_I8) {
        return PYC_COMM_ERR_INVALID;
    }
    return PYC_COMM_OK;
}

static pyc_comm_status stub_all_reduce(
    void* backend_ctx,
    pyc_comm_handle_t comm,
    const void* send_buf,
    void* recv_buf,
    size_t count,
    pyc_dtype dtype,
    pyc_reduce_op op,
    void* stream) {
    (void)stream;
    if (stub_validate_common(backend_ctx, comm, count, dtype) != PYC_COMM_OK || !send_buf || !recv_buf) {
        return PYC_COMM_ERR_INVALID;
    }
    if (op < PYC_REDUCE_SUM || op > PYC_REDUCE_AVG) {
        return PYC_COMM_ERR_INVALID;
    }
    return PYC_COMM_OK;
}

static pyc_comm_status stub_all_gather(
    void* backend_ctx,
    pyc_comm_handle_t comm,
    const void* send_buf,
    void* recv_buf,
    size_t count,
    pyc_dtype dtype,
    void* stream) {
    (void)stream;
    if (stub_validate_common(backend_ctx, comm, count, dtype) != PYC_COMM_OK || !send_buf || !recv_buf) {
        return PYC_COMM_ERR_INVALID;
    }
    return PYC_COMM_OK;
}

static pyc_comm_status stub_reduce_scatter(
    void* backend_ctx,
    pyc_comm_handle_t comm,
    const void* send_buf,
    void* recv_buf,
    size_t count,
    pyc_dtype dtype,
    pyc_reduce_op op,
    void* stream) {
    (void)stream;
    if (stub_validate_common(backend_ctx, comm, count, dtype) != PYC_COMM_OK || !send_buf || !recv_buf) {
        return PYC_COMM_ERR_INVALID;
    }
    if (op < PYC_REDUCE_SUM || op > PYC_REDUCE_AVG) {
        return PYC_COMM_ERR_INVALID;
    }
    return PYC_COMM_OK;
}

static pyc_comm_status stub_broadcast(
    void* backend_ctx,
    pyc_comm_handle_t comm,
    const void* send_buf,
    void* recv_buf,
    size_t count,
    pyc_dtype dtype,
    int root_rank,
    void* stream) {
    (void)stream;
    if (stub_validate_common(backend_ctx, comm, count, dtype) != PYC_COMM_OK || !send_buf || !recv_buf) {
        return PYC_COMM_ERR_INVALID;
    }
    if (root_rank < 0) {
        return PYC_COMM_ERR_INVALID;
    }
    return PYC_COMM_OK;
}

static pyc_comm_status stub_send(
    void* backend_ctx,
    pyc_comm_handle_t comm,
    const void* send_buf,
    size_t count,
    pyc_dtype dtype,
    int peer_rank,
    void* stream) {
    (void)stream;
    if (stub_validate_common(backend_ctx, comm, count, dtype) != PYC_COMM_OK || !send_buf) {
        return PYC_COMM_ERR_INVALID;
    }
    if (peer_rank < 0) {
        return PYC_COMM_ERR_INVALID;
    }
    return PYC_COMM_OK;
}

static pyc_comm_status stub_recv(
    void* backend_ctx,
    pyc_comm_handle_t comm,
    void* recv_buf,
    size_t count,
    pyc_dtype dtype,
    int peer_rank,
    void* stream) {
    (void)stream;
    if (stub_validate_common(backend_ctx, comm, count, dtype) != PYC_COMM_OK || !recv_buf) {
        return PYC_COMM_ERR_INVALID;
    }
    if (peer_rank < 0) {
        return PYC_COMM_ERR_INVALID;
    }
    return PYC_COMM_OK;
}

static pyc_comm_status stub_barrier(void* backend_ctx, pyc_comm_handle_t comm, void* stream) {
    (void)stream;
    if (stub_validate_common(backend_ctx, comm, 1, PYC_DTYPE_F32) != PYC_COMM_OK) {
        return PYC_COMM_ERR_INVALID;
    }
    return PYC_COMM_OK;
}

pyc_collective_comm* pyc_comm_backend_create(const char* config_json) {
    pyc_collective_comm* comm = NULL;
    pyc_stub_backend_ctx* ctx = NULL;
    (void)config_json;

    comm = (pyc_collective_comm*)calloc(1, sizeof(*comm));
    if (!comm) {
        return NULL;
    }
    ctx = (pyc_stub_backend_ctx*)calloc(1, sizeof(*ctx));
    if (!ctx) {
        free(comm);
        return NULL;
    }

    ctx->initialized = 1;
    comm->backend_ctx = ctx;
    comm->all_reduce = stub_all_reduce;
    comm->all_gather = stub_all_gather;
    comm->reduce_scatter = stub_reduce_scatter;
    comm->broadcast = stub_broadcast;
    comm->send = stub_send;
    comm->recv = stub_recv;
    comm->barrier = stub_barrier;
    return comm;
}

void pyc_comm_backend_destroy(pyc_collective_comm* comm) {
    if (!comm) {
        return;
    }
    free(comm->backend_ctx);
    free(comm);
}
