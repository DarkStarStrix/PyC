#include "pyc/collective_comm.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if defined(__has_include)
#if __has_include(<mpi.h>)
#include <mpi.h>
#define PYC_HAS_MPI_HEADER 1
#else
#define PYC_HAS_MPI_HEADER 0
#endif
#else
#define PYC_HAS_MPI_HEADER 0
#endif

typedef struct {
    int initialized;
    int transport_available;
    int strict;
    int allow_transport_calls;
    pyc_comm_handle_t self_marker;
#if defined(_WIN32)
    HMODULE transport_handle;
#else
    void* transport_handle;
#endif
#if PYC_HAS_MPI_HEADER
    int (*allreduce)(const void*, void*, int, MPI_Datatype, MPI_Op, MPI_Comm);
    int (*allgather)(const void*, int, MPI_Datatype, void*, int, MPI_Datatype, MPI_Comm);
    int (*reduce_scatter_block)(const void*, void*, int, MPI_Datatype, MPI_Op, MPI_Comm);
    int (*bcast)(void*, int, MPI_Datatype, int, MPI_Comm);
    int (*send)(const void*, int, MPI_Datatype, int, int, MPI_Comm);
    int (*recv)(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status*);
    int (*barrier)(MPI_Comm);
#endif
} pyc_mpi_backend_ctx;

static int parse_strict_mode(const char* config_json) {
    if (!config_json) {
        return 0;
    }
    return strstr(config_json, "\"strict\":true") != NULL ||
        strstr(config_json, "\"strict\":1") != NULL;
}

static int parse_has_comm_handle(const char* config_json) {
    if (!config_json) {
        return 0;
    }
    return strstr(config_json, "\"comm_handle\"") != NULL;
}

static void* lookup_symbol(
#if defined(_WIN32)
    HMODULE handle,
#else
    void* handle,
#endif
    const char* symbol) {
#if defined(_WIN32)
    return (void*)GetProcAddress(handle, symbol);
#else
    return dlsym(handle, symbol);
#endif
}

#if PYC_HAS_MPI_HEADER
static int load_required_symbols(pyc_mpi_backend_ctx* ctx) {
    if (!ctx || !ctx->transport_handle) {
        return 0;
    }
    ctx->allreduce = (int (*)(const void*, void*, int, MPI_Datatype, MPI_Op, MPI_Comm))
        lookup_symbol(ctx->transport_handle, "MPI_Allreduce");
    ctx->allgather = (int (*)(const void*, int, MPI_Datatype, void*, int, MPI_Datatype, MPI_Comm))
        lookup_symbol(ctx->transport_handle, "MPI_Allgather");
    ctx->reduce_scatter_block = (int (*)(const void*, void*, int, MPI_Datatype, MPI_Op, MPI_Comm))
        lookup_symbol(ctx->transport_handle, "MPI_Reduce_scatter_block");
    ctx->bcast = (int (*)(void*, int, MPI_Datatype, int, MPI_Comm))
        lookup_symbol(ctx->transport_handle, "MPI_Bcast");
    ctx->send = (int (*)(const void*, int, MPI_Datatype, int, int, MPI_Comm))
        lookup_symbol(ctx->transport_handle, "MPI_Send");
    ctx->recv = (int (*)(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status*))
        lookup_symbol(ctx->transport_handle, "MPI_Recv");
    ctx->barrier = (int (*)(MPI_Comm))
        lookup_symbol(ctx->transport_handle, "MPI_Barrier");
    return ctx->allreduce && ctx->allgather && ctx->reduce_scatter_block && ctx->bcast && ctx->send && ctx->recv && ctx->barrier;
}
#else
static int load_required_symbols(pyc_mpi_backend_ctx* ctx) {
    (void)ctx;
    return 0;
}
#endif

static int try_load_mpi_transport(pyc_mpi_backend_ctx* ctx) {
    if (!ctx) {
        return 0;
    }
#if defined(_WIN32)
    {
        const char* candidates[] = {"mpi.dll", "msmpi.dll", NULL};
        size_t i;
        for (i = 0; candidates[i] != NULL; ++i) {
            HMODULE h = LoadLibraryA(candidates[i]);
            if (h) {
                ctx->transport_handle = h;
                return load_required_symbols(ctx);
            }
        }
    }
#else
    {
        const char* candidates[] = {"libmpi.so", "libmpi.dylib", NULL};
        size_t i;
        for (i = 0; candidates[i] != NULL; ++i) {
            void* h = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
            if (h) {
                ctx->transport_handle = h;
                return load_required_symbols(ctx);
            }
        }
    }
#endif
    return 0;
}

static void close_mpi_transport(pyc_mpi_backend_ctx* ctx) {
    if (!ctx) {
        return;
    }
#if defined(_WIN32)
    if (ctx->transport_handle) {
        FreeLibrary(ctx->transport_handle);
        ctx->transport_handle = NULL;
    }
#else
    if (ctx->transport_handle) {
        dlclose(ctx->transport_handle);
        ctx->transport_handle = NULL;
    }
#endif
}

#if PYC_HAS_MPI_HEADER
static int map_dtype(pyc_dtype dtype, MPI_Datatype* out_dtype) {
    if (!out_dtype) {
        return 0;
    }
    switch (dtype) {
        case PYC_DTYPE_F32: *out_dtype = MPI_FLOAT; return 1;
        case PYC_DTYPE_F16: return 0;
        case PYC_DTYPE_I32: *out_dtype = MPI_INT; return 1;
        case PYC_DTYPE_I8: *out_dtype = MPI_INT8_T; return 1;
        default: return 0;
    }
}

static int map_reduce_op(pyc_reduce_op op, MPI_Op* out_op) {
    if (!out_op) {
        return 0;
    }
    switch (op) {
        case PYC_REDUCE_SUM: *out_op = MPI_SUM; return 1;
        case PYC_REDUCE_PROD: *out_op = MPI_PROD; return 1;
        case PYC_REDUCE_MIN: *out_op = MPI_MIN; return 1;
        case PYC_REDUCE_MAX: *out_op = MPI_MAX; return 1;
        case PYC_REDUCE_AVG: return 0;
        default: return 0;
    }
}
#endif

static pyc_comm_status mpi_validate_common(
    void* backend_ctx,
    pyc_comm_handle_t comm,
    size_t count,
    pyc_dtype dtype) {
    pyc_mpi_backend_ctx* ctx = (pyc_mpi_backend_ctx*)backend_ctx;
    if (!ctx || !ctx->initialized || !comm || count == 0) {
        return PYC_COMM_ERR_INVALID;
    }
    if (count > (size_t)INT_MAX) {
        return PYC_COMM_ERR_INVALID;
    }
    if (comm == ctx->self_marker) {
        return PYC_COMM_ERR_INVALID;
    }
    if (!ctx->transport_available) {
        return PYC_COMM_ERR_HARDWARE;
    }
    if (!ctx->allow_transport_calls) {
        return PYC_COMM_ERR_HARDWARE;
    }
#if PYC_HAS_MPI_HEADER
    if (!map_dtype(dtype, &(MPI_Datatype){0})) {
        return PYC_COMM_ERR_INVALID;
    }
#else
    (void)dtype;
#endif
    return PYC_COMM_OK;
}

static pyc_comm_status map_mpi_result(int code) {
    return code == 0 ? PYC_COMM_OK : PYC_COMM_ERR_HARDWARE;
}

static pyc_comm_status mpi_all_reduce(
    void* backend_ctx,
    pyc_comm_handle_t comm,
    const void* send_buf,
    void* recv_buf,
    size_t count,
    pyc_dtype dtype,
    pyc_reduce_op op,
    void* stream) {
    pyc_comm_status st = mpi_validate_common(backend_ctx, comm, count, dtype);
    (void)stream;
#if PYC_HAS_MPI_HEADER
    {
        pyc_mpi_backend_ctx* ctx = (pyc_mpi_backend_ctx*)backend_ctx;
        MPI_Datatype mpi_dtype;
        MPI_Op mpi_op;
        if (st != PYC_COMM_OK || !send_buf || !recv_buf || !ctx || !ctx->allreduce) {
            return st == PYC_COMM_OK ? PYC_COMM_ERR_HARDWARE : st;
        }
        if (!map_dtype(dtype, &mpi_dtype) || !map_reduce_op(op, &mpi_op)) {
            return PYC_COMM_ERR_INVALID;
        }
        return map_mpi_result(ctx->allreduce(send_buf, recv_buf, (int)count, mpi_dtype, mpi_op, (MPI_Comm)comm));
    }
#else
    (void)send_buf; (void)recv_buf; (void)op;
    return st;
#endif
}

static pyc_comm_status mpi_all_gather(
    void* backend_ctx,
    pyc_comm_handle_t comm,
    const void* send_buf,
    void* recv_buf,
    size_t count,
    pyc_dtype dtype,
    void* stream) {
    pyc_comm_status st = mpi_validate_common(backend_ctx, comm, count, dtype);
    (void)stream;
#if PYC_HAS_MPI_HEADER
    {
        pyc_mpi_backend_ctx* ctx = (pyc_mpi_backend_ctx*)backend_ctx;
        MPI_Datatype mpi_dtype;
        if (st != PYC_COMM_OK || !send_buf || !recv_buf || !ctx || !ctx->allgather) {
            return st == PYC_COMM_OK ? PYC_COMM_ERR_HARDWARE : st;
        }
        if (!map_dtype(dtype, &mpi_dtype)) {
            return PYC_COMM_ERR_INVALID;
        }
        return map_mpi_result(ctx->allgather(send_buf, (int)count, mpi_dtype, recv_buf, (int)count, mpi_dtype, (MPI_Comm)comm));
    }
#else
    (void)send_buf; (void)recv_buf;
    return st;
#endif
}

static pyc_comm_status mpi_reduce_scatter(
    void* backend_ctx,
    pyc_comm_handle_t comm,
    const void* send_buf,
    void* recv_buf,
    size_t count,
    pyc_dtype dtype,
    pyc_reduce_op op,
    void* stream) {
    pyc_comm_status st = mpi_validate_common(backend_ctx, comm, count, dtype);
    (void)stream;
#if PYC_HAS_MPI_HEADER
    {
        pyc_mpi_backend_ctx* ctx = (pyc_mpi_backend_ctx*)backend_ctx;
        MPI_Datatype mpi_dtype;
        MPI_Op mpi_op;
        if (st != PYC_COMM_OK || !send_buf || !recv_buf || !ctx || !ctx->reduce_scatter_block) {
            return st == PYC_COMM_OK ? PYC_COMM_ERR_HARDWARE : st;
        }
        if (!map_dtype(dtype, &mpi_dtype) || !map_reduce_op(op, &mpi_op)) {
            return PYC_COMM_ERR_INVALID;
        }
        return map_mpi_result(ctx->reduce_scatter_block(send_buf, recv_buf, (int)count, mpi_dtype, mpi_op, (MPI_Comm)comm));
    }
#else
    (void)send_buf; (void)recv_buf; (void)op;
    return st;
#endif
}

static pyc_comm_status mpi_broadcast(
    void* backend_ctx,
    pyc_comm_handle_t comm,
    const void* send_buf,
    void* recv_buf,
    size_t count,
    pyc_dtype dtype,
    int root_rank,
    void* stream) {
    pyc_comm_status st = mpi_validate_common(backend_ctx, comm, count, dtype);
    (void)stream;
#if PYC_HAS_MPI_HEADER
    {
        pyc_mpi_backend_ctx* ctx = (pyc_mpi_backend_ctx*)backend_ctx;
        MPI_Datatype mpi_dtype;
        int rc;
        if (st != PYC_COMM_OK || !send_buf || !recv_buf || root_rank < 0 || !ctx || !ctx->bcast) {
            return st == PYC_COMM_OK ? PYC_COMM_ERR_INVALID : st;
        }
        if (!map_dtype(dtype, &mpi_dtype)) {
            return PYC_COMM_ERR_INVALID;
        }
        if (send_buf != recv_buf) {
            memcpy(recv_buf, send_buf, count);
        }
        rc = ctx->bcast(recv_buf, (int)count, mpi_dtype, root_rank, (MPI_Comm)comm);
        return map_mpi_result(rc);
    }
#else
    (void)send_buf; (void)recv_buf; (void)root_rank;
    return st;
#endif
}

static pyc_comm_status mpi_send(
    void* backend_ctx,
    pyc_comm_handle_t comm,
    const void* send_buf,
    size_t count,
    pyc_dtype dtype,
    int peer_rank,
    void* stream) {
    pyc_comm_status st = mpi_validate_common(backend_ctx, comm, count, dtype);
    (void)stream;
#if PYC_HAS_MPI_HEADER
    {
        pyc_mpi_backend_ctx* ctx = (pyc_mpi_backend_ctx*)backend_ctx;
        MPI_Datatype mpi_dtype;
        if (st != PYC_COMM_OK || !send_buf || peer_rank < 0 || !ctx || !ctx->send) {
            return st == PYC_COMM_OK ? PYC_COMM_ERR_INVALID : st;
        }
        if (!map_dtype(dtype, &mpi_dtype)) {
            return PYC_COMM_ERR_INVALID;
        }
        return map_mpi_result(ctx->send(send_buf, (int)count, mpi_dtype, peer_rank, 0, (MPI_Comm)comm));
    }
#else
    (void)send_buf; (void)peer_rank;
    return st;
#endif
}

static pyc_comm_status mpi_recv(
    void* backend_ctx,
    pyc_comm_handle_t comm,
    void* recv_buf,
    size_t count,
    pyc_dtype dtype,
    int peer_rank,
    void* stream) {
    pyc_comm_status st = mpi_validate_common(backend_ctx, comm, count, dtype);
    (void)stream;
#if PYC_HAS_MPI_HEADER
    {
        pyc_mpi_backend_ctx* ctx = (pyc_mpi_backend_ctx*)backend_ctx;
        MPI_Datatype mpi_dtype;
        MPI_Status status;
        if (st != PYC_COMM_OK || !recv_buf || peer_rank < 0 || !ctx || !ctx->recv) {
            return st == PYC_COMM_OK ? PYC_COMM_ERR_INVALID : st;
        }
        if (!map_dtype(dtype, &mpi_dtype)) {
            return PYC_COMM_ERR_INVALID;
        }
        return map_mpi_result(ctx->recv(recv_buf, (int)count, mpi_dtype, peer_rank, 0, (MPI_Comm)comm, &status));
    }
#else
    (void)recv_buf; (void)peer_rank;
    return st;
#endif
}

static pyc_comm_status mpi_barrier(void* backend_ctx, pyc_comm_handle_t comm, void* stream) {
    pyc_comm_status st = mpi_validate_common(backend_ctx, comm, 1, PYC_DTYPE_I32);
    (void)stream;
#if PYC_HAS_MPI_HEADER
    {
        pyc_mpi_backend_ctx* ctx = (pyc_mpi_backend_ctx*)backend_ctx;
        if (st != PYC_COMM_OK || !ctx || !ctx->barrier) {
            return st == PYC_COMM_OK ? PYC_COMM_ERR_HARDWARE : st;
        }
        return map_mpi_result(ctx->barrier((MPI_Comm)comm));
    }
#else
    return st;
#endif
}

pyc_collective_comm* pyc_comm_backend_create(const char* config_json) {
    pyc_collective_comm* comm = (pyc_collective_comm*)calloc(1, sizeof(*comm));
    pyc_mpi_backend_ctx* ctx;
    if (!comm) {
        return NULL;
    }
    ctx = (pyc_mpi_backend_ctx*)calloc(1, sizeof(*ctx));
    if (!ctx) {
        free(comm);
        return NULL;
    }

    ctx->strict = parse_strict_mode(config_json);
    ctx->allow_transport_calls = parse_has_comm_handle(config_json);
    ctx->transport_available = try_load_mpi_transport(ctx);
    if (ctx->strict && !ctx->transport_available) {
        close_mpi_transport(ctx);
        free(ctx);
        free(comm);
        return NULL;
    }
    ctx->initialized = 1;
    ctx->self_marker = (pyc_comm_handle_t)comm;

    comm->backend_ctx = ctx;
    comm->all_reduce = mpi_all_reduce;
    comm->all_gather = mpi_all_gather;
    comm->reduce_scatter = mpi_reduce_scatter;
    comm->broadcast = mpi_broadcast;
    comm->send = mpi_send;
    comm->recv = mpi_recv;
    comm->barrier = mpi_barrier;
    return comm;
}

void pyc_comm_backend_destroy(pyc_collective_comm* comm) {
    pyc_mpi_backend_ctx* ctx;
    if (!comm) {
        return;
    }
    ctx = (pyc_mpi_backend_ctx*)comm->backend_ctx;
    close_mpi_transport(ctx);
    free(ctx);
    free(comm);
}
