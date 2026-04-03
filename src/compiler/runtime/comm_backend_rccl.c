#include "pyc/collective_comm.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

typedef int rccl_result_t;
typedef int rccl_data_type_t;
typedef int rccl_red_op_t;
typedef void* rccl_comm_t;

typedef struct {
    int initialized;
    int transport_available;
    int strict;
    int allow_transport_calls;
    pyc_comm_handle_t self_marker;
    int barrier_scratch;
#if defined(_WIN32)
    HMODULE transport_handle;
#else
    void* transport_handle;
#endif
    rccl_result_t (*all_reduce)(const void*, void*, size_t, rccl_data_type_t, rccl_red_op_t, rccl_comm_t, void*);
    rccl_result_t (*all_gather)(const void*, void*, size_t, rccl_data_type_t, rccl_comm_t, void*);
    rccl_result_t (*reduce_scatter)(const void*, void*, size_t, rccl_data_type_t, rccl_red_op_t, rccl_comm_t, void*);
    rccl_result_t (*broadcast)(const void*, void*, size_t, rccl_data_type_t, int, rccl_comm_t, void*);
    rccl_result_t (*send)(const void*, size_t, rccl_data_type_t, int, rccl_comm_t, void*);
    rccl_result_t (*recv)(void*, size_t, rccl_data_type_t, int, rccl_comm_t, void*);
    rccl_result_t (*group_start)(void);
    rccl_result_t (*group_end)(void);
} pyc_rccl_backend_ctx;

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

static int load_required_symbols(pyc_rccl_backend_ctx* ctx) {
    if (!ctx || !ctx->transport_handle) {
        return 0;
    }
    /* RCCL mirrors NCCL symbol names. */
    ctx->all_reduce = (rccl_result_t (*)(const void*, void*, size_t, rccl_data_type_t, rccl_red_op_t, rccl_comm_t, void*))
        lookup_symbol(ctx->transport_handle, "ncclAllReduce");
    ctx->all_gather = (rccl_result_t (*)(const void*, void*, size_t, rccl_data_type_t, rccl_comm_t, void*))
        lookup_symbol(ctx->transport_handle, "ncclAllGather");
    ctx->reduce_scatter = (rccl_result_t (*)(const void*, void*, size_t, rccl_data_type_t, rccl_red_op_t, rccl_comm_t, void*))
        lookup_symbol(ctx->transport_handle, "ncclReduceScatter");
    ctx->broadcast = (rccl_result_t (*)(const void*, void*, size_t, rccl_data_type_t, int, rccl_comm_t, void*))
        lookup_symbol(ctx->transport_handle, "ncclBroadcast");
    ctx->group_start = (rccl_result_t (*)(void))lookup_symbol(ctx->transport_handle, "ncclGroupStart");
    ctx->group_end = (rccl_result_t (*)(void))lookup_symbol(ctx->transport_handle, "ncclGroupEnd");
    ctx->send = (rccl_result_t (*)(const void*, size_t, rccl_data_type_t, int, rccl_comm_t, void*))
        lookup_symbol(ctx->transport_handle, "ncclSend");
    ctx->recv = (rccl_result_t (*)(void*, size_t, rccl_data_type_t, int, rccl_comm_t, void*))
        lookup_symbol(ctx->transport_handle, "ncclRecv");
    return ctx->all_reduce && ctx->all_gather && ctx->reduce_scatter && ctx->broadcast;
}

static int try_load_rccl_transport(pyc_rccl_backend_ctx* ctx) {
    if (!ctx) {
        return 0;
    }
#if defined(_WIN32)
    {
        const char* candidates[] = {"rccl.dll", NULL};
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
        const char* candidates[] = {"librccl.so", "librccl.dylib", NULL};
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

static void close_rccl_transport(pyc_rccl_backend_ctx* ctx) {
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

static int map_dtype(pyc_dtype dtype, rccl_data_type_t* out_dtype) {
    if (!out_dtype) {
        return 0;
    }
    switch (dtype) {
        case PYC_DTYPE_I8: *out_dtype = 0; return 1;
        case PYC_DTYPE_I32: *out_dtype = 2; return 1;
        case PYC_DTYPE_F16: *out_dtype = 6; return 1;
        case PYC_DTYPE_F32: *out_dtype = 7; return 1;
        default: return 0;
    }
}

static int map_reduce_op(pyc_reduce_op op, rccl_red_op_t* out_op) {
    if (!out_op) {
        return 0;
    }
    switch (op) {
        case PYC_REDUCE_SUM: *out_op = 0; return 1;
        case PYC_REDUCE_PROD: *out_op = 1; return 1;
        case PYC_REDUCE_MAX: *out_op = 2; return 1;
        case PYC_REDUCE_MIN: *out_op = 3; return 1;
        case PYC_REDUCE_AVG: *out_op = 4; return 1;
        default: return 0;
    }
}

static pyc_comm_status rccl_validate_common(
    void* backend_ctx,
    pyc_comm_handle_t comm,
    size_t count,
    pyc_dtype dtype) {
    pyc_rccl_backend_ctx* ctx = (pyc_rccl_backend_ctx*)backend_ctx;
    if (!ctx || !ctx->initialized || !comm || count == 0) {
        return PYC_COMM_ERR_INVALID;
    }
    if (!map_dtype(dtype, &(rccl_data_type_t){0})) {
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
    return PYC_COMM_OK;
}

static pyc_comm_status map_rccl_result(rccl_result_t code) {
    return code == 0 ? PYC_COMM_OK : PYC_COMM_ERR_HARDWARE;
}

static pyc_comm_status rccl_all_reduce(
    void* backend_ctx,
    pyc_comm_handle_t comm,
    const void* send_buf,
    void* recv_buf,
    size_t count,
    pyc_dtype dtype,
    pyc_reduce_op op,
    void* stream) {
    pyc_rccl_backend_ctx* ctx = (pyc_rccl_backend_ctx*)backend_ctx;
    rccl_data_type_t rccl_dtype;
    rccl_red_op_t rccl_op;
    pyc_comm_status st = rccl_validate_common(backend_ctx, comm, count, dtype);
    if (st != PYC_COMM_OK || !send_buf || !recv_buf || !ctx || !ctx->all_reduce) {
        return st == PYC_COMM_OK ? PYC_COMM_ERR_HARDWARE : st;
    }
    if (!map_dtype(dtype, &rccl_dtype) || !map_reduce_op(op, &rccl_op)) {
        return PYC_COMM_ERR_INVALID;
    }
    return map_rccl_result(ctx->all_reduce(send_buf, recv_buf, count, rccl_dtype, rccl_op, (rccl_comm_t)comm, stream));
}

static pyc_comm_status rccl_all_gather(
    void* backend_ctx,
    pyc_comm_handle_t comm,
    const void* send_buf,
    void* recv_buf,
    size_t count,
    pyc_dtype dtype,
    void* stream) {
    pyc_rccl_backend_ctx* ctx = (pyc_rccl_backend_ctx*)backend_ctx;
    rccl_data_type_t rccl_dtype;
    pyc_comm_status st = rccl_validate_common(backend_ctx, comm, count, dtype);
    if (st != PYC_COMM_OK || !send_buf || !recv_buf || !ctx || !ctx->all_gather) {
        return st == PYC_COMM_OK ? PYC_COMM_ERR_HARDWARE : st;
    }
    if (!map_dtype(dtype, &rccl_dtype)) {
        return PYC_COMM_ERR_INVALID;
    }
    return map_rccl_result(ctx->all_gather(send_buf, recv_buf, count, rccl_dtype, (rccl_comm_t)comm, stream));
}

static pyc_comm_status rccl_reduce_scatter(
    void* backend_ctx,
    pyc_comm_handle_t comm,
    const void* send_buf,
    void* recv_buf,
    size_t count,
    pyc_dtype dtype,
    pyc_reduce_op op,
    void* stream) {
    pyc_rccl_backend_ctx* ctx = (pyc_rccl_backend_ctx*)backend_ctx;
    rccl_data_type_t rccl_dtype;
    rccl_red_op_t rccl_op;
    pyc_comm_status st = rccl_validate_common(backend_ctx, comm, count, dtype);
    if (st != PYC_COMM_OK || !send_buf || !recv_buf || !ctx || !ctx->reduce_scatter) {
        return st == PYC_COMM_OK ? PYC_COMM_ERR_HARDWARE : st;
    }
    if (!map_dtype(dtype, &rccl_dtype) || !map_reduce_op(op, &rccl_op)) {
        return PYC_COMM_ERR_INVALID;
    }
    return map_rccl_result(ctx->reduce_scatter(send_buf, recv_buf, count, rccl_dtype, rccl_op, (rccl_comm_t)comm, stream));
}

static pyc_comm_status rccl_broadcast(
    void* backend_ctx,
    pyc_comm_handle_t comm,
    const void* send_buf,
    void* recv_buf,
    size_t count,
    pyc_dtype dtype,
    int root_rank,
    void* stream) {
    pyc_rccl_backend_ctx* ctx = (pyc_rccl_backend_ctx*)backend_ctx;
    rccl_data_type_t rccl_dtype;
    pyc_comm_status st = rccl_validate_common(backend_ctx, comm, count, dtype);
    if (st != PYC_COMM_OK || !send_buf || !recv_buf || root_rank < 0 || !ctx || !ctx->broadcast) {
        return st == PYC_COMM_OK ? PYC_COMM_ERR_INVALID : st;
    }
    if (!map_dtype(dtype, &rccl_dtype)) {
        return PYC_COMM_ERR_INVALID;
    }
    return map_rccl_result(ctx->broadcast(send_buf, recv_buf, count, rccl_dtype, root_rank, (rccl_comm_t)comm, stream));
}

static pyc_comm_status rccl_send(
    void* backend_ctx,
    pyc_comm_handle_t comm,
    const void* send_buf,
    size_t count,
    pyc_dtype dtype,
    int peer_rank,
    void* stream) {
    pyc_rccl_backend_ctx* ctx = (pyc_rccl_backend_ctx*)backend_ctx;
    rccl_data_type_t rccl_dtype;
    pyc_comm_status st = rccl_validate_common(backend_ctx, comm, count, dtype);
    if (st != PYC_COMM_OK || !send_buf || peer_rank < 0 || !ctx || !ctx->send) {
        return st == PYC_COMM_OK ? PYC_COMM_ERR_HARDWARE : st;
    }
    if (!map_dtype(dtype, &rccl_dtype)) {
        return PYC_COMM_ERR_INVALID;
    }
    return map_rccl_result(ctx->send(send_buf, count, rccl_dtype, peer_rank, (rccl_comm_t)comm, stream));
}

static pyc_comm_status rccl_recv(
    void* backend_ctx,
    pyc_comm_handle_t comm,
    void* recv_buf,
    size_t count,
    pyc_dtype dtype,
    int peer_rank,
    void* stream) {
    pyc_rccl_backend_ctx* ctx = (pyc_rccl_backend_ctx*)backend_ctx;
    rccl_data_type_t rccl_dtype;
    pyc_comm_status st = rccl_validate_common(backend_ctx, comm, count, dtype);
    if (st != PYC_COMM_OK || !recv_buf || peer_rank < 0 || !ctx || !ctx->recv) {
        return st == PYC_COMM_OK ? PYC_COMM_ERR_HARDWARE : st;
    }
    if (!map_dtype(dtype, &rccl_dtype)) {
        return PYC_COMM_ERR_INVALID;
    }
    return map_rccl_result(ctx->recv(recv_buf, count, rccl_dtype, peer_rank, (rccl_comm_t)comm, stream));
}

static pyc_comm_status rccl_barrier(void* backend_ctx, pyc_comm_handle_t comm, void* stream) {
    pyc_rccl_backend_ctx* ctx = (pyc_rccl_backend_ctx*)backend_ctx;
    pyc_comm_status st = rccl_validate_common(backend_ctx, comm, 1, PYC_DTYPE_I32);
    if (st != PYC_COMM_OK || !ctx || !ctx->all_reduce) {
        return st == PYC_COMM_OK ? PYC_COMM_ERR_HARDWARE : st;
    }
    if (ctx->group_start && ctx->group_end) {
        if (ctx->group_start() != 0) {
            return PYC_COMM_ERR_HARDWARE;
        }
    }
    st = map_rccl_result(ctx->all_reduce(
        &ctx->barrier_scratch,
        &ctx->barrier_scratch,
        1,
        2,
        0,
        (rccl_comm_t)comm,
        stream));
    if (ctx->group_start && ctx->group_end) {
        if (ctx->group_end() != 0) {
            return PYC_COMM_ERR_HARDWARE;
        }
    }
    return st;
}

pyc_collective_comm* pyc_comm_backend_create(const char* config_json) {
    pyc_collective_comm* comm = (pyc_collective_comm*)calloc(1, sizeof(*comm));
    pyc_rccl_backend_ctx* ctx;
    if (!comm) {
        return NULL;
    }
    ctx = (pyc_rccl_backend_ctx*)calloc(1, sizeof(*ctx));
    if (!ctx) {
        free(comm);
        return NULL;
    }

    ctx->strict = parse_strict_mode(config_json);
    ctx->allow_transport_calls = parse_has_comm_handle(config_json);
    ctx->transport_available = try_load_rccl_transport(ctx);
    if (ctx->strict && !ctx->transport_available) {
        close_rccl_transport(ctx);
        free(ctx);
        free(comm);
        return NULL;
    }
    ctx->initialized = 1;
    ctx->self_marker = (pyc_comm_handle_t)comm;

    comm->backend_ctx = ctx;
    comm->all_reduce = rccl_all_reduce;
    comm->all_gather = rccl_all_gather;
    comm->reduce_scatter = rccl_reduce_scatter;
    comm->broadcast = rccl_broadcast;
    comm->send = rccl_send;
    comm->recv = rccl_recv;
    comm->barrier = rccl_barrier;
    return comm;
}

void pyc_comm_backend_destroy(pyc_collective_comm* comm) {
    pyc_rccl_backend_ctx* ctx;
    if (!comm) {
        return;
    }
    ctx = (pyc_rccl_backend_ctx*)comm->backend_ctx;
    close_rccl_transport(ctx);
    free(ctx);
    free(comm);
}
