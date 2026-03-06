#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "pyc/distributed_runtime.h"

static double now_ms(void) {
    return ((double)clock() * 1000.0) / (double)CLOCKS_PER_SEC;
}

static const char* status_name(pyc_comm_status st) {
    switch (st) {
        case PYC_COMM_OK: return "ok";
        case PYC_COMM_ERR_TIMEOUT: return "timeout";
        case PYC_COMM_ERR_HARDWARE: return "hardware";
        case PYC_COMM_ERR_INVALID: return "invalid";
        default: return "unknown";
    }
}

int main(int argc, char** argv) {
    const char* backend = "unknown";
    const char* backend_path = NULL;
    const char* config_json = "{\"strict\":false}";
    int iters = 5000;
    int count = 1024;
    int i;
    pyc_distributed_runtime* rt;
    pyc_collective_comm* comm;
    pyc_comm_handle_t handle;
    float* send_buf;
    float* recv_buf;
    int ok_count = 0;
    int hardware_count = 0;
    int invalid_count = 0;
    int timeout_count = 0;
    pyc_comm_status last_status = PYC_COMM_OK;
    double start_ms;
    double end_ms;
    double total_ms;
    double per_iter_us;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--backend") == 0 && (i + 1) < argc) {
            backend = argv[++i];
        } else if (strcmp(argv[i], "--backend-path") == 0 && (i + 1) < argc) {
            backend_path = argv[++i];
        } else if (strcmp(argv[i], "--iters") == 0 && (i + 1) < argc) {
            iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--count") == 0 && (i + 1) < argc) {
            count = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--config-json") == 0 && (i + 1) < argc) {
            config_json = argv[++i];
        }
    }

    if (!backend_path || backend_path[0] == '\0' || iters <= 0 || count <= 0) {
        fprintf(stderr, "{\"status\":\"error\",\"error\":\"invalid_args\"}\n");
        return 2;
    }

    rt = pyc_distributed_runtime_init(backend_path, config_json, 1, 0, 0);
    if (!rt) {
        fprintf(stderr, "{\"status\":\"error\",\"error\":\"runtime_init_failed\"}\n");
        return 3;
    }
    comm = pyc_distributed_runtime_comm(rt);
    handle = pyc_distributed_runtime_handle(rt);
    if (!comm || !handle) {
        pyc_distributed_runtime_destroy(rt);
        fprintf(stderr, "{\"status\":\"error\",\"error\":\"runtime_missing_comm_or_handle\"}\n");
        return 4;
    }

    send_buf = (float*)malloc((size_t)count * sizeof(float));
    recv_buf = (float*)malloc((size_t)count * sizeof(float));
    if (!send_buf || !recv_buf) {
        free(send_buf);
        free(recv_buf);
        pyc_distributed_runtime_destroy(rt);
        fprintf(stderr, "{\"status\":\"error\",\"error\":\"oom\"}\n");
        return 5;
    }
    for (i = 0; i < count; ++i) {
        send_buf[i] = (float)(i + 1);
        recv_buf[i] = 0.0f;
    }

    start_ms = now_ms();
    for (i = 0; i < iters; ++i) {
        pyc_comm_status st = comm->all_reduce(
            comm->backend_ctx,
            handle,
            send_buf,
            recv_buf,
            (size_t)count,
            PYC_DTYPE_F32,
            PYC_REDUCE_SUM,
            NULL);
        last_status = st;
        switch (st) {
            case PYC_COMM_OK: ok_count++; break;
            case PYC_COMM_ERR_HARDWARE: hardware_count++; break;
            case PYC_COMM_ERR_INVALID: invalid_count++; break;
            case PYC_COMM_ERR_TIMEOUT: timeout_count++; break;
            default: invalid_count++; break;
        }
    }
    end_ms = now_ms();
    total_ms = end_ms - start_ms;
    per_iter_us = (iters > 0) ? ((total_ms * 1000.0) / (double)iters) : 0.0;

    printf("{\"status\":\"ok\",\"backend\":\"%s\",\"iters\":%d,\"count\":%d,"
           "\"total_ms\":%.6f,\"per_iter_us\":%.6f,"
           "\"ok\":%d,\"hardware\":%d,\"invalid\":%d,\"timeout\":%d,"
           "\"last_status\":\"%s\"}\n",
           backend,
           iters,
           count,
           total_ms,
           per_iter_us,
           ok_count,
           hardware_count,
           invalid_count,
           timeout_count,
           status_name(last_status));

    free(send_buf);
    free(recv_buf);
    pyc_distributed_runtime_destroy(rt);
    return 0;
}
