#include "pyc/cuda_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(PYC_HAVE_CUDA_RUNTIME)
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#endif

static int env_true(const char* name) {
    const char* value = getenv(name);
    if (!value) {
        return 0;
    }
    return strcmp(value, "1") == 0 || strcmp(value, "true") == 0 || strcmp(value, "TRUE") == 0;
}

void pyc_cuda_dispatch_trace_init(pyc_cuda_dispatch_trace* trace) {
    if (!trace) {
        return;
    }
    memset(trace, 0, sizeof(*trace));
    strcpy(trace->reason, "not_requested");
}

static int detect_cuda_available(char* reason, size_t reason_size) {
    if (env_true("PYC_CUDA_DISABLE")) {
        strncpy(reason, "disabled_by_env", reason_size - 1);
        return 0;
    }
    if (env_true("PYC_CUDA_SIMULATE_AVAILABLE")) {
        strncpy(reason, "cuda_simulated_available", reason_size - 1);
        return 1;
    }
#if defined(PYC_HAVE_CUDA_RUNTIME)
    {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err == cudaSuccess && device_count > 0) {
            strncpy(reason, "cuda_runtime_detected", reason_size - 1);
            return 1;
        }
        if (err != cudaSuccess) {
            snprintf(reason, reason_size, "cuda_runtime_error_%d", (int)err);
        } else {
            strncpy(reason, "cuda_runtime_no_device", reason_size - 1);
        }
        return 0;
    }
#else
    strncpy(reason, "cuda_runtime_not_built", reason_size - 1);
    return 0;
#endif
}

static size_t shape_elements(const pyc_shape* shape) {
    size_t i;
    size_t total = 1;
    if (!shape || shape->rank == 0) {
        return 0;
    }
    for (i = 0; i < shape->rank; ++i) {
        if (shape->dims[i] <= 0) {
            return 0;
        }
        total *= (size_t)shape->dims[i];
    }
    return total;
}

static int native_cuda_supported_kind(pyc_ir_op_kind kind) {
    return kind == PYC_IR_OP_INPUT ||
           kind == PYC_IR_OP_MATMUL ||
           kind == PYC_IR_OP_ADD ||
           kind == PYC_IR_OP_RELU ||
           kind == PYC_IR_OP_OUTPUT;
}

static int native_cuda_can_execute(const pyc_ir_module* module) {
    size_t i;
    if (!module || module->op_count == 0) {
        return 0;
    }
    for (i = 0; i < module->op_count; ++i) {
        const pyc_ir_op* op = &module->ops[i];
        if (!native_cuda_supported_kind(op->kind)) {
            return 0;
        }
        if (op->dtype != PYC_DTYPE_F32) {
            return 0;
        }
    }
    return 1;
}

static int parse_single_matmul_graph(
    const pyc_ir_module* module,
    int* out_lhs_id,
    int* out_rhs_id,
    int* out_matmul_id,
    int* out_output_id,
    size_t* out_m,
    size_t* out_k,
    size_t* out_n) {
    int matmul_id = -1;
    int output_id = -1;
    size_t i;

    if (!module || !out_lhs_id || !out_rhs_id || !out_matmul_id ||
        !out_output_id || !out_m || !out_k || !out_n) {
        return -1;
    }

    for (i = 0; i < module->op_count; ++i) {
        const pyc_ir_op* op = &module->ops[i];
        switch (op->kind) {
            case PYC_IR_OP_INPUT:
            case PYC_IR_OP_MATMUL:
            case PYC_IR_OP_OUTPUT:
                break;
            default:
                return -1;
        }
        if (op->kind == PYC_IR_OP_MATMUL) {
            if (matmul_id >= 0) {
                return -1;
            }
            matmul_id = (int)i;
        }
        if (op->kind == PYC_IR_OP_OUTPUT) {
            if (output_id >= 0) {
                return -1;
            }
            output_id = (int)i;
        }
    }

    if (matmul_id < 0 || output_id < 0) {
        return -1;
    }

    {
        const pyc_ir_op* matmul = &module->ops[(size_t)matmul_id];
        const pyc_ir_op* output = &module->ops[(size_t)output_id];
        int lhs_id;
        int rhs_id;
        const pyc_ir_op* lhs;
        const pyc_ir_op* rhs;

        if (matmul->input_count < 2 || output->input_count < 1) {
            return -1;
        }
        lhs_id = matmul->input_ids[0];
        rhs_id = matmul->input_ids[1];
        if (lhs_id < 0 || rhs_id < 0 ||
            (size_t)lhs_id >= module->op_count ||
            (size_t)rhs_id >= module->op_count) {
            return -1;
        }
        if (output->input_ids[0] != matmul_id) {
            return -1;
        }
        lhs = &module->ops[(size_t)lhs_id];
        rhs = &module->ops[(size_t)rhs_id];
        if (lhs->kind != PYC_IR_OP_INPUT || rhs->kind != PYC_IR_OP_INPUT) {
            return -1;
        }
        if (lhs->shape.rank != 2 || rhs->shape.rank != 2) {
            return -1;
        }
        if (lhs->shape.dims[1] != rhs->shape.dims[0]) {
            return -1;
        }

        *out_lhs_id = lhs_id;
        *out_rhs_id = rhs_id;
        *out_matmul_id = matmul_id;
        *out_output_id = output_id;
        *out_m = (size_t)lhs->shape.dims[0];
        *out_k = (size_t)lhs->shape.dims[1];
        *out_n = (size_t)rhs->shape.dims[1];
    }

    return 0;
}

static void matmul_f32_naive(
    const float* lhs,
    const float* rhs,
    float* out,
    size_t m,
    size_t k,
    size_t n) {
    size_t r;
    for (r = 0; r < m; ++r) {
        size_t c;
        for (c = 0; c < n; ++c) {
            size_t t;
            float sum = 0.0f;
            for (t = 0; t < k; ++t) {
                sum += lhs[r * k + t] * rhs[t * n + c];
            }
            out[r * n + c] = sum;
        }
    }
}

static int execute_native_cuda_graph_simulated(
    const pyc_ir_module* module,
    const pyc_tensor* inputs,
    size_t input_count,
    pyc_tensor* outputs,
    size_t output_count) {
    const float* read_ptrs[PYC_IR_MAX_OPS];
    float* owned_ptrs[PYC_IR_MAX_OPS];
    size_t elem_count[PYC_IR_MAX_OPS];
    size_t input_index = 0;
    size_t output_index = 0;
    size_t i;
    int failed = 0;

    memset(read_ptrs, 0, sizeof(read_ptrs));
    memset(owned_ptrs, 0, sizeof(owned_ptrs));
    memset(elem_count, 0, sizeof(elem_count));

    for (i = 0; i < module->op_count; ++i) {
        const pyc_ir_op* op = &module->ops[i];
        elem_count[i] = shape_elements(&op->shape);

        switch (op->kind) {
            case PYC_IR_OP_INPUT: {
                size_t required = elem_count[i] * sizeof(float);
                if (input_index >= input_count || !inputs[input_index].data) {
                    failed = 1;
                    goto cleanup;
                }
                if (required == 0 || inputs[input_index].size_bytes < required) {
                    failed = 1;
                    goto cleanup;
                }
                read_ptrs[i] = (const float*)inputs[input_index].data;
                input_index++;
                break;
            }
            case PYC_IR_OP_MATMUL: {
                int lhs_id;
                int rhs_id;
                const pyc_ir_op* lhs_op;
                const pyc_ir_op* rhs_op;
                size_t m;
                size_t k;
                size_t n;
                float* out;
                if (op->input_count < 2) {
                    failed = 1;
                    goto cleanup;
                }
                lhs_id = op->input_ids[0];
                rhs_id = op->input_ids[1];
                if (lhs_id < 0 || rhs_id < 0 ||
                    (size_t)lhs_id >= module->op_count ||
                    (size_t)rhs_id >= module->op_count) {
                    failed = 1;
                    goto cleanup;
                }
                lhs_op = &module->ops[(size_t)lhs_id];
                rhs_op = &module->ops[(size_t)rhs_id];
                if (lhs_op->shape.rank != 2 || rhs_op->shape.rank != 2) {
                    failed = 1;
                    goto cleanup;
                }
                m = (size_t)lhs_op->shape.dims[0];
                k = (size_t)lhs_op->shape.dims[1];
                n = (size_t)rhs_op->shape.dims[1];
                if (k != (size_t)rhs_op->shape.dims[0]) {
                    failed = 1;
                    goto cleanup;
                }
                out = (float*)malloc(m * n * sizeof(float));
                if (!out) {
                    failed = 1;
                    goto cleanup;
                }
                matmul_f32_naive(
                    read_ptrs[(size_t)lhs_id],
                    read_ptrs[(size_t)rhs_id],
                    out,
                    m,
                    k,
                    n);
                owned_ptrs[i] = out;
                read_ptrs[i] = out;
                break;
            }
            case PYC_IR_OP_ADD: {
                int a_id;
                int b_id;
                float* out;
                size_t j;
                if (op->input_count < 2 || elem_count[i] == 0) {
                    failed = 1;
                    goto cleanup;
                }
                a_id = op->input_ids[0];
                b_id = op->input_ids[1];
                if (a_id < 0 || b_id < 0 ||
                    (size_t)a_id >= module->op_count ||
                    (size_t)b_id >= module->op_count) {
                    failed = 1;
                    goto cleanup;
                }
                out = (float*)malloc(elem_count[i] * sizeof(float));
                if (!out) {
                    failed = 1;
                    goto cleanup;
                }
                for (j = 0; j < elem_count[i]; ++j) {
                    out[j] =
                        read_ptrs[(size_t)a_id][j] + read_ptrs[(size_t)b_id][j];
                }
                owned_ptrs[i] = out;
                read_ptrs[i] = out;
                break;
            }
            case PYC_IR_OP_RELU: {
                int src_id;
                float* out;
                size_t j;
                if (op->input_count < 1 || elem_count[i] == 0) {
                    failed = 1;
                    goto cleanup;
                }
                src_id = op->input_ids[0];
                if (src_id < 0 || (size_t)src_id >= module->op_count) {
                    failed = 1;
                    goto cleanup;
                }
                out = (float*)malloc(elem_count[i] * sizeof(float));
                if (!out) {
                    failed = 1;
                    goto cleanup;
                }
                for (j = 0; j < elem_count[i]; ++j) {
                    float v = read_ptrs[(size_t)src_id][j];
                    out[j] = v > 0.0f ? v : 0.0f;
                }
                owned_ptrs[i] = out;
                read_ptrs[i] = out;
                break;
            }
            case PYC_IR_OP_OUTPUT: {
                int src_id;
                size_t required;
                if (output_index >= output_count || !outputs[output_index].data) {
                    failed = 1;
                    goto cleanup;
                }
                if (op->input_count < 1) {
                    failed = 1;
                    goto cleanup;
                }
                src_id = op->input_ids[0];
                if (src_id < 0 || (size_t)src_id >= module->op_count) {
                    failed = 1;
                    goto cleanup;
                }
                required = elem_count[(size_t)src_id] * sizeof(float);
                if (required == 0 || outputs[output_index].size_bytes < required) {
                    failed = 1;
                    goto cleanup;
                }
                memcpy(outputs[output_index].data, read_ptrs[(size_t)src_id], required);
                output_index++;
                break;
            }
            default:
                failed = 1;
                goto cleanup;
        }
    }

cleanup:
    for (i = 0; i < module->op_count; ++i) {
        free(owned_ptrs[i]);
    }
    return failed ? -1 : 0;
}

#if defined(PYC_HAVE_CUDA_RUNTIME)
static int execute_native_cuda_graph_cuda(
    const pyc_ir_module* module,
    const pyc_tensor* inputs,
    size_t input_count,
    pyc_tensor* outputs,
    size_t output_count) {
    int lhs_id;
    int rhs_id;
    int matmul_id;
    int output_id;
    size_t m;
    size_t k;
    size_t n;
    int lhs_input_index = -1;
    int rhs_input_index = -1;
    int target_output_index = -1;
    size_t input_seen = 0;
    size_t output_seen = 0;
    size_t op_idx;
    const float* host_a;
    const float* host_b;
    float* host_out;
    size_t a_bytes;
    size_t b_bytes;
    size_t c_bytes;
    float* dev_a = NULL;
    float* dev_b = NULL;
    float* dev_c = NULL;
    cublasHandle_t handle = NULL;
    cublasStatus_t cublas_status;
    cudaError_t cuda_status;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int ok = 0;

    if (parse_single_matmul_graph(module, &lhs_id, &rhs_id, &matmul_id, &output_id, &m, &k, &n) != 0) {
        return -1;
    }
    (void)matmul_id;

    for (op_idx = 0; op_idx < module->op_count; ++op_idx) {
        const pyc_ir_op* op = &module->ops[op_idx];
        if (op->kind == PYC_IR_OP_INPUT) {
            if ((int)op_idx == lhs_id) {
                lhs_input_index = (int)input_seen;
            }
            if ((int)op_idx == rhs_id) {
                rhs_input_index = (int)input_seen;
            }
            input_seen++;
        } else if (op->kind == PYC_IR_OP_OUTPUT) {
            if ((int)op_idx == output_id) {
                target_output_index = (int)output_seen;
            }
            output_seen++;
        }
    }

    if (lhs_input_index < 0 || rhs_input_index < 0 || target_output_index < 0) {
        return -1;
    }
    if ((size_t)lhs_input_index >= input_count ||
        (size_t)rhs_input_index >= input_count ||
        (size_t)target_output_index >= output_count) {
        return -1;
    }

    host_a = (const float*)inputs[(size_t)lhs_input_index].data;
    host_b = (const float*)inputs[(size_t)rhs_input_index].data;
    host_out = (float*)outputs[(size_t)target_output_index].data;
    if (!host_a || !host_b || !host_out) {
        return -1;
    }

    a_bytes = m * k * sizeof(float);
    b_bytes = k * n * sizeof(float);
    c_bytes = m * n * sizeof(float);
    if (inputs[(size_t)lhs_input_index].size_bytes < a_bytes ||
        inputs[(size_t)rhs_input_index].size_bytes < b_bytes ||
        outputs[(size_t)target_output_index].size_bytes < c_bytes) {
        return -1;
    }

    cuda_status = cudaMalloc((void**)&dev_a, a_bytes);
    if (cuda_status != cudaSuccess) goto cleanup;
    cuda_status = cudaMalloc((void**)&dev_b, b_bytes);
    if (cuda_status != cudaSuccess) goto cleanup;
    cuda_status = cudaMalloc((void**)&dev_c, c_bytes);
    if (cuda_status != cudaSuccess) goto cleanup;

    cuda_status = cudaMemcpy(dev_a, host_a, a_bytes, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) goto cleanup;
    cuda_status = cudaMemcpy(dev_b, host_b, b_bytes, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) goto cleanup;

    cublas_status = cublasCreate(&handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    /* Row-major C=A*B via column-major GEMM with swapped operands. */
    cublas_status = cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        (int)n,
        (int)m,
        (int)k,
        &alpha,
        dev_b,
        (int)n,
        dev_a,
        (int)k,
        &beta,
        dev_c,
        (int)n);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    cuda_status = cudaMemcpy(host_out, dev_c, c_bytes, cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) goto cleanup;

    if (module->ops[(size_t)output_id].input_ids[0] != matmul_id) {
        goto cleanup;
    }

    ok = 1;

cleanup:
    if (handle) {
        (void)cublasDestroy(handle);
    }
    if (dev_a) {
        (void)cudaFree(dev_a);
    }
    if (dev_b) {
        (void)cudaFree(dev_b);
    }
    if (dev_c) {
        (void)cudaFree(dev_c);
    }
    return ok ? 0 : -1;
}
#endif

pyc_cuda_dispatch_status pyc_cuda_dispatch(
    const pyc_ir_module* module,
    const pyc_tensor* inputs,
    size_t input_count,
    pyc_tensor* outputs,
    size_t output_count,
    pyc_cpu_executor_fn cpu_executor,
    void* executor_ctx,
    pyc_cuda_dispatch_trace* trace) {
    int available;
    char reason[PYC_CUDA_REASON_MAX];
    int run_status;

    if (!module || !inputs || !outputs || !cpu_executor) {
        return PYC_CUDA_DISPATCH_ERROR;
    }

    pyc_cuda_dispatch_trace_init(trace);
    if (trace) {
        trace->cuda_requested = 1;
    }

    if (env_true("PYC_CUDA_FORCE_ERROR")) {
        if (trace) {
            trace->cuda_available = 0;
            trace->fallback_to_cpu = 0;
            strcpy(trace->reason, "forced_error");
        }
        return PYC_CUDA_DISPATCH_ERROR;
    }

    memset(reason, 0, sizeof(reason));
    available = detect_cuda_available(reason, sizeof(reason));
    if (trace) {
        trace->cuda_available = available;
    }

    if (available && !env_true("PYC_CUDA_FORCE_FALLBACK") &&
        native_cuda_can_execute(module)) {
#if defined(PYC_HAVE_CUDA_RUNTIME)
        run_status = execute_native_cuda_graph_cuda(module, inputs, input_count, outputs, output_count);
        if (run_status == 0) {
            if (trace) {
                trace->fallback_to_cpu = 0;
                strncpy(trace->reason, "cuda_native_cublas", sizeof(trace->reason) - 1);
            }
            return PYC_CUDA_DISPATCH_OK;
        }
#endif
        if (env_true("PYC_CUDA_SIMULATE_AVAILABLE")) {
            run_status = execute_native_cuda_graph_simulated(module, inputs, input_count, outputs, output_count);
            if (run_status == 0) {
                if (trace) {
                    trace->fallback_to_cpu = 0;
                    strncpy(trace->reason, "cuda_native_simulated", sizeof(trace->reason) - 1);
                }
                return PYC_CUDA_DISPATCH_OK;
            }
        }
    }

    run_status = cpu_executor(module, inputs, input_count, outputs, output_count, executor_ctx);
    if (run_status != 0) {
        if (trace) {
            trace->fallback_to_cpu = 0;
            strncpy(trace->reason, "cpu_fallback_failed", sizeof(trace->reason) - 1);
        }
        return PYC_CUDA_DISPATCH_ERROR;
    }

    if (trace) {
        trace->fallback_to_cpu = 1;
        if (env_true("PYC_CUDA_FORCE_FALLBACK")) {
            strncpy(trace->reason, "cuda_forced_fallback", sizeof(trace->reason) - 1);
        } else if (available && !native_cuda_can_execute(module)) {
            strncpy(trace->reason, "cuda_native_unsupported_fallback", sizeof(trace->reason) - 1);
        } else if (available) {
            strncpy(trace->reason, "cuda_native_failed_fallback", sizeof(trace->reason) - 1);
        } else {
            strncpy(trace->reason, reason, sizeof(trace->reason) - 1);
        }
    }
    return PYC_CUDA_DISPATCH_FALLBACK;
}
