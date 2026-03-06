#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "pyc/collective_comm.h"
#include "pyc/compiler_api.h"
#include "pyc/ir.h"
#include "pyc/kernel_registry.h"

#ifndef PYC_TEST_STUB_BACKEND_PATH
#define PYC_TEST_STUB_BACKEND_PATH ""
#endif

#ifndef PYC_TEST_INVALID_BACKEND_PATH
#define PYC_TEST_INVALID_BACKEND_PATH ""
#endif

#ifndef PYC_TEST_NCCL_BACKEND_PATH
#define PYC_TEST_NCCL_BACKEND_PATH ""
#endif

#ifndef PYC_TEST_RCCL_BACKEND_PATH
#define PYC_TEST_RCCL_BACKEND_PATH ""
#endif

#ifndef PYC_TEST_MPI_BACKEND_PATH
#define PYC_TEST_MPI_BACKEND_PATH ""
#endif

static int validate_backend_path(const char* path, int expected_error_code_base) {
    pyc_collective_comm* comm;
    pyc_comm_status st;
    int value = 1;
    pyc_comm_handle_t fake_comm = (pyc_comm_handle_t)(uintptr_t)0x1234;

    comm = pyc_load_comm_backend(path, "{\"strict\":false}");
    if (!comm) {
        fprintf(stderr, "expected backend to load in non-strict mode: %s (%s)\n", path, pyc_comm_loader_last_error());
        return expected_error_code_base;
    }
    st = comm->all_reduce(comm->backend_ctx, fake_comm, &value, &value, 1, PYC_DTYPE_I32, PYC_REDUCE_SUM, NULL);
    if (!(st == PYC_COMM_OK || st == PYC_COMM_ERR_HARDWARE)) {
        fprintf(stderr, "unexpected all_reduce status=%d for backend=%s\n", (int)st, path);
        pyc_unload_comm_backend(comm);
        return expected_error_code_base + 1;
    }
    pyc_unload_comm_backend(comm);
    return 0;
}

static void build_valid_module(pyc_ir_module* m) {
    pyc_ir_op in;
    pyc_ir_op out;

    pyc_ir_module_init(m);

    memset(&in, 0, sizeof(in));
    in.kind = PYC_IR_OP_INPUT;
    strcpy(in.name, "input0");
    in.dtype = PYC_DTYPE_F32;
    in.shape.rank = 1;
    in.shape.dims[0] = 4;
    pyc_ir_add_op(m, &in);

    memset(&out, 0, sizeof(out));
    out.kind = PYC_IR_OP_OUTPUT;
    strcpy(out.name, "output0");
    out.dtype = PYC_DTYPE_F32;
    out.shape.rank = 1;
    out.shape.dims[0] = 4;
    out.input_ids[0] = 0;
    out.input_count = 1;
    pyc_ir_add_op(m, &out);
}

int main(void) {
    pyc_collective_comm* comm;
    int value = 1;
    pyc_comm_status st;
    pyc_model_desc desc;
    pyc_compile_options options;
    pyc_compiled_model* model = NULL;
    pyc_ir_module module;
    pyc_status compile_status;
    pyc_kernel_desc kd;

    comm = pyc_load_comm_backend(PYC_TEST_STUB_BACKEND_PATH, "{\"kind\":\"stub\"}");
    if (!comm) {
        fprintf(stderr, "expected stub backend to load: %s\n", pyc_comm_loader_last_error());
        return 1;
    }

    st = comm->all_reduce(comm->backend_ctx, (pyc_comm_handle_t)comm, &value, &value, 1, PYC_DTYPE_I32, PYC_REDUCE_SUM, NULL);
    if (st != PYC_COMM_OK) {
        fprintf(stderr, "all_reduce should succeed for valid inputs\n");
        pyc_unload_comm_backend(comm);
        return 2;
    }

    st = comm->all_reduce(comm->backend_ctx, (pyc_comm_handle_t)comm, NULL, &value, 1, PYC_DTYPE_I32, PYC_REDUCE_SUM, NULL);
    if (st != PYC_COMM_ERR_INVALID) {
        fprintf(stderr, "all_reduce should reject invalid inputs\n");
        pyc_unload_comm_backend(comm);
        return 3;
    }
    pyc_unload_comm_backend(comm);

    comm = pyc_load_comm_backend("does/not/exist/backend.so", NULL);
    if (comm) {
        pyc_unload_comm_backend(comm);
        fprintf(stderr, "missing backend path should fail\n");
        return 4;
    }
    if (pyc_comm_loader_last_error()[0] == '\0') {
        fprintf(stderr, "missing backend should set loader error\n");
        return 5;
    }

    comm = pyc_load_comm_backend(PYC_TEST_INVALID_BACKEND_PATH, NULL);
    if (comm) {
        pyc_unload_comm_backend(comm);
        fprintf(stderr, "backend with missing symbols should fail\n");
        return 6;
    }
    if (strstr(pyc_comm_loader_last_error(), "missing required symbols") == NULL) {
        fprintf(stderr, "expected missing symbol error, got: %s\n", pyc_comm_loader_last_error());
        return 7;
    }

    {
        int rc = validate_backend_path(PYC_TEST_NCCL_BACKEND_PATH, 10);
        if (rc != 0) return rc;
    }
    {
        int rc = validate_backend_path(PYC_TEST_RCCL_BACKEND_PATH, 12);
        if (rc != 0) return rc;
    }
    {
        int rc = validate_backend_path(PYC_TEST_MPI_BACKEND_PATH, 14);
        if (rc != 0) return rc;
    }

    build_valid_module(&module);
    pyc_kernel_registry_reset();
    memset(&kd, 0, sizeof(kd));
    strcpy(kd.op_key, "matmul_fused");
    strcpy(kd.symbol, "kernel_cpu_v1");
    kd.backend = PYC_BACKEND_CPU;
    kd.priority = 1;
    pyc_kernel_register(&kd);

    memset(&desc, 0, sizeof(desc));
    desc.module = &module;
    desc.backend = PYC_BACKEND_CPU;

    memset(&options, 0, sizeof(options));
    options.enable_fusion = 1;
    options.enable_memory_reuse = 1;
    options.enable_autotune = 0;
    options.objective_mode = PYC_MODE_BALANCED;
    options.target_utilization_floor = 0.70;
    options.deterministic_strict = 1;
    pyc_runtime_rails_default(&options.rails);
    options.distributed.enabled = 1;
    options.distributed.backend = PYC_DIST_BACKEND_CUSTOM;
    options.distributed.strategy = PYC_DIST_STRATEGY_DATA_PARALLEL;
    options.distributed.world_size = 2;
    options.distributed.rank = 0;
    options.distributed.local_rank = 0;
    options.distributed.backend_path = PYC_TEST_STUB_BACKEND_PATH;
    options.distributed.config_json = "{\"kind\":\"stub\"}";

    compile_status = pyc_compile_model(&desc, &options, &model);
    if (compile_status != PYC_STATUS_OK || !model) {
        fprintf(stderr, "expected distributed compile to succeed, got=%d\n", (int)compile_status);
        return 8;
    }
    {
        const pyc_collective_comm* model_comm = pyc_model_distributed_comm(model);
        pyc_comm_handle_t model_handle = pyc_model_distributed_comm_handle(model);
        pyc_comm_status model_st;
        if (!model_comm || !model_handle) {
            fprintf(stderr, "expected model distributed init path to expose comm + handle\n");
            pyc_destroy_model(model);
            return 19;
        }
        model_st = model_comm->all_reduce(
            model_comm->backend_ctx,
            model_handle,
            &value,
            &value,
            1,
            PYC_DTYPE_I32,
            PYC_REDUCE_SUM,
            NULL);
        if (!(model_st == PYC_COMM_OK || model_st == PYC_COMM_ERR_HARDWARE)) {
            fprintf(stderr, "unexpected model all_reduce status=%d\n", (int)model_st);
            pyc_destroy_model(model);
            return 20;
        }
    }
    pyc_destroy_model(model);
    model = NULL;

    options.distributed.backend_path = NULL;
    compile_status = pyc_compile_model(&desc, &options, &model);
    if (compile_status != PYC_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "expected invalid argument for missing backend path, got=%d\n", (int)compile_status);
        if (model) {
            pyc_destroy_model(model);
        }
        return 9;
    }

    options.distributed.backend_path = PYC_TEST_NCCL_BACKEND_PATH;
    options.distributed.config_json = "{\"strict\":false}";
    compile_status = pyc_compile_model(&desc, &options, &model);
    if (compile_status != PYC_STATUS_OK || !model) {
        fprintf(stderr, "expected non-strict NCCL distributed compile to succeed, got=%d\n", (int)compile_status);
        return 16;
    }
    pyc_destroy_model(model);
    model = NULL;

    options.distributed.backend_path = PYC_TEST_RCCL_BACKEND_PATH;
    compile_status = pyc_compile_model(&desc, &options, &model);
    if (compile_status != PYC_STATUS_OK || !model) {
        fprintf(stderr, "expected non-strict RCCL distributed compile to succeed, got=%d\n", (int)compile_status);
        return 17;
    }
    pyc_destroy_model(model);
    model = NULL;

    options.distributed.backend_path = PYC_TEST_MPI_BACKEND_PATH;
    compile_status = pyc_compile_model(&desc, &options, &model);
    if (compile_status != PYC_STATUS_OK || !model) {
        fprintf(stderr, "expected non-strict MPI distributed compile to succeed, got=%d\n", (int)compile_status);
        return 18;
    }
    pyc_destroy_model(model);

    printf("test_collective_comm_loader: ok\n");
    return 0;
}
