#ifndef PYC_IR_H
#define PYC_IR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PYC_IR_MAX_DIMS 8
#define PYC_IR_MAX_OPS 1024
#define PYC_IR_MAX_INPUTS 8
#define PYC_IR_MAX_NAME 64

typedef enum {
    PYC_DTYPE_UNKNOWN = 0,
    PYC_DTYPE_F32,
    PYC_DTYPE_F16,
    PYC_DTYPE_I32,
    PYC_DTYPE_I8
} pyc_dtype;

typedef struct {
    size_t rank;
    int64_t dims[PYC_IR_MAX_DIMS];
} pyc_shape;

typedef enum {
    PYC_IR_OP_INPUT = 0,
    PYC_IR_OP_CONST,
    PYC_IR_OP_MATMUL,
    PYC_IR_OP_ADD,
    PYC_IR_OP_RELU,
    PYC_IR_OP_GELU,
    PYC_IR_OP_REDUCE_SUM,
    PYC_IR_OP_LAYERNORM,
    PYC_IR_OP_OUTPUT
} pyc_ir_op_kind;

typedef struct {
    pyc_ir_op_kind kind;
    char name[PYC_IR_MAX_NAME];
    pyc_dtype dtype;
    pyc_shape shape;
    int input_ids[PYC_IR_MAX_INPUTS];
    size_t input_count;
} pyc_ir_op;

typedef struct {
    pyc_ir_op ops[PYC_IR_MAX_OPS];
    size_t op_count;
} pyc_ir_module;

typedef struct {
    int code;
    char message[256];
} pyc_ir_diagnostic;

void pyc_ir_module_init(pyc_ir_module* module);
int pyc_ir_add_op(pyc_ir_module* module, const pyc_ir_op* op);
int pyc_ir_verify(const pyc_ir_module* module, pyc_ir_diagnostic* diag);
int pyc_ir_serialize(const pyc_ir_module* module, char* buffer, size_t buffer_size);

#ifdef __cplusplus
}
#endif

#endif
