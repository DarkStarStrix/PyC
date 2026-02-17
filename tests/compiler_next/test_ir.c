#include <stdio.h>
#include <string.h>

#include "pyc/ir.h"

static int test_valid_module(void) {
    pyc_ir_module m;
    pyc_ir_op a;
    pyc_ir_op b;
    pyc_ir_diagnostic d;

    pyc_ir_module_init(&m);
    memset(&a, 0, sizeof(a));
    a.kind = PYC_IR_OP_INPUT;
    strcpy(a.name, "input");
    a.dtype = PYC_DTYPE_F32;
    a.shape.rank = 1;
    a.shape.dims[0] = 4;

    memset(&b, 0, sizeof(b));
    b.kind = PYC_IR_OP_OUTPUT;
    strcpy(b.name, "out");
    b.dtype = PYC_DTYPE_F32;
    b.shape.rank = 1;
    b.shape.dims[0] = 4;
    b.input_ids[0] = 0;
    b.input_count = 1;

    if (pyc_ir_add_op(&m, &a) != 0) return 1;
    if (pyc_ir_add_op(&m, &b) != 0) return 2;
    if (pyc_ir_verify(&m, &d) != 0) return 3;
    return 0;
}

static int test_invalid_input_id(void) {
    pyc_ir_module m;
    pyc_ir_op a;
    pyc_ir_diagnostic d;

    pyc_ir_module_init(&m);
    memset(&a, 0, sizeof(a));
    a.kind = PYC_IR_OP_OUTPUT;
    strcpy(a.name, "bad");
    a.dtype = PYC_DTYPE_F32;
    a.shape.rank = 1;
    a.shape.dims[0] = 1;
    a.input_ids[0] = 7;
    a.input_count = 1;

    if (pyc_ir_add_op(&m, &a) != 0) return 10;
    if (pyc_ir_verify(&m, &d) == 0) return 11;
    if (d.code != 5) return 12;
    return 0;
}

static int test_empty_module(void) {
    pyc_ir_module m;
    pyc_ir_diagnostic d;

    pyc_ir_module_init(&m);
    if (pyc_ir_verify(&m, &d) == 0) return 20;
    if (d.code != 2) return 21;
    return 0;
}

int main(void) {
    int rc;
    rc = test_valid_module();
    if (rc) return rc;
    rc = test_invalid_input_id();
    if (rc) return rc;
    rc = test_empty_module();
    if (rc) return rc;
    printf("test_ir: ok\n");
    return 0;
}
