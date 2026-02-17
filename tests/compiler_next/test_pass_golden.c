#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pyc/ir.h"
#include "pyc/pass_manager.h"

#ifndef PYC_SOURCE_DIR
#define PYC_SOURCE_DIR "."
#endif

static int read_file(const char* path, char* out, size_t out_size) {
    FILE* f = fopen(path, "rb");
    size_t n;
    if (!f) return -1;
    n = fread(out, 1, out_size - 1, f);
    fclose(f);
    out[n] = '\0';
    return 0;
}

static void normalize_newlines(char* s) {
    char* r = s;
    char* w = s;
    while (*r) {
        if (*r == '\r') {
            r++;
            if (*r == '\n') {
                *w++ = '\n';
                r++;
            } else {
                *w++ = '\n';
            }
            continue;
        }
        *w++ = *r++;
    }
    *w = '\0';
}

int main(void) {
    pyc_ir_module m;
    pyc_ir_op op;
    pyc_pass_pipeline p;
    pyc_pass_report r;
    char got[4096];
    char expected[4096];
    const char* expected_path = PYC_SOURCE_DIR "/tests/compiler_next/golden/simple_pipeline_after.txt";

    pyc_ir_module_init(&m);

    memset(&op, 0, sizeof(op));
    op.kind = PYC_IR_OP_INPUT;
    strcpy(op.name, "lhs");
    op.dtype = PYC_DTYPE_F32;
    op.shape.rank = 2;
    op.shape.dims[0] = 2;
    op.shape.dims[1] = 3;
    if (pyc_ir_add_op(&m, &op) != 0) return 1;

    memset(&op, 0, sizeof(op));
    op.kind = PYC_IR_OP_INPUT;
    strcpy(op.name, "rhs");
    op.dtype = PYC_DTYPE_F32;
    op.shape.rank = 2;
    op.shape.dims[0] = 3;
    op.shape.dims[1] = 4;
    if (pyc_ir_add_op(&m, &op) != 0) return 2;

    memset(&op, 0, sizeof(op));
    op.kind = PYC_IR_OP_MATMUL;
    op.dtype = PYC_DTYPE_F32;
    op.shape.rank = 2;
    op.shape.dims[0] = 2;
    op.shape.dims[1] = 4;
    op.input_ids[0] = 0;
    op.input_ids[1] = 1;
    op.input_count = 2;
    if (pyc_ir_add_op(&m, &op) != 0) return 3;

    memset(&op, 0, sizeof(op));
    op.kind = PYC_IR_OP_RELU;
    op.dtype = PYC_DTYPE_F32;
    op.shape.rank = 2;
    op.shape.dims[0] = 2;
    op.shape.dims[1] = 4;
    op.input_ids[0] = 2;
    op.input_count = 1;
    if (pyc_ir_add_op(&m, &op) != 0) return 4;

    memset(&op, 0, sizeof(op));
    op.kind = PYC_IR_OP_OUTPUT;
    op.dtype = PYC_DTYPE_F32;
    op.shape.rank = 2;
    op.shape.dims[0] = 2;
    op.shape.dims[1] = 4;
    op.input_ids[0] = 3;
    op.input_count = 1;
    if (pyc_ir_add_op(&m, &op) != 0) return 5;

    pyc_pass_pipeline_default(&p);
    if (pyc_pass_pipeline_run(&p, &m, &r) != 0) return 6;
    if (r.fused_patterns < 1) return 7;

    if (pyc_ir_serialize(&m, got, sizeof(got)) != 0) return 8;
    if (read_file(expected_path, expected, sizeof(expected)) != 0) return 9;
    normalize_newlines(got);
    normalize_newlines(expected);

    if (strcmp(got, expected) != 0) {
        fprintf(stderr, "golden mismatch\nEXPECTED:\n%s\nGOT:\n%s\n", expected, got);
        return 10;
    }

    printf("test_pass_golden: ok\n");
    return 0;
}
