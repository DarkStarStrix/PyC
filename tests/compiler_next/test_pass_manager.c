#include <stdio.h>
#include <string.h>

#include "pyc/ir.h"
#include "pyc/pass_manager.h"

int main(void) {
    pyc_pass_pipeline p;
    pyc_pass_report r;
    pyc_ir_module m;
    pyc_ir_op op;

    pyc_pass_pipeline_default(&p);
    if (!(p.config.canonicalize && p.config.shape_inference && p.config.layout_propagation &&
          p.config.fusion && p.config.liveness && p.config.allocation && p.config.lowering)) {
        return 1;
    }

    pyc_ir_module_init(&m);
    if (pyc_pass_pipeline_run(&p, &m, &r) == 0) return 2;
    if (r.errors != 1) return 3;

    memset(&op, 0, sizeof(op));
    op.kind = PYC_IR_OP_INPUT;
    strcpy(op.name, "in");
    op.dtype = PYC_DTYPE_F32;
    op.shape.rank = 1;
    op.shape.dims[0] = 2;
    if (pyc_ir_add_op(&m, &op) != 0) return 4;

    if (pyc_pass_pipeline_run(&p, &m, &r) != 0) return 5;
    if (r.passes_run != 7) return 6;

    printf("test_pass_manager: ok\n");
    return 0;
}
