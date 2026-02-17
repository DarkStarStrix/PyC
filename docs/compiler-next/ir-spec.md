# Compiler-Next IR Spec (v0)

## Core Types

Defined in `include/pyc/ir.h`:

- `pyc_dtype`
- `pyc_shape`
- `pyc_ir_op_kind`
- `pyc_ir_op`
- `pyc_ir_module`
- `pyc_ir_diagnostic`

## Limits

- `PYC_IR_MAX_OPS = 1024`
- `PYC_IR_MAX_INPUTS = 8`
- `PYC_IR_MAX_DIMS = 8`

## Validation Rules

1. Module must contain at least one op.
2. Each op input id must reference an earlier op index.
3. Shape rank must not exceed `PYC_IR_MAX_DIMS`.
4. All declared dimensions must be positive.

## Planned Extensions

- Dynamic dimensions and symbolic shape constraints.
- Layout annotations and memory format tracking.
- Attribute maps for op-specific codegen hints.
