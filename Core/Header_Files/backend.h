#ifndef PYC_BACKEND_H
#define PYC_BACKEND_H

#include "ir_generator.h"

typedef enum {
    BACKEND_OBJECT,
    BACKEND_JIT
} BackendOutputMode;

int emit_backend_output(const IRCode* ir, const char* output_path, BackendOutputMode mode);

#endif
