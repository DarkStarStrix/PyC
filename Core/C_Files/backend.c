#include "backend.h"

#include <stdio.h>

int emit_backend_output(const IRCode* ir, const char* output_path, BackendOutputMode mode) {
    if (!ir || !ir->text || !output_path) {
        return -1;
    }

    FILE* out = fopen(output_path, "w");
    if (!out) {
        return -1;
    }

    if (mode == BACKEND_OBJECT) {
        fprintf(out, "# pseudo-object\n%s", ir->text);
    } else {
        fprintf(out, "# pseudo-jit-entry\n%s", ir->text);
    }

    fclose(out);
    return 0;
}

void backend_cleanup(void) {
}
