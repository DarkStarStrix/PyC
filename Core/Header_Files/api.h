#ifndef PYC_API_H
#define PYC_API_H

#include "backend.h"

typedef enum {
    API_STATUS_OK = 0,
    API_STATUS_INVALID_ARGUMENT,
    API_STATUS_IO_ERROR,
    API_STATUS_LEX_ERROR,
    API_STATUS_PARSE_ERROR,
    API_STATUS_SEMANTIC_ERROR,
    API_STATUS_IR_ERROR,
    API_STATUS_BACKEND_ERROR,
    API_STATUS_FEATURE_DISABLED
} ApiStatus;

typedef struct {
    BackendOutputMode output_mode;
    const char* output_path;
} BuildConfig;

ApiStatus build_script(const char* filename, const BuildConfig* cfg);
ApiStatus optimize_script(const char* filename, int graph_opt);
ApiStatus visualize_graph(const char* filename);
ApiStatus register_kernel(const char* kernel_file);
void cleanup_api(void);

#endif
