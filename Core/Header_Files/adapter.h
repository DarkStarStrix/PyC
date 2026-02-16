#ifndef PYC_ADAPTER_H
#define PYC_ADAPTER_H

#include <stddef.h>

typedef struct {
    int exit_code;
    char stderr_msg[256];
} AdapterResult;

int adapter_read_file(const char* path, char** out_source, size_t* out_size, char* err, size_t err_size);
int adapter_write_file(const char* path, const char* contents, char* err, size_t err_size);
AdapterResult adapter_run_command(const char* const argv[]);

#endif
