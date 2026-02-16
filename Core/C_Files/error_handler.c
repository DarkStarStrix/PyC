#include "error_handler.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#define MAX_ERRORS 64

static char g_errors[MAX_ERRORS][256];
static int g_error_count = 0;

void init_error_handler(const char* filename, const char* source) {
    (void)filename;
    (void)source;
    g_error_count = 0;
}

void add_error(const char* format, ...) {
    if (g_error_count >= MAX_ERRORS) {
        return;
    }

    va_list args;
    va_start(args, format);
    vsnprintf(g_errors[g_error_count], sizeof(g_errors[g_error_count]), format, args);
    va_end(args);
    g_error_count++;
}

int has_errors(void) {
    return g_error_count > 0;
}

void print_errors(void) {
    for (int i = 0; i < g_error_count; ++i) {
        fprintf(stderr, "error: %s\n", g_errors[i]);
    }
}

// Cleanup resources
void error_handler_cleanup(void) {
    if (source_lines) {
        for (int i = 0; i < line_count; i++) {
            free(source_lines[i]);
        }
        free(source_lines);
        source_lines = NULL;
    }

    free((void*)error_state.filename);
    free((void*)error_state.source_code);
    error_state.filename = NULL;
    error_state.source_code = NULL;

    if (error_log && error_log != stderr) {
        fclose(error_log);
        error_log = NULL;
    }

    line_count = 0;
}

void init_error_handler(const char* filename, const char* source) {
    error_handler_init(filename, source);
}

void cleanup_error_handler(void) {
    error_handler_cleanup();
}
