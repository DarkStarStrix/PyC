// Core/C_Files/error_handler.c - Enhanced error handling for PyC compiler
#include "error_handler.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

// Global error state
static void parse_source_lines(const char* source);

static char* error_strdup(const char* s) {
    size_t len = strlen(s) + 1;
    char* out = (char*)malloc(len);
    if (out) {
        memcpy(out, s, len);
    }
    return out;
}

static ErrorState error_state = {
    .has_error = 0,
    .error_count = 0,
    .warning_count = 0,
    .filename = NULL,
    .source_code = NULL
};

// Source code lines for context
static char** source_lines = NULL;
static int line_count = 0;

// Error log file
static FILE* error_log = NULL;

// Initialize the error handler
void error_handler_init(const char* filename, const char* source) {
    error_handler_cleanup();

    error_state.has_error = 0;
    error_state.error_count = 0;
    error_state.warning_count = 0;
    error_state.filename = error_strdup(filename ? filename : "unknown");
    error_state.source_code = error_strdup(source ? source : "");

    error_log = fopen("compiler_errors.log", "w");
    if (!error_log) {
        fprintf(stderr, "Warning: Failed to open error log, using stderr\n");
        error_log = stderr;
    }

    parse_source_lines(source);
}

// Parse source code into lines
void parse_source_lines(const char* source) {
    if (!source) return;

    line_count = 1;
    const char* p = source;
    while (*p) {
        if (*p++ == '\n') line_count++;
    }

    source_lines = (char**)malloc(sizeof(char*) * line_count);
    if (!source_lines) {
        fprintf(error_log, "Error: Memory allocation failed for source lines\n");
        return;
    }

    p = source;
    char* line_start = (char*)p;
    int line_idx = 0;

    while (*p) {
        if (*p == '\n') {
            size_t len = p - line_start;
            source_lines[line_idx] = (char*)malloc(len + 1);
            if (source_lines[line_idx]) {
                strncpy(source_lines[line_idx], line_start, len);
                source_lines[line_idx][len] = '\0';
            }
            line_idx++;
            line_start = (char*)(p + 1);
        }
        p++;
    }

    if (line_start < p) {
        size_t len = p - line_start;
        source_lines[line_idx] = (char*)malloc(len + 1);
        if (source_lines[line_idx]) {
            strncpy(source_lines[line_idx], line_start, len);
            source_lines[line_idx][len] = '\0';
        }
    }
}

// Report an error with detailed context
void report_error(int line, int column, const char* format, ...) {
    va_list args;
    va_start(args, format);

    fprintf(error_log, "\033[1;31mError\033[0m at %s:%d:%d: ", error_state.filename, line, column);
    vfprintf(error_log, format, args);
    fprintf(error_log, "\n");

    if (source_lines && line > 0 && line <= line_count) {
        fprintf(error_log, "  %s\n", source_lines[line - 1]);
        fprintf(error_log, "  %*s^\n", column - 1, "");
    }

    va_end(args);
    error_state.has_error = 1;
    error_state.error_count++;
}

// Report a warning
void report_warning(int line, int column, const char* format, ...) {
    va_list args;
    va_start(args, format);

    fprintf(error_log, "\033[1;33mWarning\033[0m at %s:%d:%d: ", error_state.filename, line, column);
    vfprintf(error_log, format, args);
    fprintf(error_log, "\n");

    if (source_lines && line > 0 && line <= line_count) {
        fprintf(error_log, "  %s\n", source_lines[line - 1]);
        fprintf(error_log, "  %*s\033[1;33m^\033[0m\n", column - 1, "");
    }

    va_end(args);
    error_state.warning_count++;
}

// Report a note for additional context
void report_note(int line, int column, const char* format, ...) {
    va_list args;
    va_start(args, format);

    fprintf(error_log, "\033[1;36mNote\033[0m at %s:%d:%d: ", error_state.filename, line, column);
    vfprintf(error_log, format, args);
    fprintf(error_log, "\n");

    va_end(args);
}

// Check for errors
int has_errors() {
    return error_state.has_error;
}

// Get error statistics
ErrorState* get_error_stats() {
    return &error_state;
}

// Print compilation summary
void print_error_summary() {
    fprintf(error_log, "\n--- Compilation Summary ---\n");
    fprintf(error_log, "File: %s\n", error_state.filename);
    fprintf(error_log, "Errors: %d\n", error_state.error_count);
    fprintf(error_log, "Warnings: %d\n", error_state.warning_count);

    if (error_state.error_count > 0) {
        fprintf(error_log, "\033[1;31mCompilation failed\033[0m\n");
    } else if (error_state.warning_count > 0) {
        fprintf(error_log, "\033[1;33mCompilation completed with warnings\033[0m\n");
    } else {
        fprintf(error_log, "\033[1;32mCompilation successful\033[0m\n");
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
