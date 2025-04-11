// error_handler.c - Robust error handling system
#include "error_handler.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

// Global error state
static ErrorState error_state = {
    .has_error = 0,
    .error_count = 0,
    .warning_count = 0,
    .filename = NULL,
    .source_code = NULL
};

// Current file being processed
static const char* current_filename = "unknown";

// Source code lines (for error reporting)
static char** source_lines = NULL;
static int line_count = 0;

// Error log file
static FILE* error_log = NULL;

// Initialize the error handler
void init_error_handler(const char* filename, const char* source) {
    cleanup_error_handler(); // Clean up any previous state
    
    error_state.has_error = 0;
    error_state.error_count = 0;
    error_state.warning_count = 0;
    error_state.filename = strdup(filename);
    error_state.source_code = strdup(source);
    
    current_filename = error_state.filename;
    
    // Open error log file
    error_log = fopen("compiler_errors.log", "w");
    if (!error_log) {
        fprintf(stderr, "Warning: Could not open error log file\n");
        error_log = stderr; // Fall back to stderr
    }
    
    // Split source code into lines for better error reporting
    parse_source_lines(source);
}

// Split source code into lines for better error reporting
void parse_source_lines(const char* source) {
    if (!source) return;
    
    // Count the number of lines
    line_count = 1;
    const char* p = source;
    while (*p) {
        if (*p == '\n') line_count++;
        p++;
    }
    
    // Allocate line array
    source_lines = (char**)malloc(sizeof(char*) * line_count);
    if (!source_lines) {
        fprintf(stderr, "Error: Memory allocation failed for source lines\n");
        return;
    }
    
    // Copy each line
    p = source;
    char* line_start = (char*)p;
    int line_idx = 0;
    
    while (*p) {
        if (*p == '\n') {
            // Calculate line length
            size_t len = p - line_start;
            
            // Allocate and copy the line
            source_lines[line_idx] = (char*)malloc(len + 1);
            if (source_lines[line_idx]) {
                strncpy(source_lines[line_idx], line_start, len);
                source_lines[line_idx][len] = '\0';
            }
            
            // Move to next line
            line_idx++;
            line_start = (char*)(p + 1);
        }
        p++;
    }
    
    // Handle the last line if it doesn't end with a newline
    if (line_start < p) {
        size_t len = p - line_start;
        source_lines[line_idx] = (char*)malloc(len + 1);
        if (source_lines[line_idx]) {
            strncpy(source_lines[line_idx], line_start, len);
            source_lines[line_idx][len] = '\0';
        }
    }
}

// Report an error
void report_error(int line, int column, const char* format, ...) {
    va_list args;
    va_start(args, format);
    
    fprintf(error_log, "Error at %s:%d:%d: ", current_filename, line, column);
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
    
    va_end(args);
    
    error_state.has_error = 1;
    error_state.error_count++;
}

// Report a warning
void report_warning(int line, int column, const char* format, ...) {
    va_list args;
    va_start(args, format);
    
    fprintf(error_log, "\033[1;33mWarning\033[0m at %s:%d:%d: ", current_filename, line, column);
    vfprintf(error_log, format, args);
    fprintf(error_log, "\n");
    
    // Show the line with the warning
    if (source_lines && line > 0 && line <= line_count) {
        fprintf(error_log, "  %s\n", source_lines[line-1]);
        fprintf(error_log, "  ");
        for (int i = 0; i < column - 1; i++) {
            fprintf(error_log, " ");
        }
        fprintf(error_log, "\033[1;33m^\033[0m\n");
    }
    
    va_end(args);
    
    error_state.warning_count++;
}

// Report a note (additional information)
void report_note(int line, int column, const char* format, ...) {
    va_list args;
    va_start(args, format);
    
    fprintf(error_log, "\033[1;36mNote\033[0m at %s:%d:%d: ", current_filename, line, column);
    vfprintf(error_log, format, args);
    fprintf(error_log, "\n");
    
    va_end(args);
}

// Check if there were any errors
int has_errors() {
    return error_state.has_error;
}

// Get error statistics
ErrorState* get_error_stats() {
    return &error_state;
}

// Print error summary
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

// Clean up the error handler
void cleanup_error_handler() {
    // Free source lines
    if (source_lines) {
        for (int i = 0; i < line_count; i++) {
            free(source_lines[i]);
        }
        free(source_lines);
        source_lines = NULL;
    }
    
    // Free error state strings
    free((void*)error_state.filename);
    free((void*)error_state.source_code);
    error_state.filename = NULL;
    error_state.source_code = NULL;
    
    // Close error log file
    if (error_log && error_log != stderr) {
        fclose(error_log);
        error_log = NULL;
    }
    
    line_count = 0;
}
