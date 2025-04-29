// error_handler.h - Error handling system header
#ifndef ERROR_HANDLER_H
#define ERROR_HANDLER_H

#include <stdarg.h>

#define MAX_ERROR_MSG_LEN 512
#define MAX_ERRORS 100
#define MAX_WARNINGS 100

typedef enum {
    ERR_UNDEFINED_VAR,
    ERR_TYPE_MISMATCH,
    ERR_UNDEFINED_FUNC,
    ERR_INVALID_COMPLEX_TYPE,
    ERR_SCOPE_OVERFLOW,
    ERR_TEMPLATE_MISMATCH,
    ERR_OVERLOAD_AMBIGUOUS,
    ERR_INVALID_TEMPLATE_PARAM,
    ERR_MEMORY_ALLOCATION,
    ERR_BUFFER_OVERFLOW,
    WARN_UNUSED_VARIABLE,
    WARN_IMPLICIT_CONVERSION,
    WARN_SHADOWED_VARIABLE
} DiagnosticCode;

typedef struct {
    DiagnosticCode code;
    int line;
    int column;
    char message[MAX_ERROR_MSG_LEN];
    char file[256];
    char function[256];
    int severity; // 0=info, 1=warning, 2=error, 3=fatal
    char* source_line;
    int source_line_length;
    char suggestion[MAX_ERROR_MSG_LEN];
} DiagnosticInfo;

typedef struct {
    DiagnosticInfo errors[MAX_ERRORS];
    DiagnosticInfo warnings[MAX_WARNINGS];
    int error_count;
    int warning_count;
    char source_file[256];
    char** source_lines;
    int total_lines;
} DiagnosticContext;

// Enhanced error handling functions
void init_error_handler(const char* filename, const char* source);
void report_diagnostic(DiagnosticCode code, int line, int column, const char* file, const char* function, const char* message, ...);
void suggest_fix(DiagnosticCode code, const char* identifier);
const char* get_diagnostic_message(DiagnosticCode code);
void print_diagnostic_location(int line, int column, const char* source_line);
int has_fatal_errors(void);
void clear_diagnostics(void);

// New recovery functions
void begin_error_recovery(void);
void end_error_recovery(void);
void suppress_similar_errors(DiagnosticCode code);
void set_max_errors(int max);
void enable_warning_as_error(DiagnosticCode code);

// New diagnostic utilities
void print_diagnostic_summary(void);
void export_diagnostics_to_json(const char* filename);
void suggest_quick_fixes(DiagnosticInfo* info);
void analyze_error_patterns(void);
char* get_formatted_source_snippet(int line, int context_lines);

// Stack trace support
typedef struct {
    char function[256];
    char file[256];
    int line;
} StackFrame;

void capture_stack_trace(void);
void print_stack_trace(void);
StackFrame* get_current_stack(int* depth);

#endif
