// error_handler.h - Error handling system header
#ifndef ERROR_HANDLER_H
#define ERROR_HANDLER_H

// Error state structure
typedef struct {
    int has_error;
    int error_count;
    int warning_count;
    const char* filename;
    const char* source_code;
} ErrorState;

// Initialize the error handler
void init_error_handler(const char* filename, const char* source);

// Parse source code into lines for better error reporting
void parse_source_lines(const char* source);

// Report an error
void report_error(int line, int column, const char* format, ...);

// Report a warning
void report_warning(int line, int column, const char* format, ...);

// Report a note (additional information)
void report_note(int line, int column, const char* format, ...);

// Check if there were any errors
int has_errors();

// Get error statistics
ErrorState* get_error_stats();

// Print error summary
void print_error_summary();

// Clean up the error handler
void cleanup_error_handler();

#endif // ERROR_HANDLER_H
