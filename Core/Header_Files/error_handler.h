#ifndef ERROR_HANDLER_H
#define ERROR_HANDLER_H

#include <stdarg.h>

typedef struct {
    int has_error;
    int error_count;
    int warning_count;
    const char* filename;
    const char* source_code;
} ErrorState;

void error_handler_init(const char* filename, const char* source);
void report_error(int line, int column, const char* format, ...);
void report_warning(int line, int column, const char* format, ...);
void report_note(int line, int column, const char* format, ...);
int has_errors(void);
ErrorState* get_error_stats(void);
void print_error_summary(void);
void error_handler_cleanup(void);

/* Backward-compat wrappers */
void init_error_handler(const char* filename, const char* source);
void cleanup_error_handler(void);

#endif // ERROR_HANDLER_H
