#ifndef PYC_ERROR_HANDLER_H
#define PYC_ERROR_HANDLER_H

void init_error_handler(const char* filename, const char* source);
void add_error(const char* format, ...);
int has_errors(void);
void print_errors(void);
void cleanup_error_handler(void);

#endif
