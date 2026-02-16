#ifndef BACKEND_H
#define BACKEND_H

int backend_init(void);
void backend_cleanup(void);
int backend_jit_compile_and_execute(void);
void backend_optimize_module(void);
int backend_compile_to_object(const char* output_filename);
void backend_parallel_compile(const char* output_base_filename, int num_cores);
void backend_run(const char* output_filename, int optimize);

#endif // BACKEND_H
