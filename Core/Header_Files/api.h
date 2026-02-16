#ifndef API_H
#define API_H

int api_init(void);
void compile_script(const char* filename);
void optimize_script(const char* filename, int graph_opt);
void visualize_graph(const char* filename);
void run_script(const char* filename);
void register_kernel(const char* kernel_file);
void api_cleanup(void);
void cleanup_api(void);

#endif // API_H
