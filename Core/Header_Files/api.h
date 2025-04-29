#ifndef API_H
#define API_H

void compile_script(const char* filename);
void optimize_script(const char* filename, int graph_opt);
void visualize_graph(const char* filename);
void run_script(const char* filename);
void register_kernel(const char* kernel_file);

#endif