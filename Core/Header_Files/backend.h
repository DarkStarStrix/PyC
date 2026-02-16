#ifndef BACKEND_H
#define BACKEND_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct IRNode {
    const char* operation;
    const char* value;
    struct IRNode** children;
    size_t children_count;
} IRNode;

IRNode* createIRNode(const char* operation, const char* value);
void addIRChild(IRNode* parent, IRNode* child);
void freeIR(IRNode* root);
void parseAndGenerateIR(const char* code);
void jitCompile(const char* code);

unsigned int checkCores(void);
void* compileTask(void* arg);
void scheduleTasks(int cores);

void initialize_backend(void);
int jit_compile_and_execute(void);
void optimize_module(void);
int compile_to_object(const char* output_filename);
void parallel_compile(const char* output_base_filename, int num_cores);
void backend(const char* output_filename, int optimize);

#ifdef __cplusplus
}
#endif

#endif
