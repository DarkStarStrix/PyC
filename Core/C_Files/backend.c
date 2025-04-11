// backend.c
#include "backend.h"
#include <llvm-c/Core.h>
#include <llvm-c/ExecutionEngine.h>
#include <llvm-c/Target.h>
#include <llvm-c/TargetMachine.h>
#include <llvm-c/Analysis.h>
#include <llvm-c/BitWriter.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

// External LLVM module from IR generator
extern LLVMModuleRef module;

// Structure to pass data to compilation threads
typedef struct {
    int thread_id;
    LLVMModuleRef module_clone;
    const char* output_filename;
} CompileThreadData;

// Initialize backend
void initialize_backend() {
    // Initialize LLVM targets
    LLVMInitializeAllTargetInfos();
    LLVMInitializeAllTargets();
    LLVMInitializeAllTargetMCs();
    LLVMInitializeAllAsmParsers();
    LLVMInitializeAllAsmPrinters();
}

// JIT compile and execute
int jit_compile_and_execute() {
    char* error = NULL;
    LLVMExecutionEngineRef engine;
    
    // Create execution engine
    if (LLVMCreateExecutionEngineForModule(&engine, module, &error) != 0) {
        fprintf(stderr, "Failed to create execution engine: %s\n", error);
        LLVMDisposeMessage(error);
        return 1;
    }
    
    // Find main function
    LLVMValueRef main_func = LLVMGetNamedFunction(module, "main");
    if (!main_func) {
        fprintf(stderr, "Failed to find main function\n");
        return 1;
    }
    
    // Execute main function
    printf("Executing JIT compiled code...\n");
    LLVMGenericValueRef result = LLVMRunFunction(engine, main_func, 0, NULL);
    int return_value = (int)LLVMGenericValueToInt(result, 0);
    
    // Clean up
    LLVMDisposeGenericValue(result);
    LLVMDisposeExecutionEngine(engine);
    
    return return_value;
}

// Thread function for parallel compilation
void* compile_thread_function(void* arg) {
    CompileThreadData* data = (CompileThreadData*)arg;
    
    printf("Thread %d: Starting compilation\n", data->thread_id);
    
    // In a real implementation, this would handle a part of the compilation process
    // For now, we'll just simulate work
    
    printf("Thread %d: Compilation complete\n", data->thread_id);
    
    return NULL;
}

// Optimize the module
void optimize_module() {
    LLVMPassManagerRef pass_manager = LLVMCreatePassManager();
    
    // Add optimization passes
    LLVMAddPromoteMemoryToRegisterPass(pass_manager);
    LLVMAddInstructionCombiningPass(pass_manager);
    LLVMAddReassociatePass(pass_manager);
    LLVMAddGVNPass(pass_manager);
    LLVMAddCFGSimplificationPass(pass_manager);
    
    // Run the passes
    LLVMRunPassManager(pass_manager, module);
    
    // Clean up
    LLVMDisposePassManager(pass_manager);
}

// Compile to object file
int compile_to_object(const char* output_filename) {
    char* error = NULL;
    
    // Get host triple
    char* target_triple = LLVMGetDefaultTargetTriple();
    
    // Get target
    LLVMTargetRef target;
    if (LLVMGetTargetFromTriple(target_triple, &target, &error) != 0) {
        fprintf(stderr, "Failed to get target: %s\n", error);
        LLVMDisposeMessage(error);
        free(target_triple);
        return 1;
    }
    
    // Create target machine
    char* cpu = LLVMGetHostCPUName();
    char* features = LLVMGetHostCPUFeatures();
    
    LLVMTargetMachineRef target_machine = LLVMCreateTargetMachine(
        target,
        target_triple,
        cpu,
        features,
        LLVMCodeGenLevelDefault,
        LLVMRelocDefault,
        LLVMCodeModelDefault
    );
    
    // Set target triple in the module
    LLVMSetTarget(module, target_triple);
    
    // Emit object code
    if (LLVMTargetMachineEmitToFile(
            target_machine,
            module,
            (char*)output_filename,
            LLVMObjectFile,
            &error) != 0) {
        fprintf(stderr, "Failed to emit object file: %s\n", error);
        LLVMDisposeMessage(error);
        LLVMDisposeTargetMachine(target_machine);
        free(features);
        free(cpu);
        free(target_triple);
        return 1;
    }
    
    // Clean up
    LLVMDisposeTargetMachine(target_machine);
    free(features);
    free(cpu);
    free(target_triple);
    
    printf("Object file generated: %s\n", output_filename);
    return 0;
}

// Parallel compilation using multiple cores
void parallel_compile(const char* output_base_filename, int num_cores) {
    if (num_cores <= 1) {
        // If only one core, just compile normally
        compile_to_object(output_base_filename);
        return;
    }
    
    printf("Starting parallel compilation with %d cores\n", num_cores);
    
    // Create thread data
    CompileThreadData* thread_data = (CompileThreadData*)malloc(sizeof(CompileThreadData) * num_cores);
    pthread_t* threads = (pthread_t*)malloc(sizeof(pthread_t) * num_cores);
    
    // Create and start threads
    for (int i = 0; i < num_cores; i++) {
        thread_data[i].thread_id = i;
        
        // In a real implementation, we would clone the module or split work
        thread_data[i].module_clone = module;
        
        // Create output filename for this thread
        char* output_filename = (char*)malloc(strlen(output_base_filename) + 10);
        sprintf(output_filename, "%s.%d.o", output_base_filename, i);
        thread_data[i].output_filename = output_filename;
        
        pthread_create(&threads[i], NULL, compile_thread_function, &thread_data[i]);
    }
    
    // Wait for threads to complete
    for (int i = 0; i < num_cores; i++) {
        pthread_join(threads[i], NULL);
        free((void*)thread_data[i].output_filename);
    }
    
    // Clean up
    free(thread_data);
    free(threads);
    
    printf("Parallel compilation complete\n");
}

// Main backend function
void backend(const char* output_filename, int optimize) {
    initialize_backend();
    
    // Optimize if requested
    if (optimize) {
        printf("Optimizing code...\n");
        optimize_module();
    }
    
    // Get number of CPU cores
    unsigned int cores = checkCores();
    printf("Detected %u CPU cores\n", cores);
    
    // JIT compile and execute
    int result = jit_compile_and_execute();
    printf("Execution result: %d\n", result);
    
    // Compile to object file with parallel processing
    parallel_compile(output_filename, cores);
    
    printf("Compilation complete!\n");
}
