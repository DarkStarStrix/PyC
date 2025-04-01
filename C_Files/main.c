// main.c - Main driver for the Python compiler
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "frontend.h"
#include "backend.h"
#include "parser.h"

// Forward declarations for external functions from ir_generator.c
extern void generate_ir(ASTNode* ast_root);
extern void cleanup_ir_generator();

// Print usage information
void print_usage(const char* program_name) {
    printf("Usage: %s [options] input_file\n", program_name);
    printf("Options:\n");
    printf("  -o <file>    Set output file name (default: a.out)\n");
    printf("  -O           Enable optimizations\n");
    printf("  -jit         Use JIT compilation only (don't generate object file)\n");
    printf("  -v           Enable verbose output\n");
    printf("  -h, --help   Show this help message\n");
}

int main(int argc, char** argv) {
    char* input_filename = NULL;
    char* output_filename = "a.out";
    int optimize = 0;
    int jit_only = 0;
    int verbose = 0;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_filename = argv[++i];
        } else if (strcmp(argv[i], "-O") == 0) {
            optimize = 1;
        } else if (strcmp(argv[i], "-jit") == 0) {
            jit_only = 1;
        } else if (strcmp(argv[i], "-v") == 0) {
            verbose = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        } else {
            input_filename = argv[i];
        }
    }
    
    // Check if input file is provided
    if (!input_filename) {
        fprintf(stderr, "Error: No input file specified\n");
        print_usage(argv[0]);
        return 1;
    }
    
    if (verbose) {
        printf("Compiling %s...\n", input_filename);
    }
    
    // Frontend: Parse the input file to generate AST
    ASTNode* ast_root = NULL;
    frontend(input_filename, &ast_root);
    
    if (!ast_root) {
        fprintf(stderr, "Error: Frontend processing failed\n");
        return 1;
    }
    
    if (verbose) {
        printf("Frontend processing complete\n");
    }
    
    // Middle-end: Generate LLVM IR from AST
    generate_ir(ast_root);
    
    if (verbose) {
        printf("IR generation complete\n");
    }
    
    // Backend: Compile and execute
    if (!jit_only) {
        backend(output_filename, optimize);
    } else {
        // JIT only mode
        initialize_backend();
        if (optimize) {
            printf("Optimizing code...\n");
            optimize_module();
        }
        int result = jit_compile_and_execute();
        printf("JIT execution result: %d\n", result);
    }
    
    // Cleanup
    freeAST(ast_root);
    cleanup_ir_generator();
    
    if (verbose) {
        printf("Compilation successful\n");
    }
    
    return 0;
}
