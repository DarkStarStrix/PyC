// main.c - Main driver for the Python compiler
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "frontend.h"
#include "backend.h"
#include "parser.h"
#include "api.h"

// Forward declarations for external functions from ir_generator.c
extern void generate_ir(ASTNode* ast_root);
extern void cleanup_ir_generator();

void print_usage(const char* program_name) {
    printf("Usage: %s [options] <command> <input_file>\n", program_name);
    printf("Commands:\n");
    printf("  build         Compile and optimize script\n");
    printf("  optimize      Optimize script (use with -O for graph optimization)\n");
    printf("  visualize     Visualize computational graph\n");
    printf("  run          Run optimized script\n");
    printf("  kernel       Register CUDA kernel\n");
    printf("\nOptions:\n");
    printf("  -o <file>    Set output file name (default: a.out)\n");
    printf("  -O           Enable graph optimizations\n");
    printf("  -jit         Use JIT compilation only\n");
    printf("  -v           Enable verbose output\n");
    printf("  -h, --help   Show this help message\n");
}

int main(int argc, char** argv) {
    char* input_filename = NULL;
    char* output_filename = "a.out";
    int optimize = 0;
    int jit_only = 0;
    int verbose = 0;
    const char* command = NULL;

    // Need at least command and input file
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

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
        } else if (argv[i][0] != '-' && !command) {
            command = argv[i];
            if (i + 1 < argc) {
                input_filename = argv[i + 1];
                i++;
            }
        }
    }

    if (!command || !input_filename) {
        fprintf(stderr, "Error: Command and input file required\n");
        print_usage(argv[0]);
        return 1;
    }

    if (verbose) {
        printf("Processing %s with command: %s\n", input_filename, command);
    }

    // Handle API commands
    if (strcmp(command, "build") == 0) {
        compile_script(input_filename);
        if (!jit_only) {
            backend(output_filename, optimize);
        }
    } else if (strcmp(command, "optimize") == 0) {
        optimize_script(input_filename, optimize);
    } else if (strcmp(command, "visualize") == 0) {
        visualize_graph(input_filename);
    } else if (strcmp(command, "run") == 0) {
        run_script(input_filename);
    } else if (strcmp(command, "kernel") == 0) {
        register_kernel(input_filename);
    } else {
        fprintf(stderr, "Unknown command: %s\n", command);
        print_usage(argv[0]);
        return 1;
    }

    if (verbose) {
        printf("Operation completed successfully\n");
    }

    cleanup_api();
    return 0;
}
