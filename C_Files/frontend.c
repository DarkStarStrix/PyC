// frontend.c
#include "frontend.h"
#include "lexer.h"
#include "parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Global variables
char* source_code = NULL;
extern const char* input; // From lexer.h

// Load source code from file
char* load_source(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return NULL;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Allocate buffer
    char* buffer = (char*)malloc(file_size + 1);
    if (!buffer) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        return NULL;
    }
    
    // Read file contents
    size_t read_size = fread(buffer, 1, file_size, file);
    buffer[read_size] = '\0'; // Null-terminate the string
    
    fclose(file);
    return buffer;
}

// Process Python indentation to add explicit tokens for blocks
void preprocess_indentation(char** code) {
    // In a real implementation, this would convert Python's indentation-based blocks
    // to explicit tokens that the parser can handle more easily
    // For simplicity, we're just passing through for now
}

// Main frontend function
void frontend(const char* filename, ASTNode** ast_root) {
    // Load source code
    source_code = load_source(filename);
    if (!source_code) {
        return;
    }
    
    // Preprocess indentation
    preprocess_indentation(&source_code);
    
    // Set up lexer
    input = source_code;
    
    // Parse the code
    *ast_root = parseExpression();
    
    // Print AST structure (for debugging)
    printf("AST generated successfully\n");
    
    // Free source code
    free(source_code);
}
