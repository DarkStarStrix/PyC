// frontend.c - Enhanced Python frontend implementation
#include "frontend.h"
#include "lexer.h"
#include "parser.h"
#include "symbol_table.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Global variables
char* source_code = NULL;
char* processed_code = NULL;

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
    if (!*code) return;
    
    // Count the lines in the original code
    size_t line_count = 1;
    for (char* c = *code; *c; c++) {
        if (*c == '\n') line_count++;
    }
    
    // Allocate arrays to track indentation levels
    int* indentation_levels = (int*)malloc(sizeof(int) * (line_count + 1));
    int current_line = 0;
    indentation_levels[0] = 0;
    
    // First pass: calculate indentation levels for each line
    char* line_start = *code;
    char* c = *code;
    while (*c) {
        // Find the start of the next line
        while (*c && *c != '\n') c++;
        
        // Calculate indentation (number of spaces at the beginning of the line)
        int spaces = 0;
        for (char* indent = line_start; indent < c && *indent == ' '; indent++) {
            spaces++;
        }
        
        // Store the indentation level for this line
        indentation_levels[current_line] = spaces / 4; // Assuming 4 spaces per indentation level
        
        // Move to next line
        if (*c == '\n') {
            c++;
            line_start = c;
            current_line++;
        }
    }
    
    // Calculate the size needed for the processed code
    // We need extra space for INDENT and DEDENT tokens
    size_t processed_size = strlen(*code) + line_count * 10; // Rough estimate
    processed_code = (char*)malloc(processed_size);
    if (!processed_code) {
        fprintf(stderr, "Error: Memory allocation failed during indentation processing\n");
        free(indentation_levels);
        return;
    }
    
    // Second pass: generate code with explicit indent/dedent tokens
    char* out = processed_code;
    line_start = *code;
    c = *code;
    current_line = 0;
    int prev_indent = 0;
    
    while (*c) {
        // Find the start of the next line
        char* line_end = c;
        while (*line_end && *line_end != '\n') line_end++;
        
        // Get the current indentation level
        int current_indent = indentation_levels[current_line];
        
        // Add INDENT tokens if needed
        while (current_indent > prev_indent) {
            strcpy(out, " INDENT ");
            out += 8;
            prev_indent++;
        }
        
        // Add DEDENT tokens if needed
        while (current_indent < prev_indent) {
            strcpy(out, " DEDENT ");
            out += 8;
            prev_indent--;
        }
        
        // Copy the line content (skipping the leading spaces)
        char* content_start = line_start;
        while (content_start < line_end && *content_start == ' ') content_start++;
        
        // Copy the actual content
        size_t content_len = line_end - content_start;
        memcpy(out, content_start, content_len);
        out += content_len;
        
        // Add newline
        if (*line_end == '\n') {
            *out++ = '\n';
            line_end++;
        }
        
        // Update pointers for next line
        c = line_start = line_end;
        current_line++;
    }
    
    // Add final DEDENTs at the end of the file
    while (prev_indent > 0) {
        strcpy(out, " DEDENT ");
        out += 8;
        prev_indent--;
    }
    
    *out = '\0'; // Null-terminate the processed code
    
    // Update the code pointer
    free(*code);
    *code = processed_code;
    
    free(indentation_levels);
}

// Process Python complex statements
void process_complex_statements(char** code) {
    // This would handle specific Python constructs like:
    // - List/dictionary comprehensions
    // - Lambda expressions
    // - Try/except blocks
    // - With statements
    // - Decorators
    
    // For now, we'll just print a placeholder message
    printf("Processing complex Python statements...\n");
}

// Main frontend function
int frontend_process_file(const char* filename, ASTNode** ast_root) {
    // Load source code
    source_code = load_source(filename);
    if (!source_code) {
        return 0;
    }
    
    // Initialize symbol table
    symbol_table_init();
    
    // Preprocess indentation
    preprocess_indentation(&source_code);
    
    // Process complex statements
    process_complex_statements(&source_code);
    
    // Parse the code
    parser_init(source_code);
    *ast_root = parser_parse();
    parser_cleanup();
    
    // Print AST structure (for debugging)
    if (*ast_root) {
        printf("AST generated successfully\n");
    } else {
        fprintf(stderr, "Error: AST generation failed\n");
        return 0;
    }

    return 1;
}

// Helper function to detect Python file by extension
int frontend_is_python_file(const char* filename) {
    const char* extension = strrchr(filename, '.');
    if (!extension) return 0;
    
    return strcmp(extension, ".py") == 0;
}


int frontend_init(void) {
    source_code = NULL;
    processed_code = NULL;
    return 1;
}

void frontend_cleanup(void) {
    free(source_code);
    source_code = NULL;
    processed_code = NULL;
}
