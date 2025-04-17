// Core/Header_Files/symbol_table.h - Symbol table for PyC compiler
#ifndef SYMBOL_TABLE_H
#define SYMBOL_TABLE_H
#include <llvm-c/Core.h>

typedef enum {
    SYMBOL_VARIABLE,
    SYMBOL_FUNCTION,
    SYMBOL_CLASS,
    SYMBOL_MODULE,
    SYMBOL_IMPORT
} SymbolType;

typedef struct SymbolNode {
    char* name;
    SymbolType type;
    void* data;          // Type info or metadata
    LLVMValueRef llvm_value; // LLVM IR value for variables
    int scope_id;
    struct SymbolNode* next;
} SymbolNode;

typedef struct ScopeNode {
    int scope_id;
    SymbolNode* symbols;
    struct ScopeNode* next;
} ScopeNode;

void init_symbol_table();
void enter_scope();
void exit_scope();
void add_symbol(const char* name, SymbolType type, void* data, LLVMValueRef llvm_value);
SymbolNode* lookup_symbol_current_scope(const char* name);
SymbolNode* lookup_symbol(const char* name);
void print_symbol_table();
void cleanup_symbol_table();

#endif
