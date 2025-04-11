// symbol_table.h - Symbol table header
#ifndef SYMBOL_TABLE_H
#define SYMBOL_TABLE_H

// Symbol types
typedef enum {
    SYMBOL_VARIABLE,
    SYMBOL_FUNCTION,
    SYMBOL_CLASS,
    SYMBOL_MODULE,
    SYMBOL_IMPORT
} SymbolType;

// Symbol node structure
typedef struct SymbolNode {
    char* name;
    SymbolType type;
    void* data;          // Additional data associated with the symbol
    int scope_id;        // Scope where this symbol is defined
    struct SymbolNode* next;
} SymbolNode;

// Scope node structure
typedef struct ScopeNode {
    int scope_id;
    struct SymbolNode* symbols;
    struct ScopeNode* next;
} ScopeNode;

// Initialize the symbol table
void init_symbol_table();

// Enter a new scope
void enter_scope();

// Exit the current scope
void exit_scope();

// Add a symbol to the current scope
void add_symbol(const char* name, SymbolType type, void* data);

// Look up a symbol in the current scope only
SymbolNode* lookup_symbol_current_scope(const char* name);

// Look up a symbol in all scopes (starting from the current one)
SymbolNode* lookup_symbol(const char* name);

// Print the current symbol table (for debugging)
void print_symbol_table();

// Clean up the symbol table
void cleanup_symbol_table();

#endif // SYMBOL_TABLE_H
