// symbol_table.c - Symbol table implementation
#include "symbol_table.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Current scope
static int current_scope = 0;

// Symbol table - a linked list of scopes
static ScopeNode* scope_list = NULL;

// Initialize the symbol table
void init_symbol_table() {
    // Create the global scope
    enter_scope();
}

// Enter a new scope
void enter_scope() {
    // Create a new scope node
    ScopeNode* new_scope = (ScopeNode*)malloc(sizeof(ScopeNode));
    if (!new_scope) {
        fprintf(stderr, "Error: Memory allocation failed for new scope\n");
        return;
    }
    
    // Initialize the scope
    new_scope->scope_id = current_scope++;
    new_scope->symbols = NULL;
    new_scope->next = scope_list;
    
    // Add to the scope list
    scope_list = new_scope;
    
    printf("Entered scope %d\n", new_scope->scope_id);
}

// Exit the current scope
void exit_scope() {
    if (!scope_list) {
        fprintf(stderr, "Error: Cannot exit scope - no active scopes\n");
        return;
    }
    
    printf("Exiting scope %d\n", scope_list->scope_id);
    
    // Free all symbols in this scope
    SymbolNode* current = scope_list->symbols;
    while (current) {
        SymbolNode* next = current->next;
        free(current->name);
        free(current);
        current = next;
    }
    
    // Remove the scope from the list
    ScopeNode* old_scope = scope_list;
    scope_list = scope_list->next;
    free(old_scope);
}

// Add a symbol to the current scope
void add_symbol(const char* name, SymbolType type, void* data) {
    if (!scope_list) {
        fprintf(stderr, "Error: Cannot add symbol - no active scopes\n");
        return;
    }
    
    // Check if symbol already exists in current scope
    if (lookup_symbol_current_scope(name)) {
        fprintf(stderr, "Warning: Redefinition of symbol '%s' in same scope\n", name);
    }
    
    // Create a new symbol node
    SymbolNode* new_symbol = (SymbolNode*)malloc(sizeof(SymbolNode));
    if (!new_symbol) {
        fprintf(stderr, "Error: Memory allocation failed for new symbol\n");
        return;
    }
    
    // Initialize the symbol
    new_symbol->name = strdup(name);
    new_symbol->type = type;
    new_symbol->data = data;
    new_symbol->scope_id = scope_list->scope_id;
    
    // Add to the current scope's symbol list
    new_symbol->next = scope_list->symbols;
    scope_list->symbols = new_symbol;
    
    printf("Added symbol '%s' of type %d to scope %d\n", name, type, scope_list->scope_id);
}

// Look up a symbol in the current scope only
SymbolNode* lookup_symbol_current_scope(const char* name) {
    if (!scope_list) return NULL;
    
    // Search the current scope's symbol list
    SymbolNode* current = scope_list->symbols;
    while (current) {
        if (strcmp(current->name, name) == 0) {
            return current;
        }
        current = current->next;
    }
    
    return NULL;
}

// Look up a symbol in all scopes (starting from the current one)
SymbolNode* lookup_symbol(const char* name) {
    ScopeNode* current_scope = scope_list;
    
    // Search each scope starting from the innermost
    while (current_scope) {
        SymbolNode* current_symbol = current_scope->symbols;
        while (current_symbol) {
            if (strcmp(current_symbol->name, name) == 0) {
                return current_symbol;
            }
            current_symbol = current_symbol->next;
        }
        current_scope = current_scope->next;
    }
    
    return NULL;
}

// Print the current symbol table (for debugging)
void print_symbol_table() {
    printf("\n--- Symbol Table ---\n");
    
    ScopeNode* current_scope = scope_list;
    while (current_scope) {
        printf("Scope %d:\n", current_scope->scope_id);
        
        SymbolNode* current_symbol = current_scope->symbols;
        if (!current_symbol) {
            printf("  (empty)\n");
        }
        
        while (current_symbol) {
            printf("  '%s' (type: %d)\n", current_symbol->name, current_symbol->type);
            current_symbol = current_symbol->next;
        }
        
        current_scope = current_scope->next;
    }
    
    printf("------------------\n\n");
}

// Clean up the symbol table
void cleanup_symbol_table() {
    // Exit all scopes
    while (scope_list) {
        exit_scope();
    }
    
    current_scope = 0;
}
