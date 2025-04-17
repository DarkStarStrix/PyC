// Core/C_Files/symbol_table.c - Symbol table implementation
#include "symbol_table.h"
#include "error_handler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int current_scope = 0;
static ScopeNode* scope_list = NULL;

void init_symbol_table() {
    enter_scope();
}

void enter_scope() {
    ScopeNode* new_scope = (ScopeNode*)malloc(sizeof(ScopeNode));
    if (!new_scope) {
        report_error(0, 0, "Memory allocation failed for scope");
        return;
    }
    new_scope->scope_id = current_scope++;
    new_scope->symbols = NULL;
    new_scope->next = scope_list;
    scope_list = new_scope;
}

void exit_scope() {
    if (!scope_list) {
        report_error(0, 0, "No active scopes to exit");
        return;
    }
    SymbolNode* current = scope_list->symbols;
    while (current) {
        SymbolNode* next = current->next;
        free(current->name);
        free(current);
        current = next;
    }
    ScopeNode* old_scope = scope_list;
    scope_list = scope_list->next;
    free(old_scope);
}

void add_symbol(const char* name, SymbolType type, void* data, LLVMValueRef llvm_value) {
    if (!scope_list) {
        report_error(0, 0, "No active scopes");
        return;
    }
    if (lookup_symbol_current_scope(name)) {
        report_warning(0, 0, "Redefinition of symbol '%s' in scope %d", name, scope_list->scope_id);
    }
    SymbolNode* new_symbol = (SymbolNode*)malloc(sizeof(SymbolNode));
    if (!new_symbol) {
        report_error(0, 0, "Memory allocation failed for symbol");
        return;
    }
    new_symbol->name = strdup(name);
    new_symbol->type = type;
    new_symbol->data = data;
    new_symbol->llvm_value = llvm_value;
    new_symbol->scope_id = scope_list->scope_id;
    new_symbol->next = scope_list->symbols;
    scope_list->symbols = new_symbol;
}

SymbolNode* lookup_symbol_current_scope(const char* name) {
    if (!scope_list) return NULL;
    SymbolNode* current = scope_list->symbols;
    while (current) {
        if (strcmp(current->name, name) == 0) return current;
        current = current->next;
    }
    return NULL;
}

SymbolNode* lookup_symbol(const char* name) {
    ScopeNode* scope = scope_list;
    while (scope) {
        SymbolNode* symbol = scope->symbols;
        while (symbol) {
            if (strcmp(symbol->name, name) == 0) return symbol;
            symbol = symbol->next;
        }
        scope = scope->next;
    }
    return NULL;
}

void print_symbol_table() {
    fprintf(error_log, "\n--- Symbol Table ---\n");
    ScopeNode* scope = scope_list;
    while (scope) {
        fprintf(error_log, "Scope %d:\n", scope->scope_id);
        SymbolNode* symbol = scope->symbols;
        if (!symbol) fprintf(error_log, "  (empty)\n");
        while (symbol) {
            fprintf(error_log, "  '%s' (type: %d, LLVM: %p)\n", symbol->name, symbol->type, (void*)symbol->llvm_value);
            symbol = symbol->next;
        }
        scope = scope->next;
    }
    fprintf(error_log, "------------------\n");
}

void cleanup_symbol_table() {
    while (scope_list) exit_scope();
    current_scope = 0;
}
