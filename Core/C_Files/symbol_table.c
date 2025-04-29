// Core/C_Files/symbol_table.c - Symbol table implementation
#include "symbol_table.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static VariableSymbol variables[MAX_SYMBOLS];
static FunctionSymbol functions[MAX_SYMBOLS];
static ComplexType complex_types[MAX_SYMBOLS];
static int var_count = 0;
static int func_count = 0;
static int type_count = 0;

// Add scope management
#define MAX_SCOPE_DEPTH 32
static int current_scope = 0;
static int scope_stack[MAX_SCOPE_DEPTH];

// Add type validation flags
typedef struct {
    int is_const;
    int is_volatile;
    int is_static;
    int array_dims[3];
    int is_pointer;
    int pointer_depth;
} TypeAttributes;

// Enhanced variable symbol
typedef struct {
    char name[MAX_NAME];
    char type[MAX_NAME];
    void* llvm_value;
    int scope_level;
    TypeAttributes attrs;
} EnhancedVariableSymbol;

static EnhancedVariableSymbol enhanced_variables[MAX_SYMBOLS];

// Add template support
typedef struct {
    char name[MAX_NAME];
    char template_params[MAX_SYMBOLS][MAX_NAME];
    int num_template_params;
    ComplexType base_type;
} TemplateType;

static TemplateType template_types[MAX_SYMBOLS];
static int template_count = 0;

// Add function overloading support
typedef struct {
    char name[MAX_NAME];
    FunctionSymbol* overloads;
    int num_overloads;
} OverloadedFunction;

static OverloadedFunction overloaded_functions[MAX_SYMBOLS];
static int overload_count = 0;

// Initialize the symbol table
void symbol_table_init(void) {
    var_count = 0;
    func_count = 0;
    type_count = 0;
    template_count = 0;
    overload_count = 0;
    current_scope = 0;
}

// Add a variable to the symbol table
void add_variable(const char* name, const char* type, void* llvm_value) {
    if (var_count >= MAX_SYMBOLS) {
        fprintf(stderr, "Symbol table overflow\n");
        exit(1);
    }
    strncpy(variables[var_count].name, name, MAX_NAME - 1);
    strncpy(variables[var_count].type, type, MAX_NAME - 1);
    variables[var_count].llvm_value = llvm_value;
    var_count++;
}

// Lookup a variable by name
VariableSymbol* lookup_variable(const char* name) {
    for (int i = 0; i < var_count; i++) {
        if (strcmp(variables[i].name, name) == 0) return &variables[i];
    }
    return NULL;
}

// Add a function to the symbol table
void add_function(const char* name, const char* return_type, char** param_types, int num_params) {
    if (func_count >= MAX_SYMBOLS) {
        fprintf(stderr, "Function table overflow\n");
        exit(1);
    }
    FunctionSymbol* fs = &functions[func_count];
    strncpy(fs->name, name, MAX_NAME - 1);
    strncpy(fs->return_type, return_type, MAX_NAME - 1);
    fs->param_types = malloc(num_params * sizeof(char*));
    fs->num_params = num_params;
    for (int i = 0; i < num_params; i++) {
        fs->param_types[i] = strdup(param_types[i]);
    }
    func_count++;
}

// Lookup a function by name
FunctionSymbol* lookup_function(const char* name) {
    for (int i = 0; i < func_count; i++) {
        if (strcmp(functions[i].name, name) == 0) return &functions[i];
    }
    return NULL;
}

// Add a complex type to the symbol table
void add_complex_type(const char* name, const char* type, const char* element_type, int dimensions) {
    if (type_count >= MAX_SYMBOLS) {
        fprintf(stderr, "Type table overflow\n");
        exit(1);
    }
    ComplexType* ct = &complex_types[type_count];
    strncpy(ct->name, name, MAX_NAME - 1);
    strncpy(ct->type, type, MAX_NAME - 1);
    strncpy(ct->element_type, element_type, MAX_NAME - 1);
    ct->dimensions = dimensions;
    type_count++;
}

// Lookup a complex type by name
ComplexType* lookup_complex_type(const char* name) {
    for (int i = 0; i < type_count; i++) {
        if (strcmp(complex_types[i].name, name) == 0) return &complex_types[i];
    }
    return NULL;
}

// Enter a new scope
void enter_scope(void) {
    if (current_scope >= MAX_SCOPE_DEPTH - 1) {
        fprintf(stderr, "Maximum scope depth exceeded\n");
        exit(1);
    }
    scope_stack[current_scope++] = var_count;
}

// Exit the current scope
void exit_scope(void) {
    if (current_scope <= 0) return;
    var_count = scope_stack[--current_scope];
}

// Add an enhanced variable to the symbol table
void add_enhanced_variable(const char* name, const char* type, void* llvm_value, TypeAttributes attrs) {
    if (var_count >= MAX_SYMBOLS) {
        fprintf(stderr, "Symbol table overflow\n");
        exit(1);
    }
    
    EnhancedVariableSymbol* sym = &enhanced_variables[var_count];
    strncpy(sym->name, name, MAX_NAME - 1);
    strncpy(sym->type, type, MAX_NAME - 1);
    sym->llvm_value = llvm_value;
    sym->scope_level = current_scope;
    sym->attrs = attrs;
    var_count++;
}

// Add a template type to the symbol table
void add_template_type(const char* name, char** params, int num_params, ComplexType base) {
    if (template_count >= MAX_SYMBOLS) return;
    
    TemplateType* tt = &template_types[template_count];
    strncpy(tt->name, name, MAX_NAME - 1);
    tt->num_template_params = num_params;
    
    for (int i = 0; i < num_params; i++) {
        strncpy(tt->template_params[i], params[i], MAX_NAME - 1);
    }
    
    tt->base_type = base;
    template_count++;
}

// Add a function overload to the symbol table
void add_function_overload(const char* name, FunctionSymbol new_overload) {
    // Find existing overload set or create new one
    OverloadedFunction* of = NULL;
    for (int i = 0; i < overload_count; i++) {
        if (strcmp(overloaded_functions[i].name, name) == 0) {
            of = &overloaded_functions[i];
            break;
        }
    }
    
    if (!of && overload_count < MAX_SYMBOLS) {
        of = &overloaded_functions[overload_count++];
        strncpy(of->name, name, MAX_NAME - 1);
        of->overloads = malloc(sizeof(FunctionSymbol));
        of->num_overloads = 0;
    }
    
    if (of) {
        of->overloads = realloc(of->overloads, (of->num_overloads + 1) * sizeof(FunctionSymbol));
        of->overloads[of->num_overloads++] = new_overload;
    }
}

// Free the memory allocated for the symbol table
void symbol_table_free(void) {
    for (int i = 0; i < func_count; i++) {
        for (int j = 0; j < functions[i].num_params; j++) {
            free(functions[i].param_types[j]);
        }
        free(functions[i].param_types);
    }
    
    // Free overloaded functions
    for (int i = 0; i < overload_count; i++) {
        free(overloaded_functions[i].overloads);
    }
}
