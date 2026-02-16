#ifndef SYMBOL_TABLE_H
#define SYMBOL_TABLE_H

#define MAX_SYMBOLS 256
#define MAX_NAME 64

typedef struct {
    char name[MAX_NAME];
    char return_type[MAX_NAME];
    char** param_types;
    int num_params;
} FunctionSymbol;

typedef struct {
    char name[MAX_NAME];
    char type[MAX_NAME];
    char element_type[MAX_NAME];
    int dimensions;
} ComplexType;

typedef struct {
    char name[MAX_NAME];
    char type[MAX_NAME];
    void* llvm_value;
} VariableSymbol;

void symbol_table_init(void);
void symbol_table_cleanup(void);
void add_variable(const char* name, const char* type, void* llvm_value);
VariableSymbol* lookup_variable(const char* name);
void add_function(const char* name, const char* return_type, char** param_types, int num_params);
FunctionSymbol* lookup_function(const char* name);
void add_complex_type(const char* name, const char* type, const char* element_type, int dimensions);
ComplexType* lookup_complex_type(const char* name);
void enter_scope(void);
void exit_scope(void);

#endif // SYMBOL_TABLE_H
