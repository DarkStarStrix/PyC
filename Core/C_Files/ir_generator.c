#include "Core.h"
#include "symbol_table.h"
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char* data;
    size_t len;
    size_t cap;
    int temp_id;
    int label_id;
} IRState;

static IRState ir_state;

static char* ir_strdup(const char* src) {
    size_t n = strlen(src) + 1;
    char* out = (char*)malloc(n);
    if (!out) {
        fprintf(stderr, "Out of memory while generating IR\n");
        exit(1);
    }
    memcpy(out, src, n);
    return out;
}


static void ensure_capacity(size_t extra) {
    size_t needed = ir_state.len + extra + 1;
    if (needed <= ir_state.cap) return;
    size_t new_cap = ir_state.cap == 0 ? 512 : ir_state.cap;
    while (new_cap < needed) new_cap *= 2;
    char* resized = (char*)realloc(ir_state.data, new_cap);
    if (!resized) {
        fprintf(stderr, "Out of memory while generating IR\n");
        exit(1);
    }
    ir_state.data = resized;
    ir_state.cap = new_cap;
}

static void emit(const char* text) {
    size_t n = strlen(text);
    ensure_capacity(n);
    memcpy(ir_state.data + ir_state.len, text, n);
    ir_state.len += n;
    ir_state.data[ir_state.len] = '\0';
}

static void emitf(const char* fmt, ...) {
    char buffer[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    emit(buffer);
}

static char* next_temp(void) {
    char name[32];
    snprintf(name, sizeof(name), "t%d", ir_state.temp_id++);
    return ir_strdup(name);
}

static char* next_label(const char* prefix) {
    char name[32];
    snprintf(name, sizeof(name), "%s_%d", prefix, ir_state.label_id++);
    return ir_strdup(name);
}

static char* generate_expr(ASTNode* node) {
    if (node->type == AST_INTEGER_LITERAL) {
        char lit[32];
        snprintf(lit, sizeof(lit), "%d", node->as.integer_literal.value);
        return ir_strdup(lit);
    }

    if (node->type == AST_IDENTIFIER) {
        if (!lookup_variable(node->as.identifier.name)) {
            fprintf(stderr, "Undefined identifier in IR generation: %s\n", node->as.identifier.name);
            exit(1);
        }
        return ir_strdup(node->as.identifier.name);
    }

    if (node->type == AST_BINARY_EXPRESSION) {
        char* left = generate_expr(node->as.binary_expression.left);
        char* right = generate_expr(node->as.binary_expression.right);
        char* out = next_temp();

        const char* op = "add";
        switch (node->as.binary_expression.op) {
            case AST_OP_ADD: op = "add"; break;
            case AST_OP_SUB: op = "sub"; break;
            case AST_OP_MUL: op = "mul"; break;
            case AST_OP_DIV: op = "div"; break;
        }
        emitf("%s = %s %s, %s\n", out, op, left, right);

        free(left);
        free(right);
        return out;
    }

    fprintf(stderr, "Unsupported expression node in IR generation\n");
    exit(1);
}

static void generate_statement(ASTNode* node) {
    if (node->type == AST_ASSIGNMENT) {
        char* value = generate_expr(node->as.assignment.value);
        if (!lookup_variable(node->as.assignment.name)) {
            add_variable(node->as.assignment.name, "int", NULL);
        }
        emitf("store %s, %s\n", value, node->as.assignment.name);
        free(value);
        return;
    }

    if (node->type == AST_IF_STATEMENT) {
        char* condition = generate_expr(node->as.if_statement.condition);
        char* then_label = next_label("if_then");
        char* end_label = next_label("if_end");

        emitf("br %s, %s, %s\n", condition, then_label, end_label);
        emitf("%s:\n", then_label);

        for (size_t i = 0; i < node->as.if_statement.body_count; ++i) {
            generate_statement(node->as.if_statement.body_statements[i]);
        }

        emitf("jmp %s\n", end_label);
        emitf("%s:\n", end_label);

        free(condition);
        free(then_label);
        free(end_label);
        return;
    }

    fprintf(stderr, "Unsupported statement node in IR generation\n");
    exit(1);
}

const char* generate_ir_string(ASTNode* ast_root) {
    if (!ast_root || ast_root->type != AST_PROGRAM) {
        fprintf(stderr, "AST root must be AST_PROGRAM\n");
        exit(1);
    }

    symbol_table_init();
    ir_state.len = 0;
    ir_state.temp_id = 0;
    ir_state.label_id = 0;
    emit("");

    for (size_t i = 0; i < ast_root->as.program.statement_count; ++i) {
        generate_statement(ast_root->as.program.statements[i]);
    }

    return ir_state.data;
}

void generate_ir(ASTNode* ast_root) {
    const char* ir = generate_ir_string(ast_root);
    printf("%s", ir);
}

void cleanup_ir_generator(void) {
    free(ir_state.data);
    ir_state.data = NULL;
    ir_state.cap = 0;
    ir_state.len = 0;
    symbol_table_free();
}
