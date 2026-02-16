#include "ir_generator.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void append(char** buf, size_t* len, const char* s) {
    size_t n = strlen(s);
    *buf = (char*)realloc(*buf, *len + n + 1);
    memcpy(*buf + *len, s, n + 1);
    *len += n;
}

static void emit_expr(ASTNode* expr, char* out, size_t out_size) {
    if (expr->type == AST_NUMBER || expr->type == AST_IDENTIFIER) {
        snprintf(out, out_size, "%s", expr->value);
        return;
    }
    if (expr->type == AST_BINARY_ADD) {
        char l[128], r[128];
        emit_expr(expr->left, l, sizeof(l));
        emit_expr(expr->right, r, sizeof(r));
        snprintf(out, out_size, "add i64 %s, %s", l, r);
        return;
    }
    snprintf(out, out_size, "0");
}

IRCode* generate_ir(ASTNode* ast) {
    if (!ast || ast->type != AST_PROGRAM) {
        return NULL;
    }

    IRCode* ir = (IRCode*)calloc(1, sizeof(IRCode));
    if (!ir) return NULL;

    size_t len = 0;
    append(&ir->text, &len, "; ModuleID = 'pyc'\n");
    append(&ir->text, &len, "define i64 @main() {\nentry:\n");

    for (size_t i = 0; i < ast->statement_count; ++i) {
        ASTNode* stmt = ast->statements[i];
        char expr[256];
        emit_expr(stmt->right, expr, sizeof(expr));
        char line[320];
        snprintf(line, sizeof(line), "  ; %s = %s\n", stmt->value, expr);
        append(&ir->text, &len, line);
    }

    append(&ir->text, &len, "  ret i64 0\n}\n");
    return ir;
}

void free_ir(IRCode* ir) {
    if (!ir) return;
    free(ir->text);
    free(ir);
}
