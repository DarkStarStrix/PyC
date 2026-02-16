#include "semantic.h"

#include "symbol_table.h"
#include "error_handler.h"

static void check_expr(ASTNode* n) {
    if (!n) return;
    if (n->type == AST_IDENTIFIER && !symbol_exists(n->value)) {
        add_error("line %d: use of undefined symbol '%s'", n->line, n->value);
    }
    check_expr(n->left);
    check_expr(n->right);
}

int perform_semantic_analysis(ASTNode* ast) {
    if (!ast || ast->type != AST_PROGRAM) {
        add_error("semantic analysis: invalid AST");
        return -1;
    }

    for (size_t i = 0; i < ast->statement_count; ++i) {
        ASTNode* stmt = ast->statements[i];
        check_expr(stmt->right);
        if (has_errors()) {
            return -1;
        }
        if (symbol_define(stmt->value) != 0) {
            add_error("symbol table full while defining '%s'", stmt->value);
            return -1;
        }
    }
    return 0;
}
