#include "Core.h"
#include "lexer.h"
#include "parser.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

extern const char* generate_ir_string(ASTNode* ast_root);
extern void cleanup_ir_generator(void);

static ASTNode* parse_source(const char* source) {
    lexer_init(source);
    return parse();
}

static void test_assignment_literal(void) {
    ASTNode* root = parse_source("x = 1\n");
    assert(root->type == AST_PROGRAM);
    assert(root->as.program.statement_count == 1);

    ASTNode* stmt = root->as.program.statements[0];
    assert(stmt->type == AST_ASSIGNMENT);
    assert(strcmp(stmt->as.assignment.name, "x") == 0);
    assert(stmt->as.assignment.value->type == AST_INTEGER_LITERAL);
    assert(stmt->as.assignment.value->as.integer_literal.value == 1);

    const char* ir = generate_ir_string(root);
    assert(strstr(ir, "store 1, x") != NULL);

    ast_free(root);
    cleanup_ir_generator();
    printf("test_assignment_literal passed\n");
}

static void test_assignment_binary_expression(void) {
    ASTNode* root = parse_source("x = 1 + 2\n");
    ASTNode* stmt = root->as.program.statements[0];
    ASTNode* value = stmt->as.assignment.value;

    assert(value->type == AST_BINARY_EXPRESSION);
    assert(value->as.binary_expression.op == AST_OP_ADD);
    assert(value->as.binary_expression.left->type == AST_INTEGER_LITERAL);
    assert(value->as.binary_expression.right->type == AST_INTEGER_LITERAL);

    const char* ir = generate_ir_string(root);
    assert(strstr(ir, "= add 1, 2") != NULL);
    assert(strstr(ir, "store t0, x") != NULL);

    ast_free(root);
    cleanup_ir_generator();
    printf("test_assignment_binary_expression passed\n");
}

static void test_if_block_assignment(void) {
    ASTNode* root = parse_source("x = 1\nif x:\n    y = x + 1\n");
    assert(root->as.program.statement_count == 2);

    ASTNode* if_stmt = root->as.program.statements[1];
    assert(if_stmt->type == AST_IF_STATEMENT);
    assert(if_stmt->as.if_statement.condition->type == AST_IDENTIFIER);
    assert(if_stmt->as.if_statement.body_count == 1);
    assert(if_stmt->as.if_statement.body_statements[0]->type == AST_ASSIGNMENT);

    const char* ir = generate_ir_string(root);
    assert(strstr(ir, "store 1, x") != NULL);
    assert(strstr(ir, "br x, if_then_0, if_end_1") != NULL);
    assert(strstr(ir, "store t0, y") != NULL);

    ast_free(root);
    cleanup_ir_generator();
    printf("test_if_block_assignment passed\n");
}

int main(void) {
    test_assignment_literal();
    test_assignment_binary_expression();
    test_if_block_assignment();
    puts("All parser/IR tests passed");
    return 0;
}
