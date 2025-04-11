// test_parser.c

#include "parser.h"
#include <assert.h>
#include <stdio.h>

void test_parseNumber() {
    input = "42";
    getNextToken();
    ASTNode* node = parseExpression();
    assert(node->type == AST_NODE_NUMBER);
    assert(node->data.number == 42);
    freeAST(node);
    printf("test_parseNumber passed\n");
}

void test_parseIdentifier() {
    input = "x";
    getNextToken();
    ASTNode* node = parseExpression();
    assert(node->type == AST_NODE_IDENTIFIER);
    assert(strcmp(node->data.identifier, "x") == 0);
    freeAST(node);
    printf("test_parseIdentifier passed\n");
}

void test_parseBinaryOp() {
    input = "x + 42";
    getNextToken();
    ASTNode* node = parseExpression();
    assert(node->type == AST_NODE_BINARY_OP);
    assert(node->data.binary_op.op == '+');
    assert(node->data.binary_op.left->type == AST_NODE_IDENTIFIER);
    assert(strcmp(node->data.binary_op.left->data.identifier, "x") == 0);
    assert(node->data.binary_op.right->type == AST_NODE_NUMBER);
    assert(node->data.binary_op.right->data.number == 42);
    freeAST(node);
    printf("test_parseBinaryOp passed\n");
}

int main() {
    test_parseNumber();
    test_parseIdentifier();
    test_parseBinaryOp();
    return 0;
}