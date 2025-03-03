#include "parser.h"
#include <stdlib.h>
#include <stdio.h>

static Token currentToken;

static void getNextToken() {
    currentToken = getNextToken();
}

static ASTNode* createASTNode(ASTNodeType type) {
    ASTNode* node = (ASTNode*)malloc(sizeof(ASTNode));
    node->type = type;
    return node;
}

static ASTNode* parsePrimary() {
    ASTNode* node = NULL;
    if (currentToken.type == TOKEN_NUMBER) {
        node = createASTNode(AST_NODE_NUMBER);
        node->data.number = atoi(currentToken.value);
        getNextToken();
    } else if (currentToken.type == TOKEN_IDENTIFIER) {
        node = createASTNode(AST_NODE_IDENTIFIER);
        strcpy(node->data.identifier, currentToken.value);
        getNextToken();
    } else {
        printf("Syntax error: unexpected token %s\n", currentToken.value);
        exit(1);
    }
    return node;
}

static ASTNode* parseBinaryOpRHS(int exprPrec, ASTNode* lhs) {
    while (1) {
        int tokPrec = (currentToken.type == TOKEN_OPERATOR) ? 10 : -1;

        if (tokPrec < exprPrec) {
            return lhs;
        }

        char op = currentToken.value[0];
        getNextToken();

        ASTNode* rhs = parsePrimary();

        int nextPrec = (currentToken.type == TOKEN_OPERATOR) ? 10 : -1;
        if (tokPrec < nextPrec) {
            rhs = parseBinaryOpRHS(tokPrec + 1, rhs);
        }

        ASTNode* newLHS = createASTNode(AST_NODE_BINARY_OP);
        newLHS->data.binary_op.left = lhs;
        newLHS->data.binary_op.right = rhs;
        newLHS->data.binary_op.op = op;
        lhs = newLHS;
    }
}

ASTNode* parseExpression() {
    ASTNode* lhs = parsePrimary();
    return parseBinaryOpRHS(0, lhs);
}

void freeAST(ASTNode* node) {
    if (!node) return;
    if (node->type == AST_NODE_BINARY_OP) {
        freeAST(node->data.binary_op.left);
        freeAST(node->data.binary_op.right);
    }
    free(node);
}
