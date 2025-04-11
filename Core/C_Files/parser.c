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

ASTNode* parse_if_statement() {
    expect_token(TOKEN_IF);
    ASTNode* condition = parse_expression();
    expect_token(TOKEN_COLON);
    ASTNode* then_block = parse_block();
    ASTNode* else_block = NULL;
    if (current_token.type == TOKEN_ELSE) {
        expect_token(TOKEN_ELSE);
        expect_token(TOKEN_COLON);
        else_block = parse_block();
    }
    return create_if_node(condition, then_block, else_block);
}

ASTNode* parse_block() {
    expect_token(TOKEN_INDENT);
    ASTNode* block = create_block_node();
    while (current_token.type != TOKEN_DEDENT) {
        ASTNode* stmt = parse_statement();
        add_to_block(block, stmt);
    }
    expect_token(TOKEN_DEDENT);
    return block;
}
