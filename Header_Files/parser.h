// parser.h

#ifndef PARSER_H
#define PARSER_H

#include "lexer.h"

typedef enum {
    AST_NODE_NUMBER __attribute__((unused)),
    AST_NODE_IDENTIFIER __attribute__((unused)),
    AST_NODE_BINARY_OP __attribute__((unused))
} ASTNodeType;

typedef struct ASTNode {
    __attribute__((unused)) ASTNodeType type;
    __attribute__((unused)) union {
        struct {
            struct ASTNode* left;
            struct ASTNode* right;
            char op;
        } binary_op;
        char identifier[256];
        int number;
    } data;
} ASTNode;

__attribute__((unused)) ASTNode* parseExpression();

__attribute__((unused)) void freeAST(ASTNode* node);

#endif // PARSER_H
