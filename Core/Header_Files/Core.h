#ifndef CORE_H
#define CORE_H

#include <stddef.h>

typedef enum {
    AST_PROGRAM,
    AST_ASSIGNMENT,
    AST_IF_STATEMENT,
    AST_INTEGER_LITERAL,
    AST_IDENTIFIER,
    AST_BINARY_EXPRESSION
} ASTNodeType;

typedef enum {
    AST_OP_ADD,
    AST_OP_SUB,
    AST_OP_MUL,
    AST_OP_DIV
} ASTBinaryOperator;

typedef struct ASTNode ASTNode;

struct ASTNode {
    ASTNodeType type;
    union {
        struct {
            ASTNode** statements;
            size_t statement_count;
            size_t statement_capacity;
        } program;
        struct {
            char name[256];
            ASTNode* value;
        } assignment;
        struct {
            ASTNode* condition;
            ASTNode** body_statements;
            size_t body_count;
            size_t body_capacity;
        } if_statement;
        struct {
            int value;
        } integer_literal;
        struct {
            char name[256];
        } identifier;
        struct {
            ASTBinaryOperator op;
            ASTNode* left;
            ASTNode* right;
        } binary_expression;
    } as;
};

void ast_free(ASTNode* node);

#endif
