#ifndef CORE_H
#define CORE_H

typedef enum {
    NODE_EXPRESSION,
    NODE_ASSIGNMENT,
    NODE_IF_STATEMENT,
    NODE_BLOCK
} NodeType;

typedef enum {
    EXPR_NUMBER,
    EXPR_VARIABLE,
    EXPR_BINARY_OP
} ExprType;

typedef struct ASTNode {
    NodeType type;
    union {
        struct {
            ExprType type;
            char value[256];
            struct ASTNode* left;
            struct ASTNode* right;
            int op; // TOKEN_PLUS, TOKEN_MINUS, etc.
        } expr;
        struct {
            char name[256];
            struct ASTNode* value;
        } assign;
        struct {
            struct ASTNode* condition;
            struct ASTNode* body;
            struct ASTNode* else_body;
        } if_stmt;
        struct {
            struct ASTNode** statements;
            int num_statements;
        } block;
    };
} ASTNode;

#endif
