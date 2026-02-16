#ifndef PYC_PARSER_H
#define PYC_PARSER_H

#include <stddef.h>
#include "lexer.h"

typedef enum {
    AST_NUMBER,
    AST_IDENTIFIER,
    AST_BINARY_ADD,
    AST_ASSIGNMENT,
    AST_PROGRAM
} ASTNodeType;

typedef struct ASTNode {
    ASTNodeType type;
    char* value;
    struct ASTNode* left;
    struct ASTNode* right;
    struct ASTNode** statements;
    size_t statement_count;
    int line;
} ASTNode;

ASTNode* parse_tokens(TokenArray tokens);
void free_ast(ASTNode* node);

#endif
