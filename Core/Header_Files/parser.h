#ifndef PARSER_H
#define PARSER_H

#include "Core.h"

typedef enum {
    TOKEN_NUMBER,
    TOKEN_IDENTIFIER,
    TOKEN_PLUS,
    TOKEN_MINUS,
    TOKEN_MULTIPLY,
    TOKEN_DIVIDE,
    TOKEN_ASSIGN,
    TOKEN_COLON,
    TOKEN_IF,
    TOKEN_ELSE,
    TOKEN_WHILE,
    TOKEN_DEF,
    TOKEN_INDENT,
    TOKEN_DEDENT,
    TOKEN_NEWLINE,
    TOKEN_EOF
} TokenType;

typedef struct {
    TokenType type;
    char value[256];
} Token;

void parser_init(const char* source);
ASTNode* parser_parse(void);
void parser_cleanup(void);

#endif // PARSER_H
