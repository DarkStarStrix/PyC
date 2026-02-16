#ifndef PYC_LEXER_H
#define PYC_LEXER_H

#include <stddef.h>

typedef enum {
    TOKEN_IDENTIFIER,
    TOKEN_NUMBER,
    TOKEN_ASSIGN,
    TOKEN_PLUS,
    TOKEN_NEWLINE,
    TOKEN_EOF,
    TOKEN_INVALID
} TokenType;

typedef struct {
    TokenType type;
    char* lexeme;
    int line;
} Token;

typedef struct {
    Token* data;
    size_t count;
} TokenArray;

TokenArray lexical_analysis(const char* source);
void free_tokens(TokenArray tokens);

#endif // LEXER_H
