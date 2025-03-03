// create the lexer class

// Path: lexer.c
#include <stdio.h>
#include <ctype.h>
#include <string.h>

typedef enum {
    TOKEN_IDENTIFIER,
    TOKEN_NUMBER,
    TOKEN_OPERATOR,
    TOKEN_EOF
} TokenType;

typedef struct {
    TokenType type;
    char value[256];
} Token;

const char* input;
size_t pos = 0;

Token getNextToken() {
    Token token;
    while (isspace(input[pos])) pos++;

    if (isalpha(input[pos])) {
        size_t start = pos;
        while (isalnum(input[pos])) pos++;
        strncpy(token.value, &input[start], pos - start);
        token.value[pos - start] = '\0';
        token.type = TOKEN_IDENTIFIER;
    } else if (isdigit(input[pos])) {
        size_t start = pos;
        while (isdigit(input[pos])) pos++;
        strncpy(token.value, &input[start], pos - start);
        token.value[pos - start] = '\0';
        token.type = TOKEN_NUMBER;
    } else if (input[pos] == '\0') {
        token.type = TOKEN_EOF;
    } else {
        token.value[0] = input[pos];
        token.value[1] = '\0';
        token.type = TOKEN_OPERATOR;
        pos++;
    }

    return token;
}

int main() {
    input = "int x = 42 + y;";
    Token token;
    do {
        token = getNextToken();
        printf("Token: %s, Type: %d\n", token.value, token.type);
    } while (token.type != TOKEN_EOF);

    return 0;
}

