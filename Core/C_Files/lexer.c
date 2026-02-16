#include "lexer.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

static char* copy_lexeme(const char* start, size_t n) {
    char* s = (char*)malloc(n + 1);
    if (!s) return NULL;
    memcpy(s, start, n);
    s[n] = '\0';
    return s;
}

static void push_token(TokenArray* arr, Token token) {
    arr->data = (Token*)realloc(arr->data, sizeof(Token) * (arr->count + 1));
    arr->data[arr->count++] = token;
}

TokenArray lexical_analysis(const char* source) {
    TokenArray out = {0};
    int line = 1;

    for (const char* p = source; *p; ++p) {
        if (*p == ' ' || *p == '\t' || *p == '\r') continue;
        if (*p == '\n') {
            push_token(&out, (Token){TOKEN_NEWLINE, copy_lexeme("\\n", 2), line});
            line++;
            continue;
        }
        if (*p == '=') {
            push_token(&out, (Token){TOKEN_ASSIGN, copy_lexeme("=", 1), line});
            continue;
        }
        if (*p == '+') {
            push_token(&out, (Token){TOKEN_PLUS, copy_lexeme("+", 1), line});
            continue;
        }
        if (isdigit((unsigned char)*p)) {
            const char* start = p;
            while (isdigit((unsigned char)*(p + 1))) p++;
            push_token(&out, (Token){TOKEN_NUMBER, copy_lexeme(start, (size_t)(p - start + 1)), line});
            continue;
        }
        if (isalpha((unsigned char)*p) || *p == '_') {
            const char* start = p;
            while (isalnum((unsigned char)*(p + 1)) || *(p + 1) == '_') p++;
            push_token(&out, (Token){TOKEN_IDENTIFIER, copy_lexeme(start, (size_t)(p - start + 1)), line});
            continue;
        }

        push_token(&out, (Token){TOKEN_INVALID, copy_lexeme(p, 1), line});
    }

    push_token(&out, (Token){TOKEN_EOF, copy_lexeme("", 0), line});
    return out;
}

void free_tokens(TokenArray tokens) {
    for (size_t i = 0; i < tokens.count; ++i) {
        free(tokens.data[i].lexeme);
    }
    free(tokens.data);
}
