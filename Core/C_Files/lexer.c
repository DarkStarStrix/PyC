#include "lexer.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char* source;
    size_t pos;
    int indent_stack[128];
    int indent_top;
    int pending_dedents;
    int at_line_start;
} Lexer;

static Lexer lexer;

void lexer_init(const char* source) {
    lexer.source = source;
    lexer.pos = 0;
    lexer.indent_stack[0] = 0;
    lexer.indent_top = 0;
    lexer.pending_dedents = 0;
    lexer.at_line_start = 1;
}

static void make_token(Token* token, TokenType type, const char* value) {
    token->type = type;
    strncpy(token->value, value, sizeof(token->value) - 1);
    token->value[sizeof(token->value) - 1] = '\0';
}

static int current_indent(void) {
    return lexer.indent_stack[lexer.indent_top];
}

static int consume_indent_width(void) {
    int width = 0;
    while (lexer.source[lexer.pos] == ' ') {
        width++;
        lexer.pos++;
    }
    if (lexer.source[lexer.pos] == '\t') {
        fprintf(stderr, "Tabs are not supported for indentation\n");
        exit(1);
    }
    return width;
}

Token get_next_token(void) {
    Token token = {0};

    if (lexer.pending_dedents > 0) {
        lexer.pending_dedents--;
        make_token(&token, TOKEN_DEDENT, "DEDENT");
        return token;
    }

    while (1) {
        if (lexer.at_line_start) {
            int indent = consume_indent_width();
            if (lexer.source[lexer.pos] == '\n') {
                lexer.pos++;
                make_token(&token, TOKEN_NEWLINE, "\\n");
                return token;
            }

            if (indent > current_indent()) {
                lexer.indent_stack[++lexer.indent_top] = indent;
                lexer.at_line_start = 0;
                make_token(&token, TOKEN_INDENT, "INDENT");
                return token;
            }

            if (indent < current_indent()) {
                while (lexer.indent_top > 0 && indent < current_indent()) {
                    lexer.indent_top--;
                    lexer.pending_dedents++;
                }
                if (indent != current_indent()) {
                    fprintf(stderr, "Invalid indentation\n");
                    exit(1);
                }
                if (lexer.pending_dedents > 0) {
                    lexer.pending_dedents--;
                    make_token(&token, TOKEN_DEDENT, "DEDENT");
                    return token;
                }
            }

            lexer.at_line_start = 0;
        }

        char c = lexer.source[lexer.pos];

        if (c == '\0') {
            if (lexer.indent_top > 0) {
                lexer.indent_top--;
                make_token(&token, TOKEN_DEDENT, "DEDENT");
                return token;
            }
            make_token(&token, TOKEN_EOF, "EOF");
            return token;
        }

        if (c == '#') {
            while (lexer.source[lexer.pos] != '\0' && lexer.source[lexer.pos] != '\n') {
                lexer.pos++;
            }
            continue;
        }

        if (c == '\n') {
            lexer.pos++;
            lexer.at_line_start = 1;
            make_token(&token, TOKEN_NEWLINE, "\\n");
            return token;
        }

        if (c == ' ') {
            lexer.pos++;
            continue;
        }

        if (isalpha((unsigned char)c) || c == '_') {
            size_t start = lexer.pos;
            while (isalnum((unsigned char)lexer.source[lexer.pos]) || lexer.source[lexer.pos] == '_') {
                lexer.pos++;
            }
            size_t len = lexer.pos - start;
            if (len >= sizeof(token.value)) len = sizeof(token.value) - 1;
            memcpy(token.value, lexer.source + start, len);
            token.value[len] = '\0';
            token.type = strcmp(token.value, "if") == 0 ? TOKEN_IF : TOKEN_IDENTIFIER;
            return token;
        }

        if (isdigit((unsigned char)c)) {
            size_t start = lexer.pos;
            while (isdigit((unsigned char)lexer.source[lexer.pos])) {
                lexer.pos++;
            }
            size_t len = lexer.pos - start;
            if (len >= sizeof(token.value)) len = sizeof(token.value) - 1;
            memcpy(token.value, lexer.source + start, len);
            token.value[len] = '\0';
            token.type = TOKEN_NUMBER;
            return token;
        }

        lexer.pos++;
        switch (c) {
            case '+': make_token(&token, TOKEN_PLUS, "+"); return token;
            case '-': make_token(&token, TOKEN_MINUS, "-"); return token;
            case '*': make_token(&token, TOKEN_MULTIPLY, "*"); return token;
            case '/': make_token(&token, TOKEN_DIVIDE, "/"); return token;
            case '=': make_token(&token, TOKEN_ASSIGN, "="); return token;
            case ':': make_token(&token, TOKEN_COLON, ":"); return token;
            default:
                fprintf(stderr, "Unknown character: %c\n", c);
                exit(1);
        }
    }
}

Token lexer_peek_token(void) {
    Lexer saved = lexer;
    Token token = get_next_token();
    lexer = saved;
    return token;
}
