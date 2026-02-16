#include "lexer.h"
#include "error_handler.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char* source;
    size_t pos;
    int line;
    int indent_stack[100];
    int indent_top;
    int dedent_count;
    int at_start_of_line;
} Lexer;

static Lexer lexer;

void lexer_init(const char* source) {
    lexer.source = source;
    lexer.pos = 0;
    lexer.line = 1;
    lexer.indent_stack[0] = 0;
    lexer.indent_top = 0;
    lexer.dedent_count = 0;
    lexer.at_start_of_line = 1;
}

void lexer_cleanup(void) {
    memset(&lexer, 0, sizeof(lexer));
}

int lexer_current_line(void) {
    return lexer.line;
}

static int calculate_indent_level(void) {
    int indent = 0;
    while (lexer.source[lexer.pos] == ' ') {
        indent++;
        lexer.pos++;
    }
    if (lexer.source[lexer.pos] == '\t') {
        report_error(lexer.line, 1, "Tabs are not allowed; use spaces for indentation");
        exit(1);
    }
    return indent;
}

static void handle_indentation(Token* token) {
    int indent = calculate_indent_level();
    if (indent > lexer.indent_stack[lexer.indent_top]) {
        lexer.indent_stack[++lexer.indent_top] = indent;
        token->type = TOKEN_INDENT;
        strcpy(token->value, "INDENT");
    } else if (indent < lexer.indent_stack[lexer.indent_top]) {
        while (indent < lexer.indent_stack[lexer.indent_top]) {
            lexer.indent_top--;
            lexer.dedent_count++;
        }
        if (indent != lexer.indent_stack[lexer.indent_top]) {
            report_error(lexer.line, 1, "Unindent does not match any outer indentation level");
            exit(1);
        }
        token->type = TOKEN_DEDENT;
        strcpy(token->value, "DEDENT");
        lexer.dedent_count--;
    } else {
        token->type = TOKEN_NEWLINE;
        strcpy(token->value, "\\n");
    }
}

Token get_next_token(void) {
    Token token = {0};

    if (lexer.dedent_count > 0) {
        token.type = TOKEN_DEDENT;
        strcpy(token.value, "DEDENT");
        lexer.dedent_count--;
        return token;
    }

    while (isspace((unsigned char)lexer.source[lexer.pos])) {
        if (lexer.source[lexer.pos] == '\n') {
            lexer.pos++;
            lexer.line++;
            lexer.at_start_of_line = 1;
            token.type = TOKEN_NEWLINE;
            strcpy(token.value, "\\n");
            return token;
        } else if (lexer.at_start_of_line) {
            handle_indentation(&token);
            lexer.at_start_of_line = 0;
            if (token.type != TOKEN_NEWLINE) {
                return token;
            }
        } else {
            lexer.pos++;
        }
    }

    if (lexer.source[lexer.pos] == '\0') {
        token.type = TOKEN_EOF;
        strcpy(token.value, "EOF");
        while (lexer.indent_top > 0) {
            lexer.dedent_count++;
            lexer.indent_top--;
        }
        return token;
    }

    if (lexer.source[lexer.pos] == '#') {
        while (lexer.source[lexer.pos] && lexer.source[lexer.pos] != '\n') {
            lexer.pos++;
        }
        return get_next_token();
    }

    if (isalpha((unsigned char)lexer.source[lexer.pos])) {
        const char* start = &lexer.source[lexer.pos];
        while (isalnum((unsigned char)lexer.source[lexer.pos]) || lexer.source[lexer.pos] == '_') {
            lexer.pos++;
        }
        int length = (int)(&lexer.source[lexer.pos] - start);
        strncpy(token.value, start, (size_t)length);
        token.value[length] = '\0';

        if (strcmp(token.value, "if") == 0) token.type = TOKEN_IF;
        else if (strcmp(token.value, "else") == 0) token.type = TOKEN_ELSE;
        else if (strcmp(token.value, "while") == 0) token.type = TOKEN_WHILE;
        else if (strcmp(token.value, "def") == 0) token.type = TOKEN_DEF;
        else token.type = TOKEN_IDENTIFIER;
        return token;
    }

    if (isdigit((unsigned char)lexer.source[lexer.pos])) {
        const char* start = &lexer.source[lexer.pos];
        while (isdigit((unsigned char)lexer.source[lexer.pos])) {
            lexer.pos++;
        }
        int length = (int)(&lexer.source[lexer.pos] - start);
        strncpy(token.value, start, (size_t)length);
        token.value[length] = '\0';
        token.type = TOKEN_NUMBER;
        return token;
    }

    switch (lexer.source[lexer.pos]) {
        case '+': token.type = TOKEN_PLUS; strcpy(token.value, "+"); break;
        case '-': token.type = TOKEN_MINUS; strcpy(token.value, "-"); break;
        case '*': token.type = TOKEN_MULTIPLY; strcpy(token.value, "*"); break;
        case '/': token.type = TOKEN_DIVIDE; strcpy(token.value, "/"); break;
        case '=': token.type = TOKEN_ASSIGN; strcpy(token.value, "="); break;
        case ':': token.type = TOKEN_COLON; strcpy(token.value, ":"); break;
        default:
            report_error(lexer.line, 1, "Unknown character");
            exit(1);
    }
    lexer.pos++;
    return token;
}

Token lexer_peek_token(void) {
    Lexer saved = lexer;
    Token token = get_next_token();
    lexer = saved;
    return token;
}
