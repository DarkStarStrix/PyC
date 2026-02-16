#ifndef LEXER_H
#define LEXER_H

#include "parser.h"

void lexer_init(const char* source);
void lexer_cleanup(void);
Token get_next_token(void);
Token lexer_peek_token(void);
int lexer_current_line(void);

#endif // LEXER_H
