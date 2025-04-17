#ifndef LEXER_H
#define LEXER_H

#include "parser.h"

void lexer_init(const char* source);
Token get_next_token();

#endif
