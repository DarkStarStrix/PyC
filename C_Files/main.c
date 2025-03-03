// make a compiler for python using llvm
// use a monolithic frontend to do the lexing and parsing
// then use the llvm api to generate the code
// for the backend when the user runs the compiler compile the code just in time
// and then check how many cores are in the users system and then run the code in parallel schedule the tasks in parallel
// and then print compiled in the terminal

// include the lexer and parser monolithic frontend
#ifndef LEXER_H
#define LEXER_H

typedef struct Token Token;

__attribute__((unused)) __attribute__((unused)) Token *tokenize();

__attribute__((unused)) __attribute__((unused)) void free_tokens(Token* tokens);

#endif // LEXER_H

#ifndef PARSER_H
#define PARSER_H

typedef struct Node Node;

__attribute__((unused)) __attribute__((unused)) __attribute__((unused)) Node *parse();

__attribute__((unused)) __attribute__((unused)) __attribute__((unused)) void free_ast(Node* ast);

#endif // PARSER_H

// frontend .h
#ifndef FRONTEND_H
#define FRONTEND_H

__attribute__((unused)) void frontend();

#endif // FRONTEND_H

// backend.h
#ifndef BACKEND_H
#define BACKEND_H

__attribute__((unused)) void backend();

#endif // BACKEND_H

// main.c
#include "frontend.h"
#include "backend.h"

// compile the code with
// gcc -o main.c frontend.c backend.c -Wall -wextra -pedantic -std=c11
