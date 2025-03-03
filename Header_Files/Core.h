
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>
#include <pthread.h>
#include <stdbool.h>
#include <corecrt.h>
#include "lexer.h"

#ifndef UNTITLED_CORE_H
#define UNTITLED_CORE_H

typedef struct Node {
    __attribute__((unused)) const char* type;
    __attribute__((unused)) const char* value;
    struct Node* children;
    __attribute__((unused)) size_t children_count;
} Node;

__attribute__((unused)) __attribute__((unused)) Node *parse();


#endif //UNTITLED_CORE_H
