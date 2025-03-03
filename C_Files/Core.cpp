//
// Created by kunya on 9/18/2024.
//


#include <corecrt.h>
#include "lexer.h"

typedef struct Node {
    const char* type;
    const char* value;
    struct Node* children;
    size_t children_count;
} Node;

__attribute__((unused)) Node *parse() {
    // Implementation of the parse function
    Node* ast = static_cast<Node *>(malloc(sizeof(Node)));
    ast->type = "FUNCTION";
    ast->value = "print";
    ast->children = malloc(sizeof(Node));
    ast->children[0].type = "STRING";
    ast->children[0].value = "'Hello, World!'";
    ast->children_count = 1;
    return ast;
}

__attribute__((unused)) void free_ast(Node* ast) {
    // Implementation of the free_ast function
    free(ast->children);
    free(ast);
}

