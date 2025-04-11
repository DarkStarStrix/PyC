#include <cstddef>
#include "lexer.h"

// Define the Node structure for the Abstract Syntax Tree (AST)
typedef struct Node {
    const char* type;
    const char* value;
    struct Node** children;
    size_t children_count;
} Node;

// Helper function to create a new AST node
Node* create_node(const char* type, const char* value, size_t children_count) {
    Node* node = new Node;
    node->type = type;
    node->value = value;
    node->children = (children_count > 0) ?
        new Node*[children_count] : nullptr;
    node->children_count = children_count;
    for (size_t i = 0; i < children_count; i++) {
        node->children[i] = nullptr;
    }
    return node;
}

// Parse function to build an AST (currently hardcoded as an example)
__attribute__((unused)) Node* parse() {
    Node* ast = create_node("FUNCTION", "print", 1);
    ast->children[0] = create_node("STRING", "'Hello, World!'", 0);
    return ast;
}

// Function to recursively free the AST memory
__attribute__((unused)) void free_ast(Node* ast) {
    if (ast == nullptr) return;
    for (size_t i = 0; i < ast->children_count; i++) {
        free_ast(ast->children[i]);
    }
    delete[] ast->children;
    delete ast;
}
