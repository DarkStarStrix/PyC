#include "parser.h"

#include <stdlib.h>
#include <string.h>

static ASTNode* make_node(ASTNodeType type, const char* value, int line) {
    ASTNode* node = (ASTNode*)calloc(1, sizeof(ASTNode));
    if (!node) return NULL;
    node->type = type;
    node->line = line;
    if (value) {
        node->value = strdup(value);
    }
    return node;
}

static ASTNode* parse_expression(TokenArray tokens, size_t* i) {
    ASTNode* lhs = NULL;
    Token current = tokens.data[*i];
    if (current.type == TOKEN_NUMBER) lhs = make_node(AST_NUMBER, current.lexeme, current.line);
    if (current.type == TOKEN_IDENTIFIER) lhs = make_node(AST_IDENTIFIER, current.lexeme, current.line);
    if (!lhs) return NULL;
    (*i)++;

    if (tokens.data[*i].type == TOKEN_PLUS) {
        (*i)++;
        Token rhs_tok = tokens.data[*i];
        ASTNode* rhs = NULL;
        if (rhs_tok.type == TOKEN_NUMBER) rhs = make_node(AST_NUMBER, rhs_tok.lexeme, rhs_tok.line);
        if (rhs_tok.type == TOKEN_IDENTIFIER) rhs = make_node(AST_IDENTIFIER, rhs_tok.lexeme, rhs_tok.line);
        if (!rhs) {
            free_ast(lhs);
            return NULL;
        }
        ASTNode* add = make_node(AST_BINARY_ADD, NULL, current.line);
        add->left = lhs;
        add->right = rhs;
        (*i)++;
        return add;
    }

    return lhs;
}

ASTNode* parse_tokens(TokenArray tokens) {
    ASTNode* program = make_node(AST_PROGRAM, NULL, 1);
    if (!program) return NULL;

    for (size_t i = 0; i < tokens.count;) {
        Token tok = tokens.data[i];
        if (tok.type == TOKEN_EOF) break;
        if (tok.type == TOKEN_NEWLINE) {
            i++;
            continue;
        }
        if (tok.type != TOKEN_IDENTIFIER || tokens.data[i + 1].type != TOKEN_ASSIGN) {
            free_ast(program);
            return NULL;
        }

        ASTNode* assign = make_node(AST_ASSIGNMENT, tok.lexeme, tok.line);
        i += 2;
        assign->right = parse_expression(tokens, &i);
        if (!assign->right) {
            free_ast(assign);
            free_ast(program);
            return NULL;
        }

        program->statements = (ASTNode**)realloc(program->statements, sizeof(ASTNode*) * (program->statement_count + 1));
        program->statements[program->statement_count++] = assign;

        if (tokens.data[i].type == TOKEN_NEWLINE) i++;
    }

    return program;
}

void free_ast(ASTNode* node) {
    if (!node) return;
    free(node->value);
    free_ast(node->left);
    free_ast(node->right);
    for (size_t i = 0; i < node->statement_count; ++i) free_ast(node->statements[i]);
    free(node->statements);
    free(node);
}
