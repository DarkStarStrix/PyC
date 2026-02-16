#include "parser.h"
#include "lexer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static Token current_token;
static Token lookahead_token;

static void parser_error(const char* message) {
    fprintf(stderr, "Parser error: %s\n", message);
    exit(1);
}

static ASTNode* ast_new(ASTNodeType type) {
    ASTNode* node = (ASTNode*)calloc(1, sizeof(ASTNode));
    if (!node) {
        fprintf(stderr, "Out of memory\n");
        exit(1);
    }
    node->type = type;
    return node;
}

static void program_append(ASTNode* program, ASTNode* statement) {
    if (program->as.program.statement_count == program->as.program.statement_capacity) {
        size_t new_capacity = program->as.program.statement_capacity == 0 ? 4 : program->as.program.statement_capacity * 2;
        ASTNode** new_items = (ASTNode**)realloc(program->as.program.statements, new_capacity * sizeof(ASTNode*));
        if (!new_items) {
            fprintf(stderr, "Out of memory\n");
            exit(1);
        }
        program->as.program.statements = new_items;
        program->as.program.statement_capacity = new_capacity;
    }
    program->as.program.statements[program->as.program.statement_count++] = statement;
}

static void if_body_append(ASTNode* if_node, ASTNode* statement) {
    if (if_node->as.if_statement.body_count == if_node->as.if_statement.body_capacity) {
        size_t new_capacity = if_node->as.if_statement.body_capacity == 0 ? 4 : if_node->as.if_statement.body_capacity * 2;
        ASTNode** new_items = (ASTNode**)realloc(if_node->as.if_statement.body_statements, new_capacity * sizeof(ASTNode*));
        if (!new_items) {
            fprintf(stderr, "Out of memory\n");
            exit(1);
        }
        if_node->as.if_statement.body_statements = new_items;
        if_node->as.if_statement.body_capacity = new_capacity;
    }
    if_node->as.if_statement.body_statements[if_node->as.if_statement.body_count++] = statement;
}

static void advance(void) {
    current_token = lookahead_token;
    lookahead_token = get_next_token();
}

static void expect(TokenType type, const char* message) {
    if (current_token.type != type) {
        parser_error(message);
    }
    advance();
}

static ASTNode* parse_expression(void);

static ASTNode* parse_primary(void) {
    ASTNode* node = NULL;
    if (current_token.type == TOKEN_NUMBER) {
        node = ast_new(AST_INTEGER_LITERAL);
        node->as.integer_literal.value = atoi(current_token.value);
        advance();
        return node;
    }
    if (current_token.type == TOKEN_IDENTIFIER) {
        node = ast_new(AST_IDENTIFIER);
        strncpy(node->as.identifier.name, current_token.value, sizeof(node->as.identifier.name) - 1);
        advance();
        return node;
    }

    parser_error("Expected integer literal or identifier");
    return NULL;
}

static ASTBinaryOperator token_to_binary_op(TokenType token_type) {
    switch (token_type) {
        case TOKEN_PLUS: return AST_OP_ADD;
        case TOKEN_MINUS: return AST_OP_SUB;
        case TOKEN_MULTIPLY: return AST_OP_MUL;
        case TOKEN_DIVIDE: return AST_OP_DIV;
        default:
            parser_error("Expected binary operator");
            return AST_OP_ADD;
    }
}

static ASTNode* parse_term(void) {
    ASTNode* left = parse_primary();
    while (current_token.type == TOKEN_MULTIPLY || current_token.type == TOKEN_DIVIDE) {
        ASTNode* bin = ast_new(AST_BINARY_EXPRESSION);
        bin->as.binary_expression.op = token_to_binary_op(current_token.type);
        bin->as.binary_expression.left = left;
        advance();
        bin->as.binary_expression.right = parse_primary();
        left = bin;
    }
    return left;
}

static ASTNode* parse_expression(void) {
    ASTNode* left = parse_term();
    while (current_token.type == TOKEN_PLUS || current_token.type == TOKEN_MINUS) {
        ASTNode* bin = ast_new(AST_BINARY_EXPRESSION);
        bin->as.binary_expression.op = token_to_binary_op(current_token.type);
        bin->as.binary_expression.left = left;
        advance();
        bin->as.binary_expression.right = parse_term();
        left = bin;
    }
    return left;
}

static ASTNode* parse_assignment_statement(void) {
    ASTNode* assign = ast_new(AST_ASSIGNMENT);
    strncpy(assign->as.assignment.name, current_token.value, sizeof(assign->as.assignment.name) - 1);
    advance();
    expect(TOKEN_ASSIGN, "Expected '=' in assignment");
    assign->as.assignment.value = parse_expression();
    expect(TOKEN_NEWLINE, "Expected newline after assignment");
    return assign;
}

static ASTNode* parse_statement(void);

static ASTNode* parse_if_statement(void) {
    ASTNode* if_node = ast_new(AST_IF_STATEMENT);
    expect(TOKEN_IF, "Expected 'if'");
    if_node->as.if_statement.condition = parse_expression();
    expect(TOKEN_COLON, "Expected ':' after if condition");
    expect(TOKEN_NEWLINE, "Expected newline after ':'");
    expect(TOKEN_INDENT, "Expected INDENT after if header");

    while (current_token.type != TOKEN_DEDENT && current_token.type != TOKEN_EOF) {
        if (current_token.type == TOKEN_NEWLINE) {
            advance();
            continue;
        }
        if_body_append(if_node, parse_statement());
    }
    expect(TOKEN_DEDENT, "Expected DEDENT to close if block");
    return if_node;
}

static ASTNode* parse_statement(void) {
    if (current_token.type == TOKEN_IF) {
        return parse_if_statement();
    }

    if (current_token.type == TOKEN_IDENTIFIER && lookahead_token.type == TOKEN_ASSIGN) {
        return parse_assignment_statement();
    }

    parser_error("Only assignment and if statements are allowed in v1 grammar");
    return NULL;
}

ASTNode* parse(void) {
    ASTNode* root = ast_new(AST_PROGRAM);

    current_token = get_next_token();
    lookahead_token = get_next_token();

    while (current_token.type != TOKEN_EOF) {
        if (current_token.type == TOKEN_NEWLINE) {
            advance();
            continue;
        }
        program_append(root, parse_statement());
    }

    return root;
}

void ast_free(ASTNode* node) {
    if (!node) return;

    size_t i;
    switch (node->type) {
        case AST_PROGRAM:
            for (i = 0; i < node->as.program.statement_count; ++i) {
                ast_free(node->as.program.statements[i]);
            }
            free(node->as.program.statements);
            break;
        case AST_ASSIGNMENT:
            ast_free(node->as.assignment.value);
            break;
        case AST_IF_STATEMENT:
            ast_free(node->as.if_statement.condition);
            for (i = 0; i < node->as.if_statement.body_count; ++i) {
                ast_free(node->as.if_statement.body_statements[i]);
            }
            free(node->as.if_statement.body_statements);
            break;
        case AST_BINARY_EXPRESSION:
            ast_free(node->as.binary_expression.left);
            ast_free(node->as.binary_expression.right);
            break;
        case AST_INTEGER_LITERAL:
        case AST_IDENTIFIER:
            break;
    }

    free(node);
}
