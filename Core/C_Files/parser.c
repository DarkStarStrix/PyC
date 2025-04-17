#include "parser.h"
#include "lexer.h"
#include "Core.h"
#include "symbol_table.h"
#include <stdlib.h>
#include <string.h>

static Token current_token;

static void advance() {
    current_token = get_next_token();
}

static void expect(TokenType type) {
    if (current_token.type != type) {
        char msg[256];
        sprintf(msg, "Expected token type %d, got %d", type, current_token.type);
        report_error(lexer.line, msg);
        exit(1);
    }
    advance();
}

static ASTNode* expression(); // Forward declaration

static ASTNode* statement() {
    ASTNode* node = malloc(sizeof(ASTNode));

    if (current_token.type == TOKEN_IF) {
        node->type = NODE_IF_STATEMENT;
        advance();
        node->if_stmt.condition = expression();
        expect(TOKEN_COLON);
        expect(TOKEN_NEWLINE);
        expect(TOKEN_INDENT);
        node->if_stmt.body = parse_block();
        expect(TOKEN_DEDENT);
        if (current_token.type == TOKEN_ELSE) {
            advance();
            expect(TOKEN_COLON);
            expect(TOKEN_NEWLINE);
            expect(TOKEN_INDENT);
            node->if_stmt.else_body = parse_block();
            expect(TOKEN_DEDENT);
        } else {
            node->if_stmt.else_body = NULL;
        }
    } else if (current_token.type == TOKEN_IDENTIFIER && 
               get_next_token().type == TOKEN_ASSIGN) {
        node->type = NODE_ASSIGNMENT;
        strcpy(node->assign.name, current_token.value);
        advance();
        advance(); // Skip '='
        node->assign.value = expression();
        expect(TOKEN_NEWLINE);
        add_variable(node->assign.name);
    } else {
        node->type = NODE_EXPRESSION;
        node->expr = expression();
        expect(TOKEN_NEWLINE);
    }
    return node;
}

static ASTNode* parse_block() {
    ASTNode* block = malloc(sizeof(ASTNode));
    block->type = NODE_BLOCK;
    block->block.num_statements = 0;
    block->block.statements = malloc(sizeof(ASTNode*) * 10); // Initial capacity

    while (current_token.type != TOKEN_DEDENT && current_token.type != TOKEN_EOF) {
        block->block.statements[block->block.num_statements++] = statement();
    }
    return block;
}

static ASTNode* expression() {
    ASTNode* left = malloc(sizeof(ASTNode));
    left->type = NODE_EXPRESSION;

    if (current_token.type == TOKEN_NUMBER) {
        left->expr.type = EXPR_NUMBER;
        strcpy(left->expr.value, current_token.value);
        advance();
    } else if (current_token.type == TOKEN_IDENTIFIER) {
        left->expr.type = EXPR_VARIABLE;
        strcpy(left->expr.value, current_token.value);
        if (!check_variable(left->expr.value)) {
            report_error(lexer.line, "Undefined variable");
            exit(1);
        }
        advance();
    }

    if (current_token.type == TOKEN_PLUS || current_token.type == TOKEN_MINUS) {
        ASTNode* binop = malloc(sizeof(ASTNode));
        binop->type = NODE_BINARY_OP;
        binop->binop.left = left;
        binop->binop.op = current_token.type;
        advance();
        binop->binop.right = expression();
        return binop;
    }
    return left;
}

ASTNode* parse() {
    advance();
    ASTNode* root = malloc(sizeof(ASTNode));
    root->type = NODE_BLOCK;
    root->block.num_statements = 0;
    root->block.statements = malloc(sizeof(ASTNode*) * 10);

    while (current_token.type != TOKEN_EOF) {
        if (current_token.type == TOKEN_NEWLINE) {
            advance();
            continue;
        }
        root->block.statements[root->block.num_statements++] = statement();
    }
    return root;
}
