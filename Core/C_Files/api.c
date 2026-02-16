int api_init(void) {
    return 1;
}

#include "api.h"

#include "adapter.h"
#include "error_handler.h"
#include "ir_generator.h"
#include "lexer.h"
#include "parser.h"
#include "semantic.h"
#include "symbol_table.h"

#include <stdio.h>
#include <stdlib.h>

static ApiStatus run_build_pipeline(const char* source, const BuildConfig* cfg) {
    TokenArray tokens = lexical_analysis(source);
    for (size_t i = 0; i < tokens.count; ++i) {
        if (tokens.data[i].type == TOKEN_INVALID) {
            add_error("line %d: invalid token '%s'", tokens.data[i].line, tokens.data[i].lexeme);
        }
    }
    if (has_errors()) {
        free_tokens(tokens);
        return API_STATUS_LEX_ERROR;
    }

    ASTNode* ast = parse_tokens(tokens);
    if (!ast) {
        free_tokens(tokens);
        add_error("parser failed: expected assignment lines like x = y + 1");
        return API_STATUS_PARSE_ERROR;
    }

    if (perform_semantic_analysis(ast) != 0) {
        free_ast(ast);
        free_tokens(tokens);
        return API_STATUS_SEMANTIC_ERROR;
    }

    IRCode* ir = generate_ir(ast);
    if (!ir) {
        free_ast(ast);
        free_tokens(tokens);
        add_error("IR generation failed");
        return API_STATUS_IR_ERROR;
    }

    if (emit_backend_output(ir, cfg->output_path, cfg->output_mode) != 0) {
        add_error("backend output failed for '%s'", cfg->output_path);
        free_ir(ir);
        free_ast(ast);
        free_tokens(tokens);
        return API_STATUS_BACKEND_ERROR;
    }

    free_ir(ir);
    free_ast(ast);
    free_tokens(tokens);
    return API_STATUS_OK;
}

ApiStatus build_script(const char* filename, const BuildConfig* cfg) {
    if (!filename || !cfg || !cfg->output_path) {
        return API_STATUS_INVALID_ARGUMENT;
    }

    char* source = NULL;
    size_t size = 0;
    char io_err[256] = {0};
    if (adapter_read_file(filename, &source, &size, io_err, sizeof(io_err)) != 0) {
        add_error("%s", io_err);
        return API_STATUS_IO_ERROR;
    }

    init_error_handler(filename, source);
    symbol_table_init();

    ApiStatus status = run_build_pipeline(source, cfg);
    if (status != API_STATUS_OK) {
        print_errors();
    }

    free(source);
    return status;
}

ApiStatus optimize_script(const char* filename, int graph_opt) {
    (void)filename;
    (void)graph_opt;
#ifndef ENABLE_OPTIMIZE
    return API_STATUS_FEATURE_DISABLED;
#else
    return API_STATUS_OK;
#endif
}

ApiStatus visualize_graph(const char* filename) {
    (void)filename;
#ifndef ENABLE_VISUALIZE
    return API_STATUS_FEATURE_DISABLED;
#else
    return API_STATUS_OK;
#endif
}

ApiStatus register_kernel(const char* kernel_file) {
    (void)kernel_file;
#ifndef ENABLE_KERNEL
    return API_STATUS_FEATURE_DISABLED;
#else
    return API_STATUS_OK;
#endif
}

void cleanup_api(void) {
    symbol_table_cleanup();
    cleanup_error_handler();
}

void cleanup_api(void) {
    api_cleanup();
}
