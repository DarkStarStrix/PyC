// Core/C_Files/ir_generator.c - LLVM IR generation for PyC compiler
#include "Core.h"
#include "symbol_table.h"
#include "error_handler.h"
#include <llvm-c/Core.h>
#include <llvm-c/Target.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

LLVMContextRef context;
LLVMModuleRef module;
LLVMBuilderRef builder;

void ir_generator_init(void) {
    LLVMInitializeNativeTarget();
    LLVMInitializeNativeAsmPrinter();
    context = LLVMContextCreate();
    module = LLVMModuleCreateWithNameInContext("pyc_module", context);
    builder = LLVMCreateBuilderInContext(context);
}

LLVMValueRef ast_to_llvm_ir(ASTNode* node) {
    if (!node) return NULL;

    switch (node->type) {
        case NODE_EXPRESSION:
            switch (node->expr.type) {
                case EXPR_NUMBER:
                    return LLVMConstInt(LLVMInt32TypeInContext(context), atoi(node->expr.value), 0);
                case EXPR_VARIABLE: {
                    SymbolNode* symbol = lookup_symbol(node->expr.value);
                    if (!symbol || !symbol->llvm_value) {
                        report_error(0, 0, "Undefined variable '%s'", node->expr.value);
                        return NULL;
                    }
                    return LLVMBuildLoad(builder, symbol->llvm_value, node->expr.value);
                }
                case EXPR_BINARY_OP: {
                    LLVMValueRef left = ast_to_llvm_ir(node->expr.left);
                    LLVMValueRef right = ast_to_llvm_ir(node->expr.right);
                    if (!left || !right) return NULL;
                    switch (node->expr.op) {
                        case TOKEN_PLUS: return LLVMBuildAdd(builder, left, right, "addtmp");
                        case TOKEN_MINUS: return LLVMBuildSub(builder, left, right, "subtmp");
                        case TOKEN_MULTIPLY: return LLVMBuildMul(builder, left, right, "multmp");
                        case TOKEN_DIVIDE: return LLVMBuildSDiv(builder, left, right, "divtmp");
                        default:
                            report_error(0, 0, "Unknown operator");
                            return NULL;
                    }
                }
            }
            break;

        case NODE_ASSIGNMENT: {
            LLVMValueRef value = ast_to_llvm_ir(node->assign.value);
            if (!value) return NULL;
            SymbolNode* symbol = lookup_symbol_current_scope(node->assign.name);
            if (!symbol) {
                LLVMValueRef var = LLVMBuildAlloca(builder, LLVMInt32TypeInContext(context), node->assign.name);
                add_symbol(node->assign.name, SYMBOL_VARIABLE, NULL, var);
                symbol = lookup_symbol(node->assign.name);
            }
            LLVMBuildStore(builder, value, symbol->llvm_value);
            return value;
        }

        case NODE_IF_STATEMENT: {
            LLVMValueRef condition = ast_to_llvm_ir(node->if_stmt.condition);
            if (!condition) return NULL;

            LLVMBasicBlockRef then_block = LLVMAppendBasicBlock(LLVMGetInsertBlock(builder)->parent, "then");
            LLVMBasicBlockRef else_block = node->if_stmt.else_body ? 
                LLVMAppendBasicBlock(LLVMGetInsertBlock(builder)->parent, "else") : NULL;
            LLVMBasicBlockRef end_block = LLVMAppendBasicBlock(LLVMGetInsertBlock(builder)->parent, "end");

            LLVMBuildCondBr(builder, condition, then_block, else_block ? else_block : end_block);

            LLVMPositionBuilderAtEnd(builder, then_block);
            enter_scope();
            ast_to_llvm_ir(node->if_stmt.body);
            exit_scope();
            LLVMBuildBr(builder, end_block);

            if (else_block) {
                LLVMPositionBuilderAtEnd(builder, else_block);
                enter_scope();
                ast_to_llvm_ir(node->if_stmt.else_body);
                exit_scope();
                LLVMBuildBr(builder, end_block);
            }

            LLVMPositionBuilderAtEnd(builder, end_block);
            return NULL;
        }

        case NODE_BLOCK:
            for (int i = 0; i < node->block.num_statements; i++) {
                ast_to_llvm_ir(node->block.statements[i]);
            }
            return NULL;

        default:
            report_error(0, 0, "Unsupported AST node type %d", node->type);
            return NULL;
    }
    return NULL;
}

LLVMValueRef generate_main_function(ASTNode* ast_root) {
    LLVMTypeRef return_type = LLVMInt32TypeInContext(context);
    LLVMTypeRef func_type = LLVMFunctionType(return_type, NULL, 0, 0);
    LLVMValueRef main_func = LLVMAddFunction(module, "main", func_type);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlock(main_func, "entry");
    LLVMPositionBuilderAtEnd(builder, entry);

    ast_to_llvm_ir(ast_root);
    LLVMBuildRet(builder, LLVMConstInt(LLVMInt32TypeInContext(context), 0, 0));

    return main_func;
}

void ir_generator_generate(ASTNode* ast_root) {
    ir_generator_init();
    symbol_table_init();
    generate_main_function(ast_root);

    char* error = NULL;
    LLVMVerifyModule(module, LLVMAbortProcessAction, &error);
    if (error) {
        report_error(0, 0, "Module verification failed: %s", error);
        LLVMDisposeMessage(error);
    }

    LLVMDumpModule(module);
}

void ir_generator_cleanup(void) {
    symbol_table_cleanup();
    LLVMDisposeBuilder(builder);
    LLVMDisposeModule(module);
    LLVMContextDispose(context);
}

void generate_ir(ASTNode* ast_root) {
    ir_generator_generate(ast_root);
}

void cleanup_ir_generator(void) {
    ir_generator_cleanup();
}
