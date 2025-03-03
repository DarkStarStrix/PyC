// codegen.c

#include "codegen.h"
#include <llvm-c/Core.h>
#include <llvm-c/ExecutionEngine.h>
#include <llvm-c/Target.h>
#include <llvm-c/TargetMachine.h>
#include <llvm-c/Transforms/PassManagerBuilder.h>

LLVMModuleRef module;
LLVMBuilderRef builder;
LLVMExecutionEngineRef engine;

void initCodegen() {
    LLVMInitializeNativeTarget();
    LLVMInitializeNativeAsmPrinter();
    LLVMInitializeNativeAsmParser();
    module = LLVMModuleCreateWithName("my_module");
    builder = LLVMCreateBuilder();
}

LLVMValueRef codegenAST(ASTNode* node) {
    switch (node->type) {
        case AST_NODE_NUMBER:
            return LLVMConstInt(LLVMInt32Type(), node->data.number, 0);
        case AST_NODE_IDENTIFIER:
            // Handle identifiers (variables)
            // Placeholder: return a constant for now
            return LLVMConstInt(LLVMInt32Type(), 0, 0);
        case AST_NODE_BINARY_OP: {
            LLVMValueRef left = codegenAST(node->data.binary_op.left);
            LLVMValueRef right = codegenAST(node->data.binary_op.right);
            switch (node->data.binary_op.op) {
                case '+':
                    return LLVMBuildAdd(builder, left, right, "addtmp");
                case '-':
                    return LLVMBuildSub(builder, left, right, "subtmp");
                case '*':
                    return LLVMBuildMul(builder, left, right, "multmp");
                case '/':
                    return LLVMBuildSDiv(builder, left, right, "divtmp");
                default:
                    // Error handling for unknown operator
                    fprintf(stderr, "Unknown binary operator %c\n", node->data.binary_op.op);
                    exit(1);
            }
        }
        default:
            // Error handling for unknown AST node type
            fprintf(stderr, "Unknown AST node type %d\n", node->type);
            exit(1);
    }
}

void optimizeModule() {
    LLVMPassManagerRef passManager = LLVMCreatePassManager();
    LLVMPassManagerBuilderRef passBuilder = LLVMPassManagerBuilderCreate();
    LLVMPassManagerBuilderSetOptLevel(passBuilder, 2);
    LLVMPassManagerBuilderPopulateModulePassManager(passBuilder, passManager);
    LLVMRunPassManager(passManager, module);
    LLVMPassManagerBuilderDispose(passBuilder);
    LLVMDisposePassManager(passManager);
}

void finalizeCodegen() {
    char* error = NULL;
    LLVMVerifyModule(module, LLVMAbortProcessAction, &error);
    LLVMDisposeMessage(error);
    LLVMExecutionEngineRef engine;
    if (LLVMCreateExecutionEngineForModule(&engine, module, &error) != 0) {
        fprintf(stderr, "Failed to create execution engine: %s\n", error);
        LLVMDisposeMessage(error);
        exit(1);
    }
    LLVMDisposeBuilder(builder);
}