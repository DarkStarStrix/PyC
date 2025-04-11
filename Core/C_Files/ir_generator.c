// ir_generator.c
#include "parser.h"
#include <llvm-c/Core.h>
#include <stdio.h>
#include <stdlib.h>

// Context and module for LLVM IR generation
LLVMContextRef context;
LLVMModuleRef module;
LLVMBuilderRef builder;

// Initialize LLVM for IR generation
void init_ir_generator() {
    // Initialize LLVM
    LLVMInitializeNativeTarget();
    LLVMInitializeNativeAsmPrinter();
    
    // Create context, module and builder
    context = LLVMContextCreate();
    module = LLVMModuleCreateWithNameInContext("python_module", context);
    builder = LLVMCreateBuilderInContext(context);
}

// Convert AST to LLVM IR
LLVMValueRef ast_to_llvm_ir(ASTNode* node) {
    if (!node) return NULL;
    
    switch (node->type) {
        case AST_NODE_NUMBER:
            return LLVMConstInt(LLVMInt32TypeInContext(context), node->data.number, 0);
            
        case AST_NODE_IDENTIFIER: {
            // In a real implementation, this would look up variables in a symbol table
            fprintf(stderr, "Warning: Variable lookup not yet implemented\n");
            return LLVMConstInt(LLVMInt32TypeInContext(context), 0, 0);
        }
            
        case AST_NODE_BINARY_OP: {
            LLVMValueRef left = ast_to_llvm_ir(node->data.binary_op.left);
            LLVMValueRef right = ast_to_llvm_ir(node->data.binary_op.right);
            
            if (!left || !right) return NULL;
            
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
                    fprintf(stderr, "Error: Unknown binary operator: %c\n", node->data.binary_op.op);
                    return NULL;
            }
        }
            
        default:
            fprintf(stderr, "Error: Unknown AST node type: %d\n", node->type);
            return NULL;
    }
}

// Generate function wrapper around the expression
LLVMValueRef generate_main_function(ASTNode* ast_root) {
    // Create function type and function
    LLVMTypeRef return_type = LLVMInt32TypeInContext(context);
    LLVMTypeRef function_type = LLVMFunctionType(return_type, NULL, 0, 0);
    LLVMValueRef main_func = LLVMAddFunction(module, "main", function_type);
    
    // Create basic block and set builder position
    LLVMBasicBlockRef entry = LLVMAppendBasicBlock(main_func, "entry");
    LLVMPositionBuilderAtEnd(builder, entry);
    
    // Generate IR for the AST and add return instruction
    LLVMValueRef result = ast_to_llvm_ir(ast_root);
    if (result) {
        LLVMBuildRet(builder, result);
    } else {
        // If expression evaluation failed, return 0
        LLVMBuildRet(builder, LLVMConstInt(LLVMInt32TypeInContext(context), 0, 0));
    }
    
    return main_func;
}

// Main IR generation function
void generate_ir(ASTNode* ast_root) {
    init_ir_generator();
    
    // Generate IR
    generate_main_function(ast_root);
    
    // Verify the module
    char* error = NULL;
    LLVMVerifyModule(module, LLVMAbortProcessAction, &error);
    LLVMDisposeMessage(error);
    
    // Print the generated IR (for debugging)
    LLVMDumpModule(module);
}

// Cleanup LLVM resources
void cleanup_ir_generator() {
    LLVMDisposeBuilder(builder);
    LLVMDisposeModule(module);
    LLVMContextDispose(context);
}
