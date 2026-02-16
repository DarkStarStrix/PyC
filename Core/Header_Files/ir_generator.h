#ifndef IR_GENERATOR_H
#define IR_GENERATOR_H

#include "Core.h"

void ir_generator_init(void);
void ir_generator_generate(ASTNode* ast_root);
void ir_generator_cleanup(void);

/* Backward-compat wrappers */
void generate_ir(ASTNode* ast_root);
void cleanup_ir_generator(void);

#endif // IR_GENERATOR_H
