#ifndef PYC_IR_GENERATOR_H
#define PYC_IR_GENERATOR_H

#include "parser.h"

typedef struct {
    char* text;
} IRCode;

IRCode* generate_ir(ASTNode* ast);
void free_ir(IRCode* ir);

#endif
