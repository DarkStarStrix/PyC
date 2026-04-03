#ifndef PYC_SEMANTIC_H
#define PYC_SEMANTIC_H

typedef struct PycAstNode PycAstNode;

int perform_semantic_analysis(const PycAstNode* ast);

#endif
