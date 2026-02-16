#ifndef FRONTEND_H
#define FRONTEND_H

#include "Core.h"

int frontend_init(void);
int frontend_process_file(const char* filename, ASTNode** ast_root);
int frontend_is_python_file(const char* filename);
void frontend_cleanup(void);

#endif // FRONTEND_H
