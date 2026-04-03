#ifndef PYC_SYMBOL_TABLE_H
#define PYC_SYMBOL_TABLE_H

int symbol_table_init(void);
int symbol_exists(const char* name);
int symbol_define(const char* name);
void symbol_table_cleanup(void);

#endif // SYMBOL_TABLE_H
