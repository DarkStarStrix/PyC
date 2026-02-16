#include "symbol_table.h"

#include <string.h>

#define MAX_SYMBOLS 256
#define MAX_NAME 128

static char symbols[MAX_SYMBOLS][MAX_NAME];
static int symbol_count = 0;

int symbol_table_init(void) {
    symbol_count = 0;
    return 0;
}

int symbol_exists(const char* name) {
    for (int i = 0; i < symbol_count; ++i) {
        if (strcmp(symbols[i], name) == 0) {
            return 1;
        }
    }
    return 0;
}

int symbol_define(const char* name) {
    if (symbol_count >= MAX_SYMBOLS) {
        return -1;
    }
    if (symbol_exists(name)) {
        return 0;
    }
    strncpy(symbols[symbol_count], name, MAX_NAME - 1);
    symbols[symbol_count][MAX_NAME - 1] = '\0';
    symbol_count++;
    return 0;
}

void symbol_table_cleanup(void) {
    symbol_count = 0;
}
