#include "symbol_table.h"

#include <string.h>

#include "pyc/pyc_mutex.h"

#define MAX_SYMBOLS 256
#define MAX_NAME 128

static char symbols[MAX_SYMBOLS][MAX_NAME];
static int symbol_count = 0;
static pyc_mutex symbol_mutex = PYC_MUTEX_INIT;

int symbol_table_init(void) {
    pyc_mutex_lock(&symbol_mutex);
    symbol_count = 0;
    pyc_mutex_unlock(&symbol_mutex);
    return 0;
}

/* Caller must hold symbol_mutex. */
static int symbol_exists_locked(const char* name) {
    for (int i = 0; i < symbol_count; ++i) {
        if (strcmp(symbols[i], name) == 0) {
            return 1;
        }
    }
    return 0;
}

int symbol_exists(const char* name) {
    int result;
    pyc_mutex_lock(&symbol_mutex);
    result = symbol_exists_locked(name);
    pyc_mutex_unlock(&symbol_mutex);
    return result;
}

int symbol_define(const char* name) {
    int result;
    pyc_mutex_lock(&symbol_mutex);
    if (symbol_count >= MAX_SYMBOLS) {
        result = -1;
    } else if (symbol_exists_locked(name)) {
        result = 0;
    } else {
        strncpy(symbols[symbol_count], name, MAX_NAME - 1);
        symbols[symbol_count][MAX_NAME - 1] = '\0';
        symbol_count++;
        result = 0;
    }
    pyc_mutex_unlock(&symbol_mutex);
    return result;
}

void symbol_table_cleanup(void) {
    pyc_mutex_lock(&symbol_mutex);
    symbol_count = 0;
    pyc_mutex_unlock(&symbol_mutex);
}
