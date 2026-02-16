#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "stack.h"
#include "symbol_table.h"

static long long run_stack_rounds(int rounds) {
    long long checksum = 0;
    Stack stack;

    for (int r = 0; r < rounds; ++r) {
        initStack(&stack);
        for (int i = 0; i < MAX; ++i) {
            push(&stack, i + r);
        }
        for (int i = 0; i < MAX; ++i) {
            checksum += pop(&stack);
        }
    }

    return checksum;
}

static long long run_symbol_rounds(int rounds) {
    long long checksum = 0;
    char name[64];

    for (int r = 0; r < rounds; ++r) {
        symbol_table_init();
        for (int i = 0; i < 128; ++i) {
            snprintf(name, sizeof(name), "sym_%d_%d", r, i);
            checksum += symbol_define(name);
            checksum += symbol_exists(name);
        }
        checksum += symbol_exists("sym_missing");
        symbol_table_cleanup();
    }

    return checksum;
}

int main(int argc, char** argv) {
    int rounds = 5000;
    if (argc > 1) {
        rounds = atoi(argv[1]);
        if (rounds <= 0) {
            fprintf(stderr, "rounds must be > 0\n");
            return 2;
        }
    }

    long long checksum = 0;
    checksum += run_stack_rounds(rounds);
    checksum += run_symbol_rounds(rounds);

    printf("microbench rounds=%d checksum=%lld\n", rounds, checksum);
    return 0;
}
