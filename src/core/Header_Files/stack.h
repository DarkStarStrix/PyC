#ifndef STACK_H
#define STACK_H

#define MAX 100

#if defined(__GNUC__) || defined(__clang__)
#define PYC_UNUSED __attribute__((unused))
#else
#define PYC_UNUSED
#endif

typedef struct {
    int items[MAX];
    int top;
} Stack;

PYC_UNUSED void initStack(Stack* stack);

PYC_UNUSED int isEmpty(Stack* stack);

PYC_UNUSED int isFull(Stack* stack);

PYC_UNUSED void push(Stack* stack, int value);

PYC_UNUSED int pop(Stack* stack);

PYC_UNUSED int peek(Stack* stack);

PYC_UNUSED int size(Stack* stack);

#endif // STACK_H
