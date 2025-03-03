#ifndef STACK_H
#define STACK_H

#define MAX 100

typedef struct {
    int items[MAX];
    int top;
} Stack;

__attribute__((unused)) __attribute__((unused)) void initStack(Stack* stack);

__attribute__((unused)) int isEmpty(Stack* stack);

__attribute__((unused)) int isFull(Stack* stack);

__attribute__((unused)) __attribute__((unused)) void push(Stack* stack, int value);

__attribute__((unused)) __attribute__((unused)) int pop(Stack* stack);

__attribute__((unused)) __attribute__((unused)) int peek(Stack* stack);

__attribute__((unused)) __attribute__((unused)) int size(Stack* stack);

#endif // STACK_H