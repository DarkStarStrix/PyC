#include "stack.h"

__attribute__((unused)) void initStack(Stack* stack) {
    stack->top = -1;
}

int isEmpty(Stack* stack) {
    return stack->top == -1;
}

int isFull(Stack* stack) {
    return stack->top == MAX - 1;
}

__attribute__((unused)) void push(Stack* stack, int value) {
    if (!isFull(stack)) {
        stack->items[++stack->top] = value;
    }
}

__attribute__((unused)) int pop(Stack* stack) {
    if (!isEmpty(stack)) {
        return stack->items[stack->top--];
    }
    return -1; // Return -1 if stack is empty
}

__attribute__((unused)) int peek(Stack* stack) {
    if (!isEmpty(stack)) {
        return stack->items[stack->top];
    }
    return -1; // Return -1 if stack is empty
}

__attribute__((unused)) int size(Stack* stack) {
    return stack->top + 1;
}