# PyC Compiler: High-Level Overview

What is PyC?
PyC is an experimental compiler designed to translate a subset of Python-like code into executable machine code. It leverages the LLVM infrastructure for code generation and optimization, aiming to provide a lightweight and educational tool for understanding compiler design. Additionally, PyC explores CUDA integration for GPU acceleration, targeting AI and scientific computing workloads.

## How It Works
PyC follows a traditional compiler pipeline, broken into several stages:

# Frontend:
Lexer: Converts the input source code into tokens, handling Python-style indentation and keywords.
Parser: Analyzes the tokens to build an Abstract Syntax Tree (AST), representing the program's structure.


# Symbol Table:
Manages variable scopes and tracks their usage throughout the code, preparing for IR generation.


# IR Generation:
Transforms the AST into LLVM Intermediate Representation (IR), a low-level, platform-agnostic format.


# Optimization:
Applies various LLVM optimization passes to improve the efficiency of the generated code.


# Backend:
Compiles the optimized IR into machine code, either through Just-In-Time (JIT) compilation for immediate execution or by generating object files for later use.


# Error Handling:
Provides detailed error and warning messages, helping developers identify and fix issues in their code.


# CUDA Integration (Experimental):
Aims to accelerate certain compiler tasks, like tokenization, using GPU parallelism.


# Key Components
- Lexer and Parser: Handle the syntax of the Python-like language, including indentation-based blocks.
- Symbol Table: Ensures variables are correctly scoped and accessible during compilation.
- IR Generation: Bridges the gap between high-level code and machine instructions.
- Backend: Manages code generation and execution, with support for multithreading.
- Error Handling: Enhances the development experience with informative diagnostics.
- CUDA Kernels: Explore the potential of GPU acceleration in compiler tasks.

# Current Capabilities
- PyC can currently compile basic Python-like code, including:

Variable assignments (e.g., x = 5)
Expressions (e.g., x + 42)
If statements with indentation-based blocks

It also provides:

Detailed error reporting
Basic optimizations
Experimental GPU tokenization

Future Directions
The project aims to expand its capabilities to support:

Full Python syntax, including loops and functions
Advanced optimizations for performance
Functional CUDA integration for accelerated compilation
Enhanced symbol table for complex types and functions

PyC is not just a compiler but a platform for learning and experimentation in compiler design, optimization, and GPU computing.
