### Architecture.md

# PyC Compiler: Architecture and Engineering

## Overview

PyC's architecture is designed to be modular and extensible, following a classic compiler structure while incorporating experimental features for AI and GPU acceleration. The system is built primarily in C, with components in C++ and CUDA, leveraging LLVM for code generation and optimization.

## Core Components

| Component                | Purpose                                                                               |
|--------------------------|---------------------------------------------------------------------------------------|
| **Frontend**             | Tokenizes and parses Python-like source code into Abstract Syntax Trees (ASTs).       |
| **IR System**            | Generates optimized Intermediate Representation (IR) from the AST.                    |
| **Backend**              | Generates machine code or accelerated bytecode from IR, with support for CPU and GPU. |
| **AI Graph Compiler**    | Optimizes computational graphs for tensor workflows.                                  |
| **Memory Planner**       | Dynamically allocates and minimizes tensor memory footprints.                         |
| **Custom Kernel Loader** | Integrates user-written `.cu` or `.cl` files into compiled pipelines.                 |
| **CLI Driver**           | Exposes compilation commands to users (`pyc build`, `pyc optimize`, etc.).            |

## Component Breakdown

### 1. Frontend

- **Lexer** (`lexer.c`): Tokenizes input, handling Python-like syntax and indentation.  
- **Parser** (`parser.c`): Builds AST from tokens, supporting expressions, assignments, and if statements.  
- **Frontend Utilities** (`frontend.c`): Manages source code loading and preprocessing.

### 2. Symbol Table

- **Symbol Table Manager** (`symbol_table.c`): Tracks variables and scopes, preparing for IR generation.

### 3. IR Generation

- **IR Generator** (`ir_generator.c`, `codegen.c`, `IR.c`): Converts AST to LLVM IR, supporting basic operations and conditionals.

### 4. Optimization

- **Optimizer** (`backend.c`, `codegen.c`): Applies LLVM optimization passes for efficient code generation.

### 5. Backend

- **Backend Manager** (`backend.c`): Handles JIT compilation and object file generation with multithreading.  
- **CUDA Integration** (`kernel.cu`, `matrix_mult.cu`): Experimental kernels for tokenization and matrix operations.

### 6. AI-Specific Modules

- **Graph Compiler** (`AI/graph_compiler.c`): Optimizes tensor operations and computational graphs.  
- **Memory Planner** (`AI/memory_planner.c`): Manages dynamic memory allocation for tensors.  
- **Optimizer** (`AI/optimizer.c`): Applies runtime strategies for AI models.  
- **Visualizer** (`AI/visualizer.c`): Generates visual representations of computational graphs.

### 7. CLI Interface

- **Main Entry Point** (`main.c`): Orchestrates the compiler pipeline and handles CLI commands.

### 8. Testing

- **Test Suite** (`test_parser.c`): Validates parser functionality with basic expressions and statements.

## Engineering and Feature Set

- **Modularity**: Independent components allow for easy extension and experimentation.  
- **LLVM Integration** Ensures portability and optimized code generation.  
- **Multithreading**: Accelerates backend compilation using multiple CPU cores.  
- **GPU Acceleration**: Experimental use of CUDA for compiler tasks like tokenization.  
- **AI Focus**: Specialized modules for optimizing AI workflows, including graph compilation and memory planning.  
- **CLI-First Design**: Fully operable from the terminal with simple, intuitive commands.

## How Components Connect

1. **Input Handling**: Source code is loaded and preprocessed by the frontend.  
2. **Tokenization and Parsing**: Lexer and parser transform code into an AST.  
3. **Symbol Resolution**: Symbol table manages variable scopes for IR generation.  
4. **IR Generation**: AST is converted to LLVM IR, incorporating optimizations.  
5. **Optimization**: LLVM passes enhance IR efficiency.  
6. **Code Generation**: Backend compiles IR to machine code or faster bytecode.  
7. **AI Workflow**: AI modules optimize computational graphs and manage tensor memory.  
8. **Custom Kernels**: Users can integrate custom CUDA/OpenCL kernels for acceleration.  
9. **CLI Commands**: Users interact with the compiler through a simple CLI interface.

This architecture ensures PyC is functional, extensible,
and aligned with its mission to speed up AI workflows through optimized compilation and GPU integration.
