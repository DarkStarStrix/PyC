# PyC Compiler Toolchain Documentation

This document dives into the technical details of the PyC compiler toolchain, explaining its architecture, file roles, features, and potential use cases. PyC aims to compile Python adults-like code into machine code using LLVM, with experimental CUDA integration for GPU acceleration.

## Compiler Stages

### 1. Frontend
The frontend converts Python-like source code into a structured format for compilation.

- **Lexer**: Tokenizes source code into numbers, identifiers, and operators. It’s extensible but currently basic (assumed in `lexer.h`).
- **Parser**: Constructs an Abstract Syntax Tree (AST) from tokens, handling simple expressions (`parser.c`, `Core.cpp`).
- **Indentation Preprocessing**: Planned but unimplemented, this will process Python’s indentation-based blocks (`frontend.c`).

### 2. Intermediate Representation (IR) Generation
This stage transforms the AST into LLVM IR, a portable low-level format.

- **Code Generation**: Converts AST nodes into LLVM IR instructions, supporting basic arithmetic (`codegen.c`, `ir_generator.c`).
- **Symbol Table**: Tracks variables and scopes, though currently incomplete (`symbol_table.c`).

### 3. Backend
The backend processes LLVM IR into executable machine code.

- **Optimization**: Applies LLVM passes like instruction combining and GVN (`backend.c`, `codegen.c`).
- **JIT Compilation**: Executes code immediately for testing (`backend.c`).
- **Object File Generation**: Produces executables with multi-threaded compilation (`backend.c`).

## Experimental Features

### CUDA Integration
PyC experiments with CUDA for parallel tokenization:
- **CUDA Kernel**: `kernel.cu` defines a GPU-based tokenizer, splitting source code into tokens concurrently.
- **Goal**: Combine GPU tokens with CPU parsing to accelerate the frontend, though it’s currently non-functional.

## Error Handling
A basic system (`error_handler.c`) reports errors with line and column numbers. Future improvements will add detailed diagnostics and source code snippets.

## Testing
A test suite (`test_parser.c`) validates the parser for basic expressions. It will expand as PyC develops.

## File Roles

- **`/AI/graph_compiler.c`**: Compiles computational graphs into LLVM IR, with potential GPU offloading.
- **`/Core/C_Files/`**:
  - `backend.c`: Manages JIT compilation, optimization, and multi-threaded object file generation.
  - `codegen.c`: Generates LLVM IR from the AST.
  - `Core.cpp`: Defines and manages AST nodes.
  - `error_handler.c`: Implements error reporting.
  - `frontend.c`: Loads source code and preprocesses indentation (planned).
  - `IR.c`: Provides IR utilities and basic LLVM IR generation.
  - `ir_generator.c`: Additional IR generation logic.
  - `main.c`: Compiler entry point.
  - `parser.c`: Parses tokens into an AST.
  - `stack.c`: Stack implementation for parsing.
  - `symbol_table.c`: Manages variables (incomplete).
  - `test_parser.c`: Parser unit tests.
- **`/Examples/`**: Sample scripts (`matrix_mult.py`, `simple_graph.py`) for testing.
- **`/Kernel/`**:
  - `kernel.cu`: Experimental CUDA tokenization kernel.
  - `matrix_mult.cu`: CUDA kernel for matrix multiplication, showcasing GPU capabilities.

## Technical Direction

### Future Goals
- **Full Python Support**: Parse complex constructs like loops and functions.
- **Complete Symbol Table**: Enable full variable management.
- **Enhanced Error Handling**: Provide detailed error messages.
- **Functional CUDA**: Complete GPU-based tokenization.

### Current Challenges
- Limited syntax support restricts functionality.
- Indentation and symbol table development are pending.
- CUDA integration requires optimization and testing.

## Features in Detail

- **Frontend Parsing**: Tokenizes and builds an AST for basic expressions, laying the groundwork for Python syntax.
- **IR Generation**: Produces LLVM IR, enabling optimization and portability.
- **Backend Optimization**: Leverages LLVM passes for efficient code.
- **Multi-threaded Compilation**: Uses CPU cores to speed up object file generation.
- **CUDA Experimentation**: Aims to accelerate tokenization with GPU parallelism.

## Use Cases

- **Educational Tool**: Ideal for learning compiler design, from lexing to code generation.
- **Lightweight Compiler**: Once mature, it can compile small Python-like scripts into efficient binaries.
- **AI and Scientific Computing**: With CUDA and graph compilation, PyC could target high-performance AI models or scientific computations.

PyC’s modular design and ambitious goals make it a promising project for both learning and practical application, despite its current early stage.
