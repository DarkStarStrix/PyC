PyC Compiler: Architecture and Engineering
Overview
PyC's architecture is designed to be modular and extensible, following the classic compiler structure with a focus on educational value and experimental features. The compiler is built primarily in C, with some components in C++ and CUDA, leveraging the LLVM framework for code generation and optimization.
Component Breakdown

1. Frontend
Lexer (lexer.c): 
Tokenizes the input source code, handling Python-like syntax including indentation.
Supports basic tokens (identifiers, numbers, operators) and keywords (if, else).

Parser (parser.c):
Constructs an Abstract Syntax Tree (AST) from tokens.
Handles expressions, assignments, and if statements with indentation-based blocks.

Frontend Utilities (frontend.c):
Manages source code loading and preprocessing.


2. Symbol Table
Symbol Table Manager (symbol_table.c):
Tracks variables and their scopes.
Integrates with IR generation to manage variable storage in LLVM IR.


3. IR Generation
IR Generator (ir_generator.c, codegen.c, IR.c):
Converts the AST into LLVM IR.
Supports basic operations, assignments, and conditionals.
Prepares IR for optimization and backend processing.


4. Optimization
Optimizer (backend.c, codegen.c):
Applies LLVM optimization passes like instruction combining and Global Value Numbering (GVN).
Enhances the efficiency of the generated code.


5. Backend
Backend Manager (backend.c):
Handles JIT compilation for immediate execution.
Generates object files with multithreaded compilation for efficiency.


CUDA Integration (kernel.cu, matrix_mult.cu):
Experimental kernels for tokenization and matrix multiplication.
Aims to accelerate compiler tasks using GPU parallelism.


6. Error Handling
Error Handler (error_handler.c):
Provides detailed error and warning messages with source context.
Logs issues to compiler_errors.log for debugging.


7. Testing
Test Suite (test_parser.c):
Validates the parser with basic expressions and statements.
Ensures correctness as new features are added.

Engineering and Feature Set

Modularity: Each component is designed to be independent, allowing for easy extension and modification.
LLVM Integration: Utilizes LLVM for IR generation and optimization, ensuring portability and performance.
Multithreading: Backend compilation leverages multiple CPU cores for faster object file generation.
GPU Experimentation: Explores CUDA for potential acceleration in compiler tasks, pushing the boundaries of traditional compiler design.
Educational Focus: The project serves as a learning tool, with clear separation of concerns and extensive documentation.

How Components Connect

Input Handling: Source code is read and preprocessed by the frontend.
Tokenization and Parsing: The lexer and parser transform the code into an AST.
Symbol Resolution: The symbol table manages variable scopes and prepares for IR generation.
IR Generation: The AST is converted into LLVM IR, incorporating symbol information.
Optimization: LLVM passes optimize the IR for better performance.
Code Generation: The backend compiles the optimized IR into machine code, optionally using JIT or generating object files.
Error Reporting: Throughout the process, errors and warnings are logged with detailed context.
GPU Acceleration: Experimental CUDA kernels aim to offload tasks like tokenization to the GPU.

This architecture ensures that PyC is both functional and extensible, providing a solid foundation for future development and experimentation.
