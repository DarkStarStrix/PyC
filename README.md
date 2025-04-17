# PyC: A Python-like Compiler Toolchain

This project is still under dev so some feature might not work 100% but If you want to contribute check doc.md and get an understanding then make a feature branch then make a pull request I'll chek it out 
PyC is an experimental compiler that transforms a subset of Python-like syntax into executable machine code using the LLVM infrastructure. Developed by DarkStarStrix, it serves as a learning tool and a foundation for a lightweight compiler targeting Python-like code. Written primarily in C, with C++ and CUDA components, PyC is under active development and currently supports basic expressions, assignments, and if statements.
Features

## Frontend:
Lexer: Tokenizes input with Python-style indentation (INDENT, DEDENT) and supports keywords (if, else).
Parser: Builds an AST for expressions (e.g., x + 42), assignments (e.g., x = 5), and if statements with blocks.


## Symbol Table: 
Tracks variables across scopes, integrated with LLVM IR generation.

## IR Generation: 
Produces LLVM IR for expressions, assignments, and conditionals.

## Backend:
JIT compilation for immediate execution.
Object file generation with multithreaded compilation.


Error Handling: Detailed error and warning reports with source context, logged to compiler_errors.log.
Optimization: Applies LLVM passes (e.g., instruction combining, GVN).
CUDA Integration: Experimental tokenization kernel (not yet functional).
Testing: Parser test suite (test_parser.c) for basic constructs.
Cross-Platform: Primarily tested on Windows, designed for portability.

# Current Limitations

Syntax Support: Limited to expressions, assignments, and if statements; lacks loops, functions, and full Python syntax.
Symbol Table: Supports variables but not functions or complex types.
Error Handling: Basic semantic checks; needs richer diagnostics.
CUDA: Experimental and non-operational.

```
Project Structure
darkstarstrix-pyc/
├── README.md               # Project overview
├── CODE_OF_CONDUCT.md      # Community guidelines
├── CONTRIBUTING.md         # Contribution instructions
├── Doc.md                  # Technical documentation
├── Hello.py                # Sample Python script
├── hello.spec              # PyInstaller spec file
├── LICENSE                 # Apache License 2.0
├── AI/
│   └── graph_compiler.c    # Computational graph compiler
├── Core/
│   ├── C_Files/
│   │   ├── backend.c       # JIT and object file generation
│   │   ├── codegen.c       # Additional codegen utilities
│   │   ├── Core.cpp        # AST management
│   │   ├── error_handler.c # Error reporting system
│   │   ├── frontend.c      # Source code loading
│   │   ├── IR.c            # IR utilities
│   │   ├── ir_generator.c  # LLVM IR generation
│   │   ├── lexer.c         # Tokenization with indentation
│   │   ├── main.c          # Compiler entry point
│   │   ├── parser.c        # AST construction
│   │   ├── stack.c         # Stack for parsing
│   │   ├── symbol_table.c  # Variable scoping
│   │   └── test_parser.c   # Parser unit tests
│   └── Header_Files/
│       ├── backend.h
│       ├── Core.h          # AST definitions
│       ├── error_handler.h
│       ├── frontend.h
│       ├── lexer.h
│       ├── parser.h        # Token and AST structures
│       ├── stack.h
│       └── symbol_table.h
├── Examples/
│   ├── matrix_mult.py      # Matrix multiplication example
│   └── simple_graph.py     # Computational graph example
├── hello/
│   ├── Analysis-00.toc     # PyInstaller artifacts
│   ├── base_library.zip
│   ├── EXE-00.toc
│   ├── hello.pkg
│   ├── PKG-00.toc
│   ├── PYZ-00.pyz
│   ├── PYZ-00.toc
│   ├── warn-hello.txt
│   ├── xref-hello.html
│   └── localpycs/
└── Kernel/
    ├── kernel.cu           # CUDA tokenization kernel
    └── matrix_mult.cu      # CUDA matrix multiplication
```

# Installation
Prerequisites

CMake: 3.29.6 or later
LLVM: Configured with path set in CMakeLists.txt
C/C++ Compiler: C11 and C++14 compatible (e.g., GCC, MSVC)
Python 3.x: For testing and PyInstaller
CUDA Toolkit: Optional for experimental CUDA features

# Build Steps

Clone the repository:
```
git clone https://github.com/DarkStarStrix/PyC.git
cd PyC
```

Configure with CMake:
```
mkdir build
cd build
cmake ..
```

Build the project:
```
cmake --build . --config Release
```


The executable MyCompiler will be in build/bin/.
Usage
Run the compiler:
```
./build/bin/MyCompiler [options] input_file.pc
```

# Command-Line Options

-o <file>: Output file (default: a.out)
-O: Enable LLVM optimizations
-jit: JIT compile and execute
-v: Verbose output
-h, --help: Show help

Example
Compile a file with an if statement:
```
./build/bin/MyCompiler -v -O test.pc -o test
```

test.pc
```
x = 5
if x:
    y = x + 3
```
This generates LLVM IR, applies optimizations (if -O is used), and produces an executable or executes via JIT.
How It Works

Frontend (lexer.c, parser.c, frontend.c):
Reads input, tokenizes it with indentation support, and builds an AST.
Handles expressions, assignments, and if statements.


Symbol Table (symbol_table.c):
Tracks variables across scopes, storing LLVM values for IR generation.


IR Generation (ir_generator.c, codegen.c, IR.c):
Converts the AST to LLVM IR, supporting assignments and conditionals.


Backend (backend.c):
Compiles IR to machine code via JIT or object files, with multithreading.


Error Handling (error_handler.c):
Logs errors and warnings with source context to compiler_errors.log.


CUDA (kernel.cu):
Experimental tokenization kernel (not yet functional).


# Current Progress
## Implemented
Lexer: Supports indentation, keywords, and basic tokens.
Parser: Handles expressions, assignments, and if statements.
Symbol Table: Manages variable scopes with LLVM integration.
IR Generation: Generates IR for basic constructs.
Error Handling: Detailed diagnostics with line/column info.

## Planned

Full Python syntax (loops, functions).
Enhanced symbol table for functions and types.
Functional CUDA integration.
Advanced optimizations.

## Contributing
Contributions are welcome! See CONTRIBUTING.md for guidelines:

## Fork the repository.
Create a feature branch: git checkout -b feature/your-feature.
Commit changes: git commit -m "Add feature".
Push and open a pull request.

Adhere to C11/C++14 standards and include comments.
License
Licensed under the Apache License 2.0. See LICENSE for details.
Acknowledgments
Developed by DarkStarStrix. Feedback is welcome via GitHub Issues.

