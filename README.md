# PyC - A Python Compiler (Work in Progress)

**PyC** is an experimental compiler project aimed at compiling Python-like code into executable machine code using LLVM as the backend. Written primarily in C with some C++ and CUDA components, this project explores the full compilation pipeline: frontend parsing, intermediate representation (IR) generation, optimization, and backend code generation. It is currently under active development and not yet fully functional.

## About

Developed by [DarkStarStrix](https://github.com/DarkStarStrix), PyC is both a learning exercise and a foundation for a lightweight compiler targeting a subset of Python syntax. It leverages LLVM for IR generation and supports features like JIT compilation, multi-threaded object file generation, and basic optimization passes. The project is incomplete, with several components (e.g., full Python syntax support, robust error handling) still in progress.

## Features

- **Frontend**: Loads source code, tokenizes, and parses into an Abstract Syntax Tree (AST).
- **IR Generation**: Converts AST into LLVM IR with basic arithmetic operation support.
- **Backend**: Supports JIT compilation and object file generation using LLVM, with multi-threaded compilation capabilities.
- **Optimization**: Basic LLVM optimization passes (e.g., instruction combining, GVN).
- **Cross-Platform**: Designed with portability in mind (currently tested on Windows).
- **Testing**: Includes a basic parser test suite.

### Current Limitations
- Limited language support (only basic expressions like numbers, identifiers, and binary operations).
- Incomplete symbol table and variable handling.
- No full Python indentation preprocessing.
- CUDA integration (parser.cu) is experimental and undeveloped.

## Directory Structure

```
darkstarstrix-pyc/
├── README.md           # Project documentation
├── CMakeLists.txt      # CMake build configuration
├── Hello.py            # Sample Python file for testing
├── hello.spec          # PyInstaller spec file for Hello.py
├── C_Files/            # Core C source files
│   ├── backend.c       # Backend logic (JIT, object file generation)
│   ├── codegen.c       # LLVM code generation from AST
│   ├── Core.cpp        # AST node management (C++)
│   ├── frontend.c      # Source code loading and preprocessing
│   ├── IR.c            # Intermediate Representation utilities
│   ├── ir_generator.c  # LLVM IR generation from AST
│   ├── main.c          # Compiler entry point
│   ├── parser.cu       # CUDA-based parser (placeholder)
│   ├── parser.cuh      # Parser header with AST construction
│   ├── stack.c         # Stack implementation
│   └── test_parser.c   # Parser unit tests
├── Header_Files/       # Header files
│   ├── backend.h       # Backend function declarations
│   ├── Core.h          # AST node definitions
│   ├── frontend.h      # Frontend function declarations
│   ├── lexer.h         # Lexer interface (assumed external)
│   ├── parser.h        # Parser and AST definitions
│   └── stack.h         # Stack interface
└── hello/              # PyInstaller output for Hello.py
    ├── *.toc, *.pyz, etc. # Build artifacts
```

## Installation

### Prerequisites
- **CMake** (3.29.6 or later)
- **LLVM** (installed and configured; see `CMakeLists.txt` for path)
- **C/C++ Compiler** (e.g., MSVC, GCC)
- **Python 3.x** (for testing and PyInstaller)
- **CUDA Toolkit** (optional, for `parser.cu`)

### Build Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/DarkStarStrix/PyC.git
   cd PyC
   ```

2. **Configure with CMake**:
   Update `CMakeLists.txt` with your LLVM installation path if necessary (default: `C:/Users/kunya/CLionProjects/PyC/llvm-project/build/lib/cmake/llvm`).
   ```bash
   mkdir build
   cd build
   cmake ..
   ```

3. **Build the Project**:
   ```bash
   cmake --build . --config Release  # or Debug
   ```
   The executable `MyCompiler` will be generated in `build/bin/`.

## Usage

Run the compiler with:
```bash
./build/bin/MyCompiler [options] input_file
```

### Options
- `-o <file>`: Set output file name (default: `a.out`).
- `-O`: Enable optimizations.
- `-jit`: Use JIT compilation only (no object file).
- `-v`: Enable verbose output.
- `-h, --help`: Show help message.

### Example
```bash
./build/bin/MyCompiler -v -O test_input.pc -o test_output
```
*Note*: `test_input.pc` must contain a supported expression (e.g., `x + 42`). Full Python syntax is not yet supported.

## Progress

- [x] **Lexer**: Basic tokenization (via `lexer.h`, assumed external).
- [x] **Parser**: Parses numbers, identifiers, and binary operations into an AST.
- [x] **IR Generation**: Generates LLVM IR for simple expressions.
- [x] **Backend**: JIT compilation and multi-threaded object file generation.
- [ ] **Full Python Support**: Indentation preprocessing and complex statements.
- [ ] **Symbol Table**: Variable tracking and scoping.
- [ ] **Error Handling**: Robust syntax and semantic error reporting.
- [ ] **CUDA Integration**: Functional CUDA-based parsing.

See the [Issues](https://github.com/DarkStarStrix/PyC/issues) tab for detailed tasks.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a branch for your feature or fix.
3. Submit a pull request with a clear description.

Please follow C11/C++14 standards and include comments. Use the "Provide feedback" link on GitHub for suggestions.

## Testing

Run the parser tests:
```bash
./build/bin/MyCompiler  # Assumes test_parser.c is linked
```
Current tests cover parsing numbers, identifiers, and binary operations.

## License

No official license yet. All rights are reserved by [DarkStarStrix](https://github.com/DarkStarStrix) 
