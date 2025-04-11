# PyC A AI compiler Toolchain

**PyC** is an experimental compiler project designed to compile Python-like code into executable machine code using the LLVM infrastructure as its backend. Written primarily in C, with some C++ and CUDA components, PyC explores the full compilation pipeline: frontend parsing, intermediate representation (IR) generation, optimization, and backend code generation. This project is under active development by [DarkStarStrix](https://github.com/DarkStarStrix) and serves as both a learning exercise and a foundation for a lightweight compiler targeting a subset of Python syntax. It is not yet fully functional, with several features still in development.

## Features

PyC currently supports the following features:

- **Frontend**: Loads source code, tokenizes it, and parses it into an Abstract Syntax Tree (AST).
  - Supports basic expressions (e.g., numbers, identifiers, binary operations like `+`, `-`, `*`, `/`).
- **IR Generation**: Converts the AST into LLVM Intermediate Representation (IR) for simple arithmetic operations.
- **Backend**: 
  - JIT (Just-In-Time) compilation for immediate execution.
  - Object file generation with multithreaded compilation capabilities using available CPU cores.
- **Optimization**: Applies basic LLVM optimization passes, such as instruction combining and Global Value Numbering (GVN).
- **Cross-Platform**: Designed with portability in mind, though currently tested only on Windows.
- **Testing**: Includes a basic test suite for the parser, covering numbers, identifiers, and binary operations.
- **CUDA Integration**: Experimental (and currently undeveloped) CUDA-based tokenization in `kernel.cu`.

### Current Limitations

- **Limited Language Support**: Only basic expressions (numbers, identifiers, and binary operations) are supported. Full Python syntax (e.g., loops, conditionals, functions) is not yet implemented.
- **Incomplete Symbol Table**: Variable tracking and scoping are not fully functional.
- **No Indentation Preprocessing**: Python’s indentation-based block structure is not yet processed.
- **Experimental CUDA**: The CUDA parser (`parser.cu`) is a placeholder and not operational.
- **Error Handling**: Lacks robust syntax and semantic error reporting.

## Directory Structure

The project is organized as follows:

```
darkstarstrix-pyc/
├── README.md           # Project documentation (this file)
├── CMakeLists.txt      # CMake build configuration
├── Hello.py            # Sample Python file for testing ("Hello, World!")
├── hello.spec          # PyInstaller spec file for Hello.py
├── kernel.cu           # CUDA kernel for tokenization (experimental)
├── C_Files/            # Core C source files
│   ├── backend.c       # Backend logic (JIT, object file generation)
│   ├── codegen.c       # LLVM IR generation from AST
│   ├── Core.cpp        # AST node management (C++)
│   ├── error_handler.c # Basic error handling system
│   ├── frontend.c      # Source code loading and preprocessing
│   ├── IR.c            # Intermediate Representation utilities
│   ├── ir_generator.c  # LLVM IR generation logic
│   ├── main.c          # Compiler entry point
│   ├── parser.cu       # CUDA-based parser (experimental)
│   ├── parser.cuh      # CUDA parser header with AST construction
│   ├── stack.c         # Stack implementation for parsing
│   ├── symbol_table.c  # Symbol table management (incomplete)
│   └── test_parser.c   # Parser unit tests
├── Header_Files/       # Header files
│   ├── backend.h       # Backend function declarations
│   ├── Core.h          # AST node definitions
│   ├── error_handler.h # Error handling declarations
│   ├── frontend.h      # Frontend function declarations
│   ├── lexer.h         # Lexer interface (assumed external)
│   ├── parser.h        # Parser and AST definitions
│   ├── stack.h         # Stack interface
│   └── symbol_table.h  # Symbol table interface
└── hello/              # PyInstaller output for Hello.py
    ├── *.toc, *.pyz, etc. # Build artifacts from PyInstaller
```

## Installation

### Prerequisites

To build and use PyC, you’ll need the following:

- **CMake**: Version 3.29.6 or later.
- **LLVM**: Installed and configured (update `CMakeLists.txt` with the correct path if needed).
- **C/C++ Compiler**: Compatible with C11 and C++14 (e.g., MSVC, GCC).
- **Python 3.x**: Required for testing and running PyInstaller.
- **CUDA Toolkit**: Optional, only needed for experimental CUDA features (`parser.cu`).

### Build Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/DarkStarStrix/PyC.git
   cd PyC
   ```

2. **Configure with CMake**:
   - Edit `CMakeLists.txt` to set the `LLVM_DIR` variable to your LLVM installation path (default: `C:/Users/kunya/CLionProjects/PyC/llvm-project/build/lib/cmake/llvm`).
   - Run the following commands:
     ```bash
     mkdir build
     cd build
     cmake ..
     ```

3. **Build the Project**:
   ```bash
   cmake --build . --config Release  # or Debug
   ```
   - The executable `MyCompiler` will be generated in `build/bin/`.

## Usage

Run the compiler using the following command:

```bash
./build/bin/MyCompiler [options] input_file
```

### Command-Line Options

- `-o <file>`: Specify the output file name (default: `a.out`).
- `-O`: Enable LLVM optimization passes.
- `-jit`: Perform JIT compilation and execute immediately (no object file generated).
- `-v`: Enable verbose output for debugging.
- `-h, --help`: Display the help message.

### Example

To compile a simple input file with verbose output and optimizations:

```bash
./build/bin/MyCompiler -v -O test_input.pc -o test_output
```

- **Note**: The input file (`test_input.pc`) must contain supported expressions (e.g., `x + 42`). Full Python syntax, such as the `print("Hello, World!")` in `Hello.py`, is not yet supported. The `Hello.py` file is included as a sample for future development.

### Sample Input File (`test_input.pc`)

```
x + 42
```

This will generate LLVM IR, apply optimizations (if `-O` is used), and either execute it via JIT (with `-jit`) or produce an object file.

## Current Progress

### Implemented Features
- **Lexer**: Basic tokenization of numbers, identifiers, and operators (via `lexer.h`, assumed external).
- **Parser**: Constructs an AST from basic expressions (numbers, identifiers, binary operations).
- **IR Generation**: Produces LLVM IR for simple arithmetic expressions.
- **Backend**: Supports JIT compilation and multi-threaded object file generation.

### Planned Features
- **Full Python Support**: Process indentation and handle complex statements (e.g., loops, functions).
- **Symbol Table**: Implement variable tracking and scoping.
- **Error Handling**: Add robust syntax and semantic error reporting.
- **CUDA Integration**: Develop a functional CUDA-based parser.

For detailed tasks and updates, see the [Issues](https://github.com/DarkStarStrix/PyC/issues) tab on GitHub.

## Testing

Run the parser tests with:

```bash
./build/bin/MyCompiler
```

- **Note**: This assumes `test_parser.c` is linked into the executable. The current test suite verifies parsing of numbers, identifiers, and binary operations.

## Organizational Notes

- **Code Standards**: Follow C11 and C++14 standards. Include comments for clarity.
- **Modularity**: The project separates concerns into frontend (`frontend.c`), IR generation (`ir_generator.c`, `codegen.c`), and backend (`backend.c`) components.
- **Future Plans**: Expand language support, improve error handling, and integrate CUDA for parallel parsing.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository on GitHub.
2. Create a branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes with clear messages:
   ```bash
   git commit -m "Add feature X"
   ```
4. Push your branch and submit a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

- **Guidelines**: Adhere to C11/C++14 standards, add comments, and provide a clear pull request description.
- **Feedback**: Use the "Provide feedback" link on GitHub for suggestions or questions.

## License

No official license has been established yet. All rights are reserved by [DarkStarStrix](https://github.com/DarkStarStrix).
