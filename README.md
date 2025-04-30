# PyC: A Python-like Compiler Toolchain

**Project CodeName:** `darkstarstrix-pyc`

**Mission Statement:**  
> Build a lightweight, high-performance Python compiler and optimization toolchain that speeds up AI workflows through graph optimizations, efficient tensor memory planning, and seamless custom kernel APIs — unified into a CLI-first architecture.

PyC is an experimental compiler
that transforms a subset of Python-like syntax into executable machine code using the LLVM infrastructure.
It serves as both an educational tool and a foundation for a lightweight compiler targeting Python-like code,
with a focus on AI and scientific computing workloads.

**Note:** This project is under active development. Some features, like the CLI interface, are now implemented, while others (e.g., full Python syntax, functional CUDA) are planned. Contributions are welcome—check `Doc.md` for an understanding of the project, then create a feature branch and open a pull request.

## Features

- **Frontend**:  
  - Lexer and parser for Python-like syntax with indentation support.  
  - Abstract Syntax Tree (AST) construction for expressions, assignments, and if statements.  
- **Symbol Table**:  
  - Manages variable scopes and integrates with LLVM IR generation (supports variables but not functions or complex types).  
- **IR Generation**:  
  - Converts AST to LLVM IR for expressions, assignments, and conditionals.  
- **Backend**:  
  - JIT compilation for immediate execution.  
  - Object file generation with multithreaded compilation.  
- **Optimization**:  
  - Applies LLVM passes (e.g., instruction combining, GVN).  
- **AI-Specific Modules**:  
  - Graph compiler for tensor operations.  
  - Memory planner for dynamic tensor memory allocation.  
  - Optimizer for AI model runtime strategies.  
  - Visualizer for computational graphs.  
- **CUDA Integration**:  
  - Experimental tokenization and matrix multiplication kernels (non-operational).  
- **CLI Interface**:  
  - Commands for building, optimizing, visualizing, and running Python-like scripts.  
- **Cross-Platform**:  
  - Primarily tested on Windows, designed for portability.

## Project Structure

```plaintext
darkstarstrix-pyc/
├── README.md
├── Architecture.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── Doc.md
├── Hello.py
├── hello.spec
├── LICENSE
├── Result.md
├── third_party/                # External libraries (e.g., CLI11.hpp)
├── AI/
│   ├── graph_compiler.c
│   ├── memory_planner.c
│   ├── optimizer.c
│   └── visualizer.c
├── Core/
│   ├── C_Files/
│   │   ├── backend.c
│   │   ├── codegen.c
│   │   ├── Core.cpp
│   │   ├── error_handler.c
│   │   ├── frontend.c
│   │   ├── IR.c
│   │   ├── ir_generator.c
│   │   ├── lexer.c
│   │   ├── main.cpp           # Updated to C++ for CLI11 integration
│   │   ├── parser.c
│   │   ├── stack.c
│   │   ├── symbol_table.c
│   │   └── test_parser.c
│   └── Header_Files/
│       ├── backend.h
│       ├── Core.h
│       ├── error_handler.h
│       ├── frontend.h
│       ├── graph.h
│       ├── ir_generator.h
│       ├── lexer.h
│       ├── memory_planner.h
│       ├── parser.h
│       ├── stack.h
│       └── symbol_table.h
├── Examples/
│   ├── matrix_mult.py
│   └── simple_graph.py
├── hello/
│   ├── Analysis-00.toc
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
    ├── kernel.cu
    └── matrix_mult.cu
```

## Installation

### Prerequisites

- **CMake**: 3.29.6 or later  
- **LLVM**: Configured with a path set in `CMakeLists.txt`  
- **C/C++ Compiler**: C11 and C++14 compatible (e.g., GCC, MSVC)  
- **Python 3.x**: For testing and PyInstaller  
- **CUDA Toolkit**: Optional for experimental CUDA features  
- **CLI11**: Download `CLI11.hpp` from [GitHub](https://github.com/CLIUtils/CLI11) and place it in `third_party/`.

### Build Steps

1. Clone the repository:  
   ```bash
   git clone https://github.com/DarkStarStrix/PyC.git
   cd PyC
   ```

2. Configure with CMake:  
   ```bash
   mkdir build
   cd build
   cmake ..
   ```

3. Build the project:  
   ```bash
   cmake --build . --config Release
   ```

The executable `MyCompiler` will be in `build/bin/`.

## Usage

Run the compiler using the CLI interface:  
```bash
./build/bin/MyCompiler [command] [options] input_file.pc
```

### CLI Commands

- `build file.pc`: Compiles and optimizes the Python-like script.  
- `optimize file.pc --graph`: Applies graph/tensor optimizations.  
- `visualize file.pc`: Outputs a visual computational graph diagram.  
- `run file.pc`: Executes the optimized pipeline.  
- `kernel register kernel.cu`: Registers a custom CUDA kernel into the runtime.

### Example

Compile a file with an if statement:  
```bash
./build/bin/MyCompiler build test.pc -o test
```

**test.pc**  
```python
x = 5
if x:
    y = x + 3
```

This generates LLVM IR, applies optimizations, and produces an executable.

### How it works

PyC transforms Python code into optimized machine code through several stages:

### 1. Compilation Pipeline

```plaintext
Python Code → IR → Computational Graph → Optimized CUDA/Machine Code
     ↓            ↓           ↓                    ↓
   Parser     LLVM IR    Graph Decomp      Memory Planning
```

### 2. Key Components

#### Frontend Processing
- Parses Python syntax into an Abstract Syntax Tree (AST)
- Performs type inference and validation
- Generates intermediate representation (IR)

#### Graph Compilation
```plaintext
Input → Graph Construction → Decomposition → Optimization
  ↓            ↓                 ↓              ↓
Code     Tensor Operations    Sub-graphs     Memory Layout
```

#### Memory Management
- Dynamic tensor allocation
- Memory pool management
- Smart buffer reuse
- Cache-friendly data layouts

#### CUDA Integration
```plaintext
Python Code → Optimized Graph → CUDA Kernels → Execution
                    ↑                ↑
            Custom Kernels ──────────┘
```

### 3. Usage Workflow

1. **Installation**:
   ```bash
   # Install via package manager
   pip install pyc-compiler
   
   # Or build from source
   cmake --build .
   ```

2. **Write Python Code**:
   ```python
   # model.py
   def matrix_multiply(a, b):
       return a @ b
   
   def neural_network(x):
       return matrix_multiply(x, weights)
   ```

3. **Compile & Optimize**:
   ```bash
   # Basic compilation
   pyc build model.py
   
   # With graph optimization
   pyc optimize model.py --graph
   
   # Register custom CUDA kernel
   pyc kernel register custom_matmul.cu
   ```

4. **Execution**:
   ```bash
   # Run optimized code
   pyc run model.py
   ```

### 4. Optimization Techniques

#### Graph Level
- Operation fusion
- Dead code elimination
- Memory access pattern optimization
- Kernel fusion
- Layout transformations

#### Memory Level
- Buffer reuse
- Smart allocation
- Minimal data movement
- Cache optimization

#### CUDA Integration
- Custom kernel support
- Automatic kernel selection
- Memory transfer optimization
- Stream management

### 5. Performance Benefits

- **Reduced Memory Usage**: Smart tensor management
- **Faster Execution**: Optimized computational graphs
- **GPU Acceleration**: Efficient CUDA kernels
- **Lower Latency**: Minimized data transfers
- **Better Cache Usage**: Optimized memory patterns

### 6. Monitoring & Debug

```bash
# Generate optimization visualization
pyc visualize model.py

# View memory allocation patterns
pyc analyze model.py --memory

# Profile execution
pyc profile model.py
```

### 7. API Integration

```python
from pyc import Compiler, Optimizer

# Programmatic usage
compiler = Compiler()
optimizer = Optimizer(enable_cuda=True)

# Compile and optimize
graph = compiler.compile("model.py")
optimized = optimizer.optimize(graph)

# Execute
result = optimized.run()
```

This architecture provides a seamless workflow from Python code to optimized execution, with full visibility into the optimization process and easy integration of custom components.


## Contributing

Contributions are welcome! See `CONTRIBUTING.md` for guidelines:  

1. Fork the repository.  
2. Create a feature branch: `git checkout -b feature/your-feature`.  
3. Commit changes: `git commit -m "Add feature"`.  
4. Push and open a pull request.  

Adhere to C11/C++14 standards and include comments.

## License

Licensed under the Apache License 2.0. See `LICENSE` for details.

## Acknowledgments

Developed by DarkStarStrix. Feedback is welcome via GitHub Issues.
