cmake_minimum_required(VERSION 3.10)
project(PyC_Core C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HEADER_DIR "${CMAKE_SOURCE_DIR}/Core/Header_Files")

# Shared object compilation unit for the currently stable/portable core files.
add_library(pyc_core_obj OBJECT
    Core/C_Files/stack.c
    Core/C_Files/symbol_table.c
)

target_include_directories(pyc_core_obj PUBLIC ${HEADER_DIR})

# Canonical stable core library name.
add_library(pyc_core STATIC
    $<TARGET_OBJECTS:pyc_core_obj>
)

target_include_directories(pyc_core PUBLIC ${HEADER_DIR})

# Compatibility target retained for existing scripts/CI that still reference
# `pyc_foundation`.
add_library(pyc_foundation STATIC
    $<TARGET_OBJECTS:pyc_core_obj>
)

target_include_directories(pyc_foundation PUBLIC ${HEADER_DIR})

# Always provide an executable target so CI has a deterministic build target
# across Linux/macOS/Windows, even while experimental compiler modules are
# being stabilized.
add_executable(pyc
    Core/C_Files/ci_main.c
)

target_link_libraries(pyc PRIVATE pyc_core)

option(PYC_BUILD_EXPERIMENTAL "Build experimental compiler sources" OFF)

if(PYC_BUILD_EXPERIMENTAL)
    message(STATUS "Building experimental PyC compiler sources")

    add_executable(PyC_Core
        Core/C_Files/Core.cpp
        Core/C_Files/IR.c
        Core/C_Files/api.c
        Core/C_Files/backend.c
        Core/C_Files/codegen.c
        Core/C_Files/error_handler.c
        Core/C_Files/frontend.c
        Core/C_Files/ir_generator.c
        Core/C_Files/lexer.c
        Core/C_Files/main.c
        Core/C_Files/parser.c
    )

    target_include_directories(PyC_Core PRIVATE ${HEADER_DIR})
endif()
