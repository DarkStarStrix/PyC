cmake_minimum_required(VERSION 3.10)
project(PyC_Core C)

option(ENABLE_OPTIMIZE "Enable optimize command contract" OFF)
option(ENABLE_VISUALIZE "Enable visualize command contract" OFF)
option(ENABLE_KERNEL "Enable kernel command contract" OFF)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(CORE_SRC_DIR "${CMAKE_SOURCE_DIR}/Core/C_Files")
set(CORE_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/Core/Header_Files")

set(PYC_CORE_SOURCES
    "${CORE_SRC_DIR}/api.c"
    "${CORE_SRC_DIR}/backend.c"
    "${CORE_SRC_DIR}/codegen.c"
    "${CORE_SRC_DIR}/error_handler.c"
    "${CORE_SRC_DIR}/frontend.c"
    "${CORE_SRC_DIR}/IR.c"
    "${CORE_SRC_DIR}/ir_generator.c"
    "${CORE_SRC_DIR}/lexer.c"
    "${CORE_SRC_DIR}/parser.c"
    "${CORE_SRC_DIR}/stack.c"
    "${CORE_SRC_DIR}/symbol_table.c"
)

add_library(pyc_core STATIC ${PYC_CORE_SOURCES})
target_include_directories(pyc_core
    PUBLIC
        "${CORE_INCLUDE_DIR}"
)

# Authoritative runtime entrypoint.
add_executable(pyc
    "${CMAKE_SOURCE_DIR}/Third_Party/main.cpp"
)
target_include_directories(pyc
    PRIVATE
        "${CORE_INCLUDE_DIR}"
        "${CMAKE_SOURCE_DIR}/Third_Party"
)
target_link_libraries(pyc PRIVATE pyc_core)

option(BUILD_PYC_TESTS "Build optional test executables" OFF)
if(BUILD_PYC_TESTS)
    add_executable(test_parser
        "${CORE_SRC_DIR}/test_parser.c"
        "${CORE_SRC_DIR}/parser.c"
        "${CORE_SRC_DIR}/lexer.c"
        "${CORE_SRC_DIR}/error_handler.c"
    )
    target_include_directories(test_parser PRIVATE "${CORE_INCLUDE_DIR}")
endif()
