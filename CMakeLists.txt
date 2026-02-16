cmake_minimum_required(VERSION 3.10)
project(PyC_Core C)

option(ENABLE_OPTIMIZE "Enable optimize command contract" OFF)
option(ENABLE_VISUALIZE "Enable visualize command contract" OFF)
option(ENABLE_KERNEL "Enable kernel command contract" OFF)

set(SRC_DIR "${CMAKE_SOURCE_DIR}/Core/C_Files")
set(HEADER_DIR "${CMAKE_SOURCE_DIR}/Core/Header_Files")

set(SOURCES
    ${SRC_DIR}/main.c
    ${SRC_DIR}/api.c
    ${SRC_DIR}/adapter.c
    ${SRC_DIR}/lexer.c
    ${SRC_DIR}/parser.c
    ${SRC_DIR}/semantic.c
    ${SRC_DIR}/ir_generator.c
    ${SRC_DIR}/backend.c
    ${SRC_DIR}/error_handler.c
    ${SRC_DIR}/symbol_table.c
)

add_executable(PyC_Core ${SOURCES})
target_include_directories(PyC_Core PRIVATE ${HEADER_DIR})
target_compile_features(PyC_Core PRIVATE c_std_11)

if(ENABLE_OPTIMIZE)
    target_compile_definitions(PyC_Core PRIVATE ENABLE_OPTIMIZE)
endif()
if(ENABLE_VISUALIZE)
    target_compile_definitions(PyC_Core PRIVATE ENABLE_VISUALIZE)
endif()
if(ENABLE_KERNEL)
    target_compile_definitions(PyC_Core PRIVATE ENABLE_KERNEL)
endif()
