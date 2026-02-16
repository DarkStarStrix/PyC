cmake_minimum_required(VERSION 3.10)
project(PyC_Core C)

option(ENABLE_OPTIMIZE "Enable optimize command contract" OFF)
option(ENABLE_VISUALIZE "Enable visualize command contract" OFF)
option(ENABLE_KERNEL "Enable kernel command contract" OFF)

# Set source/header directories
set(SRC_DIR "${CMAKE_SOURCE_DIR}/Core/C_Files")
set(HEADER_DIR "${CMAKE_SOURCE_DIR}/Core/Header_Files")

# Gather compiler sources and exclude test entrypoints
file(GLOB SOURCES
    "${SRC_DIR}/*.c"
    "${SRC_DIR}/*.cpp"
)

list(FILTER SOURCES EXCLUDE REGEX ".*/test_.*\\.c$")

# Add the executable
add_executable(PyC_Core ${SOURCES})

# Ensure the compiler can find header files
target_include_directories(PyC_Core PRIVATE ${HEADER_DIR})
target_compile_features(PyC_Core PRIVATE c_std_11)

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
