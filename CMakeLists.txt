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

if(ENABLE_OPTIMIZE)
    target_compile_definitions(PyC_Core PRIVATE ENABLE_OPTIMIZE)
endif()
if(ENABLE_VISUALIZE)
    target_compile_definitions(PyC_Core PRIVATE ENABLE_VISUALIZE)
endif()
if(ENABLE_KERNEL)
    target_compile_definitions(PyC_Core PRIVATE ENABLE_KERNEL)
endif()
