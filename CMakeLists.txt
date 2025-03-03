cmake_minimum_required(VERSION 3.29.6)
project(MyCompiler C CXX)

# Set CMake policies to NEW
cmake_policy(SET CMP0114 NEW)
cmake_policy(SET CMP0116 NEW)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

# Set the LLVM directory
set(LLVM_DIR "C:/Users/kunya/CLionProjects/PyC/llvm-project/build/lib/cmake/llvm")

# Add source files
add_executable(MyCompiler
    C_Files/main.c
    src/stack.c
    src/queue.c
    src/linked_list.c
    src/linked_list_node.c
    src/linked_list_iterator.c
    Header_Files/backend.h
    Header_Files/codegen.c
    Header_Files/parser.h
    Header_Files/parser.c
    Header_Files/test_parser.c
)

# Find and link LLVM libraries
find_package(LLVM REQUIRED CONFIG)
llvm_map_components_to_libnames(llvm_libs core support irreader)

target_include_directories(MyCompiler PRIVATE ${LLVM_INCLUDE_DIRS})
target_link_libraries(MyCompiler ${llvm_libs})

# Set the output directory
set_target_properties(MyCompiler PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Set the working directory
set_target_properties(MyCompiler PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Set the startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT MyCompiler)

# Set the CMake configuration
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configuration types" FORCE)

# Set the default configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Set the default flags
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /EHsc")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /Zi")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()
