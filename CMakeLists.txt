cmake_minimum_required(VERSION 3.10)
project(PyC_Core C CXX)

# Set the source directory
set(SRC_DIR "${CMAKE_SOURCE_DIR}/Core/C_Files")
set(HEADER_DIR "${CMAKE_SOURCE_DIR}/Core/Header")

# Gather all .c and .cpp files
file(GLOB SOURCES
    "${SRC_DIR}/*.c"
    "${SRC_DIR}/*.cpp"
)

# Add the executable
add_executable(PyC_Core ${SOURCES})

# Ensure the compiler can find your header files!
target_include_directories(PyC_Core PRIVATE ${HEADER_DIR})

# (Optional) Set C++ standard if needed
set_target_properties(PyC_Core PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
)

# (Optional) Link libraries here, e.g.:
# target_link_libraries(PyC_Core PRIVATE <library_name>)
