cmake_minimum_required(VERSION 3.10)
project(PyC C)

option(PYC_BUILD_EXPERIMENTAL "Build experimental PyC_Core executable" OFF)
option(PYC_BUILD_BENCHMARKS "Build benchmark targets" OFF)

set(SRC_DIR "${CMAKE_SOURCE_DIR}/Core/C_Files")
set(HEADER_DIR "${CMAKE_SOURCE_DIR}/Core/Header_Files")

set(PYC_CORE_SOURCES
    "${SRC_DIR}/adapter.c"
    "${SRC_DIR}/semantic.c"
    "${SRC_DIR}/stack.c"
    "${SRC_DIR}/symbol_table.c"
)

add_library(pyc_core_obj OBJECT ${PYC_CORE_SOURCES})
target_include_directories(pyc_core_obj PUBLIC "${HEADER_DIR}")
target_compile_features(pyc_core_obj PUBLIC c_std_11)

add_library(pyc_core STATIC $<TARGET_OBJECTS:pyc_core_obj>)
target_include_directories(pyc_core PUBLIC "${HEADER_DIR}")
target_compile_features(pyc_core PUBLIC c_std_11)

# Compatibility target kept for downstream consumers that still reference pyc_foundation.
add_library(pyc_foundation STATIC $<TARGET_OBJECTS:pyc_core_obj>)
target_include_directories(pyc_foundation PUBLIC "${HEADER_DIR}")
target_compile_features(pyc_foundation PUBLIC c_std_11)

add_executable(pyc "${SRC_DIR}/ci_main.c")
target_link_libraries(pyc PRIVATE pyc_core)

if(PYC_BUILD_EXPERIMENTAL)
  add_executable(PyC_Core
      "${SRC_DIR}/main.c"
      $<TARGET_OBJECTS:pyc_core_obj>
  )
  target_include_directories(PyC_Core PRIVATE "${HEADER_DIR}")
  target_compile_features(PyC_Core PRIVATE c_std_11)
endif()

if(PYC_BUILD_BENCHMARKS)
  add_executable(pyc_core_microbench
      "${CMAKE_SOURCE_DIR}/benchmark/workloads/core_microbench.c"
  )
  target_include_directories(pyc_core_microbench PRIVATE "${HEADER_DIR}")
  target_compile_features(pyc_core_microbench PRIVATE c_std_11)
  target_link_libraries(pyc_core_microbench PRIVATE pyc_core)
endif()
