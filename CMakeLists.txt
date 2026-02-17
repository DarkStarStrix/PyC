cmake_minimum_required(VERSION 3.10)
project(PyC C)

option(PYC_BUILD_BENCHMARKS "Build benchmark targets" OFF)
option(PYC_BUILD_COMPILER_NEXT "Build next-generation compiler stack (experimental)" ON)
option(PYC_BUILD_COMPILER_NEXT_TESTS "Build next-generation compiler smoke tests" ON)

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

if(PYC_BUILD_BENCHMARKS)
  add_executable(pyc_core_microbench
      "${CMAKE_SOURCE_DIR}/benchmark/workloads/core_microbench.c"
  )
  target_include_directories(pyc_core_microbench PRIVATE "${HEADER_DIR}")
  target_compile_features(pyc_core_microbench PRIVATE c_std_11)
  target_link_libraries(pyc_core_microbench PRIVATE pyc_core)
endif()

if(PYC_BUILD_COMPILER_NEXT)
  set(PYC_COMPILER_NEXT_SOURCES
      "${CMAKE_SOURCE_DIR}/compiler/compiler_api.c"
      "${CMAKE_SOURCE_DIR}/compiler/ir/ir.c"
      "${CMAKE_SOURCE_DIR}/compiler/passes/pass_manager.c"
      "${CMAKE_SOURCE_DIR}/compiler/runtime/runtime_allocator.c"
      "${CMAKE_SOURCE_DIR}/compiler/runtime/kernel_registry.c"
  )

  add_library(pyc_compiler_next STATIC ${PYC_COMPILER_NEXT_SOURCES})
  target_include_directories(pyc_compiler_next
      PUBLIC
          "${CMAKE_SOURCE_DIR}/include"
  )
  target_compile_features(pyc_compiler_next PUBLIC c_std_11)

  add_library(pyc_ai_bridge STATIC "${CMAKE_SOURCE_DIR}/AI/ai_bridge.c")
  target_include_directories(pyc_ai_bridge PUBLIC "${CMAKE_SOURCE_DIR}/include")
  target_link_libraries(pyc_ai_bridge PUBLIC pyc_compiler_next)
  target_compile_features(pyc_ai_bridge PUBLIC c_std_11)

  if(PYC_BUILD_COMPILER_NEXT_TESTS)
    function(pyc_add_compiler_next_test test_name source_file)
      add_executable(${test_name} "${source_file}")
      target_include_directories(${test_name} PRIVATE "${CMAKE_SOURCE_DIR}/include")
      target_link_libraries(${test_name} PRIVATE pyc_compiler_next)
      target_compile_features(${test_name} PRIVATE c_std_11)
      target_compile_definitions(${test_name} PRIVATE PYC_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
      add_test(NAME ${test_name} COMMAND ${test_name})
    endfunction()

    enable_testing()

    add_executable(pyc_compiler_next_smoke
        "${CMAKE_SOURCE_DIR}/tests/compiler_next/compiler_next_smoke.c"
    )
    target_include_directories(pyc_compiler_next_smoke PRIVATE "${CMAKE_SOURCE_DIR}/include")
    target_link_libraries(pyc_compiler_next_smoke PRIVATE pyc_compiler_next)
    target_compile_features(pyc_compiler_next_smoke PRIVATE c_std_11)
    add_test(NAME pyc_compiler_next_smoke_test COMMAND pyc_compiler_next_smoke)

    pyc_add_compiler_next_test(pyc_compiler_next_test_ir "${CMAKE_SOURCE_DIR}/tests/compiler_next/test_ir.c")
    pyc_add_compiler_next_test(pyc_compiler_next_test_pass_manager "${CMAKE_SOURCE_DIR}/tests/compiler_next/test_pass_manager.c")
    pyc_add_compiler_next_test(pyc_compiler_next_test_runtime_allocator "${CMAKE_SOURCE_DIR}/tests/compiler_next/test_runtime_allocator.c")
    pyc_add_compiler_next_test(pyc_compiler_next_test_kernel_registry "${CMAKE_SOURCE_DIR}/tests/compiler_next/test_kernel_registry.c")
    pyc_add_compiler_next_test(pyc_compiler_next_test_compiler_api "${CMAKE_SOURCE_DIR}/tests/compiler_next/test_compiler_api.c")
    pyc_add_compiler_next_test(pyc_compiler_next_test_determinism "${CMAKE_SOURCE_DIR}/tests/compiler_next/test_determinism.c")
    pyc_add_compiler_next_test(pyc_compiler_next_test_pass_golden "${CMAKE_SOURCE_DIR}/tests/compiler_next/test_pass_golden.c")
    pyc_add_compiler_next_test(pyc_compiler_next_test_policy_modes "${CMAKE_SOURCE_DIR}/tests/compiler_next/test_policy_modes.c")

    add_executable(pyc_compiler_next_test_ai_bridge "${CMAKE_SOURCE_DIR}/tests/compiler_next/test_ai_bridge.c")
    target_include_directories(pyc_compiler_next_test_ai_bridge PRIVATE "${CMAKE_SOURCE_DIR}/include")
    target_link_libraries(pyc_compiler_next_test_ai_bridge PRIVATE pyc_ai_bridge)
    target_compile_features(pyc_compiler_next_test_ai_bridge PRIVATE c_std_11)
    add_test(NAME pyc_compiler_next_test_ai_bridge COMMAND pyc_compiler_next_test_ai_bridge)

    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
      add_test(
        NAME pyc_validate_cmake_source_coverage
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/tests/validate_cmake_sources.py" "${CMAKE_SOURCE_DIR}/CMakeLists.txt" "${CMAKE_SOURCE_DIR}"
      )
    endif()
  endif()
endif()
