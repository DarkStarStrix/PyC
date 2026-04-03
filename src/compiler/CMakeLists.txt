# ============================================================
# PyC Compiler Layer
# Sources: ir, passes, kernel_registry, runtime_allocator,
#          compiler_api, cuda_backend, CUTLASS kernels
# ============================================================

# --- Core compiler sources (C, always built) ---
set(PYC_COMPILER_SOURCES
    ir/ir.c
    passes/pass_manager.c
    runtime/kernel_registry.c
    runtime/runtime_allocator.c
    runtime/runtime_control.c
    compiler_api.c
)

add_library(pyc_compiler STATIC ${PYC_COMPILER_SOURCES})
target_include_directories(pyc_compiler PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_compile_features(pyc_compiler PRIVATE c_std_11)

# Expose pyc_compiler as a shared lib for Rust FFI consumption
add_library(pyc_compiler_shared SHARED ${PYC_COMPILER_SOURCES})
target_include_directories(pyc_compiler_shared PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
set_target_properties(pyc_compiler_shared PROPERTIES
    OUTPUT_NAME "pyc_compiler"
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)

# --- CUDA backend (optional) ---
if(PYC_BUILD_CUDA)
    add_library(pyc_cuda_backend STATIC runtime/cuda_backend.cu)
    target_include_directories(pyc_cuda_backend PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_compile_features(pyc_cuda_backend PRIVATE cxx_std_17)
    set_target_properties(pyc_cuda_backend PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "80;86;89;90"   # A100, RTX3090, RTX4090, H100
    )
    if(CUBLAS_FOUND)
        target_link_libraries(pyc_cuda_backend PUBLIC CUDA::cublas)
    endif()
    target_link_libraries(pyc_compiler PUBLIC pyc_cuda_backend)
    target_link_libraries(pyc_compiler_shared PUBLIC pyc_cuda_backend)

    # --- CUTLASS kernel library ---
    add_subdirectory(cutlass_kernels)
    target_link_libraries(pyc_cuda_backend PUBLIC pyc_cutlass_kernels)
endif()

# --- AI bridge (stub, always built) ---
add_library(pyc_ai_bridge STATIC ${CMAKE_SOURCE_DIR}/src/compiler/ai/ai_bridge.c)
target_include_directories(pyc_ai_bridge PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(pyc_ai_bridge PUBLIC pyc_compiler)
