# ============================================================
# PyC CUTLASS Kernel Library
# Provides high-performance GEMM, Conv2d, and attention
# kernels via NVIDIA CUTLASS, registered into pyc_kernel_registry.
# ============================================================

set(PYC_CUTLASS_SOURCES
    cutlass_gemm.cu
    cutlass_conv2d.cu
    cutlass_attention.cu
    cutlass_registry_init.cu
)

add_library(pyc_cutlass_kernels STATIC ${PYC_CUTLASS_SOURCES})

target_include_directories(pyc_cutlass_kernels PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${PYC_CUTLASS_PATH}                     # CUTLASS headers
    ${PYC_CUTLASS_PATH}/../tools/util/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_features(pyc_cutlass_kernels PRIVATE cxx_std_17)

set_target_properties(pyc_cutlass_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80;86;89;90"
)

# CUTLASS requires these compile definitions
target_compile_definitions(pyc_cutlass_kernels PRIVATE
    CUTLASS_ENABLE_TENSOR_CORE_MMA=1
    CUTLASS_DEBUG_TRACE_LEVEL=0
)

target_link_libraries(pyc_cutlass_kernels PUBLIC
    CUDA::cudart
    CUDA::cublas
)
