# ============================================================
# PyC CUTLASS Kernel Library
# Provides high-performance GEMM, Conv2d, and attention
# kernels via NVIDIA CUTLASS, registered into pyc_kernel_registry.
# ============================================================

set(PYC_CUTLASS_SOURCES
    gemm/ada_async_kernel.cu
    registry/init.cu
)

if(PYC_CUTLASS_FULL_ENABLED)
    list(APPEND PYC_CUTLASS_SOURCES
        gemm/kernel.cu
        conv2d/kernel.cu
        attention/kernel.cu
    )
else()
    list(APPEND PYC_CUTLASS_SOURCES
        gemm/dispatch_registry.cu
    )
endif()

add_library(pyc_cutlass_kernels STATIC ${PYC_CUTLASS_SOURCES})

target_include_directories(pyc_cutlass_kernels PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

if(PYC_CUTLASS_FULL_ENABLED)
    target_include_directories(pyc_cutlass_kernels PUBLIC
        ${PYC_CUTLASS_PATH}
        ${PYC_CUTLASS_PATH}/../tools/util/include
    )
endif()

target_compile_features(pyc_cutlass_kernels PRIVATE cxx_std_17)

set_target_properties(pyc_cutlass_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "80;86;89;90"
)

# CUTLASS requires these compile definitions
target_compile_definitions(pyc_cutlass_kernels PRIVATE
    $<$<BOOL:${PYC_CUTLASS_FULL_ENABLED}>:PYC_CUTLASS_FULL_KERNELS=1>
    $<$<BOOL:${PYC_CUTLASS_FULL_ENABLED}>:CUTLASS_ENABLE_TENSOR_CORE_MMA=1>
    $<$<BOOL:${PYC_CUTLASS_FULL_ENABLED}>:CUTLASS_DEBUG_TRACE_LEVEL=0>
)

target_link_libraries(pyc_cutlass_kernels PUBLIC
    CUDA::cudart
    CUDA::cublas
)
