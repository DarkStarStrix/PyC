# ============================================================
# PyC Runtime Layer (Nexa_Vortex)
# Builds the Rust vortex_core crate via Cargo as an
# ExternalProject, then links the resulting cdylib into CMake.
# ============================================================

include(ExternalProject)

set(VORTEX_CARGO_MANIFEST "${CMAKE_SOURCE_DIR}/src/runtime/Cargo.toml")
set(VORTEX_TARGET_DIR     "${CMAKE_BINARY_DIR}/runtime/cargo_target")
set(VORTEX_LIB_PATH       "${VORTEX_TARGET_DIR}/release/libvortex_runtime.so")

# Determine Cargo build profile
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CARGO_PROFILE "debug")
    set(CARGO_PROFILE_FLAG "")
else()
    set(CARGO_PROFILE "release")
    set(CARGO_PROFILE_FLAG "--release")
endif()

ExternalProject_Add(vortex_runtime_cargo
    SOURCE_DIR        "${CMAKE_SOURCE_DIR}/src/runtime"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND
        cargo build ${CARGO_PROFILE_FLAG}
        --manifest-path "${VORTEX_CARGO_MANIFEST}"
        --target-dir    "${VORTEX_TARGET_DIR}"
    INSTALL_COMMAND   ""
    BUILD_IN_SOURCE   0
    BUILD_ALWAYS      1
    LOG_BUILD         1
)

# Expose the Rust cdylib as an IMPORTED shared library
add_library(vortex_runtime SHARED IMPORTED GLOBAL)
add_dependencies(vortex_runtime vortex_runtime_cargo)
set_target_properties(vortex_runtime PROPERTIES
    IMPORTED_LOCATION "${VORTEX_TARGET_DIR}/${CARGO_PROFILE}/libvortex_runtime.so"
)

# The runtime depends on the compiler shared lib for FFI
add_dependencies(vortex_runtime_cargo pyc_compiler_shared)
