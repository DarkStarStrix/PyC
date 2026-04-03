/// build.rs — Generates Rust FFI bindings from PyC compiler C headers
/// using bindgen, and links against libpyc_compiler.so at runtime.
use std::env;
use std::path::PathBuf;

fn main() {
    // --------------------------------------------------------
    // 1. Tell Cargo where to find libpyc_compiler.so
    //    The CMake superbuild places it in ${CMAKE_BINARY_DIR}/compiler/
    //    We resolve via PYC_COMPILER_LIB_DIR env var (set by CMake).
    // --------------------------------------------------------
    let lib_dir = env::var("PYC_COMPILER_LIB_DIR")
        .unwrap_or_else(|_| "../../build/compiler".to_string());

    println!("cargo:rustc-link-search=native={}", lib_dir);
    println!("cargo:rustc-link-lib=dylib=pyc_compiler");

    // --------------------------------------------------------
    // 2. Re-run if any PyC header changes
    // --------------------------------------------------------
    println!("cargo:rerun-if-changed=../../include/pyc/compiler_api.h");
    println!("cargo:rerun-if-changed=../../include/pyc/ir.h");
    println!("cargo:rerun-if-changed=../../include/pyc/kernel_registry.h");
    println!("cargo:rerun-if-changed=../../include/pyc/runtime_allocator.h");
    println!("cargo:rerun-if-changed=../../include/pyc/optimizer_policy.h");
    println!("cargo:rerun-if-changed=../../include/pyc/cuda_backend.h");

    // --------------------------------------------------------
    // 3. Generate bindings via bindgen
    // --------------------------------------------------------
    let include_dir = env::var("PYC_COMPILER_INCLUDE_DIR")
        .unwrap_or_else(|_| "../../include".to_string());

    let bindings = bindgen::Builder::default()
        // Master wrapper header that pulls in all pyc public headers
        .header("src/ffi/pyc_wrapper.h")
        .clang_arg(format!("-I{}", include_dir))
        // Only generate bindings for pyc_ prefixed symbols
        .allowlist_function("pyc_.*")
        .allowlist_type("pyc_.*")
        .allowlist_var("PYC_.*")
        // Derive common traits
        .derive_debug(true)
        .derive_default(true)
        .derive_copy(true)
        // Treat C enums as Rust enums
        .rustified_enum("pyc_.*")
        .generate()
        .expect("Unable to generate PyC FFI bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("pyc_bindings.rs"))
        .expect("Couldn't write bindings");
}
