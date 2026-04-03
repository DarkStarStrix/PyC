/* pyc_wrapper.h
 * Master include for bindgen — pulls in all public PyC compiler headers.
 * This file is consumed by build.rs to generate Rust FFI bindings.
 */
#include "pyc/ir.h"
#include "pyc/compiler_api.h"
#include "pyc/kernel_registry.h"
#include "pyc/runtime_allocator.h"
#include "pyc/optimizer_policy.h"
#include "pyc/cuda_backend.h"
#include "pyc/ai_bridge.h"
