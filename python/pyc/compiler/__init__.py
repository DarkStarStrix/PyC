"""
pyc.compiler — Python bindings for the PyC compiler layer.

Wraps the C-ABI of libpyc_compiler.so via ctypes, exposing:
  - IR module construction
  - Compilation with optimizer policy
  - Policy-driven kernel selection
  - Memory allocation planning
  - CUTLASS kernel registry queries
"""

import ctypes
import ctypes.util
import os
import sys
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional, List

# ----------------------------------------------------------------
# Locate and load libpyc_compiler.so
# ----------------------------------------------------------------

def _load_library() -> ctypes.CDLL:
    """Locate libpyc_compiler.so. Searches:
      1. PYC_COMPILER_LIB_DIR environment variable
      2. Standard system library paths
    """
    lib_dir = os.environ.get("PYC_COMPILER_LIB_DIR", "")
    candidates = []
    if lib_dir:
        candidates.append(os.path.join(lib_dir, "libpyc_compiler.so"))
    candidates.append(ctypes.util.find_library("pyc_compiler") or "")

    for path in candidates:
        if path and os.path.exists(path):
            return ctypes.CDLL(path)

    raise ImportError(
        "libpyc_compiler.so not found. "
        "Set PYC_COMPILER_LIB_DIR to the directory containing it, "
        "or build PyC with: cmake -B build && cmake --build build"
    )

try:
    _lib = _load_library()
    _available = True
except ImportError:
    _lib = None
    _available = False

# ----------------------------------------------------------------
# Enums (mirror pyc/optimizer_policy.h)
# ----------------------------------------------------------------

class ObjectiveMode(IntEnum):
    BALANCED          = 0
    MEMORY_FIRST      = 1
    UTILIZATION_FIRST = 2

class Backend(IntEnum):
    CPU  = 0
    CUDA = 1

# ----------------------------------------------------------------
# Data classes (mirror C structs)
# ----------------------------------------------------------------

@dataclass
class KernelDesc:
    op_key:               str   = ""
    backend:              int   = 0
    symbol:               str   = ""
    priority:             int   = 0
    estimated_occupancy:  float = 0.0
    tensor_core_eligible: bool  = False
    shared_mem_bytes:     int   = 0
    reg_pressure_class:   int   = 0

@dataclass
class AllocStats:
    peak_bytes:              int   = 0
    total_requested_bytes:   int   = 0
    reused_allocations:      int   = 0
    allocation_events:       int   = 0
    pressure_score:          float = 0.0

# ----------------------------------------------------------------
# Public API
# ----------------------------------------------------------------

def is_available() -> bool:
    """Returns True if libpyc_compiler.so was successfully loaded."""
    return _available


def select_kernel(
    op_key: str,
    backend: Backend = Backend.CUDA,
    mode: ObjectiveMode = ObjectiveMode.UTILIZATION_FIRST,
    pressure_score: float = 0.0,
) -> Optional[KernelDesc]:
    """
    Select the best kernel for `op_key` on `backend` given the current
    optimizer policy mode and memory pressure score.

    Returns a KernelDesc on success, or None if no kernel is registered.

    Example::

        kernel = pyc.compiler.select_kernel("matmul", mode=ObjectiveMode.UTILIZATION_FIRST)
        print(kernel.symbol)  # e.g. "cutlass_gemm_tensorcore_bf16"
    """
    if not _available:
        raise RuntimeError("PyC compiler library not available")

    # C struct layout mirrors pyc_kernel_desc
    class _KernelDesc(ctypes.Structure):
        _fields_ = [
            ("op_key",               ctypes.c_char * 64),
            ("backend",              ctypes.c_int),
            ("symbol",               ctypes.c_char * 128),
            ("priority",             ctypes.c_int),
            ("estimated_occupancy",  ctypes.c_double),
            ("tensor_core_eligible", ctypes.c_int),
            ("shared_mem_bytes",     ctypes.c_size_t),
            ("reg_pressure_class",   ctypes.c_int),
        ]

    class _Trace(ctypes.Structure):
        _fields_ = [
            ("selected_score",                ctypes.c_double),
            ("selected_estimated_utilization", ctypes.c_double),
            ("pressure_penalty",              ctypes.c_double),
        ]

    out   = _KernelDesc()
    trace = _Trace()

    fn = _lib.pyc_kernel_select_with_policy
    fn.restype  = ctypes.c_int
    fn.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(_KernelDesc),
        ctypes.POINTER(_Trace),
    ]

    found = fn(
        op_key.encode(),
        int(backend),
        int(mode),
        pressure_score,
        ctypes.byref(out),
        ctypes.byref(trace),
    )

    if found != 1:
        return None

    return KernelDesc(
        op_key               = out.op_key.decode().rstrip("\x00"),
        backend              = out.backend,
        symbol               = out.symbol.decode().rstrip("\x00"),
        priority             = out.priority,
        estimated_occupancy  = out.estimated_occupancy,
        tensor_core_eligible = bool(out.tensor_core_eligible),
        shared_mem_bytes     = out.shared_mem_bytes,
        reg_pressure_class   = out.reg_pressure_class,
    )


def cutlass_kernel_count(op_key: str) -> int:
    """
    Returns the number of CUTLASS kernels registered for `op_key`.
    Useful for diagnostics.

    Example::

        n = pyc.compiler.cutlass_kernel_count("matmul")
        # Returns 3 (FP16 TensorCore, BF16 TensorCore, FP32 SIMT)
    """
    if not _available:
        return 0
    fn = _lib.pyc_cutlass_kernel_count
    fn.restype  = ctypes.c_int
    fn.argtypes = [ctypes.c_char_p]
    return fn(op_key.encode())
