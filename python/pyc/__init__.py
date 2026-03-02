"""
PyC — Unified HPC Toolchain
===========================

The top-level Python SDK for the PyC toolchain. Provides a clean,
high-level interface over the three integrated layers:

  - pyc.compiler  : IR construction, pass management, kernel policy
  - pyc.runtime   : Async dispatch, NUMA memory, hardware profiling
  - pyc.distributed: FSDP sharding, collective communication
  - pyc.apps      : SciML inference API (from Nexa_Inference)

Quick start::

    import pyc

    # Detect hardware and initialize the runtime
    hw = pyc.detect_hardware()
    pipeline = pyc.init()

    # Compile and run a computation graph
    module = pyc.compiler.build_module(ops=[...])
    pyc.compiler.compile(module)
    stats = pipeline.execute(module, inputs, outputs)

    print(f"Ran on GPU: {stats.ran_on_gpu}")
    print(f"Kernel: {stats.kernel_selected}")
    print(f"Peak memory: {stats.peak_memory_bytes / 1e9:.2f} GB")
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pyc")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

# Re-export top-level convenience functions
from pyc.runtime.pipeline import init, Pipeline, PipelineConfig, PipelineStats
from pyc.runtime.hw_profile import detect_hardware, HardwareProfile
from pyc import compiler
from pyc import runtime
from pyc import distributed
from pyc import apps

__all__ = [
    "init",
    "Pipeline",
    "PipelineConfig",
    "PipelineStats",
    "detect_hardware",
    "HardwareProfile",
    "compiler",
    "runtime",
    "distributed",
    "apps",
    "__version__",
]
