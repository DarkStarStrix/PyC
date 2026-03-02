"""
pyc.runtime — Python interface to the Vortex async execution engine.

Wraps the Rust vortex_core cdylib via PyO3 (when built with the
python_ext feature) or falls back to a pure-Python simulation for
development and testing without a CUDA environment.
"""

from pyc.runtime.hw_profile import detect_hardware, HardwareProfile
from pyc.runtime.pipeline import init, Pipeline, PipelineConfig, PipelineStats

__all__ = [
    "detect_hardware",
    "HardwareProfile",
    "init",
    "Pipeline",
    "PipelineConfig",
    "PipelineStats",
]
