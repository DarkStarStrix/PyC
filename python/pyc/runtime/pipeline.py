"""
pyc.runtime.pipeline — Python interface to the async execution pipeline.

Wraps the Rust Pipeline via PyO3, or provides a pure-Python stub
for development without a CUDA build.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional, List, Any
from pyc.runtime.hw_profile import detect_hardware, HardwareProfile


@dataclass
class PipelineConfig:
    cpu_workers:          int   = 4
    queue_depth:          int   = 16
    policy_mode:          int   = 2   # UTILIZATION_FIRST
    memory_budget_bytes:  int   = 0
    numa_node:            Optional[int] = None

    @classmethod
    def from_hardware(cls, hw: HardwareProfile) -> "PipelineConfig":
        return cls(
            cpu_workers         = max(hw.cpu_cores // 2, 2),
            queue_depth         = max(hw.gpu_count, 1) * 4,
            policy_mode         = 2,
            memory_budget_bytes = 0,
            numa_node           = hw.gpu_numa_node,
        )


@dataclass
class PipelineStats:
    batch_id:            int   = 0
    preprocess_us:       int   = 0
    h2d_transfer_us:     int   = 0
    gpu_compute_us:      int   = 0
    d2h_transfer_us:     int   = 0
    total_us:            int   = 0
    ran_on_gpu:          bool  = False
    kernel_selected:     str   = ""
    peak_memory_bytes:   int   = 0


class Pipeline:
    """
    The main execution pipeline. Compiles and executes PyC IR modules
    using the async Vortex runtime and CUTLASS kernel backend.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig.from_hardware(detect_hardware())
        self._batch_counter = 0

        # Attempt to load native Rust pipeline
        try:
            from pyc._vortex_core import Pipeline as _NativePipeline
            self._native = _NativePipeline(self.config)
        except ImportError:
            self._native = None

    def execute(
        self,
        module: Any,
        inputs: List[Any],
        outputs: List[Any],
    ) -> PipelineStats:
        """
        Execute a compiled PyC IR module.

        If the native Rust pipeline is available, delegates to it.
        Otherwise, runs a Python-level simulation (CPU only).
        """
        if self._native is not None:
            raw = self._native.execute(module, inputs, outputs)
            return PipelineStats(
                batch_id          = raw.batch_id,
                preprocess_us     = raw.preprocess_us,
                h2d_transfer_us   = raw.h2d_transfer_us,
                gpu_compute_us    = raw.gpu_compute_us,
                d2h_transfer_us   = raw.d2h_transfer_us,
                total_us          = raw.total_us,
                ran_on_gpu        = raw.ran_on_gpu,
                kernel_selected   = raw.kernel_selected,
                peak_memory_bytes = raw.peak_memory_bytes,
            )

        # Python stub — for development without CUDA build
        t0 = time.perf_counter_ns()
        self._batch_counter += 1
        total_us = (time.perf_counter_ns() - t0) // 1000
        return PipelineStats(
            batch_id        = self._batch_counter,
            total_us        = total_us,
            ran_on_gpu      = False,
            kernel_selected = "cpu_stub",
        )


def init(config: Optional[PipelineConfig] = None) -> Pipeline:
    """
    Initialize the PyC runtime with hardware-aware defaults.

    Example::

        pipeline = pyc.init()
        stats = pipeline.execute(module, inputs, outputs)
    """
    hw = detect_hardware()
    cfg = config or PipelineConfig.from_hardware(hw)
    return Pipeline(cfg)
