"""
pyc.runtime.hw_profile — Hardware topology detection.

Wraps the Rust vortex_core hardware profiler. Falls back to a
pure-Python implementation when the native extension is unavailable.
"""

import os
import subprocess
import platform
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class HardwareProfile:
    cpu_cores:       int           = 1
    numa_nodes:      int           = 1
    gpu_count:       int           = 0
    gpu_numa_node:   Optional[int] = None
    total_ram_bytes: int           = 0
    cpu_arch:        str           = ""

    def __str__(self) -> str:
        return (
            f"HardwareProfile("
            f"cpu_cores={self.cpu_cores}, "
            f"numa_nodes={self.numa_nodes}, "
            f"gpu_count={self.gpu_count}, "
            f"gpu_numa_node={self.gpu_numa_node}, "
            f"ram={self.total_ram_bytes / 1e9:.1f} GB, "
            f"arch={self.cpu_arch})"
        )


def detect_hardware() -> HardwareProfile:
    """
    Detect the hardware topology of the current machine.

    Attempts to use the native Rust vortex_core extension first;
    falls back to a pure-Python implementation using os/subprocess.
    """
    try:
        from pyc._vortex_core import detect_hardware as _detect
        raw = _detect()
        return HardwareProfile(
            cpu_cores       = raw.cpu_cores,
            numa_nodes      = raw.numa_nodes,
            gpu_count       = raw.gpu_count,
            gpu_numa_node   = raw.gpu_numa_node,
            total_ram_bytes = raw.total_ram_bytes,
            cpu_arch        = raw.cpu_arch,
        )
    except ImportError:
        pass

    # Pure-Python fallback
    import multiprocessing
    cpu_cores = multiprocessing.cpu_count()

    # NUMA nodes
    numa_nodes = 1
    if platform.system() == "Linux":
        try:
            nodes = [
                d for d in os.listdir("/sys/devices/system/node")
                if d.startswith("node")
            ]
            numa_nodes = max(len(nodes), 1)
        except OSError:
            pass

    # GPU count via nvidia-smi
    gpu_count = 0
    gpu_numa_node = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpu_count = len([l for l in result.stdout.splitlines() if l.strip()])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Total RAM
    total_ram_bytes = 0
    try:
        import psutil
        total_ram_bytes = psutil.virtual_memory().total
    except ImportError:
        pass

    return HardwareProfile(
        cpu_cores       = cpu_cores,
        numa_nodes      = numa_nodes,
        gpu_count       = gpu_count,
        gpu_numa_node   = gpu_numa_node,
        total_ram_bytes = total_ram_bytes,
        cpu_arch        = platform.machine(),
    )
