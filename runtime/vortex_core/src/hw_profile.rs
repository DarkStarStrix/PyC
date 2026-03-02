//! hw_profile.rs — Hardware topology detection.
//!
//! Detects CPU cores, NUMA nodes, GPU count, and the NUMA node
//! closest to each GPU. Used by PipelineConfig::from_hardware()
//! to derive optimal dispatch and allocator settings.

use serde::{Deserialize, Serialize};
use sysinfo::{System, SystemExt, CpuExt};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HardwareProfile {
    /// Total logical CPU cores
    pub cpu_cores: usize,
    /// Number of NUMA nodes (1 on most consumer systems, 2+ on server)
    pub numa_nodes: usize,
    /// Number of CUDA-capable GPUs detected
    pub gpu_count: usize,
    /// NUMA node index closest to GPU 0 (None if single-node or unknown)
    pub gpu_numa_node: Option<usize>,
    /// Total system RAM in bytes
    pub total_ram_bytes: u64,
    /// CPU architecture string (e.g., "x86_64")
    pub cpu_arch: String,
}

/// Detect the hardware topology of the current machine.
pub fn detect_hardware() -> HardwareProfile {
    let mut sys = System::new_all();
    sys.refresh_all();

    let cpu_cores = sys.cpus().len().max(1);
    let total_ram_bytes = sys.total_memory();
    let cpu_arch = std::env::consts::ARCH.to_string();

    // NUMA node count — read from /sys/devices/system/node/ on Linux
    let numa_nodes = count_numa_nodes();

    // GPU count — attempt via nvidia-smi, fall back to 0
    let (gpu_count, gpu_numa_node) = detect_gpus();

    HardwareProfile {
        cpu_cores,
        numa_nodes,
        gpu_count,
        gpu_numa_node,
        total_ram_bytes,
        cpu_arch,
    }
}

fn count_numa_nodes() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(entries) = std::fs::read_dir("/sys/devices/system/node") {
            let count = entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.file_name()
                        .to_string_lossy()
                        .starts_with("node")
                })
                .count();
            if count > 0 {
                return count;
            }
        }
    }
    1
}

fn detect_gpus() -> (usize, Option<usize>) {
    // Try nvidia-smi to count GPUs
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output();

    let gpu_count = match output {
        Ok(out) if out.status.success() => {
            String::from_utf8_lossy(&out.stdout)
                .lines()
                .filter(|l| !l.trim().is_empty())
                .count()
        }
        _ => 0,
    };

    // Try to read GPU NUMA node from sysfs (Linux only)
    #[cfg(target_os = "linux")]
    let gpu_numa_node = {
        std::fs::read_to_string("/sys/bus/pci/devices/0000:00:00.0/numa_node")
            .ok()
            .and_then(|s| s.trim().parse::<isize>().ok())
            .and_then(|n| if n >= 0 { Some(n as usize) } else { None })
    };

    #[cfg(not(target_os = "linux"))]
    let gpu_numa_node = None;

    (gpu_count, gpu_numa_node)
}
