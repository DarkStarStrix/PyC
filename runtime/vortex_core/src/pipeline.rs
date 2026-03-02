//! pipeline.rs — The async conveyor belt pipeline.
//!
//! This is the central orchestrator of the PyC runtime. It:
//!   1. Receives a compiled `pyc_ir_module` from the compiler layer (via FFI).
//!   2. Runs the PyC memory planner to build an optimized allocation plan.
//!   3. Uses the PyC kernel registry (with CUTLASS kernels) to select kernels.
//!   4. Dispatches execution asynchronously, overlapping CPU preprocessing,
//!      DMA transfer, and GPU compute to eliminate idle bubbles.
//!   5. Publishes per-batch telemetry via the TelemetrySink.

use crate::allocator::{Allocator, AllocatorConfig};
use crate::cpu_dispatch::CpuDispatcher;
use crate::errors::VortexError;
use crate::ffi::{self, pyc_objective_mode, pyc_backend};
use crate::hw_profile::HardwareProfile;
use crate::telemetry::{TelemetryEvent, TelemetrySink};
use crossbeam_channel::{bounded, Receiver, Sender};
use std::sync::Arc;
use std::time::Instant;
use tokio::runtime::Runtime;

/// Configuration for the pipeline, derived from hardware topology.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of CPU worker threads for preprocessing
    pub cpu_workers: usize,
    /// Depth of the async work queue (number of batches in-flight)
    pub queue_depth: usize,
    /// Optimizer policy mode
    pub policy_mode: pyc_objective_mode,
    /// Memory budget in bytes (0 = unlimited)
    pub memory_budget_bytes: usize,
    /// NUMA node to pin memory to (None = auto)
    pub numa_node: Option<usize>,
}

impl PipelineConfig {
    /// Derive sensible defaults from the detected hardware topology.
    pub fn from_hardware(hw: &HardwareProfile) -> Self {
        PipelineConfig {
            cpu_workers: (hw.cpu_cores / 2).max(2),
            queue_depth: hw.gpu_count.max(1) * 4,
            policy_mode: pyc_objective_mode::PYC_MODE_UTILIZATION_FIRST,
            memory_budget_bytes: 0,
            numa_node: hw.gpu_numa_node,
        }
    }
}

/// Per-batch execution statistics.
#[derive(Debug, Default, Clone)]
pub struct PipelineStats {
    pub batch_id: u64,
    pub preprocess_us: u64,
    pub h2d_transfer_us: u64,
    pub gpu_compute_us: u64,
    pub d2h_transfer_us: u64,
    pub total_us: u64,
    pub ran_on_gpu: bool,
    pub kernel_selected: String,
    pub peak_memory_bytes: usize,
}

/// The main pipeline handle.
pub struct Pipeline {
    config: PipelineConfig,
    dispatcher: CpuDispatcher,
    allocator: Arc<Allocator>,
    telemetry: TelemetrySink,
    batch_counter: std::sync::atomic::AtomicU64,
}

impl Pipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: PipelineConfig) -> Result<Self, VortexError> {
        let allocator = Arc::new(Allocator::new(AllocatorConfig {
            numa_node: config.numa_node,
            use_pinned_memory: true,
        }));
        let dispatcher = CpuDispatcher::new(config.cpu_workers);
        let telemetry = TelemetrySink::new();
        Ok(Pipeline {
            config,
            dispatcher,
            allocator,
            telemetry,
            batch_counter: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Execute a compiled PyC IR module against the given input data.
    ///
    /// This is the hot path. It:
    ///   1. Calls the PyC memory planner to build an allocation plan.
    ///   2. Selects the best kernel via the PyC kernel registry.
    ///   3. Dispatches to GPU (with CPU fallback) via the PyC CUDA backend.
    ///   4. Emits telemetry.
    pub fn execute(
        &self,
        module: &ffi::pyc_ir_module,
        inputs: &[ffi::pyc_tensor],
        outputs: &mut [ffi::pyc_tensor],
    ) -> Result<PipelineStats, VortexError> {
        let batch_id = self
            .batch_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let t0 = Instant::now();

        // ---- Step 1: Build memory allocation plan via PyC planner ----
        let mut alloc_plan = ffi::pyc_alloc_plan::default();
        // Add tensor allocation requests derived from the IR module
        // (In full implementation, walk module ops to populate requests)
        let alloc_stats = ffi::build_alloc_plan(
            &mut alloc_plan,
            self.config.policy_mode,
            self.config.memory_budget_bytes,
        )
        .map_err(|e| VortexError::AllocPlanFailed(e.to_string()))?;

        // ---- Step 2: Select kernel via PyC kernel registry ----
        // The kernel registry now includes CUTLASS kernels registered
        // by cutlass_registry_init.cu at library load time.
        let kernel = ffi::select_kernel(
            "matmul",  // op_key — in practice derived from IR module
            ffi::pyc_backend::PYC_BACKEND_CUDA,
            self.config.policy_mode,
            alloc_stats.pressure_score,
        )
        .map_err(|e| VortexError::KernelSelectFailed(e.to_string()))?;

        let kernel_name = unsafe {
            std::ffi::CStr::from_ptr(kernel.symbol.as_ptr())
                .to_string_lossy()
                .into_owned()
        };

        // ---- Step 3: Dispatch via PyC CUDA backend ----
        let t_dispatch = Instant::now();
        let fell_back = ffi::cuda_dispatch(
            module,
            inputs,
            outputs,
            None,        // CPU fallback fn — None uses PyC's built-in fallback
            std::ptr::null_mut(),
        )
        .map_err(|e| VortexError::DispatchFailed(e.to_string()))?;

        let gpu_us = t_dispatch.elapsed().as_micros() as u64;
        let total_us = t0.elapsed().as_micros() as u64;

        // ---- Step 4: Emit telemetry ----
        let stats = PipelineStats {
            batch_id,
            preprocess_us: 0,  // populated by full implementation
            h2d_transfer_us: 0,
            gpu_compute_us: gpu_us,
            d2h_transfer_us: 0,
            total_us,
            ran_on_gpu: !fell_back,
            kernel_selected: kernel_name,
            peak_memory_bytes: alloc_stats.peak_bytes,
        };

        self.telemetry.emit(TelemetryEvent::BatchComplete {
            batch_id,
            total_us,
            ran_on_gpu: !fell_back,
        });

        Ok(stats)
    }
}
