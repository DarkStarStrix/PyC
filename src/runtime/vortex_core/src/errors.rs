//! errors.rs — Unified error type for the vortex_core runtime.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum VortexError {
    #[error("Allocation failed: {0}")]
    AllocationFailed(String),

    #[error("Allocation plan failed: {0}")]
    AllocPlanFailed(String),

    #[error("Kernel selection failed: {0}")]
    KernelSelectFailed(String),

    #[error("Dispatch failed: {0}")]
    DispatchFailed(String),

    #[error("Hardware detection failed: {0}")]
    HardwareDetectionFailed(String),

    #[error("FFI error: {0}")]
    FfiError(String),

    #[error("Pipeline initialization failed: {0}")]
    PipelineInitFailed(String),
}
