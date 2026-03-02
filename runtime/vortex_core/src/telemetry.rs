//! telemetry.rs — Lightweight telemetry broadcaster.
//!
//! Emits structured events to registered sinks (stdout, file, Python callback).
//! Uses a crossbeam channel internally to avoid blocking the hot path.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event")]
pub enum TelemetryEvent {
    BatchComplete {
        batch_id: u64,
        total_us: u64,
        ran_on_gpu: bool,
    },
    KernelSelected {
        op_key: String,
        symbol: String,
        estimated_occupancy: f64,
    },
    MemoryPressure {
        peak_bytes: usize,
        pressure_score: f64,
    },
    WorkerQueueDepth {
        depth: usize,
    },
}

/// A sink that receives telemetry events.
pub struct TelemetrySink {
    sender: crossbeam_channel::Sender<TelemetryEvent>,
    _receiver_thread: std::thread::JoinHandle<()>,
}

impl TelemetrySink {
    pub fn new() -> Self {
        let (tx, rx) = crossbeam_channel::unbounded();
        let thread = std::thread::Builder::new()
            .name("pyc-telemetry".to_string())
            .spawn(move || {
                while let Ok(event) = rx.recv() {
                    if let Ok(json) = serde_json::to_string(&event) {
                        log::debug!("[telemetry] {}", json);
                    }
                }
            })
            .expect("failed to spawn telemetry thread");
        TelemetrySink {
            sender: tx,
            _receiver_thread: thread,
        }
    }

    /// Emit an event. Non-blocking; drops the event if the channel is full.
    pub fn emit(&self, event: TelemetryEvent) {
        let _ = self.sender.try_send(event);
    }
}
