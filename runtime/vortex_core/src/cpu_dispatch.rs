//! cpu_dispatch.rs — Lock-free async CPU dispatcher.
//!
//! Replaces the original `std::sync::mpsc` + `Mutex` dispatcher with
//! `crossbeam_channel`, which provides a lock-free MPMC queue with
//! significantly lower tail latency under contention (up to 15x in
//! benchmarks vs. Mutex-based channels).
//!
//! The dispatcher is the "CPU side" of the conveyor belt pipeline:
//!   - Producer: the Python control plane or Rust pipeline submits jobs
//!   - Consumer: a pool of CPU worker threads executes preprocessing,
//!     then hands off to the GPU via the PyC CUDA backend.

use crossbeam_channel::{bounded, Receiver, Sender};
use std::thread;
use crate::errors::VortexError;

type Job = Box<dyn FnOnce() + Send + 'static>;

/// A handle returned by `dispatch()` that can be used to await completion.
pub struct DispatchHandle {
    receiver: crossbeam_channel::Receiver<Result<(), String>>,
}

impl DispatchHandle {
    /// Block until the dispatched job completes.
    pub fn join(self) -> Result<(), VortexError> {
        self.receiver
            .recv()
            .map_err(|_| VortexError::DispatchFailed("worker channel closed".to_string()))?
            .map_err(|e| VortexError::DispatchFailed(e))
    }
}

/// The CPU dispatcher. Maintains a pool of worker threads fed by a
/// lock-free bounded channel.
pub struct CpuDispatcher {
    sender: Sender<Job>,
    _workers: Vec<thread::JoinHandle<()>>,
}

impl CpuDispatcher {
    /// Create a dispatcher with `num_workers` threads and a queue depth
    /// of `queue_depth` jobs. Bounded queue provides back-pressure,
    /// preventing unbounded memory growth when the GPU is the bottleneck.
    pub fn new(num_workers: usize) -> Self {
        let queue_depth = num_workers * 8;
        let (sender, receiver) = bounded::<Job>(queue_depth);

        let workers = (0..num_workers)
            .map(|id| {
                let rx = receiver.clone();
                thread::Builder::new()
                    .name(format!("pyc-worker-{}", id))
                    .spawn(move || {
                        log::debug!("PyC worker {} started", id);
                        while let Ok(job) = rx.recv() {
                            job();
                        }
                        log::debug!("PyC worker {} exiting", id);
                    })
                    .expect("failed to spawn worker thread")
            })
            .collect();

        CpuDispatcher {
            sender,
            _workers: workers,
        }
    }

    /// Submit a job to the worker pool. Returns immediately (non-blocking).
    /// Back-pressure: blocks if the queue is full (bounded channel).
    pub fn dispatch<F>(&self, f: F) -> Result<(), VortexError>
    where
        F: FnOnce() + Send + 'static,
    {
        self.sender
            .send(Box::new(f))
            .map_err(|_| VortexError::DispatchFailed("worker pool shut down".to_string()))
    }

    /// Submit a job and return a handle to await its result.
    pub fn dispatch_with_result<F, R>(&self, f: F) -> Result<DispatchHandle, VortexError>
    where
        F: FnOnce() -> Result<R, String> + Send + 'static,
        R: Send + 'static,
    {
        let (tx, rx) = crossbeam_channel::bounded(1);
        self.dispatch(move || {
            let result = f().map(|_| ());
            let _ = tx.send(result);
        })?;
        Ok(DispatchHandle { receiver: rx })
    }

    /// Returns the number of jobs currently queued (approximate).
    pub fn queue_len(&self) -> usize {
        self.sender.len()
    }
}
