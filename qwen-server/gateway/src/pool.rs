use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Bounded request admission + single in-flight generation.
///
/// `queue` bounds total requests in the system (waiting + running); when it is
/// full, the gateway returns 503. `inflight` (capacity 1) serializes the actual
/// generation, because only one copy of the model fits in memory.
#[derive(Clone)]
pub struct Pool {
    queue: Arc<Semaphore>,
    inflight: Arc<Semaphore>,
    max_queue: usize,
}

impl Pool {
    pub fn new(max_queue: usize) -> Self {
        Self {
            queue: Arc::new(Semaphore::new(max_queue)),
            inflight: Arc::new(Semaphore::new(1)),
            max_queue,
        }
    }

    /// Admit a request, or None if the queue is full (-> 503). Holding the
    /// returned permit keeps a queue slot occupied; drop it when done.
    pub fn try_admit(&self) -> Option<OwnedSemaphorePermit> {
        let permit = self.queue.clone().try_acquire_owned().ok()?;
        metrics::gauge!("qwen_queue_depth").set(self.queue_depth() as f64);
        Some(permit)
    }

    /// Await the single in-flight slot. Drop the permit to release it.
    pub async fn acquire_inflight(&self) -> OwnedSemaphorePermit {
        let permit = self.inflight.clone().acquire_owned().await.expect("semaphore");
        metrics::gauge!("qwen_in_flight").set(1.0);
        permit
    }

    pub fn release_inflight(&self) {
        metrics::gauge!("qwen_in_flight").set(0.0);
    }

    pub fn refresh_queue_gauge(&self) {
        metrics::gauge!("qwen_queue_depth").set(self.queue_depth() as f64);
    }

    fn queue_depth(&self) -> usize {
        self.max_queue - self.queue.available_permits()
    }
}
