use std::sync::{Arc, Mutex};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Bounded request admission + bounded in-flight generations.
///
/// `queue` bounds total requests in the system (waiting + running); when it is
/// full, the gateway returns 503. `inflight` bounds the generations sent to the
/// worker at once: the worker's `max_batch` (Health), 1 for a worker that
/// serializes, or `GATEWAY_MAX_INFLIGHT`.
///
/// The gauges follow the permits: `qwen_queue_depth` / `qwen_in_flight` go up when
/// a permit is taken and down when it is dropped, on every path (finish, error,
/// client disconnect while waiting or generating).
///
/// The in-flight capacity can change at runtime (`resize`, from the Health poll).
/// Growing adds permits; shrinking forgets the free ones and, for slots still held,
/// records a debt that the next dropped slots pay by forgetting their permit.
#[derive(Clone)]
pub struct Pool {
    queue: Arc<Semaphore>,
    inflight: Arc<Semaphore>,
    cap: Arc<Mutex<Cap>>,
}

struct Cap {
    cap: usize,
    owed: usize, // permits still to forget (capacity shrank while they were held)
}

/// A held slot; dropping it frees the slot and decrements its gauge.
pub struct Slot {
    permit: Option<OwnedSemaphorePermit>,
    gauge: &'static str,
    cap: Option<Arc<Mutex<Cap>>>, // in-flight slots: pay a shrink debt on drop
}

impl Drop for Slot {
    fn drop(&mut self) {
        metrics::gauge!(self.gauge).decrement(1.0);
        if let Some(cap) = &self.cap {
            let mut c = cap.lock().unwrap_or_else(|e| e.into_inner());
            if c.owed > 0 {
                c.owed -= 1;
                if let Some(p) = self.permit.take() {
                    p.forget();
                }
            }
        }
    }
}

/// `GATEWAY_MAX_INFLIGHT` if it is a positive integer (a fixed in-flight capacity);
/// None = follow the worker's `max_batch`.
pub fn fixed_capacity(env: Option<&str>) -> Option<usize> {
    let s = env.map(str::trim).filter(|s| !s.is_empty())?;
    match s.parse::<usize>() {
        Ok(n) if n >= 1 => Some(n),
        _ => {
            tracing::warn!(value = s, "GATEWAY_MAX_INFLIGHT is not a positive integer; \
                           using the worker's max_batch");
            None
        }
    }
}

/// In-flight capacity that follows the worker: its `max_batch` (0 from a worker that
/// predates the field = 1).
pub fn worker_capacity(max_batch: u32) -> usize {
    (max_batch as usize).max(1)
}

impl Pool {
    pub fn new(max_queue: usize, cap: usize) -> Self {
        let cap = cap.max(1);
        metrics::gauge!("qwen_queue_depth").set(0.0);
        metrics::gauge!("qwen_in_flight").set(0.0);
        metrics::gauge!("qwen_inflight_capacity").set(cap as f64);
        Self {
            queue: Arc::new(Semaphore::new(max_queue)),
            inflight: Arc::new(Semaphore::new(cap)),
            cap: Arc::new(Mutex::new(Cap { cap, owed: 0 })),
        }
    }

    pub fn capacity(&self) -> usize {
        self.cap.lock().unwrap_or_else(|e| e.into_inner()).cap
    }

    /// Set the in-flight capacity (min 1). Running generations keep their slots; with a
    /// smaller capacity no new one starts until the running count is below it.
    pub fn resize(&self, new: usize) {
        let new = new.max(1);
        let mut c = self.cap.lock().unwrap_or_else(|e| e.into_inner());
        if new > c.cap {
            let grow = new - c.cap;
            let paid = grow.min(c.owed);
            c.owed -= paid;
            self.inflight.add_permits(grow - paid);
        } else if new < c.cap {
            let shrink = c.cap - new;
            let forgot = self.inflight.forget_permits(shrink);
            c.owed += shrink - forgot;
        }
        c.cap = new;
        metrics::gauge!("qwen_inflight_capacity").set(new as f64);
    }

    /// Admit a request, or None if the queue is full (-> 503). Holding the
    /// returned slot keeps a queue place occupied.
    pub fn try_admit(&self) -> Option<Slot> {
        let permit = self.queue.clone().try_acquire_owned().ok()?;
        metrics::gauge!("qwen_queue_depth").increment(1.0);
        Some(Slot { permit: Some(permit), gauge: "qwen_queue_depth", cap: None })
    }

    /// Await an in-flight slot (one of `capacity`).
    pub async fn acquire_inflight(&self) -> Slot {
        let permit = self.inflight.clone().acquire_owned().await.expect("semaphore");
        metrics::gauge!("qwen_in_flight").increment(1.0);
        Slot { permit: Some(permit), gauge: "qwen_in_flight", cap: Some(self.cap.clone()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrics_exporter_prometheus::PrometheusBuilder;

    #[test]
    fn capacity_from_env_or_worker() {
        assert_eq!(worker_capacity(0), 1);   // old worker: no max_batch
        assert_eq!(worker_capacity(1), 1);
        assert_eq!(worker_capacity(8), 8);
        assert_eq!(fixed_capacity(None), None);
        assert_eq!(fixed_capacity(Some("3")), Some(3));
        assert_eq!(fixed_capacity(Some(" 12 ")), Some(12));
        assert_eq!(fixed_capacity(Some("")), None);
        assert_eq!(fixed_capacity(Some("0")), None);
        assert_eq!(fixed_capacity(Some("-2")), None);
        assert_eq!(fixed_capacity(Some("x")), None);
    }

    impl Pool {
        // held in-flight slots = capacity + debt - free permits
        fn in_flight(&self) -> usize {
            let c = self.cap.lock().unwrap();
            c.cap + c.owed - self.inflight.available_permits()
        }
        fn queue_depth(&self) -> usize { 3 - self.queue.available_permits() }
    }

    fn gauge(text: &str, name: &str) -> f64 {
        text.lines()
            .find_map(|l| l.strip_prefix(name)?.strip_prefix(' ')?.trim().parse().ok())
            .unwrap_or_else(|| panic!("{name} not in\n{text}"))
    }

    #[test]
    fn permits_and_gauges() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
            rt.block_on(async {
                let pool = Pool::new(3, 2);   // in_flight / queue_depth above assume these
                let r = handle.render();
                assert_eq!(gauge(&r, "qwen_inflight_capacity"), 2.0);
                assert_eq!(gauge(&r, "qwen_in_flight"), 0.0);

                let a = pool.try_admit().unwrap();
                let b = pool.try_admit().unwrap();
                let c = pool.try_admit().unwrap();
                assert!(pool.try_admit().is_none(), "4th request must be a 503");
                assert_eq!(gauge(&handle.render(), "qwen_queue_depth"), 3.0);

                let ia = pool.acquire_inflight().await;
                let ib = pool.acquire_inflight().await;
                assert_eq!(pool.in_flight(), 2);
                assert_eq!(gauge(&handle.render(), "qwen_in_flight"), 2.0);
                // the third waits for a slot
                let p2 = pool.clone();
                let mut third = Box::pin(async move { p2.acquire_inflight().await });
                assert!(futures::poll!(&mut third).is_pending());

                drop(ia);
                drop(a);
                let ic = third.await;
                assert_eq!(pool.in_flight(), 2);
                assert_eq!(pool.queue_depth(), 2);
                let r = handle.render();
                assert_eq!(gauge(&r, "qwen_in_flight"), 2.0);
                assert_eq!(gauge(&r, "qwen_queue_depth"), 2.0);

                // a waiter dropped before it got a slot (client gone) leaves no trace
                let d = pool.try_admit().unwrap();
                let p3 = pool.clone();
                let mut waiting = Box::pin(async move { p3.acquire_inflight().await });
                assert!(futures::poll!(&mut waiting).is_pending());
                drop(waiting);
                drop(d);

                drop((ib, ic, b, c));
                assert_eq!(pool.in_flight(), 0);
                assert_eq!(pool.queue_depth(), 0);
                let r = handle.render();
                assert_eq!(gauge(&r, "qwen_in_flight"), 0.0);
                assert_eq!(gauge(&r, "qwen_queue_depth"), 0.0);
            });
        });
    }

    #[test]
    fn resize_with_held_slots() {
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let pool = Pool::new(100, 4);
            let mut held = vec![];
            for _ in 0..4 {
                held.push(pool.acquire_inflight().await);
            }
            // shrink to 2 while 4 run: nothing new starts until 2 have finished
            pool.resize(2);
            assert_eq!(pool.capacity(), 2);
            assert_eq!(pool.in_flight(), 4);
            let p = pool.clone();
            let mut next = Box::pin(async move { p.acquire_inflight().await });
            drop(held.pop());
            assert!(futures::poll!(&mut next).is_pending());
            drop(held.pop());
            assert!(futures::poll!(&mut next).is_pending());   // 2 running = capacity
            drop(held.pop());
            let n = next.await;                                 // 1 running -> 2nd slot
            assert_eq!(pool.in_flight(), 2);
            assert!(pool.inflight.try_acquire().is_err());
            // grow back to 4: two more start at once
            pool.resize(4);
            let a = pool.acquire_inflight().await;
            let b = pool.acquire_inflight().await;
            assert_eq!(pool.in_flight(), 4);
            assert!(pool.inflight.try_acquire().is_err());
            drop((a, b, n, held));
            assert_eq!(pool.in_flight(), 0);
            assert_eq!(pool.inflight.available_permits(), 4);
            // grow while a debt is outstanding pays the debt first
            let mut held: Vec<_> = Vec::new();
            for _ in 0..4 {
                held.push(pool.acquire_inflight().await);
            }
            pool.resize(1);                                     // debt 3
            pool.resize(3);                                     // debt 1, no new permits
            assert!(pool.inflight.try_acquire().is_err());
            held.clear();
            assert_eq!(pool.inflight.available_permits(), 3);
            assert_eq!(pool.in_flight(), 0);
        });
    }
}
