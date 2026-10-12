use std::time::Duration;

use anyhow::Result;
use tonic::transport::Channel;
use tracing::{info, warn};

use crate::pb::worker_client::WorkerClient;
use crate::pb::Empty;
use crate::pool::{self, Pool};

pub type Client = WorkerClient<Channel>;

/// Connect to the worker with backoff, then block until Health reports ready.
/// Returns the client, the id of the model the worker has loaded and the worker's
/// `max_batch` (generations it decodes concurrently; 0 from an older worker).
pub async fn connect(addr: String) -> Result<(Client, String, u32)> {
    let endpoint = format!("http://{addr}");
    let mut backoff = Duration::from_millis(500);
    loop {
        match WorkerClient::connect(endpoint.clone()).await {
            Ok(mut client) => match client.health(Empty {}).await {
                Ok(reply) => {
                    let r = reply.into_inner();
                    if r.ready {
                        info!(model = %r.model, path = %r.model_path, load_ms = r.load_ms,
                              max_batch = r.max_batch, "worker ready");
                        metrics::gauge!("qwen_worker_up").set(1.0);
                        return Ok((client, r.model, r.max_batch));
                    }
                    warn!("worker connected but not ready yet; retrying");
                }
                Err(e) => warn!(error = %e, "Health call failed; retrying"),
            },
            Err(e) => warn!(error = %e, "worker connect failed; retrying"),
        }
        metrics::gauge!("qwen_worker_up").set(0.0);
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(Duration::from_secs(10));
    }
}

const POLL: Duration = Duration::from_secs(15);
const POLL_TIMEOUT: Duration = Duration::from_secs(5);

/// Every 15 s ask the worker's Health: keep `qwen_worker_up` current and, unless the
/// capacity is fixed (GATEWAY_MAX_INFLIGHT), follow a changed `max_batch` (a worker
/// restarted with another batch size or model).
pub fn spawn_health_poll(mut client: Client, pool: Pool, fixed: Option<usize>, mut max_batch: u32) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(POLL);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        tick.tick().await; // the first tick is immediate; connect() just asked
        let mut up = true;
        loop {
            tick.tick().await;
            let mut req = tonic::Request::new(Empty {});
            req.set_timeout(POLL_TIMEOUT);
            let reply = match tokio::time::timeout(POLL_TIMEOUT, client.health(req)).await {
                Ok(Ok(r)) => r.into_inner(),
                Ok(Err(e)) => {
                    if up {
                        warn!(error = %e, "worker Health failed");
                    }
                    up = false;
                    metrics::gauge!("qwen_worker_up").set(0.0);
                    continue;
                }
                Err(_) => {
                    // alive but no free gRPC thread (or stuck): leave qwen_worker_up as is
                    warn!(timeout_s = POLL_TIMEOUT.as_secs(), "worker Health timed out");
                    continue;
                }
            };
            if reply.ready != up {
                info!(ready = reply.ready, model = %reply.model, "worker Health changed");
            }
            up = reply.ready;
            metrics::gauge!("qwen_worker_up").set(if up { 1.0 } else { 0.0 });
            if !reply.ready || reply.max_batch == max_batch {
                continue;
            }
            let new = pool::worker_capacity(reply.max_batch);
            match fixed {
                None => {
                    warn!(old = max_batch, new = reply.max_batch, capacity = new,
                          "worker max_batch changed; in-flight capacity follows");
                    pool.resize(new);
                }
                Some(cap) if cap > new => {
                    warn!(worker_max_batch = reply.max_batch, capacity = cap,
                          "worker max_batch changed and is now below GATEWAY_MAX_INFLIGHT");
                }
                Some(_) => info!(old = max_batch, new = reply.max_batch, "worker max_batch changed"),
            }
            max_batch = reply.max_batch;
        }
    });
}
