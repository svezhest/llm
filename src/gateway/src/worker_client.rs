use std::time::Duration;

use anyhow::Result;
use tonic::transport::Channel;
use tracing::{info, warn};

use crate::pb::worker_client::WorkerClient;
use crate::pb::Empty;

pub type Client = WorkerClient<Channel>;

/// Connect to the worker with backoff, then block until Health reports ready.
/// Returns the client and the id of the model the worker has loaded.
pub async fn connect(addr: String) -> Result<(Client, String)> {
    let endpoint = format!("http://{addr}");
    let mut backoff = Duration::from_millis(500);
    loop {
        match WorkerClient::connect(endpoint.clone()).await {
            Ok(mut client) => match client.health(Empty {}).await {
                Ok(reply) => {
                    let r = reply.into_inner();
                    if r.ready {
                        info!(model = %r.model, path = %r.model_path, load_ms = r.load_ms, "worker ready");
                        metrics::gauge!("qwen_worker_up").set(1.0);
                        return Ok((client, r.model));
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
