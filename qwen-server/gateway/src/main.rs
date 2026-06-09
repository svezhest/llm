mod api;
mod logging;
mod metrics;
mod pb;
mod pool;
mod worker_client;

use std::env;

use axum::{routing::get, routing::post, Router};
use tracing::info;

use worker_client::Client;

#[derive(Clone)]
pub struct AppState {
    pub pool: pool::Pool,
    pub client: Client,
    pub metrics: metrics_exporter_prometheus::PrometheusHandle,
    pub model_name: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    logging::init();
    let metrics_handle = metrics::install();

    let worker_addr = env::var("WORKER_ADDR").unwrap_or_else(|_| "127.0.0.1:50051".into());
    let listen = env::var("GATEWAY_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".into());
    let max_queue: usize = env::var("MAX_QUEUE").ok().and_then(|s| s.parse().ok()).unwrap_or(32);

    info!(%worker_addr, %listen, max_queue, "starting gateway");
    let client = worker_client::connect(worker_addr).await?;

    let state = AppState {
        pool: pool::Pool::new(max_queue),
        client,
        metrics: metrics_handle,
        model_name: env::var("MODEL_NAME").unwrap_or_else(|_| "qwen3.6-35b-a3b".into()),
    };

    let metrics_state = state.clone();
    let app = Router::new()
        .route("/v1/chat/completions", post(api::chat_completions))
        .route("/v1/models", get(api::list_models))
        .route("/health", get(api::health))
        .route("/metrics", get(move || {
            let h = metrics_state.metrics.clone();
            async move { h.render() }
        }))
        .with_state(state);

    let l = tokio::net::TcpListener::bind(&listen).await?;
    info!("listening on {listen}");
    axum::serve(l, app).await?;
    Ok(())
}
