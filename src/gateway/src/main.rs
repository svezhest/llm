mod api;
mod journal;
mod logging;
mod metrics;
mod pb;
mod pool;
mod request;
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
    /// Model id the worker reported at connect; a fallback only — responses carry
    /// the id from each generation's Start event, /v1/models asks Health.
    pub worker_model: String,
    pub journal: Option<journal::Journal>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    logging::init();
    let metrics_handle = metrics::install();

    let worker_addr = env::var("WORKER_ADDR").unwrap_or_else(|_| "127.0.0.1:50051".into());
    let listen = env::var("GATEWAY_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".into());
    let max_queue: usize = env::var("MAX_QUEUE").ok().and_then(|s| s.parse().ok()).unwrap_or(32);

    info!(%worker_addr, %listen, max_queue, "starting gateway");
    let (client, worker_model) = worker_client::connect(worker_addr).await?;

    let state = AppState {
        pool: pool::Pool::new(max_queue),
        client,
        metrics: metrics_handle,
        worker_model,
        journal: journal::Journal::from_env()?,
    };
    if state.journal.is_some() {
        info!(path = %env::var("GATEWAY_JOURNAL").unwrap_or_default(), "request journal on");
    }

    let metrics_state = state.clone();
    let app = Router::new()
        .route("/v1/chat/completions", post(api::chat_completions))
        .route("/v1/models", get(api::list_models))
        .route("/health", get(api::health))
        .route("/metrics", get(move || {
            let h = metrics_state.metrics.clone();
            async move { h.render() }
        }))
        // Vision payloads: a base64 PNG snapshot easily tops axum's 2MB default.
        .layer(axum::extract::DefaultBodyLimit::max(32 * 1024 * 1024))
        .with_state(state);

    let l = tokio::net::TcpListener::bind(&listen).await?;
    info!("listening on {listen}");
    axum::serve(l, app).await?;
    Ok(())
}
