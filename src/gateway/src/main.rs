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
use tracing::{info, warn};

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
    let max_queue: usize = match env::var("MAX_QUEUE") {
        Err(_) => 32,
        Ok(s) => match s.trim().parse() {
            Ok(n) if n >= 1 => n,
            _ => {
                warn!(value = %s, "MAX_QUEUE is not a positive integer; using 32");
                32
            }
        },
    };

    info!(%worker_addr, %listen, max_queue, "starting gateway");
    let (client, worker_model, max_batch) = worker_client::connect(worker_addr).await?;
    // Generations sent to the worker at once: GATEWAY_MAX_INFLIGHT, else the worker's
    // max_batch (followed at runtime by the Health poll).
    let fixed = pool::fixed_capacity(env::var("GATEWAY_MAX_INFLIGHT").ok().as_deref());
    let cap = fixed.unwrap_or_else(|| pool::worker_capacity(max_batch));
    info!(capacity = cap, worker_max_batch = max_batch, fixed = fixed.is_some(), "in-flight capacity");
    if fixed.is_some_and(|f| f > pool::worker_capacity(max_batch)) {
        warn!(capacity = cap, worker_max_batch = max_batch,
              "GATEWAY_MAX_INFLIGHT > the worker's max_batch: the extra generations wait inside \
               the worker holding its gRPC threads, or are refused (503)");
    }
    if max_queue < cap {
        warn!(max_queue, capacity = cap, "MAX_QUEUE < in-flight capacity: the queue bounds concurrency");
    }

    let pool = pool::Pool::new(max_queue, cap);
    worker_client::spawn_health_poll(client.clone(), pool.clone(), fixed, max_batch);
    let state = AppState {
        pool,
        client,
        metrics: metrics_handle,
        worker_model,
        journal: journal::Journal::from_env()?,
    };
    if state.journal.is_some() {
        info!(path = %env::var("GATEWAY_JOURNAL").unwrap_or_default(), "request journal on");
    }

    let journal = state.journal.clone();
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
    // On SIGINT/SIGTERM exit at once, as before (open streams are cut, no destructors
    // run), but first write the journal lines still queued.
    tokio::select! {
        r = axum::serve(l, app) => r?,
        code = shutdown_signal() => {
            info!("signal received, exiting");
            if let Some(j) = journal {
                let _ = tokio::task::spawn_blocking(move || j.flush(std::time::Duration::from_secs(5)))
                    .await;
            }
            std::process::exit(code);
        }
    }
    Ok(())
}

/// Wait for SIGINT / SIGTERM; returns the shell's exit status for it (128 + signo).
async fn shutdown_signal() -> i32 {
    use tokio::signal::unix::{signal, SignalKind};
    let mut term = signal(SignalKind::terminate()).expect("SIGTERM handler");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => 130,
        _ = term.recv() => 143,
    }
}
