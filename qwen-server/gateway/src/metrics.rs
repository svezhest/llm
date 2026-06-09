use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

/// Install the Prometheus recorder and return a handle whose `.render()`
/// produces the text exposition format served at GET /metrics.
pub fn install() -> PrometheusHandle {
    let handle = PrometheusBuilder::new()
        .install_recorder()
        .expect("install prometheus recorder");

    // Describe metrics up-front so they appear before first use.
    metrics::describe_counter!("qwen_requests_total", "Chat completion requests by status");
    metrics::describe_counter!("qwen_errors_total", "Worker/gateway errors by kind");
    metrics::describe_counter!("qwen_cancellations_total", "Client-disconnect cancellations");
    metrics::describe_counter!("qwen_tokens_generated_total", "Tokens streamed to clients");
    metrics::describe_histogram!("qwen_request_duration_seconds", "End-to-end request duration");
    metrics::describe_histogram!("qwen_ttft_seconds", "Time to first token");
    metrics::describe_gauge!("qwen_queue_depth", "Requests waiting + in flight");
    metrics::describe_gauge!("qwen_in_flight", "Generations currently running");
    metrics::describe_gauge!("qwen_worker_up", "1 if the worker Health is ready");
    handle
}
