use metrics_exporter_prometheus::{BuildError, Matcher, PrometheusBuilder, PrometheusHandle};

/// Seconds: TTFT and queue waits (tens of ms) up to a long-context generation (minutes).
const SECONDS: &[f64] = &[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0,
                          80.0, 160.0, 320.0, 640.0];
/// Mean decode batch size of a request (1 = alone).
const BATCH: &[f64] = &[1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 16.0, 32.0];

/// Every histogram here has buckets, so it renders as a Prometheus histogram
/// (`_bucket`/`_sum`/`_count`, for histogram_quantile) — the exporter renders a metric
/// either as buckets or as a summary, never both.
fn builder() -> Result<PrometheusBuilder, BuildError> {
    PrometheusBuilder::new()
        .set_buckets_for_metric(Matcher::Suffix("_seconds".into()), SECONDS)?
        .set_buckets_for_metric(Matcher::Full("qwen_batch_mean".into()), BATCH)
}

/// Install the Prometheus recorder and return a handle whose `.render()`
/// produces the text exposition format served at GET /metrics.
pub fn install() -> PrometheusHandle {
    let handle = builder()
        .expect("histogram buckets")
        .install_recorder()
        .expect("install prometheus recorder");

    // Describe metrics up-front so they appear before first use.
    metrics::describe_counter!("qwen_requests_total", "Chat completion requests by status");
    metrics::describe_counter!("qwen_errors_total", "Worker/gateway errors by kind");
    metrics::describe_counter!("qwen_cancellations_total", "Client-disconnect cancellations");
    metrics::describe_counter!("qwen_journal_dropped_total", "Journal lines dropped (writer queue full)");
    metrics::describe_counter!("qwen_tokens_generated_total", "Tokens streamed to clients");
    metrics::describe_histogram!("qwen_request_duration_seconds",
                                 "Request duration from arrival at the gateway (queue waits included)");
    metrics::describe_histogram!("qwen_ttft_seconds",
                                 "Time to first token from arrival at the gateway (streaming)");
    metrics::describe_histogram!("qwen_gateway_queue_seconds", "Wait for an in-flight slot in the gateway");
    metrics::describe_gauge!("qwen_queue_depth", "Requests waiting + in flight");
    metrics::describe_gauge!("qwen_in_flight", "Generations sent to the worker (count, up to qwen_inflight_capacity)");
    metrics::describe_gauge!("qwen_inflight_capacity", "In-flight limit: the worker's max_batch or GATEWAY_MAX_INFLIGHT");
    metrics::describe_histogram!("qwen_worker_queue_seconds", "Worker-reported wait from arrival to prefill start");
    metrics::describe_histogram!("qwen_batch_mean", "Worker-reported mean decode batch size over a request");
    metrics::describe_gauge!("qwen_worker_up", "1 if the worker Health is ready");
    handle
}

#[cfg(test)]
mod tests {
    #[test]
    fn histograms_have_buckets() {
        let recorder = super::builder().unwrap().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            metrics::histogram!("qwen_ttft_seconds").record(0.3);
            metrics::histogram!("qwen_batch_mean").record(2.5);
        });
        let text = handle.render();
        assert!(text.contains("qwen_ttft_seconds_bucket{le=\"0.5\"} 1"), "{text}");
        assert!(text.contains("qwen_ttft_seconds_bucket{le=\"0.25\"} 0"), "{text}");
        assert!(text.contains("qwen_batch_mean_bucket{le=\"3\"} 1"), "{text}");
        assert!(!text.contains("quantile="), "{text}");
    }
}
