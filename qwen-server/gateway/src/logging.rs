use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Classical leveled logging. `RUST_LOG` controls the filter
/// (e.g. `info`, `qwen_gateway=debug,tower_http=warn`). Defaults to `info`.
pub fn init() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    // Pretty to console; switch to JSON via LOG_FORMAT=json for Loki/Promtail.
    let json = std::env::var("LOG_FORMAT").as_deref() == Ok("json");
    let registry = tracing_subscriber::registry().with(filter);
    if json {
        registry.with(fmt::layer().json().flatten_event(true)).init();
    } else {
        registry.with(fmt::layer().with_target(true)).init();
    }
}
