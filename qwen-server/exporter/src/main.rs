//! Prometheus exporter for Apple Silicon GPU/ANE/power via `powermetrics`.
//!
//! `powermetrics` needs root, so run this exporter under sudo (or a launchd
//! service loaded as root). Each scrape takes one ~200ms sample.
//!
//!   sudo MAC_EXPORTER_ADDR=0.0.0.0:9101 qwen-mac-exporter
//!
//! Exposes: mac_gpu_busy_ratio, mac_gpu_freq_mhz, mac_ane_power_mw,
//!          mac_package_power_mw, mac_gpu_power_mw.

use std::process::Command;

use axum::{routing::get, Router};
use tracing::{info, warn};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().with_env_filter(
        tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
    ).init();

    let addr = std::env::var("MAC_EXPORTER_ADDR").unwrap_or_else(|_| "0.0.0.0:9101".into());
    let app = Router::new().route("/metrics", get(metrics)).route("/health", get(|| async { "ok" }));
    let l = tokio::net::TcpListener::bind(&addr).await.expect("bind");
    info!("mac-exporter on {addr}");
    axum::serve(l, app).await.unwrap();
}

async fn metrics() -> String {
    match sample() {
        Ok(s) => render(&s),
        Err(e) => {
            warn!(error = %e, "powermetrics sample failed (need root?)");
            "# powermetrics unavailable (run as root)\n".to_string()
        }
    }
}

#[derive(Default)]
struct Sample {
    gpu_busy: f64,    // ratio 0..1
    gpu_freq: f64,    // MHz
    ane_mw: f64,
    gpu_mw: f64,
    pkg_mw: f64,
}

fn sample() -> Result<Sample, String> {
    // One 200ms sample of GPU + ANE counters.
    let out = Command::new("powermetrics")
        .args(["-n", "1", "-i", "200", "--samplers", "gpu_power"])
        .output()
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!("exit {:?}", out.status.code()));
    }
    Ok(parse(&String::from_utf8_lossy(&out.stdout)))
}

fn parse(text: &str) -> Sample {
    let mut s = Sample::default();
    for line in text.lines() {
        let l = line.trim();
        if let Some(v) = after(l, "GPU HW active residency:") {
            s.gpu_busy = pct(&v);
        } else if let Some(v) = after(l, "GPU HW active frequency:") {
            s.gpu_freq = num(&v);
        } else if let Some(v) = after(l, "ANE Power:") {
            s.ane_mw = num(&v);
        } else if let Some(v) = after(l, "GPU Power:") {
            s.gpu_mw = num(&v);
        } else if let Some(v) = after(l, "Combined Power (CPU + GPU + ANE):") {
            s.pkg_mw = num(&v);
        }
    }
    s
}

fn after(line: &str, key: &str) -> Option<String> {
    line.strip_prefix(key).map(|r| r.trim().to_string())
}

/// First float in a string (e.g. "1234.5 mW" -> 1234.5, "37.21%" -> 37.21).
fn num(s: &str) -> f64 {
    s.chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
        .collect::<String>()
        .parse()
        .unwrap_or(0.0)
}

fn pct(s: &str) -> f64 {
    num(s) / 100.0
}

fn render(s: &Sample) -> String {
    format!(
        "# HELP mac_gpu_busy_ratio GPU active residency (0..1)\n\
         # TYPE mac_gpu_busy_ratio gauge\nmac_gpu_busy_ratio {}\n\
         # TYPE mac_gpu_freq_mhz gauge\nmac_gpu_freq_mhz {}\n\
         # TYPE mac_ane_power_mw gauge\nmac_ane_power_mw {}\n\
         # TYPE mac_gpu_power_mw gauge\nmac_gpu_power_mw {}\n\
         # TYPE mac_package_power_mw gauge\nmac_package_power_mw {}\n",
        s.gpu_busy, s.gpu_freq, s.ane_mw, s.gpu_mw, s.pkg_mw
    )
}
