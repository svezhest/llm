//! Optional request/response journal (JSONL). Enabled by `GATEWAY_JOURNAL=<path>`.
//!
//! One line per request: `{"ts", "id", "request", "response"}`. `request` is the body
//! exactly as the client sent it (only raw CR/LF bytes are dropped: in valid JSON they can
//! only be insignificant whitespace, so the line stays one line and the JSON stays the same).
//! `response` is the chat.completion object; for streamed requests it is assembled from
//! the chunks and carries `"stream": true`. Errors are logged as `{"error": ...}`.
//!
//! Lines are formatted and written by one writer thread fed through a channel, so the
//! request path never blocks on the file; the channel keeps the order of `write` calls.
//! The channel holds at most 10k lines: past that (a stuck disk) lines are dropped and
//! counted in `qwen_journal_dropped_total`, never queued without bound.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::body::Bytes;
use serde_json::Value;
use tracing::warn;

enum Msg {
    Line { ts: f64, id: String, body: Bytes, response: Value },
    /// Reply once every line sent before it is written.
    Flush(mpsc::Sender<()>),
}

const CAPACITY: usize = 10_000;

#[derive(Clone)]
pub struct Journal {
    tx: mpsc::SyncSender<Msg>,
    dropped: Arc<AtomicU64>,
}

impl Journal {
    pub fn from_env() -> anyhow::Result<Option<Journal>> {
        match std::env::var("GATEWAY_JOURNAL") {
            Ok(path) if !path.is_empty() => {
                let f = OpenOptions::new().create(true).append(true).open(&path)?;
                Ok(Some(Journal::start(f)))
            }
            _ => Ok(None),
        }
    }

    fn start(mut f: File) -> Journal {
        let (tx, rx) = mpsc::sync_channel::<Msg>(CAPACITY);
        std::thread::Builder::new()
            .name("journal".into())
            .spawn(move || {
                for msg in rx {
                    match msg {
                        Msg::Line { ts, id, body, response } => {
                            // unbuffered File: each line is one write(2), in the kernel once
                            // written, so nothing is lost at exit but lines still queued
                            if let Err(e) = f.write_all(line(ts, &id, &body, &response).as_bytes()) {
                                warn!(error = %e, "journal write failed");
                            }
                        }
                        Msg::Flush(done) => {
                            let _ = done.send(());
                        }
                    }
                }
            })
            .expect("spawn journal thread");
        Journal { tx, dropped: Arc::new(AtomicU64::new(0)) }
    }

    /// Queue a line; never blocks (a full channel drops it).
    pub fn write(&self, id: &str, body: &Bytes, response: Value) {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let msg = Msg::Line { ts, id: id.to_string(), body: body.clone(), response };
        if let Err(e) = self.tx.try_send(msg) {
            metrics::counter!("qwen_journal_dropped_total").increment(1);
            let n = self.dropped.fetch_add(1, Ordering::Relaxed) + 1;
            if n == 1 || n % 1000 == 0 {   // don't flood the log while the disk is stuck
                let why = if matches!(e, mpsc::TrySendError::Full(_)) { "full" } else { "writer gone" };
                warn!(id, dropped = n, why, "journal line dropped");
            }
        }
    }

    /// Block until every line queued so far is written (best effort, at shutdown).
    pub fn flush(&self, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        let (tx, rx) = mpsc::channel();
        let mut msg = Msg::Flush(tx);
        loop {
            match self.tx.try_send(msg) {
                Ok(()) => break,
                Err(mpsc::TrySendError::Full(m)) if Instant::now() < deadline => {
                    msg = m;
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(_) => {
                    warn!("journal flush: writer stuck or gone");
                    return;
                }
            }
        }
        if rx.recv_timeout(deadline.saturating_duration_since(Instant::now())).is_err() {
            warn!("journal flush timed out");
        }
    }
}

fn line(ts: f64, id: &str, body: &[u8], response: &Value) -> String {
    let request: String = if serde_json::from_slice::<Value>(body).is_ok() {
        let bytes: Vec<u8> = body.iter().copied().filter(|&b| b != b'\n' && b != b'\r').collect();
        String::from_utf8(bytes).unwrap_or_default()
    } else {
        // not JSON: keep it as a string
        Value::String(String::from_utf8_lossy(body).into_owned()).to_string()
    };
    format!(
        "{{\"ts\":{ts:.3},\"id\":{},\"request\":{request},\"response\":{response}}}\n",
        Value::String(id.to_string())
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn lines_in_order_after_flush() {
        let path = std::env::temp_dir().join(format!("journal-test-{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let f = OpenOptions::new().create(true).append(true).open(&path).unwrap();
        let j = Journal::start(f);
        for i in 0..100 {
            j.write(&format!("id{i}"), &Bytes::from("{\"a\":\n1}"), json!({"n": i}));
        }
        j.write("x", &Bytes::from("not json"), json!({}));
        j.flush(Duration::from_secs(5));
        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let lines: Vec<Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(lines.len(), 101);
        for (i, l) in lines[..100].iter().enumerate() {
            assert_eq!(l["id"], json!(format!("id{i}")));
            assert_eq!(l["request"], json!({"a": 1}));
            assert_eq!(l["response"], json!({"n": i}));
        }
        assert_eq!(lines[100]["request"], json!("not json"));
    }

    #[test]
    fn full_channel_drops_instead_of_blocking() {
        // a writer that never runs: the receiver is never drained
        let (tx, _rx) = mpsc::sync_channel::<Msg>(2);
        let j = Journal { tx, dropped: Arc::new(AtomicU64::new(0)) };
        for i in 0..5 {
            j.write(&format!("id{i}"), &Bytes::from("{}"), json!({}));
        }
        assert_eq!(j.dropped.load(Ordering::Relaxed), 3);
        let t = Instant::now();
        j.flush(Duration::from_millis(50));   // gives up, does not hang
        assert!(t.elapsed() < Duration::from_secs(2));
    }
}
