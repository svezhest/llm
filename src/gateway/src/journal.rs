//! Optional request/response journal (JSONL). Enabled by `GATEWAY_JOURNAL=<path>`.
//!
//! One line per request: `{"ts", "id", "request", "response"}`. `request` is the body
//! exactly as the client sent it (only raw CR/LF bytes are dropped: in valid JSON they can
//! only be insignificant whitespace, so the line stays one line and the JSON stays the same).
//! `response` is the chat.completion object; for streamed requests it is assembled from
//! the chunks and carries `"stream": true`. Errors are logged as `{"error": ...}`.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;
use tracing::warn;

#[derive(Clone)]
pub struct Journal(Arc<Mutex<File>>);

impl Journal {
    pub fn from_env() -> anyhow::Result<Option<Journal>> {
        match std::env::var("GATEWAY_JOURNAL") {
            Ok(path) if !path.is_empty() => {
                let f = OpenOptions::new().create(true).append(true).open(&path)?;
                Ok(Some(Journal(Arc::new(Mutex::new(f)))))
            }
            _ => Ok(None),
        }
    }

    pub fn write(&self, id: &str, body: &[u8], response: &Value) {
        let request: String = if serde_json::from_slice::<Value>(body).is_ok() {
            let bytes: Vec<u8> = body.iter().copied().filter(|&b| b != b'\n' && b != b'\r').collect();
            String::from_utf8(bytes).unwrap_or_default()
        } else {
            // not JSON: keep it as a string
            Value::String(String::from_utf8_lossy(body).into_owned()).to_string()
        };
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let line = format!(
            "{{\"ts\":{ts:.3},\"id\":{},\"request\":{request},\"response\":{response}}}\n",
            Value::String(id.to_string())
        );
        let mut f = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Err(e) = f.write_all(line.as_bytes()) {
            warn!(error = %e, "journal write failed");
        }
    }
}
