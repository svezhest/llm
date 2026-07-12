use std::convert::Infallible;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::{
    extract::State,
    http::StatusCode,
    response::sse::{Event as SseEvent, Sse},
    response::{IntoResponse, Response},
    Json,
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::{info, warn};
use uuid::Uuid;

use crate::pb;
use crate::AppState;

// ---- OpenAI request shape -------------------------------------------------

#[derive(Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<InMessage>,
    #[serde(default)]
    pub stream: bool,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stop: Vec<String>,
    pub seed: Option<u64>,
    // Extensions (Qwen reasoning controls).
    #[serde(default)]
    pub enable_thinking: bool,
    #[serde(default)]
    pub thinking_budget: u32,
    // Tool / function calling (OpenAI shape).
    pub tools: Option<Value>,
    pub tool_choice: Option<Value>,
}

#[derive(Deserialize)]
pub struct InMessage {
    pub role: String,
    // string OR array of OpenAI content parts. May be null for assistant tool-call turns.
    #[serde(default)]
    pub content: Value,
    // Tool-call conversation fields (passed through for multi-turn tool use).
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Value>,
}

/// Flatten OpenAI content (string or parts array) into (text, image urls).
fn flatten_content(v: &Value) -> (String, Vec<String>) {
    match v {
        Value::String(s) => (s.clone(), vec![]),
        Value::Array(parts) => {
            let mut text = String::new();
            let mut images = vec![];
            for p in parts {
                match p.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(t) = p.get("text").and_then(Value::as_str) {
                            text.push_str(t);
                        }
                    }
                    Some("image_url") => {
                        // Accept both shapes: {image_url:{url:"..."}} and image_url:"...".
                        let url = match p.get("image_url") {
                            Some(Value::String(s)) => Some(s.clone()),
                            Some(o) => o.get("url").and_then(Value::as_str).map(String::from),
                            None => None,
                        };
                        match url {
                            Some(u) => images.push(u),
                            None => warn!(part = %p, "image_url part: no extractable url"),
                        }
                    }
                    other => {
                        let keys: Vec<&str> = p.as_object()
                            .map(|o| o.keys().map(String::as_str).collect())
                            .unwrap_or_default();
                        warn!(part_type = ?other, keys = ?keys, "unparsed content part");
                    }
                }
            }
            (text, images)
        }
        _ => (String::new(), vec![]),
    }
}

fn build_pb_request(id: &str, req: &ChatRequest) -> pb::Request {
    let mut images = vec![];
    let mut messages = vec![];
    for m in &req.messages {
        let (text, imgs) = flatten_content(&m.content);
        images.extend(imgs);
        messages.push(pb::Message {
            role: m.role.clone(),
            content: text,
            name: m.name.clone().unwrap_or_default(),
            tool_call_id: m.tool_call_id.clone().unwrap_or_default(),
            tool_calls_json: m
                .tool_calls
                .as_ref()
                .map(|v| v.to_string())
                .unwrap_or_default(),
        });
    }
    let tools_json = req.tools.as_ref().map(|v| v.to_string()).unwrap_or_default();
    let tool_choice = match &req.tool_choice {
        Some(Value::String(s)) => s.clone(),
        Some(v) => v.to_string(),
        None => String::new(),
    };
    pb::Request {
        id: id.to_string(),
        messages,
        images,
        params: Some(pb::Params {
            max_tokens: req.max_tokens.unwrap_or(1024),
            temperature: req.temperature.unwrap_or(0.7),
            top_p: req.top_p.unwrap_or(1.0),
            enable_thinking: req.enable_thinking,
            thinking_budget: req.thinking_budget,
            seed: req.seed,
            stop: req.stop.clone(),
            tools_json,
            tool_choice,
        }),
    }
}

/// Build the OpenAI `tool_calls` JSON array from worker ToolCalls.
fn tool_calls_json(calls: &[pb::ToolCall]) -> Vec<Value> {
    calls
        .iter()
        .enumerate()
        .map(|(i, c)| {
            json!({
                "index": i,
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments},
            })
        })
        .collect()
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

// ---- handlers -------------------------------------------------------------

pub async fn chat_completions(
    State(st): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Response {
    let admit = match st.pool.try_admit() {
        Some(p) => p,
        None => {
            metrics::counter!("qwen_requests_total", "status" => "rejected").increment(1);
            return (StatusCode::SERVICE_UNAVAILABLE, "queue full").into_response();
        }
    };
    let id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let pbreq = build_pb_request(&id, &req);
    let model = req.model.clone().unwrap_or_else(|| st.model_name.clone());

    if req.stream {
        stream_response(st, admit, id, model, pbreq).await.into_response()
    } else {
        json_response(st, admit, id, model, pbreq).await.into_response()
    }
}

/// Owns the request's queue + in-flight permits and guarantees cleanup on ANY
/// exit path (clean finish OR client disconnect): frees both slots, refreshes
/// the gauges, and — unless disarmed — fires Worker.Cancel.
struct ReqGuard {
    st: AppState,
    id: String,
    armed: bool,
    admit: Option<tokio::sync::OwnedSemaphorePermit>,
    inflight: Option<tokio::sync::OwnedSemaphorePermit>,
}
impl Drop for ReqGuard {
    fn drop(&mut self) {
        // Release the slots first, THEN refresh gauges so depth reflects reality.
        let _ = self.inflight.take();
        let _ = self.admit.take();
        self.st.pool.release_inflight();
        self.st.pool.refresh_queue_gauge();
        if self.armed {
            metrics::counter!("qwen_cancellations_total").increment(1);
            let mut client = self.st.client.clone();
            let id = self.id.clone();
            tokio::spawn(async move {
                let _ = client.cancel(pb::CancelReq { id }).await;
            });
        }
    }
}

async fn stream_response(
    st: AppState,
    admit: tokio::sync::OwnedSemaphorePermit,
    id: String,
    model: String,
    pbreq: pb::Request,
) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
    let body = async_stream::stream! {
        let inflight = st.pool.acquire_inflight().await;
        let mut guard = ReqGuard {
            st: st.clone(), id: id.clone(), armed: true,
            admit: Some(admit), inflight: Some(inflight),
        };
        let start = Instant::now();
        let created = now_unix();
        let mut ttft_done = false;
        let mut first_role = true;

        let mut client = st.client.clone();
        match client.generate(pbreq).await {
            Ok(resp) => {
                let mut stream = resp.into_inner();
                loop {
                    match stream.message().await {
                        Ok(Some(ev)) => {
                            if let Some(sse) = event_to_sse(&id, &model, created, ev,
                                                            &mut ttft_done, &start, &mut first_role) {
                                yield Ok(sse);
                            }
                        }
                        Ok(None) => break,
                        Err(e) => {
                            warn!(error = %e, "worker stream error");
                            metrics::counter!("qwen_errors_total", "kind" => "stream").increment(1);
                            yield Ok(SseEvent::default().data(
                                json!({"error": {"message": e.to_string()}}).to_string()));
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                metrics::counter!("qwen_errors_total", "kind" => "rpc").increment(1);
                yield Ok(SseEvent::default().data(
                    json!({"error": {"message": e.to_string()}}).to_string()));
            }
        }

        metrics::histogram!("qwen_request_duration_seconds").record(start.elapsed().as_secs_f64());
        metrics::counter!("qwen_requests_total", "status" => "ok").increment(1);
        guard.armed = false;   // completed cleanly -> guard drop frees slots, no cancel
        yield Ok(SseEvent::default().data("[DONE]"));
        drop(guard);
    };

    Sse::new(body)
}

/// Map one worker Event to an OpenAI chunk SSE frame (or None to skip).
fn event_to_sse(
    id: &str,
    model: &str,
    created: u64,
    ev: pb::Event,
    ttft_done: &mut bool,
    start: &Instant,
    first_role: &mut bool,
) -> Option<SseEvent> {
    let kind = ev.kind?;
    match kind {
        pb::event::Kind::Start(_) => None,
        pb::event::Kind::Token(t) => {
            if !*ttft_done {
                metrics::histogram!("qwen_ttft_seconds").record(start.elapsed().as_secs_f64());
                *ttft_done = true;
            }
            metrics::counter!("qwen_tokens_generated_total").increment(1);
            let mut delta = serde_json::Map::new();
            if *first_role {
                delta.insert("role".into(), json!("assistant"));
                *first_role = false;
            }
            if t.thinking {
                delta.insert("reasoning_content".into(), json!(t.text));
            } else {
                delta.insert("content".into(), json!(t.text));
            }
            Some(chunk_frame(id, model, created, Value::Object(delta), None))
        }
        pb::event::Kind::Done(d) => {
            info!(id, finish = %d.finish,
                  gen_tokens = d.usage.as_ref().map(|u| u.gen_tokens).unwrap_or(0),
                  tool_calls = d.tool_calls.len(),
                  "done");
            let delta = if d.tool_calls.is_empty() {
                json!({})
            } else {
                json!({ "tool_calls": tool_calls_json(&d.tool_calls) })
            };
            Some(chunk_frame(id, model, created, delta, Some(&d.finish)))
        }
        pb::event::Kind::Error(e) => {
            metrics::counter!("qwen_errors_total", "kind" => e.kind.clone()).increment(1);
            Some(SseEvent::default().data(
                json!({"error": {"message": e.msg, "type": e.kind}}).to_string()))
        }
    }
}

fn chunk_frame(id: &str, model: &str, created: u64, delta: Value, finish: Option<&str>) -> SseEvent {
    let body = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    });
    SseEvent::default().data(body.to_string())
}

async fn json_response(
    st: AppState,
    admit: tokio::sync::OwnedSemaphorePermit,
    id: String,
    model: String,
    pbreq: pb::Request,
) -> Response {
    let inflight = st.pool.acquire_inflight().await;
    let mut guard = ReqGuard {
        st: st.clone(), id: id.clone(), armed: true,
        admit: Some(admit), inflight: Some(inflight),
    };
    let start = Instant::now();
    let created = now_unix();

    let mut content = String::new();
    let mut reasoning = String::new();
    let mut finish = "stop".to_string();
    let mut usage = pb::Usage::default();
    let mut tool_calls: Vec<pb::ToolCall> = vec![];

    let mut client = st.client.clone();
    let result = match client.generate(pbreq).await {
        Ok(resp) => {
            let mut stream = resp.into_inner();
            let mut err: Option<(String, String)> = None;
            while let Ok(Some(ev)) = stream.message().await {
                match ev.kind {
                    Some(pb::event::Kind::Token(t)) => {
                        if t.thinking { reasoning.push_str(&t.text); }
                        else { content.push_str(&t.text); }
                    }
                    Some(pb::event::Kind::Done(d)) => {
                        tool_calls = d.tool_calls;
                        finish = d.finish;
                        if let Some(u) = d.usage { usage = u; }
                    }
                    Some(pb::event::Kind::Error(e)) => { err = Some((e.kind, e.msg)); }
                    _ => {}
                }
            }
            err
        }
        Err(e) => Some(("rpc".into(), e.to_string())),
    };

    metrics::histogram!("qwen_request_duration_seconds").record(start.elapsed().as_secs_f64());
    guard.armed = false;
    drop(guard);

    if let Some((kind, msg)) = result {
        metrics::counter!("qwen_requests_total", "status" => "error").increment(1);
        // Client errors (e.g. oversized prompt) → 400; worker faults → 502.
        let status = if kind == "context_length_exceeded" {
            StatusCode::BAD_REQUEST
        } else {
            StatusCode::BAD_GATEWAY
        };
        return (status, Json(json!({"error": {"message": msg, "type": kind}})))
            .into_response();
    }
    metrics::counter!("qwen_requests_total", "status" => "ok").increment(1);

    let mut msg = serde_json::Map::new();
    msg.insert("role".into(), json!("assistant"));
    if tool_calls.is_empty() {
        msg.insert("content".into(), json!(content));
    } else {
        // OpenAI: content is null when the assistant makes tool calls.
        msg.insert("content".into(),
                   if content.is_empty() { Value::Null } else { json!(content) });
        msg.insert("tool_calls".into(), json!(tool_calls_json(&tool_calls)));
    }
    if !reasoning.is_empty() {
        msg.insert("reasoning_content".into(), json!(reasoning));
    }
    Json(json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": Value::Object(msg), "finish_reason": finish}],
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.gen_tokens,
            "total_tokens": usage.prompt_tokens + usage.gen_tokens,
        },
    }))
    .into_response()
}

// ---- misc endpoints -------------------------------------------------------

#[derive(Serialize)]
pub struct ModelList {
    object: &'static str,
    data: Vec<ModelInfo>,
}
#[derive(Serialize)]
pub struct ModelInfo {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

pub async fn list_models(State(st): State<AppState>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![ModelInfo {
            id: st.model_name.clone(),
            object: "model",
            owned_by: "local",
        }],
    })
}

pub async fn health() -> &'static str {
    "ok"
}
