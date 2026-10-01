use std::convert::Infallible;
use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::{
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::sse::{Event as SseEvent, Sse},
    response::{IntoResponse, Response},
    Json,
};
use futures::stream::Stream;
use serde::Serialize;
use serde_json::{json, Value};
use tracing::{info, warn};
use uuid::Uuid;

use crate::pb;
use crate::request::{self, BadRequest, ChatRequest};
use crate::AppState;

/// vLLM's error shape: {"error": {"message", "type", "param", "code"}}.
fn error(status: StatusCode, kind: &str, msg: &str, param: Option<&str>) -> Response {
    let body = json!({"error": {"message": msg, "type": kind, "param": param, "code": status.as_u16()}});
    (status, Json(body)).into_response()
}

fn bad_request(e: BadRequest) -> Response {
    error(StatusCode::BAD_REQUEST, "BadRequestError", &e.msg, e.param)
}

/// Extra names `model` may carry (GATEWAY_SERVED_MODEL_NAMES, comma-separated, like
/// vLLM's --served-model-name); the loaded model's id is always accepted.
fn served_names() -> &'static [String] {
    static N: OnceLock<Vec<String>> = OnceLock::new();
    N.get_or_init(|| {
        std::env::var("GATEWAY_SERVED_MODEL_NAMES").unwrap_or_default()
            .split(',').map(str::trim).filter(|s| !s.is_empty()).map(String::from).collect()
    })
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

fn usage_json(u: &pb::Usage) -> Value {
    json!({
        "prompt_tokens": u.prompt_tokens,
        "total_tokens": u.prompt_tokens + u.gen_tokens,
        "completion_tokens": u.gen_tokens,
        "completion_tokens_details": {"reasoning_tokens": u.reasoning_tokens},
    })
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// What the answer shows besides the generation itself.
#[derive(Clone, Copy)]
struct Show {
    reasoning: bool, // include_reasoning (vLLM; default true)
    usage: bool,     // stream_options.include_usage
}

// ---- handlers -------------------------------------------------------------

pub async fn chat_completions(State(st): State<AppState>, body: Bytes) -> Response {
    // Parsed by hand (not the Json extractor) so the journal gets the body as it came.
    let raw: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return error(StatusCode::BAD_REQUEST, "BadRequestError", &e.to_string(), None),
    };
    let Some(fields) = raw.as_object() else {
        return error(StatusCode::BAD_REQUEST, "BadRequestError", "request body must be a JSON object", None);
    };
    if let Err(e) = request::check_fields(fields) {
        return bad_request(e);
    }
    // vLLM answers 400 (not 422) for a request that does not fit the schema too.
    let req: ChatRequest = match serde_json::from_value(raw.clone()) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, "BadRequestError", &e.to_string(), None),
    };
    if let Err(e) = request::validate(&req) {
        return bad_request(e);
    }
    // vLLM: the response id is chatcmpl-<request_id> when the client gives one
    let id = format!("chatcmpl-{}", req.request_id.clone()
        .unwrap_or_else(|| Uuid::new_v4().simple().to_string()));
    let pbreq = match request::to_pb(&id, &req) {
        Ok(r) => r,
        Err(e) => return bad_request(e),
    };
    // What the worker would refuse (chat template, context length, schema / tool
    // grammars, images on a text-only model) is refused here, before queueing, so it
    // is a 400 for streaming requests too, not an error frame after a 200.
    let mut client = st.client.clone();
    match client.check(pbreq.clone()).await {
        Ok(r) => {
            let r = r.into_inner();
            if let Some(m) = &req.model {
                if *m != r.model && !served_names().contains(m) {
                    return error(StatusCode::NOT_FOUND, "NotFoundError",
                                 &format!("The model `{m}` does not exist."), Some("model"));
                }
            }
            if r.internal {
                return error(StatusCode::INTERNAL_SERVER_ERROR, "InternalServerError", &r.error, None);
            }
            if !r.error.is_empty() {
                return error(StatusCode::BAD_REQUEST, "BadRequestError", &r.error, None);
            }
        }
        Err(e) => {
            metrics::counter!("qwen_errors_total", "kind" => "rpc").increment(1);
            let msg = if e.code() == tonic::Code::Unimplemented {
                "worker is older than the gateway (no Check RPC)".to_string()
            } else {
                e.to_string()
            };
            return error(StatusCode::BAD_GATEWAY, "rpc", &msg, None);
        }
    }
    let admit = match st.pool.try_admit() {
        Some(p) => p,
        None => {
            metrics::counter!("qwen_requests_total", "status" => "rejected").increment(1);
            return error(StatusCode::SERVICE_UNAVAILABLE, "ServiceUnavailableError", "queue full", None);
        }
    };
    let show = Show {
        reasoning: req.include_reasoning.unwrap_or(true),
        usage: req.stream_options.as_ref().is_some_and(|o| o.include_usage),
    };
    if req.stream {
        stream_response(st, admit, id, pbreq, body, show).await.into_response()
    } else {
        json_response(st, admit, id, pbreq, body, show).await.into_response()
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
    pbreq: pb::Request,
    body: Bytes,
    show: Show,
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
        // The whole answer, assembled from the chunks (journal, usage, errors).
        let mut acc = Acc::default();
        // The model that generates this answer, from the worker's Start event.
        let mut model = st.worker_model.clone();

        let mut client = st.client.clone();
        match client.generate(pbreq).await {
            Ok(resp) => {
                let mut stream = resp.into_inner();
                loop {
                    match stream.message().await {
                        Ok(Some(ev)) => {
                            if let Some(pb::event::Kind::Start(s)) = &ev.kind {
                                if !s.model.is_empty() { model = s.model.clone(); }
                            }
                            acc.add(&ev);
                            for sse in event_to_sse(&id, &model, created, ev, &mut ttft_done,
                                                    &start, show) {
                                yield Ok(sse);
                            }
                        }
                        Ok(None) => {
                            if acc.err.is_none() && acc.finish.is_none() {
                                // the worker went away mid-answer: say so, don't end it as if complete
                                let msg = "worker stream ended without a Done event";
                                metrics::counter!("qwen_errors_total", "kind" => "stream").increment(1);
                                acc.err = Some(("stream".into(), msg.into()));
                                yield Ok(error_frame(msg, "stream"));
                            }
                            break;
                        }
                        Err(e) => {
                            warn!(error = %e, "worker stream error");
                            metrics::counter!("qwen_errors_total", "kind" => "stream").increment(1);
                            acc.err = Some(("stream".into(), e.to_string()));
                            yield Ok(error_frame(&e.to_string(), "stream"));
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                metrics::counter!("qwen_errors_total", "kind" => "rpc").increment(1);
                acc.err = Some(("rpc".into(), e.to_string()));
                yield Ok(error_frame(&e.to_string(), "rpc"));
            }
        }

        metrics::histogram!("qwen_request_duration_seconds").record(start.elapsed().as_secs_f64());
        let status = if acc.err.is_some() { "error" } else { "ok" };
        metrics::counter!("qwen_requests_total", "status" => status).increment(1);
        guard.armed = false;   // completed (or failed) -> guard drop frees slots, no cancel
        if let Some(j) = &st.journal {
            let mut resp = acc.response(&id, &model, created, show);
            resp["stream"] = json!(true);
            j.write(&id, &body, &resp);
        }
        yield Ok(SseEvent::default().data("[DONE]"));
        drop(guard);
    };

    Sse::new(body)
}

fn error_frame(msg: &str, kind: &str) -> SseEvent {
    SseEvent::default().data(json!({"error": {"message": msg, "type": kind, "code": 500}}).to_string())
}

/// Map one worker Event to OpenAI chunk SSE frames (vLLM's shape: a first chunk with
/// the role and empty content, then deltas, then the finish chunk and, with
/// stream_options.include_usage, a usage chunk with no choices).
fn event_to_sse(
    id: &str,
    model: &str,
    created: u64,
    ev: pb::Event,
    ttft_done: &mut bool,
    start: &Instant,
    show: Show,
) -> Vec<SseEvent> {
    let Some(kind) = ev.kind else { return vec![] };
    match kind {
        pb::event::Kind::Start(_) => {
            vec![chunk_frame(id, model, created, json!({"role": "assistant", "content": ""}), None)]
        }
        pb::event::Kind::Token(t) => {
            if !*ttft_done {
                metrics::histogram!("qwen_ttft_seconds").record(start.elapsed().as_secs_f64());
                *ttft_done = true;
            }
            metrics::counter!("qwen_tokens_generated_total").increment(1);
            let delta = if !t.thinking {
                json!({"content": t.text})
            } else if show.reasoning {
                // vLLM >= 0.2x names it `reasoning`; `reasoning_content` is the older name
                json!({"reasoning": t.text, "reasoning_content": t.text})
            } else {
                return vec![];
            };
            vec![chunk_frame(id, model, created, delta, None)]
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
            let mut out = vec![chunk_frame(id, model, created, delta, Some(&d.finish))];
            if show.usage {
                let body = json!({
                    "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": [], "usage": usage_json(&d.usage.unwrap_or_default()),
                });
                out.push(SseEvent::default().data(body.to_string()));
            }
            out
        }
        pb::event::Kind::Error(e) => {
            metrics::counter!("qwen_errors_total", "kind" => e.kind.clone()).increment(1);
            vec![SseEvent::default().data(
                json!({"error": {"message": e.msg, "type": e.kind, "code": 500}}).to_string())]
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
    pbreq: pb::Request,
    body: Bytes,
    show: Show,
) -> Response {
    let inflight = st.pool.acquire_inflight().await;
    let mut guard = ReqGuard {
        st: st.clone(), id: id.clone(), armed: true,
        admit: Some(admit), inflight: Some(inflight),
    };
    let start = Instant::now();
    let created = now_unix();

    let mut acc = Acc::default();
    let mut client = st.client.clone();
    match client.generate(pbreq).await {
        Ok(resp) => {
            let mut stream = resp.into_inner();
            loop {
                match stream.message().await {
                    Ok(Some(ev)) => acc.add(&ev),
                    Ok(None) => break,
                    Err(e) => {
                        warn!(error = %e, "worker stream error");
                        acc.err = Some(("stream".into(), e.to_string()));
                        break;
                    }
                }
            }
            if acc.err.is_none() && acc.finish.is_none() {
                acc.err = Some(("stream".into(), "worker stream ended without a Done event".into()));
            }
        }
        Err(e) => acc.err = Some(("rpc".into(), e.to_string())),
    }

    metrics::histogram!("qwen_request_duration_seconds").record(start.elapsed().as_secs_f64());
    guard.armed = false;
    drop(guard);

    let model = acc.model.clone().unwrap_or_else(|| st.worker_model.clone());
    let out = acc.response(&id, &model, created, show);
    if let Some(j) = &st.journal {
        j.write(&id, &body, &out);
    }
    if let Some((kind, msg)) = &acc.err {
        metrics::counter!("qwen_requests_total", "status" => "error").increment(1);
        metrics::counter!("qwen_errors_total", "kind" => kind.clone()).increment(1);
        // A request the worker refuses -> 400; a lost worker -> 502; a failed generation -> 500.
        return match kind.as_str() {
            "context_length_exceeded" | "invalid_request" => {
                error(StatusCode::BAD_REQUEST, "BadRequestError", msg, None)
            }
            "rpc" | "stream" => error(StatusCode::BAD_GATEWAY, kind, msg, None),
            _ => error(StatusCode::INTERNAL_SERVER_ERROR, "InternalServerError", msg, None),
        };
    }
    metrics::counter!("qwen_requests_total", "status" => "ok").increment(1);
    Json(out).into_response()
}

/// Worker events of one request folded into a chat.completion.
#[derive(Default)]
struct Acc {
    content: String,
    reasoning: String,
    finish: Option<String>,
    usage: pb::Usage,
    tool_calls: Vec<pb::ToolCall>,
    err: Option<(String, String)>,
    model: Option<String>,
}

impl Acc {
    fn add(&mut self, ev: &pb::Event) {
        match &ev.kind {
            Some(pb::event::Kind::Start(s)) if !s.model.is_empty() => {
                self.model = Some(s.model.clone());
            }
            Some(pb::event::Kind::Token(t)) => {
                if t.thinking { self.reasoning.push_str(&t.text); }
                else { self.content.push_str(&t.text); }
            }
            Some(pb::event::Kind::Done(d)) => {
                self.tool_calls = d.tool_calls.clone();
                self.finish = Some(d.finish.clone());
                if let Some(u) = &d.usage { self.usage = u.clone(); }
            }
            Some(pb::event::Kind::Error(e)) => { self.err = Some((e.kind.clone(), e.msg.clone())); }
            _ => {}
        }
    }

    fn response(&self, id: &str, model: &str, created: u64, show: Show) -> Value {
        if let Some((kind, msg)) = &self.err {
            return json!({"error": {"message": msg, "type": kind}});
        }
        let mut msg = serde_json::Map::new();
        msg.insert("role".into(), json!("assistant"));
        // vLLM: no answer text at all (only reasoning, or only tool calls) = null
        let content = if self.content.is_empty() && (!self.reasoning.is_empty() || !self.tool_calls.is_empty()) {
            Value::Null
        } else {
            json!(self.content)
        };
        msg.insert("content".into(), content);
        if !self.tool_calls.is_empty() {
            msg.insert("tool_calls".into(), json!(tool_calls_json(&self.tool_calls)));
        }
        if show.reasoning && !self.reasoning.is_empty() {
            // vLLM >= 0.2x names it `reasoning`; `reasoning_content` is the older name
            msg.insert("reasoning".into(), json!(self.reasoning));
            msg.insert("reasoning_content".into(), json!(self.reasoning));
        }
        json!({
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "message": Value::Object(msg),
                         "finish_reason": self.finish.as_deref().unwrap_or("stop")}],
            "usage": usage_json(&self.usage),
        })
    }
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

/// The model the worker has loaded right now (Health), plus GATEWAY_SERVED_MODEL_NAMES.
pub async fn list_models(State(st): State<AppState>) -> Response {
    let mut client = st.client.clone();
    match client.health(pb::Empty {}).await {
        Ok(r) => {
            let mut ids = vec![r.into_inner().model];
            ids.extend(served_names().iter().cloned());
            Json(ModelList {
                object: "list",
                data: ids.into_iter().map(|id| ModelInfo { id, object: "model", owned_by: "local" })
                    .collect(),
            })
            .into_response()
        }
        Err(e) => error(StatusCode::SERVICE_UNAVAILABLE, "rpc",
                        &format!("worker unavailable: {}", e.message()), None),
    }
}

pub async fn health() -> &'static str {
    "ok"
}
