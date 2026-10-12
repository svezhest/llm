use std::convert::Infallible;
use std::sync::OnceLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::{
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::sse::{Event as SseEvent, KeepAlive, Sse},
    response::{IntoResponse, Response},
    Json,
};
use futures::stream::Stream;
use serde::Serialize;
use serde_json::{json, Value};
use tracing::{info, warn};
use uuid::Uuid;

use crate::pb;
use crate::pool::Slot;
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
    // Request clock: duration and TTFT are what the client sees, queue waits included.
    let t0 = Instant::now();
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
    // vLLM: the response id is chatcmpl-<request_id> when the client gives one. The
    // worker gets an id unique per generation (Cancel goes by it): a client reusing a
    // request_id for concurrent requests must not cancel the other one.
    let fresh = Uuid::new_v4().simple().to_string();
    let (id, worker_id) = match &req.request_id {
        Some(rid) => (format!("chatcmpl-{rid}"), format!("chatcmpl-{rid}-{}", &fresh[..12])),
        None => (format!("chatcmpl-{fresh}"), format!("chatcmpl-{fresh}")),
    };
    let pbreq = match request::to_pb(&worker_id, &req) {
        Ok(r) => r,
        Err(e) => return bad_request(e),
    };
    // A full queue is a 503 before the worker is asked anything.
    let Some(admit) = st.pool.try_admit() else {
        metrics::counter!("qwen_requests_total", "status" => "rejected").increment(1);
        return error(StatusCode::SERVICE_UNAVAILABLE, "ServiceUnavailableError", "queue full", None);
    };
    // What the worker would refuse (chat template, context length, schema / tool
    // grammars, images on a text-only model) is refused here, before the request waits
    // for an in-flight slot, so it is a 400 for streaming requests too, not an error
    // frame after a 200. (tonic takes the message by value, hence the clone; Generate
    // needs the same request after.) A worker with no free gRPC thread would leave
    // Check waiting behind running generations: past GATEWAY_CHECK_TIMEOUT_S it is a 503.
    let mut client = st.client.clone();
    let limit = check_timeout();
    let mut creq = tonic::Request::new(pbreq.clone());
    creq.set_timeout(limit);
    match tokio::time::timeout(limit, client.check(creq)).await {
        Ok(Ok(r)) => {
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
        Ok(Err(e)) if is_busy(&e) || e.code() == tonic::Code::DeadlineExceeded => {
            return busy(&format!("worker busy: {}", e.message()));
        }
        Ok(Err(e)) => {
            metrics::counter!("qwen_errors_total", "kind" => "rpc").increment(1);
            let msg = if e.code() == tonic::Code::Unimplemented {
                "worker is older than the gateway (no Check RPC)".to_string()
            } else {
                e.to_string()
            };
            return error(StatusCode::BAD_GATEWAY, "rpc", &msg, None);
        }
        Err(_) => {
            warn!(id = %id, timeout_s = limit.as_secs_f64(), "worker Check timed out");
            return busy(&format!("worker busy: no answer to the request check within {}s",
                                 limit.as_secs_f64()));
        }
    }
    let job = Job {
        id,
        pbreq,
        body,
        show: Show {
            reasoning: req.include_reasoning.unwrap_or(true),
            usage: req.stream_options.as_ref().is_some_and(|o| o.include_usage),
        },
        t0,
        admit,
    };
    if req.stream {
        stream_response(st, job).await.into_response()
    } else {
        json_response(st, job).await.into_response()
    }
}

/// GATEWAY_CHECK_TIMEOUT_S (default 30): how long Check may take before a 503.
fn check_timeout() -> Duration {
    static T: OnceLock<Duration> = OnceLock::new();
    *T.get_or_init(|| {
        let s = std::env::var("GATEWAY_CHECK_TIMEOUT_S").ok();
        match s.as_deref().map(str::trim).map(str::parse::<f64>) {
            None => Duration::from_secs(30),
            Some(Ok(v)) if v > 0.0 && v.is_finite() => Duration::from_secs_f64(v),
            Some(_) => {
                warn!(value = ?s, "GATEWAY_CHECK_TIMEOUT_S is not a positive number; using 30");
                Duration::from_secs(30)
            }
        }
    })
}

/// The worker refused for load (RESOURCE_EXHAUSTED: no room for another generation).
fn is_busy(e: &tonic::Status) -> bool {
    e.code() == tonic::Code::ResourceExhausted
}

/// 503 with the same shape as a full queue; the caller's slots drop on return.
fn busy(msg: &str) -> Response {
    metrics::counter!("qwen_requests_total", "status" => "rejected").increment(1);
    metrics::counter!("qwen_errors_total", "kind" => "busy").increment(1);
    error(StatusCode::SERVICE_UNAVAILABLE, "ServiceUnavailableError", msg, None)
}

/// A request that passed the checks and holds a queue place.
struct Job {
    id: String,           // response id (chatcmpl-...); pbreq.id is the worker's
    pbreq: pb::Request,
    body: Bytes,          // as received, for the journal
    show: Show,
    t0: Instant,          // handler entry
    admit: Slot,
}

/// Owns the request's queue + in-flight slots and guarantees cleanup on ANY
/// exit path (clean finish OR client disconnect): frees both slots (each Slot
/// updates its gauge on drop) and — unless disarmed — fires Worker.Cancel.
struct ReqGuard {
    st: AppState,
    worker_id: String,
    armed: bool,
    admit: Option<Slot>,
    inflight: Option<Slot>,
}
impl Drop for ReqGuard {
    fn drop(&mut self) {
        let _ = self.inflight.take();
        let _ = self.admit.take();
        if self.armed {
            metrics::counter!("qwen_cancellations_total").increment(1);
            let mut client = self.st.client.clone();
            let id = self.worker_id.clone();
            tokio::spawn(async move {
                let _ = client.cancel(pb::CancelReq { id }).await;
            });
        }
    }
}

/// Wait for an in-flight slot; the guard owns both slots from here on.
async fn begin(st: &AppState, worker_id: &str, admit: Slot) -> (ReqGuard, Duration) {
    let w = Instant::now();
    let inflight = st.pool.acquire_inflight().await;
    let waited = w.elapsed();
    metrics::histogram!("qwen_gateway_queue_seconds").record(waited.as_secs_f64());
    let guard = ReqGuard {
        st: st.clone(), worker_id: worker_id.to_string(), armed: true,
        admit: Some(admit), inflight: Some(inflight),
    };
    (guard, waited)
}

/// Error kind of a failed Generate call / stream: `busy` (-> 503) or `fallback`.
fn rpc_kind(e: &tonic::Status, fallback: &str) -> String {
    if is_busy(e) { "busy".into() } else { fallback.into() }
}

async fn stream_response(
    st: AppState,
    job: Job,
) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
    let Job { id, pbreq, body, show, t0, admit } = job;
    let body = async_stream::stream! {
        let worker_id = pbreq.id.clone();
        let (mut guard, waited) = begin(&st, &worker_id, admit).await;
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
                            match &ev.kind {
                                Some(pb::event::Kind::Start(s)) if !s.model.is_empty() => {
                                    model = s.model.clone();
                                }
                                Some(pb::event::Kind::Done(d)) => on_done(&id, &worker_id, d, waited),
                                _ => {}
                            }
                            acc.add(&ev);
                            for sse in event_to_sse(&id, &model, created, ev, &mut ttft_done,
                                                    &t0, show) {
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
                            warn!(id = %id, worker_id = %worker_id, error = %e, "worker stream error");
                            let kind = rpc_kind(&e, "stream");
                            metrics::counter!("qwen_errors_total", "kind" => kind.clone()).increment(1);
                            yield Ok(error_frame(&e.to_string(), &kind));
                            acc.err = Some((kind, e.to_string()));
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                let kind = rpc_kind(&e, "rpc");
                metrics::counter!("qwen_errors_total", "kind" => kind.clone()).increment(1);
                yield Ok(error_frame(&e.to_string(), &kind));
                acc.err = Some((kind, e.to_string()));
            }
        }

        metrics::histogram!("qwen_request_duration_seconds").record(t0.elapsed().as_secs_f64());
        let status = match &acc.err {
            None => "ok",
            Some((k, _)) if k == "busy" => "rejected",
            Some(_) => "error",
        };
        metrics::counter!("qwen_requests_total", "status" => status).increment(1);
        guard.armed = false;   // completed (or failed) -> guard drop frees slots, no cancel
        drop(guard);
        if let Some(j) = &st.journal {
            let mut resp = acc.response(&id, &model, created, show);
            resp["stream"] = json!(true);
            j.write(&id, &body, resp);
        }
        yield Ok(SseEvent::default().data("[DONE]"));
    };

    // Comment lines while nothing else is sent (waiting for a slot, a long prefill) so
    // clients and proxies do not time the stream out; OpenAI clients skip them.
    Sse::new(body).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)).text("keep-alive"))
}

/// An error frame in the stream; `busy` (the worker refused for load) carries 503.
fn error_frame(msg: &str, kind: &str) -> SseEvent {
    let (kind, code) = if kind == "busy" { ("ServiceUnavailableError", 503) } else { (kind, 500) };
    SseEvent::default().data(json!({"error": {"message": msg, "type": kind, "code": code}}).to_string())
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
    t0: &Instant,
    show: Show,
) -> Vec<SseEvent> {
    let Some(kind) = ev.kind else { return vec![] };
    match kind {
        pb::event::Kind::Start(_) => {
            vec![chunk_frame(id, model, created, json!({"role": "assistant", "content": ""}), None)]
        }
        pb::event::Kind::Token(t) => {
            if !*ttft_done {
                metrics::histogram!("qwen_ttft_seconds").record(t0.elapsed().as_secs_f64());
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

/// Log a finished generation and record the worker-reported scheduling metrics.
/// queue_ms / batch_mean are recorded only from a worker that reports them (an older
/// one leaves batch_mean 0; a batching one always reports >= 1).
fn on_done(id: &str, worker_id: &str, d: &pb::Done, waited: Duration) {
    let u = d.usage.unwrap_or_default();
    if u.batch_mean > 0.0 {
        metrics::histogram!("qwen_worker_queue_seconds").record(f64::from(u.queue_ms) / 1000.0);
        metrics::histogram!("qwen_batch_mean").record(f64::from(u.batch_mean));
    }
    info!(id, worker_id, finish = %d.finish, prompt_tokens = u.prompt_tokens,
          gen_tokens = u.gen_tokens, queue_wait_ms = waited.as_millis() as u64, ttft_ms = u.ttft_ms,
          tps = u.tps, queue_ms = u.queue_ms, batch_mean = u.batch_mean,
          tool_calls = d.tool_calls.len(), "done");
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

async fn json_response(st: AppState, job: Job) -> Response {
    let Job { id, pbreq, body, show, t0, admit } = job;
    let worker_id = pbreq.id.clone();
    let (mut guard, waited) = begin(&st, &worker_id, admit).await;
    let created = now_unix();

    let mut acc = Acc::default();
    let mut client = st.client.clone();
    match client.generate(pbreq).await {
        Ok(resp) => {
            let mut stream = resp.into_inner();
            loop {
                match stream.message().await {
                    Ok(Some(ev)) => {
                        if let Some(pb::event::Kind::Done(d)) = &ev.kind {
                            on_done(&id, &worker_id, d, waited);
                        }
                        acc.add(&ev);
                    }
                    Ok(None) => break,
                    Err(e) => {
                        warn!(id = %id, worker_id = %worker_id, error = %e, "worker stream error");
                        acc.err = Some((rpc_kind(&e, "stream"), e.to_string()));
                        break;
                    }
                }
            }
            if acc.err.is_none() && acc.finish.is_none() {
                acc.err = Some(("stream".into(), "worker stream ended without a Done event".into()));
            }
        }
        Err(e) => acc.err = Some((rpc_kind(&e, "rpc"), e.to_string())),
    }

    metrics::histogram!("qwen_request_duration_seconds").record(t0.elapsed().as_secs_f64());
    guard.armed = false;
    drop(guard);

    let model = acc.model.clone().unwrap_or_else(|| st.worker_model.clone());
    let out = acc.response(&id, &model, created, show);
    if let Some(j) = &st.journal {
        j.write(&id, &body, out.clone());
    }
    if let Some((kind, msg)) = &acc.err {
        if kind == "busy" {
            return busy(&format!("worker busy: {msg}"));
        }
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
                if let Some(u) = d.usage { self.usage = u; }
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
