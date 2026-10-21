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

/// OpenAI / vLLM usage; `prompt_tokens_details.cached_tokens` = prompt tokens the worker
/// resumed from its prompt cache (0 with the cache off or from an older worker).
fn usage_json(u: &pb::Usage) -> Value {
    json!({
        "prompt_tokens": u.prompt_tokens,
        "total_tokens": u.prompt_tokens + u.gen_tokens,
        "completion_tokens": u.gen_tokens,
        "prompt_tokens_details": {"cached_tokens": u.cached_tokens},
        "completion_tokens_details": {"reasoning_tokens": u.reasoning_tokens},
    })
}

/// A log-probability as JSON: the f32's shortest decimal (what the worker computed,
/// without the f64 widening's noise digits); a non-finite value (never from the
/// worker's fp32 log-softmax of raw logits) is null.
fn lp_json(x: f32) -> Value {
    if !x.is_finite() {
        return Value::Null;
    }
    x.to_string().parse::<f64>().map_or(Value::Null, Value::from)
}

/// OpenAI's `logprobs.content` entries: one per generated token, in order (reasoning,
/// answer and the final EOS alike); `bytes` = the UTF-8 of `token` (the worker's
/// `decode([id])`, so a token that is part of a multi-byte character shows U+FFFD).
fn content_logprobs_json(lp: &[pb::TokenLogprob]) -> Vec<Value> {
    lp.iter()
        .map(|t| {
            let top: Vec<Value> = t.top.iter()
                .map(|x| json!({"token": x.token, "logprob": lp_json(x.logprob),
                                "bytes": x.token.as_bytes()}))
                .collect();
            json!({"token": t.token, "logprob": lp_json(t.logprob), "bytes": t.token.as_bytes(),
                   "top_logprobs": top})
        })
        .collect()
}

/// Teacher forcing over the prompt: `prompt_logprobs` (one entry per prompt token, null
/// where it was not scored: below `first` — position 0, and up to the depth of a
/// prompt-cache hit) and `prompt_logprobs_messages` (per assistant span [start, end):
/// how many of its positions were scored and the sum of their log p; null sum when
/// none was).
fn prompt_logprobs_json(p: &pb::PromptLogprobs) -> (Value, Value) {
    let n = p.tokens.len();
    let first = p.first as usize;
    let at = |i: usize| i.checked_sub(first).and_then(|k| p.logprobs.get(k)).copied();
    let positions: Vec<Value> = p.tokens.iter().enumerate()
        .map(|(i, t)| match at(i) {
            Some(lp) => json!({"token": t.token, "token_id": t.token_id, "logprob": lp_json(lp)}),
            None => Value::Null,
        })
        .collect();
    let spans: Vec<Value> = p.spans.iter()
        .map(|s| {
            let (a, b) = (s.start as usize, (s.end as usize).min(n));
            let scored: Vec<f32> = (a..b.max(a)).filter_map(at).collect();
            let sum = if scored.is_empty() {
                Value::Null
            } else {
                Value::from(scored.iter().map(|&x| f64::from(x)).sum::<f64>())
            };
            json!({"message": s.message, "role": "assistant", "start": s.start, "end": s.end,
                   "tokens": s.end.saturating_sub(s.start), "scored": scored.len(),
                   "logprob": sum})
        })
        .collect();
    (Value::Array(positions), Value::Array(spans))
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// What the answer shows besides the generation itself.
#[derive(Clone, Copy)]
struct Show {
    reasoning: bool,       // include_reasoning (vLLM; default true)
    usage: bool,           // stream_options.include_usage
    logprobs: bool,        // logprobs: true -> choices[].logprobs.content
    prompt_logprobs: bool, // prompt_logprobs -> prompt_logprobs + prompt_logprobs_messages
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
            logprobs: req.logprobs_on(),
            prompt_logprobs: req.prompt_logprobs_on(),
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
            j.write(&id, &body, for_journal(resp));
        }
        yield Ok(SseEvent::default().data("[DONE]"));
    };

    // Comment lines while nothing else is sent (waiting for a slot, a long prefill) so
    // clients and proxies do not time the stream out; OpenAI clients skip them.
    Sse::new(body).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)).text("keep-alive"))
}

/// The journal keeps a response without its per-token log-probabilities (a 30K-token
/// prompt is ~2 MB of `prompt_logprobs`, a 4K answer with top_logprobs 20 several MB of
/// `logprobs.content`, and attribution runs send thousands): `prompt_logprobs` becomes
/// `prompt_logprobs_positions` = its length, `choices[0].logprobs` becomes
/// `{"tokens": n, "logprob": sum}`; the per-message sums stay.
fn for_journal(mut resp: Value) -> Value {
    let Some(o) = resp.as_object_mut() else { return resp };
    if let Some(Value::Array(a)) = o.get("prompt_logprobs") {
        let n = a.len();
        o.remove("prompt_logprobs");
        o.insert("prompt_logprobs_positions".into(), json!(n));
    }
    let lp = o.get_mut("choices").and_then(|c| c.get_mut(0)).and_then(|c| c.get_mut("logprobs"));
    if let Some(lp) = lp {
        if let Some(Value::Array(a)) = lp.get("content") {
            let sum: f64 = a.iter().filter_map(|t| t["logprob"].as_f64()).sum();
            *lp = json!({"tokens": a.len(), "logprob": sum});
        }
    }
    resp
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
    event_chunks(id, model, created, ev, ttft_done, t0, show)
        .into_iter()
        .map(|body| SseEvent::default().data(body.to_string()))
        .collect()
}

/// The JSON bodies of `event_to_sse`'s frames.
fn event_chunks(
    id: &str,
    model: &str,
    created: u64,
    ev: pb::Event,
    ttft_done: &mut bool,
    t0: &Instant,
    show: Show,
) -> Vec<Value> {
    let Some(kind) = ev.kind else { return vec![] };
    match kind {
        pb::event::Kind::Start(_) => {
            vec![chunk_json(id, model, created, json!({"role": "assistant", "content": ""}), None)]
        }
        pb::event::Kind::Token(t) => {
            if !*ttft_done {
                metrics::histogram!("qwen_ttft_seconds").record(t0.elapsed().as_secs_f64());
                *ttft_done = true;
            }
            if !t.text.is_empty() {
                metrics::counter!("qwen_tokens_generated_total").increment(1);
            }
            // `lp` is not aligned with `text`: the tokens read since the previous event.
            // A chunk that carries only log-probabilities (a held-back piece, the EOS,
            // hidden reasoning) has an empty delta.
            let lp = (show.logprobs && !t.lp.is_empty()).then(|| content_logprobs_json(&t.lp));
            let delta = if !t.thinking {
                if t.text.is_empty() && lp.is_some() { json!({}) } else { json!({"content": t.text}) }
            } else if show.reasoning && !t.text.is_empty() {
                // vLLM >= 0.2x names it `reasoning`; `reasoning_content` is the older name
                json!({"reasoning": t.text, "reasoning_content": t.text})
            } else if lp.is_some() {
                json!({})
            } else {
                return vec![];
            };
            let mut body = chunk_json(id, model, created, delta, None);
            if let Some(lp) = lp {
                body["choices"][0]["logprobs"] = json!({"content": lp});
            }
            vec![body]
        }
        pb::event::Kind::PromptLogprobs(p) => {
            if !show.prompt_logprobs {
                return vec![];
            }
            let (positions, spans) = prompt_logprobs_json(&p);
            vec![json!({
                "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [], "prompt_logprobs": positions, "prompt_logprobs_messages": spans,
            })]
        }
        pb::event::Kind::Done(d) => {
            let delta = if d.tool_calls.is_empty() {
                json!({})
            } else {
                json!({ "tool_calls": tool_calls_json(&d.tool_calls) })
            };
            let mut out = vec![chunk_json(id, model, created, delta, Some(&d.finish))];
            if show.usage {
                out.push(json!({
                    "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": [], "usage": usage_json(&d.usage.unwrap_or_default()),
                }));
            }
            out
        }
        pb::event::Kind::Error(e) => {
            metrics::counter!("qwen_errors_total", "kind" => e.kind.clone()).increment(1);
            vec![json!({"error": {"message": e.msg, "type": e.kind, "code": 500}})]
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
    metrics::counter!("qwen_gateway_cached_tokens_total").increment(u64::from(u.cached_tokens));
    info!(id, worker_id, finish = %d.finish, prompt_tokens = u.prompt_tokens,
          cached_tokens = u.cached_tokens,
          gen_tokens = u.gen_tokens, queue_wait_ms = waited.as_millis() as u64, ttft_ms = u.ttft_ms,
          tps = u.tps, queue_ms = u.queue_ms, batch_mean = u.batch_mean,
          tool_calls = d.tool_calls.len(), "done");
}

fn chunk_json(id: &str, model: &str, created: u64, delta: Value, finish: Option<&str>) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    })
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
        j.write(&id, &body, for_journal(out.clone()));
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
    lp: Vec<pb::TokenLogprob>,           // every Token event's `lp`, in order
    prompt: Option<pb::PromptLogprobs>,  // the PromptLogprobs event (before Done)
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
                self.lp.extend(t.lp.iter().cloned());
            }
            Some(pb::event::Kind::PromptLogprobs(p)) => { self.prompt = Some(p.clone()); }
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
        // OpenAI: `logprobs` is always in the choice, null unless asked for
        let logprobs = if show.logprobs {
            json!({"content": content_logprobs_json(&self.lp)})
        } else {
            Value::Null
        };
        let mut out = json!({
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "message": Value::Object(msg), "logprobs": logprobs,
                         "finish_reason": self.finish.as_deref().unwrap_or("stop")}],
            "usage": usage_json(&self.usage),
        });
        if show.prompt_logprobs {
            // null pair: a worker that sent no PromptLogprobs event (older than the field)
            let (positions, spans) = self.prompt.as_ref().map(prompt_logprobs_json)
                .unwrap_or((Value::Null, Value::Null));
            out["prompt_logprobs"] = positions;
            out["prompt_logprobs_messages"] = spans;
        }
        out
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_reports_cached_tokens() {
        let u = pb::Usage { prompt_tokens: 100, gen_tokens: 7, reasoning_tokens: 3,
                            cached_tokens: 64, ..Default::default() };
        assert_eq!(usage_json(&u), json!({
            "prompt_tokens": 100, "total_tokens": 107, "completion_tokens": 7,
            "prompt_tokens_details": {"cached_tokens": 64},
            "completion_tokens_details": {"reasoning_tokens": 3},
        }));
        // an older worker (no field): 0, the object is always there (as vLLM)
        let old = usage_json(&pb::Usage::default());
        assert_eq!(old["prompt_tokens_details"], json!({"cached_tokens": 0}));
    }

    fn tl(id: u32, token: &str, logprob: f32, top: &[(u32, &str, f32)]) -> pb::TokenLogprob {
        pb::TokenLogprob {
            token_id: id, token: token.into(), logprob,
            top: top.iter().map(|&(token_id, t, logprob)| pb::TopLogprob {
                token_id, token: t.into(), logprob }).collect(),
        }
    }

    fn ev(kind: pb::event::Kind) -> pb::Event {
        pb::Event { id: "w".into(), kind: Some(kind) }
    }

    fn token(text: &str, thinking: bool, lp: Vec<pb::TokenLogprob>) -> pb::Event {
        ev(pb::event::Kind::Token(pb::Token { text: text.into(), thinking, lp }))
    }

    fn done(finish: &str) -> pb::Event {
        ev(pb::event::Kind::Done(pb::Done { finish: finish.into(), usage: None, tool_calls: vec![] }))
    }

    /// 8 prompt tokens, scored from position 3 (a cache hit at depth 2); spans [1, 4) (partly
    /// below `first`) and [5, 8) (all scored).
    fn prompt_ev() -> pb::Event {
        let toks = ["<a>", "b", "c", "d", "e", "f", "g", "<|im_end|>"];
        ev(pb::event::Kind::PromptLogprobs(pb::PromptLogprobs {
            first: 3,
            tokens: toks.iter().enumerate().map(|(i, t)| tl(i as u32 + 10, t, 0.0, &[])).collect(),
            logprobs: vec![-0.5, -1.25, -2.0, -0.25, -0.125],
            spans: vec![pb::Span { message: 1, start: 1, end: 4 },
                        pb::Span { message: 3, start: 5, end: 8 },
                        pb::Span { message: 5, start: 0, end: 2 }],
        }))
    }

    const ALL: Show = Show { reasoning: true, usage: false, logprobs: true, prompt_logprobs: true };
    const NONE: Show = Show { reasoning: true, usage: false, logprobs: false, prompt_logprobs: false };

    #[test]
    fn lp_json_is_the_f32_decimal() {
        assert_eq!(lp_json(-2.3), json!(-2.3));
        assert_eq!(lp_json(-0.0001), json!(-0.0001));
        assert_eq!(lp_json(f32::NEG_INFINITY), Value::Null);
        assert_eq!(lp_json(f32::NAN), Value::Null);
    }

    #[test]
    fn acc_collects_logprobs() {
        let mut acc = Acc::default();
        for e in [
            ev(pb::event::Kind::Start(pb::Start { role: "assistant".into(), model: "m".into() })),
            token("hm", true, vec![tl(1, "hm", -0.5, &[(1, "hm", -0.5), (2, "ok", -1.5)])]),
            token("Hé", false, vec![tl(3, "H", -0.25, &[]), tl(4, "é", -1.0, &[])]),
            token("", false, vec![tl(5, "<|im_end|>", -0.125, &[])]),
            prompt_ev(),
            done("stop"),
        ] {
            acc.add(&e);
        }
        let out = acc.response("id", "m", 0, ALL);
        let c = &out["choices"][0];
        assert_eq!(c["message"]["content"], json!("Hé"));
        assert_eq!(c["logprobs"], json!({"content": [
            {"token": "hm", "logprob": -0.5, "bytes": [104, 109], "top_logprobs": [
                {"token": "hm", "logprob": -0.5, "bytes": [104, 109]},
                {"token": "ok", "logprob": -1.5, "bytes": [111, 107]}]},
            {"token": "H", "logprob": -0.25, "bytes": [72], "top_logprobs": []},
            {"token": "é", "logprob": -1.0, "bytes": [195, 169], "top_logprobs": []},
            {"token": "<|im_end|>", "logprob": -0.125, "bytes": "<|im_end|>".as_bytes(),
             "top_logprobs": []},
        ]}));
        assert_eq!(out["prompt_logprobs"], json!([
            null, null, null,
            {"token": "d", "token_id": 13, "logprob": -0.5},
            {"token": "e", "token_id": 14, "logprob": -1.25},
            {"token": "f", "token_id": 15, "logprob": -2.0},
            {"token": "g", "token_id": 16, "logprob": -0.25},
            {"token": "<|im_end|>", "token_id": 17, "logprob": -0.125},
        ]));
        assert_eq!(out["prompt_logprobs_messages"], json!([
            // [1, 4): positions 1, 2 below first -> 1 of 3 scored
            {"message": 1, "role": "assistant", "start": 1, "end": 4, "tokens": 3, "scored": 1,
             "logprob": -0.5},
            {"message": 3, "role": "assistant", "start": 5, "end": 8, "tokens": 3, "scored": 3,
             "logprob": -2.375},
            // nothing scored: a null sum, not 0
            {"message": 5, "role": "assistant", "start": 0, "end": 2, "tokens": 2, "scored": 0,
             "logprob": null},
        ]));
        // the journal drops the per-position array, keeps the sums
        let j = for_journal(out.clone());
        assert!(j.get("prompt_logprobs").is_none());
        assert_eq!(j["prompt_logprobs_positions"], json!(8));
        assert_eq!(j["prompt_logprobs_messages"], out["prompt_logprobs_messages"]);
        assert_eq!(j["choices"][0]["logprobs"], json!({"tokens": 4, "logprob": -1.875}));
        assert_eq!(j["choices"][0]["message"], out["choices"][0]["message"]);

        // not asked for: logprobs null (OpenAI), no prompt keys
        let plain = acc.response("id", "m", 0, NONE);
        assert_eq!(plain["choices"][0]["logprobs"], Value::Null);
        assert!(plain.get("prompt_logprobs").is_none() && plain.get("prompt_logprobs_messages").is_none());
        assert_eq!(for_journal(plain.clone()), plain);
    }

    #[test]
    fn prefill_only_answer() {
        // max_tokens 0: no Token events, PromptLogprobs then Done(length)
        let mut acc = Acc::default();
        acc.add(&prompt_ev());
        acc.add(&done("length"));
        let out = acc.response("id", "m", 0, ALL);
        assert_eq!(out["choices"][0]["finish_reason"], json!("length"));
        assert_eq!(out["choices"][0]["message"]["content"], json!(""));
        assert_eq!(out["choices"][0]["logprobs"], json!({"content": []}));
        assert_eq!(out["prompt_logprobs"].as_array().unwrap().len(), 8);
        // asked for, but the worker sent no event: null, not missing
        let mut old = Acc::default();
        old.add(&done("length"));
        let out = old.response("id", "m", 0, ALL);
        assert_eq!((&out["prompt_logprobs"], &out["prompt_logprobs_messages"]), (&Value::Null, &Value::Null));
    }

    fn sse(ev: pb::Event, show: Show) -> Vec<Value> {
        let mut ttft = true;
        event_chunks("id", "m", 0, ev, &mut ttft, &Instant::now(), show)
    }

    #[test]
    fn sse_chunks_carry_logprobs() {
        let lp = || vec![tl(3, "H", -0.25, &[(3, "H", -0.25)])];
        // text + lp: one chunk, logprobs beside the delta
        let c = sse(token("H", false, lp()), ALL);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0]["choices"][0]["delta"], json!({"content": "H"}));
        assert_eq!(c[0]["choices"][0]["logprobs"], json!({"content": [
            {"token": "H", "logprob": -0.25, "bytes": [72],
             "top_logprobs": [{"token": "H", "logprob": -0.25, "bytes": [72]}]}]}));
        assert_eq!(c[0]["choices"][0]["finish_reason"], Value::Null);
        // not asked for: the chunk as before, no logprobs key
        let c = sse(token("H", false, lp()), NONE);
        assert_eq!(c[0]["choices"][0], json!({"index": 0, "delta": {"content": "H"}, "finish_reason": null}));
        // lp with no text (the EOS, a held-back piece): emitted with an empty delta
        let c = sse(token("", false, lp()), ALL);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0]["choices"][0]["delta"], json!({}));
        assert_eq!(c[0]["choices"][0]["logprobs"]["content"][0]["token"], json!("H"));
        // hidden reasoning (include_reasoning false) still reports its tokens' logprobs
        let hidden = Show { reasoning: false, ..ALL };
        let c = sse(token("hm", true, lp()), hidden);
        assert_eq!(c[0]["choices"][0]["delta"], json!({}));
        assert!(sse(token("hm", true, lp()), Show { reasoning: false, ..NONE }).is_empty());
        let c = sse(token("hm", true, lp()), ALL);
        assert_eq!(c[0]["choices"][0]["delta"], json!({"reasoning": "hm", "reasoning_content": "hm"}));
        // the prompt event: its own chunk with no choices; nothing when not asked for
        let c = sse(prompt_ev(), ALL);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0]["choices"], json!([]));
        assert_eq!(c[0]["object"], json!("chat.completion.chunk"));
        assert_eq!(c[0]["prompt_logprobs"].as_array().unwrap().len(), 8);
        assert_eq!(c[0]["prompt_logprobs"][2], Value::Null);
        assert_eq!(c[0]["prompt_logprobs_messages"][1]["logprob"], json!(-2.375));
        assert!(sse(prompt_ev(), NONE).is_empty());
        // the finish chunk is unchanged
        let c = sse(done("length"), ALL);
        assert_eq!(c[0]["choices"][0], json!({"index": 0, "delta": {}, "finish_reason": "length"}));
    }
}
