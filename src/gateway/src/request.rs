//! OpenAI chat request -> worker Request, checked the way vLLM's /v1/chat/completions
//! checks it (v0.30, the reference server for these models): the same 400s for
//! out-of-range values, fields vLLM does not know ignored but logged, and every vLLM
//! feature this gateway does not implement refused with a 400 instead of accepted and
//! silently ignored. Template, context-length and grammar checks are the worker's
//! (Check RPC).

use std::sync::OnceLock;

use serde::Deserialize;
use serde_json::{json, Map, Value};
use tracing::{info, warn};

use crate::pb;

pub struct BadRequest {
    pub msg: String,
    pub param: Option<&'static str>,
}

fn bad<T>(msg: impl Into<String>, param: &'static str) -> Result<T, BadRequest> {
    Err(BadRequest { msg: msg.into(), param: Some(param) })
}

#[derive(Deserialize)]
pub struct ChatRequest {
    pub model: Option<String>,
    pub messages: Vec<InMessage>,
    #[serde(default)]
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,
    // numbers may come as strings, as pydantic (vLLM) accepts them
    #[serde(default, deserialize_with = "lax_i64")]
    pub max_tokens: Option<i64>,
    #[serde(default, deserialize_with = "lax_i64")]
    pub max_completion_tokens: Option<i64>,
    #[serde(default, deserialize_with = "lax_f64")]
    pub temperature: Option<f64>,
    #[serde(default, deserialize_with = "lax_f64")]
    pub top_p: Option<f64>,
    #[serde(default, deserialize_with = "lax_i64")]
    pub top_k: Option<i64>,
    #[serde(default, deserialize_with = "lax_f64")]
    pub min_p: Option<f64>,
    #[serde(default, deserialize_with = "lax_f64")]
    pub presence_penalty: Option<f64>,
    #[serde(default, deserialize_with = "lax_f64")]
    pub frequency_penalty: Option<f64>,
    #[serde(default, deserialize_with = "lax_f64")]
    pub repetition_penalty: Option<f64>,
    // OpenAI allows `stop` as a single string or an array of strings.
    #[serde(default, deserialize_with = "string_or_vec")]
    pub stop: Vec<String>,
    #[serde(default, deserialize_with = "lax_i64")]
    pub seed: Option<i64>,
    #[serde(default, deserialize_with = "lax_i64")]
    pub n: Option<i64>,
    pub logprobs: Option<bool>,
    #[serde(default, deserialize_with = "lax_i64")]
    pub top_logprobs: Option<i64>,
    pub logit_bias: Option<Map<String, Value>>,
    pub tools: Option<Vec<Value>>,
    pub tool_choice: Option<Value>,
    pub parallel_tool_calls: Option<bool>,
    pub response_format: Option<Value>,
    pub reasoning_effort: Option<String>,
    pub chat_template_kwargs: Option<Map<String, Value>>,
    pub include_reasoning: Option<bool>,
    #[serde(default, deserialize_with = "lax_i64")]
    pub thinking_token_budget: Option<i64>,
    // vLLM extras implemented in the worker
    pub request_id: Option<String>,
    #[serde(default, deserialize_with = "lax_i64")]
    pub min_tokens: Option<i64>,
    #[serde(default)]
    pub stop_token_ids: Option<Vec<i64>>,
    #[serde(default)]
    pub ignore_eos: bool,
    #[serde(default)]
    pub bad_words: Option<Vec<String>>,
    #[serde(default)]
    pub allowed_token_ids: Option<Vec<i64>>,
    #[serde(default)]
    pub include_stop_str_in_output: bool,
    pub add_generation_prompt: Option<bool>,
    #[serde(default)]
    pub continue_final_message: bool,
    // Extensions of this gateway (not vLLM): `enable_thinking` = chat_template_kwargs'
    // one, `thinking_budget` = thinking_token_budget with 0 meaning unlimited.
    pub enable_thinking: Option<bool>,
    pub thinking_budget: Option<u32>,
}

#[derive(Deserialize, Default)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
    #[serde(default)]
    pub continuous_usage_stats: bool,
}

#[derive(Deserialize)]
pub struct InMessage {
    pub role: String,
    // string OR array of OpenAI content parts. May be null for assistant tool-call turns.
    #[serde(default)]
    pub content: Value,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Value>,
    // Earlier reasoning of an assistant turn; vLLM reads `reasoning`, and renames the
    // older `reasoning_content` to it.
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

fn lax_f64<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<f64>, D::Error> {
    match Option::<Value>::deserialize(d)? {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => Ok(n.as_f64()),
        Some(Value::String(s)) => s.trim().parse().map(Some)
            .map_err(|_| serde::de::Error::custom(format!("not a number: {s:?}"))),
        Some(v) => Err(serde::de::Error::custom(format!("not a number: {v}"))),
    }
}

fn lax_i64<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<i64>, D::Error> {
    match lax_f64(d)? {
        None => Ok(None),
        Some(f) if f.fract() == 0.0 && f.abs() < 9.2e18 => Ok(Some(f as i64)),
        Some(f) => Err(serde::de::Error::custom(format!("not an integer: {f}"))),
    }
}

fn string_or_vec<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum V { One(String), Many(Vec<String>) }
    Ok(match Option::<V>::deserialize(d)? {
        Some(V::One(s)) => vec![s],
        Some(V::Many(v)) => v,
        None => vec![],
    })
}

/// Fields of vLLM's ChatCompletionRequest this gateway honours (or, like `user`,
/// that have no effect in vLLM either), plus its own extensions.
const HANDLED: &[&str] = &[
    "messages", "model", "stream", "stream_options", "max_tokens", "max_completion_tokens",
    "temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty",
    "repetition_penalty", "stop", "seed", "n", "logprobs", "top_logprobs", "logit_bias",
    "tools", "tool_choice", "parallel_tool_calls", "response_format", "reasoning_effort",
    "chat_template_kwargs", "include_reasoning", "thinking_token_budget", "user", "priority",
    "session_id", "enable_thinking", "thinking_budget", "request_id", "min_tokens",
    "stop_token_ids", "ignore_eos", "bad_words", "allowed_token_ids",
    "include_stop_str_in_output", "add_generation_prompt", "continue_final_message",
    // no effect here: skip_special_tokens / spaces_between_special_tokens (vLLM's qwen3
    // parser forces skip_special_tokens off itself), cache_salt (the prompt cache never
    // changes an answer and is not shared between tenants)
    "skip_special_tokens", "spaces_between_special_tokens", "cache_salt",
];

/// vLLM request fields not implemented here, with vLLM's default: anything else -> 400.
fn unsupported_default(name: &str) -> Option<Value> {
    Some(match name {
        "use_beam_search" | "echo" | "add_special_tokens" | "return_token_offsets"
        | "return_assistant_tokens_mask" => json!(false),
        "watermarking" => json!(true),
        "length_penalty" | "stream_interval" => json!(1),
        "routed_experts_prompt_start" => json!(0),
        "truncate_prompt_tokens" | "truncation_side" | "prompt_logprobs" | "logprob_token_ids"
        | "documents" | "chat_template" | "media_io_kwargs"
        | "mm_processor_kwargs" | "structured_outputs"
        | "return_tokens_as_token_ids" | "return_token_ids" | "return_prompt_text"
        | "kv_transfer_params" | "ec_transfer_params" | "vllm_xargs"
        | "repetition_detection" => Value::Null,
        _ => return None,
    })
}

fn is_default(v: &Value, d: &Value) -> bool {
    match (v, d) {
        (Value::Null, _) => true,
        (Value::Number(a), Value::Number(b)) => a.as_f64() == b.as_f64(),
        (Value::Array(a), Value::Array(b)) => a.is_empty() && b.is_empty(),
        _ => v == d,
    }
}

/// Top-level keys: legacy function calling and unimplemented vLLM features -> 400;
/// keys vLLM itself does not know are ignored (as vLLM does), with a warning.
pub fn check_fields(body: &Map<String, Value>) -> Result<(), BadRequest> {
    let mut ignored = vec![];
    for (k, v) in body {
        if HANDLED.contains(&k.as_str()) {
            continue;
        }
        if k == "functions" || k == "function_call" {
            return bad(format!("`{k}` (legacy function calling) is not supported; use `tools` \
                                and `tool_choice`"), "functions");
        }
        match unsupported_default(k) {
            Some(d) if is_default(v, &d) => {}
            Some(_) => return bad(format!("`{k}` is not supported by this server"), "request"),
            None => ignored.push(k.as_str()),
        }
    }
    if !ignored.is_empty() {
        warn!(fields = ?ignored, "request fields ignored (unknown to vLLM's chat request too)");
    }
    Ok(())
}

/// vLLM's SamplingParams._verify_args and request validators, same messages.
pub fn validate(r: &ChatRequest) -> Result<(), BadRequest> {
    match r.n {
        Some(n) if n < 1 => return bad(format!("n must be at least 1, got {n}."), "n"),
        Some(n) if n > 1 => return bad("n > 1 is not supported by this server", "n"),
        _ => {}
    }
    for (name, v) in [("presence_penalty", r.presence_penalty), ("frequency_penalty", r.frequency_penalty)] {
        if let Some(v) = v {
            if !(-2.0..=2.0).contains(&v) {
                return bad(format!("{name} must be in [-2, 2], got {v}."), "sampling");
            }
        }
    }
    if let Some(v) = r.repetition_penalty {
        if !v.is_finite() || v <= 0.0 {
            return bad(format!("repetition_penalty must be greater than zero, got {v}."),
                       "repetition_penalty");
        }
    }
    if let Some(t) = r.temperature {
        if !t.is_finite() || !(0.0..=2.0).contains(&t) {
            return bad(format!("temperature must be in [0, 2], got {t}."), "temperature");
        }
    }
    if let Some(p) = r.top_p {
        if !(p > 0.0 && p <= 1.0) {
            return bad(format!("top_p must be in (0, 1], got {p}."), "top_p");
        }
    }
    if let Some(k) = r.top_k {
        if k < -1 {
            return bad(format!("top_k must be 0 (disable), or at least 1, got {k}."), "top_k");
        }
    }
    if let Some(m) = r.min_p {
        if !(0.0..=1.0).contains(&m) {
            return bad(format!("min_p must be in [0, 1], got {m}."), "min_p");
        }
    }
    if let Some(m) = r.max_completion_tokens.or(r.max_tokens) {
        if m < 1 {
            return bad(format!("max_tokens must be at least 1, got {m}."), "max_tokens");
        }
    }
    if r.stop.iter().any(String::is_empty) {
        return bad("stop cannot contain an empty string.", "stop");
    }
    if r.logit_bias.as_ref().is_some_and(|b| !b.is_empty()) {
        return bad("logit_bias is not supported by this server", "logit_bias");
    }
    if let Some(t) = r.top_logprobs {
        if t != 0 && r.logprobs != Some(true) {
            return bad("when using `top_logprobs`, `logprobs` must be set to true.", "top_logprobs");
        }
    }
    if r.logprobs == Some(true) {
        return bad("logprobs are not supported by this server", "logprobs");
    }
    if let Some(m) = r.min_tokens {
        if m < 0 {
            return bad(format!("min_tokens must be greater than or equal to 0, got {m}."), "min_tokens");
        }
        if let Some(x) = r.max_completion_tokens.or(r.max_tokens) {
            if m > x {
                return bad(format!("min_tokens must be less than or equal to max_tokens={x}, got {m}."),
                           "min_tokens");
            }
        }
    }
    if r.stop_token_ids.iter().flatten().chain(r.allowed_token_ids.iter().flatten()).any(|&t| t < 0) {
        return bad("token ids must be non-negative", "stop_token_ids");
    }
    if r.allowed_token_ids.as_ref().is_some_and(Vec::is_empty) {
        return bad("allowed_token_ids is not allowed to be empty.", "allowed_token_ids");
    }
    if r.bad_words.iter().flatten().any(String::is_empty) {
        return bad("bad_words cannot contain an empty string.", "bad_words");
    }
    if r.continue_final_message && r.add_generation_prompt == Some(true) {
        return bad("Cannot set both `continue_final_message` and `add_generation_prompt` to True.",
                   "continue_final_message");
    }
    // vLLM: an empty stream_options object is falsy and passes without stream
    if let Some(o) = r.stream_options.as_ref().filter(|o| o.include_usage || o.continuous_usage_stats) {
        if !r.stream {
            return bad("Stream options can only be defined when `stream=True`.", "stream_options");
        }
        if o.continuous_usage_stats {
            return bad("stream_options.continuous_usage_stats is not supported by this server",
                       "stream_options");
        }
    }
    if let Some(e) = &r.reasoning_effort {
        if !["none", "minimal", "low", "medium", "high", "xhigh", "max"].contains(&e.as_str()) {
            return bad(format!("Invalid reasoning_effort: {e}"), "reasoning_effort");
        }
    }
    if let Some(b) = r.thinking_token_budget {
        if b < -1 {
            return bad(format!("thinking_token_budget must be -1 or >= 0, got {b}."),
                       "thinking_token_budget");
        }
    }
    response_format_json(r.response_format.as_ref())?;
    check_tools(r)
}

/// vLLM's check_tool_usage.
fn check_tools(r: &ChatRequest) -> Result<(), BadRequest> {
    if r.tools.as_ref().is_some_and(Vec::is_empty) {
        return bad("`tools` must not be an empty array. Either provide at least one tool or \
                    omit the field entirely.", "tools");
    }
    for t in r.tools.iter().flatten() {
        let name = t.get("function").and_then(|f| f.get("name")).and_then(Value::as_str);
        // `type` defaults to "function" in vLLM's ChatCompletionToolsParam
        let kind = t.get("type").map_or(Some("function"), Value::as_str);
        if kind != Some("function") || name.is_none() {
            return bad("each tool must be {\"type\": \"function\", \"function\": {\"name\": ...}}",
                       "tools");
        }
    }
    let usage = "Correct usage: `{\"type\": \"function\", \"function\": {\"name\": \"my_function\"}}`";
    match &r.tool_choice {
        None | Some(Value::Null) => Ok(()),
        Some(Value::String(s)) if s == "none" => Ok(()),
        Some(_) if r.tools.is_none() => {
            bad("When using `tool_choice`, `tools` must be set.", "tool_choice")
        }
        Some(Value::String(s)) if s == "auto" || s == "required" => Ok(()),
        Some(Value::Object(o)) => {
            if o.get("type").and_then(Value::as_str) != Some("function") {
                return bad(format!("Invalid value for `tool_choice`: type must be \"function\". {usage}"),
                           "tool_choice.type");
            }
            let Some(f) = o.get("function").and_then(Value::as_object) else {
                return bad(format!("Invalid value for `function` in `tool_choice`! {usage}"),
                           "tool_choice.function");
            };
            let Some(name) = f.get("name").and_then(Value::as_str).filter(|n| !n.is_empty()) else {
                return bad(format!("Expected field `name` in `function` in `tool_choice`! {usage}"),
                           "tool_choice.function.name");
            };
            let known = r.tools.iter().flatten().any(|t| {
                t.get("function").and_then(|f| f.get("name")).and_then(Value::as_str) == Some(name)
            });
            if known {
                Ok(())
            } else {
                bad("The tool specified in `tool_choice` does not match any of the specified `tools`",
                    "tool_choice")
            }
        }
        Some(v) => bad(format!("Invalid value for `tool_choice`: {v}! Only named tools, \"none\", \
                                \"auto\" or \"required\" are supported."), "tool_choice"),
    }
}

/// `response_format` as the JSON passed to the worker: empty for text (or absent).
/// Whether a schema compiles is the worker's call (Check).
fn response_format_json(rf: Option<&Value>) -> Result<String, BadRequest> {
    let rf = match rf {
        None | Some(Value::Null) => return Ok(String::new()),
        Some(v) => v,
    };
    match rf.get("type").and_then(Value::as_str) {
        Some("text") => Ok(String::new()),
        Some("json_object") => Ok(json!({"type": "json_object"}).to_string()),
        Some("json_schema") => match rf.get("json_schema") {
            None | Some(Value::Null) => bad("When response_format type is 'json_schema', the \
                                             'json_schema' field must be provided.", "response_format"),
            Some(js) if !js.get("name").is_some_and(Value::is_string) => {
                bad("response_format.json_schema.name is required", "response_format")
            }
            Some(js) => match js.get("schema") {
                Some(Value::Object(_)) => Ok(rf.to_string()),
                _ => bad("response_format.json_schema.schema must be a JSON Schema object",
                         "response_format"),
            },
        },
        Some(t) => bad(format!("unsupported response_format type '{t}' (supported: text, \
                                json_object, json_schema)"), "response_format"),
        None => bad("response_format must be an object with a string `type`", "response_format"),
    }
}

/// Content (string or parts) -> (text, image urls). Text parts are concatenated as the
/// chat template does; a part type this server cannot take is a 400, as in vLLM.
fn flatten_content(v: &Value) -> Result<(String, Vec<String>), BadRequest> {
    match v {
        Value::Null => Ok((String::new(), vec![])),
        Value::String(s) => Ok((s.clone(), vec![])),
        Value::Array(parts) => {
            let mut text = String::new();
            let mut images = vec![];
            for p in parts {
                match p.get("type").and_then(Value::as_str) {
                    Some("text" | "input_text" | "output_text") => {
                        if let Some(t) = p.get("text").and_then(Value::as_str) {
                            text.push_str(t);
                        }
                    }
                    Some("refusal") => {
                        if let Some(t) = p.get("refusal").and_then(Value::as_str) {
                            text.push_str(t);
                        }
                    }
                    Some(kind @ ("image_url" | "input_image")) => {
                        // {image_url: {url: ...}} or image_url: "..."
                        let url = match p.get(kind) {
                            Some(Value::String(s)) => Some(s.clone()),
                            Some(o) => o.get("url").and_then(Value::as_str).map(String::from),
                            None => None,
                        };
                        match url {
                            Some(u) => images.push(u),
                            None => return bad(format!("{kind} part without a url"), "messages"),
                        }
                    }
                    other => {
                        return bad(format!("Unsupported chat content part type: {:?}",
                                           other.unwrap_or("<none>")), "messages")
                    }
                }
            }
            Ok((text, images))
        }
        _ => bad("message content must be a string, an array of parts or null", "messages"),
    }
}

/// GATEWAY_THINKING=on|off forces the template's enable_thinking for every request
/// (unset: as the request says, else the template default — the Qwen3.5 template thinks).
fn forced_thinking() -> Option<bool> {
    static F: OnceLock<Option<bool>> = OnceLock::new();
    *F.get_or_init(|| match std::env::var("GATEWAY_THINKING").as_deref() {
        Ok("on") => Some(true),
        Ok("off") => Some(false),
        _ => None,
    })
}

/// Generation limit when the request sets neither max_tokens nor max_completion_tokens
/// (GATEWAY_DEFAULT_MAX_TOKENS; 0 / unset = the rest of the context, as vLLM).
fn default_max_tokens() -> u32 {
    static LIMIT: OnceLock<u32> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("GATEWAY_DEFAULT_MAX_TOKENS").ok().and_then(|s| s.parse().ok()).unwrap_or(0)
    })
}

/// GATEWAY_OPENAI_DEFAULTS=1: omitted sampling fields take the OpenAI / vLLM defaults
/// (temperature 1, top_p 1, top_k 0, min_p 0, penalties 0 / 1) instead of the worker's
/// env (greedy, presence penalty 1.5).
fn openai_defaults() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("GATEWAY_OPENAI_DEFAULTS").map(|s| s == "1").unwrap_or(false))
}

/// The template kwargs vLLM would render with: the request's chat_template_kwargs,
/// `reasoning_effort` (also enable_thinking = effort != "none" unless set), this
/// gateway's top-level `enable_thinking`, then GATEWAY_THINKING over all of them.
fn template_kwargs(r: &ChatRequest, id: &str) -> Map<String, Value> {
    let mut kw = r.chat_template_kwargs.clone().unwrap_or_default();
    // vLLM merges its own add_generation_prompt / continue_final_message over these
    // (merge_kwargs: the request fields win); they travel as Params fields here
    for k in ["add_generation_prompt", "continue_final_message"] {
        if kw.remove(k).is_some() {
            info!(id, key = k, "chat_template_kwargs entry overridden by the request field (as vLLM)");
        }
    }
    if let Some(e) = &r.reasoning_effort {
        kw.insert("reasoning_effort".into(), json!(e));
        if !kw.contains_key("enable_thinking") {
            kw.insert("enable_thinking".into(), json!(e != "none"));
        }
    }
    if let Some(t) = r.enable_thinking {
        kw.entry("enable_thinking").or_insert(json!(t));
    }
    if let Some(f) = forced_thinking() {
        if let Some(asked) = kw.get("enable_thinking").and_then(Value::as_bool) {
            if asked != f {
                info!(id, asked, forced = f, "GATEWAY_THINKING overrides the request's enable_thinking");
            }
        }
        kw.insert("enable_thinking".into(), json!(f));
    }
    kw
}

pub fn to_pb(id: &str, r: &ChatRequest) -> Result<pb::Request, BadRequest> {
    let mut images = vec![];
    let mut messages = vec![];
    for m in &r.messages {
        let (text, imgs) = flatten_content(&m.content)?;
        images.extend(imgs);
        if let Some(calls) = &m.tool_calls {
            let ok = calls.as_array().is_some_and(|a| a.iter().all(|c| {
                c.get("function").is_some_and(Value::is_object)
                    && c.get("type").and_then(Value::as_str).unwrap_or("function") == "function"
            }));
            if !ok {
                return bad("chat completions only support assistant tool_calls of type \
                            'function'.", "tool_calls");
            }
        }
        messages.push(pb::Message {
            role: m.role.clone(),
            content: text,
            name: m.name.clone().unwrap_or_default(),
            tool_call_id: m.tool_call_id.clone().unwrap_or_default(),
            tool_calls_json: m.tool_calls.as_ref().filter(|v| !v.is_null())
                .map(|v| v.to_string()).unwrap_or_default(),
            reasoning: m.reasoning.clone().or_else(|| m.reasoning_content.clone()),
        });
    }
    let tools_json = r.tools.as_ref().map(|v| Value::from(v.clone()).to_string()).unwrap_or_default();
    let tool_choice = match &r.tool_choice {
        Some(Value::String(s)) => s.clone(),
        None | Some(Value::Null) => String::new(),
        Some(v) => v.to_string(),
    };
    let kwargs = template_kwargs(r, id);
    let budget = match (r.thinking_token_budget, r.thinking_budget) {
        (Some(b), _) if b >= 0 => Some(b as u32),
        (Some(_), _) => None,                  // -1 = unlimited
        (None, Some(b)) if b > 0 => Some(b),
        _ => None,
    };
    let openai = openai_defaults();
    let d = |v: Option<f64>, def: f32| v.map(|x| x as f32).or(openai.then_some(def));
    Ok(pb::Request {
        id: id.to_string(),
        messages,
        images,
        params: Some(pb::Params {
            max_tokens: r.max_completion_tokens.or(r.max_tokens).map(|m| m.min(u32::MAX as i64) as u32),
            default_max_tokens: default_max_tokens(),
            thinking_budget: budget,
            seed: r.seed.filter(|&s| s != -1).map(|s| s as u64),
            stop: r.stop.clone(),
            tools_json,
            tool_choice,
            parallel_tool_calls: r.parallel_tool_calls,
            temperature: d(r.temperature, 1.0),
            top_p: d(r.top_p, 1.0),
            top_k: r.top_k.map(|k| k.max(0) as u32).or(openai.then_some(0)),
            min_p: d(r.min_p, 0.0),
            presence_penalty: d(r.presence_penalty, 0.0),
            frequency_penalty: d(r.frequency_penalty, 0.0),
            repetition_penalty: d(r.repetition_penalty, 1.0),
            response_format_json: response_format_json(r.response_format.as_ref())?,
            chat_template_kwargs_json: if kwargs.is_empty() { String::new() }
                                       else { Value::Object(kwargs).to_string() },
            min_tokens: r.min_tokens.unwrap_or(0) as u32,
            stop_token_ids: r.stop_token_ids.iter().flatten().map(|&t| t as u32).collect(),
            ignore_eos: r.ignore_eos,
            bad_words: r.bad_words.clone().unwrap_or_default(),
            allowed_token_ids: r.allowed_token_ids.iter().flatten().map(|&t| t as u32).collect(),
            include_stop_str_in_output: r.include_stop_str_in_output,
            // continue_final_message implies no generation prompt
            add_generation_prompt: r.add_generation_prompt.or(r.continue_final_message.then_some(false)),
            continue_final_message: r.continue_final_message,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rf(s: &str) -> Result<String, String> {
        let v: Value = serde_json::from_str(s).unwrap();
        response_format_json(Some(&v)).map_err(|e| e.msg)
    }

    #[test]
    fn response_format_types() {
        assert_eq!(response_format_json(None).ok(), Some(String::new()));
        assert_eq!(rf("null"), Ok(String::new()));
        assert_eq!(rf(r#"{"type": "text"}"#), Ok(String::new()));
        assert_eq!(rf(r#"{"type": "json_object"}"#), Ok(r#"{"type":"json_object"}"#.to_string()));
        // property order (z before a) survives the round trip
        let s = r#"{"type":"json_schema","json_schema":{"name":"x","strict":true,"schema":{"type":"object","properties":{"z":{},"a":{}}}}}"#;
        assert_eq!(rf(s), Ok(s.to_string()));
    }

    #[test]
    fn response_format_rejected() {
        for s in [
            r#"{"type": "xml"}"#,
            r#"{"type": "json_schema"}"#,
            r#"{"type": "json_schema", "json_schema": {"schema": "nope"}}"#,
            r#"{"json_schema": {"schema": {}}}"#,
            r#""json_object""#,
        ] {
            assert!(rf(s).is_err(), "{s}");
        }
    }

    fn req(extra: Value) -> Result<ChatRequest, String> {
        let mut body = json!({"messages": [{"role": "user", "content": "hi"}]});
        body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        check_fields(body.as_object().unwrap()).map_err(|e| e.msg)?;
        let r: ChatRequest = serde_json::from_value(body).map_err(|e| e.to_string())?;
        validate(&r).map_err(|e| e.msg)?;
        Ok(r)
    }

    #[test]
    fn vllm_validation() {
        for bad in [
            json!({"top_p": 0}), json!({"temperature": 2.5}), json!({"max_tokens": 0}),
            json!({"n": 2}), json!({"logit_bias": {"1": 5}}), json!({"logprobs": true}),
            json!({"top_k": -2}), json!({"presence_penalty": 3}), json!({"repetition_penalty": 0}),
            json!({"stop": [""]}), json!({"functions": []}), json!({"echo": true}),
            json!({"stream_options": {"include_usage": true}}), json!({"tools": []}),
            json!({"response_format": {"type": "json_schema", "json_schema": {"schema": {}}}}),
            json!({"tools": [{"type": "function", "function": {"name": "f"}}],
                   "tool_choice": {"type": "x", "function": {"name": "f"}}}),
            json!({"temperature": "hot"}),
            json!({"tool_choice": "required"}), json!({"reasoning_effort": "ultra"}),
            json!({"tools": [{"type": "function", "function": {"name": "f"}}],
                   "tool_choice": {"type": "function", "function": {"name": "g"}}}),
        ] {
            assert!(req(bad.clone()).is_err(), "{bad}");
        }
        for good in [
            json!({"top_p": 1, "temperature": 0, "n": 1, "logprobs": false, "top_logprobs": 0}),
            json!({"ignore_eos": true, "skip_special_tokens": false, "something_else": 1}),
            json!({"temperature": "0.5", "max_tokens": "10", "stream_options": {}}),
            json!({"tools": [{"function": {"name": "f"}}], "min_tokens": 3, "bad_words": ["x"]}),
            json!({"tool_choice": "none"}), json!({"seed": -1}),
            json!({"tools": [{"type": "function", "function": {"name": "f"}}],
                   "tool_choice": {"type": "function", "function": {"name": "f"}}}),
        ] {
            assert!(req(good.clone()).is_ok(), "{good}");
        }
    }

    #[test]
    fn thinking_kwargs() {
        let r = req(json!({"reasoning_effort": "none"})).ok().unwrap();
        assert_eq!(template_kwargs(&r, "x").get("enable_thinking"), Some(&json!(false)));
        let r = req(json!({"reasoning_effort": "high",
                           "chat_template_kwargs": {"enable_thinking": false}})).ok().unwrap();
        assert_eq!(template_kwargs(&r, "x").get("enable_thinking"), Some(&json!(false)));
        let r = req(json!({})).ok().unwrap();
        assert!(template_kwargs(&r, "x").is_empty());
    }
}
