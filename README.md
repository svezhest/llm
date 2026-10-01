# llm

Local LLM inference on Apple Silicon (MLX): a **Rust gateway** (OpenAI-compatible
HTTP/SSE) talking to thin **Python MLX workers** over **gRPC**, with Prometheus
metrics, Grafana dashboards, Loki logs, a request queue, and leveled logging.

```
clients ──HTTP/SSE (OpenAI)──► gateway (Rust) ──gRPC stream──► mlx worker(s) (Python)
                                 /metrics, queue, logs           mlx-vlm load + generate
        Prometheus ◄─scrape─ gateway + mac-exporter ; Grafana ◄ Prometheus + Loki ◄ Promtail
```

Why the split: Rust for everything (API, queue, metrics, logging, gRPC); Python
only where MLX must load. Process isolation, no PyO3. The `.proto` is the single
source of truth. Inference stays native on the host (Metal is not in Docker);
only the observability stack is containerized.

## Models (`weights/`)

| dir | model | quant | size | arch |
|---|---|---|---|---|
| `ornith10-9b-mlx-8bit` **(default)** | Ornith-1.0-9B | MLX **8-bit** g64 | ~10.4 GB | dense VL |
| `ornith15-9b-mlx-8bit` | Ornith-1.5-9B (`ornith-ai`) | MLX **8-bit** g64 | ~9.5 GB | dense, **text-only** (VL arch, vision tower stripped from the checkpoint; worker loads it non-strictly) |
| `qwen36-35b-a3b-mlx-ud3` | Qwen3.6-35B-A3B | MLX affine g64, **3-bit UD** (mixed: embed 6b, attn/gates 8b, shared-expert 4b, MoE experts 3b) | ~16 GB | VL MoE (256 experts, ~3B active) |
| `qwen38-27b-mlx-5bit` | Qwen3.8-27B | MLX affine g64, **5-bit** | ~18.1 GB | dense hybrid VL (64 layers; Gated DeltaNet + full attention every 4th, 4 KV heads, head_dim 256; native 256K ctx) |

All four are `model_type qwen3_5` (`qwen3_5_moe` for the 35B) in mlx-vlm — one
worker code path; only the env knobs differ.

Weights live in `weights/` (git-ignored). MLX loads from a plain path — no HF
cache needed. Download recipe for a slow, throttled link: multi-connection
`aria2c -c -x16 -s16` per shard with `--check-integrity --checksum=sha-256=<lfs.oid>`
(oids from `https://huggingface.co/api/models/<repo>/tree/main`), `HF_HUB_DISABLE_XET=1`,
never hard-kill (orphans partials). See `CLAUDE.md`. Scripted, resumable:
`scripts/hf_pull.sh lmstudio-community/Qwen3.8-27B-MLX-5bit weights/qwen38-27b-mlx-5bit`
(re-run the same line after any interruption; finished shards are skipped by checksum).

## Layout

- `src/proto/worker.proto` — gRPC contract.
- `src/gateway/` (Rust) — axum HTTP + tonic gRPC client + queue + metrics + tracing.
- `src/worker/`  (Python) — grpcio server, mlx-vlm; load + generate + reasoning budget.
- `src/exporter/` (Rust) — `powermetrics` → `/metrics` (GPU/ANE/mem/watts).
- `deploy/` — docker-compose (prometheus/grafana/loki/promtail) + launchd plists.
- `Cargo.toml` — Rust workspace (members `src/gateway`, `src/exporter`).
- `pyproject.toml` — root tooling/download env; `src/worker/` is its own uv project.

## Run (dev)

The **shipped default is Ornith 9B — 128K context, fp16 KV** (light: ~16 GB
resident, no wired-limit tuning needed). The 35B and 27B are the heavier alternatives below.

1. **Worker** — run the 9B (override the built-in default 35B path via `QWEN_MODEL`):
   ```zsh
   cd src/worker
   uv run --group proto bash gen_proto.sh    # once (and after a proto change): codegen worker_pb2*.py
   QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 \
     uv run python worker.py                  # serves 127.0.0.1:50051
   # optional, greedy requests only: +DFlash speculative decoding (~1.4–1.6x decode)
   #   QWEN_DRAFT=../../weights/qwen35-9b-dflash
   ```

2. **Gateway** (after the worker logs "model loaded"):
   ```zsh
   cargo build --release
   RUST_LOG=info ./target/release/qwen-gateway   # 0.0.0.0:8080
   ```

3. **Call it** (OpenAI-compatible):
   ```zsh
   curl -N http://localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{
     "model": "ornith15-9b-mlx-8bit", "stream": true,
     "messages": [{"role":"user","content":"Привет! Кто ты?"}]
   }'
   ```

**Heavier alternative — Qwen 35B-A3B** (the worker's built-in default path). Needs
the Metal wired-limit bump first (once per boot, resets on reboot — see
*Context & memory*), and runs q4 KV:
```zsh
sudo sysctl iogpu.wired_limit_mb=28672    # 28 GB cap (default ~21.8 GB)
QWEN_WIRED_LIMIT_GB=27 QWEN_PREFILL_STEP=512 uv run python worker.py   # q4 KV (default)
```

**Qwen3.8-27B (5-bit)** — the strongest dense option; ~24 GB resident at 128K, so
it needs the same wired-limit bump as the 35B, but runs **q8 KV**:
```zsh
sudo sysctl iogpu.wired_limit_mb=28672                 # once per boot
cd src/worker
QWEN_MODEL=../../weights/qwen38-27b-mlx-5bit QWEN_KV_BITS=8 \
  QWEN_WIRED_LIMIT_GB=27 QWEN_PREFILL_STEP=512 uv run python worker.py
# gateway (after "model loaded"):
RUST_LOG=info ./target/release/qwen-gateway
```
The model id (`/v1/models`, `"model"` in every response) is the worker's weights
dir name — `ornith15-9b-mlx-8bit`, `qwen38-27b-mlx-5bit`, … — reported by the worker
itself (Health, and the Start of each generation), so it always names what is
loaded; `/v1/models` is 503 while the worker is down. The request's `"model"` must
be that id or a `GATEWAY_SERVED_MODEL_NAMES` alias, anything else is a 404 (see
**model** below).

**API behaviour = vLLM.** The reference is `vllm serve` (v0.30) with the card's
recipe (`--tool-call-parser qwen3_xml --reasoning-parser qwen3`); where this gateway
cannot do what vLLM does it says so with a 400 instead of ignoring the field.
- **Sampling** (vLLM order, float32): penalties → temperature → min_p → top_k → top_p
  (over the top_k survivors) → draw; temperature 0 = argmax, (0, 0.01) is raised to
  0.01. `presence_penalty` / `frequency_penalty` are additive over all generated
  tokens, `repetition_penalty` multiplicative over prompt + output. Ranges as vLLM
  (400 otherwise): temperature [0, 2], top_p (0, 1] (`top_p: 0` is a 400), top_k ≥ -1,
  min_p [0, 1], presence/frequency [-2, 2], repetition > 0, max_tokens ≥ 1. A request
  that omits a field gets the worker env default (`GATEWAY_OPENAI_DEFAULTS=1`: vLLM's
  defaults — T 1, top_p 1, top_k 0, min_p 0, penalties 0 / 1). `seed` makes a sampled
  request reproducible (`-1` = none).
- **Length**: `max_tokens` / `max_completion_tokens` (the latter wins) beyond what
  the context leaves is a 400 with vLLM's message; omitted = the rest of the context
  (`GATEWAY_DEFAULT_MAX_TOKENS` caps it if set).
- **stop** (string or array): the earliest match wins, over the whole output —
  reasoning included, as vLLM; a stop split across tokens is caught and never emitted.
- **Reasoning**: no setting = the chat template's default, which for these models is
  *thinking on*. `chat_template_kwargs: {"enable_thinking": false}`, `reasoning_effort`
  (`"none"` = off, anything else = on; also passed to the template) or this gateway's
  top-level `enable_thinking` set it; `GATEWAY_THINKING=on|off` forces it for every
  request (and logs when it overrides one). Budget: `thinking_token_budget` (vLLM;
  -1 = unlimited) or `thinking_budget` (0 = unlimited). The answer is split like
  vLLM's `qwen3` parser: reasoning → `reasoning` (and `reasoning_content`, the older
  name), no tags in either; no `</think>` at all → everything is reasoning and
  `content` is null; `include_reasoning: false` drops it; a `<tool_call>` inside the
  reasoning ends it (the call goes to the answer); with thinking off a stray `</think>`
  is dropped from `content`. `usage.completion_tokens_details.reasoning_tokens` counts
  the generated tokens before the token that ends the reasoning. History: an assistant
  message's `reasoning` / `reasoning_content` goes back into the template.
- **chat_template_kwargs**: `chat_template` / `tokenize` there are a 400 (vLLM's
  message); keys the template does not use are ignored (logged), as vLLM;
  `add_generation_prompt` / `continue_final_message` there are overridden by the
  request's own fields (vLLM's `merge_kwargs`). A template error is a 400.
- **Roles**: `developer` becomes `system` (this template does not know it), and all
  system messages are merged into one at the front, `"\n\n"`-joined — vLLM's
  renderers/hf.py.
- **Tool calls**: parsed as vLLM's `qwen3_xml`: values lose exactly one leading and
  one trailing newline; types come only from the tool's schema (a `"false"` string
  parameter stays a string; a parameter not in the schema stays a string); content is
  the text before the first call (null if empty), text after a call is dropped (logged);
  a `<tool_call>` with no `<function=` is no call: it is dropped up to its `</tool_call>`
  (logged) and the text before and after it stays `content`; a call cut off by `max_tokens` is still a call and `finish_reason`
  is `tool_calls`. Without `tools` a `<tool_call>` in the answer is cut from `content`
  together with everything after it (vLLM's parser runs anyway; no calls returned).
  `tool_choice`: `"none"` keeps the tools in the prompt and returns no calls;
  `"required"` / a named function / a `strict` tool are grammar-constrained as
  xgrammar's `qwen_3_coder` structural tag (vLLM with VLLM_ENFORCE_STRICT_TOOL_CALLING):
  declared names, parameters in schema order, required present, values by schema, no
  keys beyond the schema's. `"required"`: the answer opens with a call — no text
  before it (`content` null), text and more calls may follow; with
  `parallel_tool_calls: false` exactly one call and nothing after it. A named function:
  exactly that one call, nothing before or after; it finishes with `"stop"`, as in
  vLLM. `"auto"`: `parallel_tool_calls: false` = at most one call. Reasoning stays free
  in all modes (the grammar starts after it). Call ids are `chatcmpl-tool-<16 hex>`
  derived from the prompt, seed and answer: the same request gives the same ids.
- **Streaming**: a first chunk `{"role": "assistant", "content": ""}`, then deltas;
  `stream_options.include_usage` adds the final usage chunk (`choices: []`). A worker
  lost mid-answer is an error frame + `[DONE]` (counted as an error), or a 502 without
  streaming — never a truncated 200.
- **400 instead of silence**: `n` > 1, `logit_bias`, `logprobs` / `top_logprobs`, legacy
  `functions` / `function_call`, content parts other than text / image, images on a
  text-only checkpoint, roles the template rejects, `response_format.json_schema`
  without `name`, a `tool_choice` object whose `type` is not `"function"`, and every
  other vLLM field set to a non-default value (`echo`, `use_beam_search`,
  `truncate_prompt_tokens`, `structured_outputs`, `documents`, …). Fields vLLM does
  not know are ignored, as in vLLM, but logged.
- **Also as vLLM**: `request_id` (the response id becomes `chatcmpl-<request_id>`),
  `min_tokens` (EOS / `stop_token_ids` / stop strings held off until then),
  `stop_token_ids`, `ignore_eos`, `bad_words`, `allowed_token_ids`,
  `include_stop_str_in_output`, `continue_final_message`, `add_generation_prompt:
  false`, `stream_options: {}` without streaming, a tool without `"type"`, numbers sent
  as strings; `skip_special_tokens` / `spaces_between_special_tokens` / `cache_salt`
  are accepted and have no effect (vLLM's qwen3 parser turns special-token skipping
  off itself; the prompt cache never changes an answer). The same checks run before queueing (worker `Check` RPC), so streams get the
  400 too. Errors have vLLM's shape `{"error": {"message", "type", "param", "code"}}`.
- **model**: the loaded model's id (its weights dir name) or a name in
  `GATEWAY_SERVED_MODEL_NAMES` (vLLM's `--served-model-name`); anything else is a
  404 `The model ... does not exist.`, as vLLM.

The worker env's own default is argmax + presence penalty 1.5 — the Qwen card's
*non-thinking* setting, and **wrong for reasoning and benchmarks**: it suppresses
repeated tokens inside long `<think>` traces. For evals run the card's
thinking-mode sampler:
```zsh
QWEN_PRESENCE_PENALTY=0 QWEN_TEMPERATURE=1.0 QWEN_TOP_P=0.95 QWEN_TOP_K=20 QWEN_MIN_P=0 \
  QWEN_MODEL=... uv run python worker.py
```
(Qwen3.8 card: thinking = T 1.0 / top_p 0.95 / top_k 20 / min_p 0 / presence 0;
instruct = T 0.7 / top_p 0.80 / top_k 20 / presence 1.5.)

Vision (VL checkpoints): put an `image_url` part in the message content.

**Structured output** (`response_format`): `{"type": "json_schema", "json_schema":
{"schema": {...}}}` decodes under the schema's grammar (llguidance token masks),
`{"type": "json_object"}` under "any JSON object", `"text"` as before. Required keys
are present; keys beyond `properties` are **not** allowed unless the schema says so
(`additionalProperties` / `unevaluatedProperties` / `patternProperties`), as vLLM
compiles schemas (xgrammar `strict_mode=True`); types, enums, `const`, lengths and
item counts hold. `strict` is implied. EOS is allowed only where the JSON is complete; a `max_tokens` cut still
ends mid-JSON (`finish_reason: "length"`). Thinking: the grammar starts after
`</think>`. Whitespace is free, as in vLLM (pretty-printed JSON is fine);
`QWEN_JSON_MAX_WS=N` caps a whitespace run at N characters if a model starts
padding. With `tools` and `tool_choice: "auto"` the answer is either the schema JSON
or constrained tool calls (vLLM would make tool calls impossible here); the model
is not told the format — without an instruction in the prompt it tends to keep
calling tools (that is its choice, not a grammar fault). An unknown type or a schema
that does not compile is a 400, streaming included. Grammar requests skip DFlash
(logged). Cost: ~0.2 ms of CPU per token, no step-ahead in the decode loop — 9B
31 → 30 tok/s.

Quick CLI (no gateway): `uv run python chat.py "Привет!" --model weights/ornith10-9b-mlx-8bit`.

## Prompt cache (`QWEN_PROMPT_CACHE=1`, default off)

Multi-slot **prefix cache** for text-only prompts. Each entry is the cache
state after the prefill of `prompt[:-1]`. Full-attention KV can be cut to any
prefix by slicing; the gated-delta-net recurrent state cannot be rewound, so it
is **checkpointed during prefill** (~52 MB each on the 9B) at the slab end
closest below: every `QWEN_PROMPT_CACHE_STRIDE` (2048) tokens, message
boundaries (end of the system/tools block, start of the last two messages and
of the reply) and the point where the prompt diverged from the closest cached
one. Slab ends depend only on the tokens before them (a message start at least
`QWEN_PREFILL_MIN_STEP` past the previous end, else a full step), with or
without the cache: the kernels round differently for another slab shape
(split-K in thin matmuls, SDPA key blocks), and cache-dependent cuts used to
flip greedy answers. A new prompt resumes from the deepest checkpoint that is
also its own slab end inside its longest common prefix with any entry, so the
output is bitwise the same as a cold run whatever the cache holds. A different
conversation over the same system prompt, an edited / regenerated turn and
interleaved sessions all hit, not only a strict continuation. Entries are
LRU-evicted past `QWEN_PROMPT_CACHE_GB` (3; the newest always stays), at most
`QWEN_PROMPT_CACHE_CKPTS` (12) checkpoints each; an entry that is a prefix of a
newer one is dropped. mlx-vlm's own trim is never used (broken for quantized KV
tuples). Vision requests never touch the cache.

Measured on Ornith-1.5 9B (fp16 KV, 7.5K-token system prompt, temperature 0),
TTFT cold → cached, output byte-identical to the cold run in all five:

| scenario | cold | cached |
|---|---|---|
| other conversation, same system prompt | 14.4 s | 0.20 s |
| back to conversation A after B | 15.6 s | 0.31 s |
| back to B | 16.5 s | 0.24 s |
| edited last user message | 17.1 s | 0.22 s |

(the old single-slot snapshot missed the first, second and fourth). No cost on
a cold prefill (±noise over 3 A/B runs). Through the gateway, 4.3K prompt: 9.4 s
→ 0.32 s. First-token logprobs checked bitwise equal (9K-token prompt, q4 KV
past 5K): cold, `QWEN_PROMPT_CACHE=0`, exact repeat, next turn, same system
prompt only. Logs: `prompt cache HIT n/m tokens reused (common
prefix k)` / `MISS` / `store: n tokens, c checkpoints, MB`.

## Context & memory

Context is capped at **128K tokens** (`MAX_CONTEXT=131072`, prompt + generation;
override with `QWEN_MAX_CONTEXT`).

**KV cache precision is per-model** (`QWEN_KV_BITS`, default 4), because the right
value tracks the *weight* precision and the memory budget:

| model | weights | KV (`QWEN_KV_BITS`) | KV @ 128K | resident @ 128K | why |
|---|---|---|---|---|---|
| Qwen 35B-A3B | 3-bit, ~18 GB | **q4** (default) | ~2.6 GB | ~21 GB (tight) | 4-bit KV is the balanced match for 3-bit weights; q8 KV would blow the 32 GB budget |
| **Ornith 9B** | 8-bit, ~10 GB | **fp16 (`0`)** | ~4 GB | **~16 GB** | headroom to spare, and an unquantized cache is *faster* than q8 here: at 32K prefill 72 vs 82 s, decode 31.9 vs 27.3 tok/s (quantized-cache prefill dequantizes per slab; the q8 decode SDPA kernel is slower than fp16) |
| **Qwen3.8-27B** | 5-bit, ~18.1 GB | **q8** | ~3.9 GB | **~24 GB** | 5-bit weights deserve q8 KV; fits under the 27 GB wired limit with the bump |

**KV math for the hybrid models.** Only the full-attention layers (every 4th) hold
a growing KV cache; the Gated DeltaNet layers keep a small constant state. Per token
at q8: `2 (K,V) × full-attn layers × 4 KV heads × head_dim 256 × 1 byte` → 9B
(8 layers) ≈ 16 KB, 27B (16 layers) ≈ 32 KB. So the 27B is ~3.9 GB at 128K and
~6.1 GB at 200K; resident ≈ weights 18.1 + KV + ~2 GB prefill activations ≈ 24 GB
(128K) / 26 GB (200K). **To run the 27B at 200K**: `QWEN_MAX_CONTEXT=204800` plus
the wired bump (`sudo sysctl iogpu.wired_limit_mb=28672`, `QWEN_WIRED_LIMIT_GB=27`,
`QWEN_PREFILL_STEP=512`) — ~1 GB of slack under the limit, so keep prefill chunked.
The 9B runs q8 at 128K with no `sudo sysctl` needed (256K ≈ 4 GB KV also fits).

Oversized prompts are rejected with **HTTP 400 `context_length_exceeded`** before
any allocation (no OOM). Image tokens are counted for real (a 4K image ≈ 10k
tokens), so vision prompts are validated accurately, not undercounted; the
prepared inputs (image preprocessing included) are then fed straight into
generation, so images are processed once, not twice.

**Fitting the 35B / 27B long context on 32 GB.** Weights are ~16–18 GB; the default
Metal wired limit (~21.8 GB) leaves almost nothing for KV + prefill activations,
so big prompts OOM. Three things keep it alive:
- **`sudo sysctl iogpu.wired_limit_mb=28672`** raises the kernel working-set cap
  to 28 GB (resets on reboot). `QWEN_WIRED_LIMIT_GB` then pins MLX's wired limit
  (clamped to the cap — set it *below* the sysctl value, e.g. 27).
- **Chunked prefill** (`QWEN_PREFILL_STEP`, default 2048) bounds the prefill
  activation peak so a 50K-token prompt doesn't spike memory in one matmul. The
  slab shrinks automatically as the context grows (`slab × cache_offset ≤
  QWEN_PREFILL_BUDGET`, default 2048 × 32K, floor `QWEN_PREFILL_MIN_STEP` 256):
  a fixed 2048 slab at 120K peaked at 26.7 GB on the 9B, the adaptive one stays
  near the 512-slab footprint there. Set `QWEN_PREFILL_STEP=512` for the
  35B/27B at long context if memory is tight.
- **Buffer pool** (`QWEN_CLEAR_CACHE`, default `1`; `QWEN_CACHE_LIMIT_GB`,
  default `3`). Freed Metal buffers sit in MLX's allocator free-list, still
  resident and counted against the wired working set; the list is size-bucketed
  and by default unbounded, so request N's prefill activations stay parked
  while request N+1 allocates fresh ones on top — the second-request OOM.
  `QWEN_CLEAR_CACHE=1` drops the pool after every generation (verified:
  56K-token prompt × 2 back-to-back, no OOM). `QWEN_CACHE_LIMIT_GB` is applied
  at load via `mx.set_cache_limit`, so `QWEN_CLEAR_CACHE=0` keeps a bounded
  pool across requests instead. Measured on the 9B, 9K-token prompt × 2:
  clear=1 → footprint 11 GB, TTFT 31.3 / 29.8 s; clear=0 → footprint 14 GB
  (pool pinned at the 3 GB limit), TTFT 30.7 / 29.2 s. Peak MLX memory
  12.0 / 12.8 GB in both modes. The pool buys ~2% on TTFT for 3 GB of
  footprint, hence the default stays `1`; the tight 35B/27B setups must keep it.

The **9B-8bit** model needs none of this — ~10 GB resident leaves plenty of
headroom on 32 GB.

See **CLAUDE.md** for design decisions, implementation notes, and gotchas.

## Observability
```zsh
cd deploy && QWEN_LOG_DIR=../logs docker compose up -d
```
- Grafana: http://localhost:3000 (anonymous admin) → dashboard **qwen-server**.
- Prometheus: http://localhost:9090 ; gateway metrics: http://localhost:8080/metrics.
- mac-exporter (GPU/ANE/watts) needs root: `sudo ./target/release/qwen-mac-exporter`.
- Set `LOG_FORMAT=json` on the gateway so Promtail can label by level.

## Resilience (launchd)
Edit paths in `deploy/launchd/*.plist`, then load them (worker + gateway as user
agents, exporter as a root daemon). `watchdog.sh` kickstarts the gateway if
`/health` goes dark. See comments inside each plist.

## Env knobs
| var | where | default |
|---|---|---|
| `QWEN_MODEL` | worker | `weights/qwen36-35b-a3b-mlx-ud3` (repo-relative) |
| `WORKER_ADDR` | gateway | `127.0.0.1:50051` |
| `GATEWAY_ADDR` | gateway | `0.0.0.0:8080` |
| `MAX_QUEUE` | gateway | `32` |
| `GATEWAY_DEFAULT_MAX_TOKENS` | gateway | `0` = the rest of the context (vLLM) — generation limit when the request has neither `max_tokens` nor `max_completion_tokens` |
| `RUST_LOG` / `LOG_FORMAT` | gateway | `info` / pretty |
| `GATEWAY_JOURNAL` | gateway | unset — path of a JSONL journal: one line per request, `{ts, id, request, response}`; `request` is the body as sent, `response` the chat.completion (streamed ones assembled, `"stream": true`) |
| `GATEWAY_OPENAI_DEFAULTS` | gateway | unset — `1`: omitted sampling fields get the OpenAI / vLLM defaults (temperature 1, top_p 1, top_k 0, min_p 0, presence / frequency 0, repetition 1) instead of the worker's env (greedy, presence penalty 1.5) |
| `GATEWAY_THINKING` | gateway | unset (as the request says, else the template default: thinking) — `on` / `off` forces it for every request, logged when it overrides one |
| `GATEWAY_SERVED_MODEL_NAMES` | gateway | unset — comma-separated extra names `model` may carry (vLLM `--served-model-name`); the loaded id is always accepted, anything else is a 404 |
| `LOG_LEVEL` | worker | `INFO` |
| `QWEN_MAX_CONTEXT` | worker | `131072` |
| `QWEN_KV_BITS` | worker | `4` (35B); `0` = fp16 cache for the 9B, `8` for the 27B |
| `QWEN_FP16` | worker | `1` — cast bf16 params to fp16 (M1/M2 have no native bf16: prefill ×1.45; KL vs bf16 0.0003). `0` = bf16 |
| `QWEN_DRAFT` / `QWEN_DRAFT_BLOCK` | worker | unset / `0` (adaptive) — DFlash drafter dir for speculative decoding (greedy, text-only, fp16 KV) |
| `QWEN_WIRED_LIMIT_GB` | worker | `0` (off; clamped to kernel cap) |
| `QWEN_PREFILL_STEP` / `QWEN_PREFILL_BUDGET` / `QWEN_PREFILL_MIN_STEP` | worker | `2048` / `67108864` (2048×32K) / `256` — slab shrinks so slab×context ≤ budget |
| `QWEN_TEMPERATURE` / `QWEN_TOP_P` / `QWEN_TOP_K` / `QWEN_MIN_P` | worker | `0.0` (argmax) / `1.0` / `0` / `0.0` (off) |
| `QWEN_PRESENCE_PENALTY` | worker | `1.5` (set `0` for reasoning/benchmarks) |
| `QWEN_FREQUENCY_PENALTY` / `QWEN_REPETITION_PENALTY` | worker | `0` / `1.0` |
| `QWEN_JSON_MAX_WS` | worker | `0` (off) — cap on a whitespace run in a `response_format` answer |
| `QWEN_PROMPT_CACHE` | worker | `0` (set `1`: multi-slot prefix cache with recurrent-state checkpoints) |
| `QWEN_PROMPT_CACHE_GB` / `_STRIDE` / `_CKPTS` | worker | `3` / `2048` / `12` — LRU budget, checkpoint stride, checkpoints per entry |
| `QWEN_DRAFT_SAMPLING` | worker | `0` (set `1`: speculative path also for temperature/presence-penalty requests, exact sample-match; code ×1.1–1.2, prose ×0.8) |
| `QWEN_CLEAR_CACHE` / `QWEN_CACHE_LIMIT_GB` | worker | `1` / `3` (drop MLX buffer pool after each request / pool cap) |
| `MAC_EXPORTER_ADDR` | exporter | `0.0.0.0:9101` |
