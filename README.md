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

**Status: a research project, published as is.** Built and measured on one machine
(M1 Max, 32 GB) against pinned library versions (mlx 0.32.3, mlx-vlm 0.7.6, whose
classes the worker patches at run time); no support, no stability promises. The
interesting parts are probably the prompt cache for hybrid (Gated DeltaNet) models,
the M1 matmul kernels (`pfqmm.py`, `mrqmm.py`), DFlash speculative decoding with the
prompt cache, and `docs/perf/` — every change's measurements, raw data and the dead
ends. Much of the code and the reports were written with AI coding agents (Claude
Code); `CLAUDE.md` is the project memory they worked from.

## Models (`weights/`)

| dir | model | quant | size | arch |
|---|---|---|---|---|
| `ornith10-9b-mlx-8bit` **(default)** | Ornith-1.0-9B | MLX **8-bit** g64 | ~10.4 GB | dense VL |
| `ornith15-9b-mlx-8bit` | Ornith-1.5-9B (`ornith-ai`) | MLX **8-bit** g64 | ~9.5 GB | dense, **text-only** (VL arch, vision tower stripped from the checkpoint; mlx-vlm 0.7 loads it without a tower, image requests → 400) |
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
- `src/worker/`  (Python) — grpcio server, mlx 0.32.3 + mlx-vlm 0.7.6 (no mlx-lm); load + generate + reasoning budget.
  `worker.py` is the entry point and a facade (`import worker` = every name of the package);
  the code is in `qwen_worker/`:

  | module | what |
  |---|---|
  | `config` | `QWEN_*` env knobs and constants |
  | `chat` | tool-call parsing, images, chat messages, special-token-safe tokenization |
  | `sampler` | vLLM-order sampler, penalties, grouped per-row draws |
  | `pcache` | prompt cache: KV segments, entries, the prefix cache |
  | `sizing` | prefill slab policy, memory budget arithmetic |
  | `stream` | detokenizer, stop strings, request controls, reasoning split |
  | `grammar` | structured output / tool-call grammars (llguidance) |
  | `nosync` | `QWEN_DEBUG_NO_SYNC` |
  | `rows` | batched decode over per-row caches, stacked GDN state |
  | `kernels` | mlx-vlm patches: fused SDPA, DFlash verify, fused GDN in-projection |
  | `logprobs` | log-probabilities, assistant spans |
  | `request` | request / row objects |
  | `scheduler` + `sched_*` | `_Scheduler` and its mixins: admit, prefill, spec (DFlash), store, cascade, decode |
  | `engine` | `Engine`: load, checks, `generate` |
  | `server` | gRPC servicer, `serve`, `main` |
- `src/exporter/` (Rust) — `powermetrics` → `/metrics` (GPU/ANE/mem/watts).
- `deploy/` — docker-compose (prometheus/grafana/loki/promtail) + launchd plists.
- `Cargo.toml` — Rust workspace (members `src/gateway`, `src/exporter`).
- `pyproject.toml` — root tooling/download env; `src/worker/` is its own uv project.

## Run (dev)

The **shipped default is Ornith 9B — 128K context, fp16 KV** (light: ~16 GB
resident, no wired-limit tuning needed). The 35B and 27B are the heavier alternatives below.

1. **Worker** — run the 9B (override the built-in default 35B path via `QWEN_MODEL`).
   It batches concurrent requests (continuous batching, up to `QWEN_MAX_BATCH=16`
   rows, memory permitting); `QWEN_MAX_BATCH=1` serves one request at a time, as
   before:
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

   Worker tests (CPU, tiny random models, ~15 s): `cd src/worker && uv run --group dev pytest -q --ignore tests/test_mrqmm.py` (the Metal parts of `tests/test_pfqmm.py` run with `PFQMM_GPU=1`)
   (`tests/test_mrqmm.py` runs Metal kernels on the GPU; it is skipped without Metal).

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
  min_p [0, 1], presence/frequency [-2, 2], repetition > 0, max_tokens ≥ 1 (0 only with
  `prompt_logprobs`, see **Log-probabilities**). A request
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
  request (and logs when it overrides one). Later wins: `chat_template_kwargs` <
  top-level `enable_thinking` < `GATEWAY_THINKING`; `reasoning_effort` decides only
  when `chat_template_kwargs` has no `enable_thinking`. Budget: `thinking_token_budget` (vLLM;
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
- **400 instead of silence**: `n` > 1, `logit_bias`, `top_logprobs` > 20, `prompt_logprobs`
  other than true/false/0/1, legacy
  `functions` / `function_call`, content parts other than text / image, images on a
  text-only checkpoint, roles the template rejects, `response_format.json_schema`
  without `name`, a `tool_choice` object whose `type` is not `"function"`, and every
  other vLLM field set to a non-default value (`echo`, `use_beam_search`,
  `truncate_prompt_tokens`, `structured_outputs`, `documents`, …). Fields vLLM does
  not know are ignored, as in vLLM, but logged.
- **Also as vLLM**: `request_id` (the response id becomes `chatcmpl-<request_id>`),
  `min_tokens` (EOS / `stop_token_ids` / stop strings held off until then),
  `stop_token_ids`, `ignore_eos`, `bad_words`, `allowed_token_ids` (caps vLLM does not
  have, 400 beyond: `stop` 64 strings, `stop_token_ids` 1024, `bad_words` 1024,
  `allowed_token_ids` 2^18; ids must fit uint32),
  `include_stop_str_in_output`, `continue_final_message`, `add_generation_prompt:
  false`, `stream_options: {}` without streaming, a tool without `"type"`, numbers sent
  as strings; `skip_special_tokens` / `spaces_between_special_tokens` / `cache_salt`
  are accepted and have no effect (vLLM's qwen3 parser turns special-token skipping
  off itself; the prompt cache never changes an answer). The same checks run before the request waits for a generation slot (worker `Check` RPC), so streams get the
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

**Log-probabilities** (`logprobs`, `top_logprobs`, `prompt_logprobs`, `prompt_cache`;
design: `docs/perf/2026-10-21/design/logprobs.md`). All values are *raw*: fp32 `logit −
logsumexp` of the model's logits before penalties, temperature, grammar or token masks
(vLLM's `logprobs_mode=raw_logprobs`), so they do not depend on the sampler settings.

| field | type | meaning | 400 |
|---|---|---|---|
| `logprobs` | bool | log p of every generated token: reasoning, answer and the final EOS / stop token (listed although its text is not in `content`) | — |
| `top_logprobs` | int 0..20 | + the k most likely tokens per step, descending; needs `logprobs: true` | `top_logprobs must be between 0 and 20, got N.` / ``when using `top_logprobs`, `logprobs` must be set to true.`` |
| `prompt_logprobs` | `true`/`false`/`0`/`1`/null | teacher forcing: log p of every prompt token given the tokens before it, computed in the prefill. As vLLM's int: `0` and `1` are **on** (vLLM's int is the top-k per position beside the token's own log p; we return only the own one), `false`/null off | `prompt_logprobs: only true/false/0/1 are supported (top-k over the prompt is not)`; with images: the worker's 400 (text-only) |
| `max_tokens` / `max_completion_tokens` | int | `0` = prefill only, allowed only with `prompt_logprobs` on: no decode step, `finish_reason: "length"`, `completion_tokens: 0`, `content: ""`; the prompt is stored in the prompt cache as after a generation | `max_tokens must be at least 1, got 0.` (vLLM's) without `prompt_logprobs` |
| `prompt_cache` | `true`/`false`/`"off"`/`"read"` (extension) | `true`/absent: normal. `false` or `"off"`: no prompt-cache lookup (cold prefill: every position scored), the entry is still stored. `"read"`: lookup as normal, nothing stored | `prompt_cache must be true, false, "off" or "read"` |

`prompt_logprobs` also works with `stream: true` (vLLM refuses that) and costs an extra
`lm_head` over every prefill row (~+10 % prefill FLOPs), only for requests that ask.
**Attribution workflow** (one recorded trace, many edited variants): send the baseline
with `prompt_cache: false` (cold, every position scored, and it is stored), then every
variant with `prompt_cache: "read"` — each resumes from the baseline's entry up to its
first edit and stores nothing, so thousands of variants neither evict the baseline nor
churn the cache with their tails. A variant's unscored positions (before its hit
depth) are the baseline's values there (causality).

Response (non-streaming):
```json
{"choices": [{"index": 0, "message": {...}, "finish_reason": "stop",
              "logprobs": {"content": [
                {"token": "Hi", "logprob": -0.031, "bytes": [72, 105],
                 "top_logprobs": [{"token": "Hi", "logprob": -0.031, "bytes": [72, 105]},
                                  {"token": "Hello", "logprob": -3.6, "bytes": [72, 101, 108, 108, 111]}]},
                {"token": "<|im_end|>", "logprob": -0.002, "bytes": [...], "top_logprobs": []}]}}],
 "prompt_logprobs": [null, null, {"token": "user", "token_id": 846, "logprob": -2.3}, ...],
 "prompt_logprobs_messages": [{"message": 3, "role": "assistant", "start": 1200, "end": 1500,
                               "tokens": 300, "scored": 300, "logprob": -45.2}],
 "usage": {...}}
```
- `choices[0].logprobs` is OpenAI's shape; `null` when `logprobs` was not asked for.
  `token` is the tokenizer's `decode([id])` of one token and `bytes` the UTF-8 of that
  string, as vLLM: lossy for a token holding part of a multi-byte character (CJK, emoji
  split across tokens) — it shows U+FFFD (`[239, 191, 189]`), so joining `bytes` does
  not rebuild such characters.
- `prompt_logprobs` (only when asked for) has one entry per prompt token (the rendered
  chat template, as `usage.prompt_tokens` counts it); `null` where a position was not
  scored, i.e. every position ≤ `usage.prompt_tokens_details.cached_tokens`: position 0
  on a cold prefill (nothing before it), up to the hit depth on a prompt-cache hit (the
  cache keeps no logits, not even of its last position). Positions past that are a
  cold prefill's values (causality, same slabs: bit-exact) — except for a hit through a
  stored reply (`QWEN_PROMPT_CACHE_REPLY=1`), whose KV came from decode kernels (the
  batched-decode numerics class; `QWEN_PROMPT_CACHE_REPLY=0` keeps hits bit-exact).
  The shape differs from vLLM's (a `{token_id: {logprob, rank, decoded_token}}` dict
  per position).
- `prompt_logprobs_messages`: one entry per `assistant` message of the request
  (`message` = its index in `messages`). `[start, end)` = the token positions the model
  *generated* for it: after `<|im_start|>assistant\n` and the template's `<think>`
  pre-open — with empty reasoning the whole `<think>\n\n</think>\n\n` header is
  skipped (the prompt of a thinking-off generation), with reasoning only `<think>\n`
  (reasoning + `\n</think>\n\n` + content were generated) — through its `<|im_end|>`
  inclusive. `tokens = end − start`; `scored` = how many of those positions have a value
  (fewer than `tokens` when a cache hit reaches into the span); `logprob` = the sum
  over the scored ones (`null` when none is). The rule excludes only the template's
  pre-open, so a span can hold fewer tokens than the model generated: a thinking-on
  reply whose reasoning is sent back empty renders as the thinking-off header (the
  model's own `\n</think>\n\n` is not scored), and templates that drop the think block
  of earlier turns (27B / 35B: assistant turns before the last user message) leave
  only the content, scored in a context the model never saw. The per-token
  `prompt_logprobs` list is authoritative.

Streaming: a chunk's `choices[0].logprobs.content` holds the tokens read since the
previous chunk — **not aligned** with the chunk's text delta (stop-string hold-back,
multi-byte characters, the reasoning split); a chunk may carry logprobs with an empty
`delta: {}` (the final EOS, a held-back piece, reasoning hidden by `include_reasoning:
false`). Concatenated over the stream they are the `content` list above. The prompt
log-probabilities come as one extra chunk with `"choices": []` and the top-level
`prompt_logprobs` + `prompt_logprobs_messages`, just before the finish chunk. The
journal (`GATEWAY_JOURNAL`) keeps `prompt_logprobs_messages` but replaces the
per-position array by `prompt_logprobs_positions` (its length) and
`choices[0].logprobs` by `{"tokens": n, "logprob": sum}`.

Numerics: for an answer generated at temperature 0 (presence penalty 0), the
teacher-forced sum over its span (the same messages + that answer as the last
assistant message, `add_generation_prompt: false`, `max_tokens: 0`) equals the sum of
its generation `logprobs`, EOS included, up to kernel rounding: the prefill scores
positions with M-row matmuls (pfqmm / qmm, fused SDPA) where decode used 1-row qmv,
so expect small per-token differences, not bit equality. Under batched decode with
B ≥ 3 rows the generation-side values are in the mrqmm numerics class (top-1 ≥ 99.7 %,
KL ~1e-5 vs solo) and depend on the batch; `QWEN_MAX_BATCH=1` makes them
reproducible. At temperature 0 the drawn token is usually `top_logprobs[0]` but not
always: the greedy argmax runs in the model dtype (fp16) and `top_logprobs` is sorted
in fp32, so on a near-tie the drawn token can be `top_logprobs[1]` (or two equal
values come in the other order) — do not assert it. Requests with neither flag run
exactly the same graph as before.

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
interleaved sessions all hit, not only a strict continuation. An entry that is
a prefix of a newer one (the previous turn of a conversation) is absorbed by it:
its checkpoints move over and it is dropped, so a conversation holds one entry,
not one per turn (on quantized KV an unquantized entry is kept under a quantized
newer one: the only resume point below `quantized_kv_start`). Its end state moves
over only where it is a slab end of the new prompt, so an exact repeat of the
older prompt resumes at its deepest checkpoint instead. At most
`QWEN_PROMPT_CACHE_CKPTS` (12) checkpoints per entry: the end of the system block
and the last three message starts are never thinned. An entry that absorbed an older
one (a **chain**: a conversation or an agent episode continuing turn by turn) keeps
fewer: the end of the system block, the last `QWEN_PROMPT_CACHE_CHAIN_MARKS` (3)
message starts, the stride checkpoints within its last `QWEN_PROMPT_CACHE_CHAIN_TAIL`
(4096) tokens (regenerate / an edited last turn), those inside the first user message
(found by its role token, system message or not; a shared document: another
conversation over it can diverge anywhere inside, and
then resumes near the divergence instead of re-prefilling the document; thinned by
the cap) and its end state; the next turn resumes at those, so the strides and
boundaries deeper in the history were dead weight (AppWorld agent, 9B: an 11K-token
entry carried ~10 states of 49 MB beside 355 MB of KV, 932 MB in all; with the chain
policy ~4–5 states: ~670 MB at 11K, ~1.65 GB at 41K instead of 2.0 GB, simulated
through the scheduler). The document strides cost nothing for AppWorld (its first
user message is ~300 tokens) and 2–4 more states (100–200 MB) per entry when the first
message is an 8K document. Repeating an older turn still hits, at its last message
start for the previous turn, else at the end of the system block (or a stride of the
first user message); `QWEN_PROMPT_CACHE_CHAIN_MARKS=-1`
turns the chain policy off. **Fresh entries** (nothing absorbed: a new conversation, a MISS)
keep the cap-12 policy below `QWEN_PROMPT_CACHE_CHAIN_FRESH_TOKENS` (16384 tokens; `0`:
the cap policy at any length), so a sibling conversation over a short shared prompt still finds deep strides;
from there on they take the chain policy too, plus the checkpoint at the divergence point
from the closest other entry (a cancelled attempt of the same prompt does not count) and the
strides inside the system block (thinned by the cap, like the first user message's: a new
session whose long system prompt differs near its end resumes at the last one). AppWorld:
an entry stored after a MISS on a 50K history kept 12 states; simulated 2.28 → 2.04 GB on
the 9B, 12 → 7 states. **A cancelled prefill keeps
its prefix**: a text request cancelled mid-prefill (a client timeout, the Cancel RPC) after
at least one slab, or failing outside the forward, stores the entry for the prefix it
prefilled — up to its last completed slab end, so the hit is bit-identical to a cold
run — through the same store (budget checks, absorption, chain policy; logged as a
normal store line + `(cancelled at pos/len)`). A retry of the same prompt resumes there
instead of re-prefilling it (the 2026-10-17 AppWorld run re-prefilled a 114K prompt twice
for 8 minutes after client timeouts). The retry absorbs it without counting as a chain.
Image prompts store nothing when cancelled (DFlash prefills do, as plain ones). `QWEN_PROMPT_CACHE_GB` (3) is a soft cap: past it entries
go at store (the newest stays, alone if need be) in the **eviction order** below. Entries **share their KV**: an entry is a chain of immutable
segments, a store copies only the tokens past its resume point out of the request's
cache and reuses the older entry's segments for the rest, so 16 conversations over
one 8K system prompt hold it once (~1.4 GB for all 16 on the 9B by the design
estimate, not yet measured; every entry used to carry its own copy: 6 fit in 3.2 GB,
measured); a hit concatenates the pieces once
into the buffer its prefill would have built anyway (bit-identical). The cache's
live bytes (every buffer once) count against the batch memory budget: a store
evicts entries down to what the running rows leave; a new entry that does not
fit there even alone is not kept, and then the cache stays as it was (nothing
absorbed, evicted or copied; counted as an eviction). An admission or a growing row
evicts entries when only the cache is in the way (an admission spares the entry
it would resume from at least one slab deep, and evicts nothing when that would not
get it in; a growing row never evicts the entry a running prefill resumed from).
When that entry alone keeps the next request out, the request waits for a running
row to end only if the hit saves at least `QWEN_PROMPT_CACHE_KEEP_TOKENS` (8192)
tokens of prefill, and at most `QWEN_PROMPT_CACHE_KEEP_WAIT_S` (10 s); then (or at
once for a shallower hit) the entry is evicted and the request starts cold.
Eviction order (every eviction above): entries a waiting request would resume from
(a slab deep or more; the next `QWEN_MAX_BATCH` in line and an admitted request whose
prefill has not started yet) go last; when a request gives up its hit, that entry goes
first; stale entries go first, least recently used first: not used within the last
round (more than twice as many text-request lookups + stores since they were stored or
hit as there are entries: a finished conversation or episode; vision requests do not
count, they would age live episodes) or, unless a waiting request would resume from
it, idle for `QWEN_PROMPT_CACHE_STALE_S` (300 s: an idle cache); when every entry was used within the
round (N conversations / agent episodes taking turns), the most recently used goes,
since its next turn is the furthest away. Plain LRU evicts exactly the episode whose
turn is next, and its full re-prefill stores an entry that evicts the next one in line:
8 episodes round-robin in a cap that holds 7 missed 28 of 32 turns with LRU and 5 with
this order (CPU test); an event simulation of the AppWorld load (8 episodes, 5–8 GB of
cache) served 13–60 % more requests than LRU at 5–8 GB (equal at 10 GB), and evicting
the largest entry instead of the most recent one (also tried) 3–10 % fewer than this
order. A
failed store only logs; the request goes on. mlx-vlm's own trim is never used
(broken for quantized KV tuples). Vision requests never touch the cache (nor tick its
round clock).

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
prefix k)` / `MISS` / `store: n tokens, c checkpoints, MB (e entries, X MB total)
reused=r gap=g absorbed=a pieces=p states=S MB` (gap: common-prefix tokens it could not resume
from; pieces: segments in its KV chain; states: its recurrent-state checkpoints + end state).
Responses carry `usage.prompt_tokens_details.cached_tokens` (vLLM's field);
the gateway counts `qwen_gateway_cached_tokens_total`; the worker's Health reports
entries, live bytes, hits, misses and evictions (a new entry not kept at store counts
as one; gateway debug log on change).

**Reply KV (`QWEN_PROMPT_CACHE_REPLY`, default on since the 2026-10-17 GPU gate).** When a
text request ends with
`stop` or `length`, its entry is extended by the tokens the model consumed while it
decoded and by the recurrent state after them, which becomes the new end state:

* the consumed tokens are the prompt's last token and every generated token fed back,
  including the EOS the step-ahead already fed (a grammar / sync row never feeds it);
  a `length` row fed its last generated token too, so its entry is prompt + completion;
* the next turn (prompt + reply + a new message: chats, agent loops) then resumes
  *after* the reply, not at the last message start;
* only the reply's KV is copied (the prompt part is the prompt entry's segments), and
  the copy goes through the same budget checks as any store;
* the copies are queued without waiting, between two decode steps;
* the extended entry keeps its prompt entry's recency in the eviction order (a
  conversation that just finished is not "most recently used").

On the AppWorld traces the template re-renders every reply to the ids the model
generated (1344/1344 pairs). That saves 20.8 % of the prefilled tokens, 148 per call:
the reply, the assistant header, and after a short tool output the stretch back to the
previous slab end (`docs/perf/2026-10-17/design/reply_kv.md`).

Nothing extra is stored for:

* cancelled requests and errors. Stop strings usually end the row as a cancel, but the
  handler's cancel can arrive after the row already finished with `stop`/`length`. That
  reply is then stored with the tokens past the string; the next turn falls back, which
  is harmless;
* replies that generated a reasoning section: thinking on, i.e. generation started
  inside the template's open `<think>` (the same split the worker's `_GenState` makes
  between reasoning and content). Clients rarely send the reasoning back, so the copy
  would be wasted, and the prompt-end state (a regenerate's full hit) would be lost.
  With thinking off a `<think>` id the model emits inside its content is content: the
  reply is stored;
* vision rows (DFlash rows are stored: the tokens their target cache took in the rounds, the
  entry's end state carrying the drafter's context, `QWEN_DRAFT_TAIL`; not when the stop token
  was an accepted draft with more accepted drafts after it — the cache ran past the stop);
* rows still in a cascade group when they end (their KV is in the group's stack; logged);
* rows whose prompt entry was evicted while they decoded.

Other effects:

* A regenerate of the same prompt resumes at the last message start (~10 tokens of
  prefill), since the prompt-end state goes with the extension.
* A reply the next prompt re-tokenizes differently falls back to that point too, and
  its entry is absorbed.
* Log: `id=… prompt cache reply kept: n tokens (+g generated) …`.
* `id=… prompt cache reply not kept: its prompt entry is gone …` / `…: reasoning …`.
* `id=… prompt cache reply skipped: in a cascade group …`.
* `prompt cache reply not kept: over the batch's headroom …`.
* `scripts/perf/agentic/worker_log.py` counts the lines above apart from prefill stores.

**Numerics.** The reply's KV comes from decode kernels (qmv at B ≤ 2, mrqmm at 3–16
rows, cascade-released rows), and the next turn's first slab starts at the reply end.
A hit through a reply is therefore *not* bit-identical to a cold prefill of the same
tokens. It is in the batched-decode numerics class: on the CPU a T-token slab and T
one-token steps already differ in the GDN recurrence from layer 0 on. On the GPU gate
(9B, `docs/perf/2026-10-17/reply_kv_gpu.md`) a hit through a reply vs a cold prefill
agreed on top-1 at 99.88 % of 862 teacher-forced positions, KL 5.8e-5. With the knob on,
a greedy answer can depend on what the cache holds; `QWEN_PROMPT_CACHE_REPLY=0` is the
way back to the bit-exact path (an answer independent of the cache contents).

### One KV copy for a single stream (`QWEN_PROMPT_CACHE_MOVE`, default off)

With the copies above, a conversation's KV exists twice while its request runs: in the
row and in the cache entry (`store` copies, `restore` copies). At fp16 that is what limits
the context of a single-stream setup long before the weights do: the 27B with its
drafter (64 KiB of KV per token, 6.6 GiB for rows + cache under a 27 GiB cap) kept
entries only up to ~40K tokens — a longer prompt ran, was "not kept", and every turn was
cold. `QWEN_PROMPT_CACHE_MOVE=1` keeps **one** copy:

* a hit on the conversation's own entry (its ids are a prefix of the new prompt, or the
  new prompt covers its reply) **moves** the entry's buffers into the row, layer by layer
  (the transient is one layer's KV), and the entry leaves the cache while the row runs;
* nothing is copied at the prefill's end: the prompt entry's states are kept aside and
  the store is deferred;
* when the row ends, its own buffers become the entry: the prompt entry, and over the
  same buffers the reply extension (`QWEN_PROMPT_CACHE_REPLY`). A reply that is not stored
  (reasoning) is cut off the buffers when it is a KVCache step (256 tokens) or longer;
* a prefill that stops early gives the entry back as it was (nothing prefilled past it)
  or stores its prefix (the partial store), again the row's own buffers.

Answers, reused tokens and resume positions are those of the copies (the parity tests
play every scenario both ways and compare the cache contents bit for bit;
`tests/test_pcache_move.py`); on the GPU a hit equals a cold prefill of the same prompt
token for token (27B, 8K, with and without thinking).

Only for a lone stream: `QWEN_MAX_BATCH=1`, fp16 KV (`QWEN_KV_BITS=0`), no disk tier
(the load log says `prompt cache move: on` or why not). What it costs:

* the entry is out of the cache while its row runs: a second request of the same
  conversation arriving meanwhile would miss (single stream: it waits anyway);
* a row that **fails in a forward** (out of memory, a kernel error) stores nothing and the
  entry it took is lost — the copies would have kept it. The next turn is a cold prefill;
* a sibling conversation (same system prompt, another branch) still gets copies of the
  shared part while there is room; when a copy does not fit beside the entry, the entry
  itself moves into the new row (logged), i.e. the abandoned branch is gone;
* the checkpoints a prefill takes are held by the row until it ends and are not in the
  cache's byte count (thinned to `QWEN_PROMPT_CACHE_CKPTS` as they come);
* `restore` and the end-of-row store still build one layer at a time.

**27B + DFlash2 at 60K** (measured 2026-11-11, `docs/perf/2026-11-10/dflash2_27b.md` §8;
27 GiB wired limit): cold prefill of 60.1K tokens 521 s, then every turn resumes in
0.9–2.1 s; decode 11.5–12.7 tok/s with thinking, 14.3–16.0 without (plain decode there: 9.6);
process peak 26.4 GiB (prefill), 25.6–26.0 in decode; the cache holds the one entry
(4.4 GiB) and memory does not grow from turn to turn. The recipe:

```zsh
sudo sysctl iogpu.wired_limit_mb=28672          # once per boot
cd src/worker && QWEN_MODEL=../../weights/qwen38-27b-mlx-5bit QWEN_KV_BITS=0 QWEN_MRQMM5=1 \
  QWEN_DRAFT=../../weights/qwen38-27b-dflash2 QWEN_MAX_BATCH=1 QWEN_MAX_CONTEXT=65536 \
  QWEN_PROMPT_CACHE=1 QWEN_PROMPT_CACHE_MOVE=1 QWEN_PROMPT_CACHE_GB=6 QWEN_PROMPT_CACHE_CKPTS=4 \
  QWEN_WIRED_LIMIT_GB=27 QWEN_CACHE_LIMIT_GB=1 QWEN_SDPA_FUSED_MIN_KV=8192 QWEN_PREFILL_BUDGET=16777216 \
  QWEN_PRESENCE_PENALTY=0 uv run python worker.py
```

Why each of the memory knobs: weights + drafter are 19.4 GiB and 60K of fp16 KV 3.7 GiB, so
everything else has ~3 GiB. `QWEN_SDPA_FUSED_MIN_KV=8192` + `QWEN_PREFILL_BUDGET=16777216`
cap the unfused attention scores of a prefill slab at 0.75 GiB (the default lets them reach
2.8 GiB just below 32K keys: the first 60K run died there); `QWEN_PROMPT_CACHE_CKPTS=4`
bounds the recurrent states (147 MiB each on the 27B) the row and the entry hold;
`QWEN_CACHE_LIMIT_GB=1` the buffer pool. A DFlash round itself needs ~1.3 GiB above the
resting row: the verify block's per-token recurrent states (block 8). Above the wired
limit MLX does not fail at once, it crawls (the same run at `_CKPTS=8` and before the two
fixes of §8: 0.6 tok/s, then Metal's out-of-memory) — keep the peak under it.

### Cascade attention (`QWEN_CASCADE`, default on with fp16 KV)

Rows that share a long prompt prefix (one system prompt, the prompt cache on) can
decode their attention as one *group*: the prefix's KV once, in fp32, and each row's
own tokens in a stacked fp32 suffix (`src/worker/cascade.py`; design
`docs/perf/2026-10-12/design/cascade_plan.md` Part B). A group forms when a row joins
and at least `QWEN_CASCADE_MIN_ROWS` (8) running rows share ≥ `QWEN_CASCADE_MIN_PREFIX`
(4096) prompt tokens ending on a prefill slab end of each (their KV there are the same
bits); the earlier rows are converted and their own KV freed, later rows join it.

Memory rule: at the rows' real lengths, the group's fp32 prefix + stack (+ its step
transients) must take no more than their own fp16 caches would, and a group is never
charged less than its prefix + stacks + 2 contiguous fp16 rows (what dissolving its
last 2 members builds beside it). 9B, ~150-token suffixes: 3 rows never win (3 × 32K /
64K / 96K: 4.2 / 8.3 / 12.4 GiB vs 3.2 / 6.3 / 9.5), 4 only from 32K, 5–16 at 4K–32K.
The reservation is charged per group (fp32 stack at the longest member reservation),
so grouping can raise `reserved` (at 4K with the default 4096 reserved tokens:
+1.0–1.7 GiB at 8–16 rows); a formation or join that raises it is taken only while `reserved` + the
prompt cache's live bytes stay within the budget (else skipped; nothing is evicted for
it). Rows leave when they finish or when their own suffix passes P/2; the group
dissolves (rows get contiguous caches back) when fewer than 3 members are left (so
always at B ≤ 2; unchecked, inside the group's charge) or the memory rule fails. A P/2
leave, a memory dissolve and `set_cascade(False)` go one member per decode step (the
last < 3 together), each checked with the group still alive: `reserved` + the released
rows' fp16 caches + the prompt cache within the batch headroom, nothing credited for
the group's shrink; until it fits the rows stay grouped.

Only fp16 KV (`QWEN_KV_BITS=0`, the 9B), `QWEN_PROMPT_CACHE=1` and `QWEN_MAX_BATCH` ≥ 3
— otherwise it is off whatever the switch says (the load log says why); text-only
rows that resumed from the prompt cache or stored an entry; never vision or DFlash
rows. Grouped rows are not bit-equal to the per-row kernel (fp32 statistics, another
summation order; closer to an fp32 reference on the 9B layout): numerically the same
class as mrqmm's batched decode (B4: top-1 ≥ 99.71 %, KL +0–13 % over today's per-row
batch), so a greedy answer at B ≥ 3 can also depend on the cache contents and arrival
order. B ≤ 2 and ungrouped rows are unchanged (bit-exact); `QWEN_CASCADE=0` restores
the per-row path bit for bit, `QWEN_MAX_BATCH=1` full determinism (with
`QWEN_PROMPT_CACHE_REPLY=0` too: a hit through a reply differs from cold even at B = 1).

Measured (9B, 2026-10-12, `docs/perf/2026-10-12/cascade_gpu.md`): decode step 8K × 16
rows 66.4 → 53.7 ms (×1.24), 32K × 8 83.6 → 68.2 ms (×1.23); 16 × 8K active memory
−3 GiB (peak 17.6–18.1 → 15.0–15.3 GB); 16 × 32K all admitted (was 12 + 4 waiting),
peak 25.3 → 23.2 GB. Caveats: forming a group builds the fp32 prefix while the members
still hold their fp16 caches (+2.2 GB peak at 8 × 32K); 32K × 16 steps at ~92 ms vs
~62 ms projected (open, needs a profile). The dissolve floor and the staged releases
came after that run (CPU-tested only).

Logs: `cascade: group formed: P=…, n rows (…)`, `joined`, `left (why)`,
`dissolved (why)`; the `batch:` line's `groups= grouped=`; Health `cascade_groups` /
`cascade_rows` (gateway debug log on change). `QWEN_CASCADE_CHECK=1` verifies every
new member's prefix KV bitwise (a sync per join).

## Context & memory

Context is capped at **128K tokens** (`MAX_CONTEXT=131072`, prompt + generation;
override with `QWEN_MAX_CONTEXT`).

**KV cache precision is per-model** (`QWEN_KV_BITS`, default 4), because the right
value tracks the *weight* precision and the memory budget:

| model | weights | KV (`QWEN_KV_BITS`) | KV @ 128K | resident @ 128K | why |
|---|---|---|---|---|---|
| Qwen 35B-A3B | 3-bit, ~18 GB | **q4** (default) | ~0.75 GB (fp16 would be ~2.6) | ~21 GB (tight) | 4-bit KV is the balanced match for 3-bit weights; q8 KV would blow the 32 GB budget |
| **Ornith 9B** | 8-bit, ~10 GB | **fp16 (`0`)** | ~4 GB | **~16 GB** | headroom to spare, and an unquantized cache is *faster* than q8 here: at 32K prefill 72 vs 82 s, decode 31.9 vs 27.3 tok/s (quantized-cache prefill dequantizes per slab; the q8 decode SDPA kernel is slower than fp16) |
| **Qwen3.8-27B** | 5-bit, ~18.1 GB | **q8** | ~3.9 GB | **~24 GB** | 5-bit weights deserve q8 KV; fits under the 27 GB wired limit with the bump |

**KV math for the hybrid models.** Only the full-attention layers (every 4th) hold
a growing KV cache; the Gated DeltaNet layers keep a small constant state. Per token
at q8: `2 (K,V) × full-attn layers × 4 KV heads × head_dim 256 × (1 byte + 4/64 for
the fp16 scale and bias per 64)` → 9B (8 layers) ≈ 17 KiB, 27B (16 layers) ≈ 34 KiB.
So the 27B is ~4.25 GiB at 128K and ~6.6 GiB at 200K; resident ≈ weights 18.1 + KV + ~2 GB prefill activations ≈ 24 GB
(128K) / 26 GB (200K). **To run the 27B at 200K**: `QWEN_MAX_CONTEXT=204800` plus
the wired bump (`sudo sysctl iogpu.wired_limit_mb=28672`, `QWEN_WIRED_LIMIT_GB=27`,
`QWEN_PREFILL_STEP=512`) — ~1 GB of slack under the limit, so keep prefill chunked.
The 9B needs no `sudo sysctl`: its fp16 KV is ~4 GiB at 128K (q8 would be ~2.1 GiB).

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
  activation peak so a 50K-token prompt doesn't spike memory in one matmul. Where
  the attention materializes its scores (quantized KV — the 35B/27B — or fp16 KV
  without the fused kernel) the slab shrinks as the context grows (`slab ×
  cache_offset ≤ QWEN_PREFILL_BUDGET`, default 2048 × 32K, floor
  `QWEN_PREFILL_MIN_STEP` 256). An fp16-KV model with `QWEN_SDPA_FUSED=1` (the
  9B with `QWEN_KV_BITS=0`) keeps 2048 once a slab reaches 32K keys: the fused
  kernel holds no score matrix (9B 120K prefill: ×1.27, peak −1.0…1.4 GB); below
  32K keys the budget rule still applies. The cost under batching: a slab runs
  uninterrupted, so while a long prompt prefills (a 2048 slab at 120K takes ~9 s,
  ~3× the old shrunk slab) the other running requests emit no tokens. Set `QWEN_PREFILL_STEP=512` for the
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

**Batching memory.** The worker admits a request next to running ones only if its
reservation fits a budget: KV for `prompt + min(max_tokens, QWEN_BATCH_RESERVE_TOKENS)`
tokens (4096 by default, so clients that omit `max_tokens` still batch) plus 2× the
row's recurrent state (the queued step's copy beside the current one) — 9B: 32
KiB/token fp16 + 2 × 49 MiB; 27B q8: 34 KiB + 2 × 147 MiB; 35B q4: 5.6 KiB + 2 × 61
MiB. The budget (`QWEN_BATCH_MEM_GB`, else derived at
load) is cap − weights − prefill reserve (`QWEN_PREFILL_BUDGET` × q heads × 2 B +
0.4 GiB; with the fixed fp16 slab at least `QWEN_PREFILL_STEP` × 32K × q heads × 2 B) − 1 GiB − `QWEN_CACHE_LIMIT_GB` (if
`QWEN_CLEAR_CACHE=0`); the prompt cache's live bytes count inside it and give way on demand; cap = `QWEN_WIRED_LIMIT_GB` or the device working set (the
`iogpu.wired_limit_mb` sysctl once raised). Logged at load (`batching: …`); 9B at
the 28 GiB sysctl: 13.8 GiB. A request that does not fit waits inside the worker
(never a 503 for memory); an empty batch always takes the next request. Rows are
re-accounted at their real length every step: a row may outgrow its reservation
(the headroom outside the budget absorbs it), and while the batch is over budget
nothing new is admitted. Growth past budget + prefill reserve would exhaust that
headroom, so then the growing row ends with `finish_reason: "length"` while other
rows run (a lone row is never cut; set `GATEWAY_DEFAULT_MAX_TOKENS` to bound
clients that send no `max_tokens`). If the budget cannot hold two typical rows, the worker
reports `max_batch` 1. Measured: 8 rows × 8K-token prompts peak at 17.5 GB on the
9B. How the gateway uses `max_batch` (`GATEWAY_MAX_INFLIGHT`, several gateways on
one worker, metrics): see *Observability* and *Env knobs*.

See **CLAUDE.md** for design decisions, implementation notes, and gotchas.

## Observability
```zsh
cd deploy && QWEN_LOG_DIR=../logs docker compose up -d
```
- Grafana: http://localhost:3000 (anonymous admin) → dashboard **qwen-server**.
- Prometheus: http://localhost:9090 ; gateway metrics: http://localhost:8080/metrics.
- mac-exporter (GPU/ANE/watts) needs root: `sudo ./target/release/qwen-mac-exporter`.
- Set `LOG_FORMAT=json` on the gateway so Promtail can label by level.
- Concurrency: `qwen_queue_depth` (waiting + running), `qwen_in_flight` (sent to the
  worker, ≤ `qwen_inflight_capacity`), `qwen_gateway_queue_seconds` (wait for an
  in-flight slot), and from the worker's usage `qwen_worker_queue_seconds` (wait
  inside the worker before prefill) and `qwen_batch_mean` (mean decode batch a request
  shared; both only from a worker that reports them). The `done` log line has
  `queue_wait_ms` (gateway), `queue_ms` / `batch_mean` (worker), the response `id` and
  the `worker_id` (unique per generation; `chatcmpl-<request_id>-<12 hex>` when the
  client sets `request_id`).
- `qwen_request_duration_seconds` and `qwen_ttft_seconds` (streaming) run from the
  request's arrival at the gateway — queue waits included, what the client sees.
  All histograms are Prometheus histograms (`_bucket`, for `histogram_quantile`).
- 503 `ServiceUnavailableError` = load shed, retry later: queue full (`MAX_QUEUE`), the
  worker's `Check` not answered within `GATEWAY_CHECK_TIMEOUT_S`, or the worker refusing
  a generation (gRPC `RESOURCE_EXHAUSTED`; for a stream already open this is an error
  frame with `code: 503`). A waiting stream gets `: keep-alive` comment lines every 15 s.

## Resilience (launchd)
Edit paths in `deploy/launchd/*.plist`, then load them (worker + gateway as user
agents, exporter as a root daemon). `watchdog.sh` kickstarts the gateway if
`/health` goes dark. See comments inside each plist. The worker restarts itself
through launchd's KeepAlive when its scheduler gives up (3 consecutive failed
batch steps: every request gets an error, then the process exits with 70);
`kill` (SIGTERM) stops it gracefully — running and queued requests get
`Error(shutdown)` first.

## Env knobs
| var | where | default |
|---|---|---|
| `QWEN_MODEL` | worker | `weights/qwen36-35b-a3b-mlx-ud3` (repo-relative) |
| `WORKER_ADDR` | gateway | `127.0.0.1:50051` |
| `GATEWAY_ADDR` | gateway | `0.0.0.0:8080` |
| `MAX_QUEUE` | gateway | `32` — requests admitted (waiting + running); beyond = 503 |
| `GATEWAY_MAX_INFLIGHT` | gateway | unset = the worker's `max_batch` (Health, re-read every 15 s; 1 for a worker without it) — generations sent to the worker at once, the rest wait in the gateway queue. Several gateways on one worker should split its `max_batch` |
| `GATEWAY_CHECK_TIMEOUT_S` | gateway | `30` — the worker's pre-queue `Check` taking longer is a 503 |
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
| `QWEN_VERIFY_EXACT_MAX_T` | worker | `8` — DFlash verify blocks up to this many tokens use mlx-vlm's exact verify kernels (output = plain greedy); longer blocks use `quantized_matmul` (the exact kernel is ×9–12 slower at 12–16 rows). 9B coding tasks: 48 tok/s vs 37 plain. Empty = default |
| `QWEN_WIRED_LIMIT_GB` | worker | `0` (off; clamped to kernel cap) |
| `QWEN_PREFILL_STEP` / `QWEN_PREFILL_BUDGET` / `QWEN_PREFILL_MIN_STEP` | worker | `2048` / `67108864` (2048×32K) / `256` — slab shrinks so slab×context ≤ budget (quantized KV; fp16 KV only below 32K keys or without the fused SDPA — the 9B with `QWEN_KV_BITS=0` keeps 2048 from 32K keys on) |
| `QWEN_TEMPERATURE` / `QWEN_TOP_P` / `QWEN_TOP_K` / `QWEN_MIN_P` | worker | `0.0` (argmax) / `1.0` / `0` / `0.0` (off) |
| `QWEN_PRESENCE_PENALTY` | worker | `1.5` (set `0` for reasoning/benchmarks) |
| `QWEN_FREQUENCY_PENALTY` / `QWEN_REPETITION_PENALTY` | worker | `0` / `1.0` |
| `QWEN_JSON_MAX_WS` | worker | `0` (off) — cap on a whitespace run in a `response_format` answer |
| `QWEN_PROMPT_CACHE` | worker | `0` (set `1`: multi-slot prefix cache with recurrent-state checkpoints) |
| `QWEN_PROMPT_CACHE_GB` / `_STRIDE` / `_CKPTS` | worker | `3` / `2048` / `12` — soft cap (evicted at store in the round-aware order, live bytes counted in the batch budget and evicted on demand), checkpoint stride, checkpoints per entry |
| `QWEN_PROMPT_CACHE_STALE_S` | worker | `300` — an entry unused this long (and wanted by no waiting request) is stale: evicted first, LRU, even within the round of turns; `0` = off (the tick rule only) |
| `QWEN_PROMPT_CACHE_CHAIN_MARKS` / `_CHAIN_TAIL` | worker | `3` / `4096` — an entry that absorbed the previous turn keeps only the end of the system block, this many last message starts, the strides in this many last tokens and inside the first user message, and its end state (`-1` marks: off, the `_CKPTS` policy everywhere) |
| `QWEN_PROMPT_CACHE_CHAIN_FRESH_TOKENS` | worker | `16384` — a fresh entry (absorbed nothing) of at least this many tokens takes the chain policy too (+ the divergence point and the system block's strides); shorter ones keep the `_CKPTS` policy; `0` = never (fresh entries always the `_CKPTS` policy) |
| `QWEN_PROMPT_CACHE_REPLY` | worker | `1` — a finished request's entry is extended by its decoded tokens' KV + end state, so the next turn resumes after the reply (agentic loads: −21 % prefill tokens on AppWorld); a hit through a reply is not bit-identical to a cold prefill (batched-decode numerics class: top-1 99.88 %, KL 5.8e-5 on the GPU gate; see *Prompt cache*); `0` = off, bit-exact |
| `QWEN_PROMPT_CACHE_MOVE` | worker | `0` — one KV copy per conversation for a single stream (see *One KV copy*): a hit moves the entry's KV into the row, the row's own buffers become the entry when it ends; only with `QWEN_MAX_BATCH=1`, `QWEN_KV_BITS=0`, no disk tier (else off, the load log says why). Same answers and hits as the copies; a row that fails in a forward loses its entry. 27B + DFlash2: entries past ~40K tokens were "not kept", with it 60K conversations hit (TTFT 1.4–2.1 s vs 521 s cold) |
| `QWEN_PROMPT_CACHE_KEEP_TOKENS` / `_KEEP_WAIT_S` | worker | `8192` / `10` — a request kept out only by the cache entry it would resume from waits for a running row to end only for a hit this deep, and only this long; else the entry is evicted and the request starts cold (8K tokens ≈ 14 s of 9B prefill, more than the wait) |
| `QWEN_CASCADE` | worker | `1` — cascade decode attention for rows sharing a prompt prefix (see *Cascade attention*); only with `QWEN_KV_BITS=0` + `QWEN_PROMPT_CACHE=1` + max_batch ≥ 3, else off; `0` = the per-row path bit for bit |
| `QWEN_CASCADE_MIN_ROWS` / `_MIN_PREFIX` / `_CHECK` | worker | `8` (1–2 → 3, 0 = off) / `4096` (0 = off) / `0` — rows sharing a prefix before a group forms, its minimum length in tokens, bitwise prefix check per join |
| `QWEN_DRAFT_FP32` | worker | unset = auto (on for DFlash2 drafters: the 27B's overflows fp16 → no draft accepted) — the drafter's activations in fp32, weights stay `QWEN_DRAFT_BITS`; `1` / `0` force |
| `QWEN_MRQMM5` | worker | `0` — the multi-row matmul kernel for 5-bit modules (the 27B), 3..16 rows; needed for DFlash2 on the 27B (×1.77 code / ×1.41 thinking vs ×0.8–1.1 without); off by default: the 27B's batched decode through it is not gated |
| `QWEN_DRAFT_TAIL` / `QWEN_DRAFT_TAIL_END` | worker | `128` / `2048` — with the prompt cache, every state a DFlash row stores keeps the drafter's context there: the target's hidden states of the last N positions (27B: 51 KiB each); a DFlash row resuming from it starts its drafter with them instead of only the tokens the hit left to prefill. `TAIL`: checkpoints; `TAIL_END`: the entry's end (prompt end, reply end; capped at the drafter's window, ≤ 105 MB per entry on the 27B) — a turn that re-emits the previous answer's code needs the whole source in the drafter's sight. `0` = off |
| `QWEN_VERIFY_ATT` | worker | `1` — a DFlash verify block's attention through MLX's vector attention kernel, several queries per call (queries × GQA factor ≤ 32: 5 on the 27B, 8 on the 9B; longer blocks in chunks) instead of mlx-vlm's one query per call, which reads the row's whole KV once per block token. Bit-equal to the one-query calls (checked on random data at load, else off). 27B, block 8: round 570 → 296 ms at 60K keys (×1.9), 210 → 202 ms at 24K; `0` = the library's calls |
| `QWEN_DRAFT_SAMPLING` | worker | `0` (set `1`: speculative path also for temperature/presence-penalty requests, exact sample-match; code ×1.1–1.2, prose ×0.8) |
| `QWEN_CLEAR_CACHE` / `QWEN_CACHE_LIMIT_GB` | worker | `1` / `3` (drop MLX buffer pool after each request / pool cap) |
| `QWEN_MAX_BATCH` | worker | `16` — requests decoded together (continuous batching); `1` = one at a time, as before. Reported to the gateway (Health) as its in-flight limit |
| `QWEN_BATCH_MEM_GB` | worker | `0` = derived (see *Context & memory*) — GiB the running rows may reserve |
| `QWEN_BATCH_RESERVE_TOKENS` | worker | `4096` — generation tokens a row reserves at admission (`prompt + min(max_tokens, this)`); real lengths re-accounted every step |
| `QWEN_BATCH_PREFILL_SHARE` | worker | `0.5` — a new prompt's prefill share of the GPU while a full batch decodes (more when fewer rows run); `1` = prefill first |
| `QWEN_PFQMM` | worker | `1` — prefill matmul with 64-row tiles (`pfqmm.py`, MLX's own qmm_t code): ×1.03–1.28 per matmul on M1/M2 (8-bit ×1.10–1.14), bit-identical, self-tested per (bits, group) at load; `0` = MLX's own. Needs fp16 scales: `QWEN_FP16=0` leaves it off |
| `QWEN_MRQMM` | worker | `1` — multi-row 8-bit matmul kernel for batched decode (3..16 rows, 8-bit g64 fp16-scale modules with ≥ 512 outputs; 9B decode step ×2.5 at 8 rows, ×1.8 at 16); `0` = MLX's own |
| `QWEN_GDN_FUSE` | worker | `1` — a decode step's four GDN in-projections as one matmul (no extra memory; −2% decode step). Needs fp16 scales: with `QWEN_FP16=0` it fuses nothing |
| `QWEN_SDPA_FUSED` | worker | `1` — MLX's fused attention kernel for long fp16-KV prefill slabs (≥ 512 queries over ≥ 32K keys): 64K prefill −3…5%, −1.3 GB peak |
| `QWEN_SDPA_FUSED_MIN_KV` | worker | `32768` — keys from which a long fp16-KV prefill slab takes the fused kernel. Below it the unfused graph holds slab × keys × heads scores (27B, 24 heads: 2.8 GiB just under 32K, which the prefill reserve is sized for); `8192` with `QWEN_PREFILL_BUDGET=16777216` caps them at 0.75 GiB and the reserve at 1.15 GiB (per call the fused kernel is ×1.05–1.12 slower up to 28K keys, 9B) — for setups short of memory, e.g. the 27B with its drafter at 60K |
| `QWEN_GDN_FAST` | worker | `0` — `mx.fast.gated_delta_update` for prefill slabs (×1.02 per slab, but it rounds the decay to fp16: KL 0.013 on 8K tokens; off) |
| `QWEN_GEN_SLOTS` | worker | `max(2·max_batch, 8)` — concurrent Generate RPCs; beyond = gRPC `RESOURCE_EXHAUSTED` (gateway 503). gRPC threads = slots + 8 (replaces `QWEN_GRPC_THREADS`) |
| `MAC_EXPORTER_ADDR` | exporter | `0.0.0.0:9101` |

## License

MIT (`LICENSE`). Code derived from MLX, mlx-vlm (MIT) and vLLM (Apache 2.0) and the
projects whose ideas it borrows are listed in `THIRD_PARTY_NOTICES.md`.
