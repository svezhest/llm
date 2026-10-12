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
- **400 instead of silence**: `n` > 1, `logit_bias`, `logprobs` / `top_logprobs`, legacy
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
and the last three message starts are never thinned. `QWEN_PROMPT_CACHE_GB` (3) is
a soft cap: past it the least recently used entries go at store (the newest stays,
alone if need be). Entries **share their KV**: an entry is a chain of immutable
segments, a store copies only the tokens past its resume point out of the request's
cache and reuses the older entry's segments for the rest, so 16 conversations over
one 8K system prompt hold it once (~1.4 GB for all 16 on the 9B by the design
estimate, not yet measured; every entry used to carry its own copy: 6 fit in 3.2 GB,
measured); a hit concatenates the pieces once
into the buffer its prefill would have built anyway (bit-identical). The cache's
live bytes (every buffer once) count against the batch memory budget: a store
evicts LRU entries down to what the running rows leave; a new entry that does not
fit there even alone is not kept, and then the cache stays as it was (nothing
absorbed, evicted or copied; counted as an eviction). An admission or a growing row
evicts LRU entries when only the cache is in the way (an admission spares the entry
it would resume from at least one slab deep, and evicts nothing when that would not
get it in; a growing row never evicts the entry a running prefill resumed from).
When that entry alone keeps the next request out, the request waits for a running
row to end only if the hit saves at least `QWEN_PROMPT_CACHE_KEEP_TOKENS` (8192)
tokens of prefill, and at most `QWEN_PROMPT_CACHE_KEEP_WAIT_S` (10 s); then (or at
once for a shallower hit) the entry is evicted and the request starts cold. A
failed store only logs; the request goes on. mlx-vlm's own trim is never used
(broken for quantized KV tuples). Vision requests never touch the cache.

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
reused=r gap=g absorbed=a pieces=p` (gap: common-prefix tokens it could not resume
from; pieces: segments in its KV chain).
Responses carry `usage.prompt_tokens_details.cached_tokens` (vLLM's field);
the gateway counts `qwen_gateway_cached_tokens_total`; the worker's Health reports
entries, live bytes, hits, misses and evictions (a new entry not kept at store counts
as one; gateway debug log on change).

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
| `QWEN_PROMPT_CACHE_GB` / `_STRIDE` / `_CKPTS` | worker | `3` / `2048` / `12` — soft cap (LRU at store, live bytes counted in the batch budget and evicted on demand), checkpoint stride, checkpoints per entry |
| `QWEN_PROMPT_CACHE_KEEP_TOKENS` / `_KEEP_WAIT_S` | worker | `8192` / `10` — a request kept out only by the cache entry it would resume from waits for a running row to end only for a hit this deep, and only this long; else the entry is evicted and the request starts cold (8K tokens ≈ 14 s of 9B prefill, more than the wait) |
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
| `QWEN_GDN_FAST` | worker | `0` — `mx.fast.gated_delta_update` for prefill slabs (×1.02 per slab, but it rounds the decay to fp16: KL 0.013 on 8K tokens; off) |
| `QWEN_GEN_SLOTS` | worker | `max(2·max_batch, 8)` — concurrent Generate RPCs; beyond = gRPC `RESOURCE_EXHAUSTED` (gateway 503). gRPC threads = slots + 8 (replaces `QWEN_GRPC_THREADS`) |
| `MAC_EXPORTER_ADDR` | exporter | `0.0.0.0:9101` |
