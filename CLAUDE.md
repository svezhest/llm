# CLAUDE.md — llm

Local LLM inference on Apple Silicon (MLX). **Rust gateway** (OpenAI-compatible
HTTP/SSE) ⇄ **gRPC** ⇄ thin **Python MLX worker(s)**. Metrics (Prometheus),
Grafana dashboards, Loki logs, request queue, leveled logging. Inference runs
**native** (Metal not in Docker); only the observability stack is containerized.
See `README.md` for run instructions, `BACKLOG.md` for the plan + phase status.

## Layout (repo root = `/Users/user/Projects/llm`)

- `weights/` — model dirs (git-ignored). `qwen36-35b-a3b-mlx-ud3`, `ornith10-9b-mlx-8bit`, `ornith15-9b-mlx-8bit`,
  `qwen38-27b-mlx-5bit`. Pull with `scripts/hf_pull.sh <repo> <dir>` (resumable aria2c).
- `src/proto/worker.proto` — gRPC contract (single source of truth; both sides codegen).
- `src/gateway/` (Rust) — axum HTTP + tonic gRPC client + queue + metrics + tracing.
  `src/{main,api,request,pool,metrics,worker_client,logging,journal,pb}.rs`.
- `src/worker/` (Python) — grpcio server, mlx-vlm; load + generate only. Own uv project.
- `src/exporter/` (Rust) — `powermetrics` → `/metrics` (GPU/ANE/watts; needs root).
- `deploy/` — docker-compose (prometheus/grafana/loki/promtail) + launchd plists.
- `Cargo.toml` — workspace, members `src/gateway` + `src/exporter`. `target/` at root.
- `pyproject.toml` / `uv.lock` — root tooling/download env (`chat.py`, hf-hub, mlx-vlm).
  Separate from `src/worker/`'s own uv project (grpcio runtime). **Not merged** — a
  deliberate split; the worker's runtime env is isolated from tooling.

Build paths are relative and survive the layout: gateway `build.rs` → `../proto`,
worker `gen_proto.sh` → `../proto`. Model default is repo-relative (`_REPO_ROOT`
from `__file__`), so the tree can move without editing code.

## Run order

```zsh
sudo sysctl iogpu.wired_limit_mb=28672                   # once/boot, for 35B long ctx
cd src/worker && uv run --group proto bash gen_proto.sh  # once: codegen worker_pb2*.py
QWEN_WIRED_LIMIT_GB=27 QWEN_PREFILL_STEP=512 uv run python worker.py   # 127.0.0.1:50051
# after "model loaded":
cargo build --release && RUST_LOG=info ./target/release/qwen-gateway   # 0.0.0.0:8080
# observability (optional): cd deploy && QWEN_LOG_DIR=../logs docker compose up -d
```

Stop everything: `kill $(pgrep -f 'qwen-gateway|python worker.py')`. The MLX 35B
worker holds ~16–18 GB — kill it to reclaim memory. Never `kill -9` a download
(orphans partials). `ps`/RSS under-reports MLX (wired memory); use the macOS
resource manager / `footprint` for the real size.

## Models

- **`qwen36-35b-a3b-mlx-ud3`** — Qwen3.6-35B-A3B, VL MoE (256 experts, ~3B active).
  MLX affine g64, 3-bit **Unsloth Dynamic** (mixed per-layer: embed 6b, attn/gates
  8b, shared-expert 4b, MoE experts 3b). ~16 GB. Needs the wired-limit dance on 32 GB.
- **`ornith10-9b-mlx-8bit`** — Ornith-1.0-9B (deepreinforce-ai, agentic-coding RL
  finetune), dense VL, MLX 8-bit. ~10.4 GB. No wired-limit dance; 8-bit ≈ fp16 quality.
  Card benchmarks (SWE-bench 69%) are self-reported and implausibly high for the
  size — **evaluate empirically, don't trust the card**.
- **`ornith15-9b-mlx-8bit`** — Ornith-1.5-9B (`ornith-ai/Ornith-1.5-9B-MLX-8bit`, the
  official MLX quant). Same arch/quant as the 1.0 but the checkpoint is **text-only**:
  no `vision_config`, no `visual.*` weights, no `preprocessor_config.json`. mlx-vlm
  0.6.2 still builds the vision tower and `load_weights` is strict, so `Engine.load`
  detects the missing `vision_config` and loads non-strictly (tower stays random;
  image requests will not work — text only). ~9.5 GB, 11.6 GB loaded, ~36 tok/s.
  Empirically (5 coding tasks, temp 0) 1.5 ≥ 1.0: same correctness on 4, and 1.0
  leaked its thinking into the answer and hit the length cap on the SQL task while
  1.5 answered cleanly. 1.5 thinks longer on trivial prompts (2–5× more tokens).
- **`qwen38-27b-mlx-5bit`** — Qwen3.8-27B (lmstudio-community), dense hybrid VL,
  `model_type qwen3_5` like the 9B (64 layers, full attn every 4th, 4 KV heads,
  head_dim 256, native 256K). MLX 5-bit g64, ~18.1 GB. Run: `QWEN_KV_BITS=8` +
  wired-limit dance (~24 GB @128K; 200K via `QWEN_MAX_CONTEXT=204800`, ~26 GB).

Download over a slow, throttled link — hard-won recipe:
- `HF_HUB_DISABLE_XET=1` always (Xet backend hangs silently at 0 MB/s here).
- Get each LFS file's sha256 from `…/api/models/<repo>/tree/main` (`lfs.oid`).
- `aria2c -c -x16 -s16 -k1M -j1 --max-tries=0 --check-integrity=true
  --checksum=sha-256=<oid> --header="Authorization: Bearer <token>" --dir=<weights/x> -o <file> <resolve-url>`.
  Multi-connection beats single curl ~12× (CDN throttles per-connection); the
  checksum defeats proxy-corrupted-but-full-size shards.
- **Never hard-kill** `hf download`/aria2 — orphans/duplicates partials. Token at
  `~/.cache/huggingface/token`. Runtime is **mlx-vlm** (VL models), not mlx-lm.

## Key design decisions (locked — don't relitigate)

- **Public API**: OpenAI-compatible HTTP/SSE (`/v1/chat/completions`), cross-OS.
- **Gateway ⇄ worker**: gRPC (tonic ⇄ grpcio), server-streaming `Generate→stream Event`.
- **Queue + serialized worker** (only one ~18 GB copy fits in 32 GB). "pooling" in
  the plan = metrics pooling (Prometheus pull), not a worker pool.
- **Rust everywhere except the MLX worker** (process isolation, no PyO3).
- **Images** = paths/URLs on the wire, never bytes.

## Direction: multi-model routing (planned, not yet built)

Goal: several models / inference backends behind one gateway. `llama-server`
already speaks OpenAI API, so for a single model the gateway is optional — its
value is **routing + queue + metrics**. Plan: gateway keeps a registry
`model name → backend addr` and proxies by the request's `model` field (e.g.
`qwen3.6-35b-a3b`→:50051, `ornith-9b`→:50052). Since both current models are
mlx-vlm, this is "pick one of N same-protocol workers", not multi-protocol —
keep it that simple until a non-MLX (GGUF/llama.cpp) backend actually lands.

## Performance notes (measured 2026-09-11 on M1 Max 32 GB, Ornith-1.5 9B 8-bit)

- **M1/M2 have no native bf16.** All our MLX quants store scales/norms/embeddings
  in bf16, and an 8-bit `quantized_matmul` with ≥512 rows runs 4.5 TFLOPS in bf16
  vs 7.8 in fp16. `QWEN_FP16=1` (default) casts the language model to fp16 after
  load: prefill 8K 26.5→15.5 s, 32K 118→82 s (with the 2048 slab); decode
  unchanged (bandwidth-bound, ~28 ms/token ≈ 9.5 GB / 400 GB/s). Verified
  teacher-forced on 3.4K tokens: KL(bf16‖fp16)=0.0003, top-1 agreement 99.4%
  (9B); 0.0004 / 99.2% on the 27B; 0.0019 / 98.8% on the 35B (3-bit MoE
  amplifies the noise — still under a Q6 quant's 0.004; set `QWEN_FP16=0` if
  fidelity beats speed there). 35B prefill 8K: 16.6 → 11.5 s, decode +4%.
- **Prefill slab**: 512→2048 is +15% at 8K (only matters once fp16 makes the
  GEMMs the bottleneck); 4096 gains nothing. `QWEN_CLEAR_CACHE` 0/1 is ±2%.
  Slab shrinks with context (`PREFILL_BUDGET`) — a fixed 2048 at 120K peaked at
  26.7 GB, the adaptive slab 19.9 GB at the same speed (453 s). Prefill is MLP-bound (~60%), GDN ~22%, attention ~7% at 2K, growing
  to ~18% at 32K (quadratic). 260 tok/s at 120K, 500 at 8K.
- **KV quantization costs speed** on this model: q8 vs fp16 cache at 32K is −12%
  prefill and −15% decode. The 9B runs fp16 KV (`QWEN_KV_BITS=0`, 32 KB/token,
  ~4 GB @128K). The 27B/35B keep q8/q4 for memory.
- **Dead ends (measured 2026-09-18, don't retry blindly)**: (1) dequantize→fp16
  GEMM instead of 8-bit qmm on prefill slabs: +12% on up/gate, −3% on down, ~5%
  of MLP — the GPU is at ~8 TFLOPS either way. (2) MLX ≤0.32.2 has **no fused
  SDPA for head_dim 256** (q_len > 8): the fallback materializes scores (8 GB at
  slab 2048 × 120K — the real reason for `PREFILL_BUDGET`) and is 1.7–1.9×
  slower than the fused kernel at equal FLOPs (232 vs 136 ms, slab 512 × 120K);
  attention is ~half of a 120K prefill. MLX `main` lists 256 in the full kernel
  — **bump mlx when it ships** (expect ~−20% prefill at 120K and slab 2048 at
  any context); building main needs full Xcode (`metal` compiler; CLT lacks it).
  (3) 8-bit qmm cost by rows on M1: M=2 ×1.17, M=4 ×1.97, M=6…32 flat ×2.6
  (dequantization-bound) — a custom verify kernel has ≤1.3× headroom. (4) DFlash
  with fallback to plain decoding on low acceptance: loses (8-round probe ≈0.5 s,
  drafter restarts cold); spec for sampling requests is exact but only ×1.1–1.2
  on code / ×0.8 on prose → opt-in `QWEN_DRAFT_SAMPLING=1`.
- **Logits on prefill slabs are never computed** (MLX is lazy; `skip_logits=True`
  is passed anyway to keep the graph small) — don't expect a win there.
- **DFlash speculative decoding** (`QWEN_DRAFT=weights/qwen35-9b-dflash`, the
  z-lab drafter for base Qwen3.5-9B, 2.5 GB bf16, quantized to 4-bit at load via
  `QWEN_DRAFT_BITS`): works on Ornith with ~60% token acceptance (mean 3–5
  accepted per round), output identical to plain greedy on the test prompts
  (the target verifies, so drafter precision only moves speed). Decode
  ×1.43–1.64 on the 5 coding tasks (33 → 47–53 tok/s), 63.7 tok/s end-to-end on
  a code answer (67% acceptance), ~120 tok/s on trivially predictable text;
  holds at 8K context. Tiny answers (<10 tokens) come out slower — first rounds
  have no context. Block size: adaptive (default) ≥ fixed 4/8/16/32. The
  ceiling is an MLX kernel property: 8-bit qmm with 4–16 rows costs ~2.3× a
  1-row qmm on M1 (M=32 drops to 1.7×), so a 16-token verify forward ≈ 70 ms vs
  30 ms decode; the 4-bit drafter block ≈ 15 ms (lm_head over 15 rows alone
  9 ms). Greedy-only (temperature 0, presence_penalty 0), text-only, no
  thinking budget, fp16 KV (`QWEN_KV_BITS=0`, else the drafter is ignored).

## Implementation notes worth keeping

- **Context = 128K** (`MAX_CONTEXT=131072`, prompt + generation). **KV cache is
  per-model** via `QWEN_KV_BITS` (default 4) — `stream_generate(..., kv_bits=N,
  kv_group_size=64, quantized_kv_start=5000)` (MLX analog of llama.cpp
  `--cache-type-k`; K and V uniform, fp16 warmup for the first 5000 tokens). Pick N
  by weight precision + budget: **35B (3-bit, ~18 GB) → q4** (~2.6 GB KV @ 128K; q8
  would blow the 32 GB budget); **9B (8-bit, ~10 GB) → q8** (~8 GB KV @ 128K, ~18 GB
  total, near-lossless — no reason to compress to q4); **27B (5-bit, ~18 GB) → q8**
  (~3.9 GB @128K, ~24 GB total). Only full-attention layers (every 4th) hold KV:
  q8 ≈ 16 KB/token (9B) / 32 KB/token (27B). Env sets sampler defaults;
  requests may override temperature/top_p/top_k/min_p/presence_penalty (proto
  `optional` fields; absent = env default, temperature -1 / top_p 0 likewise).
- **The worker owns its prefill/decode loop** (`Engine._run`), a trimmed copy of
  mlx-vlm 0.6.2 `generate_step`+`stream_generate` (same sampler, logits
  processors, KV quantization, ThinkingBudgetCriteria, detokenizer). Reasons:
  the prompt-cache snapshot has to be taken at the exact point where the cache
  holds prompt[:-1] and nothing else (the library's `PromptCacheState` snapshots
  after the first generated token and trims — impossible for ArraysCache,
  broken for quantized tuples); pre-tokenized vision inputs go in directly;
  `finish_reason` `"length"` and pool clearing are ours. Output verified
  byte-identical to `stream_generate` at temperature 0. When bumping mlx-vlm,
  diff `generate/ar.py::generate_step` against `_run`.
- **Prompt cache** (`QWEN_PROMPT_CACHE=1`): multi-slot prefix cache
  (`_PrefixCache`/`_Entry`). KV is sliceable to any prefix; the GDN recurrent
  state is not, so `_run`/`_run_spec` checkpoint it (~52 MB, fp32 32×128×128 ×
  24 layers on the 9B) at slab ends forced onto `cuts`: every 2048 tokens,
  `<|im_start|>` boundaries (end of system block, last two messages) and the
  divergence point from the closest entry. Hit = deepest checkpoint ≤ longest
  common prefix with any entry (`_copy_cache(upto=, states=)`). LRU by
  `QWEN_PROMPT_CACHE_GB` (3). Never trim with the library; never share mx.array
  *objects* between cache copies (`a[i] = v` rebinds the Python object in place
  — slice to get a new one, buffers are copy-on-write). On a hit set
  `language_model._rope_deltas` to zeros (text mRoPE continues from the cache
  offset; a previous vision request leaves its own delta there). Text-only.
  7.5K shared system prompt, new conversation: TTFT 14.4 → 0.20 s, byte-identical.
- **Buffer pool**: `QWEN_CLEAR_CACHE` (1) / `QWEN_CACHE_LIMIT_GB` (3, via
  `mx.set_cache_limit` at load). See README "Context & memory" for the numbers.
- **Stop sequences**: `_StopFilter` in the chunk loop keeps back the longest
  stream tail that is a prefix of a stop string; earliest match wins; applied to the
  whole output, reasoning included (vLLM's detokenizer does the same).
- **OOM avoidance on 32 GB** (all three needed for the 35B at long context):
  `sudo sysctl iogpu.wired_limit_mb=28672` (kernel cap) + `QWEN_WIRED_LIMIT_GB=27`
  (MLX wired limit, clamped to cap) + `QWEN_PREFILL_STEP=512` (chunked prefill
  bounds the activation peak) + `mx.clear_cache()` after every generation (else
  request N's buffer pool carries into request N+1's prefill → the second-request
  OOM). Verified: 56K-token prompt × 2 back-to-back, no OOM.
- **Reasoning budget = native mlx-vlm `ThinkingBudgetCriteria`** (forces `\n</think>`
  into the token stream; KV cache stays consistent). Do NOT hand-splice + restart
  generation — the earlier crutch corrupted answers and is removed. Thinking on/off
  is the template's `enable_thinking` (gateway: chat_template_kwargs / reasoning_effort
  / GATEWAY_THINKING; unset = template default = thinking). The template *pre-opens*
  `<think>`, so the worker decides "generation starts inside reasoning" from the
  prompt ids (last of `<think>`/`</think>` is an open one), not from a flag.
  `_GenState` = vLLM's `qwen3` reasoning parser (see README).
- **Prompt-length validation** (worker, before generation → `context_length_exceeded`
  → HTTP 400): count image tokens via `mlx_vlm.utils.prepare_inputs(...).input_ids.shape[-1]`,
  NOT a text-only `encode` — a 4096×2645 image expands to ~10644 tokens (text-only
  would say ~15). **No text fallback for vision** (would undercount → OOM); raises
  `prompt_count_failed` instead. Text-only uses a plain `encode`.
- **Tool calls**: Qwen3.5 emits `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n
  </parameter>\n</function>\n</tool_call>`. `parse_tool_calls` is a port of vLLM's
  `qwen3_xml` (vllm/parser/qwen3.py + parser_engine coercion, v0.30): one newline
  trimmed each side, types only from the tool schema (`coerce_to_schema_type` order
  null > integer > number > boolean > object/array > string, else json.loads, else
  the string). No `json.loads` of schema-less values — `"false"` stays a string.
- **Behaviour reference = vLLM v0.30** (`vllm serve` + card recipe: qwen3_xml tool
  parser, qwen3 reasoning parser). Anything vLLM does that we don't is a 400, never
  "accept and ignore"; vLLM-unknown fields are ignored + logged (as vLLM, which logs
  at debug). Request checks: gateway `request.rs` (field table, SamplingParams ranges,
  tool_choice rules) + worker `Check` RPC (template, context length, grammars, images
  on text-only) before queueing → 400 for streams too.
- **Sampler** = vLLM order on float32: penalties (presence/frequency over generated
  tokens, repetition over prompt+output; `_Penalties`) → temperature → min_p → top_k →
  top_p → draw (`make_sampler`, per-request key). Greedy stays argmax of
  `logits - logsumexp` in the model dtype — byte-identical to the old path. The old
  512-token presence window is gone (vLLM has none).
- **Structured output / constrained tools** = **llguidance** (already an mlx-vlm 0.6.2
  dependency; Metal bitmask kernel; no grammar compile step, mask ~0.05–0.2 ms on the
  248K vocab; best JSON Schema coverage). `_Guide` masks logits before the sampler, so
  seed/sampler semantics are unchanged. The mask for token n+1 needs token n, so the
  guided loop does **not** queue the next step ahead (a step-ahead + separate masked
  sample measured 18 tok/s and degrading; mask-in-step is 30 vs 31 plain). Thinking:
  grammar activates on `</think>` (or a `<tool_call>` ending reasoning), before it only
  EOS is masked. JSON whitespace free (llguidance's lexer skips whitespace, so a
  `whitespace_pattern` does NOT bound runs — `QWEN_JSON_MAX_WS` masks by leading
  whitespace per token instead). Tool grammars (`_grammar`, Lark): one rule per
  function in xgrammar's `qwen_3_coder` layout, params in schema order, string values
  lazy raw text up to `\n</parameter>\n`, others `%json`. llguidance treats every
  added token as special: `<tool_call>` goes by id (`<[248058]>`), and free-text
  regexes can never produce it. DFlash is skipped for grammar requests.
  `required` = call first, no text before (parallel_tool_calls false: one call, nothing
  after); named = exactly that call; checked against xgrammar 0.2.8's own
  `get_model_structural_tag("qwen_3_coder")`. Schemas go through `_strict`
  (xgrammar strict_mode: no extra keys unless the schema allows them).
- **Request shaping as vLLM's renderer**: `developer` → `system` + all system messages
  merged at the front (`_developer_to_system`, renderers/hf.py); chat_template_kwargs:
  `chat_template`/`tokenize` → 400, keys not in the template dropped (logged), the
  request's `add_generation_prompt`/`continue_final_message` win (gateway). `Check`
  never 502s on a worker exception: template errors → 400, anything else → 500.
  No tools: `<tool_call>` + all text after it is cut from the answer (vLLM's parser).
- **Key order**: gateway serde_json has `preserve_order` — tools, tool_call history,
  response_format reach the worker in the client's order (sorted before: a tools
  prompt changed under the client's feet; schema property order is what the grammar
  emits).
- **Model id** = the worker's weights dir name (Health.model, Start.model, Check.model);
  `model` in a request must be it or a GATEWAY_SERVED_MODEL_NAMES alias, else 404
  (vLLM). `MODEL_NAME` is gone.
- **Gateway `ReqGuard`** (Drop) owns both semaphore permits — frees them, refreshes
  the queue/in-flight gauges, AND fires `Worker.Cancel` on client disconnect. Without
  owning the permits there, `queue_depth` stuck non-zero after a mid-stream disconnect.

## Env knobs

| var | where | default |
|---|---|---|
| `QWEN_MODEL` | worker | `weights/qwen36-35b-a3b-mlx-ud3` (repo-relative) |
| `WORKER_ADDR` / `GATEWAY_ADDR` / `MAX_QUEUE` | gateway | `127.0.0.1:50051` / `0.0.0.0:8080` / `32` |
| `RUST_LOG` / `LOG_FORMAT` | gateway | `info` / pretty (set `json` for Loki) |
| `LOG_LEVEL` | worker | `INFO` |
| `QWEN_MAX_CONTEXT` | worker | `131072` |
| `QWEN_KV_BITS` | worker | `4` (35B); `0` (fp16 cache) for the 9B; `8` for the 27B |
| `QWEN_FP16` | worker | `1` (cast bf16→fp16 after load; `0` = bf16) |
| `QWEN_DRAFT` / `QWEN_DRAFT_BLOCK` | worker | unset / `0` — DFlash drafter dir (`weights/qwen35-9b-dflash`), speculative decode for greedy text requests |
| `QWEN_WIRED_LIMIT_GB` / `QWEN_PREFILL_STEP` | worker | `0` (off; clamped to cap) / `2048`, auto-shrunk so slab×offset ≤ `QWEN_PREFILL_BUDGET` (2048×32K) |
| `QWEN_TEMPERATURE` / `QWEN_TOP_P` / `QWEN_TOP_K` / `QWEN_MIN_P` | worker | `0.0` (argmax) / `1.0` / `0` / `0.0` |
| `QWEN_PRESENCE_PENALTY` / `QWEN_FREQUENCY_PENALTY` / `QWEN_REPETITION_PENALTY` | worker | `1.5` / `0` / `1.0` — **benchmarks/reasoning: presence `0`** + card sampler (see README) |
| `QWEN_JSON_MAX_WS` | worker | `0` (off) — whitespace-run cap in response_format answers |
| `GATEWAY_OPENAI_DEFAULTS` / `GATEWAY_THINKING` / `GATEWAY_SERVED_MODEL_NAMES` / `GATEWAY_DEFAULT_MAX_TOKENS` | gateway | unset / unset (`on`/`off` forces) / unset / `0` (= rest of context) — see README |
| `QWEN_PROMPT_CACHE` / `_GB` / `_STRIDE` / `_CKPTS` | worker | `0` / `3` / `2048` / `12` |
| `QWEN_DRAFT_SAMPLING` | worker | `0` |
| `QWEN_CLEAR_CACHE` / `QWEN_CACHE_LIMIT_GB` | worker | `1` / `3` |
| `MAC_EXPORTER_ADDR` | exporter | `0.0.0.0:9101` (run under sudo) |

## Gotchas

- **MLX random state is per thread, and mlx_lm's sampler is compiled against the
  main thread's key.** gRPC handlers run in pool threads, so `mlx_lm.make_sampler`
  there drew every token from the same never-advancing key: temperature 0.7
  returned byte-identical answers, and `mx.random.seed` from the handler thread
  changed nothing. The worker's own `make_sampler` (worker.py) reuses mlx_lm's
  top_p/min_p/top_k filters but draws with an explicit per-request key split
  per step; request `seed` feeds that key (reproducible, verified through the
  gateway). Test any sampler change from a thread, not the REPL main thread.
- DFlash greedy and the plain loop are greedy-equivalent only up to floating
  point: the 16-row verify qmm and the 1-row decode qmm round differently, so
  on a near-tie they can pick different tokens (seen at token ~40 of one
  prompt: `temperature 0` vs `top_k 1`). Both are deterministic.
- DFlash's verify block is adaptive on acceptance stats kept across requests, so the
  same greedy request can round differently (near-tie flip) between runs: seen once
  in 2026-10 (first pair differed, the next 5 identical). A fixed `QWEN_DRAFT_BLOCK`
  should avoid it (not verified); the plain loop is unaffected.
- The gateway waits for the worker's Health before it listens: no worker = no 8080.
- A scheme-less proxy variable (`HTTP_PROXY=localhost:<port>`) → breaks Python `requests`.
  Use `http://...` or `HF_HUB_OFFLINE=1` (models are cached locally in `weights/`).
- mac-exporter needs root (`powermetrics`); load it as a `/Library/LaunchDaemons`
  daemon, not a user agent.
- Observability stack is shared infra — `deploy/` containers are `deploy-*`; don't
  touch the user's other Docker containers.
- The venv breaks on a repo move (absolute paths baked in) — `uv sync` at the root
  and in `src/worker/` to repair.
