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
   uv run --group proto bash gen_proto.sh    # once: codegen worker_pb2*.py
   QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 \
     uv run python worker.py                  # serves 127.0.0.1:50051
   # optional, greedy requests only: +DFlash speculative decoding (~1.4–1.6x decode)
   #   QWEN_DRAFT=../../weights/qwen35-9b-dflash
   ```

2. **Gateway** (after the worker logs "model loaded"):
   ```zsh
   cargo build --release
   RUST_LOG=info MODEL_NAME=ornith10-9b ./target/release/qwen-gateway   # 0.0.0.0:8080
   ```

3. **Call it** (OpenAI-compatible):
   ```zsh
   curl -N http://localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{
     "model": "ornith10-9b", "stream": true,
     "messages": [{"role":"user","content":"Привет! Кто ты?"}]
   }'
   ```

**Heavier alternative — Qwen 35B-A3B** (the worker's built-in default path). Needs
the Metal wired-limit bump first (once per boot, resets on reboot — see
*Context & memory*), and runs q4 KV:
```zsh
sudo sysctl iogpu.wired_limit_mb=28672    # 28 GB cap (default ~21.8 GB)
QWEN_WIRED_LIMIT_GB=27 QWEN_PREFILL_STEP=512 uv run python worker.py   # q4 KV (default)
# gateway: leave MODEL_NAME at its default qwen3.6-35b-a3b; call with that model name.
```

**Qwen3.8-27B (5-bit)** — the strongest dense option; ~24 GB resident at 128K, so
it needs the same wired-limit bump as the 35B, but runs **q8 KV**:
```zsh
sudo sysctl iogpu.wired_limit_mb=28672                 # once per boot
cd src/worker
QWEN_MODEL=../../weights/qwen38-27b-mlx-5bit QWEN_KV_BITS=8 \
  QWEN_WIRED_LIMIT_GB=27 QWEN_PREFILL_STEP=512 uv run python worker.py
# gateway (after "model loaded"):
RUST_LOG=info MODEL_NAME=qwen3.8-27b ./target/release/qwen-gateway
```
`MODEL_NAME` is only the label the gateway answers to (`"model"` in requests and
`/v1/models`); the worker's `QWEN_MODEL` path decides what's loaded. Keep the pair
consistent: `ornith10-9b` ⇄ `ornith10-9b-mlx-8bit`, `qwen3.6-35b-a3b` ⇄
`qwen36-35b-a3b-mlx-ud3`, `qwen3.8-27b` ⇄ `qwen38-27b-mlx-5bit`.

**Sampling**: env sets the process-wide defaults; a request may override
`temperature`, `top_p`, `top_k`, `min_p` and `presence_penalty` (the last three
are the usual llama.cpp/vLLM extensions); `seed` makes a sampled request
reproducible (same seed + params ⇒ same text). `stop` (string or array) is honoured:
the worker holds back the stream tail until it knows a stop sequence is not
starting there, so a stop split across tokens is caught and never emitted;
`finish_reason` is `"stop"` for EOS/stop-string, `"length"` for `max_tokens`.
Stop sequences apply to the answer, not to `reasoning_content`. The shipped
default is argmax + presence penalty 1.5 over 512 tokens — the Qwen card's
*non-thinking* setting, and **wrong for reasoning and benchmarks**: it
suppresses repeated tokens inside long `<think>` traces. For evals run the
card's thinking-mode sampler:
```zsh
QWEN_PRESENCE_PENALTY=0 QWEN_TEMPERATURE=1.0 QWEN_TOP_P=0.95 QWEN_TOP_K=20 QWEN_MIN_P=0 \
  QWEN_MODEL=... uv run python worker.py
```
(Qwen3.8 card: thinking = T 1.0 / top_p 0.95 / top_k 20 / min_p 0 / presence 0;
instruct = T 0.7 / top_p 0.80 / top_k 20 / presence 1.5.)

Vision (any model): put an `image_url` part in the message content. Reasoning:
`"enable_thinking": true, "thinking_budget": 256` (0 = unlimited). Thinking streams
as `delta.reasoning_content`, the answer as `delta.content`.

Quick CLI (no gateway): `uv run python chat.py "Привет!" --model weights/ornith10-9b-mlx-8bit`.

## Prompt cache (`QWEN_PROMPT_CACHE=1`, default off)

Multi-slot **prefix cache** for text-only prompts. Each entry is the cache
state after the prefill of `prompt[:-1]`. Full-attention KV can be cut to any
prefix by slicing; the gated-delta-net recurrent state cannot be rewound, so it
is **checkpointed during prefill** (~52 MB each on the 9B): every
`QWEN_PROMPT_CACHE_STRIDE` (2048) tokens, at message boundaries (end of the
system/tools block, start of the last two messages) and at the point where the
prompt diverged from the closest cached one. Prefill slabs are cut on those
positions, so a checkpoint is free. A new prompt resumes from the deepest
checkpoint inside its longest common prefix with any entry — so a different
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
→ 0.32 s. Resuming from a checkpoint changes slab boundaries, so logits move by
fp16 noise — on a near-tie the greedy token can differ from a cold run (seen
once with the DFlash path). Logs: `prompt cache HIT n/m tokens reused (common
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
| `MODEL_NAME` | gateway | `qwen3.6-35b-a3b` |
| `RUST_LOG` / `LOG_FORMAT` | gateway | `info` / pretty |
| `LOG_LEVEL` | worker | `INFO` |
| `QWEN_MAX_CONTEXT` | worker | `131072` |
| `QWEN_KV_BITS` | worker | `4` (35B); `0` = fp16 cache for the 9B, `8` for the 27B |
| `QWEN_FP16` | worker | `1` — cast bf16 params to fp16 (M1/M2 have no native bf16: prefill ×1.45; KL vs bf16 0.0003). `0` = bf16 |
| `QWEN_DRAFT` / `QWEN_DRAFT_BLOCK` | worker | unset / `0` (adaptive) — DFlash drafter dir for speculative decoding (greedy, text-only, fp16 KV) |
| `QWEN_WIRED_LIMIT_GB` | worker | `0` (off; clamped to kernel cap) |
| `QWEN_PREFILL_STEP` / `QWEN_PREFILL_BUDGET` / `QWEN_PREFILL_MIN_STEP` | worker | `2048` / `67108864` (2048×32K) / `256` — slab shrinks so slab×context ≤ budget |
| `QWEN_TEMPERATURE` / `QWEN_TOP_P` / `QWEN_TOP_K` / `QWEN_MIN_P` | worker | `0.0` (argmax) / `1.0` / `0` / `0.0` (off) |
| `QWEN_PRESENCE_PENALTY` | worker | `1.5` (set `0` for reasoning/benchmarks) |
| `QWEN_PRESENCE_CONTEXT` | worker | `512` |
| `QWEN_PROMPT_CACHE` | worker | `0` (set `1`: multi-slot prefix cache with recurrent-state checkpoints) |
| `QWEN_PROMPT_CACHE_GB` / `_STRIDE` / `_CKPTS` | worker | `3` / `2048` / `12` — LRU budget, checkpoint stride, checkpoints per entry |
| `QWEN_DRAFT_SAMPLING` | worker | `0` (set `1`: speculative path also for temperature/presence-penalty requests, exact sample-match; code ×1.1–1.2, prose ×0.8) |
| `QWEN_CLEAR_CACHE` / `QWEN_CACHE_LIMIT_GB` | worker | `1` / `3` (drop MLX buffer pool after each request / pool cap) |
| `MAC_EXPORTER_ADDR` | exporter | `0.0.0.0:9101` |
