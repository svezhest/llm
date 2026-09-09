# CLAUDE.md — llm

Local LLM inference on Apple Silicon (MLX). **Rust gateway** (OpenAI-compatible
HTTP/SSE) ⇄ **gRPC** ⇄ thin **Python MLX worker(s)**. Metrics (Prometheus),
Grafana dashboards, Loki logs, request queue, leveled logging. Inference runs
**native** (Metal not in Docker); only the observability stack is containerized.
See `README.md` for run instructions, `BACKLOG.md` for the plan + phase status.

## Layout (repo root = `/Users/user/Projects/llm`)

- `weights/` — model dirs (git-ignored). `qwen36-35b-a3b-mlx-ud3`, `ornith10-9b-mlx-8bit`,
  `qwen38-27b-mlx-5bit`. Pull with `scripts/hf_pull.sh <repo> <dir>` (resumable aria2c).
- `src/proto/worker.proto` — gRPC contract (single source of truth; both sides codegen).
- `src/gateway/` (Rust) — axum HTTP + tonic gRPC client + queue + metrics + tracing.
  `src/{main,api,pool,metrics,worker_client,logging,pb}.rs`.
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
- **Prompt cache** (`QWEN_PROMPT_CACHE=1`): single `_Snapshot(ids[:-1], cache
  copy)`. Hit = new ids strictly extend the snapshot ids; delta-only prefill from
  a `_copy_cache` of it. Never trim; never share mx.array *objects* between cache
  copies (`a[i] = v` rebinds the Python object in place — slice to get a new one,
  buffers are copy-on-write). On a hit set `language_model._rope_deltas` to zeros
  (text mRoPE continues from the cache offset; a previous vision request leaves
  its own delta there). Text-only conversations only.
- **Buffer pool**: `QWEN_CLEAR_CACHE` (1) / `QWEN_CACHE_LIMIT_GB` (3, via
  `mx.set_cache_limit` at load). See README "Context & memory" for the numbers.
- **Stop sequences**: `_StopFilter` in the chunk loop keeps back the longest
  stream tail that is a prefix of a stop string; applied to answer text only.
- **OOM avoidance on 32 GB** (all three needed for the 35B at long context):
  `sudo sysctl iogpu.wired_limit_mb=28672` (kernel cap) + `QWEN_WIRED_LIMIT_GB=27`
  (MLX wired limit, clamped to cap) + `QWEN_PREFILL_STEP=512` (chunked prefill
  bounds the activation peak) + `mx.clear_cache()` after every generation (else
  request N's buffer pool carries into request N+1's prefill → the second-request
  OOM). Verified: 56K-token prompt × 2 back-to-back, no OOM.
- **Reasoning budget = native mlx-vlm `thinking_budget` kwarg** (forces `\n</think>`
  into the token stream; KV cache stays consistent). Do NOT hand-splice + restart
  generation — the earlier crutch corrupted answers and is removed. Also pass
  `enable_thinking`; the chat template *pre-opens* `<think>` so generation STARTS
  inside the think block (literal `<think>` never emitted; `_GenState` seeds
  `in_thinking=enable_thinking`). Thinking → `delta.reasoning_content`, answer → `delta.content`.
- **Prompt-length validation** (worker, before generation → `context_length_exceeded`
  → HTTP 400): count image tokens via `mlx_vlm.utils.prepare_inputs(...).input_ids.shape[-1]`,
  NOT a text-only `encode` — a 4096×2645 image expands to ~10644 tokens (text-only
  would say ~15). **No text fallback for vision** (would undercount → OOM); raises
  `prompt_count_failed` instead. Text-only uses a plain `encode`.
- **Tool calls**: Qwen3 emits them inside `<tool_call>…</tool_call>`; this MLX build
  uses the **Hermes-XML** payload form (`<function=NAME><parameter=KEY>V</parameter>…`),
  not JSON. Parser handles both.
- **Gateway `ReqGuard`** (Drop) owns both semaphore permits — frees them, refreshes
  the queue/in-flight gauges, AND fires `Worker.Cancel` on client disconnect. Without
  owning the permits there, `queue_depth` stuck non-zero after a mid-stream disconnect.

## Env knobs

| var | where | default |
|---|---|---|
| `QWEN_MODEL` | worker | `weights/qwen36-35b-a3b-mlx-ud3` (repo-relative) |
| `WORKER_ADDR` / `GATEWAY_ADDR` / `MAX_QUEUE` / `MODEL_NAME` | gateway | `127.0.0.1:50051` / `0.0.0.0:8080` / `32` / `qwen3.6-35b-a3b` |
| `RUST_LOG` / `LOG_FORMAT` | gateway | `info` / pretty (set `json` for Loki) |
| `LOG_LEVEL` | worker | `INFO` |
| `QWEN_MAX_CONTEXT` | worker | `131072` |
| `QWEN_KV_BITS` | worker | `4` (35B); `8` for the 9B and 27B |
| `QWEN_WIRED_LIMIT_GB` / `QWEN_PREFILL_STEP` | worker | `0` (off; clamped to cap) / `512` |
| `QWEN_TEMPERATURE` / `QWEN_TOP_P` / `QWEN_TOP_K` / `QWEN_MIN_P` | worker | `0.0` (argmax) / `1.0` / `0` / `0.0` |
| `QWEN_PRESENCE_PENALTY` / `QWEN_PRESENCE_CONTEXT` | worker | `1.5` / `512` — **benchmarks/reasoning: `0`** + card sampler (see README) |
| `QWEN_PROMPT_CACHE` | worker | `0` |
| `QWEN_CLEAR_CACHE` / `QWEN_CACHE_LIMIT_GB` | worker | `1` / `3` |
| `MAC_EXPORTER_ADDR` | exporter | `0.0.0.0:9101` (run under sudo) |

## Gotchas

- A scheme-less proxy variable (`HTTP_PROXY=localhost:<port>`) → breaks Python `requests`.
  Use `http://...` or `HF_HUB_OFFLINE=1` (models are cached locally in `weights/`).
- mac-exporter needs root (`powermetrics`); load it as a `/Library/LaunchDaemons`
  daemon, not a user agent.
- Observability stack is shared infra — `deploy/` containers are `deploy-*`; don't
  touch the user's other Docker containers.
- The venv breaks on a repo move (absolute paths baked in) — `uv sync` at the root
  and in `src/worker/` to repair.
