# AGENTS.md — qwen-server

Production inference service for the local **Qwen3.6-35B-A3B** MLX VLM.
**Rust gateway** (OpenAI-compatible HTTP/SSE) ⇄ **gRPC** ⇄ thin **Python MLX
worker**. Metrics (Prometheus), Grafana dashboards, Loki logs, request queue,
leveled logging. See `README.md` for run instructions and `../BACKLOG.md` for
the full plan + phase status.

## Status (2026-06-08): working, end-to-end verified

Worker loads in ~6–7 s; gateway streams OpenAI SSE + non-stream; metrics/gauges
correct; cancel-on-disconnect works; reasoning budget + vision validated. Nothing
outstanding blocks use. Inference runs **native** (Metal not in Docker); only the
observability stack is containerized.

## Layout

- `proto/worker.proto` — gRPC contract (single source of truth; both sides codegen).
- `gateway/` (Rust) — axum HTTP + tonic gRPC client + queue + metrics + tracing.
  `src/{main,api,pool,metrics,worker_client,logging,pb}.rs`.
- `worker/` (Python) — grpcio server, mlx-vlm; load + generate only.
- `exporter/` (Rust) — `powermetrics` → `/metrics` (GPU/ANE/watts; needs root).
- `deploy/` — docker-compose (prometheus/grafana/loki/promtail) + launchd plists.

## Run order

```zsh
cd worker && uv run --group proto bash gen_proto.sh    # once: codegen worker_pb2*.py
uv run python worker.py                                  # serves 127.0.0.1:50051 (loads model)
# after "model loaded":
cd .. && cargo build --release
RUST_LOG=info ./target/release/qwen-gateway             # serves 0.0.0.0:8080
# observability (optional): cd deploy && QWEN_LOG_DIR=../logs docker compose up -d
```

Stop everything: `kill $(pgrep -f 'qwen-gateway|python worker.py')`. The MLX
worker holds ~18 GB — kill it to reclaim memory. `ps`/RSS under-reports MLX
(wired memory); use the macOS resource manager / `footprint` to see the real size.

## Key design decisions (locked — don't relitigate)

- **Public API**: OpenAI-compatible HTTP/SSE (`/v1/chat/completions`), cross-OS.
- **Gateway ⇄ worker**: gRPC (tonic ⇄ grpcio), server-streaming `Generate→stream Event`.
- **Queue + 1 MLX worker** (only one ~18 GB copy fits in 32 GB). "pooling" in the
  plan = metrics pooling (Prometheus pull), not a worker pool.
- **Rust everywhere except the MLX worker** (process isolation, no PyO3).
- **Images** = paths/URLs on the wire, never bytes.

## Implementation notes worth keeping

- **KV cache q4 @ 64K**: weights are 3-bit, so a 4-bit KV cache is the balanced
  match — `stream_generate(..., kv_bits=4, kv_group_size=64, quantized_kv_start=5000)`
  (MLX analog of llama.cpp `--cache-type-k q4_0`; applies to K and V uniformly,
  fp16 warmup for the first 5000 tokens). ~20 KiB/token → context fixed at **64K**
  (`MAX_CONTEXT=65536`, ~1.3 GB KV). q8 ≈ 40 KiB/tok; fp16 ≈ 80.
- **Reasoning budget = native mlx-vlm `thinking_budget` kwarg** (forces `\n</think>`
  into the token stream; KV cache stays consistent). Do NOT hand-splice + restart
  generation — the earlier crutch corrupted answers and is removed. Also pass
  `enable_thinking`; the chat template *pre-opens* `<think>` so generation STARTS
  inside the think block (the literal `<think>` token is never emitted; `_GenState`
  seeds `in_thinking=enable_thinking`). Thinking streams as `delta.reasoning_content`,
  answer as `delta.content`.
- **Prompt-length validation** (worker, before generation → `context_length_exceeded`
  → HTTP 400): count image tokens via `mlx_vlm.utils.prepare_inputs(...).input_ids.shape[-1]`,
  NOT a text-only `encode` — a 4096×2645 image expands to ~10644 tokens (text-only
  would say ~15). **No text fallback for vision** (would undercount → OOM); raises
  `prompt_count_failed` instead. Text-only uses a plain `encode`.
- **Gateway `ReqGuard`** (Drop) owns both semaphore permits — frees them, refreshes
  the queue/in-flight gauges, AND fires `Worker.Cancel` on client disconnect. Without
  owning the permits there, `queue_depth` stuck non-zero after a mid-stream disconnect.

## Env knobs

| var | where | default |
|---|---|---|
| `WORKER_ADDR` / `GATEWAY_ADDR` / `MAX_QUEUE` / `MODEL_NAME` | gateway | `127.0.0.1:50051` / `0.0.0.0:8080` / `32` / `qwen3.6-35b-a3b` |
| `RUST_LOG` / `LOG_FORMAT` | gateway | `info` / pretty (set `json` for Loki) |
| `LOG_LEVEL` | worker | `INFO` |
| `MAC_EXPORTER_ADDR` | exporter | `0.0.0.0:9101` (run under sudo) |

## Gotchas

- A scheme-less proxy variable (`HTTP_PROXY=localhost:<port>`) → breaks Python `requests`.
  Use `http://...` or `HF_HUB_OFFLINE=1` (model is cached locally).
- mac-exporter needs root (`powermetrics`); load it as a `/Library/LaunchDaemons`
  daemon, not a user agent.
- Observability stack is shared infra — `deploy/` containers are `deploy-*`; don't
  touch the user's other Docker containers.
