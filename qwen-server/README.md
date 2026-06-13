# qwen-server

Production inference service for the local Qwen3.6-35B-A3B MLX model:
a **Rust gateway** (OpenAI-compatible HTTP/SSE) talking to a **thin Python MLX
worker** over **gRPC**, with Prometheus metrics, Grafana dashboards, Loki logs,
a request queue, and leveled logging.

```
clients ──HTTP/SSE (OpenAI)──► qwen-gateway (Rust) ──gRPC stream──► mlx-worker (Python)
                                  /metrics, queue, logs              mlx-vlm load + generate
        Prometheus ◄─scrape─ gateway + mac-exporter ; Grafana ◄ Prometheus + Loki ◄ Promtail
```

Why this split: Rust for everything (API, queue, metrics, logging, gRPC); Python
only where MLX must load. Process isolation, no PyO3. The `.proto` is the single
source of truth. Inference stays native on the host (Metal is not in Docker);
only the observability stack is containerized.

## Layout
- `proto/worker.proto` — gRPC contract.
- `gateway/` (Rust) — axum HTTP + tonic gRPC client + queue + metrics + tracing.
- `worker/` (Python) — grpcio server, mlx-vlm; load + generate + reasoning budget.
- `exporter/` (Rust) — `powermetrics` → `/metrics` (GPU/ANE/mem/watts).
- `deploy/` — docker-compose (prometheus/grafana/loki/promtail) + launchd plists.

## Run (dev)

0. **Raise the Metal wired limit** (once per boot, needed for long context on
   32 GB — see *Context & memory* below). Resets on reboot:
   ```zsh
   sudo sysctl iogpu.wired_limit_mb=28672    # 28 GB cap (default ~21.8 GB)
   ```

1. **Worker** (loads the model, ~18 GB):
   ```zsh
   cd worker
   uv run --group proto bash gen_proto.sh   # once: codegen worker_pb2*.py
   QWEN_WIRED_LIMIT_GB=27 QWEN_PREFILL_STEP=512 \
     uv run python worker.py                  # serves 127.0.0.1:50051
   ```

2. **Gateway** (after the worker logs "model loaded"):
   ```zsh
   cargo build --release
   RUST_LOG=info ./target/release/qwen-gateway   # serves 0.0.0.0:8080
   ```

3. **Call it** (OpenAI-compatible):
   ```zsh
   curl -N http://localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{
     "model": "qwen3.6-35b-a3b",
     "stream": true,
     "messages": [{"role":"user","content":"Привет! Кто ты?"}]
   }'
   ```
   Vision: put an `image_url` part in the message content. Reasoning:
   `"enable_thinking": true, "thinking_budget": 256` (0 = unlimited). Thinking
   text streams as `delta.reasoning_content`, the answer as `delta.content`.

## Context & memory

Context is capped at **128K tokens** (`MAX_CONTEXT=131072`, prompt + generation).
The weights are 3-bit, so the worker runs a **q4 KV cache** (`kv_bits=4`, ~20 KiB/
token → ~2.6 GB at 128K) — the balanced match for a 3-bit model on 32 GB.

Oversized prompts are rejected with **HTTP 400 `context_length_exceeded`** before
any allocation (no OOM). Image tokens are counted for real (a 4K image ≈ 10k
tokens), so vision prompts are validated accurately, not undercounted.

**Fitting long context on 32 GB.** Weights are ~18 GB; the default Metal wired
limit (~21.8 GB) leaves almost nothing for KV + prefill activations, so big
prompts OOM. Three things keep it alive:
- **`sudo sysctl iogpu.wired_limit_mb=28672`** raises the kernel working-set cap
  to 28 GB (resets on reboot). `QWEN_WIRED_LIMIT_GB` then pins MLX's wired limit
  (clamped to the cap — set it *below* the sysctl value, e.g. 27).
- **Chunked prefill** (`QWEN_PREFILL_STEP=512`) bounds the prefill activation
  peak so a 50K-token prompt doesn't spike memory in one matmul.
- **`mx.clear_cache()` runs after every generation**, so MLX's buffer pool from
  one request never carries into the next request's prefill (the cause of the
  second-request OOM). Verified: 56K-token prompt × 2 back-to-back, no OOM.

See **AGENTS.md** for design decisions, implementation notes, and gotchas.

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
| `WORKER_ADDR` | gateway | `127.0.0.1:50051` |
| `GATEWAY_ADDR` | gateway | `0.0.0.0:8080` |
| `MAX_QUEUE` | gateway | `32` |
| `MODEL_NAME` | gateway | `qwen3.6-35b-a3b` |
| `RUST_LOG` / `LOG_FORMAT` | gateway | `info` / pretty |
| `LOG_LEVEL` | worker | `INFO` |
| `QWEN_WIRED_LIMIT_GB` | worker | `0` (off; clamped to kernel cap) |
| `QWEN_PREFILL_STEP` | worker | `512` |
| `QWEN_TEMPERATURE` | worker | `0.0` (argmax) |
| `QWEN_PRESENCE_PENALTY` | worker | `1.5` |
| `QWEN_PRESENCE_CONTEXT` | worker | `512` |
| `MAC_EXPORTER_ADDR` | exporter | `0.0.0.0:9101` |
