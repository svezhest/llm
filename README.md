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
| `qwen36-35b-a3b-mlx-ud3` | Qwen3.6-35B-A3B | MLX affine g64, **3-bit UD** (mixed: embed 6b, attn/gates 8b, shared-expert 4b, MoE experts 3b) | ~16 GB | VL MoE (256 experts, ~3B active) |

Weights live in `weights/` (git-ignored). MLX loads from a plain path — no HF
cache needed. Download recipe for a slow, throttled link: multi-connection
`aria2c -c -x16 -s16` per shard with `--check-integrity --checksum=sha-256=<lfs.oid>`
(oids from `https://huggingface.co/api/models/<repo>/tree/main`), `HF_HUB_DISABLE_XET=1`,
never hard-kill (orphans partials). See `CLAUDE.md`.

## Layout

- `src/proto/worker.proto` — gRPC contract.
- `src/gateway/` (Rust) — axum HTTP + tonic gRPC client + queue + metrics + tracing.
- `src/worker/`  (Python) — grpcio server, mlx-vlm; load + generate + reasoning budget.
- `src/exporter/` (Rust) — `powermetrics` → `/metrics` (GPU/ANE/mem/watts).
- `deploy/` — docker-compose (prometheus/grafana/loki/promtail) + launchd plists.
- `Cargo.toml` — Rust workspace (members `src/gateway`, `src/exporter`).
- `pyproject.toml` — root tooling/download env; `src/worker/` is its own uv project.

## Run (dev)

The **shipped default is Ornith 9B — 128K context, q8 KV** (light: ~18 GB
resident, no wired-limit tuning needed). The 35B is the heavier alternative below.

1. **Worker** — run the 9B (override the built-in default 35B path via `QWEN_MODEL`):
   ```zsh
   cd src/worker
   uv run --group proto bash gen_proto.sh    # once: codegen worker_pb2*.py
   QWEN_MODEL=../../weights/ornith10-9b-mlx-8bit QWEN_KV_BITS=8 \
     uv run python worker.py                  # serves 127.0.0.1:50051
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

Vision (either model): put an `image_url` part in the message content. Reasoning:
`"enable_thinking": true, "thinking_budget": 256` (0 = unlimited). Thinking streams
as `delta.reasoning_content`, the answer as `delta.content`.

Quick CLI (no gateway): `uv run python chat.py "Привет!" --model weights/ornith10-9b-mlx-8bit`.

## Context & memory

Context is capped at **128K tokens** (`MAX_CONTEXT=131072`, prompt + generation).

**KV cache precision is per-model** (`QWEN_KV_BITS`, default 4), because the right
value tracks the *weight* precision and the memory budget:

| model | weights | KV (`QWEN_KV_BITS`) | KV @ 128K | resident @ 128K | why |
|---|---|---|---|---|---|
| Qwen 35B-A3B | 3-bit, ~18 GB | **q4** (default) | ~2.6 GB | ~21 GB (tight) | 4-bit KV is the balanced match for 3-bit weights; q8 KV would blow the 32 GB budget |
| **Ornith 9B** | 8-bit, ~10 GB | **q8** | ~8 GB | **~18 GB** | 8-bit weights + tons of headroom → run near-lossless q8 KV; no reason to compress the cache to 4-bit and eat the quality hit |

So the 9B runs `QWEN_KV_BITS=8` at the full 128K — that's the configuration we
ship for it. Its head_dim is 256 (2× a typical model), so KV is the dominant cost,
not the weights; even at q8 it's only ~8 GB, and the whole thing fits in ~18 GB with
no `sudo sysctl` needed. (256K would also fit at q8, ~26 GB, but needs the wired bump
and raising `MAX_CONTEXT`.)

Oversized prompts are rejected with **HTTP 400 `context_length_exceeded`** before
any allocation (no OOM). Image tokens are counted for real (a 4K image ≈ 10k
tokens), so vision prompts are validated accurately, not undercounted.

**Fitting the 35B long context on 32 GB.** Weights are ~16–18 GB; the default
Metal wired limit (~21.8 GB) leaves almost nothing for KV + prefill activations,
so big prompts OOM. Three things keep it alive:
- **`sudo sysctl iogpu.wired_limit_mb=28672`** raises the kernel working-set cap
  to 28 GB (resets on reboot). `QWEN_WIRED_LIMIT_GB` then pins MLX's wired limit
  (clamped to the cap — set it *below* the sysctl value, e.g. 27).
- **Chunked prefill** (`QWEN_PREFILL_STEP=512`) bounds the prefill activation
  peak so a 50K-token prompt doesn't spike memory in one matmul.
- **`mx.clear_cache()` runs after every generation**, so MLX's buffer pool from
  one request never carries into the next request's prefill (the cause of the
  second-request OOM). Verified: 56K-token prompt × 2 back-to-back, no OOM.

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
| `QWEN_KV_BITS` | worker | `4` (35B); set `8` for the 9B |
| `QWEN_WIRED_LIMIT_GB` | worker | `0` (off; clamped to kernel cap) |
| `QWEN_PREFILL_STEP` | worker | `512` |
| `QWEN_TEMPERATURE` | worker | `0.0` (argmax) |
| `QWEN_PRESENCE_PENALTY` | worker | `1.5` |
| `QWEN_PRESENCE_CONTEXT` | worker | `512` |
| `MAC_EXPORTER_ADDR` | exporter | `0.0.0.0:9101` |
