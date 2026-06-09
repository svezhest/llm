# Backlog

---

# PRIORITY: qwen-server (production inference service)

Replace the test CLI (`chat.py`) with a real service: Rust gateway + thin Python MLX
worker, talking over **gRPC**, with metrics (Prometheus pull), Grafana, request
queue, and proper leveled logging.

## Locked decisions
- **Public API**: OpenAI-compatible HTTP/SSE (`/v1/chat/completions`) — cross-OS clients (Pi, Open WebUI, curl).
- **Gateway ↔ worker transport**: **gRPC** (tonic in Rust, grpcio in Python). Network + cross-OS + native streaming. Same `.proto` is the single source of truth. Loopback for the local case; ready for remote/heterogeneous workers (Case B) with no redesign.
- **Worker pool**: bounded request **queue + 1 MLX worker** (model ~18–20 GB → only one copy fits in 32 GB). Queue depth + wait time are metrics. (Continuous batching = later, optional.)
- **"pooling" = metrics pooling**: Prometheus scrapes (pulls) the gateway `/metrics`.
- **Images**: passed as **paths/URLs**, not bytes (keeps the wire text-only).
- **Language split**: Rust for everything (gateway, API, pool, metrics, logging, gRPC client); Python only where MLX is loaded (worker = load + generate). No PyO3 — process isolation.

## Architecture
```
            HTTP/SSE (OpenAI API)                gRPC (server-streaming)
 clients ───────────────────────► [qwen-gateway] ───────────────────► [mlx-worker]
 (any OS, remote)                   Rust / axum     Generate stream      Python / grpcio
                                    - OpenAI /v1        of Event           - mlx-vlm load() once
                                    - request queue (1)                    - generate() → tokens
                                    - /metrics (Prometheus)                - Cancel / Shutdown
                                    - tracing logs
                                         ▲ scrape
                    [Prometheus] ────────┘────────── [mac-exporter] (powermetrics: GPU/ANE/mem/watt)
                         │
                    [Grafana]  ◄── dashboards     [Loki] ◄── [Promtail] ◄── gateway+worker logs
```

## Repo layout (proposed)
```
qwen-server/
  proto/worker.proto              # gRPC contract (shared source of truth)
  gateway/         (Rust)         # axum HTTP + tonic gRPC client + queue + metrics + tracing
    Cargo.toml  build.rs  src/{main,api,pool,metrics,worker_client,logging}.rs
  worker/          (Python)       # grpcio server, mlx-vlm; load + generate only
    pyproject.toml  worker.py  gen_proto.sh
  exporter/        (Rust)         # powermetrics → /metrics (mac GPU/ANE/mem/watt)
  deploy/
    docker-compose.yml            # prometheus + grafana + loki + promtail
    prometheus.yml  grafana/dashboards/*.json  loki/  promtail/
    launchd/*.plist               # KeepAlive services for gateway + worker + exporter
  README.md
```

## worker.proto (gRPC service)
```proto
syntax = "proto3";
package qwenworker;

service Worker {
  rpc Generate(Request) returns (stream Event);   // server-streaming token stream
  rpc Cancel(CancelReq) returns (Ack);
  rpc Health(Empty)     returns (HealthReply);
  rpc Shutdown(Empty)   returns (Ack);
}

message Request {
  string id = 1;
  repeated Message messages = 2;       // OpenAI-style chat
  repeated string images = 3;          // PATHS/URLs, not bytes
  Params params = 4;
}
message Message { string role = 1; string content = 2; }
message Params {
  uint32 max_tokens = 1; float temperature = 2; float top_p = 3;
  bool enable_thinking = 4; uint32 thinking_budget = 5;   // 0 = unlimited
  optional uint64 seed = 6; repeated string stop = 7;
}

message Event {
  string id = 1;
  oneof kind { Start start = 2; Token token = 3; Done done = 4; Error error = 5; }
}
message Start { string role = 1; }
message Token { string text = 1; bool thinking = 2; }   // thinking=true while inside <think>
message Done  { string finish = 1; Usage usage = 2; }
message Error { string kind = 1; string msg = 2; }
message Usage {
  uint32 prompt_tokens = 1; uint32 gen_tokens = 2;
  float tps = 3; uint32 ttft_ms = 4; float peak_gb = 5;
}

message CancelReq { string id = 1; }
message HealthReply { bool ready = 1; string model = 2; uint32 load_ms = 3; }
message Ack {} message Empty {}
```

## Phases & tasks

> Implemented under `qwen-server/`. End-to-end verified 2026-06-08: worker loads
> in ~7s, gateway streams OpenAI SSE + non-stream, metrics/gauges correct,
> cancel-on-disconnect fires, reasoning budget force-closes `<think>`.

### P0 — proto + scaffolding ✅
- [x] `proto/worker.proto`; field numbers locked.
- [x] Rust workspace (`gateway`, `exporter`) + `build.rs` with tonic-build (compiles clean).
- [x] Python `worker/` with grpcio + `gen_proto.sh` → `worker_pb2*.py`.

### P1 — Python MLX worker (thin) ✅
- [x] grpcio server implementing `Worker`. `Health` returns ready/model/load_ms.
- [x] Load `mlx-vlm` once at startup (path = `~/models/Qwen3.6-35B-A3B-UD-MLX-3bit`).
- [x] `Generate`: apply_chat_template (with `enable_thinking`), `stream_generate` → `Start`/`Token`/`Done`; exceptions → `Error{kind}`.
- [x] **Reasoning budget**: native mlx-vlm `thinking_budget` kwarg (forces `\n</think>` into the token stream — KV cache stays consistent). Replaced the earlier manual-splice crutch. `_GenState` now only tags thinking vs answer. Verified: budget=30 → cut + correct answer.
- [x] **KV cache q4 @ 64K**: weights are 3-bit, so a 4-bit KV cache (`kv_bits=4`, group 64, fp16 warmup first 5000 tok) is the balanced match — 64K ≈ 1.3 GB. `MAX_CONTEXT=65536`.
- [x] **Prompt-length validation (incl. vision)**: worker counts real input_ids via `prepare_inputs` (expands image tokens, same path generation uses); text-only `encode` fallback. If ≥ MAX_CONTEXT → `Error{context_length_exceeded}` (gateway → HTTP 400) BEFORE generation (no OOM). Generation clamped to remaining budget; `prompt_tokens` logged. Verified: 80k text → 400; 1536×1024 img = 1558 tok; 4096×2645 img = **10644 tok** (text-only count would say ~15 — why vision counting is mandatory).
- [x] `Cancel`: cooperative stop via per-id `threading.Event`, checked each token.
- [x] Python `logging` with levels, to stdout.

### P2 — Rust gateway (axum + tonic) ✅
- [x] tonic gRPC client; reconnect/backoff; awaits `Health` ready (sets `qwen_worker_up`).
- [x] **Request queue**: bounded `Semaphore` admission (503 on overflow) + single in-flight `Semaphore(1)`; depth/in-flight gauges.
- [x] OpenAI API: `POST /v1/chat/completions` (stream + non-stream), `GET /v1/models`, `GET /health`.
- [x] SSE: worker `Event` → `chat.completion.chunk` (thinking → `reasoning_content`); `ReqGuard` Drop propagates `Cancel` on disconnect + frees slots + refreshes gauges.
- [x] Maps max_tokens, temperature, top_p, stop, seed, enable_thinking, thinking_budget; vision via OpenAI `image_url` content parts → `images` paths/URLs.

### P3 — Logging (classical leveled) ✅
- [x] `tracing` + `EnvFilter` (`RUST_LOG`); `LOG_FORMAT=json` for Loki.
- [x] Per-request info logs (id, finish, tokens). Worker `logging` aligned. Promtail labels by `level`.
- [ ] (later) richer per-request span with queue-wait timing.

### P4 — Metrics (Prometheus pull / "pooling") ✅
- [x] `/metrics` via `metrics-exporter-prometheus`.
- [x] requests_total{status}, request_duration, ttft, tokens_generated, queue_depth, in_flight, worker_up, errors_total{kind}, cancellations.
- [x] `exporter/` (Rust): wraps `powermetrics` → GPU busy/freq, ANE, GPU/package watts → `/metrics` (needs root).

### P5 — Prometheus + Grafana + Loki (docker-compose) ✅
- [x] `deploy/docker-compose.yml`: prometheus, grafana, loki, promtail.
- [x] `prometheus.yml` scrapes gateway + mac-exporter via `host.docker.internal`.
- [x] Promtail tails `logs/*.log` → Loki; Grafana datasources + dashboard auto-provisioned.
- [x] Grafana dashboard JSON: ttft/duration/tok-s, queue/in-flight, errors, GPU/watts, logs panel.
- [ ] (todo) `docker compose up` smoke test (not yet run on this machine).

### P6 — Resilience (launchd) — "pod-like" ✅
- [x] launchd plists (KeepAlive + RunAtLoad + ThrottleInterval) for gateway, worker, exporter.
- [x] `watchdog.sh`: curl `/health` → `launchctl kickstart -k` if wedged.
- [ ] (todo) load plists + verify auto-restart on this machine.

### P7 — wrap-up
- [x] README: run order, env table, observability, resilience.
- [ ] (opt) make `chat.py` a thin client of the OpenAI endpoint (kept as offline tool for now).

## Open / later
- Continuous batching in the worker (higher throughput, more complex on mlx).
- TLS + auth on the public API if exposed beyond localhost.
- **Case B**: Windows + another model as a second gRPC worker behind the SAME gateway; inference Dockerizable there via WSL2 + CUDA. Reuses proto, gateway, metrics, Grafana unchanged.

---

# qwen CLI (done-ish)
- [ ] (opt) hide/strip `<think>…</think>` from output when thinking is on (will be moot once gateway exists).

# asr-server (built; weights downloaded; not run live)
- [ ] **End-to-end mic run** — verify the two `# VERIFY` spots: parakeet `_iter_sentences`, diart `itertracks`. Confirm real Frame stream.
- [ ] Run the **Rust client** vs the protobuf server (`--port 8003`, no `--json`).
- [ ] **DeepFilterNet** capture frontend (denoise 48k→16k).
- [ ] (opt) overlap-aware `segmentation-3.0` + `wespeaker` for hard audio.

# ✅ Done
- Downloaded + SHA256-verified Qwen (4/4 shards) over a slow link; inference ~59 tok/s.
- `chat.py`: streaming, `--image` vision, `--enable-thinking` (default off), `--max-tokens`, `--temp`.
- asr-server: deps pinned (torch 2.4.1, hub<1.0, matplotlib≥3.8), warnings silenced, all weights fetched.
- Download recipe in README + memory; proxy-scheme gotcha noted.
