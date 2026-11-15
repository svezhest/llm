# llm

Local LLM inference on Apple Silicon (MLX): a **Rust gateway** (OpenAI-compatible
HTTP/SSE) talking to thin **Python MLX workers** over **gRPC**, with Prometheus
metrics, Grafana dashboards, Loki logs, a request queue, and leveled logging.

**New here? Start with the [tutorial](docs/TUTORIAL.md)** — install, first run, memory recipes per model, the API in practice, tuning, observability, benchmarking and development.

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
