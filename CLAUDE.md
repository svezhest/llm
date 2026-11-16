# CLAUDE.md — llm

Local LLM inference on Apple Silicon (MLX). **Rust gateway** (OpenAI-compatible
HTTP/SSE) ⇄ **gRPC** ⇄ thin **Python MLX worker(s)**. Metrics (Prometheus),
Grafana dashboards, Loki logs, request queue, leveled logging. Inference runs
**native** (Metal not in Docker); only the observability stack is containerized.
See `README.md` for run instructions, `BACKLOG.md` for the plan + phase status.

This file holds only what a session must know before it touches anything. Everything
measured, every mechanism and every knob lives in topic files under `docs/`; the index
with the current state of each topic is [docs/MAP.md](docs/MAP.md).

## Layout

- `weights/` — model dirs (git-ignored). `qwen36-35b-a3b-mlx-ud3`, `ornith10-9b-mlx-8bit`, `ornith15-9b-mlx-8bit`,
  `qwen38-27b-mlx-5bit`, `qwen38-27b-mlx-4bit`, `qwen38-27b-mlx-oq4e`. Pull with `scripts/hf_pull.sh <repo> <dir>` (resumable aria2c).
- `src/proto/worker.proto` — gRPC contract (single source of truth; both sides codegen).
- `src/gateway/` (Rust) — axum HTTP + tonic gRPC client + queue + metrics + tracing.
  `src/{main,api,request,pool,metrics,worker_client,logging,journal,pb}.rs`.
- `src/worker/` (Python) — grpcio server, mlx-vlm; load + generate only. Own uv project.
  `worker.py` is the entry point and a facade over the package `qwen_worker/`: which
  module holds what, and the rules for names tests patch, are in [docs/layout.md](docs/layout.md).

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

## Tests

- Tests: `cd src/worker && uv run --group dev pytest -q --ignore
  tests/test_mrqmm.py` (CPU, tiny random models, ~15 s; the scheduler's key
  tests also run on `dense_fused`: 8-bit fp16-scale weights + `QWEN_GDN_FUSE`).
  `tests/test_mrqmm.py` runs Metal kernels (skipped without Metal; also the
  248K-vocab grouped-sampling check) — only with the GPU lease;
  likewise `PFQMM_GPU=1 … pytest tests/test_pfqmm.py` (the prefill kernel; its
  dispatch rules run in the CPU suite).

## Key design decisions (locked — don't relitigate)

- **Public API**: OpenAI-compatible HTTP/SSE (`/v1/chat/completions`), cross-OS.
- **Gateway ⇄ worker**: gRPC (tonic ⇄ grpcio), server-streaming `Generate→stream Event`.
- **One model copy, continuous batching** (only one ~18 GB copy fits in 32 GB): one
  scheduler thread in the worker owns MLX and decodes every running request in one
  `(B,1)` forward per step — per-row KV caches (the objects a lone request has) +
  the GDN state of all rows stacked on axis 0; prefill one request at a time,
  interleaved with decode. Gateway in-flight = the worker's `max_batch` (Health).
  `QWEN_MAX_BATCH=1` = the old serialized worker. "pooling" in the plan = metrics
  pooling (Prometheus pull), not a worker pool.
- **Rust everywhere except the MLX worker** (process isolation, no PyO3).
- **Images** = paths/URLs on the wire, never bytes.

Planned, not built: multi-model routing in the gateway
([docs/api.md](docs/api.md#direction-multi-model-routing-planned-not-yet-built)).

## Hard invariants

Each line is a rule that broke something once; the topic file has the story.

- MLX graphs are built and evaluated only on the scheduler thread after load; the scheduler
  thread never returns ([scheduler](docs/scheduler.md)).
- Never put a new op on a step's outputs before the next step is queued (read tokens per row
  with `r.y.item()`) ([decode](docs/decode.md)).
- Never share mx.array *objects* between cache copies (slice to get a new one); never trim a
  cache with the library ([prompt-cache](docs/prompt-cache.md)).
- A GDN checkpoint at p is valid for an entry only where p is a slab end of *that entry's*
  prompt; the slab policy is frozen at load (`_FIXED_SLAB`, `_bounds` determinism)
  ([prompt-cache](docs/prompt-cache.md), [prefill](docs/prefill.md)).
- The prefill/decode loop is our trimmed copy of mlx-vlm's `generate_step`: on every mlx-vlm
  bump diff the functions listed in [scheduler](docs/scheduler.md).
- One model process on the GPU at a time (one ~18 GB copy fits in 32 GB): stop a worker
  before starting another, and nothing else on the GPU while measuring.
- Behaviour reference = vLLM v0.30: what vLLM does that we don't is a 400, never "accept and
  ignore"; vLLM-unknown fields are ignored and logged ([api](docs/api.md)).
- Vision prompts are counted through the image processor, never a text-only encode, and
  there is no text fallback ([api](docs/api.md)).
- The thinking budget goes only through mlx-vlm's `ThinkingBudgetCriteria`; never splice
  `</think>` by hand and restart generation ([api](docs/api.md)).
- Break reference cycles that hold rows, leases or groups (`_Own.close`, `_unlink`): a cycle
  keeps GBs of KV alive until Python's cyclic GC ([scheduler](docs/scheduler.md),
  [prompt-cache](docs/prompt-cache.md)).
- A new Metal kernel (or a new width of one) self-tests at load and falls back to MLX on
  failure, as pfqmm, mrqmv, vatt and 4 / 6-bit mrqmm do ([kernels](docs/kernels.md)).
- A module-level name a test patches, or the code rebinds, follows the facade rules in
  [layout](docs/layout.md).
- Never hard-kill a download (`hf download` / aria2c): it orphans partials
  ([models](docs/models.md)).

## Exactness, in one paragraph

B ≤ 2 rows are bit-identical to a solo row; B ≥ 3 (mrqmm), cascade groups, hits through a
reply (reply KV) and DFlash with the mrqmm verify are in the "batched-decode numerics class"
(top-1 ≥ 99.7 %, near-ties can flip a greedy answer). Determinism at any batch size:
`QWEN_BATCH_INVARIANT=1` + `QWEN_PROMPT_CACHE_REPLY=0` ([scheduler](docs/scheduler.md)).

## Measuring, in brief ([docs/measuring.md](docs/measuring.md))

- The GPU's speed drifts ±20–30 % over minutes: never compare two separate runs. A/B inside
  one process, interleaved, medians over ≥ 3 rounds.
- zsh nices background jobs (`BG_NICE`): launch worker / gateway / bench with
  `setopt NO_BG_NICE` or in the foreground.
- Nothing else on the GPU: cargo, pytest, another worker, video, Activity Monitor.
- Scripts that write files use `PERF_OUT`; reports and raw data go to `docs/perf/<date>/`.

## Gotchas

- MLX random state is per thread: a sampler compiled against the main thread's key draws
  the same token stream from every gRPC handler thread. Test sampler changes from a thread
  ([api](docs/api.md#gotchas)).
- `QWEN_KV_BITS` defaults to 4 (the 35B's value); the 9B wants `0` and the 27B `8`. A wrong
  value only runs slower or bigger, nothing fails ([models](docs/models.md)).
- `QWEN_FP16=0` silently turns off every fp16-scale fast path (pfqmm, mrqmm, GDN fuse)
  ([kernels](docs/kernels.md)).
- Past the wired limit MLX crawls (well under 1 tok/s) before Metal OOMs: a sudden crawl is
  memory, not a kernel ([drafters](docs/drafters.md), 27B long context).
- DFlash greedy is not bit-equal to plain greedy (near-ties), but deterministic run to run
  ([drafters](docs/drafters.md#gotchas)).
- The gateway waits for the worker's Health before it listens: no worker = no 8080.
- A scheme-less proxy variable (`HTTP_PROXY=localhost:<port>`) → breaks Python `requests`.
  Use `http://...` or `HF_HUB_OFFLINE=1` (models are cached locally in `weights/`).
- mac-exporter needs root (`powermetrics`); load it as a `/Library/LaunchDaemons`
  daemon, not a user agent.
- Observability stack is shared infra — `deploy/` containers are `deploy-*`; don't
  touch the user's other Docker containers.
- The venv breaks on a repo move (absolute paths baked in) — `uv sync` at the root
  and in `src/worker/` to repair.

## Topic files

| topic | file | read it before touching |
|---|---|---|
| state of every topic | [docs/MAP.md](docs/MAP.md) | anything: start here |
| code layout, facade | [docs/layout.md](docs/layout.md) | adding a module or a name tests patch |
| models, KV format, memory, downloads | [docs/models.md](docs/models.md) | a new checkpoint, `QWEN_KV_BITS`, wired limit, library bumps |
| prefill | [docs/prefill.md](docs/prefill.md) | `sizing` (slab policy), fused SDPA, the fp16 cast |
| decode and batching performance | [docs/decode.md](docs/decode.md) | the decode step, step-ahead, the `batch:` log line |
| Metal kernels | [docs/kernels.md](docs/kernels.md) | `mrqmm.py`, `mrqmv.py`, `pfqmm.py`, `vatt.py`, GDN fuse, `kernels` |
| speculative decoding | [docs/drafters.md](docs/drafters.md) | `sched_spec`, DFlash / DFlash2, MTP, `lookup`, verify |
| prompt cache | [docs/prompt-cache.md](docs/prompt-cache.md) | `pcache`, `sched_store`, `pcdisk.py`, reply KV, MOVE |
| scheduler | [docs/scheduler.md](docs/scheduler.md) | `scheduler` / `sched_*`, admission, memory budget, cascade |
| API and gateway | [docs/api.md](docs/api.md) | `src/gateway/`, `chat`, `sampler`, `grammar`, `logprobs`, `stream` |
| env knobs (full table) | [docs/knobs.md](docs/knobs.md) | adding or changing any knob |
| measuring | [docs/measuring.md](docs/measuring.md) | any benchmark or performance claim |

## Where new knowledge goes

- A number goes into a report under `docs/perf/<date>/` (with its raw data).
- A conclusion goes into the topic file: one or two lines + a link to the report.
- The state of a topic goes into [docs/MAP.md](docs/MAP.md).
- The next step goes into `BACKLOG.md`.
- A new knob gets its row in [docs/knobs.md](docs/knobs.md).
- This file gets a line only for a new invariant or a silent gotcha, and stays under 16 KB.
