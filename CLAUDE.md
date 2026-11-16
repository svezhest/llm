# CLAUDE.md — llm

Local LLM inference on Apple Silicon (MLX). **Rust gateway** (OpenAI-compatible
HTTP/SSE) ⇄ **gRPC** ⇄ thin **Python MLX worker(s)**. Metrics (Prometheus),
Grafana dashboards, Loki logs, request queue, leveled logging. Inference runs
**native** (Metal not in Docker); only the observability stack is containerized.
See `README.md` for run instructions, `BACKLOG.md` for the plan + phase status.

## Layout (repo root = `/Users/user/Projects/llm`)

- `weights/` — model dirs (git-ignored). `qwen36-35b-a3b-mlx-ud3`, `ornith10-9b-mlx-8bit`, `ornith15-9b-mlx-8bit`,
  `qwen38-27b-mlx-5bit`, `qwen38-27b-mlx-4bit`, `qwen38-27b-mlx-oq4e`. Pull with `scripts/hf_pull.sh <repo> <dir>` (resumable aria2c).
- `src/proto/worker.proto` — gRPC contract (single source of truth; both sides codegen).
- `src/gateway/` (Rust) — axum HTTP + tonic gRPC client + queue + metrics + tracing.
  `src/{main,api,request,pool,metrics,worker_client,logging,journal,pb}.rs`.
- `src/worker/` (Python) — grpcio server, mlx-vlm; load + generate only. Own uv project.
  `worker.py` = the entry point (`uv run python worker.py`) + a facade over the package
  `qwen_worker/`: `import worker` still gives every name (`W._Scheduler`, `W.PREFILL_STEP`, …),
  and `setattr(worker, NAME, v)` / `monkeypatch.setattr(worker, …)` rebinds NAME in every
  module that holds it (`tests/test_facade.py` checks that each patch reaches its readers).
  So "worker.py" / `_Scheduler._fits` in the notes below = a name in `qwen_worker/`.
  Modules: `config` (QWEN_* knobs, constants; `SDPA_FUSED` / `DRAFT_BLOCK` / `BATCH_INVARIANT`
  are changed at load, read as `cfg.NAME`), `chat` (tool-call parser, images, messages,
  `_PlainContent`), `sampler`, `pcache` (`_Seg` / `_Entry` / `_PrefixCache`, cache-array
  helpers), `sizing` (`_step` / `_bounds` slab policy, `_FIXED_SLAB`, budget math), `stream`
  (detokenizer, `_StopFilter`, `_Control`, `_GenState`, `_done`), `grammar` (llguidance;
  `QWEN_JSON_MAX_WS`), `nosync`, `rows` (`RowsKV`, `_rows_attention`, stacked GDN), `kernels`
  (mlx-vlm patches: fused SDPA, DFlash verify, GDN fuse, batch-invariant checks), `lookup` (prompt lookup drafts: `_Lookup`, `_Gate`, `_DraftCtx`), `logprobs`,
  `request` (`_Req` / `_Row` / `_Own`), `scheduler` (`_Scheduler` core) + its mixins
  `sched_admit` / `sched_prefill` / `sched_spec` / `sched_store` / `sched_cascade` /
  `sched_decode`, `engine` (`Engine`), `server` (servicer, `serve`, `main`). A new module-level
  name a test patches must be read bare in its home or via `from .home import NAME`; a name
  the code rebinds at run time must live in one module and be read qualified elsewhere.
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
  0.7.6 drops the weightless vision tower (`vision_tower = None`) and loads strictly
  (0.6.2 needed a `strict=False` monkeypatch and kept a random tower); image requests
  → 400. ~9.5 GB on disk = 9.5 GB loaded (was 13.0 with the random tower and 0.6.2's
  fused-GDN weight copy), ~37 tok/s.
  Empirically (5 coding tasks, temp 0) 1.5 ≥ 1.0: same correctness on 4, and 1.0
  leaked its thinking into the answer and hit the length cap on the SQL task while
  1.5 answered cleanly. 1.5 thinks longer on trivial prompts (2–5× more tokens).
- **`qwen38-27b-mlx-5bit`** — Qwen3.8-27B (lmstudio-community), dense hybrid VL,
  `model_type qwen3_5` like the 9B (64 layers, full attn every 4th, 4 KV heads,
  head_dim 256, native 256K). MLX 5-bit g64, ~18.1 GB. Run: `QWEN_KV_BITS=8` +
  wired-limit dance (~24 GB @128K; 200K via `QWEN_MAX_CONTEXT=204800`, ~26 GB).
- **`qwen38-27b-mlx-oq4e`** — the same 27B, Jundot's oQ4e (oQ/oMLX mixed precision, MTP weights
  dropped at load): affine g64, base 4 bits with per-module overrides in `config.json` (lm_head 6,
  embed 5, attention 5 (last layer's k/v 6), GDN out 5, z / b / a 5 in layers 0–18, b / a 8 or 6 in
  three late layers, MLP down 5 in layers 0–17, the rest 4). LM ~14.85 GiB vs ~14.09 for the uniform
  4-bit (from the headers, MTP excluded); teacher-forced KL vs the 5-bit (`kl_models.py`, 3 × 3072
  tokens) 0.0236 / top-1 93.6 % vs 0.0534 / 90.1 % for the uniform 4-bit. Fast paths for it:
  `QWEN_MRQMM4/5/6=1` + the width-grouped GDN fusion (see "oQ4e" below).

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
- **One model copy, continuous batching** (only one ~18 GB copy fits in 32 GB): one
  scheduler thread in the worker owns MLX and decodes every running request in one
  `(B,1)` forward per step — per-row KV caches (the objects a lone request has) +
  the GDN state of all rows stacked on axis 0; prefill one request at a time,
  interleaved with decode. Gateway in-flight = the worker's `max_batch` (Health).
  `QWEN_MAX_BATCH=1` = the old serialized worker. "pooling" in the plan = metrics
  pooling (Prometheus pull), not a worker pool.
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

- **Library bump (2026-10-12): mlx 0.32.3, mlx-vlm 0.7.6, mlx-lm dropped** (src/worker
  only; root tooling env untouched). Exactness, teacher-forced 3.4K tokens, old → new
  stack: 9B KL 1.5e-5 / top-1 99.76%; 27B 9.2e-5 / 99.97%; 35B 6.3e-3 / 97.7% — the
  3-bit MoE's rounding floor, not a regression: vs an fp32 reference fp16 is equally
  far on both stacks (new 8.3e-3 / 96.97%, old 7.5e-3 / 97.32%; bf16 is worse on
  both, 1.1–1.4e-2), old-fp32 vs new-fp32 1.6e-4. Greedy identity corpus vs old: 9B
  12/15 (near-ties at chars 221–518 + a seeded draw), 27B 14/15, 35B 6/15; within
  the new stack byte-identical. Speed, interleaved A/B: 9B decode step 26.8 / 42.0 /
  61.2 → 26.6 / 41.7 / 61.4 ms at B = 1 / 8 / 16 (equal); 9B prefill 8K/32K/64K
  equal (±2%, drift); 35B prefill 8K 15.0/13.8 → 13.1/12.8 s (−9%), decode 65 → 68
  tok/s; 27B decode 14.8 → 15.5 tok/s. Memory: 9B loaded 13.0 → 9.5 GB, 8 × 8K
  batch peak 16.8 → 13.3 GB. DFlash ×1.5 (see below). Raw: `docs/perf/2026-10-12/` (`bump_*`, `design/bump_apply.md`).
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
  GEMMs the bottleneck); 4096 gains nothing (qmm rate is flat in M). `QWEN_CLEAR_CACHE`
  0/1 is ±2%. Quantized KV (27B/35B) shrinks the slab with context (`PREFILL_BUDGET`:
  the quantized SDPA materializes slab × keys × heads scores). fp16 KV + fused SDPA
  (the 9B with `QWEN_KV_BITS=0`) keeps 2048 at any context since 2026-10-12 (`_step`):
  a slab reaching 32K keys takes the fused kernel, which holds no scores (at 120K the
  old 546-token slab ran it at 3.8 vs 5.7 TFLOPS); below 32K keys the budget rule
  still applies (at the defaults it never shrinks there). The policy is frozen at
  load (`_FIXED_SLAB`), so `_bounds` cuts stay the same for the process's life; a
  fused-SDPA failure at run time fails a slab that would pass the budget unfused
  instead of materializing its scores. **Trade-off under batching**: one slab per
  scheduler tick, uninterrupted — a 2048 slab at 120K is ~8.4–9.5 s (≈5 s at 32K)
  vs ~3.2 s for the old 546 slab, so while a >64K prompt prefills every running
  decode row's inter-token gap is ~3× longer. Shrinking the slab only under load
  would change slab shapes and with them the bits (`_bounds` determinism); not done.
- **Prefill profile (9B, 2026-10-12, `docs/perf/2026-10-12/prefill_profile.md`)**: op
  classes of a 2048-token slab (each op synced and timed in-process), share of the
  slab at cache offset 0 / 8K / 32K / 120K: 8-bit GEMMs 91 / 83 / 68 / 41% (MLP
  gate/up alone 42% at 0), attention 1 / 9 / 26 / 54% (fused 5.7–6.9 TFLOPS from
  32K), GDN recurrence 3.6 / 3.3 / 2.7 / 1.6%, norms + swiglu + conv + rope + other
  elementwise ~3–4%, KV concat ≤1.3%, host (graph build + clear_cache) <0.2%. GEMMs
  ran ~8.1 TFLOPS against MLX's own fp16 GEMM at 9.5 on the same shapes: the
  quantized kernel's 32×32 tiles on non-NAX GPUs, fixed by `pfqmm.py`
  (`QWEN_PFQMM=1`: MLX's qmm_t code from the wheel's headers via `metal_kernel`,
  64-row tiles → 9.0 TFLOPS, ×1.10–1.14 per 8-bit matmul, ×1.03–1.28 over 3..8 bits,
  **bit-identical**; taken only where MLX runs qmm_t, ≥ 64 rows (the last tile is MLX's own 32-row one when ≤ 32 rows are left: 2026-10-17, ×1.06–1.19 GEMM time on short slabs with M % 64 ∈ (0, 32], `docs/perf/2026-10-17/pfqmm_short_slabs.md`), (1, M, K) inputs, g64;
  self-tested at load per (bits, group) on the first module and row count where MLX
  surely runs qmm_t — a key with no such module is logged "untested" and stays off). Full prefill,
  in-process interleaved A/B, old → new (pfqmm + fixed fp16 slab): 8K 17.7 → 14.4 s
  (×1.23), 32K 89 → 74 s (×1.20), 64K 215 → 177 s (×1.22), 120K 498 → 391 s
  (×1.27); peak at 120K 17.7–18.1 → 16.65 GB. Caches bit-identical up to 48K; at
  64K KL 1.2e-5, top-1 100% (teacher-forced tail; the GDN b/a projections, N=32, run
  MLX split-K with split 16 for slabs ≤ 1024 rows vs 8 above, and the old policy
  reached 1024-row slabs past 64K). Next lever: attention at long
  context (the fused head_dim-256 kernel at ~6 TF is 54% of a 120K slab; its tile
  variants and a chunked unfused attention were slower — see BACKLOG).
- **KV quantization costs speed** on this model: q8 vs fp16 cache at 32K is −12%
  prefill and −15% decode. The 9B runs fp16 KV (`QWEN_KV_BITS=0`, 32 KB/token,
  ~4 GB @128K). The 27B/35B keep q8/q4 for memory.
- **Dead ends (measured 2026-09-18, don't retry blindly)**: (1) dequantize→fp16
  GEMM instead of 8-bit qmm on prefill slabs: +12% on up/gate, −3% on down, ~5%
  of MLP — the GPU is at ~8 TFLOPS either way. (2) Fused SDPA for head_dim 256:
  mlx 0.32.3 has the kernel but its default dispatch still takes the unfused graph
  (materializes slab × keys × heads scores — the real reason for `PREFILL_BUDGET`).
  `QWEN_SDPA_FUSED=1` forces it for fp16-KV prefill slabs ≥ 512 over ≥ 32K keys.
  Per call (9B layout, time fused / default; final interleaved re-measure
  2026-10-12): up to 28K keys the default path is faster (fused ×1.05–1.12);
  from 32K fused wins: ×0.89 at 2048 × 32K (×0.95 at 1024, ×1.03 at 512), ×0.83
  at 2048 × 40K, ×0.59–0.68 at 64K, ×0.67–0.72 at 512 × 120K (an earlier single
  probe said ×1.06 at 2048 × 32K; not reproduced). The default path jumps at 32K
  keys (+34% for +14% keys), so the gate stays there. End-to-end much less than hoped: 64K prefill −3…5%
  in-process (203/215 vs 219/222 s), peak −1.3 GB; ≤ 32K unchanged. KL fused vs
  unfused at 64K 7e-6, 100% top-1. Slab 2048 at every offset for fp16 KV: done
  2026-10-12 (see Prefill profile; ×1.27 at 120K). Below 32K keys the slab still
  obeys `QWEN_PREFILL_BUDGET`, and the batch prefill reserve is sized for
  max(`QWEN_PREFILL_BUDGET`, `QWEN_PREFILL_STEP` × 32K) scores (= the old reserve at
  the defaults).
  (2b) `QWEN_GDN_FAST=1` (`mx.fast.gated_delta_update` for ≥ 16-token slabs): KL
  vs off 6.3e-5 / 99.85% (9B, 3.4K tokens), 1.1e-5 (27B), 9e-3 (35B); 8K prefill
  ×0.994. Re-measured with pfqmm: ×1.024 per slab, but 8K real tokens KL 0.013 /
  97.0% top-1 (fp16 decay drifts) — stays off.
  (2c) Prefill (2026-10-12): mx.compile of the elementwise chains (≤ ~1% to gain),
  KV preallocation instead of the per-slab concat (no gain, peak +0.3 GB), chunked
  unfused attention for long slabs (×0.4–0.76, huge fp32 transients), other fused-
  attention tiles (WN=1 / Q in registers / BK 32 all slower; BQ 64 does not fit).
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
  ×1.43–1.64 on the 5 coding tasks (33 → 47–53 tok/s; 2026-10-12, mlx-vlm 0.7.6 in
  the batching worker: median 48 tok/s vs 37 plain, the old stack had fallen to 32 —
  i.e. slower than plain), 63.7 tok/s end-to-end on
  a code answer (67% acceptance), ~120 tok/s on trivially predictable text;
  holds at 8K context. Tiny answers (<10 tokens) come out slower — first rounds
  have no context. Block size: with `QWEN_VERIFY_MRQMM=1` (default) and `QWEN_DRAFT_BLOCK` 0 a fixed
  block 8; else adaptive ≥ fixed 4/8/16/32. The
  ceiling is an MLX kernel property: 8-bit qmm with 4–16 rows costs ~2.3× a
  1-row qmm on M1 (M=32 drops to 1.7×), so a 16-token verify forward ≈ 70 ms vs
  30 ms decode; the 4-bit drafter block ≈ 15 ms (lm_head over 15 rows alone
  9 ms). Greedy-only (temperature 0, presence_penalty 0), text-only (a
  thinking budget is honoured inside the rounds since 2026-11-11). Quantized KV since 2026-11-15 (before: fp16 only, the
  drafter was ignored): `_spec` quantizes like the plain loop (after every slab, step and committed round) and cuts a
  block at `quantized_kv_start` while the cache is fp16, so every position has the KV format of the plain loop (CPU:
  q8 DFlash == plain greedy); its verify attends quantized KV one query per call (`QWEN_VERIFY_ATT` is fp16-only);
  speed not gated (`docs/perf/2026-11-15/design/kv_q8.md`).
  mlx-vlm 0.7 verifies through exact multi-row qmv kernels (rows bit-identical to a
  1-token decode, so DFlash output == plain greedy, 15/15 corpus); they spill
  registers from ~10 rows, so `QWEN_VERIFY_EXACT_MAX_T=8` keeps them for blocks
  ≤ 8 and uses `quantized_matmul` above. Median tok/s on the 5 tasks (2 rounds):
  cap 0 = 35, 6 = 50, **8 = 46–52**, 10/12 = 51–54 but one task collapses to 27,
  16 = 39 (that task 23). Caps 6–16 all 15/15 identical to plain greedy; 0 → 14/15.
- **Continuous batching (measured 2026-10-11/12)**: 9B decode step 41 / 51 / 90 /
  127 / 133 ms at B = 1 / 2 / 4 / 8 / 16 (the design microbench and the worker agree
  — no scheduler overhead). The ceiling is the M1 qmm row cost: MLX runs a per-row
  qmv below 10 rows for 4096-wide projections (below 6 for wider), then a flat
  qmm_splitk, so 16 rows cost about 8 (hence `QWEN_MAX_BATCH=16`). 27B (5-bit)
  pays poorly: ×1.4 at B=2 … ×1.9 at B=8 (×3.7 at 16). 35B MoE: rows read ~B×8
  distinct routed experts, so with distinct prompts batching gains only at large B
  (aggregate 38 / 69 tok/s at 8 / 16 concurrent vs ~40 solo decode; a microbench
  that clones one prompt into every row overstates it). Prefill-heavy load gains
  little (prefill is compute-bound and serialized; arrivals in waves): the share
  of decode time scales with occupancy so batches form at all with 8K prompts.
  A prefill slab runs uninterrupted in one tick: with the fixed 2048 fp16 slab a
  120K-offset slab stalls every decode row ~9 s (see Prefill slab).
  End-to-end (2026-10-12, through the gateway, ~610-token prompts, `ignore_eos`, closed
  loop; agg tok/s / TTFT p50; serialized = `QWEN_MAX_BATCH=1` on the same code):

  | c | 256 serialized | 256 batched | 1024 batched |
  |---|---|---|---|
  | 1 | 31.9 / 1.2 s | 31.2 / 1.3 s | 34.2 / 1.7 s |
  | 2 | 31.6 / 9.3 s | 48.7 / 2.4 s | |
  | 4 | 31.4 / 25 s | 82.8 / 3.0 s | |
  | 8 | 31.8 / 58 s | 110.4 / 5.3 s | 142 / 7.0 s |
  | 16 | | 115.9 / 10.4 s | 170 / 12.9 s |

  Pre-change serialized (f2bcbdf, drifting machine): 22.5 tok/s at every c, TTFT p50 80 s
  at c=8 (256) / 315 s (1024). Per-request decode 37 / 30 / 29 / 21 / 12 tok/s at c = 1 /
  2 / 4 / 8 / 16. `QWEN_MRQMM=0`: c=8 64.6, c=16 87.5 tok/s. Prefill TTFT single stream:
  8K 22.1 → 17.2 s, 32K 72.5 s (old: ~82–89 s, in-process). Peak 10.4 → 11.7 GB (c=16,
  256) / 12.05 GB (1024), old 13.8 GB. TTFT p95 at c=16 is 24 s: arrivals in waves,
  prefill one prompt at a time. Raw: `docs/perf/2026-10-12/final_bench.md` (+ `raw/`).
  Shared-prefix waves with cascade groups (`QWEN_CASCADE=1`, `docs/perf/2026-10-12/
  cascade_gpu.md`): step 8K × 16 rows 66.4 → 53.7 ms (×1.24), 32K × 8 83.6 → 68.2 ms;
  16 × 32K all admitted (peak 25.3 → 23.2 GB); 16 × 8K active −3 GiB; formation
  transient +2.2 GB at 8 × 32K; 32K × 16 at 92 ms vs ~62 projected (open, profile it).
- **Fused GDN decode in-projection (`QWEN_GDN_FUSE=1`, 2026-10-12; by width groups since 2026-11-15)**: mlx-vlm 0.7
  dropped 0.6.2's `_decode_quantized_linears_fused`, so a decode step ran qkv/z/b/a
  as 4 launches per GDN layer. `_install_gdn_fuse` concatenates the four weights once
  and points the modules at row views of it (no extra memory; 0.6.2 kept a second
  1.2 GB copy); `_gdn_call` runs one matmul (qmv at B ≤ 2, mrqmm for 3..16 rows on
  8-bit models with b/a riding along, else `quantized_matmul`'s own qmv/qmm — e.g.
  `QWEN_MRQMM=0` or the 5-bit 27B) and hands the outputs to the library `__call__`.
  Bit-identical at B ≤ 2; in-run A/B −2.3% at B=1, −1.6% at B=16. The concatenated
  shapes' mrqmm kernels are compiled at install. Needs fp16 scales/biases: with
  `QWEN_FP16=0` it fuses nothing ("0 layers" in the load log). Mixed-width layers
  (oQ4e): `_gdn_groups` concatenates the projections that share (bits, group, mode)
  (≥ 2 of them); a projection alone at its width keeps its own weights and call.
  `_fused_in` is a tuple of `_InGroup`s (a uniform layer: one group of four, the old
  concatenation and split); a group takes mrqmm only with N ≥ `MIN_N` (b+a, N = 96,
  stays on `quantized_matmul`). B ≤ 2 bit-identical to the unfused calls on the GPU
  (`test_worker_gdn_fuse_mixed`, tiny mixed model through the worker; the CPU suite checks B = 1).
- **Multi-row 8-bit matmul (`mrqmm.py`, `QWEN_MRQMM=1`, measured 2026-10-12)**:
  `mx.fast.metal_kernel`, simdgroup MMA computing `Cᵀ = W·xᵀ` for ≤16 rows,
  A fragments built in registers from one 16-byte weight load per 8 rows × 64 k
  (k permuted within each 64-group so x is one 16-byte load per row too), fp16
  dequant (0x6400 trick), fp32 accumulate, no threadgroup memory in the K loop.
  9B linears per step (all 8-bit shapes incl. lm_head): qmm 47 / 59 / 85 / 85 ms
  → 32 / 32 / 33 / 46 ms at 3 / 4 / 8 / 16 rows (a 1-row qmv step is 30): weight
  streaming stays at qmv speed to 8 rows, 16 rows are MMA-bound (~5.8 TFLOPS).
  Whole decode step (same process A/B, ctx 1K): 56 / 86 / 93 → 31 / 34 / 52 ms at
  B = 4 / 8 / 16. Through the gateway (600-token prompts, 512-token answers):
  c=8 52 → 93 tok/s, c=16 69 → 81–99. Numerics: ≤ qmm's own error vs an fp32
  reference on every real 9B/35B shape (rms rel 2.9e-4 = fp16 output rounding);
  9B teacher-forced 100% top-1 at B=3/4/8 (KL 8e-6), 99.7% at 16 (KL 1e-5; qmm
  99.8%, 1.1e-5). Only 8-bit affine g64 fp16-scale modules with N ≥ 512 (35B:
  attention + GDN, not the router / experts / shared expert / lm_head; 27B
  5-bit: none), only (rows, 1, K) inputs with 3 ≤ rows ≤ 16 — B ≤ 2 stays
  bit-exact qmv, prefill and DFlash (always (1, T, K)) never take it. Tried: NA 8
  fragments per simdgroup spills (×5 slower); 2×8 / 4×4 / 2×4 tiles are within 3%.
- **4-bit mrqmm (`QWEN_MRQMM4`, 2026-11-15)**: `_WSIDE[4]` (existed since 2026-11-10 for the bench, now dispatched):
  a 64-group is 8 aligned words, a thread loads 2 (one uint2) and regroups them into 16-bit lanes once per group;
  config as 5 bits (4×8 at ≤ 8 rows, 2×8 above: KS 8 in both buckets, a row's bits the same at 3..16 rows). GPU unit
  tests (synthetic, the 27B 4-bit shapes incl. lm_head): within fp16 rounding of fp32 and ≤ `quantized_matmul`'s
  error, rows independent of the row count, self-test passes. Selection is `mrqmm._bits_on` + `supported` only, so
  every 5-bit user (MRQuantizedLinear, `_gdn_call`, `_verify_linear(s)`, `_verify_head`) takes 4 bits unchanged.
  Random 4-bit weights (2026-11-10): 8 rows ×1.64 of a solo row, MMA-bound like 5 bits — a 4-bit build gains its
  smaller weights (solo row 46.5 vs 55 ms of linears), not a cheaper kernel. Model-level gate 2026-11-15 (`docs/perf/2026-11-16/profile_27b.md` §5): ×1.75 DFlash, numerics in class, identity 12/16 at near-ties; default off.
- **oQ4e / mixed precision (2026-11-15, `docs/perf/2026-11-15/design/oq4e_speed.md`, GPU gate
  `scripts/perf/dflash2/gate_oq4e.sh` pending)**: measured that day (separate processes, ~2K prompt):
  plain 11.6–11.9 vs 17.2 tok/s (uniform 4-bit), DFlash 16.6–17.6 vs 37.7 (round 259–274 vs 153 ms,
  3.57 vs 4.8 drafts accepted). Measured drift-free (gate steps 1–3, `oq4e_bench.py`, both
  checkpoints' tensors in one process): the linears of a 1-row forward 48.4 vs 45.2 ms (×1.07; 5 / 6-bit
  qmv streams as fast as 4-bit, the 6-bit head 349 vs 352 GB/s), so ~25 % of the measured plain gap
  is outside the kernels: drift between the processes or the weights' residency, to be split by
  `decode_probe.py` (step 5, a model run). DFlash costs found: mlx-vlm's exact verify kernels take 4/5/8 bits only, so the 6-bit lm_head went
  through per-token qmv (~8 head reads per round) and the drafter's head (the target's) through
  `quantized_matmul`; fixed by `QWEN_MRQMM6` (6-bit weight side `_WSIDE[6]`: 3 aligned words per
  thread, 4 lane words, one shift + mask per step; self-tested at load). The GDN in-projections
  were never fused (mixed widths); now by width groups (30 layers qkv+z 4 + b+a 5, 15 qkv alone +
  z+b+a 5, 2 qkv+z + b+a 8, 1 qkv+z + b, a alone). Re-quantizing the head to 8 bits was rejected
  (+0.3 GiB, a second rounding, +0.9 ms per plain token). Verify block T = 8, linears: 117.8 →
  87.6 ms (head 22.8 → 4.0 via mrqmm6, drafter head 11.6 → 4.0, GDN in 17.5 → 15.0) vs 80.8 for
  the uniform 4-bit; the 6-bit kernel 2.65e-4 rms rel (≤ qmm), 4×8 config kept. `oq4e_bench.py`: per-width qmv, verify
  block per kind, lm_head paths, kernel accuracy, both checkpoints in one process.
- **Decode step profile (2026-10-12, B ≥ 2)**: the worker's step was GPU-bound but
  the step-ahead was broken. The token read `mx.concatenate([r.y…]).tolist()` queued
  a new kernel *behind* the step just submitted, so the read waited for step k+1 too,
  and the GPU idled while the host built step k+2 (3–6 ms of Python under gRPC load).
  The fix is `r.y.item()` per row, which waits on the step's own event: in-run A/B −7…12% at
  B=8, −5…7% at B=16. Never put a new op on a step's outputs before the next step is
  queued. Per-row draws in a batch are grouped (`sample_rows`): rows with equal sampler
  settings are filtered as one (G,V) stack and drawn with their own keys; an error
  building a group's draw fails that group's rows only. MLX row ops are bit-identical
  per row in a stack (Metal, 248K vocab, ties included: `tests/test_mrqmm.py::
  test_sample_rows_full_vocab_stack_equals_solo`). The card
  sampler at B=16 goes from 14.8 to 7.8 ms of GPU, a −18% step. Host costs are small:
  graph build 2–4 ms CPU, encode 4–9 ms, greedy sampler 0.5 ms GPU, quantize /
  clear_cache / GDN gathers negligible. MLX encodes at most 10 command buffers ahead
  (50 ops / 50M elements each on M1 Max), so at B=16 host time still leaks into the
  step. `MLX_MAX_OPS/MB_PER_BUFFER=200` hides it but adds +2–3 GB of prefill peak
  (buffers live until their command buffer completes), so it is rejected. Per-row
  attention is 4–8% of the step at 1K and 16–27% at 8K (~2× its bandwidth floor),
  which makes a multi-row decode-attention kernel the next lever. This machine's GPU
  speed drifts ±30% over minutes (probably thermal), so A/B *within* a run
  (`scripts/perf/prof/step_prof.py --ab … --ab-steps N`). Details:
  `docs/perf/2026-10-12/step_profile.md`.
- **DeepInfer port (2026-10-24, `docs/perf/2026-10-24/`, ideas: `deepinfer_ideas.md`)**: ideas
  from deepseek-ai/DeepInfer (clone at `~/Git/DeepInfer`), each behind a knob (Env knobs).
  (0) The `batch:` log line has `step_ms` (decode ticks' wall per step), `gbps` (bytes a
  step must read — weights + KV + 2× GDN, `_step_bytes` — over it), `decode_s` /
  `prefill_s` per window (`analyze_worker_log.py` prints them). The old "~200 ms/step at
  rows=6" was the log (rows/prefilling are a snapshot, steps/s a 10 s window with 4–9 s
  slabs in it): AppWorld replay decode steps are 36–60 ms at 2–7 rows (200–300 GB/s),
  and **prefill is 81 % of the scheduler's busy time** (`step_metric.md`) — req/min moves
  with prefill / cache misses, not the decode step. Cascade's "32K × 16 at 92 ms vs ~62
  projected" does not reproduce on today's code (62.5 ms; the 2026-10-12 run was ~18 %
  slow). `QWEN_PROMPT_CACHE_CHAIN_TAIL=0` saves 1.2–1.5 states per entry (−4…8 %) and
  costs +3–10K replayed tokens only when an older message is edited: default stays
  4096; for append-only agentic runs under cache pressure 0 is free (`design/chain_tail.md`).
  Special-token strings in message text become control tokens (as in vLLM: a tool
  output with `<|im_end|>\n<|im_start|>system` forges a turn; `split_special_tokens`
  does not help — `<tool_call>`, `<think>` are special=False): `QWEN_SPLIT_SPECIAL_TOKENS`.
  Disk tier of the prompt cache (`QWEN_PROMPT_CACHE_DISK`, `design/pcdisk.md`): the biggest
  AppWorld lever of the day (×2.3 req/min at a 3 GB RAM cache: evictions become 0.3 s loads
  instead of 50–200 s re-prefills); off by default only because it needs a directory.

- **DFlash2 on the 27B (2026-11-10/11, `docs/perf/2026-11-10/dflash2_27b.md`)**: `QWEN_DRAFT=weights/qwen38-27b-dflash2`
  (z-lab/Qwen3.8-27B-DFlash2, 5 layers, block 8, target layers 5/19/33/47/61) + `QWEN_KV_BITS=0` + `QWEN_MRQMM5=1` +
  wired-limit dance. Decode, in-run A/B, greedy: code without thinking 13.3 → 23.4 tok/s (×1.77, 3.9 tokens / 164 ms
  round), with thinking 13.4 → 19.0–19.3 (×1.41–1.44, 3.2 tokens / round at block 8; blocks 4 / 5 / 8 equal, adaptive
  worse: the verify cost is flat over 3..8 rows). Without the 5-bit kernel ×0.81–1.10 (a block-8 verify through
  `quantized_matmul` = 4.4 steps; mlx-vlm's exact verify kernels on 5 bits: 1.6 s at block 8). The drafter must run fp32
  activations (`QWEN_DRAFT_FP32`, auto: fp16 overflows → 0 % accepted); its block is 36 ms at 4 bits. Not bit-equal to
  plain greedy with the kernel (15/16 answers without thinking, 12/16 with; 4 of the 5 divergences at a top-2 logit gap
  ≤ 0.016, one at 0.25). mlx-vlm's default runtime block for DFlash2 is 5 (adaptive 3..5); `_setup_verify_mrqmm` fixes 8.
  A thinking budget no longer turns DFlash off (`_spec`: the block is cut where the budget forces the next token,
  `\n</think>` go through plain steps; 27B 8/8 == plain with the budget). Prompt cache + drafter (2026-11-11, report §7):
  hits work, output == cold (30/30 pairs incl. hits through a reply). The entry keeps the drafter's context
  (`QWEN_DRAFT_TAIL` / `QWEN_DRAFT_TAIL_END`: hidden states of the last positions before each state), so an exact repeat runs the cold
  run's rounds (before: 1 token of context, 5–20 % fewer drafts accepted) and so does the next turn; DFlash rows now
  store reply KV (also demoted ones; `_spec_reply`: `row.fed` = generated tokens, the cache holds the first
  pos − len(prompt); a stop token among accepted drafts with more after it → nothing stored) and a cancelled prefill's
  prefix. Next turn without thinking: TTFT 0.75–0.9 s vs 31–36 s cold. `_spec`'s prefill window bug fixed (the oldest
  chunk was dropped while all but the NEWEST covered the window: slabs 2048 + 2048 + 40 left the drafter 40 positions).
  States stored by plain rows carry no context. Peak 23.2 GB (26.3–26.7 GB = 24.5–24.9 GiB with a 3 GB cache of four entries, wired limit
  27 GiB). **Long context (2026-11-11, report §8)**: with the copies an entry is kept only to ~40K tokens (rows + cache
  share KV budget 3.24 GiB + prefill reserve 3.4 GiB, fp16 KV is 64 KiB/token and held twice: row + entry) — longer
  prompts were "not kept", every turn cold. `QWEN_PROMPT_CACHE_MOVE=1` (single stream: one KV copy, see Prompt cache)
  + `QWEN_PROMPT_CACHE_CKPTS=4 QWEN_CACHE_LIMIT_GB=1 QWEN_SDPA_FUSED_MIN_KV=8192 QWEN_PREFILL_BUDGET=16777216
  QWEN_PROMPT_CACHE_GB=6 QWEN_MAX_CONTEXT=65536`: a 60.1K-token conversation, cold prefill 521 s, next turns TTFT
  1.4–2.1 s (thinking: resume at the last message start), decode 11.5–12.7 tok/s with thinking, peak 26.4 GiB
  (prefill) / 25.6–26.0 (decode) under the 27 GiB limit. Three things had to be fixed for it: (1) `_own_states`: after
  a partly accepted verify block mlx-vlm leaves each GDN layer's state as a slice of the block's per-token states,
  pinning all of them between rounds (7 × 147 MiB at block 8; decode transient above the resting row 2.05 → 1.27 GiB);
  (2) `QWEN_VERIFY_ATT`: mlx-vlm's verify attention is one query per call, i.e. the whole KV read per block token
  (round 570 ms at 60K keys vs 159 at 8K — slower than plain decode); MLX's vector attention kernel (queries × GQA
  factor ≤ 32 → 5 per call on the 27B) is bit-equal and reads it once: 296 ms; (3) a reference cycle (`_Own` ↔ its
  store closure ↔ the row) kept each finished row's states / KV until Python's cyclic GC (+0.4 GiB a turn with
  thinking, +1.3 without): `_Own.close`, and every request of `tests/test_pcache_move.py` checks that its row dies
  with its last reference. Above the wired limit MLX crawls before it fails (0.6 tok/s, then Metal OOM): the first
  60K runs. Without thinking at 60K: DFlash 14.3–16.0 tok/s vs plain 9.6 (×1.5–1.67), hits through the reply TTFT ~1 s;
  with thinking at 24K 17.0 vs plain 14.1. Full 64K and plain-with-thinking at 60K not measured.
  **Drafter (2026-11-16, 4-bit target, `docs/perf/2026-11-16/drafter.md`)**: the fp32 drafter's linears on
  `mrqmm_wide` (`QWEN_DRAFT_MRQMM=1`, default): block 31.3 → 17.4 ms, DFlash ×1.117 at 2K. `QWEN_DRAFT_BITS` 4 / 5 / 8:
  acceptance 3.24 / 3.21 / 3.25 of 7 (240 probes) at +1.5–2 ms per block for 5 / 8 bits, e2e 8 bits ×0.99 of 4: stays 4.
  Short Russian prose: 0.71 drafts of 7 accepted at any width, DFlash 11.1 vs plain 14.8 tok/s even with the fast
  drafter (break-even on the 27B ≈ 1 accepted draft per round): wants an acceptance-based fallback to plain.

- **One-pass verify / decode attention (`vatt.py`, 2026-11-16, `docs/perf/2026-11-16/verify_att2.md`)**: at long
  context the 27B's verify block attention (G 6, `QWEN_VERIFY_ATT=1`: 5 + 3 queries per vector-kernel call, ~30 GB/s)
  was ~45 % of a DFlash round and MLX's decode SDPA read the KV at ~90 GB/s. `QWEN_VERIFY_ATT=2` / `QWEN_DECODE_ATT`:
  steel-attention MMA loop (via `cascade_mma.steel_headers`) over key chunks, 16-row query blocks, causal tail per
  block position, fp32 partials + merge, KV buffers read in place. One layer at 60K: verify 8.14 → 2.45 ms (4.9 TFLOPS:
  compute-bound, 49 FLOP/byte; GEMM-rate floor ~1.3 ms), decode 2.77 → 0.91 ms (276 GB/s of ~339). 27B 4-bit at 60K:
  DFlash 17.97 → 26.54 tok/s, plain 10.61 → 14.37; at 32K 22.31 → 30.13 / 14.87 → 16.53; short prompts equal; tokens unchanged on the long runs; KL 1.2–1.3e-5. `QWEN_VERIFY_ATT=2` is the default since 2026-11-16; `QWEN_DECODE_ATT` stays off (recipe knob). Tiles: 16x16 best;
  48-row single blocks, WN = 1 and split smem regions slower (sweep in the report).

## Implementation notes worth keeping

- **Context = 128K** (`MAX_CONTEXT=131072`, prompt + generation). **KV cache is
  per-model** via `QWEN_KV_BITS` (default 4) — `stream_generate(..., kv_bits=N,
  kv_group_size=64, quantized_kv_start=5000)` (MLX analog of llama.cpp
  `--cache-type-k`; K and V uniform, fp16 warmup for the first 5000 tokens). Pick N
  by weight precision + budget: **35B (3-bit, ~18 GB) → q4** (5.6 KiB/token, ~0.7 GiB
  @ 128K; the balanced match for 3-bit weights, and the 35B is the tightest fit);
  **9B (8-bit, ~10 GB) → fp16** (`QWEN_KV_BITS=0`, 32 KiB/token, ~4 GiB @ 128K:
  faster than q8 here, see README; q8 would be 17 KiB/token, ~2.1 GiB); **27B
  (5-bit, ~18 GB) → q8** (34 KiB/token, ~4.25 GiB @128K, ~24 GB total). Only
  full-attention layers (every 4th) hold KV (`_kv_token_bytes`: bits/8 + an fp16
  scale and bias per 64 elements). Env sets sampler defaults;
  requests may override temperature/top_p/top_k/min_p/presence_penalty (proto
  `optional` fields; absent = env default, temperature -1 / top_p 0 likewise).
- **The worker owns its prefill/decode loop** (`_Scheduler._prefill` + `_iterate`), a trimmed copy of
  mlx-vlm 0.6.2 `generate_step`+`stream_generate`, re-checked against 0.7.6 (same sampler, logits
  processors, KV quantization, ThinkingBudgetCriteria, detokenizer). Reasons:
  the prompt-cache snapshot has to be taken at the exact point where the cache
  holds prompt[:-1] and nothing else (the library's `PromptCacheState` snapshots
  after the first generated token and trims — impossible for ArraysCache,
  broken for quantized tuples); pre-tokenized vision inputs go in directly;
  `finish_reason` `"length"` and pool clearing are ours. Output verified
  byte-identical to `stream_generate` at temperature 0. When bumping mlx-vlm,
  diff `generate/ar.py::generate_step` against `_prefill`/`_iterate`/`_sample`,
  `Qwen3_5Attention.__call__`/`_prepare_projected_qkv` against `_rows_attention`, and
  `Qwen3_5GatedDeltaNet.__call__` against `_gdn_call`. 0.7 returns `position_ids`/
  `rope_deltas` from `get_input_embeddings` (counted from 0): text drops them (a
  prompt-cache suffix would land at 0), vision slices them per slab. DFlash verify
  is `speculative_verify=True` + `commit_/abort_speculative_round` every round.
- **Prompt cache** (`QWEN_PROMPT_CACHE=1`): multi-slot prefix cache
  (`_PrefixCache`/`_Entry`). KV is sliceable to any prefix; the GDN recurrent
  state is not, so `_prefill`/`_spec` checkpoint it (~52 MB, fp32 32×128×128 ×
  24 layers on the 9B) at slab ends forced onto `cuts`: every 2048 tokens,
  `<|im_start|>` boundaries (end of system block, last two messages) and the
  divergence point from the closest entry. Hit = deepest checkpoint ≤ longest
  common prefix with any entry. **Segments** (`_Seg`, since 2026-10-12): an
  entry's KV is a chain of pieces `(seg, lo, hi)` over immutable per-FA-layer
  buffers of its own; `store` copies only `[at, len)` out of the row (the parent's
  pieces cover `[0, at)`; all of `[0, len)` when the row's prefill quantized an
  fp16 parent's part) and never keeps a view of the row (its first decode write
  stays in place; phase 1 pinned the whole buffer and the row COW-copied it);
  `restore(e, at, first)` concatenates the pieces + a zeros tail of
  round_up(first slab, 256) = the buffer the first `update_and_fetch` built from
  the old `[:at]` view (bit-identical, tested). Bytes count every buffer once; a
  segment dies with the last entry holding it. 16 siblings of an 8K prompt:
  ~1.38 GB for all 16 by the plan's estimate (cascade_plan A.7; not yet measured —
  phase 1 measured 6 kept in 3.18 GB); a chat keeps one chain (+ the
  dead tail of each resumed turn, ≤ a slab + the reply opener). A checkpoint at p
  is valid for an entry only where p is a slab end of *that entry's* prompt (its
  KV over [0, p) was cut there). An older entry whose ids are a prefix of the new
  one (the previous turn) is absorbed at store: its checkpoints move over (shared
  objects), its end state only if that position is a slab end of the new prompt
  (usually not: an exact repeat of the older prompt then resumes at its deepest
  checkpoint), and it is dropped — except an unquantized entry under a quantized
  new one (27B/35B: the only resume point below `quantized_kv_start`). Thinning
  to `QWEN_PROMPT_CACHE_CKPTS` (`_thin`) never drops the end of the system block
  (`starts[1]`) or the last three `<|im_start|>` (mapped to slab ends, inherited
  ones too); strides and older turns' boundaries go first, closest pair's left.
  **Chain policy** (2026-10-16, AppWorld handoff: an 11K agentic entry was 932 MB,
  ~60 % GDN states): an entry that absorbs an older one (`covered`) keeps only
  `chain[0]` = the end of the system block + the last `QWEN_PROMPT_CACHE_CHAIN_MARKS`
  (3) message starts + the stride checkpoints ≥ end − `QWEN_PROMPT_CACHE_CHAIN_TAIL`
  (4096) + the strides inside the first user message (the first `<|im_start|>user` block,
  by `Engine._im_user`; unknown role → `starts[1]..starts[2]`; a shared
  document: a sibling conversation diverging inside it resumes near the divergence, not
  at the system block), thinned with `chain[1]` (the marks) protected; + its end state.
  Trade-off (review 2026-10-16 #6, `chain_ckpts_sim.py` with `SIM_USER0`): free for
  AppWorld (300-token first message: 11.4K entry 669 MB either way), +2–4 states
  (+100–200 MB) per entry for an 8K first message (13.5K 687 → 785 MB, 34K 1415 → 1562);
  the alternative (chain filter only from the 2nd absorption) was rejected: it keeps
  nothing of the document past turn 3. The quantized floor is applied by `store` (the
  chain is not pre-filtered). Computed in
  `_prompt_cache` (`chain`), applied in `store`; `CHAIN_MARKS=-1` = off. Fresh entries
  (no chain absorbed) keep the cap-12 policy below `QWEN_PROMPT_CACHE_CHAIN_FRESH_TOKENS`
  (16384, 0 = never; length of the whole prompt), from there `fresh` = `chain` + the
  divergence point from the closest *other* entry (`diverge`: the prompt's own partial
  entries are ignored, their `_Entry.diverge` carried) + the strides inside the system
  block (before the first user message, unprotected) (2026-10-17: a 50K MISS entry
  12 → 7 states, 2.28 → 2.04 GB simulated). Byte-identical (a subset of valid checkpoints).
  Simulated through the scheduler (CPU, production knobs, AppWorld-shaped turns,
  9B bytes, `docs/perf/2026-10-16/chain_ckpts_sim.py`): 11.4K 1012 → 669 MB (12 → 5
  states), 41K 1995 → 1652 MB, resume
  positions unchanged; an older turn repeated resumes at its last message start
  (previous turn) or the system block. Store log `states=` = checkpoints + end MB.
  `QWEN_PROMPT_CACHE_GB` (3) is a soft cap (at store, the newest stays, alone
  if over with a warning); **eviction order** (`_PrefixCache.victim`, every eviction:
  store, `_fits`, `_regrow`): entries a waiting request would resume from ≥ a slab
  deep (`_Scheduler._wanted`: the admitted request whose prefill has not started + the
  first `max_batch` waiting, the queue copied under `cv`; peeks memoized per request in
  `req.peeked`: `pc.version` bumps once per kept store (publish / rollback), an eviction
  invalidates only the memos that pointed at it; weakref)
  last; stale ones (`_PrefixCache.stale`: `pc.seq − e.seq > 2 × len(entries)`, the clock
  ticking at every text lookup and store — no hit / store of it within a round of turns;
  vision does not tick: 2 ticks per vision request made live episodes stale, 28/32
  misses with one image per text turn, review round 2 — or, unless wanted, unused for
  `QWEN_PROMPT_CACHE_STALE_S` (300 s): an idle cache) first, LRU; all live → MRU (round-robin episodes: the latest
  one's next turn is the furthest; LRU evicted the next in line, and its re-prefill
  the one after: 28/32 misses vs 5 in `test_round_robin_episodes_keep_hitting`;
  event sim of the AppWorld load +13–60 % requests served at 5–8 GB, "largest
  first" 3–10 % worse than MRU: `docs/perf/2026-10-16/evict_sim.py`; `evict_sim.py clock`:
  the lookup+store clock gives identical results, and so does + 300 s / 120 s wall-clock
  age when wanted entries are exempt — without the exemption 300 s cost 6–12 % of the
  requests served at 4–6 GB, where queue waits pass 300 s); the cache's live bytes (`_PrefixCache.bytes`: every
  buffer once) count against the batch budget: a store first sizes the new entry
  (its segment from the row's shapes, `_Seg.estimate`, exact) and, if it alone is
  over budget + prefill reserve − reserved, drops it before anything changes —
  nothing absorbed, evicted or copied, logged "not kept", counted in `evictions`;
  else it absorbs, evicts entries (victim order) down to that headroom, then copies `[at, len)`
  (`wanted` is asked before anything changes and only when an eviction is coming; its
  failure is logged and costs the order only; anything raising between the publish and
  the segment rolls back: the new entry out, the absorbed ones back — never a
  half-built entry, whose hits would restore a too-short KV);
  `_fits` evicts on demand but spares the entry `peek` says the request would resume
  from when that is ≥ `QWEN_PREFILL_STEP` deep (memoized `req.bounds`, `req.peeked`), and evicts
  nothing when even that would not admit it (budget or live check). When that entry
  alone keeps the head out, the head waits for a row to end only for a hit ≥
  `QWEN_PROMPT_CACHE_KEEP_TOKENS` (8192) and at most `QWEN_PROMPT_CACHE_KEEP_WAIT_S`
  (10 s, from its first refusal: `req.keep_since`); then, or at once for a shallower
  hit, the entry goes too, first (logged "gives up its N-token hit"). `_regrow` never evicts
  the running prefill's parent (`row.parent`; the snapshot closure holds an
  `_Inherit`, not the entry, and drops it and the checkpoint dict right after its
  store — a DFlash run keeps the closure for its whole decode). Quantized entries keep
  no checkpoint below `quantized_kv_start` (never resumable; `_Entry.low`). A
  restore holds copies, so eviction frees memory at once, except the pieces an
  in-flight prefill resumed from (its `_Inherit`, released at its store). A failed store is
  logged, the request goes on. Store log: `… MB (n entries, X MB total) reused=r
  gap=split−reused absorbed=k pieces=c states=S MB`; `Usage.cached_tokens` = reused; Health has
  entries / bytes / hits / misses / evictions. Never trim with the library; never share mx.array
  *objects* between cache copies (`a[i] = v` rebinds the Python object in place
  — slice to get a new one, buffers are copy-on-write). On a hit set
  `language_model._rope_deltas` to zeros (text mRoPE continues from the cache
  offset; a previous vision request leaves its own delta there). Text-only.
  7.5K shared system prompt, new conversation: TTFT 14.4 → 0.20 s, byte-identical.
  **Partial store (2026-10-17)**: a text `_prefill` that stops early — cancel at the slab
  loop's top, or an exception outside the forward (`intact`: after the slab's eval) —
  with ≥ 1 slab past `reused` calls `snapshot(cache, upto=pos, why)`: ids = prompt[:pos],
  pos the last completed slab end (∈ `ends`, offsets checked), the row's state = end
  state, checkpoints so far, same `pc.store` (limit, absorb, chain/fresh, `_Inherit`
  released); `_Entry.partial` (absorbing only partial entries is no chain: `chained` =
  any non-partial or chained covered entry), logged as the normal store line +
  ` (cancelled at pos/len)`. Runs in `_advance` like the full store. DFlash (`_spec`)
  prefills too since 2026-11-11; vision: nothing. Retry resumes at pos, bit-identical to cold. A store exception there
  is logged (`_store_prefix`), never changes the outcome. `docs/perf/2026-10-17/
  design/partial_store.md`, `tests/test_partial_store.py`.
  **Reply KV (`QWEN_PROMPT_CACHE_REPLY`, default 1 since the 2026-10-17 GPU gate;
  `0` = off)**: at a row's end
  (`stop`/`length`; not cancel (stop strings usually; a late cancel after stop/length
  stores, harmlessly), errors, vision, rows still in a cascade group (logged
  `id=… prompt cache reply skipped: in a cascade group …`), replies that generated a
  reasoning section — `_GenState.for_prompt(ids, _think)`: the prompt open in `<think>`,
  the handler's own split; a think id inside the content with thinking off is content,
  stored) `_drop_done` → `_reply_cache` + `_store_reply` extend the row's prompt
  entry (`row.entry`, weakref; gone = nothing) to ids = prompt + `row.fed[:pos − len]`
  (tokens as `_iterate` read them; the step-ahead fed the EOS it stops on, a sync row
  did not, a `length` row fed its last generated token too: entry = prompt +
  completion) through `pc.store` (`inherit` = its
  whole chain, so only `[len−1, pos)` is copied; budget checks as any store; `reply=` →
  `_Entry.reply`, whose end `best()` accepts though it is no slab end; `tick=False`:
  a turn stays 2 clock ticks; `sync=False`: copies left lazy, `_drop_done` queues every
  finishing row's copies + end states in ONE `async_eval` — no host wait between steps;
  `spare=(task_row.parent,)`: never evicts the running prefill's parent, as `_regrow`,
  and "not kept" when it does not fit beside it; `clock=` the prompt entry's
  (used, seq): a reply keeps its prefill's recency, else a just-finished conversation
  — next turn imminent — would be the MRU victim). A failed pass `async_eval` drops the
  reply entries and restores the prompt entries they absorbed.
  The GDN end state is always a gather `a[[i]]` (own buffer; at B = 1 too: right after a
  2 → 1 shrink the row's state is `_gdn_split`'s view of the old stack). Absorbed
  checkpoints only `p < L`. The prompt-end state is dropped (a regenerate resumes at the last
  message start); a reply entry whose prompt part a new store covers is absorbed (the
  reply re-tokenized differently / regenerated). Phase 0 (AppWorld traces, CPU): 100 %
  of 1344 replies re-render to the same ids, −20.8 % prefill tokens.
  **Numerics**: reply KV from decode kernels + the next slab starting at the reply end
  → a hit through a reply ≠ cold prefill bitwise (CPU: GDN recurrence of a T-slab vs T
  steps differs from layer 0): the batched-decode numerics class — GPU gate (9B, hit
  through a reply vs cold, teacher-forced 862 positions) top-1 99.88 %, KL 5.8e-5;
  greedy answers can depend on cache contents. `QWEN_PROMPT_CACHE_REPLY=0` is the way
  back (bit-exact, `_bounds` determinism). Gate: −21 % prefill tokens / +6.5 % req/min
  on the AppWorld replay, no visible decode cost. Tests `tests/test_reply_kv.py`; design
  `docs/perf/2026-10-17/design/reply_kv.md`, gate `docs/perf/2026-10-17/reply_kv_gpu.md`.
  **One KV copy (`QWEN_PROMPT_CACHE_MOVE`, 2026-11-11, off by default; `_move_on`: max_batch 1, no disk tier; quantized
  KV since 2026-11-15: only quantized entries move, an fp16 one (< `quantized_kv_start`) is copied, a row still fp16
  at its prefill's end stores by copy, an fp16 parent of a quantizing row inherits no pieces; parity tests
  `tests/test_kvq8.py`)**: `_prompt_cache` takes the conversation's own entry (`taken`: its ids are a prefix of the prompt or the
  prompt covers its reply; a sibling's entry only when a copy does not fit beside it, logged) with
  `restore(move=True)`: the entry leaves `pc.entries`, its exclusive segments are released layer by layer as the
  row's buffers are built, the KV-less entry stays as the lease's `husk` (`_Own`: husk / at / lo / L / end / store).
  `snapshot()` at the prefill's end only saves the end states (`lease.end`); the row's end goes through `_settle`
  (`_drop_done`, `_spec_reply`, the max_tokens = 0 path): `_Seg.take` turns the row's own buffers into the segment
  (views; own slices when the buffers run a KVCache step past what is kept), `lease.store` = the deferred prompt
  store with `seg=` / `moved=husk`, then `_store_reply(seg=whole.sub(L, pos))` over the same buffers. Early stops:
  `_settle_prefix` (`unmove` when nothing was prefilled past the entry, else the partial store); a failed forward:
  `_lost` (nothing evaluated or stored: the arrays hang on the failed graph). Checkpoints of a running prefill are
  thinned live to `QWEN_PROMPT_CACHE_CKPTS` (they are the row's, uncounted, until the store). `_Own.close()` at
  every exit: `store` is a closure over the lease and the row (a cycle that kept a dead row's KV and states until
  the cyclic GC). Parity with the copies (answers, reused, cache contents bit for bit) + no-cycle check per request:
  `tests/test_pcache_move.py`. README "One KV copy for a single stream".
- **Continuous batching (`_Scheduler`, worker.py)**. Invariants:
  - MLX graphs are built and evaluated only on the scheduler thread after load
    (MLX ops on two threads at once crashed in tests). Handlers render, tokenize,
    run the image processor — numpy, whose `to_mlx` wraps the results as
    *constant* MLX arrays on the handler thread (allocations only, no graph or
    stream; read back to numpy at once) — so a slow image URL never stalls the
    batch; build `_Control`/`_Guide` (no MLX); then turn token ids into events.
    The processor runs with `padding=False` (padding would flip the shared Rust
    tokenizer's state under other handlers: "Already borrowed"). The scheduler
    thread never returns: MLX tears down its thread-local stream on exit, which
    segfaulted the process.
  - B=1 = the plain library call on the row's own caches with an explicit position
    → byte-identical to the pre-batching worker (GPU: 15/15 corpus at max_batch 1
    and 16). B≥2: `RowsKV` + `_rows_attention` (projections batched, cache update +
    SDPA per row with the B=1 calls), stacked GDN, positions `(B,1)` = KV length +
    the row's mRoPE delta; decode never reads `_rope_deltas`/`_position_ids` (only
    the one running prefill writes them). Rows differ from solo only by the matmul
    kernels' row-count rounding: B≤2 always bit-exact (per-row qmv); 3..16 rows
    take mrqmm on 8-bit modules (`QWEN_MRQMM`), qmv/qmm on the others. Measured
    with stock MLX: 9B teacher-forced 100% top-1 at B=8 (KL 9e-6), 99.8% at 16;
    with mrqmm 100% at B=3/4/8 (KL 8e-6), 99.7% at 16 (KL 1e-5) — so
    under load a greedy answer can differ run to run; `QWEN_MAX_BATCH=1` restores
    determinism (+ `QWEN_PROMPT_CACHE_REPLY=0`: a hit through a reply differs from
    cold even at B = 1). **Batch-invariant mode (`QWEN_BATCH_INVARIANT=1`, 2026-10-24,
    `docs/perf/2026-10-24/design/batch_invariant.md`)**: determinism at any batch size
    instead: mrqmm with one fixed KS (8) for 1..16 rows and also at B ≤ 2 (plain linears,
    lm_head, fused GDN), cascade and DFlash off — a row in a batch of N == the same row
    solo, bit for bit (9B GPU gate: 5947 teacher-forced steps at B = 1..16, logprobs and
    top-5 `==`; control with the mode off diverges). Step cost vs off: B=1 ×1.18, B=2
    ×0.86, B=4/8 ×1.00, B=16 ×0.97 → for reproducible agentic runs use it with
    `QWEN_PROMPT_CACHE_REPLY=0` at `QWEN_MAX_BATCH=16` instead of `MAX_BATCH=1/2`.
    Models with modules mrqmm does not take (27B 5-bit, 35B experts / lm_head,
    `QWEN_MRQMM=0`, `QWEN_FP16=0`) get max_batch 2 (per-row qmv exact there); an
    unquantized `nn.Linear` turns the mode off. Rests on MMA results not depending on
    the column's position and on the MB=1/2 kernel variants compiling `acc += red[s]`
    alike (true on the 9B shapes; others not gated). Side findings, not applied: at B=2
    mrqmm is 14 % faster than 2-row qmv; at 9..16 rows 2×8 beats the default 4×4 by 2.6 %. Cascade groups (default on with fp16 KV + the prompt cache: ≥ 8
    rows sharing ≥ 4096 prompt tokens, below) change only the grouped rows'
    attention, in the same numerics class as mrqmm (top-1 ≥ 99.71 %); B ≤ 2 stays
    bit-exact and `QWEN_CASCADE=0` restores this per-row path bit for bit.
  - Admission FIFO, no overtaking; an empty batch always admits. Reservation =
    prompt + min(max_tokens, `QWEN_BATCH_RESERVE_TOKENS` 4096) tokens of KV + 2×
    GDN state (quantized KV: + the fp16 warm-up copy, ≤ `quantized_kv_start` + a slab, until the row's cache is
    quantized — `_kvq_mark`, `row.kvq` — then N-bit only); rows are re-accounted at their real length every step (growth past
    the reservation is allowed and holds back new admissions; past budget +
    prefill reserve the growing row ends with `length` while other rows run — a
    lone row is never cut); the prompt cache's live bytes count beside the
    reservations and give way first (evicted on demand, `victim` order); live check `active +
    reserve + prefill reserve ≤ cap`. Cap in `_batch_cap`, budget in `_batch_budget`.
  - Time share: every prefill slab, the last one too (a cache hit's only slab),
    sets `resume` (`_slab_done`), so decode steps run between short prefills
    instead of waiting for the queue to drain (the 16 × 8K shared burst held every
    first token until the last hit was prefilled).
  - Step-ahead (`async_eval`) as before; a grammar or multi-token bad-words row
    puts the whole batch in sync mode while it is there. Thinking budget forces the
    row's next input. Vision: prefill serialized, decode batched with its delta.
    DFlash only for a request admitted with nothing else running or waiting
    (a thinking budget is honoured inside its rounds since 2026-11-11);
    demoted to a batch row at a round boundary when a waiting request would be
    admitted next to it.
  - Generate RPCs beyond `QWEN_GEN_SLOTS` (`max(2·max_batch, 8)`) →
    `RESOURCE_EXHAUSTED` (gateway 503); gRPC pool = slots + 8 threads, so
    Check/Cancel/Health have spare threads (grpc's own backstop refuses RPCs only
    past slots + 64 queued). A failed decode step fails its rows only; a lone
    row's failure is no strike. 3 consecutive failed steps with several rows close
    the scheduler: Error(worker_fault) to everything, Health `ready=false`, and
    the process exits (70) after 1 s so launchd restarts it. A tick running
    > 300 s only makes Health not ready (handlers waiting on it give up) until it
    ends; idle time never counts. Shutdown RPC / SIGTERM / Ctrl-C →
    Error(shutdown) to everything; exit(1) if the current tick does not end
    within 10 s.
  - `Usage.peak_gb` / log `peak=`: MLX peak since the batch was last empty, read by
    the scheduler when the request ends (= the process peak during the request
    when rows overlap). `Usage.queue_ms` / `batch_mean` / `cached_tokens` are
    worker-side (`queue_ms` = arrival → the first prefill slab, which waits for
    decode's share after admission; the worker's `ttft_ms` counts from there,
    `_advance`); the gateway returns `usage.prompt_tokens_details.cached_tokens`
    (always, 0 without a hit) and counts `qwen_gateway_cached_tokens_total`.
  - **Cascade groups (`QWEN_CASCADE`, default 1 since the B4 GPU gate, 2026-10-12)**:
    rows sharing prompt ids `[0, P)` with P a slab end of each (→ same KV bits:
    `_bounds` + hit == cold) decode attention as one `_PrefixGroup` (`cascade.PrefixGroup`:
    fp32 prefix once + stacked fp32 suffixes; `_rows_attention` → `_cascade_rows` with
    the step's `_StepPlan`; `begin_step`/`end_step` around the forward in
    `_grouped_forward`). Formed in `_join` (`_cascade_join`) when ≥ `QWEN_CASCADE_MIN_ROWS`
    (8) rows share ≥ `_MIN_PREFIX` (4096) tokens (earlier rows converted, their FA caches
    → `_GroupRef`, freed); later rows `add` (the row's own S-token copies evaluated, the
    stack concat lazy). A row a group refuses forms no second group on that group's prefix.
    Memory rule `_group_wins`: at the members' REAL lengths the group's charge
    (`_group_bytes`) ≤ their own fp16 caches (9B, ~150-token suffixes: 3 rows never win,
    4 only from 32K, 5–16 at 4K–32K). Budget guard `_charge_fits`: a formation / join
    that raises `reserved` is taken only while `reserved` + the prompt cache's live bytes
    ≤ budget (refused, nothing evicted; the charge is reservation-based and grows with
    `_regrow` like any row's, so growth can still pass the budget). `_compact` →
    `_cascade_tidy`: done rows `take`n out (lazy); forced dissolve at < 3 live members
    (covers B ≤ 2: today's bit-exact path), unchecked — its peak (2 contiguous rows beside
    the live group) is inside the charge by `_group_bytes`' floor (prefix + stacks + 2 ×
    (P + s) fp16); optional releases (a member whose own suffix > P//2, the longest first;
    a memory dissolve and `set_cascade(False)`, the cheapest first) one member per group
    and decode step (`released_at`; the last < 3 together), each checked with the group
    alive: `reserved` + the released rows' fp16 caches + prompt cache ≤ budget + prefill
    reserve, nothing credited for the group's shrink (release evaluates beside it, take is
    lazy) — else they stay grouped. A tidy that raises fails that group's rows only. No
    reference cycles: `_GroupRef` holds a weakref to its row and no group; `_unlink` on
    every drop (a leaked group would hold GBs until Python's gen-2 GC). Eligible: fp16 KV
    only, `QWEN_PROMPT_CACHE=1`, max_batch ≥ 3 (else off whatever the switch; the load log
    says why), text rows that resumed from the cache or stored an entry (`row.cached`),
    never vision or DFlash (`row.spec`, also after demotion). Numerics: the same class as
    mrqmm's batched decode (B4: top-1 ≥ 99.71 %, KL +0–13 % over today's per-row batch,
    per-layer error below the per-row kernel's); B ≤ 2 bit-exact; ungrouped rows of a
    mixed batch take the per-row calls verbatim (bit-identical; the output take is an
    exact copy); `QWEN_CASCADE=0` restores today's per-row path bit for bit. A released
    row's `[0, P)` are the cold bits; its suffix holds what it decoded while grouped — so
    B ≥ 3 greedy answers also depend on cache contents / arrival order (determinism:
    `QWEN_CASCADE=0` or `QWEN_MAX_BATCH=1`, with `QWEN_PROMPT_CACHE_REPLY=0`). Accounting: a group is charged once
    (`_group_bytes`: fp32 prefix padded to chunks + stack at the longest reservation ×1.05
    + max(score transients + one layer's old+new stack, the 2-row dissolve floor)),
    members keep `_member_bytes` (GDN). A failed step drops every group (`_fail_rows`).
    `_plan` raises on any member/batch/length disagreement (`check_lens` every step).
    Health `cascade_groups`/`cascade_rows`; `batch:` log `groups= grouped=`.
    `set_cascade(on)` for tools (scheduler thread only). Members are kept in batch order
    (a group = one slice of the batch: no gather / output take). Measured (9B,
    `docs/perf/2026-10-12/cascade_gpu.md`): step 8K × 16 rows 66.4 → 53.7 ms (×1.24),
    32K × 8 83.6 → 68.2 ms; 16 × 32K now all admitted (was 12 + 4 waiting), peak 25.3 →
    23.2 GB; 16 × 8K active −3 GiB. Caveats: formation builds the fp32 prefix while the
    members still hold their fp16 caches (+2.2 GB peak at 8 × 32K); 32K × 16 steps at
    92 ms vs ~62 projected (open, needs a profile). The floor and the staged releases
    came after B4 (CPU tests only). Tests `tests/test_cascade_sched.py` (+ `test_cascade.py`).
  - Tests: `cd src/worker && uv run --group dev pytest -q --ignore
    tests/test_mrqmm.py` (CPU, tiny random models, ~15 s; the scheduler's key
    tests also run on `dense_fused`: 8-bit fp16-scale weights + `QWEN_GDN_FUSE`).
    `tests/test_mrqmm.py` runs Metal kernels (skipped without Metal; also the
    248K-vocab grouped-sampling check) — only with the GPU lease;
    likewise `PFQMM_GPU=1 … pytest tests/test_pfqmm.py` (the prefill kernel; its
    dispatch rules run in the CPU suite).
- **RoPE position bugs fixed with batching (both confirmed on the real models)**:
  (A1) DFlash's first target call after prefill rotated the last prompt token as
  position 0 (`_rope_deltas=None` → `get_rope_index` of a 1-token input; |Δlogit|
  0.88 on the 9B) — `_spec` now sets zero deltas. (A2) an image prompt longer
  than one prefill slab gave image tokens text positions from slab 2 on (|Δlogit|
  0.30 at slab 512) — vision slabs now pass `position_ids[..., pos:pos+n]`, sliced
  from what `get_input_embeddings` returns (0.7 no longer sets `_position_ids`).
  Outputs of DFlash and of multi-slab vision prompts therefore differ from older
  workers.
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
  into the token stream; KV cache stays consistent; 0.7: `prompt_preopens_thinking=True`
  or it never counts, `pop_forced_token_id()`). mlx-vlm ≥ 0.6.3 renders
  `enable_thinking=False` when unset — `_render` defaults it to True (template default). Do NOT hand-splice + restart
  generation — the earlier crutch corrupted answers and is removed. Thinking on/off
  is the template's `enable_thinking` (gateway, later wins: chat_template_kwargs <
  top-level enable_thinking < GATEWAY_THINKING; reasoning_effort fills it only when
  chat_template_kwargs did not; unset = template default = thinking). The template *pre-opens*
  `<think>`, so the worker decides "generation starts inside reasoning" from the
  prompt ids (last of `<think>`/`</think>` is an open one), not from a flag.
  `_GenState` = vLLM's `qwen3` reasoning parser (see README).
- **Prompt-length validation** (worker, before generation → `context_length_exceeded`
  → HTTP 400): count image tokens via the processor output (`_vision_inputs`, mlx-vlm `prepare_inputs` minus MLX),
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
- **Log-probabilities (2026-10-22, `docs/perf/2026-10-21/design/logprobs.md`, review
  `reviews/logprobs.md`)**: `logprobs` / `top_logprobs` (≤ 20) per generated token (reasoning,
  answer and the final EOS / stop token; raw logits before penalties / temperature, fp32
  `logit − logsumexp`, built in `_step` on the same `async_eval` as the token and read after
  `r.y.item()` only for rows that asked — the off path adds no op) and `prompt_logprobs`
  (teacher forcing: the prefill keeps `return_hidden`'s post-norm hidden state and runs
  `lm_head` in `QWEN_LOGPROBS_CHUNK` (256) row chunks, fp32 log-softmax of the next prompt
  token; ~+10–12 % of a slab's FLOPs, only for such requests). Scored positions
  `[start + 1, n)`: position 0 never, nothing at or below the cache resume point (its logits
  are not kept) — `prompt_cache: false`/`"off"` = no lookup (cold, every position; still
  stored), `"read"` = lookup, nothing stored (variants of one baseline); the lookup for
  prompt_logprobs is over `ids[:-1]` so the last token is always scored. `max_tokens: 0`
  (only with prompt_logprobs) ends the row inside `_prefill` after the store (`length`, no
  `_join`, no decode step; the entry = prompt[:-1] as a generation's). Spans
  (`_assistant_spans`): the k-th `<|im_start|>assistant` block ↔ the k-th assistant message,
  matched from the end on a count mismatch (logged); `[start, end)` = after the header + the
  template's `<think>` pre-open ([`<think>`, `\n\n`, `</think>`, `\n\n`] = thinking off,
  [`<think>`, `\n`] = on) through `<|im_end|>`. The Ornith template renders a history
  assistant message exactly as generation prompt + reply (tested on the real tokenizer), so
  teacher-forced sums equal generation logprobs up to prefill-vs-decode kernel rounding
  (CPU 8-bit fp16: ≤ 3e-3 per token). Wire: `Token.lp` (not aligned with `text`; an empty
  Token may carry the EOS's lp), one `PromptLogprobs` event before Done; gateway: OpenAI
  `choices[0].logprobs` (always present, null when off), top-level `prompt_logprobs` (null
  where not scored) + `prompt_logprobs_messages` (per span: tokens / scored / logprob sum),
  journal keeps only counts and sums. DFlash never takes such requests. Tests
  `tests/test_logprobs.py`; real-model check `scripts/perf/logprobs/check_teacher_forcing.py` (9B, 2026-10-22,
  `docs/perf/2026-10-21/logprobs_check_real.json`: 5 AppWorld calls, generation byte-equal with the flags
  off, |Σtf − Σgen| ≤ 0.019 nat on 23–50 tokens, hit == cold values, 9.5–12.4K cold scored in 28–38 s).
- **Structured output / constrained tools** = **llguidance** (already an mlx-vlm
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
- **Gateway `ReqGuard`** (Drop) owns both slots (`pool::Slot`: permit + its gauge,
  decremented on drop) AND fires `Worker.Cancel` on client disconnect. Without
  owning the permits there, `queue_depth` stuck non-zero after a mid-stream disconnect.
- **Gateway admission**: `MAX_QUEUE` admits (waiting + running; 503 beyond, before any
  worker call) → `Check` (400/500) → wait for one of N in-flight slots → `Generate`.
  N = the worker's `HealthReply.max_batch` (0/1 = serialized; re-read every 15 s,
  the semaphore resized) or a fixed `GATEWAY_MAX_INFLIGHT`. Several gateways on one
  worker each send up to their own N, so they should split `max_batch`. Oversubscribing
  is NOT harmless: every Generate holds a worker gRPC thread while it waits in the
  worker, so Check/Cancel/Health starve, and the worker caps concurrent Generates with
  `RESOURCE_EXHAUSTED` → gateway 503 (slots freed). Check has a deadline
  (`GATEWAY_CHECK_TIMEOUT_S`, 30) → 503. The worker gets a per-generation id
  (Cancel goes by it; a reused client `request_id` must not cancel another request);
  responses keep `chatcmpl-<request_id>`. Duration/TTFT metrics start at handler
  entry; `qwen_gateway_queue_seconds` = wait for a slot. All histograms have buckets.
- **Gateway list caps** (vLLM has none): `stop` ≤ 64, `stop_token_ids` ≤ 1024,
  `bad_words` ≤ 1024, `allowed_token_ids` ≤ 2^18; token ids must fit uint32 (the
  vocab check is the worker's). Journal lines go through a writer thread (bounded
  10k-line channel, ordered, dropped + counted when full; flushed on SIGINT/SIGTERM,
  exit 130/143).

## Env knobs

| var | where | default |
|---|---|---|
| `QWEN_MODEL` | worker | `weights/qwen36-35b-a3b-mlx-ud3` (repo-relative) |
| `WORKER_ADDR` / `GATEWAY_ADDR` / `MAX_QUEUE` | gateway | `127.0.0.1:50051` / `0.0.0.0:8080` / `32` |
| `GATEWAY_MAX_INFLIGHT` / `GATEWAY_CHECK_TIMEOUT_S` | gateway | unset = the worker's `max_batch` (Health, polled every 15 s; 1 for an old worker) — generations sent to the worker at once / `30` (Check slower → 503) |
| `RUST_LOG` / `LOG_FORMAT` | gateway | `info` / pretty (set `json` for Loki) |
| `LOG_LEVEL` | worker | `INFO` |
| `QWEN_MAX_CONTEXT` | worker | `131072` |
| `QWEN_KV_BITS` | worker | `4` (35B); `0` (fp16 cache) for the 9B; `8` for the 27B — since 2026-11-15 quantized KV also works with the drafter (DFlash) and `QWEN_PROMPT_CACHE_MOVE` (27B q8: 34 vs 64 KiB/token; `docs/perf/2026-11-15/design/kv_q8.md`, GPU gate pending) |
| `QWEN_FP16` | worker | `1` (cast bf16→fp16 after load; `0` = bf16) |
| `QWEN_TEXT_ONLY` | worker | `0` — `1`: a VL checkpoint loads without its vision tower (`engine._load_model`: `mlx_vlm.load(lazy=True)`, `vision_tower = None` — the text-only checkpoint's state — then the eval, so the `vision_tower.*` tensors are never read from disk: 27B 0.86 GiB, 35B 0.83, 9B 0.85 bf16; CPU test: load peak = the LM's bytes); image requests → 400 (`invalid_request`, as a text-only checkpoint); text bit-identical. Load log: `vision tower not loaded (N tensors, X GiB …)`. GPU memory before/after on the 27B not measured yet |
| `QWEN_DRAFT` / `QWEN_DRAFT_BLOCK` | worker | unset / `0` — DFlash drafter dir (`weights/qwen35-9b-dflash`), speculative decode for greedy text requests |
| `QWEN_WIRED_LIMIT_GB` / `QWEN_PREFILL_STEP` | worker | `0` (off; clamped to cap) / `2048`; shrunk so slab×offset ≤ `QWEN_PREFILL_BUDGET` (2048×32K) for quantized KV, and for fp16 KV while the slab stays under 32K keys; fp16 KV + fused SDPA (9B with `QWEN_KV_BITS=0`) keeps it from 32K keys on |
| `QWEN_TEMPERATURE` / `QWEN_TOP_P` / `QWEN_TOP_K` / `QWEN_MIN_P` | worker | `0.0` (argmax) / `1.0` / `0` / `0.0` |
| `QWEN_PRESENCE_PENALTY` / `QWEN_FREQUENCY_PENALTY` / `QWEN_REPETITION_PENALTY` | worker | `1.5` / `0` / `1.0` — **benchmarks/reasoning: presence `0`** + card sampler (see README) |
| `QWEN_JSON_MAX_WS` | worker | `0` (off) — whitespace-run cap in response_format answers |
| `GATEWAY_OPENAI_DEFAULTS` / `GATEWAY_THINKING` / `GATEWAY_SERVED_MODEL_NAMES` / `GATEWAY_DEFAULT_MAX_TOKENS` | gateway | unset / unset (`on`/`off` forces) / unset / `0` (= rest of context) — see README |
| `QWEN_PROMPT_CACHE` / `_GB` / `_STRIDE` / `_CKPTS` | worker | `0` / `3` (soft cap: evicted at store; counted live in the batch budget, evicted on demand; order: wanted by the queue last, stale LRU first, else MRU) / `2048` / `12` (system-block end + last 3 message starts never thinned) |
| `QWEN_PROMPT_CACHE_DISK` (+ `_GB` / `_KEEP` / `_QUEUE_GB` / `_MIN_TOKENS` / `_WAIT_S` / `_HELD_WAIT_S` / `_TIMEOUT_S` / `_NOCACHE`) | worker | unset = off / `32` / `0` / `4` / `4096` / `10` / `5` / `30` / `1` — SSD tier of the prompt cache (`pcdisk.py`): evicted entries (≥ MIN_TOKENS) are written by a background thread (zero-copy views, one file per segment / GDN state, deduplicated, LRU under `_GB`), a waiting request whose disk hit is ≥ a slab deeper than its RAM hit gets the entry loaded (reader thread → constant mx.arrays, inserted on the scheduler under the usual budget); RAM held by pending writes / loads counts in every budget check (`pc.charged()`); disk hit == RAM hit == cold bit for bit; directory cleared at start unless `_KEEP=1` with the same validity key. AppWorld replay at `QWEN_PROMPT_CACHE_GB=3`: 10.3 → 23.4 req/min (×2.3), MISS 35 → 7 %, load p50 0.32 s; peak +3 GB (longer rows, not the tier) |
| `QWEN_PROMPT_CACHE_STALE_S` | worker | `300` — unused this long and wanted by no waiting request = stale (evicted first, LRU) even within the round; `0` = off |
| `QWEN_PROMPT_CACHE_CHAIN_MARKS` / `_CHAIN_TAIL` | worker | `3` / `4096` — an entry that absorbed the previous turn keeps only the system-block end, the last N message starts, strides in the last N tokens and in the first user message + its end state; `-1` marks = off |
| `QWEN_PROMPT_CACHE_CHAIN_FRESH_TOKENS` | worker | `16384` — fresh entries this long take the chain policy (+ the divergence point); shorter: the cap policy; `0` = never |
| `QWEN_PROMPT_CACHE_REPLY` | worker | `1` — a finished request's entry is extended by its decoded tokens' KV + end state; the next turn resumes after the reply (AppWorld: −21 % prefill tokens); not bit-identical to a cold prefill (batched-decode class: top-1 99.88 %, KL 5.8e-5 on the gate); `0` = off, bit-exact |
| `QWEN_PROMPT_CACHE_MOVE` | worker | `0` — one KV copy per conversation (single stream only: `QWEN_MAX_BATCH=1`, no disk tier; else off, load log; quantized KV: quantized entries move, the fp16 ones below `quantized_kv_start` are copied): a hit on the conversation's own entry moves its buffers into the row (the entry leaves the cache), the prefill-end store is deferred, the ended row's own buffers become the entry (prompt + reply). Same answers / hits as the copies (parity tests; GPU: hit == cold token for token). A row failing in a forward loses the entry; a sibling's entry is copied while a copy fits, else it moves. The cache gives way to a lone row down to the batch budget, never into the prefill reserve; a sibling's shared prefix that would pin more than its own bytes of the parent's segments is kept as the row's own copy (`_pins`; 2026-11-15, the stand's OOMs: a 332-token prefix held a dead 27K entry's 1.6 GB). 27B + drafter: 60K conversations hit (before: entries > ~40K "not kept") |
| `QWEN_PROMPT_CACHE_KEEP_TOKENS` / `_KEEP_WAIT_S` | worker | `8192` / `10` — when only the entry the head of the queue would resume from keeps it out, it waits for a row to end only for a hit this deep and only this long; else the entry is evicted and the head admitted cold (8K ≈ 14 s of 9B prefill > the 10 s wait) |
| `QWEN_DRAFT_SAMPLING` | worker | `0` — 27B 4-bit + DFlash2 at T 0.6 / top_p 0.95 / top_k 20 (2026-11-16, `docs/perf/2026-11-16/drafter.md` §5): vs plain sampling ×1.34–1.41 on 2K code / en / ru prompts, ×1.0 on short English prose, ×0.70 on short Russian prose (< 1 draft accepted per round; greedy DFlash also ×0.75 of plain there) — stays opt-in |
| `QWEN_LOOKUP` | worker | `0` — prompt lookup decoding (`lookup.py`, `_spec`): a round's drafts are the continuation of the latest earlier occurrence (prompt + output) of the text's last 3..`_NMAX` tokens, verified as a drafter's block (DFlash's exactness class). `draft`: rows the drafter takes, a lookup round when matched, else the drafter's block (its context = every position since its last block, `_DraftCtx`); `1`: the same + without a drafter lookup rounds between plain step-ahead bursts (not bit-equal to plain: the rounds' verify rows); `splice`: drafter every round, the continuation where it agrees (slower than `draft`, not recommended). 27B 4-bit + DFlash2, greedy, 384 tokens, in-run A/B: `draft` vs DFlash: rewrite a 2K-token file with one change ×2.4–2.8 (71–79 vs 27–30 tok/s; plain 18), re-emit the previous answer's code ×1.55–1.67 (80–84 vs 48–54), ordinary code / English / Russian 2–3.6K prompts ×0.985–1.05; answers token-identical to DFlash's (gate passed on this set). Without a drafter ×2.9–3.9 on copies, ±1 % elsewhere, not bit-exact vs plain. Default off until the 9B / long context are measured: `docs/perf/2026-11-16/lookup_draft.md` |
| `QWEN_LOOKUP_N` / `_NMAX` / `_MIN` / `_K` / `_K0` / `_BACKOFF` / `_BURST` | worker | `3` / `32` / `4` / `15` / `7` / `1` / `16` — index n-gram / longest match scored / shortest match taken / max drafts (≤ 15: mrqmm's 16 rows) / first drafts of a round (2k+1 after a fully accepted one, K0 after a miss; 27B 4-bit verify 4 rows 99 ms = 8 rows 106, 12 = 16 rows 155–159) / a round keeping < half its drafts doubles the match the next needs (reset at a full accept) / plain steps per burst without a drafter |
| `QWEN_VERIFY_ATT` | worker | `2` (since 2026-11-16; `1` before) — `1`: a DFlash verify block's attention through MLX's vector attention kernel, `32 // GQA factor` queries per call (27B 5, 9B 8; longer blocks in chunks) instead of mlx-vlm's one query per call (the row's whole KV read per block token); bit-equal to those calls (self-test on random data at load, `_probe_verify_attention`, else off); 27B block 8: round 570 → 296 ms at 60K keys, 210 → 202 at 24K; fp16 KV, lone row; `0` = the library's calls; quantized KV: the library's one-query calls (no quantized kernel yet); **`2`** = one pass (`vatt.py`, 2026-11-16): the block's G × T query rows per kv head attend the KV together (steel-attention MMA loop over key chunks, causal tail per block position, fp32 partials + LSE merge, the KVCache buffers read in place), KV read once a round whatever T (≤ 16, head_dim 256); not the per-query calls' bits (teacher-forced 32K: KL 1.3e-5 vs `1`, top-1 100 %; error vs fp32 2.0e-4 < MLX's 2.6e-4); self-tested at load for T 2..16 (`_probe_vatt`), else `1`. 27B 4-bit, one layer: 60K keys 8.14 → 2.45 ms, 80K 10.2 → 3.07, 8K 1.18 → 0.46; DFlash at 60K 17.97 → 26.54 tok/s (×1.48, round 284 → 192 ms), 32K 22.31 → 30.13 (×1.35), short prompts equal (30.9); tokens == plain 9/9 at 32K and at 60K; 16 short tasks 12/16 == plain, the same as `1` on the 4-bit 27B (divergences at gaps ≤ 0.016). 9B layout one layer: ×2.3–3.1 vs `1`, not gated at model level. Default `2` since 2026-11-16 (the short-task identity is the mrqmm verify's, equal for `1` and `2`); `1` = the library's bits. `docs/perf/2026-11-16/verify_att2.md` |
| `QWEN_DECODE_ATT` | worker | `0` (off) — `N`: a decode step's attention over ≥ N fp16 keys through the same one-pass kernel (T = 1: the G query heads as one 8-row MMA block) instead of MLX's 2-pass `sdpa_vector` (27B layout: ~90 GB/s); every decode row (lone row and each row of a batch, via the `_fused_sdpa` override), not prefill / quantized KV; self-tested at load (`_probe_decode_att`). 27B one layer: 60K 2.77 → 0.91 ms (276 GB/s), 32K 1.07 → 0.55, 8K equal; plain decode at 60K 10.61 → 14.37 tok/s (×1.35), tokens == plain 3/3; decode-step KL 1.2e-5, top-1 100 %. Not MLX's bits. 9B layout (G 4): MLX's kernel already reads ~300 GB/s there and vatt is 8–17 % slower at ≥ 8K keys, so keep it off for the 9B; 32K plain 14.87 → 16.53 tok/s (×1.11). Recipe value `8192` (27B) |
| `QWEN_DRAFT_TAIL` / `QWEN_DRAFT_TAIL_END` | worker | `128` / `2048` — every prompt-cache state a DFlash row stores carries the drafter's context there: the target's hidden states of the last N positions (27B: 51 KiB each), as one more item after the state's recurrent layers (`_with_tail`: rides through inherit / absorb / thinning / the disk tier, counted in the entry's bytes). `TAIL`: checkpoints (6.6 MB each); `_END`: the entry's end state — prompt end, cancelled prefix, reply end — capped at the drafter's window − 1 (≤ 105 MB per entry). A DFlash row resuming from a state starts its drafter with them; a checkpoint inside the end state's window takes the part of the end context before it (`_resume_tail`, no extra memory). 27B replay (`ctx_len.py`): rounds for the same 768 tokens 245 / 226 / 217 / 218 / 221 at N = 1 / 16 / 32 / 128 / 2047 (cold); but a turn re-emitting the previous answer's code needs the whole source (128 at the reply end: 52 rounds vs the cold run's 38; window: 38). `0` = off |
| `QWEN_DRAFT_FP32` | worker | unset = auto (on for DFlash2 drafters) — the drafter's activations in fp32 (weights stay `QWEN_DRAFT_BITS`): the 27B's DFlash2 overflows fp16 after its first layer (residual 5.7e4 → NaN logits, 0 drafts accepted); fp32 3.16 of 7 accepted at 4 bits vs 3.28 unquantized, block 36 vs 32 ms; `1` / `0` force. `docs/perf/2026-11-10/dflash2_27b.md` |
| `QWEN_DRAFT_MRQMM` | worker | `1` — the fp32 drafter's quantized linears (DFlash2: `QWEN_DRAFT_FP32`, `QWEN_DRAFT_BITS` 4 / 5 / 8; 47 of its modules) through `mrqmm_wide` for 1..16 rows: each row prescaled by an exact power of two (max \|x\| in [2^10, 2^11)), cast to fp16, the kernel's fp32 sums written in fp32 × the inverse scale (`kernels._drafter_linear`); scales / biases held in fp16 (checked exact, else the module stays). 27B 4-bit (2026-11-16, `docs/perf/2026-11-16/drafter.md`): drafter block 31.3 → 17.4 ms (MLP 17.1 → 5.7), the same drafts on 240 probes, DFlash 25.8 → 29.6 tok/s (×1.117 in-run; code / en / ru 2K ×1.13 / 1.12 / 1.11), identity vs plain unchanged (12/16, same divergence points). Only the drafter (the target verifies). `0` = `quantized_matmul` in fp32 |
| `QWEN_DRAFT_COMPILE` | worker | `0` — the DFlash2 drafter's grouped dynamic convolution compiled (`kernels._grouped_conv`): block −0.65 ms, decode +0.3 % (under the 3 % gate) |
| `QWEN_CASCADE` | worker | `1` — cascade decode attention for rows sharing a prompt prefix (fp16 KV + prompt cache + max_batch ≥ 3 only, else off; unset/empty = `1`, anything but `1` off); `0` = today's per-row path bit for bit |
| `QWEN_CASCADE_MIN_ROWS` / `_MIN_PREFIX` / `_CHECK` | worker | `8` (1–2 → 3, 0 = off) / `4096` (0 = off) / `0` (bitwise prefix check per join; a sync) |
| `QWEN_CLEAR_CACHE` / `QWEN_CACHE_LIMIT_GB` | worker | `1` / `3` |
| `QWEN_MAX_BATCH` | worker | `16` — rows decoded together (`1` = serialized); Health reports it (1 when the budget holds < 2 typical rows) |
| `QWEN_PREFILL_RESERVE_GB` | worker | `0` = estimated (`_prefill_reserve`: attention scores + 0.4 GiB) — memory kept outside the batch budget for a prefill slab's activations; the estimate is 1.15 GiB for the 27B + drafter at `QWEN_PREFILL_BUDGET=16777216`, its DFlash prefill peaked 2.0–2.7 GiB above the KV (stand log 2026-11-14): set the measured peak there |
| `QWEN_BATCH_MEM_GB` / `QWEN_BATCH_RESERVE_TOKENS` | worker | `0` = derived (cap − weights − prefill reserve − 1 GiB − pool if kept; the prompt cache's live bytes count inside it) / `4096` (generation tokens reserved per row at admission) |
| `QWEN_BATCH_PREFILL_SHARE` | worker | `0.5` — prefill's time share when the batch is full; with B of max_batch rows it is `1-(1-s)·B/max` |
| `QWEN_PFQMM` | worker | `1` — prefill matmul (`pfqmm.py`): MLX's own qmm_t code with 64-row tiles (a 32-row last tile when ≤ 32 rows are left), ≥ 64 rows where MLX runs qmm_t; ×1.03–1.28 per matmul (8-bit ×1.10–1.14), bit-identical, self-tested per key at load; `0` = MLX `quantized_matmul`. Needs fp16 scales: `QWEN_FP16=0` leaves it off |
| `QWEN_MRQMM` | worker | `1` — multi-row 8-bit matmul kernel (`mrqmm.py`) for batched decode with 3..16 rows (8-bit affine g64 fp16 modules with N ≥ 512 only); `0` = MLX `quantized_matmul` |
| `QWEN_MRQMM5` | worker | `0` — the same multi-row kernel for 5-bit affine g64 modules (the 27B: every linear incl. lm_head and the fused GDN in-projection), 3..16 rows: batched decode, the fused GDN in-projection and the DFlash verify block + drafter lm_head. Linears of one 27B forward at 8 rows 249 → 89 ms (×1.61 of a solo row, stock ×4.5). Off by default: the 27B's batched decode (B ≥ 3) is not gated yet; DFlash2 on the 27B needs it (`docs/perf/2026-11-10/dflash2_27b.md`). Ignored with `QWEN_BATCH_INVARIANT=1` |
| `QWEN_MRQMM4` | worker | `0` — the same multi-row kernel for 4-bit affine g64 modules (`weights/qwen38-27b-mlx-4bit`, lmstudio-community: every linear 4-bit), everywhere `QWEN_MRQMM5` takes 5-bit ones (batched decode 3..16 rows, fused GDN in-projection, DFlash verify block + drafter lm_head). Own knob: separate GPU gate, and a mixed-width model (35B: 4-bit shared expert) is not pulled in by `QWEN_MRQMM5`. Self-tested on the GPU at load before its first dispatch (`mrqmm._selftest`: unpack exact on one-hot x at every in-group offset, both row buckets + real-scale rows 3/8/9/16 vs fp32 and `quantized_matmul`); a failure logs `self-test FAILED` and leaves 4-bit on `quantized_matmul`. Gated 2026-11-15 on the 27B 4-bit (`docs/perf/2026-11-16/profile_27b.md` §5): DFlash ×1.75 (32.3 vs 18.4 tok/s; without it ×0.72 of plain), batched step ×1.5–2.95 vs qmm, teacher-forced top-1 100 / 100 / 99.80 % at B = 3/8/16 (KL ≤ 1e-5, stock qmm 100 / 99.90 / 99.85 %), DFlash identity 12/16 with all 4 divergences at fp16 near-ties; stays off by default (identity count < 15/16, the 35B's shared expert ungated) — set it for the 4-bit 27B. Ignored with `QWEN_BATCH_INVARIANT=1` |
| `QWEN_MRQMM6` | worker | `0` — the same multi-row kernel for 6-bit affine g64 modules (oQ4e: the lm_head 248320×5120 and the last layer's k/v; N ≥ 512), everywhere `QWEN_MRQMM5` takes 5-bit ones; above all the DFlash verify block's lm_head and the drafter's head (without it: mlx-vlm's per-token path, ~8 head reads per round, and `quantized_matmul`). Self-tested at load like 4 bits; gated 2026-11-15 on oQ4e (`docs/perf/2026-11-15/design/oq4e_speed.md`): without it the DFlash round loses 37.1 → 30.2 tok/s, with it oQ4e equals the uniform 4-bit (37.1 vs 37.6), identity `m8/8` 15/16 and `plainnf` 16/16 — set it for oQ4e; off by default. Ignored with `QWEN_BATCH_INVARIANT=1` |
| `QWEN_GDN_FUSE` | worker | `1` — one matmul for a decode step's 4 GDN in-projections (weights concatenated at load, modules hold views); a mixed-width layer (oQ4e) fuses each group of ≥ 2 same-width projections (load log `fused by width`), the rest stay separate; `0` = 4 launches. Needs fp16 scales (`QWEN_FP16=1`): with `QWEN_FP16=0` it fuses nothing |
| `QWEN_SDPA_FUSED` | worker | `1` — force MLX's fused SDPA for fp16-KV prefill slabs ≥ 512 over ≥ 32K keys (per call ×0.89 at 2048 × 32K, ×0.6–0.7 at 64K+; 64K prefill −3…5%, −1.3 GB peak) |
| `QWEN_SDPA_FUSED_MIN_KV` | worker | `32768` — keys from which a long fp16 prefill slab takes the fused SDPA kernel (the gate of `QWEN_SDPA_FUSED`); `8192` + `QWEN_PREFILL_BUDGET=16777216` cap the unfused scores at 0.75 GiB and the prefill reserve at 1.15 GiB instead of 3.4 (27B, 24 heads: the hump just under 32K keys is 2.8 GiB) — for the 27B + drafter at long context |
| `QWEN_KVQ_PREFILL_FUSED` | worker | `0` — quantized KV (`QWEN_KV_BITS` 4/8): prefill slabs the fused-SDPA gate takes with fp16 KV attend the layer's keys / values dequantized to fp16 (`mx.dequantize`, `kernels._dequantized_kv`) through the fused kernel instead of the library's quantized SDPA (two quantized matmuls over materialized fp16 scores); the slab stays `QWEN_PREFILL_STEP` at any context (`_FIXED_SLAB`; off: shrunk to `QWEN_PREFILL_BUDGET` // offset, 546 at 120K); prefill reserve + one layer's dequantized K + V at `QWEN_MAX_CONTEXT` × 1.6 (the measured transient: 504 MiB at 80K keys; 27B 128K: 819 MiB). 27B 4-bit (2026-11-15): 32K teacher-forced KL vs fp16 KV 5.6e-5 / top-1 99.70 % (q8 without it 6.0e-5 / 99.66 %); prefill attention per token = fp16's (q8 ×1.2 at 32K, ×1.5 at 80K). Off until the e2e gate (`scripts/perf/kvq8/`) |
| `QWEN_GDN_FAST` | worker | `0` — `mx.fast.gated_delta_update` for prefill slabs (×1.02 per slab; rounds the decay to fp16: KL 0.013 on 8K real tokens — rejected) |
| `QWEN_VERIFY_EXACT_MAX_T` | worker | `8` — DFlash verify blocks ≤ this use mlx-vlm's exact verify kernels (DFlash == greedy), longer ones `quantized_matmul`; 0 = never, 16 = every adaptive block (the drafter's block_size 16; slower; a `QWEN_DRAFT_BLOCK` > 16 still has longer blocks on `quantized_matmul`); empty = default |
| `QWEN_BATCH_INVARIANT` | worker | `0` — batch-invariant decode (above, Continuous batching): row in a batch == solo bit for bit at 1..16 rows on the 9B; cascade + DFlash off; B=1 step ×1.18 |
| `QWEN_VERIFY_MRQMM` | worker | `1` — DFlash verify blocks T ∈ [3,16] through mrqmm as (T,1,K) + the drafter's lm_head; with `QWEN_DRAFT_BLOCK` unset a fixed block 8 (DFlash ×1.47 vs the old adaptive exact path: 59–68 tok/s vs plain 34–36); `head` = lm_head only (exact verify, ×1.05); `0` = the old path |
| `QWEN_SAMPLER_TOPK_FAST` | worker | `0` — with top_k > 0: min_p / top_p / the draw over the k candidates (ties at the k-th as one bucket) instead of a full-vocab argsort; same distribution, different RNG stream for a seed (hence off); card sampler at B=16 7.8 → 3.5 ms, e2e step −7 % |
| `QWEN_PREFILL_LANE` / `_BURST` | worker | `2048` / `4` — a text request with ≤ N tokens left to prefill (after its prompt-cache hit) is prefilled whole between the slabs of a running long prefill (≤ BURST in a row); the long prefill's slabs are unchanged (bit-identical); gate: short requests during a 40K prefill finish 19 s vs 76 s, the long one ×1.13; `0` = one prefill at a time |
| `QWEN_PREEMPT` | worker | `2` — a row outgrowing the batch's headroom is preempted (its own KV buffers become a prompt-cache entry, no copy; back to the head of the queue; resumes from that hit) instead of ended with `length`, at most N times per request; the stream continues bit for bit (sampler key, penalties, grammar, logprobs); not for vision / DFlash / cascade-grouped / prompt_logprobs / prompt_cache read-off rows; `0` = the cut |
| `QWEN_DEBUG_NO_SYNC` | worker | `0` — debug: any CPU↔GPU sync (`mx.eval`, `.item`, `.tolist`, `np.asarray` …) while a decode step is built raises `SyncInStep` (catches step-ahead regressions; on in test_lane / test_preempt) |
| `QWEN_CASCADE_KERNEL` | worker | `0` — cascade groups attend the shared prefix with a simdgroup-MMA Metal kernel (`cascade_mma.py`) over the donor's fp16 buffer (aliased, no copy) instead of an fp32 copy: −2.3 GB peak at 16 × 32K, no formation transient, groups win from 4 rows; attention ×1.4 at ≤ 8 rows, ×0.84 at 32K × 16 (hence off) |
| `QWEN_SPLIT_SPECIAL_TOKENS` | worker | `0` (as vLLM) — added-token strings inside user / tool text and tool-call arguments tokenized as plain text, the template's own control tokens by id (`<|im_start|>` in a tool output otherwise becomes a real turn boundary); `docs/perf/2026-10-24/design/special_tokens.md` |
| `QWEN_LOGPROBS_CHUNK` | worker | `256` — rows per `lm_head` chunk when scoring a prompt (`prompt_logprobs`): 256 × 248K × 6 B ≈ 0.38 GB transient |
| `QWEN_GEN_SLOTS` | worker | `max(2·max_batch, 8)` — concurrent Generate RPCs, beyond = `RESOURCE_EXHAUSTED`; gRPC threads = slots + 8 (replaces `QWEN_GRPC_THREADS`) |
| `MAC_EXPORTER_ADDR` | exporter | `0.0.0.0:9101` (run under sudo) |

## Measuring performance on this machine (lessons from 2026-10-11/12)

- The GPU speed drifts ±20–30% over minutes (user activity, thermal, power state): the
  9B solo decode probe read 18–38 tok/s across one night. Never compare two separate
  runs; A/B *inside* one process, interleaved, medians over ≥ 3 rounds
  (`scripts/perf/prof/step_prof.py --ab`, `prof/ab_prefill.py`). For end-to-end runs
  `scripts/bench_concurrency.py --repeat 2 --alternate --probe 64 --cooldown 30` and
  compare each level against its probe; a 64-token probe still misses a slow window
  (11:24–11:37 on 2026-10-12 ran 20–40% slow with probes at 38), so repeat a level
  that disagrees with its neighbours before believing it.
- zsh gives background jobs nice 5 (`BG_NICE`): launch the worker/gateway/bench with
  `setopt NO_BG_NICE` (or in the foreground); the first baseline ran its worker at nice 5
  unnoticed.
- Nothing else on the GPU during a measurement: cargo/pytest/another worker/video
  playback all show up in the numbers. Close Activity Monitor too.
- Closed-loop clients (N in flight, each re-sends on completion) make arrivals come
  in waves: at c=16 all rows finish together, 16 prompts prefill one at a time, so
  TTFT p95 is ~N × prefill — a property of the load, not of the scheduler.
- Aggregate tok/s through the gateway counts completion tokens over the level's wall
  time, prefill included; the worker log's `batch=`/`queue_ms=` join by
  `chatcmpl-<request_id>-<12 hex>` (the gateway's per-generation id).
- Worker-side TTFT (`Usage.ttft_ms`, the `w_ttft_ms` columns of `bench_concurrency.py` /
  `scripts/perf/final/pcache_tables.py`) counts from the start of prefill since Part A
  of the segment store (2026-10-12; `queue_ms` + `ttft_ms` = arrival → first token).
  Before it, `t0` was admission, so a request admitted behind another prefill had the
  decode-share wait in `ttft_ms` (and not in `queue_ms`): `w_ttft_ms` columns from
  before (e.g. `prompt_cache_bench.md`) are not comparable for such requests. The
  gateway's `qwen_ttft_seconds` (its own clock) is unaffected.
- The old (f2bcbdf) baseline was taken on a drifting machine; "old → new" at c=1
  (22.5 → 31.9 tok/s) is mostly the machine. The honest batching gain is the
  serialized-vs-batched pair on the same code (`docs/perf/2026-10-12/final_bench.md`).
- Reports, raw JSON and the tools: `docs/perf/<date>/` + `scripts/perf/` (prof, microbench,
  final driver). Scripts that write files use `PERF_OUT` (default `/tmp/perf`).

## Gotchas

- **MLX random state is per thread, and mlx_lm's sampler is compiled against the
  main thread's key.** gRPC handlers run in pool threads, so `mlx_lm.make_sampler`
  there drew every token from the same never-advancing key: temperature 0.7
  returned byte-identical answers, and `mx.random.seed` from the handler thread
  changed nothing. The worker's own `make_sampler` (worker.py, no mlx_lm since the
  0.7.6 bump) filters top_p/min_p/top_k itself and draws with an explicit per-request key split
  per step; request `seed` feeds that key (reproducible, verified through the
  gateway). Test any sampler change from a thread, not the REPL main thread.
- DFlash greedy and the plain loop are byte-identical only for verify blocks
  ≤ `QWEN_VERIFY_EXACT_MAX_T` (8): longer blocks go through `quantized_matmul`,
  which rounds differently from the 1-row decode, so a near-tie can flip there
  (the adaptive block reaches 16; 15/15 corpus identical at cap 8 nonetheless).
- DFlash's verify block was adaptive on acceptance stats; since mlx-vlm 0.7.6
  `draft.reset` clears them per request, so a request's blocks depend on itself only
  (checked by code, 2026-10-24). Since 2026-10-24 the default is `QWEN_VERIFY_MRQMM=1`
  with a fixed block 8 (verify rows through mrqmm as (T,1,K)): DFlash ≠ plain greedy
  bitwise (batched-decode class; 15/16 identical on the corpus, the old adaptive path
  13/16) but run-to-run deterministic (T ≤ 8 → always mrqmm's MB=1). `head` = only the
  drafter's lm_head through mrqmm (verify exact, +4.7 %).
- The gateway waits for the worker's Health before it listens: no worker = no 8080.
- A scheme-less proxy variable (`HTTP_PROXY=localhost:<port>`) → breaks Python `requests`.
  Use `http://...` or `HF_HUB_OFFLINE=1` (models are cached locally in `weights/`).
- mac-exporter needs root (`powermetrics`); load it as a `/Library/LaunchDaemons`
  daemon, not a user agent.
- Observability stack is shared infra — `deploy/` containers are `deploy-*`; don't
  touch the user's other Docker containers.
- The venv breaks on a repo move (absolute paths baked in) — `uv sync` at the root
  and in `src/worker/` to repair.
