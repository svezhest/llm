# Final end-to-end benchmark: Ornith-1.5 9B through the gateway (2026-10-12)

Code under test: the working tree as of 2026-10-12 11:12 (continuous batching + mrqmm + GDN
in-projection fuse + pfqmm + fixed 2048 fp16 slab; mlx 0.32.3 / mlx-vlm 0.7.6 in src/worker).
Gateway binary `target/release/qwen-gateway` built 2026-10-11 23:42, newer than every
`src/gateway/src/*.rs` (no rebuild needed); restarted on 127.0.0.1:8090 at 11:14, it follows the
worker's `max_batch` through the 15 s Health poll (logged `max_batch changed ... new=1` / `new=16`
on the worker restarts). Machine: M1 Max 32 GB, macOS 27; nothing else on the GPU; worker,
gateway and bench at nice 0. Same methodology as `baseline.md` (closed loop, distinct ~610-token
prompts, temperature 0, presence 0, thinking off, `ignore_eos`, stream + include_usage,
max(8, 2N) requests per level, 30 s idle + a 64-token single-stream probe before every level).

## Headline

256 tokens, ~610-token prompts. agg = aggregate tok/s (all completion tokens / level wall time,
prefill included); TTFT p50 includes the queue wait.

| conc | old serialized (f2bcbdf) agg / TTFT p50 | final serialized (`QWEN_MAX_BATCH=1`) agg / TTFT p50 | **final batched (defaults)** agg / TTFT p50 | batched vs final serialized | batched vs old |
|---|---|---|---|---|---|
| 1 | 22.5 / 2.39 s | 31.9 / 1.22 s | 31.2 / 1.25 s | ×0.98 | ×1.39 |
| 2 | 23.0 / 13.1 s | 31.6 / 9.28 s | 48.7 / 2.41 s | ×1.54 | ×2.1 |
| 4 | 22.1 / 35.0 s | 31.4 / 25.4 s | 82.8 / 2.97 s | ×2.64 | ×3.7 |
| 8 | 21.5 / 80.3 s | 31.8 / 57.5 s | 110.4 / 5.32 s | ×3.47 | ×5.1 |
| 16 | — | (= 31.8, serialized) | 115.9 / 10.4 s | ×3.6 | ×5.4 (vs old c=8) |

1024 tokens:

| conc | old serialized agg / TTFT p50 | final batched agg / TTFT p50 | × |
|---|---|---|---|
| 1 | 25.0 / 2.31 s | 34.2 / 1.69 s | ×1.37 |
| 8 | 22.5 / 315 s | 141.9 (134.7, 149.1) / 6.95 s | ×6.3 |
| 16 | — | 169.5 (162.8, 176.2) / 12.9 s | ×7.5 (vs old c=8) |

Prefill, single stream, final defaults vs the pre-change serialized baseline (TTFT through the
gateway, 128-token answers; worker-side ttft in the raw tables differs by ≤ 0.1 s):

| prompt | before | after (final) | × |
|---|---|---|---|
| ~8.15K tokens | 22.1 s (21.9 / 22.1 / 22.8; `ref_prefill_8000.json`, 8160 tok) | 17.2 s (16.95 / 17.42, probe 17.97 at 8.1K) | ×1.28 |
| ~32.7K tokens | no gateway measurement before (baseline.md stops at 8K); in-process old stack 83.5 / 94.6 s at 32768 random ids (`prefill_profile.md`, pre-pfqmm), CLAUDE.md 2026-09: 82 s | 72.5 s (70.26 / 74.70 at 32789 / 32676 tok, probe 70.25 at 32366) | ×1.23 vs 89 s (in-process old mean), ×1.13 vs 82 s |

Peak worker memory (`mx.get_peak_memory`): 10.4 GB at c=1 → 11.0 (c=8) → 11.7 GB (c=16, 256 tok),
12.05 GB at c=16 × 1024 tok; 11.8 GB at 8K, 14.4 GB at 32K. Old serialized: 13.8 GB at every
concurrency (15.2 GB at 8K). No errors, no 503s, no stream error frames in any run.

### Reading the numbers

- **Batching is the big lever**: at the same code and machine state, aggregate throughput goes
  ×1.54 / ×2.64 / ×3.47 / ×3.6 over the serialized worker at 2 / 4 / 8 / 16 clients (256
  tokens), and ×4.1 / ×5.0 at 8 / 16 with 1024-token answers (vs the c=1 rate, which is the serialized rate) (longer answers amortize the
  serialized prefill waves). TTFT p50 stops being queue wait: 57.5 s → 5.3 s at c=8.
- **c=1 costs nothing**: batched worker 31.2 vs serialized 31.9 agg tok/s, decode 37.3 vs 37.6
  tok/s per request (within run noise).
- **Per-request decode falls with batch size** (37.3 / 30.5 / 28.7 / 21.2 / 12.3 tok/s at 1 / 2 /
  4 / 8 / 16 rows), the price of sharing the step; c=16 adds only 5% aggregate over c=8 at 256
  tokens (step 16 rows ≈ MMA-bound, see mrqmm notes) but +19% at 1024.
- **TTFT p95 at c=16 (24 s / 35 s)** is the closed loop's wave effect: with `ignore_eos` all rows
  finish together, the next 16 prompts arrive together, and prefill runs one prompt at a time
  interleaved with decode (`QWEN_BATCH_PREFILL_SHARE` 0.5), so the 16th waits ~15 prefills
  (`queue_ms` mean 8.1 s).
- **Old → final serialized (+41% at c=1)** is mostly not code: per-request decode 28.6 → 37.6 tok/s
  where the bump A/B measured equal decode; the old sweeps ran on a drifting machine (probes
  17.8-30.4 tok/s, video playback in the background), today's probes sat at 38.0-38.6. The c=1
  TTFT 2.38 → 1.22 s is partly pfqmm + fixed slab (×1.23 at 8K in-process) and partly that.
  Compare batching against the final serialized column.
- **mrqmm ablation**: `QWEN_MRQMM=0` drops c=8 from 110.4 to 64.6 tok/s (×1.71 for mrqmm) and
  c=16 from 115.9 to 87.5 (×1.32); per-request decode 21.2 → 9.9 and 12.3 → 7.9 tok/s.

## Commands and env

```zsh
setopt NO_BG_NICE                       # zsh would start & jobs at nice 5
S=$SCRATCH/results
# gateway (repo root)
GATEWAY_ADDR=127.0.0.1:8090 RUST_LOG=info nohup ./target/release/qwen-gateway >> $S/gateway_final.log 2>&1 &
# worker (src/worker); <env> = nothing (defaults) | QWEN_MRQMM=0 | QWEN_MAX_BATCH=1
QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 <env> \
  nohup uv run python worker.py >> $S/worker_final{,_mrqmm0,_mb1}.log 2>&1 &
# all bench runs: repo root
B="uv run python scripts/bench_concurrency.py --url http://127.0.0.1:8090 --probe 64 --cooldown 30"
$B --concurrency 1,2,4,8,16 --repeat 2 --alternate --max-tokens 256  --worker-log $S/worker_final.log --out $S/final_256.json
$B --concurrency 1,8,16     --repeat 2 --alternate --max-tokens 1024 --worker-log $S/worker_final.log --out $S/final_1024.json
$B --concurrency 1 --requests-per-level 2 --max-tokens 128 --prompt-tokens 8000|32000 \
   --worker-log $S/worker_final.log --out $S/final_prefill_{8000,32000}.json
# QWEN_MRQMM=0 worker
$B --concurrency 8,16 --max-tokens 256 --worker-log $S/worker_final_mrqmm0.log --out $S/final_mrqmm0_256.json
# QWEN_MAX_BATCH=1 worker
$B --concurrency 1,2,4,8 --repeat 2 --alternate --max-tokens 256 --worker-log $S/worker_final_mb1.log --out $S/final_serial_256.json
# defaults again: order check for the slow window
$B --concurrency 16,8,4 --max-tokens 256 --worker-log $S/worker_final.log --out $S/final_256_check_desc.json
$B --concurrency 4,8    --max-tokens 256 --worker-log $S/worker_final.log --out $S/final_256_check_asc.json
```

Driver scripts: `scratchpad/final/run_final.zsh`, `final/extra.zsh`; tables: `final/tables.py`,
`final/pooled.py`. Worker defaults in force: QWEN_FP16=1, QWEN_MAX_BATCH=16, QWEN_MRQMM=1,
QWEN_PFQMM=1, QWEN_GDN_FUSE=1, QWEN_SDPA_FUSED (on: "prefill slab 2048 at any context"),
QWEN_BATCH_PREFILL_SHARE 0.5, prompt cache off, no DFlash, KV budget 15.75 GiB.

Bench fix made during this run: the gateway now sends the worker `chatcmpl-<request_id>-<12 hex>`,
so `bench_concurrency.py --worker-log` joined nothing (no peak / queue_ms / batch_mean in the
JSONs of final_256 / final_1024). `parse_worker_log` now strips that suffix (scripts/
bench_concurrency.py); the tables here join the worker logs post hoc the same way.

## Machine state

- Probe (64-token single stream, after 30 s idle) across the session: 38.0-38.6 tok/s before
  every level except 29.6 at 11:31 (b1024 rep 0 c=1) and the long-context probes (36.7 at 8K,
  33.9 at 32K context, expected). Probe TTFT 1.35-1.44 s. Serialized runs: decode 37.4-37.6
  tok/s per request, ±0.5% between reps; no drift like the 2026-10-11 baseline.
- **Slow window ~11:24-11:37**: final_256 rep 1 c=8 / 4 / 2 (67.4 / 59.1 / 41.7 vs 110.4 /
  82.8 / 48.7) and final_1024 rep 0 c=1 (23.5 vs 34.2) ran slow: worker step rate at 8 rows
  18.7 vs 22-23 steps/s, at 1 row 21-27 vs 32-33 steps/s, prefill slower too (queue spacing
  ~2.3 vs 1.2 s per admitted prompt), while the 64-token probes mostly still read 38. Not
  order-dependent: the same levels rerun 12:24-12:30 in descending (16,8,4) and ascending (4,8)
  order gave 110.7 / 110.4 and 83.2 / 82.8 tok/s, i.e. rep 0's numbers exactly. No foreign
  requests in the worker log; WindowServer at 40-50% CPU throughout (display kept awake by
  caffeinate); no thermal warnings in `pmset -g therm` (it never reports any here); no root for
  powermetrics. Most likely thermal/power-state. Those runs are tabulated separately and kept
  out of the headline.
- `ioreg` GPU utilization during cooldowns was 0% (nothing else on the GPU; sampled 11:27-11:34,
  `final_gpu_util.log`).

## Tables (pooled: percentiles over all requests of the selected runs)

**Batched (final defaults, QWEN_MAX_BATCH=16), 256 tokens, normal machine state** — final_256 rep 0 (all) + rep 1 c=16/c=1 + the two check runs:

| conc | runs | agg tok/s (runs) | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 31.2 (31.8, 30.5) | 1.25 | 1.58 | 8.13 | 8.50 | 37.3 | 1.00 | 6 | 10.43 |
| 2 | 1 | 48.7 (48.7) | 2.41 | 2.60 | 10.41 | 11.22 | 30.5 | 1.99 | 181 | 10.50 |
| 4 | 3 | 82.8 (82.6, 83.2, 82.8) | 2.97 | 4.89 | 12.19 | 15.72 | 28.7 | 3.94 | 938 | 10.69 |
| 8 | 3 | 110.4 (110.3, 110.7, 110.4) | 5.32 | 10.34 | 17.68 | 24.64 | 21.2 | 7.53 | 3102 | 11.02 |
| 16 | 3 | 115.9 (116.1, 114.7, 115.9) | 10.36 | 23.88 | 32.33 | 48.08 | 12.3 | 12.71 | 8100 | 11.69 |

**Serialized reference, final code (QWEN_MAX_BATCH=1), 256 tokens** (final_serial_256, 2 reps):

| conc | runs | agg tok/s (runs) | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 31.9 (31.8, 31.9) | 1.22 | 1.25 | 8.03 | 8.07 | 37.6 | 1.00 | 5 | 10.41 |
| 2 | 2 | 31.6 (31.5, 31.7) | 9.28 | 9.50 | 16.10 | 16.44 | 37.4 | 1.00 | 8 | 10.42 |
| 4 | 2 | 31.4 (31.8, 31.0) | 25.37 | 26.45 | 32.18 | 33.30 | 37.5 | 1.00 | 8 | 10.43 |
| 8 | 2 | 31.8 (31.8, 31.8) | 57.52 | 57.67 | 64.33 | 64.52 | 37.5 | 1.00 | 8 | 10.42 |

**Old serialized baseline (f2bcbdf, mlx 0.31.2 / mlx-vlm 0.6.2), 256 tokens** (baseline_256_nice0, 2 reps; percentiles recomputed over all requests, so they differ slightly from baseline.md's per-rep averages):

| conc | runs | agg tok/s (runs) | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 22.5 (23.3, 21.6) | 2.39 | 3.09 | 11.34 | 12.58 | 28.6 | - | - | 13.82 |
| 2 | 2 | 23.0 (23.5, 22.5) | 13.10 | 14.77 | 21.66 | 23.63 | 29.0 | - | - | 13.83 |
| 4 | 2 | 22.1 (22.9, 21.3) | 34.95 | 39.62 | 43.64 | 49.65 | 27.9 | - | - | 13.83 |
| 8 | 2 | 21.5 (22.2, 20.8) | 80.26 | 88.95 | 89.24 | 98.69 | 27.6 | - | - | 13.83 |

**Batched, slow window 11:24-11:29** (final_256 rep 1 c=8/4/2; see Machine state):

| conc | runs | agg tok/s (runs) | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 41.7 (41.7) | 2.91 | 3.16 | 12.15 | 13.62 | 26.4 | 1.99 | 198 | 10.52 |
| 4 | 1 | 59.1 (59.1) | 5.31 | 7.61 | 17.10 | 21.84 | 21.5 | 3.93 | 1963 | 10.68 |
| 8 | 1 | 67.4 (67.4) | 8.74 | 18.86 | 29.00 | 40.55 | 13.7 | 7.43 | 5863 | 11.02 |

### 1024 tokens

**Batched, 1024 tokens** (final_1024; c=1 rep 1 only — rep 0 fell in the slow window):

| conc | runs | agg tok/s (runs) | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 34.2 (34.2) | 1.69 | 1.85 | 29.86 | 30.86 | 36.2 | 1.00 | 6 | 10.44 |
| 8 | 2 | 141.9 (134.7, 149.1) | 6.95 | 15.81 | 57.02 | 67.42 | 21.0 | 7.86 | 4786 | 11.22 |
| 16 | 2 | 169.5 (162.8, 176.2) | 12.94 | 34.58 | 94.15 | 114.66 | 13.1 | 15.14 | 11702 | 12.05 |

c=1 rep 0 (slow window, 11:31-11:37):

| conc | runs | agg tok/s (runs) | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 23.5 (23.5) | 2.53 | 3.30 | 42.85 | 48.72 | 25.1 | 1.00 | 26 | 10.40 |

**Old serialized baseline, 1024 tokens** (baseline_1024, 1 rep):

| conc | runs | agg tok/s (runs) | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 25.0 (25.0) | 2.31 | 2.94 | 40.84 | 46.79 | 26.7 | - | - | 13.80 |
| 8 | 1 | 22.5 (22.5) | 314.56 | 332.77 | 358.92 | 375.90 | 24.2 | - | - | 13.81 |

### Ablation: mrqmm off

**QWEN_MRQMM=0**, final defaults otherwise (final_mrqmm0_256, 1 rep):

| conc | runs | agg tok/s (runs) | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 1 | 64.6 (64.6) | 4.45 | 10.54 | 30.88 | 38.04 | 9.9 | 7.75 | 2762 | 11.03 |
| 16 | 1 | 87.5 (87.5) | 11.27 | 23.41 | 43.55 | 58.53 | 7.9 | 14.12 | 8440 | 11.65 |


Same levels with mrqmm on (normal state): c=8 110.4 tok/s, decode 21.2 tok/s/req; c=16 115.9, 12.3.

## Raw per-run tables (`final/tables.py`; batch_mean / queue_ms / peak joined from the worker logs)

### final_256.json  (run_tag ca4977, max_tokens 256, prompt-tokens 600)

| rep | conc | reqs | prompt tok | wall s | agg tok/s | probe tok/s | agg/probe | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 8 | 609 | 64.4 | 31.8 | 38.6 | 0.82 | 1.22 | 1.23 | 8.05 | 8.06 | 37.5 | 1.00 | 6 | 10.40 | 0 |
| 0 | 2 | 8 | 622 | 42.1 | 48.7 | 38.5 | 1.26 | 2.41 | 2.60 | 10.41 | 11.22 | 30.5 | 1.99 | 181 | 10.50 | 0 |
| 0 | 4 | 8 | 620 | 24.8 | 82.6 | 38.4 | 2.15 | 3.03 | 4.84 | 12.21 | 15.31 | 28.6 | 3.94 | 951 | 10.67 | 0 |
| 0 | 8 | 16 | 620 | 37.1 | 110.3 | 38.5 | 2.87 | 5.39 | 10.22 | 17.75 | 23.66 | 21.1 | 7.53 | 3110 | 11.01 | 0 |
| 0 | 16 | 32 | 633 | 70.6 | 116.1 | 38.4 | 3.02 | 10.25 | 22.80 | 32.18 | 47.40 | 12.3 | 12.74 | 8036 | 11.69 | 0 |
| 1 | 16 | 32 | 636 | 71.4 | 114.7 | 38.4 | 2.99 | 10.61 | 22.82 | 32.87 | 47.15 | 12.2 | 12.64 | 8189 | 11.66 | 0 |
| 1 | 8 | 16 | 634 | 60.8 | 67.4 | 38.0 | 1.78 | 8.74 | 18.86 | 29.00 | 40.55 | 13.7 | 7.43 | 5863 | 11.02 | 0 |
| 1 | 4 | 8 | 632 | 34.6 | 59.1 | 38.5 | 1.54 | 5.31 | 7.61 | 17.10 | 21.84 | 21.5 | 3.93 | 1963 | 10.68 | 0 |
| 1 | 2 | 8 | 632 | 49.1 | 41.7 | 38.5 | 1.09 | 2.91 | 3.16 | 12.15 | 13.62 | 26.4 | 1.99 | 198 | 10.52 | 0 |
| 1 | 1 | 8 | 633 | 67.1 | 30.5 | 38.4 | 0.79 | 1.51 | 1.60 | 8.39 | 8.55 | 37.1 | 1.00 | 6 | 10.43 | 0 |

Pooled over repetitions (percentiles over all requests of the level):

| conc | agg tok/s mean (min-max) | agg/probe | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 31.2 (30.5-31.8) | 0.81 | 1.25 | 1.58 | 8.13 | 8.50 | 37.3 | 1.00 | 6 | 10.43 |
| 2 | 45.2 (41.7-48.7) | 1.17 | 2.44 | 3.12 | 11.18 | 13.06 | 28.4 | 1.99 | 189 | 10.52 |
| 4 | 70.9 (59.1-82.6) | 1.84 | 3.51 | 7.50 | 14.55 | 20.60 | 25.1 | 3.94 | 1457 | 10.68 |
| 8 | 88.9 (67.4-110.3) | 2.32 | 6.92 | 17.33 | 22.16 | 38.26 | 17.4 | 7.48 | 4487 | 11.02 |
| 16 | 115.4 (114.7-116.1) | 3.00 | 10.34 | 23.61 | 32.33 | 48.01 | 12.3 | 12.69 | 8113 | 11.69 |

### final_1024.json  (run_tag f20a01, max_tokens 1024, prompt-tokens 600)

| rep | conc | reqs | prompt tok | wall s | agg tok/s | probe tok/s | agg/probe | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 8 | 608 | 349.0 | 23.5 | 29.6 | 0.79 | 2.53 | 3.30 | 42.85 | 48.72 | 25.1 | 1.00 | 26 | 10.40 | 0 |
| 0 | 8 | 16 | 620 | 121.6 | 134.7 | 38.2 | 3.52 | 7.28 | 16.66 | 59.54 | 69.32 | 20.0 | 7.86 | 5112 | 11.17 | 0 |
| 0 | 16 | 32 | 629 | 201.3 | 162.8 | 38.3 | 4.25 | 13.79 | 34.75 | 95.92 | 117.18 | 12.6 | 15.14 | 12046 | 12.05 | 0 |
| 1 | 16 | 32 | 632 | 185.9 | 176.2 | 38.3 | 4.60 | 12.46 | 33.30 | 88.13 | 110.96 | 13.7 | 15.15 | 11358 | 12.05 | 0 |
| 1 | 8 | 16 | 631 | 109.9 | 149.1 | 38.5 | 3.88 | 6.62 | 14.84 | 53.87 | 64.19 | 22.0 | 7.87 | 4460 | 11.22 | 0 |
| 1 | 1 | 8 | 634 | 239.9 | 34.2 | 38.4 | 0.89 | 1.69 | 1.85 | 29.86 | 30.86 | 36.2 | 1.00 | 6 | 10.44 | 0 |

Pooled over repetitions (percentiles over all requests of the level):

| conc | agg tok/s mean (min-max) | agg/probe | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 28.8 (23.5-34.2) | 0.84 | 2.17 | 2.96 | 35.58 | 46.61 | 30.6 | 1.00 | 16 | 10.44 |
| 8 | 141.9 (134.7-149.1) | 3.70 | 6.95 | 15.81 | 57.02 | 67.42 | 21.0 | 7.86 | 4786 | 11.22 |
| 16 | 169.5 (162.8-176.2) | 4.42 | 12.94 | 34.58 | 94.15 | 114.66 | 13.1 | 15.14 | 11702 | 12.05 |

### final_serial_256.json  (run_tag f5702c, max_tokens 256, prompt-tokens 600)

| rep | conc | reqs | prompt tok | wall s | agg tok/s | probe tok/s | agg/probe | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 8 | 608 | 64.3 | 31.8 | 38.6 | 0.82 | 1.22 | 1.24 | 8.03 | 8.07 | 37.6 | 1.00 | 5 | 10.40 | 0 |
| 0 | 2 | 8 | 621 | 65.0 | 31.5 | 38.6 | 0.82 | 9.31 | 9.51 | 16.16 | 16.45 | 37.3 | 1.00 | 8 | 10.42 | 0 |
| 0 | 4 | 8 | 621 | 64.4 | 31.8 | 38.4 | 0.83 | 25.36 | 25.38 | 32.18 | 32.21 | 37.6 | 1.00 | 8 | 10.43 | 0 |
| 0 | 8 | 16 | 620 | 128.8 | 31.8 | 38.4 | 0.83 | 57.51 | 57.61 | 64.32 | 64.45 | 37.6 | 1.00 | 8 | 10.42 | 0 |
| 1 | 8 | 16 | 622 | 129.0 | 31.8 | 38.5 | 0.82 | 57.55 | 57.70 | 64.37 | 64.55 | 37.5 | 1.00 | 8 | 10.42 | 0 |
| 1 | 4 | 8 | 620 | 66.1 | 31.0 | 38.5 | 0.81 | 26.01 | 26.46 | 32.84 | 33.31 | 37.4 | 1.00 | 8 | 10.42 | 0 |
| 1 | 2 | 8 | 620 | 64.6 | 31.7 | 38.4 | 0.82 | 9.28 | 9.48 | 16.08 | 16.29 | 37.5 | 1.00 | 8 | 10.42 | 0 |
| 1 | 1 | 8 | 621 | 64.2 | 31.9 | 38.5 | 0.83 | 1.22 | 1.24 | 8.03 | 8.04 | 37.6 | 1.00 | 6 | 10.41 | 0 |

Pooled over repetitions (percentiles over all requests of the level):

| conc | agg tok/s mean (min-max) | agg/probe | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 31.9 (31.8-31.9) | 0.83 | 1.22 | 1.25 | 8.03 | 8.07 | 37.6 | 1.00 | 5 | 10.41 |
| 2 | 31.6 (31.5-31.7) | 0.82 | 9.28 | 9.50 | 16.10 | 16.44 | 37.4 | 1.00 | 8 | 10.42 |
| 4 | 31.4 (31.0-31.8) | 0.82 | 25.37 | 26.45 | 32.18 | 33.30 | 37.5 | 1.00 | 8 | 10.43 |
| 8 | 31.8 (31.8-31.8) | 0.83 | 57.52 | 57.67 | 64.33 | 64.52 | 37.5 | 1.00 | 8 | 10.42 |

### final_mrqmm0_256.json  (run_tag 14a217, max_tokens 256, prompt-tokens 600)

| rep | conc | reqs | prompt tok | wall s | agg tok/s | probe tok/s | agg/probe | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 8 | 16 | 621 | 63.4 | 64.6 | 38.6 | 1.67 | 4.45 | 10.54 | 30.88 | 38.04 | 9.9 | 7.75 | 2762 | 11.03 | 0 |
| 0 | 16 | 32 | 611 | 93.6 | 87.5 | 38.5 | 2.27 | 11.27 | 23.41 | 43.55 | 58.53 | 7.9 | 14.12 | 8440 | 11.65 | 0 |

Pooled over repetitions (percentiles over all requests of the level):

| conc | agg tok/s mean (min-max) | agg/probe | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 64.6 (64.6-64.6) | 1.67 | 4.45 | 10.54 | 30.88 | 38.04 | 9.9 | 7.75 | 2762 | 11.03 |
| 16 | 87.5 (87.5-87.5) | 2.27 | 11.27 | 23.41 | 43.55 | 58.53 | 7.9 | 14.12 | 8440 | 11.65 |

### final_256_check_desc.json  (run_tag 387919, max_tokens 256, prompt-tokens 600)

| rep | conc | reqs | prompt tok | wall s | agg tok/s | probe tok/s | agg/probe | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 16 | 32 | 622 | 70.7 | 115.9 | 38.5 | 3.01 | 10.40 | 22.78 | 32.34 | 46.92 | 12.3 | 12.74 | 8074 | 11.64 | 0 |
| 0 | 8 | 16 | 626 | 37.0 | 110.7 | 38.4 | 2.88 | 5.29 | 10.21 | 17.68 | 23.64 | 21.2 | 7.53 | 3087 | 11.02 | 0 |
| 0 | 4 | 8 | 609 | 24.6 | 83.2 | 38.4 | 2.17 | 2.96 | 4.74 | 12.13 | 15.17 | 28.7 | 3.94 | 911 | 10.69 | 0 |

Pooled over repetitions (percentiles over all requests of the level):

| conc | agg tok/s mean (min-max) | agg/probe | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 83.2 (83.2-83.2) | 2.17 | 2.96 | 4.74 | 12.13 | 15.17 | 28.7 | 3.94 | 911 | 10.69 |
| 8 | 110.7 (110.7-110.7) | 2.88 | 5.29 | 10.21 | 17.68 | 23.64 | 21.2 | 7.53 | 3087 | 11.02 |
| 16 | 115.9 (115.9-115.9) | 3.01 | 10.40 | 22.78 | 32.34 | 46.92 | 12.3 | 12.74 | 8074 | 11.64 |

### final_256_check_asc.json  (run_tag 6785ce, max_tokens 256, prompt-tokens 600)

| rep | conc | reqs | prompt tok | wall s | agg tok/s | probe tok/s | agg/probe | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 4 | 8 | 607 | 24.7 | 82.8 | 38.5 | 2.15 | 3.03 | 4.84 | 12.20 | 15.29 | 28.7 | 3.94 | 952 | 10.65 | 0 |
| 0 | 8 | 16 | 620 | 37.1 | 110.4 | 38.4 | 2.87 | 5.42 | 10.16 | 17.72 | 23.60 | 21.1 | 7.52 | 3109 | 11.00 | 0 |

Pooled over repetitions (percentiles over all requests of the level):

| conc | agg tok/s mean (min-max) | agg/probe | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 82.8 (82.8-82.8) | 2.15 | 3.03 | 4.84 | 12.20 | 15.29 | 28.7 | 3.94 | 952 | 10.65 |
| 8 | 110.4 (110.4-110.4) | 2.87 | 5.42 | 10.16 | 17.72 | 23.60 | 21.1 | 7.52 | 3109 | 11.00 |

### final_prefill_8000.json  (run_tag 5b572d, max_tokens 128, prompt-tokens 8000)

| rep | conc | reqs | prompt tok | wall s | agg tok/s | probe tok/s | agg/probe | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 2 | 8154 | 41.4 | 6.2 | 36.7 | 0.17 | 17.18 | 17.39 | 20.68 | 20.89 | 36.6 | 1.00 | 8 | 11.81 | 0 |

Pooled over repetitions (percentiles over all requests of the level):

| conc | agg tok/s mean (min-max) | agg/probe | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6.2 (6.2-6.2) | 0.17 | 17.18 | 17.39 | 20.68 | 20.89 | 36.6 | 1.00 | 8 | 11.81 |

prefill requests: [(8160, 17.42, 17362, 36.6, 11.81), (8149, 16.95, 16894, 36.6, 11.8)]

### final_prefill_32000.json  (run_tag a166b0, max_tokens 128, prompt-tokens 32000)

| rep | conc | reqs | prompt tok | wall s | agg tok/s | probe tok/s | agg/probe | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 2 | 32732 | 152.7 | 1.7 | 33.9 | 0.05 | 72.48 | 74.48 | 76.35 | 78.37 | 33.1 | 1.00 | 27 | 14.36 | 0 |

Pooled over repetitions (percentiles over all requests of the level):

| conc | agg tok/s mean (min-max) | agg/probe | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.7 (1.7-1.7) | 0.05 | 72.48 | 74.48 | 76.35 | 78.37 | 33.1 | 1.00 | 27 | 14.36 |

prefill requests: [(32789, 70.26, 70151, 33.3, 14.26), (32676, 74.7, 74596, 32.9, 14.36)]

## Files (scratchpad/results)

- JSON + console: `final_256`, `final_1024`, `final_serial_256`, `final_mrqmm0_256`,
  `final_prefill_8000`, `final_prefill_32000`, `final_256_check_desc`, `final_256_check_asc`
  (`.json` / `.txt`)
- logs: `worker_final.log` (defaults; still running), `worker_final_mrqmm0.log`,
  `worker_final_mb1.log`, `gateway_final.log` (:8090, still running), `final_status.txt`
  (driver timeline), `final_gpu_util.log`
