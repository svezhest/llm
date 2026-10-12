# Prompt cache under continuous batching: 16 × 8K shared vs distinct prefixes

2026-10-12, 14:01–14:37, M1 Max 32 GB, Ornith-1.5 9B 8-bit, fp16 KV (`QWEN_KV_BITS=0`),
defaults otherwise (max_batch 16, fp16, pfqmm, mrqmm, GDN fuse, no DFlash). Through the
gateway (:8090), streaming, temperature 0, presence 0, thinking off, `ignore_eos`,
`max_tokens 256`, `--probe 64 --cooldown 30`. Driver `scripts/perf/final/run_pcache.zsh`,
tables `scripts/perf/final/pcache_tables.py`, raw `raw/pcache_*` (JSON/TXT per phase,
`pcache_status.txt` = phase timestamps + Health lines, `pcache_worker_cache_{on,off}.log`).

**Prompt shape** (`bench_concurrency.py --prefix`): the ~8K-token filler now sits in the
**system** message, the user message is only `Question #i: <task> …` (~40–50 tokens).
`shared` = one filler per level (one system prompt for all 16), `distinct` = one per
request. The system/user boundary is an `<|im_start|>` cut where the worker checkpoints.
Calibration lands at 8.0–8.35K prompt tokens (shared ran 8.2–8.35K, distinct 8.0–8.35K).

## Setup

- Cache on: `QWEN_PROMPT_CACHE=1` (3 GB). Worker log:
  `batching: max_batch 16, KV budget 12.75 GiB (cap 28.0 GiB, prefill reserve 2.4 GiB)` —
  15.75 GiB with the cache off, so the 3 GB come straight out of the batch budget. The
  gateway kept in-flight capacity 16 (no `max_batch changed` line; Health `ready=true`
  after each restart).
- Cache off (control B1): `KV budget 15.75 GiB`, same max_batch 16.
- Errors: none. 0 HTTP errors / 503s, 0 `ended at … tokens`, 0 refused Generates, all
  requests `finish=length` at 256 (128 in A5).

| phase | worker | args | local time | wall of the level |
|---|---|---|---|---|
| A1a | cache on | shared, c=16, 16 req | 14:02:11–14:03:53 | 37.5 s |
| A2a | cache on | distinct, c=16, 16 req | 14:03:53–14:10:15 | 314.9 s |
| A1b | cache on | shared, c=16, 16 req | 14:10:15–14:12:16 | 44.3 s |
| A2b | cache on | distinct, c=16, 16 req | 14:12:16–14:20:00 | 390.8 s |
| A3 | cache on | shared, c=1, 8 req | 14:20:00–14:23:00 | 93.5 s |
| A4 | cache on | distinct, c=1, 3 req | 14:23:00–14:25:44 | 80.0 s |
| A5 | cache on | shared, c=8, 8 req, 32K, max_tokens 128 | 14:25:44–14:30:52 | 90.4 s |
| B1 | cache **off** | shared, c=16, 16 req | 14:31:10–14:37:34 | 312.7 s |

(phase time includes calibration ~15–20 s / ~70 s at 32K, cooldown 30 s, probe.) A1/A2 were run
**interleaved** (A1a, A2a, A1b, A2b) instead of `--repeat 2` back to back, to spread drift.

## Summary

| phase | prompt | TTFT p50 s | TTFT p95 s | wall s | agg tok/s | decode tok/s/req | peak GB | hits | reused (mean) | probe decode / probe prefill |
|---|---|---|---|---|---|---|---|---|---|---|
| A1a shared c=16 | 8340 | **20.46** | **20.46** | 37.5 | **109.1** | 15.0 | 17.71 | 15/16 | 8192 | 36.6 / 572 tok/s |
| A1b shared c=16 | 8285 | 23.90 | 23.91 | 44.3 | 92.4 | 12.5 | 17.71 | 15/16 | 8192 | 35.7 / 407 tok/s |
| A2a distinct c=16 | 8207 | 150.18 | 289.94 | 314.9 | 13.0 | 5.1 | 16.74 | 0/16 | 0 | 36.3 / 578 tok/s |
| A2b distinct c=16 | 8257 | 178.29 | 360.14 | 390.8 | 10.5 | 4.3 | 16.80 | 0/16 | 0 | 36.7 / 460 tok/s |
| B1 shared c=16, **cache off** | 8277 | 154.79 | 290.71 | 312.7 | 13.1 | 5.5 | 13.51 | – | – | 36.8 / 490 tok/s |
| A3 shared c=1 | 8198 | 0.39 (hit) / 22.50 (cold) | | 93.5 | 21.9 | 30.1 | 14.79 | 7/8 | 8153 | 35.7 / 357 tok/s |
| A4 distinct c=1 | 8053 | 18.56 | 19.05 | 80.0 | 9.6 | 31.6 | 14.61 | 0/3 | 0 | 36.1 / 397 tok/s |
| A5 shared c=8, 32K | 32791 | 80.84 | 80.84 | 90.4 | 11.3 | 13.4 | 20.48 | 7/8 | 32746 | 32.8 / 396 tok/s |

"probe prefill" = the 64-token probe's prompt tokens / its TTFT (its own ~8K distinct
prompt; 34.9K for A5), i.e. the cold-prefill speed just before the level.

**Drift.** Decode was stable (probes 35.7–36.8 tok/s at 8K), but **prefill drifted by ~1.6×**:
probe prefill 572–578 tok/s in A1a/A2a, 357–460 tok/s from A1b on (thermal, most likely —
compute-bound prefill throttles, bandwidth-bound decode does not). A1b vs A1a (+3.4 s
TTFT) and A2b vs A2a (+24% wall) are that drift: the cold prefill in A1b took 23.8 s vs
14.3 s in A1a, and A3's cold request 22.4 s vs A4's 18–19 s (A3's probe was the slowest,
357 tok/s). Compare shared vs distinct within a pair (A1a/A2a are both fast-window,
A1b/A2b both slow-window); the effect is ~10× so drift does not change the conclusion.

## A1: one wave of 16 requests on one shared 8K system prompt

Per request in worker admission order (all 16 sent at t=0; `w queue` = waiting in the
worker before admission, `w ttft` = admission → first token):

#### A1a (prefix shared, c=16, filler 20162 chars)

wall 37.5s, agg 109.1 tok/s, TTFT p50 20.46 / p95 20.46 s, decode/req mean 15.0, probe 36.6 tok/s (ttft 15.46 s), peak max 17.71 GB, errors 0, hit rate 0.94, reused mean 7680

| # (worker admission order) | start s | prompt | TTFT s | w queue s | w ttft s | hit | reused | prefilled | decode tok/s | w tps | batch | peak GB | store MB | entries / total MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.00 | 8336 | 20.46 | 0.01 | 20.36 | miss | 0 | 8336 | 15.0 | 14.9 | 16.0 | 17.71 | 531 | 3 / 1750 |
| 2 | 0.00 | 8344 | 20.46 | 14.28 | 6.09 | HIT | 8192 | 152 | 15.0 | 14.9 | 16.0 | 17.71 | 531 | 4 / 2281 |
| 3 | 0.00 | 8342 | 20.46 | 14.67 | 5.70 | HIT | 8192 | 150 | 15.0 | 14.9 | 16.0 | 17.71 | 531 | 5 / 2812 |
| 4 | 0.00 | 8343 | 20.46 | 15.05 | 5.32 | HIT | 8192 | 151 | 15.0 | 14.9 | 16.0 | 17.71 | 531 | 5 / 2722 |
| 5 | 0.00 | 8341 | 20.46 | 15.43 | 4.94 | HIT | 8192 | 149 | 15.0 | 14.9 | 16.0 | 17.71 | 531 | 5 / 2654 |
| 6 | 0.00 | 8340 | 20.46 | 15.83 | 4.53 | HIT | 8192 | 148 | 15.0 | 14.9 | 16.0 | 17.71 | 531 | 6 / 3185 |
| 7 | 0.00 | 8340 | 20.46 | 16.23 | 4.14 | HIT | 8192 | 148 | 15.0 | 14.9 | 16.0 | 17.71 | 531 | 6 / 3185 |
| 8 | 0.00 | 8338 | 20.46 | 16.62 | 3.74 | HIT | 8192 | 146 | 15.0 | 14.9 | 16.0 | 17.71 | 531 | 6 / 3185 |
| 9 | 0.00 | 8334 | 20.46 | 17.01 | 3.34 | HIT | 8192 | 142 | 15.0 | 14.9 | 16.0 | 17.71 | 531 | 6 / 3184 |
| 10 | 0.00 | 8342 | 20.46 | 17.39 | 2.97 | HIT | 8192 | 150 | 15.0 | 14.9 | 16.0 | 17.71 | 531 | 6 / 3184 |
| 11 | 0.00 | 8341 | 20.46 | 17.77 | 2.58 | HIT | 8192 | 149 | 15.0 | 14.9 | 16.0 | 17.71 | 531 | 6 / 3185 |
| 12 | 0.01 | 8344 | 20.46 | 18.15 | 2.20 | HIT | 8192 | 152 | 15.0 | 14.9 | 16.0 | 17.71 | 531 | 6 / 3185 |
| 13 | 0.00 | 8336 | 20.46 | 18.52 | 1.82 | HIT | 8192 | 144 | 15.0 | 14.9 | 16.0 | 17.71 | 531 | 6 / 3185 |
| 14 | 0.01 | 8339 | 20.46 | 18.90 | 1.44 | HIT | 8192 | 147 | 15.0 | 14.9 | 16.0 | 17.71 | 531 | 6 / 3185 |
| 15 | 0.00 | 8342 | 20.46 | 19.30 | 1.04 | HIT | 8192 | 150 | 15.0 | 14.9 | 16.0 | 17.71 | 531 | 6 / 3185 |
| 16 | 0.00 | 8342 | 20.46 | 19.69 | 0.65 | HIT | 8192 | 150 | 15.0 | 14.9 | 16.0 | 17.71 | 531 | 6 / 3185 |

#### A1b (prefix shared, c=16, filler 19925 chars)

wall 44.3s, agg 92.4 tok/s, TTFT p50 23.90 / p95 23.91 s, decode/req mean 12.5, probe 35.7 tok/s (ttft 21.45 s), peak max 17.71 GB, errors 0, hit rate 0.94, reused mean 7680

| # (worker admission order) | start s | prompt | TTFT s | w queue s | w ttft s | hit | reused | prefilled | decode tok/s | w tps | batch | peak GB | store MB | entries / total MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.00 | 8288 | 23.91 | 0.01 | 23.79 | miss | 0 | 8288 | 12.5 | 12.5 | 16.0 | 17.71 | 529 | 5 / 2803 |
| 2 | 0.00 | 8285 | 23.91 | 19.19 | 4.61 | HIT | 8192 | 93 | 12.5 | 12.5 | 16.0 | 17.71 | 529 | 5 / 2804 |
| 3 | 0.00 | 8289 | 23.91 | 19.48 | 4.32 | HIT | 8192 | 97 | 12.5 | 12.5 | 16.0 | 17.71 | 529 | 5 / 2806 |
| 4 | 0.00 | 8287 | 23.91 | 19.79 | 4.01 | HIT | 8192 | 95 | 12.5 | 12.5 | 16.0 | 17.71 | 529 | 5 / 2712 |
| 5 | 0.00 | 8283 | 23.91 | 20.08 | 3.73 | HIT | 8192 | 91 | 12.5 | 12.5 | 16.0 | 17.71 | 529 | 5 / 2645 |
| 6 | 0.00 | 8285 | 23.90 | 20.38 | 3.42 | HIT | 8192 | 93 | 12.5 | 12.5 | 16.0 | 17.71 | 529 | 6 / 3174 |
| 7 | 0.00 | 8287 | 23.90 | 20.68 | 3.12 | HIT | 8192 | 95 | 12.5 | 12.5 | 16.0 | 17.71 | 529 | 6 / 3174 |
| 8 | 0.00 | 8286 | 23.90 | 20.97 | 2.83 | HIT | 8192 | 94 | 12.5 | 12.5 | 16.0 | 17.71 | 529 | 6 / 3174 |
| 9 | 0.00 | 8281 | 23.91 | 21.26 | 2.54 | HIT | 8192 | 89 | 12.5 | 12.5 | 16.0 | 17.71 | 529 | 6 / 3174 |
| 10 | 0.00 | 8279 | 23.90 | 21.53 | 2.27 | HIT | 8192 | 87 | 12.5 | 12.5 | 16.0 | 17.71 | 529 | 6 / 3174 |
| 11 | 0.00 | 8281 | 23.90 | 21.80 | 1.99 | HIT | 8192 | 89 | 12.5 | 12.5 | 16.0 | 17.71 | 529 | 6 / 3174 |
| 12 | 0.00 | 8287 | 23.90 | 22.09 | 1.70 | HIT | 8192 | 95 | 12.5 | 12.5 | 16.0 | 17.71 | 529 | 6 / 3174 |
| 13 | 0.00 | 8287 | 23.90 | 22.38 | 1.40 | HIT | 8192 | 95 | 12.5 | 12.5 | 16.0 | 17.71 | 529 | 6 / 3174 |
| 14 | 0.00 | 8289 | 23.90 | 22.67 | 1.11 | HIT | 8192 | 97 | 12.5 | 12.5 | 16.0 | 17.71 | 529 | 6 / 3174 |
| 15 | 0.00 | 8286 | 23.90 | 22.95 | 0.82 | HIT | 8192 | 94 | 12.5 | 12.5 | 16.0 | 17.71 | 529 | 6 / 3174 |
| 16 | 0.01 | 8284 | 23.90 | 23.25 | 0.53 | HIT | 8192 | 92 | 12.5 | 12.5 | 16.0 | 17.71 | 529 | 6 / 3174 |

**Shape.** Every request's first token arrives at the same moment (20.46 s in A1a, to
10 ms). Worker timeline of A1a (`raw/pcache_worker_cache_on.log`):

| t (s from arrival) | event |
|---|---|
| 0.00 | 16 Generates arrive; request #1 admitted, MISS (common prefix 40: only the calibration/probe entries) |
| 0.00–14.27 | cold prefill of 8336 tokens (584 tok/s), 4 GDN checkpoints, store 531 MB |
| 14.28–20.11 | 15 HITs, one after the other, **0.385 s each**: reuse 8192, prefill 142–152 tokens, store a 531 MB entry |
| 14.3–20.1 | `batch: rows=8 waiting=8 … steps/s=0.0` — no decode step while hits are admitted |
| ~20.4 | first batched decode step: all 16 rows emit their first token |
| 20.46–37.5 | 16 rows decode together: 255 tokens in 17.1 s = 15.0 tok/s/row, **239 tok/s** aggregate (67 ms step at B=16, ~8.3K context); A1b: 200 tok/s (80 ms) |

So a hit's TTFT in a wave = **queueing behind the cold prefill (14.3 s, 70%) + queueing
behind the other hits (≤5.8 s, 28%) + its own restore+prefill+store (0.385 s, 2%)**.
The cold request itself pays the 5.8 s too: 20.4 s instead of 14.3 s.

Why no decode between the hits: a hit's prefill (~150 tokens) is a single slab, and the
last slab of any prefill "joins at once" without `_slab_done`, so `resume` (decode's time
share) is never pushed forward; the next tick admits the next waiting request and
advances it immediately. Rows that already joined do not step until the waiting queue is
empty. Harmless here (5.8 s), but with a long queue of hits every running row's
inter-token gap grows by ~0.4 s per admitted hit and nobody's first token appears until
the queue drains. A fix (not done) would be to call `_slab_done` for the single-slab
prefill too, or emit the row's first token on join.

Why 8192 and not the system-block end (~8300): checkpoints exist only at slab ends, and
`_bounds` cuts at a message start only ≥ `QWEN_PREFILL_MIN_STEP` (256) past the previous
cut. The system block ended 93–150 tokens after the 8192 stride cut, so there was no cut at
the boundary and hits re-prefill ~100–150 tokens instead of ~45 (A3's prompts ended
the system block at 8153, ≥ 256 past 6144: cut there, 39–48 tokens prefilled). Cost
~0.1 s per hit; prompt-length dependent, not a bug.

## A2 / B1: 16 distinct 8K prompts (cache on) vs 16 shared 8K prompts with the cache off

#### A2a (prefix distinct, c=16, filler 17949 chars)

wall 314.9s, agg 13.0 tok/s, TTFT p50 150.18 / p95 289.94 s, decode/req mean 5.1, probe 36.3 tok/s (ttft 13.96 s), peak max 16.74 GB, errors 0, hit rate 0.00, reused mean 0

| # (worker admission order) | start s | prompt | TTFT s | w queue s | w ttft s | hit | reused | prefilled | decode tok/s | w tps | batch | peak GB | store MB | entries / total MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.00 | 8162 | 17.51 | 0.02 | 17.44 | miss | 0 | 8162 | 2.8 | 2.8 | 3.9 | 16.74 | 525 | 5 / 2818 |
| 2 | 0.00 | 8103 | 32.63 | 13.96 | 18.61 | miss | 0 | 8103 | 3.4 | 3.4 | 4.1 | 16.74 | 523 | 5 / 2810 |
| 3 | 0.00 | 8185 | 48.81 | 28.89 | 19.87 | miss | 0 | 8185 | 3.7 | 3.7 | 4.4 | 16.74 | 526 | 5 / 2805 |
| 4 | 0.00 | 8133 | 65.87 | 45.02 | 20.79 | miss | 0 | 8133 | 3.9 | 3.9 | 4.5 | 16.74 | 524 | 5 / 2620 |
| 5 | 0.00 | 8145 | 83.93 | 61.91 | 21.96 | miss | 0 | 8145 | 3.9 | 3.9 | 4.4 | 16.74 | 524 | 6 / 3144 |
| 6 | 0.00 | 8181 | 103.16 | 79.95 | 23.15 | miss | 0 | 8181 | 3.6 | 3.6 | 4.0 | 16.74 | 526 | 6 / 3148 |
| 7 | 0.00 | 8191 | 122.24 | 99.15 | 23.02 | miss | 0 | 8191 | 3.3 | 3.3 | 3.9 | 16.74 | 526 | 6 / 3149 |
| 8 | 0.00 | 8153 | 140.71 | 118.23 | 22.41 | miss | 0 | 8153 | 3.2 | 3.2 | 4.0 | 16.74 | 525 | 6 / 3150 |
| 9 | 0.00 | 8312 | 159.66 | 136.68 | 22.90 | miss | 0 | 8312 | 3.3 | 3.3 | 4.0 | 16.74 | 530 | 6 / 3154 |
| 10 | 0.00 | 8256 | 179.10 | 155.51 | 23.52 | miss | 0 | 8256 | 3.3 | 3.3 | 4.0 | 16.74 | 528 | 6 / 3159 |
| 11 | 0.00 | 8316 | 199.25 | 174.81 | 24.36 | miss | 0 | 8316 | 3.4 | 3.4 | 4.0 | 16.74 | 530 | 6 / 3164 |
| 12 | 0.00 | 8248 | 220.03 | 194.79 | 25.16 | miss | 0 | 8248 | 3.1 | 3.1 | 3.9 | 16.74 | 528 | 6 / 3166 |
| 13 | 0.00 | 8251 | 241.38 | 215.48 | 25.82 | miss | 0 | 8251 | 4.0 | 4.0 | 3.8 | 16.74 | 528 | 6 / 3168 |
| 14 | 0.00 | 8198 | 263.26 | 236.70 | 26.48 | miss | 0 | 8198 | 5.5 | 5.5 | 3.6 | 16.74 | 526 | 6 / 3170 |
| 15 | 0.00 | 8263 | 285.44 | 258.38 | 26.98 | miss | 0 | 8263 | 9.4 | 9.4 | 3.2 | 16.74 | 528 | 6 / 3168 |
| 16 | 0.00 | 8221 | 303.45 | 280.55 | 22.82 | miss | 0 | 8221 | 22.4 | 22.3 | 2.4 | 16.74 | 527 | 6 / 3167 |

#### A2b (prefix distinct, c=16, filler 18473 chars)

wall 390.8s, agg 10.5 tok/s, TTFT p50 178.29 / p95 360.14 s, decode/req mean 4.3, probe 36.7 tok/s (ttft 17.72 s), peak max 16.80 GB, errors 0, hit rate 0.00, reused mean 0

| # (worker admission order) | start s | prompt | TTFT s | w queue s | w ttft s | hit | reused | prefilled | decode tok/s | w tps | batch | peak GB | store MB | entries / total MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.00 | 8187 | 21.80 | 0.01 | 21.73 | miss | 0 | 8187 | 2.6 | 2.6 | 3.7 | 16.80 | 526 | 5 / 2757 |
| 2 | 0.00 | 8194 | 39.51 | 17.59 | 21.86 | miss | 0 | 8194 | 3.0 | 3.0 | 4.0 | 16.80 | 526 | 5 / 2754 |
| 3 | 0.00 | 8222 | 58.05 | 35.28 | 22.71 | miss | 0 | 8222 | 3.4 | 3.4 | 4.3 | 16.80 | 527 | 5 / 2752 |
| 4 | 0.00 | 8229 | 77.52 | 53.75 | 23.71 | miss | 0 | 8229 | 3.6 | 3.6 | 4.4 | 16.80 | 527 | 5 / 2631 |
| 5 | 0.00 | 8184 | 98.15 | 73.12 | 24.96 | miss | 0 | 8184 | 3.4 | 3.4 | 4.2 | 16.80 | 526 | 6 / 3156 |
| 6 | 0.00 | 8213 | 120.35 | 93.65 | 26.64 | miss | 0 | 8213 | 3.1 | 3.1 | 3.8 | 16.80 | 527 | 6 / 3158 |
| 7 | 0.00 | 8158 | 143.25 | 115.62 | 27.56 | miss | 0 | 8158 | 2.8 | 2.8 | 3.6 | 16.80 | 525 | 6 / 3157 |
| 8 | 0.00 | 8310 | 166.52 | 138.35 | 28.10 | miss | 0 | 8310 | 2.9 | 2.9 | 3.7 | 16.80 | 530 | 6 / 3161 |
| 9 | 0.00 | 8273 | 190.06 | 161.24 | 28.75 | miss | 0 | 8273 | 2.8 | 2.8 | 3.8 | 16.80 | 529 | 6 / 3163 |
| 10 | 0.00 | 8269 | 214.31 | 184.85 | 29.39 | miss | 0 | 8269 | 2.7 | 2.7 | 3.8 | 16.80 | 528 | 6 / 3164 |
| 11 | 0.00 | 8227 | 239.43 | 208.94 | 30.41 | miss | 0 | 8227 | 2.5 | 2.5 | 3.7 | 16.80 | 527 | 6 / 3166 |
| 12 | 0.00 | 8317 | 266.56 | 233.99 | 32.49 | miss | 0 | 8317 | 2.5 | 2.5 | 3.7 | 16.80 | 530 | 6 / 3169 |
| 13 | 0.00 | 8345 | 295.44 | 260.54 | 34.81 | miss | 0 | 8345 | 3.0 | 3.0 | 3.8 | 16.80 | 531 | 6 / 3175 |
| 14 | 0.00 | 8320 | 324.93 | 288.89 | 35.95 | miss | 0 | 8320 | 4.3 | 4.3 | 3.6 | 16.80 | 530 | 6 / 3175 |
| 15 | 0.00 | 8311 | 354.39 | 318.62 | 35.69 | miss | 0 | 8311 | 7.5 | 7.5 | 3.1 | 16.80 | 530 | 6 / 3177 |
| 16 | 0.00 | 8348 | 377.37 | 348.04 | 29.23 | miss | 0 | 8348 | 19.1 | 19.0 | 2.3 | 16.80 | 531 | 6 / 3179 |

#### B1 (prefix shared, c=16, filler 20526 chars)

wall 312.7s, agg 13.1 tok/s, TTFT p50 154.79 / p95 290.71 s, decode/req mean 5.5, probe 36.8 tok/s (ttft 17.98 s), peak max 13.51 GB, errors 0, hit rate -, reused mean -

| # (worker admission order) | start s | prompt | TTFT s | w queue s | w ttft s | hit | reused | prefilled | decode tok/s | w tps | batch | peak GB | store MB | entries / total MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.00 | 8273 | 20.06 | 0.01 | 19.98 | - | - | - | 2.8 | 2.8 | 3.9 | 13.51 | - | - |
| 2 | 0.00 | 8281 | 36.15 | 16.16 | 19.93 | - | - | - | 3.4 | 3.4 | 4.2 | 13.51 | - | - |
| 3 | 0.00 | 8279 | 52.66 | 32.33 | 20.26 | - | - | - | 3.7 | 3.7 | 4.4 | 13.51 | - | - |
| 4 | 0.00 | 8280 | 69.80 | 48.84 | 20.90 | - | - | - | 3.9 | 3.9 | 4.5 | 13.51 | - | - |
| 5 | 0.00 | 8278 | 87.79 | 65.95 | 21.77 | - | - | - | 3.8 | 3.8 | 4.4 | 13.51 | - | - |
| 6 | 0.00 | 8273 | 106.76 | 83.84 | 22.85 | - | - | - | 3.5 | 3.5 | 4.1 | 13.51 | - | - |
| 7 | 0.00 | 8271 | 126.21 | 102.76 | 23.39 | - | - | - | 3.3 | 3.3 | 3.8 | 13.51 | - | - |
| 8 | 0.00 | 8275 | 145.09 | 122.12 | 22.90 | - | - | - | 3.4 | 3.4 | 3.9 | 13.51 | - | - |
| 9 | 0.00 | 8279 | 164.50 | 140.93 | 23.50 | - | - | - | 3.4 | 3.4 | 4.0 | 13.51 | - | - |
| 10 | 0.00 | 8277 | 184.51 | 160.32 | 24.12 | - | - | - | 3.3 | 3.3 | 3.9 | 13.51 | - | - |
| 11 | 0.00 | 8277 | 204.71 | 180.16 | 24.47 | - | - | - | 3.3 | 3.3 | 3.9 | 13.51 | - | - |
| 12 | 0.00 | 8279 | 225.35 | 200.28 | 24.98 | - | - | - | 3.3 | 3.3 | 3.9 | 13.51 | - | - |
| 13 | 0.00 | 8278 | 245.77 | 220.92 | 24.77 | - | - | - | 4.3 | 4.3 | 3.9 | 13.51 | - | - |
| 14 | 0.00 | 8276 | 266.23 | 241.34 | 24.80 | - | - | - | 6.1 | 6.1 | 3.7 | 13.51 | - | - |
| 15 | 0.00 | 8281 | 286.67 | 261.77 | 24.82 | - | - | - | 10.6 | 10.6 | 3.2 | 13.51 | - | - |
| 16 | 0.00 | 8279 | 302.83 | 282.21 | 20.54 | - | - | - | 25.9 | 25.8 | 2.5 | 13.51 | - | - |
Prefill-bound: the 16 cold prefills run one at a time (~17–20 s each at 8.2K, rising to
~25–35 s per prompt by the end because decode rows take their share of the GPU), so TTFT
grows linearly with admission order and only ~4 rows decode at once (`batch` 3.6–4.5),
each at 3–5 tok/s (decode gets `1 - s` of the time while a prompt is waiting). **A2a ≈ B1
within 1%** (wall 314.9 vs 312.7 s, TTFT p50 150.2 vs 154.8 s, agg 13.0 vs 13.1 tok/s,
B1's probe prefill was ~15% slower, 490 vs 578 tok/s, so B1 may even be slightly flattered
by ordering — within the drift either way): with distinct prompts the
cache costs nothing measurable (checkpoint captures + a 0.53 GB store per request), and
B1 shows that the identical shared prompt without the cache is just as slow — the cache
is the only variable. A2b is the slow-window repeat (+24% wall).

## A3 / A4: sequential (c=1)

#### A3 (prefix shared, c=1, filler 19539 chars)

wall 93.5s, agg 21.9 tok/s, TTFT p50 0.39 / p95 14.79 s, decode/req mean 30.1, probe 35.7 tok/s (ttft 24.14 s), peak max 14.79 GB, errors 0, hit rate 0.88, reused mean 7134

| # (worker admission order) | start s | prompt | TTFT s | w queue s | w ttft s | hit | reused | prefilled | decode tok/s | w tps | batch | peak GB | store MB | entries / total MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.00 | 8202 | 22.50 | 0.01 | 22.44 | miss | 0 | 8202 | 29.3 | 29.1 | 1.0 | 14.79 | 526 | 5 / 2809 |
| 2 | 31.24 | 8200 | 0.47 | 0.01 | 0.40 | HIT | 8153 | 47 | 28.1 | 28.0 | 1.0 | 12.81 | 526 | 5 / 2805 |
| 3 | 40.82 | 8194 | 0.39 | 0.01 | 0.34 | HIT | 8153 | 41 | 30.6 | 30.4 | 1.0 | 12.58 | 526 | 5 / 2800 |
| 4 | 49.59 | 8196 | 0.39 | 0.01 | 0.33 | HIT | 8153 | 43 | 31.0 | 30.9 | 1.0 | 12.38 | 526 | 5 / 2696 |
| 5 | 58.23 | 8198 | 0.40 | 0.01 | 0.34 | HIT | 8153 | 45 | 30.9 | 30.8 | 1.0 | 12.08 | 526 | 5 / 2631 |
| 6 | 66.90 | 8201 | 0.37 | 0.01 | 0.32 | HIT | 8153 | 48 | 31.1 | 30.9 | 1.0 | 12.15 | 526 | 6 / 3157 |
| 7 | 75.52 | 8199 | 0.39 | 0.01 | 0.33 | HIT | 8153 | 46 | 30.9 | 30.8 | 1.0 | 12.15 | 526 | 6 / 3157 |
| 8 | 84.19 | 8192 | 0.39 | 0.01 | 0.33 | HIT | 8153 | 39 | 28.6 | 28.5 | 1.0 | 12.15 | 526 | 6 / 3157 |

#### A4 (prefix distinct, c=1, filler 18315 chars)

wall 80.0s, agg 9.6 tok/s, TTFT p50 18.56 / p95 19.05 s, decode/req mean 31.6, probe 36.1 tok/s (ttft 19.97 s), peak max 14.61 GB, errors 0, hit rate 0.00, reused mean 0

| # (worker admission order) | start s | prompt | TTFT s | w queue s | w ttft s | hit | reused | prefilled | decode tok/s | w tps | batch | peak GB | store MB | entries / total MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.00 | 8038 | 19.11 | 0.01 | 19.05 | miss | 0 | 8038 | 31.5 | 31.4 | 1.0 | 14.23 | 521 | 5 / 2742 |
| 2 | 27.22 | 8101 | 18.56 | 0.01 | 18.50 | miss | 0 | 8101 | 31.5 | 31.3 | 1.0 | 14.50 | 523 | 5 / 2738 |
| 3 | 53.92 | 8019 | 17.98 | 0.01 | 17.92 | miss | 0 | 8019 | 31.7 | 31.6 | 1.0 | 14.61 | 520 | 5 / 2733 |

A hit in isolation: **0.33–0.40 s TTFT (worker 0.32–0.40 s) vs 18–22 s cold — ×50–60**.
Worker log timestamps of a hit: Generate → `HIT` line 15 ms (render, tokenize, lookup;
the restore slices lazily), `HIT` → `store` ~265 ms (prefill of 39–48 tokens on an 8.2K
cache, which also materializes the restored copy, + the new entry's copy and eval),
then ~60 ms to the first token (first decode step). Decode after a hit is the same as
after a cold prefill: 28.0–31.1 tok/s (A3) vs 31.5–31.7 (A4); the hits' 28.0 / 28.6 outliers
are within the run's noise. A3's cold request (22.5 s) vs A4 (18–19 s) is the prefill drift
(probe 357 vs 397 tok/s), not the cache. Peak at c=1: 14.6–14.8 GB with a cold prefill
(cache entries + prefill transients), 12.1–12.8 GB for a hit.

## A5: 8 requests on one shared 32K prefix (max_tokens 128)

#### A5 (prefix shared, c=8, filler 82786 chars)

wall 90.4s, agg 11.3 tok/s, TTFT p50 80.84 / p95 80.84 s, decode/req mean 13.4, probe 32.8 tok/s (ttft 87.99 s), peak max 20.48 GB, errors 0, hit rate 0.88, reused mean 28653

| # (worker admission order) | start s | prompt | TTFT s | w queue s | w ttft s | hit | reused | prefilled | decode tok/s | w tps | batch | peak GB | store MB | entries / total MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.00 | 32785 | 80.83 | 0.03 | 80.69 | miss | 0 | 32785 | 13.4 | 13.3 | 8.0 | 20.48 | 1744 | 1 / 1744 |
| 2 | 0.00 | 32793 | 80.84 | 78.72 | 2.01 | HIT | 32746 | 47 | 13.4 | 13.3 | 8.0 | 20.48 | 1744 | 1 / 1744 |
| 3 | 0.00 | 32795 | 80.84 | 78.96 | 1.76 | HIT | 32746 | 49 | 13.4 | 13.3 | 8.0 | 20.48 | 1744 | 1 / 1744 |
| 4 | 0.00 | 32794 | 80.84 | 79.20 | 1.53 | HIT | 32746 | 48 | 13.4 | 13.3 | 8.0 | 20.48 | 1744 | 1 / 1744 |
| 5 | 0.00 | 32792 | 80.84 | 79.43 | 1.29 | HIT | 32746 | 46 | 13.4 | 13.3 | 8.0 | 20.48 | 1744 | 1 / 1744 |
| 6 | 0.00 | 32791 | 80.84 | 79.66 | 1.06 | HIT | 32746 | 45 | 13.4 | 13.3 | 8.0 | 20.48 | 1744 | 1 / 1744 |
| 7 | 0.00 | 32789 | 80.84 | 79.89 | 0.83 | HIT | 32746 | 43 | 13.4 | 13.3 | 8.0 | 20.48 | 1744 | 1 / 1744 |
| 8 | 0.00 | 32787 | 80.84 | 80.12 | 0.61 | HIT | 32746 | 41 | 13.4 | 13.3 | 8.0 | 20.48 | 1744 | 1 / 1744 |

Budget line said it fits (8 rows × (32.8K + 128) × 32 KiB ≈ 8.0 GiB + GDN state of
12.75 GiB): the batch log shows `rows=8 … reserved=9.27/12.75GiB active=18.96GiB`, no
admission waits beyond the cold prefill, no `ended at`. Same shape as A1: the cold 32.8K
prefill took 80.7 s (406 tok/s; the slow window — the earlier final bench measured 72.5 s),
then 7 hits at **0.21–0.25 s each** (reuse 32746, prefill 41–49 tokens: a hit at 32K costs
no more than at 8K), first tokens at 80.84 s for all 8. Decode: 8 rows × 13.4 tok/s =
107 tok/s aggregate at 32.8K context. Peak **20.48 GB** (weights 9.5 + 8 × 1.07 GB KV +
one 1.74 GB cache entry + transients). A 32K entry is 1.74 GB (1.07 GB KV + 12
checkpoints, the `QWEN_PROMPT_CACHE_CKPTS` cap), so the 3 GB cache holds **one** such
entry: every store evicted the previous one, and hits continued only because each new
entry carries the same prefix. Without the cache this level would be 8 × ~75 s of
prefill ≈ 10 min before the last first token, vs 81 s.

## Cache store sizes and evictions

An 8.2–8.3K entry is **521–531 MB**: 272 MB fp16 KV (8.3K × 32 KiB) + 4 GDN checkpoints
and the final recurrent state (~52 MB each). The 3 GiB (3221 MB) budget therefore holds **6
entries of 8K** (max total 3185 MB). Per run (stores include the calibration and probe
requests):

| run | stores | evictions | max total MB | entries at the end |
|---|---|---|---|---|
| A1a shared | 18 | 12 | 3185 | 6 |
| A2a distinct | 18 | 18 | 3170 | 6 |
| A1b shared | 18 | 18 | 3174 | 6 |
| A2b distinct | 18 | 18 | 3179 | 6 |
| A3 shared c=1 | 10 | 10 | 3157 | 6 |
| A4 distinct c=1 | 5 | 6 | 2756 | 5 |
| A5 shared 32K | 10 | 14 | 2928 | 1 |

So yes, **16 entries of 8K blow the 3 GB budget**: LRU evicts ~10 per wave. It costs
nothing in the shared case (hit rate 15/16 every time, the cold one is the only miss; the
LRU keeps the newest, and every new entry holds the shared prefix again), but the cache is
wasteful there: 6 sibling entries each store their own copy of the same 8192-token prefix
(6 × 0.53 GB for ~0.53 GB of distinct information). With distinct prompts every entry is
dead weight for this load (0/16 hits; it would only help a later turn of the same
conversation). Entries sharing a prefix with an existing one (a store with a `parent`)
could keep only the suffix, or the cache could skip storing a sibling whose own suffix is
~50 tokens — not done.

## Analysis

- **TTFT, 16 concurrent on one 8K prefix**: p50 **20.5 s (cache) vs 154.8 s (no cache)**,
  p95 **20.5 vs 290.7 s** (×7.6 / ×14); wall 37.5 vs 312.7 s (×8.3); aggregate 109 vs 13.1
  tok/s (×8.3). A2 (distinct, cache on) = B1 (shared, cache off) within 1%.
- **Where a hit's TTFT goes (in the wave)**: ~70% waiting for the one cold prefill, ~28%
  waiting for the other hits (0.385 s each, serialized, no decode in between), ~2% its own
  work. Restore itself is ~free (lazy slices, ≤ 15 ms to the HIT line); the measurable
  per-hit cost is the suffix prefill + storing the new entry, 0.27 s (45 tokens) –
  0.38 s (150 tokens) at 8K, 0.21–0.25 s at 32K.
- **Peak memory**: shared 17.71 GB (16 rows of 8.3K decoding at once + 3.2 GB cache) vs
  distinct 16.74–16.80 GB (only ~4 rows at once, + 3.2 GB cache) vs 13.51 GB with the cache
  off (B1, ~4 rows). The cache's 3 GB is visible one-for-one (16.7 → 13.5); the shared run
  is higher only because the cache lets all 16 rows reach decode together.
- **Decode tok/s**: not a like-for-like comparison — in the shared wave all 16 rows decode
  together (15.0 / 12.5 tok/s per row, 239 / 200 tok/s aggregate), in the distinct wave
  decode is starved by the serialized prefills (3–5 tok/s per row, ~4 rows at a time). At
  c=1, decode after a hit = after a cold prefill (≈ 30–31 tok/s).
- **Budget**: the 3 GB cache shrinks the batch KV budget 15.75 → 12.75 GiB; max_batch
  stays 16 and none of these loads came close (6.07 GiB reserved at 16 × 8.3K, 9.27 GiB at
  8 × 32.8K).
- **Caveats**: one wave per configuration per drift window (A1/A2 twice, interleaved;
  B1 once); prefill speed drifted ~1.6× during the session (decode did not); no other
  client used this worker during the runs (every Generate id in both worker logs is
  `bench-*`).
