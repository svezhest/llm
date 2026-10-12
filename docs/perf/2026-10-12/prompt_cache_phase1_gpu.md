# Prompt cache phase 1 on the GPU: identity, shared waves, pressure

2026-10-12, 15:07–15:49, M1 Max 32 GB, Ornith-1.5 9B 8-bit, fp16 KV (`QWEN_KV_BITS=0`),
defaults otherwise (max_batch 16, fp16, pfqmm, mrqmm, GDN fuse, no DFlash). Code under test:
the uncommitted phase-1 changes in `src/worker/worker.py`:

- an entry whose ids are a prefix of the new one is absorbed into it
- the last slab of a prefill is charged through `_slab_done`
- protected checkpoints
- the cache's live bytes count inside the batch budget, with on-demand LRU eviction
- `Usage.cached_tokens` and the Health counters

The gateway on :8090 was restarted on the rebuilt binary (pid 27313, same env:
`GATEWAY_ADDR=127.0.0.1:8090 RUST_LOG=info`). The baseline is `prompt_cache_bench.md`
(14:01–14:37, the same machine, old cache code).

The drivers are `scripts/perf/final/run_pcache2.zsh` and `run_pcache2_part2.zsh`. The
identity clients are `scripts/perf/final/pcache2_v1.py` and `pcache2_v1x.py`, and the
tables come from `pcache_tables.py` pointed at the `pcache2_` files. Raw data is in
`raw/pcache2_*`:

- JSON and TXT per phase
- `pcache2_status.txt`: phase timestamps, thermal state, Health counters after each phase
- `pcache2_worker_*.log`: one worker log per launch

| launch | worker env (+ `QWEN_KV_BITS=0`) | phases |
|---|---|---|
| R1 15:07 | `QWEN_PROMPT_CACHE=1` | V1 on |
| R2 15:09 | cache off | V1 off (cold references) |
| R3 15:12 | `QWEN_PROMPT_CACHE=1 QWEN_PROMPT_CACHE_GB=1` | V4 gb1 |
| R4 15:19 | `QWEN_PROMPT_CACHE=1` | V2 s1, V2 d (= V4 at the default 3 GB), V2 s2, V3, V4 32K |
| R5 15:38 | cache off | V1x off |
| R6 15:40 | `QWEN_PROMPT_CACHE=1 QWEN_PROMPT_CACHE_GB=8 QWEN_BATCH_MEM_GB=6` | V1x on, V4c |
| R7 15:49 | `QWEN_PROMPT_CACHE=1` | left running (log `raw/pcache2_worker_final.log`) |

The load line with the cache on now reads:

`batching: max_batch 16, KV budget 15.75 GiB (cap 28.0 GiB, prefill reserve 2.4 GiB), …, prompt cache 0.00 GiB live inside the budget (soft cap 3.0 GiB, evicted on demand)`

The budget is the same as with the cache off. The old code had 12.75 GiB, because it
subtracted the 3 GB up front.

**Errors:** none in any of the 7 worker logs:

- 0 `ended at … tokens`
- 0 `store failed`
- 0 "newest entry alone over QWEN_PROMPT_CACHE_GB"
- 0 Traceback / ERROR
- 0 `RESOURCE_EXHAUSTED`
- 0 HTTP errors in any bench phase

Every Generate id in the logs is `bench-*` or `v1-*`, so no other client used the worker.

## V1: hit == cold, byte-for-byte (26/26)

All prompts were greedy (temperature 0, presence 0), sequential (B=1), with no
`ignore_eos`, through the gateway. The cache-on results come from R1/R6. Each one was
compared with the same messages run once on a cache-off worker (R2/R5). The comparison
covers content, reasoning, finish, completion_tokens and prompt_tokens.

| case | prompt | `cached_tokens` (usage) | worker log | result |
|---|---|---|---|---|
| code0..4 cold (no system, 30–46 tok, max 128) | 30–46 | 0 | MISS → store, 0 ckpts, 53 MB | 5/5 identical |
| code0..4 again (hit at the entry end) | 30–46 | prompt−1 (45/37/33/38/29) | HIT, prefilling 1, store `absorbed=1` | 5/5 identical |
| sysA cold (7.5K system + task) | 7405 | 0 | MISS → store 7404 tok, 4 ckpts, 500 MB | identical |
| sysA again | 7405 | 7404 | HIT 7404, prefilling 1, `absorbed=1` | identical |
| sysB, sysC siblings (same system, other task) | 7396 / 7395 | 7359 / 7359 | HIT at the system-end cut, prefilling 37/36, `gap=5/3` | 2/2 identical |
| chatT turn 1/2/3 (thinking **on**, 2.3K system, max 1024, reasoning not sent back) | 2647 / 2684 / 2719 | 0 / 2625 / 2625 | MISS; HIT `gap=21 absorbed=1`; HIT `gap=58 absorbed=1` | 3/3 identical (finish stop) |
| chatN turn 1/2/3 (thinking off, max 128) | 2643 / 2791 / 2899 | 0 / 2625 / 2625 | MISS; HIT `gap=17 absorbed=1`; HIT `gap=165 absorbed=1` | 3/3 identical |
| **V1x** P2 cold (7.5K system + 5K user) | 12272 | 0 | MISS, 6 ckpts, 763 MB | identical |
| P3 (same system, other 5K user: **multi-slab suffix**, 3 slabs) | 12276 | 7359 | HIT 7359, prefilling 4917 | identical |
| P4 (shares the first ~5K of the system prompt, then diverges) | 7697 | 4096 | HIT at the 4096 stride checkpoint, prefilling 3601, `gap=990` | identical |
| sysA after P2 (shorter prompt, P2's system-end checkpoint) | 7405 | 7359 | HIT, prefilling 46 | identical (vs R2's sysA) |
| P2, P3 again (exact repeats) | 12272 / 12276 | 12271 / 12275 | HIT, prefilling 1, `absorbed=1` | 2/2 identical |

**Result: 26/26 byte-identical, with no divergence anywhere.** The cases above cover hits
at the entry end, at the system-block cut, at a 2048-stride checkpoint inside the system
prompt, a multi-slab prefill after a hit, both thinking modes, and a 3-turn chat.

The gateway's `usage.prompt_tokens_details.cached_tokens` equals the worker's
`reused`/`cached=` on every request. This also holds for all 128 bench requests of
V2–V4 (`w_cached == reused`).

**Absorb.** After each chat turn the turn-before entry was absorbed:

- the Health `entries` count stays at 9 through chatT 1→2→3, and at 10 through chatN 1→2→3
- the store lines show `absorbed=1`
- one entry is left per conversation

The previous turn's ids are an exact prefix of the next prompt. With the Ornith template
and the reasoning not sent back, the common prefix equals the old entry's length
(2646 / 2683 for chatT). Exact repeats also absorb their predecessor (`absorbed=1`, entry
count unchanged).

**Gap in chats.** Every turn resumed at **2625**, the end of the system block. It did not
resume at the previous turn's end. `gap` grows by the length of each turn: 21 → 58 tokens
for chatT, 17 → 165 for chatN.

The cause is that checkpoints exist only at slab ends. `_bounds` puts a message-start cut
only ≥ `QWEN_PREFILL_MIN_STEP` (256) past the previous cut, so short turns create no new
cut. The absorbed entry's end state is not a slab end of the new prompt, and is dropped as
documented.

This works as designed, and it is cheap at these sizes: turns 2/3 took 1.4–2.7 s in total
vs 6.4–7.9 s cold. Once a conversation has grown ≥ 256 tokens past the last cut, a cut
appears and the resume point moves forward. A chat of many short turns re-prefills up to
~256 tokens per turn.

## V2: 16 concurrent requests on one shared 8K system prompt (old → new)

The flags were `--prefix shared --concurrency 16 --requests-per-level 16 --max-tokens 256 --ignore-eos on --prompt-tokens 8000 --probe 64 --cooldown 30`.
The runs went shared → distinct → shared on one worker (R4).

| run | code | probe decode / probe TTFT (8.2K cold) | cold request's first token (worker prefill) | hits start → last first token | TTFT p50 / p95 s | wall s | agg tok/s | peak GB | entries / MB at the end |
|---|---|---|---|---|---|---|---|---|---|
| A1a | old | 36.6 / 15.46 s (fast window) | 20.46 (14.27) | 14.28 → 20.46 (all 16 at 20.46) | 20.46 / 20.46 | 37.5 | 109.1 | 17.71 | 6 / 3185 |
| A1b | old | 35.7 / 21.45 s (slow) | 23.91 (19.19) | 19.19 → 23.90 (all 16 at 23.90) | 23.90 / 23.91 | 44.3 | 92.4 | 17.71 | 6 / 3174 |
| **V2 s1** | new | 36.0 / 20.46 s (slow) | **18.95** (18.61) | 18.28 → 24.55 (first tokens 19.25, 19.29, … 24.55) | 21.23 / 24.11 | 43.1 | 95.0 | 18.27 | 9 / 3107 |
| **V2 s2** | new | 36.5 / 19.67 s (slow) | **18.48** (18.05) | 17.63 → 25.94 | 21.02 / 25.39 | 43.7 | 93.7 | 18.24 | 9 / 3120 |

Both new runs fell in the slow prefill window, with probe TTFT 19.7–20.5 s like A1b's
21.5 s. The like-for-like comparison is therefore **A1b → V2**.

- **The cold request's first token arrives when its own prefill ends:** 18.95 / 18.48 s,
  vs 23.91 s in A1b. In the old code it also waited for all 15 hits (+4.7 s in A1b,
  +6.2 s in A1a). In the fast window this would be ~14 s, which is A1a's prefill time.
- **Hits now release their first tokens one by one:** 19.25 → 24.55 s (s1) and
  18.52 → 25.94 s (s2). In the old code all 16 arrived together after the last hit. The
  last hit comes **+0.6 to +2.0 s later** than in A1b: the hit window is 6.3 / 8.3 s vs
  4.7 s (A1b) and 6.2 s (A1a). That window includes decode steps for the rows that have
  joined. The per-hit restore + prefill + store is unchanged at ~0.26–0.30 s from the HIT
  line to the store line. The average spacing grows from ~0.4 s at the start to ~0.7 s at
  the end, as more rows join: the hit time is charged, and decode takes its share of the
  time.
- **Decode during the hit queue is > 0.** The worker's 10 s batch lines show:

  ```
  old A1a 14:03:32  batch: rows=8 waiting=8 prefilling=- steps/s=0.0 mean_rows=0.00
  new s1  15:21:19  batch: rows=4 waiting=12 prefilling=- steps/s=0.3 mean_rows=2.00
  new s2  15:30:12  batch: rows=4 waiting=12 prefilling=- steps/s=0.4 mean_rows=2.25
  ```

  The next window, which contains the rest of the hits, shows 9.5 / 8.2 steps/s at
  mean_rows 14.2 / 13.5. The progressive first-token times above are the direct evidence.
- **TTFT p50** 21.0–21.2 s vs 23.9 s (A1b) and 20.5 s (A1a). **p95** 24.1–25.4 s vs
  23.9 s. **Wall and aggregate** are equal to A1b within noise: 43.1/43.7 vs 44.3 s, and
  95.0/93.7 vs 92.4 tok/s. The steady 16-row decode ran at 13.0–13.3 steps/s vs 12.3–14.7
  in the old runs.
- **Cache and peak.** At the end of the wave there were 9 entries (3.11 GB) instead of 6.
  Peak was 18.24–18.27 GB vs 17.71 GB, which is **+0.55 GB**.

  This is the accounting fix, not a leak. Siblings inherit their parent's 4 checkpoint
  states (the same objects). The old code summed every entry's `nbytes`, so it counted
  those states once per entry: its "3185 MB" was really 6 × (272 MB KV + 52 MB end
  state) + 4 × 52 MB ≈ 2.15 GB. The new `_PrefixCache.bytes` counts each array once, so
  the 3 GB soft cap now holds 9 entries, ≈ 3.1 GB real. Active memory at 16 rows was
  16.86 GiB vs 15.95 GiB, a difference of +0.9 GiB, which matches. Each sibling's own
  data is ~322 MB: a fresh KV copy of the whole 8.2K prefix plus its end state.
- Evictions (Health, at the 3 GB store-time cap): 9 in s1 (calibration and probe
  included), 15 in s2. The hit rate was 15/16 in both. Each hit reused 8192 tokens; the
  system block ended 68–121 tokens past the 8192 stride cut, so no boundary cut existed,
  as before.

### V2 distinct (drift control; also V4 at the default 3 GB)

| run | code | probe TTFT | first cold prefill (worker) | wall s | TTFT p50 / p95 s | agg tok/s | decode/req | batch mean | peak GB |
|---|---|---|---|---|---|---|---|---|---|
| A2a | old | 13.96 s | 17.44 | 314.9 | 150.2 / 289.9 | 13.0 | 5.1 | 2.4–4.5 | 16.74 |
| A2b | old | 17.72 s | 21.73 | 390.8 | 178.3 / 360.1 | 10.5 | 4.3 | 2.3–4.4 | 16.80 |
| V4 gb1 (1 GB cap) | new | 13.86 s | 17.92 | 347.0 | 163.5 / 320.1 | 11.8 | 4.8 | 2.3–4.4 | 14.69 |
| **V2 d** (3 GB) | new | 18.01 s | 21.46 | 341.9 | 173.9 / 318.6 | 12.0 | 5.2 | 2.4–4.3 | 16.81 |
| V4c (8 GB cap, 6 GiB budget) | new | 14.70 s | 17.99 | 318.3 | 149.2 / 294.1 | 12.9 | 5.3 | 2.4–4.5 | 17.81 |

For 16 distinct 8K prompts the new code is **within drift of the old**, with no consistent
sign:

- V2 d is −12.5% wall vs A2b, which matches its probe window and first prefill time
- V4 gb1 is +10% vs A2a, also a matching window
- V4c is +1% vs A2a

The `_slab_done` charge after the last slab lets decode run between cold prefills. That
could lengthen TTFT, but it does not show up beyond this noise. The same ~4-row
prefill-bound shape as before. 0/16 hits in each run.

## V3: c=1 sequential on a shared prefix (A3 equivalent)

| run | probe TTFT | cold TTFT | hit TTFT (client) | hit worker ttft | reused / prefilled | decode tok/s | peak GB (cold / hit) |
|---|---|---|---|---|---|---|---|
| A3 old | 24.14 s | 22.50 | 0.37–0.47 | 0.32–0.40 | 8153 / 39–48 | 28.0–31.1 | 14.79 / 12.1–12.8 |
| **V3 new** | 17.43 s | 16.16 | **0.26–0.34** | 0.21–0.28 | 8127 / 39–48 | 34.7–36.0 | 15.06 / 12.9–13.05 |

The hit cost is unchanged or slightly lower: 0.26–0.34 s vs 0.37–0.47 s. That
difference is within drift; the new run was in a faster window. Decode after a hit
equals the probe's decode. 7/8 requests hit.

## V4: cache pressure and on-demand eviction

- **Soft cap 1 GB, 16 distinct 8K requests (`V4 gb1`).** A 0.53 GB entry fits twice, so
  the cache held 2 entries (1.05 GB) throughout. There were 16 store-time LRU evictions,
  1 per store (Health `evictions=16`, `misses=18`, `hits=0`). The rows never pressed the
  15.75 GiB budget (~4 rows × 0.37 GiB), so there was no on-demand eviction. There were
  no `ended at`, no warnings and no thrash. Peak was 14.69 GB, vs 16.81 GB with the 3 GB
  cap (V2 d). Speed is as above.
- **Default 3 GB (`V2 d`).** 18 stores and 21 evictions at store (Health 9 → 30), 6
  entries / 3.17 GB at the end, and 0 on-demand evictions, because the budget was not the
  constraint.
- **On-demand eviction at admission (`V4c`: `QWEN_PROMPT_CACHE_GB=8`, `QWEN_BATCH_MEM_GB=6`).**
  The cache grew past what the 6 GiB budget leaves next to the rows. Admissions then
  evicted LRU entries: **9 log lines, 13 entries**, for example:

  ```
  prompt cache: 3 entries evicted to admit id=…-c16-5 (3.77 GiB left)
  prompt cache: 1 entries evicted to admit id=…-c16-15 (3.94 GiB left)
  ```

  All 13 evictions in Health were on demand; none came from the 8 GB soft cap. The batch
  ran 4–5 rows with `reserved` 1.9 GiB + `pcache` 3.4–3.9 GiB ≤ 6 GiB. There was one
  eviction per admission and no thrash: no entry was evicted and then needed again. There
  were no `ended at` lines and no OOM. Peak was 17.81 GB, and wall time was the same as
  A2a.

  After the wave the cache stood at 4.76 GB (9 entries). This is allowed: with no rows,
  only the soft cap applies, and an empty batch always admits.
- **Real budget pressure with a shared 32K prefix (`V4 32K`: c=16, 16 req, ~33.3K prompts,
  max_tokens 128, default worker).** The cold prefill took 74.7 s; the probe was 79.9 s at
  ~35K. Then:
  - Hits joined at ~0.3 s spacing up to 12 rows. Two admissions near the limit each
    evicted 1 entry (`1 entries evicted to admit … (1.64 GiB left)`, 15:37:47). The batch
    line then read `rows=12 waiting=4 reserved=14.11 pcache=1.64`, which sums to
    15.75 GiB, the budget exactly.
  - The last 4 requests waited ~13 s for rows to finish. They were then admitted as hits,
    and two of those admissions evicted 1 entry each (15:38:00–01). That makes 4 on-demand
    eviction lines in total.

  The entry they resume from was kept, as the `keep` rule says. They waited rather than
  losing the prefix. Hit rate was 15/16, with 33275 tokens reused per hit. Peak was
  **25.27 GB** (active 23.4 GiB at 12 rows × 33.3K + weights + 1.64 GiB cache), under the
  28 GiB cap. There were no `ended at` lines and no errors.

  TTFT p50 77.2 s / p95 92.7 s, wall 99.5 s. Without the cache this would be 16 × ~75 s
  of prefill.

  Churn: every hit stores a 1.76 GB sibling (its own 1.09 GB KV copy of the shared 33K
  prefix), which the next admission evicts again. It is harmless here but wasteful; this
  is the sibling-duplication issue from `prompt_cache_bench.md`.
- **Not exercised on the GPU:** the `_regrow` path, in which the cache gives way for a row
  that outgrows its reservation. That needs rows running past `QWEN_BATCH_RESERVE_TOKENS`
  (4096 generated tokens) under a full budget. Only the CPU tests cover it.

## Drift and caveats

- **Decode** was stable: probe 35.8–37.1 tok/s at 8K, and 32.8 at 33K.
- **Prefill** drifted ~1.45×. Probe TTFT at 8.2K was 13.9–14.7 s in R3/R6 and 17.4–20.5 s
  in R4. All new shared waves (V2 s1/s2) fell in the slow window, so they are compared
  with A1b (old, slow), not A1a.
- One wave per configuration and window, so the distinct-wave differences (±10%) are
  drift.
- Load average was 2.6–4.2 throughout (`pcache2_status.txt`, `pmset -g therm` empty = no
  thermal warning).
- The driver's "wait for the gateway's Health changed" step sometimes hit its 60 s timeout
  after a worker restart. The new gateway logs `worker ready` on its first poll. This cost
  wall-clock time only, not results.

## State left

- Worker on :50051: pid 31776, `QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 QWEN_PROMPT_CACHE=1`,
  log in the scratchpad `pcache2/worker_final.log`.
- Gateway on 127.0.0.1:8090: pid 27313, the new binary (`cached_tokens` in usage), log
  appended to the scratchpad `results/gateway_final.log`.
- Nothing else was touched (:8080/:8081 gateways, docker, aria2).
