# What determinism costs: AppWorld replay at `QWEN_MAX_BATCH` 1 / 2 / 16, 2026-10-17

**Question.** The user wants reproducible AppWorld runs. The batched path is bit-exact only at B ≤ 2. At B ≥ 3 the
batch uses mrqmm and cascade, so a near-tie can flip a greedy token. What does sequential mode cost on this
prefill-bound load?

**Answer.**

| mode | requests/min | vs B=16 | reproducible |
|---|---|---|---|
| `QWEN_MAX_BATCH=16` | 21.4 / 21.7 (two runs) | – | no |
| `QWEN_MAX_BATCH=2` | **19.2** | **−11 %** | yes |
| `QWEN_MAX_BATCH=1` | 16.7 | −22 % | yes |

- **Identity.** The B=1 and B=2 runs gave byte-identical answers: 175/175 matched calls, every task identical, and
  every prompt length equal.
- **Two B=16 runs agree on only 77 % of their replies.** 5 of 9 tasks diverge, at a median call 12.
- **Latency.**
  - B=2: p50 22.4 s, p90 **36.0 s**.
  - B=16: p50 15.0–15.7 s, p90 38.0–41.9 s.
  - B=1: p50 25.7 s, p90 42.8 s.
- **Recommendation: `QWEN_MAX_BATCH=2` for reproducible runs.** It is the cheapest exact mode, and it keeps 89 %
  of B=16's throughput. B=1 buys nothing more: its answers are the same as B=2's.

## Method

- **Replay.**
  - `scripts/perf/agentic/replay.py --own-replies --ignore-eos off --episodes 8 --duration-min 10`, through our
    gateway on :8090 (thinking off).
  - It replays the recorded AppWorld calls (19 tasks, `docs/perf/2026-10-16/replay_bench.md`) with the model's
    own replies in the history.
  - New this time: every record carries `reply_sha` (sha1 of content + tool calls), so runs can be compared
    reply by reply.
- **Driver.** `scripts/perf/agentic/run_replay_batch.sh NAME MAX_BATCH`, once per run:
  1. a fresh worker under `sched_timer.py`;
  2. the **cool** probe (`bench_concurrency.py`: one cold ~8K prompt + 128 tokens);
  3. a fresh worker for the replay, with memsampler;
  4. the **warm** probe right after the replay, on that worker.
- **Worker env.** Each worker ran:

  ```
  QWEN_MODEL=.../ornith15-9b-mlx-8bit QWEN_KV_BITS=0 QWEN_PROMPT_CACHE=1 QWEN_PROMPT_CACHE_GB=6
  QWEN_BATCH_RESERVE_TOKENS=1024 QWEN_PRESENCE_PENALTY=0 QWEN_PROMPT_CACHE_REPLY=1 QWEN_MAX_BATCH=<B>
  ```

  `sched_timer` now also logs one `sched_slab:` line per prefill tick (input to `slabs.md`). The lines are
  log-only; the schedule is unchanged.
- **Code.** HEAD 778ef6c (worker.py md5 `0f42fb87…`).
  - The worker modules were frozen from HEAD into a scratch copy (`SCHED_WORKER_DIR`). The reason: another
    session's uncommitted `pfqmm.py` change (short-slab tiles) is in the working tree, and it must not leak into
    these numbers.
- **Run order.** b16a → b1 → b16b → b2, 06:17–07:10. GPU lock held; nice 0; one model process at a time.
- **Gateway check.** `qwen_inflight_capacity` was checked to equal the worker's max_batch before each replay:
  - at B=1, the gateway sends one generation at a time, and AppWorld's 8 processes wait in the gateway;
  - "wait before the worker" below = client latency − the worker's arrival→done.
- **Foreign requests:** 0. The only non-replay Generates in each worker log are our own warm probe.
- **Drift control.**
  - **b16a** and **b16b** (the same setting, 26 min apart): 21.4 vs 21.7/min.
  - **Warm probes** after every run: 19.4–19.8 s TTFT, 33.3–35.2 tok/s.
  - **Cool probes:** 17.7–17.9 s, except b16a's (13.8 s), which came after ~20 min of GPU idle.
  - So the four runs ran in the same (warm) machine state.

## Results (10-min window, 8 episodes)

`raw/replay_b1_compare.md` (`scripts/perf/agentic/batch_compare.py`):

| | b16a (B=16) | **b1 (B=1)** | b16b (B=16) | **b2 (B=2)** |
|---|---|---|---|---|
| requests/min (window) | 21.4 | **16.7** | 21.7 | **19.2** |
| requests in window / total / errors | 214 / 222 / 0 | 167 / 175 / 0 | 217 / 225 / 0 | 192 / 200 / 0 |
| latency mean / p50 / p90 s | 21.9 / 15.7 / 38.0 | 28.0 / 25.7 / 42.8 | 22.6 / 15.0 / 41.9 | 24.3 / 22.4 / 36.0 |
| wait before the worker (gateway) mean / p50 / p90 s | 0.0 / 0.0 / 0.0 | **24.4** / 22.4 / 38.0 | 0.0 / 0.0 / 0.0 | **18.1** / 16.7 / 27.2 |
| worker queue mean / p50 / p90 s | 7.9 / 1.4 / 19.9 | 0.0 / 0.0 / 0.0 | 8.8 / 1.3 / 23.0 | 0.4 / 0.0 / 0.2 |
| worker ttft mean / p50 / p90 s | 3.3 / 1.4 / 6.9 | 1.9 / 0.9 / 4.5 | 3.4 / 1.4 / 6.6 | 2.1 / 1.0 / 4.5 |
| worker decode mean / p50 / p90 s | 10.7 / 8.9 / 19.1 | **1.6** / 1.4 / 3.0 | 10.3 / 8.8 / 18.5 | 3.6 / 3.0 / 6.3 |
| scheduler prefill / decode / idle % | 79 / 16 / 5 | **50 / 45** / 5 | 81 / 15 / 4 | 54 / 42 / 5 |
| ms per prefill slab / per decode step | 987 / 46 | 889 / 30 | 993 / 45 | 842 / 40 |
| mean decode rows per step | 5.35 | 1.00 | 5.44 | 1.66 |
| prefill / decode tok/s (wall) | 290 / 19.0 | 188 / 15.2 | 253 / 19.0 | 203 / 17.5 |
| cache hit rate / mean reused | 94.6 % / 9.8K | 94.9 % / 8.4K | 94.7 % / 9.8K | 95.5 % / 9.2K |
| prompt mean (tokens) / mean call index | 10345 / 12.3 | 8759 / 9.5 | 10447 / 12.4 | 9533 / 10.8 |
| misses: total / of them capacity misses (not a task start), tokens | 12 / 3 (24K, 17K, 8K) | 9 / 0 | 12 / 3 (28K, 15K, 15K) | 9 / 0 |
| reply kept / entry gone | 220 / 2 | 175 / 0 | 224 / 1 | 200 / 0 |
| pcache / active GiB max, peak GB | 6.0 / 18.8, 22.1 | 6.0 / 15.4, 19.2 | 6.0 / 19.1, 22.1 | 6.0 / 16.0, 19.2 |
| probe 8K TTFT s / decode tok/s, cool → warm | 13.8 / 36.7 → 19.8 / 33.8 | 17.9 / 36.4 → 19.4 / 34.3 | 17.7 / 36.5 → 19.4 / 35.2 | 17.8 / 36.4 → 19.6 / 33.3 |

### Where the time goes

- **B=16:** the scheduler is ~80 % in prefill, as in every earlier replay. A decode step serves 5.4 rows on
  average, so a request's ~53 tokens cost about 0.45 s of GPU.
- **B=1:**
  - Each request's decode runs alone: 53 tokens × 30 ms = 1.6 s, against ~1.9 s of prefill.
  - So decode rises to **45 %** of the scheduler, and throughput drops by the decode it no longer shares.
  - Model: 600 s × 95 % busy / (1.9 + 1.6 s) ≈ 16.3/min (measured 16.7).
- **B=2:** half of that comes back.
  - Steps run at 1.66 rows on average: 28 ms at 1 row, 46 ms at 2.
  - Decode is 42 % of the scheduler; 19.2/min.
  - The B ≤ 2 path stays per-row qmv: bit-exact, but 2 rows cost ~1.6× one.
- **Latency moves from the worker's queue to the gateway.**
  - At B=1/2 the gateway's semaphore holds the other 7 processes, so worker queue time is ~0. TTFT is lower
    (0.9–1.0 s p50, no decode interleave) and decode is short (1.6 / 3.6 s).
  - The client still waits 24.4 / 18.1 s on average in front of the worker.
  - p50 rises (15–16 → 22–26 s), but B=2's **p90 is the best of all four runs** (36.0 s). Closed-loop FIFO with
    one or two in flight has no long decode tail. Under B=16, a 200-token reply decodes beside a stream of
    prefills (decode p90 19 s).
- **The comparison slightly favours B=1/2.** The heavier requests fell on B=16:
  - Its episodes got further into their tasks in 10 min (mean call 12.3 vs 9.5, prompt 10.3K vs 8.8K).
  - B=16 had 3 **capacity misses** per run: 15–28K-token conversations whose entry the 6 GB soft cap evicted by
    LRU at a store. They cost 40–60K extra prefill tokens, roughly 15–25 % of the window's prefill time at the in-run ~2.6 ms/token.
  - B=1/2 never reached that cache size in 10 min.
  - On matched calls that hit in both runs, the prefilled tokens are identical (510 vs 510 tok/call b16a/b1, 488
    vs 488 b16b/b2).
  - So the −11 % / −22 % are, if anything, a little generous to the exact modes.

  (Side finding: with 8 episodes past ~10K tokens each, 6 GB holds ~7–8 entries, and LRU starts to drop live
  conversations; `QWEN_PROMPT_CACHE_GB=8` has the room, peak 22.1 GB.)

### Reproducibility (`reply_sha` per task × call)

| pair | matched calls | same reply | same prompt length | tasks identical over all matched calls | first divergence (median call) |
|---|---|---|---|---|---|
| **b1 vs b2** | 175 | **175 (100 %)** | 175 | **9 / 9** | – |
| b16a vs b16b | 219 | 168 (77 %) | 175 | 5 / 9 | 12 |
| b16a vs b1 | 175 | 128 (73 %) | 133 | 5 / 9 | 5 |
| b16a vs b2 | 199 | 145 (73 %) | 150 | 5 / 9 | 5 |
| b1 vs b16b | 175 | 124 (71 %) | 131 | 5 / 9 | 5.5 |
| b16b vs b2 | 200 | 140 (70 %) | 147 | 5 / 9 | 5.5 |

- **B=1 and B=2 produced the same trajectories**, with different timing:
  - the prompt cache's state at each call differed between the runs;
  - B=2's rows overlapped.

  The B ≤ 2 kernels are bit-exact per row. With `QWEN_PROMPT_CACHE_REPLY=1` the resume points were the same in
  both runs here.
- **Caveat for the reply-KV mode.** A hit through the reply KV is not bit-identical to a cold prefill (the
  reply_kv gate: top-1 99.88 %). So a call whose reply entry is evicted (or not kept) in one run and kept in the
  other could still flip:
  - b1/b2 had 0 "entry gone" events; the B=16 runs had 1–2.
  - **Strict** reproducibility independent of cache state needs `QWEN_PROMPT_CACHE_REPLY=0` (hit == cold by
    `_bounds`).
  - With the cache on and 8 episodes, `QWEN_MAX_BATCH=2` + REPLY=1 was exactly reproducible over these
    175 calls.
- **At B=16 the two runs share 77 % of replies.** Once a reply flips, every later prompt of that task differs (own
  replies), so 4 of 9 tasks diverge. That is the expected mrqmm/cascade near-tie class, not an error.

## Recommendation

- **For reproducible AppWorld runs:** set `QWEN_MAX_BATCH=2`. Cost −11 % requests/min, p90 latency unchanged.
- **Do not use B=1:** −22 % for the same answers.
- **Fully cache-state-independent identity:** add `QWEN_PROMPT_CACHE_REPLY=0`. That costs reply-KV's −21 % of
  prefill tokens per call, ~6 % of throughput (reply_kv_gpu.md).
- **Nothing in the gateway needs to change.** In-flight follows the worker's max_batch (polled every 15 s), and
  the AppWorld processes simply wait there.

## Raw

| what | files |
|---|---|
| per run (`raw/replay_b1_<run>.*`, run = b16a, b1, b16b, b2) | `.json` (records incl. `reply_sha`, joined worker fields, summary), `.txt` (replay summary) |
| worker logs incl. `sched_slab:` lines | `raw/replay_b1_w_<run>.log` |
| memory series | `raw/replay_b1_mem_<run>.jsonl` |
| probes | `raw/replay_b1_probe_<run>_{cool,warm}.txt` |
| the table | `raw/replay_b1_compare.md` |

Tools:

- `scripts/perf/agentic/run_replay_batch.sh`: the driver;
- `scripts/perf/agentic/batch_compare.py`: the table and identity;
- `scripts/perf/agentic/sched_timer.py`: `sched_slab:` lines;
- `scripts/perf/agentic/replay.py`: `reply_sha`.
