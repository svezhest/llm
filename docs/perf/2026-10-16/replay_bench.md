# Agentic replay of the AppWorld load: memory knobs, batch size, prefill share

16.10.2026, 02:30–05:00, M1 Max 32 GB (MacBookPro18,2, `powermode 0`), Ornith-1.5 9B 8-bit, fp16 KV,
prompt cache on, max_batch 16. Gateway :8090 (thinking off). Measured on a GPU free of other jobs
(lock); other agents ran CPU-only pytest meanwhile (they share the SoC's power/thermal budget).

## Method

- `scripts/perf/agentic/replay.py`: N episodes in a closed loop. Episode i replays the recorded LM calls
  of AppWorld task i (`~/Projects/appworld/experiments/outputs/ornith9b_react_dev/tasks/*/logs/lm_calls.jsonl`,
  19 tasks, 507 calls, sorted by task id). Every call sends the **recorded** `input.messages`
  (non-streaming, temperature 0, seed 100, no presence_penalty, as AppWorld did), with
  `max_tokens` = the recorded completion length + `ignore_eos` (the decode load is reproduced
  exactly; 0 mismatches). It sleeps the recorded gap between calls (median 0.14 s). When a task is
  done, the episode takes the next task not yet started. The worker log is joined by request id.
- `scripts/perf/agentic/memsampler.py` (every 10 s): `vm.swapusage`, the worker's MEM/CMPRS/PAGEINS
  (`top`), `vm_stat` rates (pageins, swapins, decompressions), the biggest memory users.
- `scripts/perf/agentic/sched_timer.py` launches the worker unchanged, but wraps `_Scheduler._advance`
  (one prefill slab, synced) and `_iterate` (one decode step; it waits for the previous step), and logs
  the scheduler thread's time split every 10 s (`sched_time:` lines). This gives prefill vs decode
  GPU time. The batch log's `steps/s` cannot give it: 75–85 % of its 10 s ticks contain a prefill.
- `scripts/perf/agentic/worker_log.py` extends `analyze_worker_log.py`; it adds the time split,
  stores, events and ms/step by rows. `compare.py` prints the tables below.
- Every run gets a fresh worker. A probe (one cold ~8K prompt + 128 decode tokens through the
  gateway) runs on a fresh worker before the run (**cool**) and right after it (**warm**).
- Code: **R0** = HEAD 8a1625c `worker.py` (the user's run). **R0', R1, R2, R3** = the working tree
  frozen at 03:22, with the other agent's chain-checkpoint policy and the eviction order
  (stale → MRU, waiting entries spared). Its diff: `raw/replay_worker_wt0322_vs_head.diff`.
  `r0new` is an extra run of an intermediate working-tree version (02:35); it is not in the
  tables.
- Run windows are 15 min (R3: 10 min) with 8 episodes (R2: 16). In 15 min an episode set gets
  through 12–20 tasks, so each task runs once and the cache starts empty (the first call of each
  task is a miss).

Raw data: `raw/replay_<run>.{json,txt}`, `raw/replay_worker_<run>.log`, `raw/replay_mem_<run>.jsonl`,
`raw/replay_probe_<run>_{cool,warm}.txt`, `raw/replay_prefill_prof_432.json`, `raw/replay_driver.log`.

## Results (15 min, 8 episodes unless noted)

Knobs: R0/R0' `QWEN_PROMPT_CACHE_GB=12` (CKPTS 12, reserve 4096); R1 `GB=6 CKPTS=3 RESERVE_TOKENS=1024`;
R2 = R1 with 16 episodes and `GB=9`.

| | R0 (HEAD, 12 GB) | R0' (wt, 12 GB) | R1 (wt, 6 GB) | R2 (wt, 9 GB, 16 ep) |
|---|---|---|---|---|
| requests/min | 16.9 | 19.9 | 16.8 | 15.9 |
| latency p50 / p90 s | 25.9 / 46.4 | 21.7 / 41.4 | 25.6 / 46.0 | 55.1 / 96.0 |
| queue mean / p90 s | 6.7 / 16.6 | 6.5 / 17.3 | 7.6 / 20.5 | 36.3 / 69.1 |
| ttft mean / p90 s | 4.4 / 11.0 | 3.8 / 10.2 | 4.6 / 11.7 | 5.1 / 11.9 |
| decode mean / p90 s | 16.4 / 28.1 | 13.4 / 22.7 | 15.7 / 25.6 | 17.2 / 29.7 |
| cache hit rate / mean reused | 94.7 % / 9.9K | 94.8 % / 10.4K | 94.2 % / 9.8K | 93.3 % / 7.3K |
| misses (= task starts) / evictions to admit | 14 / 3 | 16 / 1 | 15 / 0 | 17 / 0 |
| prefill / decode tok/s wall | 242 / 16.5 | 291 / 19.5 | 242 / 16.4 | 253 / 15.5 |
| scheduler: prefill / decode / idle % | 76 / 20 / 4 | 78 / 18 / 4 | 79 / 18 / 4 | 80 / 17 / 3 |
| ms per prefill slab / decode step | 1283 / 66 | 1123 / 48 | 1335 / 55 | 1414 / 56 |
| decode step ms at 4 / 6 / 8 rows | 58 / 69 / 89 | 45 / 50 / 57 | 51 / 59 / 65 | 52 / 61 / 65 |
| in-run prefill ms per new token | 3.42 | 2.92 | 3.58 | 3.69 |
| store MB / checkpoints per entry | 922 / 9.8 | 659 / 4.2 | 560 / 2.8 | 479 / 2.8 |
| pcache / active GiB max, peak GB | 11.7 / 24.0, **29.3** | 11.4 / 24.1, 27.6 | 6.0 / 18.4, 22.9 | 9.0 / 22.3, 26.3 |
| pageins/s, decompr./s max (1-min means) | **6.2K, 97K** | 1.8K, 24K | 0.2K, 2.8K | 1.1K, 14K |
| worker CMPRS max GB | 0.69 | 0.61 | 0.62 | 0.68 |
| probe 8K TTFT s / decode tok/s: cool → warm | – → 23.6 / 25.6 | 13.6 / 36.6 → 20.9 / 24.2 | – → 25.6 / 24.3 | 13.8 / 36.6 → 24.4 / 24.4 |

R0 per minute: memory pressure begins after minute 9, when pcache passes ~10.8 GiB. The decode
step grows with it:

| min | 2 | 5 | 8 | 9 | 11 | 13 | 14 |
|---|---|---|---|---|---|---|---|
| pcache GiB | 5.5 | 8.3 | 10.0 | 10.8 | 11.7 | 11.6 | 11.6 |
| free MB | 1622 | 70 | 1484 | 611 | 15 | 239 | 20 |
| pageins/s | 43 | 290 | 403 | 1718 | 2811 | 4694 | 4672 |
| swapins/s | 20 | 29 | 67 | 754 | 2320 | 6360 | 6565 |
| decode step ms (5–6 rows) | 44 | 56 | 59 | 73 | 88 | 99 | 126 |

R1 has no such phase: pageins stay at or below 232/s, and the step is 55–67 ms from minute 4 to
minute 15.

### R3: `QWEN_BATCH_PREFILL_SHARE` (R1 knobs, 10 min each)

| | R1, share 0.5 (first 10 min) | share 0.3 | share 0.7 |
|---|---|---|---|
| requests/min | 18.5 | 19.5 | 20.0 |
| latency p50 / p90 s | 23.3 / 41.8 | 22.4 / 37.4 | 20.8 / 40.0 |
| queue / ttft / decode mean s | 6.8 / 4.1 / 14.4 | 8.4 / 3.9 / 11.8 | 4.2 / 3.6 / 15.4 |
| scheduler prefill / decode % | 77 / 18 | 75 / 19 | 79 / 16 |
| mean rows | 4.80 | 4.34 | 5.70 |
| probe warm TTFT s | 25.6 (after 15 min) | 20.9 | 20.6 |

The 0.3 and 0.7 runs ran back to back in the same machine state (cool and warm probes are equal).
Throughput does not change (19.5 vs 20.0/min). The share only moves latency between the queue and
decode. At 0.7 the queue is half as long and decode 30 % longer, and p50 is 7 % lower. The R1
column ran in a slower machine state, so compare it only with care.

### Thermal state (not memory) sets the pace

- Fresh worker, cool GPU: 8K cold TTFT 13.6–13.8 s, decode 36.6 tok/s. Right after any 10–15 min
  run: 20.6–25.6 s and 24–28 tok/s. That is ×1.5–1.85 slower, even after R1, which had no memory
  pressure.
- I re-probed the R2 worker after 4 min idle, with its cache still full (9 GiB pcache, peak 21.5 GB):
  13.5 / 14.9 s and 36.7 / 36.6 tok/s, back to the cool numbers. So the slowdown under sustained load
  comes from the GPU's power/thermal throttling, not from memory. The ~1.2× spread between runs
  (R0' and R3 faster than R0, R1 and R2; same code in R0' and R1) is this machine state drifting.
  Other agents' CPU tests running at the same time share the package's thermal budget. **Throughput
  differences below ~20 % between separate 15-min runs are noise here.**

## Handoff hypotheses

1. **"Decode at long context is twice as slow" (260 ms/step at B=8): rejected.** In the scheduler
   thread's own timing, the decode step at ~10K context is 45–65 ms at 4–8 rows (R0', R1, R2) and
   89 ms at 8 rows in R0 (its swap minutes). That compares with 31–52 ms at 1K context. The
   handoff's 3.8 steps/s came from 10 s ticks in which prefill took 75–85 % of the time.
   Decode is only **17–20 % of the scheduler's time**. Under memory pressure the step does grow:
   R0 went 56 → 99–126 ms in minutes 13–14.
2. **Prefill strictly one at a time: true, and it is the limiter.** Prefill is 75–80 % of the
   scheduler's time in every run, and the GPU is idle 3–6 %. Requests average ~860 new tokens
   (p50 430, p90 2.8K) at 10–12K context. Packed prefill of several short suffixes would barely
   raise throughput. In isolation the GEMMs of a 432-token slab already run at 7.5–8 TFLOPS, vs
   8.2–8.6 for a 2048 slab (below). It could only shorten queue waits.
3. **Prefill interleave stifles decode: true for latency, irrelevant for throughput.** With
   5 of 16 rows busy, share 0.5 gives prefill an effective 84 % (`1-(1-s)·B/max`). Moving the share
   from 0.3 to 0.7 leaves requests/min unchanged and only trades queue time against decode time
   (R3). 0.7 is slightly better end to end (p50 −7 %, queue halved). This is not a lever for
   throughput.
4. **Memory near the cap: confirmed as a time bomb, not the cause of the measured 15-min
   throughput.** With HEAD and 12 GB, the cache passes ~10.8 GiB after ~9 min, peak reaches 29.3 GB
   (cap 28) and free memory drops to tens of MB. Then pageins reach 4.7–6.2K/s, swapins 6–9K/s,
   decompressions 60–97K/s, and the decode step doubles. It did not reproduce as **8 GB CMPRS of the
   worker**: the worker stayed at 0.4–0.7 GB compressed, and the other processes (other
   applications) took the paging. The user's run also had 8 AppWorld processes and 25 min
   to fill up, which explains the bigger effect there. Within 15 min, R0 and R1 have the same
   throughput (16.9 vs 16.8/min): the pressure only arrives in R0's last 5 minutes.
   - The working-tree chain policy alone (R0') cuts entries from 922 to 659 MB (4.2 states). With
     `CKPTS=3` (R1) they are 560 MB (2.8 states). The 6 GB cap (R1) keeps active ≤ 18.4 GiB and
     peak 22.9 GB with no pressure, and no capacity misses (15 misses = task starts). So R1's knobs
     are right for long runs. Keep them; they leave ~9 GB to the user's apps.
5. **LRU in a round robin = worst case: not observed in these windows.** Every run has 0 capacity
   misses: misses equal task starts (the 313-token common prefix). That holds for R0 (HEAD LRU,
   12 GB, 8 episodes) and for R2 (working-tree stale/MRU order, 9 GB, 16 episodes, 18 entries max).
   R0 had 3 evictions to admit and none of them turned into a miss. The thrash the handoff saw
   (16 processes at 3 GB / 12 GB) needs either a smaller cache or longer histories than 15 min
   produce. A 25–30 min R2 would test it; I did not run it.

**16 episodes (R2):** no throughput gain (15.9/min vs 16.8), and latency doubles (p50 55 s, queue
36 s mean). The GPU is already saturated at 8 episodes, so extra concurrency only queues.

## What still limits throughput (target for the kernel work)

1. **Prefill compute: 75–80 % of the GPU time.** Each request brings ~860 new tokens (tool output),
   costing 2.9–3.7 ms per token in the run. A 432-token slab at a 12K offset, in isolation and cool
   (`prefill_prof.py --offsets 0,12288 --slab 432,2048 --pfqmm 1`), takes 840 ms (514 tok/s). The
   breakdown:

   | | share | rate |
   |---|---|---|
   | 8-bit GEMMs | ~74 % (MLP gate/up 33 %, down 17 %, GDN in 13 %) | 7.5–8.0 TFLOPS |
   | attention SDPA (unfused path below 32K keys) | 11 % | 6.2 TFLOPS |
   | GDN recurrence | 3 % | |
   | norms / elementwise / other | ~10 % | |

   At offset 0 a 2048 slab runs 622 tok/s, at 12K 512 tok/s (SDPA 12 %). Inside the run the same
   ~430-token suffix takes 1.55–2.0 s (×1.85–2.4). The ×1.5–1.85 thermal throttle explains most of
   that. The remaining ~10–30 % comes from three per-request extras:
   - ~2.2 slabs per request: message-start cuts split even a 400-token suffix.
   - Prompt-cache restore/store and checkpoint snapshots.
   - Waiting for the decode step queued ahead of the slab.

   Possible levers:
   - More efficient qmm for M ≈ 100–600 rows (pfqmm is at ~8 of 10.4 TFLOPS).
   - Faster attention for short query slabs over 10–40K keys (11 % at 12K, growing with context).
   - Fewer forced slab boundaries in short hit suffixes.

   Each saving converts to throughput at about 0.8:1.
2. **Thermal/power throttling ×1.5–1.85 under sustained load.** It hits prefill and decode alike, and
   it is the biggest single factor after the volume of prefill. Anything that cuts work (or energy
   per token) pays twice. The user could try High Power Mode (`sudo pmset -a powermode 2`; currently
   `0` = automatic) and compare warm probes; that needs the user, and was not done.
3. **Decode: 17–20 % of the time.** The step is 45–65 ms at 4–8 rows at ~10K context. Per-row
   attention is a part of that, so even a decode attention twice as fast (rattn) is worth
   ≤ ~5–8 % of throughput on this load. It matters more for decode-heavy loads.
4. Memory is not a limiter with R1's knobs. With 12 GB caches it becomes one after ~10 min (the
   decode step doubles, and other apps page).

Estimated ceiling at the measured costs: one request needs ~860 × 2.9–3.6 ms of prefill
(2.5–3.1 s) plus its share of decode (~0.6–0.7 s). That is ~17–20 requests/min, which is what
8 episodes already reach.

## Reproduce

```zsh
# worker with the scheduler time split (env as usual; SCHED_WORKER_DIR = a frozen worker.py copy, optional)
cd src/worker && QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 QWEN_PROMPT_CACHE=1 \
  QWEN_PROMPT_CACHE_GB=6 QWEN_PROMPT_CACHE_CKPTS=3 QWEN_BATCH_RESERVE_TOKENS=1024 \
  PYTHONPATH=. uv run python ../../scripts/perf/agentic/sched_timer.py >> /tmp/perf/worker_r1.log 2>&1 &
python3 scripts/perf/agentic/memsampler.py --out /tmp/perf/mem_r1.jsonl --duration-min 19 \
  --pattern '\.venv/bin/python3 ([^ ]*/)?(worker|sched_timer)\.py' &
python3 scripts/perf/agentic/replay.py --episodes 8 --duration-min 15 --tag r1 \
  --worker-log /tmp/perf/worker_r1.log --mem-log /tmp/perf/mem_r1.jsonl --out /tmp/perf/replay_r1.json
python3 scripts/perf/agentic/replay.py --summarize /tmp/perf/replay_r1.json [--window-min 10]
python3 scripts/perf/agentic/compare.py docs/perf/2026-10-16/raw/replay_r{0,0p,1,2}.json
```
