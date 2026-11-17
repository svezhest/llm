# Measuring performance on this machine

Current conclusions (the state of all topics: [MAP.md](MAP.md)):

- Never compare two separate runs: the GPU drifts ±20–30 % over minutes. A/B inside one
  process, interleaved, medians over ≥ 3 rounds.
- Nothing else on the GPU, and no `BG_NICE` on the measured processes.
- Systems that cannot share a process (ours, llama.cpp, mlx-vlm) are compared cell by cell at
  the same position of sessions with one fixed cell order: a session loses 16–23 % of its
  speed in its first 2–4 minutes of load, repeatably (two sessions agree within a few %), and
  the memory-bandwidth probe does not see it. A plan must interleave the classes it reports
  (`scripts/perf/bench27/plans.json` `headline2`; [perf/2026-11-17/decode_27b_bench.md](perf/2026-11-17/decode_27b_bench.md)).
- CPU work beside a GPU measurement distorts it too (pytest shares the memory bandwidth): the
  CPU suite gets its own slot.
- Reports and raw data go to `docs/perf/<date>/`; conclusions to the topic file.

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
