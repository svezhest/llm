# Baseline: concurrency on Ornith-1.5 9B (serialized worker, commit f2bcbdf)

Date 2026-10-11, M1 Max 32 GB, macOS 27. mlx 0.31.2 / mlx-vlm 0.6.2.
Benchmark: `scripts/bench_concurrency.py` (new, stdlib-only, in the repo).

## Setup

Worker (README command, defaults otherwise: QWEN_FP16=1, QWEN_PREFILL_STEP=2048,
prompt cache off, no DFlash, clear cache after request, pool limit 3 GB):

```zsh
cd src/worker
setopt NO_BG_NICE      # see "nice 5" below; without it zsh runs `&` jobs at nice 5
QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 \
  nohup uv run python worker.py > $SCRATCH/results/worker_baseline.log 2>&1 &
```

Gateway: `GATEWAY_ADDR=127.0.0.1:8090 RUST_LOG=info ./target/release/qwen-gateway`
(MAX_QUEUE default 32, in-flight 1).

Requests: stream, temperature 0, presence_penalty 0, `chat_template_kwargs
{"enable_thinking": false}`, `ignore_eos: true` (every answer is exactly max_tokens),
`stream_options.include_usage` (the gateway sends usage only with it; verified, all
requests had usage), distinct prompts (24 coding/explanation tasks + per-request
distinct pseudo-log filler, ~610 tokens), closed loop: N client threads, each
sends its next request as soon as the previous one finishes; per level
max(8, 2N) requests. TTFT = send -> first content token, so it includes the queue
wait. Decode tok/s = completion tokens / (end - first token).

Commands (repo root, `S=$SCRATCH/results`):

```zsh
# main sweep (final, nice 0): 2 repetitions, order 1,2,4,8 then 8,4,2,1,
# 30 s idle + a 64-token single-stream probe before every level
uv run python scripts/bench_concurrency.py --concurrency 1,2,4,8 --repeat 2 --alternate \
  --probe 64 --cooldown 30 --max-tokens 256 --prompt-tokens 600 \
  --worker-log $S/worker_baseline.log --out $S/baseline_256_nice0.json
# long decode
uv run python scripts/bench_concurrency.py --concurrency 1,8 --probe 64 --cooldown 30 \
  --max-tokens 1024 --prompt-tokens 600 --worker-log $S/worker_baseline.log --out $S/baseline_1024.json
# prefill reference
uv run python scripts/bench_concurrency.py --concurrency 1 --requests-per-level 3 --cooldown 30 \
  --max-tokens 128 --prompt-tokens 4000|8000 --worker-log ... --out $S/ref_prefill_{4000,8000}.json
```

## Results: max_tokens 256, prompt ~610 tokens (final run, worker at nice 0)

Per repetition (`baseline_256_nice0.json`):

| rep | conc | reqs | wall s | agg tok/s | TTFT mean s | TTFT p95 s | lat mean s | lat p95 s | decode tok/s/req | probe tok/s | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 8 | 87.9 | 23.3 | 2.31 | 2.98 | 10.99 | 12.00 | 29.6 | 30.4 | 13.81 |
| 0 | 2 | 8 | 87.3 | 23.5 | 11.78 | 13.90 | 20.41 | 22.25 | 29.7 | 29.4 | 13.82 |
| 0 | 4 | 8 | 89.4 | 22.9 | 27.51 | 36.27 | 36.38 | 45.01 | 28.9 | 24.7 | 13.83 |
| 0 | 8 | 16 | 184.6 | 22.2 | 62.12 | 83.54 | 71.18 | 93.42 | 28.4 | 23.9 | 13.82 |
| 1 | 8 | 16 | 197.3 | 20.8 | 67.56 | 89.41 | 77.12 | 99.06 | 26.9 | 17.8 | 13.83 |
| 1 | 4 | 8 | 96.0 | 21.3 | 30.03 | 39.83 | 39.59 | 49.85 | 26.9 | 19.2 | 13.82 |
| 1 | 2 | 8 | 90.9 | 22.5 | 12.34 | 14.98 | 21.43 | 24.51 | 28.3 | 23.7 | 13.83 |
| 1 | 1 | 8 | 94.6 | 21.6 | 2.50 | 2.92 | 11.83 | 12.73 | 27.7 | 20.9 | 13.82 |

Averaged over the 2 repetitions:

| conc | agg tok/s (min-max) | TTFT p50 s | TTFT p95 s | lat mean s | lat p50 s | lat p95 s | decode tok/s/req |
|---|---|---|---|---|---|---|---|
| 1 | 22.5 (21.6-23.3) | 2.38 | 2.95 | 11.41 | 11.23 | 12.37 | 28.6 |
| 2 | 23.0 (22.5-23.5) | 13.04 | 14.44 | 20.92 | 21.72 | 23.38 | 29.0 |
| 4 | 22.1 (21.3-22.9) | 36.20 | 38.05 | 37.99 | 44.88 | 47.43 | 27.9 |
| 8 | 21.5 (20.8-22.2) | 82.99 | 86.48 | 74.15 | 92.16 | 96.24 | 27.6 |

Aggregate tok/s includes prefill time: 256 / (2.3 s prefill + 256/28.6 s decode)
= 22.7, i.e. the measured aggregate is exactly one stream's rate at every level.

## Results: max_tokens 1024 (long decode), c=1 and c=8 (`baseline_1024.json`)

| conc | reqs | wall s | agg tok/s | TTFT mean s | TTFT p50 s | TTFT p95 s | lat mean s | lat p50 s | lat p95 s | decode tok/s/req | probe tok/s | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 328.1 | 25.0 | 2.43 | 2.3 | 2.94 | 41.01 | 40.8 | 46.79 | 26.7 | 21.8 | 13.80 |
| 8 | 16 | 727.6 | 22.5 | 240.94 | 314.6 | 332.77 | 283.48 | 358.9 | 375.90 | 24.2 | 18.8 | 13.81 |

## First sweeps (worker accidentally at nice 5; kept for the drift picture)

Sweep 1, machine cold, pristine f2bcbdf worker and gateway (`baseline_256.json`):

| conc | reqs | wall s | agg tok/s | TTFT mean s | TTFT p95 s | lat mean s | lat p95 s | decode tok/s/req | peak GB |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 72.4 | 28.3 | 1.51 | 1.58 | 9.05 | 9.16 | 34.0 | 13.81 |
| 2 | 8 | 75.2 | 27.2 | 9.93 | 11.52 | 17.67 | 19.13 | 33.1 | 13.81 |
| 4 | 8 | 80.9 | 25.3 | 23.75 | 31.33 | 32.04 | 40.84 | 31.1 | 13.81 |
| 8 | 16 | 198.7 | 20.6 | 65.58 | 96.38 | 75.30 | 106.96 | 26.8 | 13.81 |

The fall from 28.3 to 20.6 is drift over time, not concurrency: a c=1 run right
after it gave 18.9 agg / 22.9 decode tok/s. Sweep 2 (2 reps, alternating order,
`baseline_256_x2.json`): c=1 21.4, c=2 20.1, c=4 20.7, c=8 17.7 tok/s aggregate,
the second repetition's probes at 15-17 tok/s.

## Single-stream reference

| prompt tok | worker TTFT s | prefill tok/s | decode tok/s (worker) | peak GB |
|---|---|---|---|---|
| 610 (cold machine, sweep 1) | 1.43-1.57 | ~410 | 33.5-34.1 | 13.80 |
| 610 (after 30 s idle, probes) | 1.65-1.69 | ~365 | 18-30 (drifts) | 13.8 |
| 610 (back to back) | 2.1-2.5 | ~260 | 27-30 | 13.81 |
| 4045 | 12.1-13.0 | 311-336 | 31.7-32.7 | 14.77 |
| 8160 | 21.9-22.7 | 359-373 | 26.5-32.7 | 15.23 |

CLAUDE.md says decode ~33-36 tok/s (seen here only on a cold machine: 34.0) and 8K
prefill 15.5 s (~530 tok/s); today 8K took ~22 s. The gateway adds ~40 ms to
TTFT (client TTFT − worker ttft_ms), and client-side decode tok/s equals the
worker's within 0.2 tok/s.

## Memory

Worker peak per request (`mx.get_peak_memory`, logged per request): 13.72-13.83 GB at
every concurrency and both lengths (610-token prompt; active 13.01-13.04 GB after
the request, pool ≤ 0.08 GB); 14.77 GB at 4K, 15.23 GB at 8K. No growth over
~150 requests (serialized: one KV cache at a time).

## Observations

- **Serialized as expected**: aggregate tok/s stays at one stream's rate
  (21.5-23 tok/s at 256 tokens, 22.5-25 at 1024) whatever the concurrency. Latency
  grows linearly with N: steady-state (p50) latency ≈ N × service time (11.2 / 21.7 /
  44.9 / 92.2 s for N = 1/2/4/8, ~11.2 s per request; the mean is lower because the
  first N requests of a closed loop wait for fewer predecessors); TTFT is mostly queue wait
  (p50 83 s at N=8 for 256 tokens, 315 s for 1024). Per-request decode tok/s does not
  depend on N.
- **No errors, no 503s** (MAX_QUEUE 32 > 8 clients), no stream error frames. Usage
  arrives in the final chunk only with `stream_options.include_usage`.
- **Strong drift in decode speed on this machine, not in prefill.** Decode went
  34 → 16-30 tok/s over minutes of sustained load and recovers only partly after 30-45 s
  idle; prefill TTFT after a 30 s idle stayed at 1.65-1.69 s throughout. Background
  load at the time: WindowServer 40-50 % CPU, a browser playing video
  (VTDecoderXPCService), Activity Monitor; GPU at 12 % before the runs and 0 % later.
  Plain single numbers are therefore ±20 %; comparisons need the same conditions:
  use `--repeat 2 --alternate --probe 64 --cooldown 30` and compare agg tok/s
  against the probe, or interleave A/B runs.
- **nice 5 trap**: zsh's default `BG_NICE` starts `cmd &` at nice 5; the first
  worker and gateway ran at nice 5 (so do the user's own gateways on :8080/:8081,
  `STAT SN`). After restarting at nice 0 (`setopt NO_BG_NICE`) the sustained
  decode in the second repetition was 27-28 instead of 20-23 tok/s, though the
  drift above muddies how much of that is nice. Benchmarks should start the
  worker at nice 0.
- TTFT back to back (2.2-2.5 s) is higher than after an idle period (1.67 s) or
  on a cold machine (1.5 s) for the same 610-token prompt: prefill is also slower
  when the machine is hot.
- `ignore_eos` keeps every answer at exactly max_tokens (`--ignore-eos off` to
  let the model stop).

## Caveat: code under test

Another agent edited the repo during this run (gateway pool/journal, proto, worker
batching). Sweep 1 (19:51) ran the pristine f2bcbdf worker and gateway. At 20:29
I restarted both at nice 0: the gateway binary had been rebuilt by someone else at
20:26 (new pool.rs, `GATEWAY_MAX_INFLIGHT` / Health `max_batch`); it logged
`in-flight capacity capacity=1 worker_max_batch=0` and `queue_ms=0 batch_mean=0.0`
for every request, so the worker it talked to had no batching (Health.max_batch
unset) and the gateway serialized as before. The results are the same
serialized behaviour, but the nice-0 worker's worker.py is the file as of 20:29,
not guaranteed byte-identical to f2bcbdf.

## Files

- `scripts/bench_concurrency.py` (repo)
- results: `baseline_256_nice0.{json,txt}` (final), `baseline_1024.{json,txt}`,
  `ref_prefill_{4000,8000}.json`, `baseline_256.{json,txt}` (sweep 1),
  `baseline_256_x2.{json,txt}` (sweep 2), `hot_c1.json`, `nice0_c1_check.json`,
  logs `worker_baseline.log` / `gateway_baseline.log` (running instance, nice 0),
  `worker_baseline_nice5.log` / `gateway_baseline_nice5.log` (sweeps 1-2).
