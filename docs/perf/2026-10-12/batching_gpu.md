# Continuous batching: GPU results

Date: 2026-10-11. Machine: M1 Max 32 GB. Libraries: mlx 0.31.2 / mlx-vlm 0.6.2.

* Worker: `src/worker/worker.py` (batching).
* Reference: the pre-change copy, `scratchpad/orig_worker/worker.py`.
* Raw files: `results/bgpu/`. Scripts: `gpu_tests/`.

Every worker and gateway ran at **nice 0**, started with `setopt NO_BG_NICE` (`gpu_tests/start_worker.sh`). Nothing else heavy ran during the timed runs.

The machine drifts a lot. The 64-token probe before each level ranged from **17.7 to 37.8 tok/s** in the same session, so compare runs through `agg/probe`.

## Summary

* **B=1 is byte-identical to the old worker.**
  * 9B, `QWEN_MAX_BATCH=1`: 15/15 corpus cases and 16/16 sequential prompts.
  * 9B, `QWEN_MAX_BATCH=8` with sequential requests: 15/15.
  * 35B (q4 KV): 15/15 and 4/4.
  * 27B (q8 KV, `QWEN_PREFILL_STEP=512`): 15/15 and 4/4.
  * The corpus covers: greedy, seeded sampling, penalties, thinking ± budget, stop strings, min_tokens + ignore_eos, bad_words, json_schema, tools auto / required.
* **Batched numerics, 9B, teacher-forced.**
  * B=2 and B=4 are **bit-identical** to solo. Each row still goes through its own per-row qmv.
  * B=8: 100% top-1 agreement, mean KL 8.5e-6.
  * B=16: 99.80% top-1, mean KL 1.1e-5.
  * Free-running answers that differ are near-tie flips.
* **Throughput, 9B (through the gateway, ~600-token prompts, ignore_eos).**
  * Old worker: 22.0–25.5 tok/s at any concurrency.
  * New worker, `QWEN_MAX_BATCH=16`:
    * c=16 with 256-token answers: **56.0 tok/s** (agg/probe 2.57).
    * c=16 with 1024-token answers: **81.2 tok/s** (agg/probe 3.37).
  * New worker, `QWEN_MAX_BATCH=8`:
    * c=8: 40.3 tok/s (1.86).
    * c=8 with 1024-token answers: 46.1 tok/s (2.61).
  * Mean latency at c=8 dropped from 74 s (serialized baseline) to 48 s.
* **The ceiling is a kernel property, not the scheduler.**
  * The design microbench (no scheduler) gives the same ms/step as the worker: 9B **41 / 51 / 90 / 127 / 133 ms** at B = 1 / 2 / 4 / 8 / 16.
  * MLX on M1 runs a per-row `qmv` below 10 rows for the 4096-wide projections (below 6 for wider layers), then `qmm_splitk`, whose cost is flat. So **B=16 costs about the same per step as B=8**, and 16 is the efficient batch on the 9B.
* **35B MoE: batching pays only at large B.**
  * Distinct prompts read about B×8 distinct routed experts.
  * Aggregate in the worker: 19 / 28 / 38 / 69 tok/s at 2 / 4 / 8 / 16 concurrent, against 38–42 tok/s solo decode.
  * The design microbench overstates the MoE gain: it clones one prompt into every row, so all rows route to the same experts.
* **Both appendix-A position bugs are confirmed on the real models and fixed:**
  * A1 (DFlash): max |Δlogit| 0.88 before the fix, 0 after.
  * A2 (vision prompt longer than one slab): 0.30 before; 0.05 after, which is the fp16 slab-shape noise floor.
* **Behaviour changes made during the GPU phase** (details in `design/worker_changes.md`):
  1. The prefill time share now scales with batch occupancy (memory test, §6).
  2. The reservation counts the recurrent state twice (memory test, §6).
  3. Check rejects out-of-vocab `stop_token_ids` / `allowed_token_ids` (coordinator request).
  4. The log line puts `queue_ms=` and `batch=` last, so `bench_concurrency.py` keeps parsing it; the script's regex now also reads them.

## 1. Identity (B=1) vs the pre-change worker

```
cd src/worker; W="PYTHONPATH=. .venv/bin/python $SP/gpu_tests/wtest.py"
gpu_tests/start_worker.sh <log> $SP/orig_worker/worker.py QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0
$W corpus --out bgpu/corpus_old.json; $W seq --n 16 --max-tokens 256 --out bgpu/seq16_old.json
# the same with worker.py and QWEN_MAX_BATCH=1, then with the default (8); then:
$W compare bgpu/corpus_old.json bgpu/corpus_mb1.json         # 15/15 byte-identical
$W compare bgpu/seq16_old.json  bgpu/seq16_mb1.json          # 16/16
$W compare bgpu/corpus_old.json bgpu/corpus_mb8.json         # 15/15
gpu_tests/smoke_big.sh smoke35 QWEN_MODEL=../../weights/qwen36-35b-a3b-mlx-ud3 QWEN_KV_BITS=4 QWEN_WIRED_LIMIT_GB=27   # 15/15, 4/4
gpu_tests/smoke_big.sh smoke27 QWEN_MODEL=../../weights/qwen38-27b-mlx-5bit QWEN_KV_BITS=8 QWEN_WIRED_LIMIT_GB=27 QWEN_PREFILL_STEP=512   # 15/15, 4/4
```

* The 27B ran at `iogpu.wired_limit_mb=28672`, which the user set during the session.
* 27B speed numbers are not comparable. It peaks at 22.5 GB and the machine had 9.5 GB of swap in use. Tokens/s swung 4–13 in both workers; on the 4 sequential prompts the new worker was faster on all 4.

## 2. Agreement under batching (9B, teacher-forced, 128 steps)

`PYTHONPATH=. QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 .venv/bin/python $SP/gpu_tests/agree.py`

```
B= 2: top-1 agreement 100.00% (256/256), KL mean 0.00e+00 p99 0.00e+00 max 0.00e+00, peak 13.2 GB
B= 4: top-1 agreement 100.00% (512/512), KL mean 0.00e+00 p99 0.00e+00 max 0.00e+00, peak 13.7 GB
B= 8: top-1 agreement 100.00% (1024/1024), KL mean 8.53e-06 p99 9.19e-05 max 1.07e-03, peak 14.3 GB
B=16: top-1 agreement 99.80% (2044/2048), KL mean 1.13e-05 p99 1.10e-04 max 1.67e-03, peak 15.5 GB
```

**Free-running** (`$W conc --n N` vs sequential): all answers finish. Any differences are near-tie flips at B≥8, the same class as the DFlash verify noise.

**35B with B=4 concurrent:** the 4 answers were byte-identical to the solo ones.

## 3. Decode ceiling by batch size (design microbench, no scheduler)

9B, ctx 1K (`MB_MODES=lib1,rows MB_BATCHES=1,2,4,8,16`):

| B | ms/step | aggregate tok/s | × vs B=1 |
|---|---|---|---|
| 1 | 41.1 | 24.3 | 1.00 |
| 2 | 50.7 | 39.5 | 1.62 |
| 4 | 90.5 | 44.2 | 1.82 |
| 8 | 127.0 | 63.0 | 2.59 |
| 16 | 133.0 | 120.3 | 4.95 |

* The worker matched it: 127 ms/step at B=8 in the `conc 8` run.
* So the per-row attention loop and the per-row sampling cost nothing measurable. The B=4..8 cost is the per-row qmv of the ≤4096-wide projections.
* **Recommendation:** `QWEN_MAX_BATCH=16` for the 9B. The code default stays 8, the value the gateway was rolled out with (see §5).

35B, ctx 1K. All rows are clones of one prompt here, so expert routing is shared and the numbers are optimistic:

```
model qwen36-35b-a3b-mlx-ud3 loaded in 8.7s, active 17.37 GB; KV 5.6 KiB/token (kv_bits=4), GDN 64.4 MB/row; modes=['lib1', 'rows'] sync=False ragged=False row_sampler=True
ctx   1024 B  1 lib1:    35.4 ms/step     28.3 tok/s aggregate    28.3 tok/s per row  x1.00 vs B=1  peak 18.38 GB
ctx   1024 B  2 rows:    49.1 ms/step     40.7 tok/s aggregate    20.4 tok/s per row  x1.44 vs B=1  peak 18.52 GB
ctx   1024 B  4 rows:    62.1 ms/step     64.4 tok/s aggregate    16.1 tok/s per row  x2.28 vs B=1  peak 18.77 GB
ctx   1024 B  8 rows:    82.1 ms/step     97.5 tok/s aggregate    12.2 tok/s per row  x3.45 vs B=1  peak 19.17 GB
ctx   1024 B 16 rows:   149.3 ms/step    107.1 tok/s aggregate     6.7 tok/s per row  x3.79 vs B=1  peak 20.02 GB
ctx   1024 B 32 rows:   230.6 ms/step    138.8 tok/s aggregate     4.3 tok/s per row  x4.91 vs B=1  peak 21.65 GB
```

The real worker with distinct prompts (`conc`, 128 tokens) gave 10.4 (c=1, cold) / 19.4 / 27.8 / 38.5 / 69.1 tok/s at c = 1/2/4/8/16. Sequential solo decode was 36–42 tok/s.

## 4. Feature checks inside a batch (9B, `QWEN_MAX_BATCH=8`; all PASS)

| suite | result |
|---|---|
| cancel (8 streams; 2 Cancel RPCs at 80/300 chars; 1 dropped stream) | Each cancelled row ends `cancelled` / dropped. The others finish `length`. Health ok. `active=` returns to 12.92 GB (the load baseline) after each batch. |
| stops (4 different stop sets, 2 free rows) | 3 hit `stop`, 1 ran to `length`. No stop text leaked. |
| mixed max_tokens 1/7/64/512 (ignore_eos) | Exact counts, all `length`. |
| thinking on/off mixed + budget 32 | The reasoning/answer split is right. The budget row has 34 reasoning tokens (32 + forced `\n`), then an answer. |
| grammar: 4 json_schema rows + 4 free rows | No grammar errors. The 2 rows that finished are valid schema JSON; the other 2 hit max_tokens 200 (long tag lists). Sync mode cost nothing visible (free rows ran at the same 8.3 tok/s per row as the schema rows). |
| 503 overflow: gateway `MAX_QUEUE=4 GATEWAY_MAX_INFLIGHT=2`, 10 concurrent requests | **4 × 200, 6 × 503**, exactly as expected. |
| prompt cache (`QWEN_PROMPT_CACHE=1`), 8 conversations over one 7.4K-token system prompt | Cold TTFT 24.6 s. Then **7/7 HIT** (7359 tokens reused), TTFT 0.3 s for the first and 3.4 s for the rest. The 34-token prefills are serialized and each one stores a ~250 MB entry. |
| DFlash (`QWEN_DRAFT=../../weights/qwen35-9b-dflash`) | Solo outputs 14/15 identical to the old worker's DFlash; code0 differs because of the A1 fix. 13/15 identical to plain greedy (near-tie flips, as before). New DFlash ran ~10% faster than old DFlash in the same session (28.8 vs 26.3 tok/s, code0..4). Demotion test: the second request arrived 3 s in, the log shows `dflash: 34 tokens ... demoted`, the first answer equals its DFlash-solo answer, and the second equals plain solo. |
| out-of-vocab ids | CPU-tested (400 `invalid_request` before queueing). |

## 5. Concurrency benchmark (gateway :8090, `scripts/bench_concurrency.py`)

Command (repo root): `.venv/bin/python scripts/bench_concurrency.py --concurrency <levels> --max-tokens <256|1024> --probe 64 --cooldown 30 --prompt-tokens 600 [--repeat 2 --alternate] --worker-log <worker log> --out bgpu/bench/<run>.json`

The whole interleaved session is `gpu_tests/bench_all.sh`, run in this order: old c=1 → new MB=8 (256 ×2 alternating, then 1024) → new MB=16 (256 ×2, then 1024 at c=16) → old c=1. Worker env: `QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0`, plus `QWEN_MAX_BATCH=16` for the mb16 runs. Gateway: `GATEWAY_ADDR=127.0.0.1:8090 RUST_LOG=info`, capacity taken from Health.max_batch (old worker: 1).

| run | conc | reps | reqs | agg tok/s (min-max) | agg/probe | probe tok/s | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| old_c1_a | 1 | 1 | 8 | 25.5 (25.5-25.5) | 0.79 | 32.2 | 1.55 | 2.85 | 9.82 | 11.00 | 31.1 | - | - | 13.78 |
| mb8_256 | 1 | 2 | 16 | 23.8 (23.2-24.4) | 0.86 | 28.2 | 2.44 | 3.13 | 10.75 | 11.61 | 31.0 | 1.00 | 2 | 13.84 |
| mb8_256 | 2 | 2 | 16 | 29.4 (28.8-29.9) | 1.20 | 24.5 | 4.15 | 5.15 | 17.25 | 19.86 | 18.9 | 1.98 | 424 | 13.9 |
| mb8_256 | 4 | 2 | 16 | 31.9 (31.0-32.9) | 1.37 | 23.5 | 5.66 | 10.66 | 31.85 | 37.69 | 10.3 | 3.89 | 2242 | 14.09 |
| mb8_256 | 8 | 2 | 32 | 40.3 (40.2-40.4) | 1.86 | 21.7 | 9.08 | 22.24 | 48.46 | 63.50 | 6.8 | 7.44 | 6145 | 14.43 |
| mb8_256 | 16 | 2 | 64 | 39.1 (37.8-40.4) | 1.76 | 22.4 | 58.06 | 73.09 | 100.54 | 118.39 | 6.0 | 7.55 | 3351 | 14.43 |
| mb8_1024 | 1 | 1 | 8 | 24.4 (24.4-24.4) | 1.03 | 23.7 | 2.77 | 3.28 | 42.60 | 49.92 | 26.5 | 1.00 | 4 | 13.79 |
| mb8_1024 | 8 | 1 | 16 | 46.1 (46.1-46.1) | 2.61 | 17.7 | 11.75 | 26.28 | 175.70 | 186.29 | 6.3 | 7.86 | 7599 | 14.57 |
| mb16_256 | 1 | 2 | 16 | 22.3 (20.6-24.0) | 1.00 | 23.1 | 2.65 | 3.07 | 11.47 | 12.98 | 29.1 | 1.00 | 2 | 13.82 |
| mb16_256 | 2 | 2 | 16 | 27.9 (26.0-29.8) | 0.94 | 30.5 | 4.64 | 5.52 | 18.23 | 21.68 | 18.1 | 1.99 | 445 | 13.91 |
| mb16_256 | 4 | 2 | 16 | 32.8 (32.7-33.0) | 1.35 | 24.4 | 7.14 | 10.65 | 31.67 | 38.30 | 10.5 | 3.94 | 2007 | 14.07 |
| mb16_256 | 8 | 2 | 32 | 40.9 (39.0-42.8) | 1.58 | 28.5 | 8.12 | 20.30 | 49.31 | 62.56 | 6.7 | 7.75 | 5742 | 14.42 |
| mb16_256 | 16 | 2 | 64 | 56.0 (53.8-58.2) | 2.57 | 22.1 | 18.03 | 41.70 | 68.97 | 95.82 | 5.3 | 14.09 | 14765 | 15.1 |
| mb16_1024 | 16 | 1 | 32 | 81.2 (81.2-81.2) | 3.37 | 24.1 | 24.71 | 40.38 | 201.79 | 215.76 | 5.9 | 15.52 | 17254 | 15.47 |
| old_c1_b | 1 | 1 | 8 | 22.0 (22.0-22.0) | 1.14 | 19.3 | 2.74 | 3.11 | 11.30 | 13.28 | 28.9 | - | - | 13.81 |

How to read the table:

* `batch_mean` and `queue_ms` are the worker's per-request `Usage` values, joined from the worker log.
* TTFT and latency are measured at the client; TTFT includes the queue wait.
* The old worker's two c=1 runs (start and end of the session) bracket the drift: 25.5 tok/s at probe 32.2, and 22.0 at probe 19.3.

Takeaways:

* **c=1: no regression.** Agg/probe was 0.86 (MB=8) and 1.00 (MB=16), against 0.79 and 1.14 for the two old-worker runs. Per-request decode was 31.0 / 29.1 tok/s, against 31.1 / 28.9 for the old worker.
* **c=2..8:**
  * Aggregate is ×1.3–1.8 the old worker's.
  * Mean latency at c=8 is 48 s, against 74 s in the serialized baseline (`baseline.md`).
  * TTFT p50 is 9 s, against 83 s.
* **Why the gain at 256 tokens is capped near ×1.8.** The closed-loop clients start and finish in waves. Eight ~600-token prefills (1.2–1.7 s each) run back to back between waves. This is the design's V2 "piggyback decode on prefill slabs" item.
* **Long answers show the decode gain.** 1024 tokens: c=8 46.1 tok/s (agg/probe 2.61); c=16 at MB=16 81.2 tok/s (agg/probe 3.37). The old worker would be ~22–25 tok/s.
* **MB=16 vs MB=8:** equal up to c=8. At c=16, MB=16 gives 56 tok/s against 39, with TTFT p50 18 s against 58 s. Recommend 16 for the 9B.

## 6. Memory (9B, 8 rows with ~8.1K-token prompts, 256 tokens each)

| run | wall s | agg tok/s | batch_mean | peak GB | notes |
|---|---|---|---|---|---|
| fixed share 0.5 (first version) | 268.6 | 7.6 | 1.0 | 15.2 | Batches never formed: a lone decoding row took half the GPU from the next 20 s prefill. Slower than serial (≈250 s). |
| occupancy-scaled share (fix) | 231.7 | 8.8 | 2.0–3.5 | 16.5 | Prefill-bound load; batching can only save the decode part. |
| share 1 (all prefill first) | 238.5 | 8.6 | 8.0 | **16.15** | Peak at a true B=8 with 8K contexts: active 15.5 GB against 11.6 GB of weights. |
| `QWEN_BATCH_MEM_GB=0.8`, 4 rows | 132.9 | 7.7 | 2.0 | 15.5 | Admission held 2 rows; the rest waited (log `waiting=2`). No OOM, no errors. |

* Prefill speed matched the baseline (8K: ~22 s).
* The reservation underestimated actual use: 2.77 GB reserved against ~3.9 GB used above the weights. The step-ahead keeps two copies of the stacked recurrent state, so the reservation now counts it twice (≈3.6 GB).

## 7. RoPE position bugs (appendix A), real models

```
PYTHONPATH=. QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 .venv/bin/python $SP/gpu_tests/rope_checks.py a1
PYTHONPATH=. QWEN_MODEL=../../weights/ornith10-9b-mlx-8bit QWEN_KV_BITS=0 .venv/bin/python $SP/gpu_tests/rope_checks.py a2
```

```
A1: prompt 4638 tokens
  as _spec does it (no position_ids, _position_ids None): max|dlogit| 0.8828, top-1 271 vs 271, top-5 overlap 4/5
  fixed (_rope_deltas = zeros): max|dlogit| 0.0000, top-1 271 vs 271, top-5 overlap 5/5
A2: prompt 1042 tokens (image included)
  slab 512 as the worker does it vs one slab: max|dlogit| 0.2969, top-1 760 vs 760, top-5 overlap 5/5
  slab 512 with sliced _position_ids vs one slab: max|dlogit| 0.0508, top-1 760 vs 760, top-5 overlap 5/5
```

Both are fixed in the worker:

* `_spec` resets `_rope_deltas` to zeros.
* Vision slabs pass sliced `_position_ids`.

## 8. State left running

* **Worker:** new `worker.py` on 127.0.0.1:50051, 9B, `QWEN_KV_BITS=0 QWEN_MAX_BATCH=8`, nice 0. Log: `results/worker_batching.log`.
* **Gateway:** `target/release/qwen-gateway` on :8090, nice 0, capacity 8. Log: `results/gateway_batching.log`.
* The gateways on :8080 / :8081 were not touched.

## 9. Round 2: GPU re-verification after the review fixes (2026-10-12)

Correctness only. No throughput bench was run: a Metal kernel agent ran short GPU bursts in parallel. Raw files are in `results/r2/`.

Setup:
* 9B: `QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0`, with the new defaults (`QWEN_MAX_BATCH=16`, `QWEN_BATCH_RESERVE_TOKENS=4096`).
* Vision: ornith10.
* Everything at nice 0. The gateway on :8090 is the new binary.

| check | result |
|---|---|
| Identity corpus (15 cases) at `QWEN_MAX_BATCH=1` vs the pre-change worker | **PASS**: 15/15 byte-identical |
| 16 sequential prompts at `QWEN_MAX_BATCH=1` | **PASS**: 16/16 |
| Identity corpus at `QWEN_MAX_BATCH=16`, sequential requests | **PASS**: 15/15 |
| cancel (2 Cancel RPCs + 1 dropped stream among 8 streams) | **PASS**. Health ready, max_batch 16. |
| stops / mixed max_tokens / thinking + budget | **PASS** / **PASS** / **PASS** |
| grammar (4 json_schema + 4 free rows) | **PASS** on rerun. The suite prints FAIL because 2 answers hit the token cap: j2 also does that alone (the greedy tag list loops); j1 diverges into the same loop at B=8 (kernel rounding). Ran as a batch of 4 (the bit-exact qmv range), all 4 schema rows were **byte-identical to solo**, so the guided path is exact in a batch. |
| 16 concurrent (256 tokens) vs solo | 10/16 identical. The 6 others diverge at chars 221–753: B=16 near-tie flips, as in round 1 (teacher-forced 99.8% top-1). |
| vision (ornith10): file image 213 tokens, data-URI image 1047 tokens, old vs new | **PASS**: 2/2 byte-identical. The handler-side image path changed no output. |
| a slow image URL (served after 6 s) arriving while 4 text rows stream | **PASS**. The text rows' largest inter-token gap around the fetch was 1.0–2.4 s; that is the vision prefill, not the 6 s fetch. The vision row answered normally. |
| 503, gateway queue: `MAX_QUEUE=4 GATEWAY_MAX_INFLIGHT=2`, 10 concurrent requests | **PASS**: 4 × 200, 6 × 503 |
| 503, worker slots: `QWEN_GEN_SLOTS=3`, default gateway (in-flight 16), 10 concurrent requests | **PASS**: 3 × 200, 7 × 503. The worker logged 7 "refused" (RESOURCE_EXHAUSTED); the next request got 200. |
| memory, 8 rows × 8.1K-token prompts, 256 tokens each | **PASS**: wall 237.8 s, batch_mean 3.4–5.7, peak 17.47 GB, no errors. Round 1 at max_batch 8: batch_mean 2–3.5, peak 16.5 GB. |
| memory, the same 8 × 8K **without max_tokens** (natural stop; reservation capped at 4096 tokens) | **PASS**: all 8 batched (batch_mean 5.0–7.8; answers 697–2489 tokens). Wall 575 s, 22.2 tok/s aggregate, peak 17.47 GB. Max reserved 3.95 GiB with 8 rows, against ~3.8 GiB measured above the weights. No errors. Under round 1's rule each request would have reserved the rest of the 128K context and run alone. |

**Left running:** the new worker on 127.0.0.1:50051 with default `QWEN_MAX_BATCH=16` (nice 0, log `results/r2/w_mem.log`), and the new gateway on :8090 (nice 0, default env, log `results/r2/gw_slots.log`). :8080 and :8081 were not touched.

## 10. Round 3 re-check (2026-10-12; mrqmm integrated, QWEN_MRQMM=1 default)

| check | result |
|---|---|
| Identity corpus at `QWEN_MAX_BATCH=1` vs the pre-change worker | **PASS**: 15/15 |
| Identity corpus at the defaults (max_batch 16, mrqmm on), sequential | **PASS**: 15/15 |
| SIGTERM, idle worker | **PASS**: "worker stopped", exit 0 |
| SIGTERM with 3 streams in flight | **PASS**: all 3 got `Error(shutdown)` at once; exit 0; no segfault or fatal in the log |
| Vision on ornith10 (`padding=False`) vs the reference, plus a slow URL mid-batch | **PASS**: 2/2 identical. Largest text-row gap around the 6 s fetch: 1.55 s. |
| 8 × 8K memory test at the defaults | **PASS**: wall 215 s, batch_mean 2.4–4.2, peak 16.82 GB, max reserved 2.23 GiB. **Growth cap fired 0 times.** No errors. |
| `pytest tests/test_mrqmm.py` (Metal) | **PASS**: 133 passed, 1 skipped (real-weights test, opt-in) |
| CPU suite (`--ignore tests/test_mrqmm.py`) | **PASS**: 54 passed |

Raw files in `results/r3/`. The worker is left running with the defaults on :50051 (nice 0, log `results/worker_r3.log`).
