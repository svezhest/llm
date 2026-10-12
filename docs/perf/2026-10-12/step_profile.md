# Worker decode-step profile (9B, batched decode)

Date 2026-10-12, M1 Max 32 GB, mlx 0.31.2 / mlx-vlm 0.6.2, Ornith-1.5 9B 8-bit, `QWEN_KV_BITS=0`,
defaults otherwise (max_batch 16, mrqmm on). I held the GPU lease. The worker was stopped for the
in-process runs and restarted afterwards; it is running now with defaults on :50051 at nice 0.

## Measurement caveat: the GPU speed drifts by ±30 %

The same step changed from 37 to 60 ms at B=8 within minutes, and the whole machine ran at
×1.5–1.8 in a "slow state" (8K prefill took 25–27 s instead of 15.5 s). Nothing else used the GPU
(ioreg utilization was ~0 between runs). The drift is probably thermal or power state, since it grows over
long continuous runs. Because of it, comparisons between processes are only indicative. Every claim
below comes from an **in-run A/B**: the profiler swaps the implementation every 12–16 decode
steps inside one batch, drops the 2 steps after each swap, and compares the medians.

## Tools (all in `scratchpad/prof/`)

* **`step_prof.py`** builds the real `Engine` the way `worker.py` does and wraps the scheduler's
  methods. It does not edit the repo. It times these per tick:
  * `tick`, `_iterate`, `_step`;
  * `LanguageModel.__call__`, which is the graph build;
  * `_sample`, `_compact`, `_regrow`, `quantize`;
  * `mx.async_eval`, which is the graph walk plus Metal encoding;
  * the host wait in `mx.array.tolist` and `.item()`;
  * `mx.clear_cache`.

  It records both wall time and thread CPU time. Modes:
  * load from in-process handlers (`Engine.generate`), or from a bare queue consumer;
  * `--serve`: the gRPC server in-process, loaded by `bench_concurrency` through the gateway on :8090;
  * `--ab X,Y --ab-steps N`: the step-interleaved A/B;
  * slack probes `sN`, which add N ms of host time per step;
  * `--dot`, which counts the graph nodes;
  * `--cprofile`.
* **`variants.py`** measures GPU-time variants of the batched step in one process: no attention,
  no cache update, and one batched greedy argmax.
* **`sampler_bench.py`** measures the samplers on (B, V) logits in isolation.
* **`batch_exact.py`** and **`sample_rows_check.py`** check bit-exactness on Metal.
* **`prefill_check.py`** measures prefill time and peak memory under an MLX env setting.

Raw outputs are in `results/prof/`.

## Breakdown (before the fixes)

In-process, real handlers, ~1K context, steady B rows, median ms per step:

| B | period | Python build (fwd, CPU) | async_eval wall / CPU | host wait on tokens | sampler build | consume + tick | quantize | clear_cache /100 steps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 27.5 | 2.2 | 25.2 / 4.4 | 0.01 | 0.01 | 0.05 | 0.00 | 0 |
| 8 | 36.8 | 2.7 | 23.9 / 4.3 | 9.4 | 0.03 | 0.09 | 0.01 | 0 |
| 16 | 54.4 | 3.7 | 37.2 / 6.0 | 13.0 | 0.06 | 0.17 | 0.01 | 0.9 |

Through gRPC with gateway load (`bench_concurrency` c=8/16, 600-token prompts, 512 tokens):

| B | period | fwd CPU | async_eval CPU | wait |
|---:|---:|---:|---:|---:|
| 8 | 47.0 | 5.9 | 8.0 | 11.5 |
| 16 | 81.4 | 5.6 | 8.8 | 19.7 |

The same microbench (`mrqmm2/step_ab.py`), run in the same session, gave 37.2 ms (B=8) and
55.2 ms (B=16).

The items from the task:

* **(a) Python graph build**
  * 2.2 / 2.7 / 3.7 ms CPU at B = 1 / 8 / 16. About half of it is in `_rows_attention`: 128
    `update_and_fetch` calls at B=16 take 8.5 µs each, 1.1 ms in all.
  * Under gRPC load the same work takes 1.3–2× the CPU time. The scheduler thread runs slower
    next to ~40 gRPC and handler threads.
  * There is no GIL wait in the build: its wall time equals its CPU time.
* **(b) GPU time** is the step: the host is busy only 7–14 ms per step.
  * `async_eval` wall ≫ CPU because of MLX's back-pressure. It commits a command buffer every
    50 kernels or 50M input elements on M1 Max, and blocks the encoder while more than 10
    buffers are in flight (`MAX_ACTIVE_TASKS`).
* **(c) Per-row post-processing**
  * The greedy sampler build is negligible: 0.03–0.07 ms.
  * Its GPU cost is 0.5 ms at B=16. Batching the greedy argmax gave nothing (`variants.py`
    bsamp: ×1.00–1.02).
  * Penalties cost ~1 ms of GPU at B=16.
  * The **card sampler** (temperature 0.6, top_p 0.95, top_k 20) costs **14.8 ms of GPU per step
    at B=16** row by row: 16 full-vocabulary topk and argsort launches. That is +20–40 % of a
    B=16 step.
* **(d) Per-row attention**
  * Graph at B=8: 1716 nodes. They include 128 SliceUpdate, 64 SDPA (2-pass) and 352 Slice
    (views). The update + SDPA per row is 2 SliceUpdates + 2 SDPA passes.
  * Cost of the whole per-row update + SDPA (`variants.py` noattn, in-process A/B):

    | context | B=8 | B=16 |
    |---|---:|---:|
    | 1K | 3.7 % | 7.5 % |
    | 8K | 16 % | 27 % |

  * At 8K and B=16 that is ~24 ms of attention against a ~11 ms bandwidth floor (MLX's 2-pass
    kernel already reads K/V once per GQA group). This is the next GPU-side lever: a fused
    multi-row decode-attention kernel. It is not done here.
* **(e) `quantize(row.cache)`**: 0.01 ms; a no-op for fp16 KV (`kv_bits=None` returns at once).
* **(f) GDN stack/gather on join/leave**
  * The cost lands in the step after a join or leave.
  * A leave at B=16 copies ~0.8 GB once. That is negligible over 512-token rows.
  * `_compact` is 0.01 ms per step.
* **(g) Handler threads / GIL**
  * In-process, the bare consumer and the real handlers measured the same within noise.
  * Under gRPC the scheduler's CPU work grows (see (a)). That mattered only because of (i).
* **(h) `mx.clear_cache`**: 0.3–0.9 calls per 100 steps (every 256 steps plus row ends).
  Negligible.
* **(i) Step-ahead was broken for B ≥ 2.** This is the main finding.
  * `_iterate` read the tokens with `mx.concatenate([r.y ...]).tolist()`. That concatenate is a
    new kernel, queued on the generation stream **behind step k+1, which was just submitted**.
  * So the read waited for step k+1 too (the 9–20 ms `wait` above). The GPU then sat idle while
    the host built and encoded step k+2.
  * The microbench reads the async-evaluated token array itself, so it never had the bug.
  * B=1 used `y.item()` and was fine.

## Changes (`src/worker/worker.py`)

1. **Token read**: `toks = [r.y.item() for r in live]`.
   * Each row's token was evaluated by its step's `async_eval`, so `item()` waits for that step
     only. No new op is queued.
   * No numeric change; host wait 9–20 ms → 0.02 ms.
2. **Grouped draws in a batch (`_Sampler`, `sample_rows`)**
   * `make_sampler` now returns a `_Sampler` object. Calling it is the same per-row sampler,
     with the same ops in the same order.
   * In a batched step (B ≥ 2), rows without a grammar are grouped by (sampler settings, logits
     dtype). Each group is filtered as one (G, V) stack: temperature / min_p / top_k / top_p,
     or the greedy argmax(x − logsumexp). Each row then draws with its own key.
   * B=1 and grammar rows keep the per-row path untouched (`_sample` = `_logits` + guide/sampler).
   * **Bit-exactness, checked on Metal.** MLX's row ops give each row of a stack bit-for-bit the
     row's own result. Checked: topk, argsort (ties included), softmax, cumsum, logsumexp, max.
     Details in `batch_exact.py`.
   * **Token-level check.** `sample_rows` vs per-row samplers: 1600/1600 identical over 25 steps
     with mixed settings and dtypes. The filtered stacks are bitwise equal (`sample_rows_check.py`).
   * Samplers in isolation, B=16:

     | sampler | per row | grouped |
     |---|---:|---:|
     | card | 14.8 ms | 7.8 ms |
     | top_p | 9.5 ms | 6.5 ms |
     | top_k | 6.8 ms | 4.4 ms |
     | greedy | 0.5 ms | 0.5 ms |

   * One behaviour change: an exception while drawing fails the rows of that group, not a single
     row (caught per group inside `sample_rows`, review r3 L2; the other groups are drawn).
     Graph-building errors there are practically impossible; grammar errors stay per row.
3. **Tests**: `tests/test_scheduler.py` gains two tests.
   * `test_sample_rows_equal_per_row`: mixed settings and dtypes, over steps.
   * `test_grouped_sampling_rows_equal_solo`: 3 seeded card-sampler rows + 2 greedy + 1
     penalized, batched == solo.
   * CPU suite: **56 passed**.

## Before / after

### Steady-state step, in-run A/B (median ms per step, the same batch alternating every 12–16 steps)

| load | B | before | after | Δ |
|---|---:|---:|---:|---:|
| in-process handlers, greedy, ~1K context | 8 | 48.6 | 44.8 | −7.8 % |
| | 16 | 65.5 | 61.1 | −6.8 % |
| gRPC + gateway (bench c=8/16), greedy, run 1 (token read only) | 8 | 49.8 | 44.0 | −11.8 % |
| | 16 | 82.1 | 76.4 | −6.9 % |
| gRPC + gateway, run 2 (both changes, `oldall` vs `new`) | 8 | 60.1 | 54.6 | −9.2 % |
| | 16 | 89.9 | 85.0 | −5.4 % |
| in-process, card sampler (grouped draw only) | 8 | 44.0 | 40.5 | −8.0 % |
| | 16 | 71.3 | 58.6 | **−17.8 %** |

Absolute values differ between rows of the table: the machine state drifted between runs.

### bench_concurrency

Command: `--concurrency 8,16 --max-tokens 512 --repeat 1 --probe 64 --prompt-tokens 600`, through
the gateway on :8090, ignore_eos, greedy. Run order: after₁, before₁, before₂, after₂ (ABBA).
"before" = the in-process server with the old step code (`--ab oldall`); "after" = `worker.py`.

| run | c | agg tok/s | TTFT mean s | lat mean s | decode tok/s/req | probe tok/s |
|---|---:|---:|---:|---:|---:|---:|
| after₁ | 8 | 91.1 | 10.93 | 44.2 | 15.7 | 36.8 |
| before₁ | 8 | 76.2 | 12.78 | 52.9 | 13.0 | 34.6 |
| before₂ | 8 | 84.0 | 11.43 | 48.0 | 14.3 | 36.3 |
| after₂ | 8 | 95.1 | 10.38 | 42.3 | 16.4 | 37.1 |
| after₁ | 16 | 87.5 | 23.28 | 89.3 | 8.0 | 35.5 |
| before₁ | 16 | 88.1 | 22.97 | 89.2 | 8.0 | 34.8 |
| before₂ | 16 | 91.7 | 22.61 | 85.8 | 8.4 | 35.4 |
| after₂ | 16 | 116.0 | 17.68 | 67.6 | 10.6 | 36.7 |

* **c=8**: +16 % aggregate (93.1 vs 80.1 mean).
* **c=16**: +13 % (101.8 vs 89.9), but noisy. The c=16 level is dominated by serialized ~1.2–2 s
  prefills (arrival waves), and after₂ caught a fast machine state.

JSON: `results/prof/bench_{before,after}_{1,2}.json`.

## Verification

* **Identity**: corpus 15/15 byte-identical to `bgpu/corpus_old.json` at `QWEN_MAX_BATCH=1` and at the
  defaults; seq16 16/16 identical (`results/prof/gpu/`).
* **Agreement** (`agree.py`), unchanged from the mrqmm baseline:
  * B=8: 100.00 %, KL mean 8.65e-6;
  * B=16: 99.71 % (2042/2048), KL mean 9.70e-6.
* **conc 16** vs solo: 8/16 identical, first divergence at 254–702 characters. These are the same
  positions as the mrqmm run (the same near-tie class).
* **Feature suites**:
  * stops, think, mixedmax: PASS.
  * grammar: j0/j1/j3 valid; j2 is the known "Aliens" tag list that hits max_tokens 500. The
    suite's strict check prints FAIL for it, exactly as in the qmm and mrqmm runs.

## Tried and rejected

* **Larger Metal command buffers** (`MLX_MAX_OPS_PER_BUFFER=200 MLX_MAX_MB_PER_BUFFER=200`)
  * They give the step-ahead room. With them the slack probe shows +4 ms of host time per step
    costs nothing, against +2–5 ms at the defaults.
  * Decode gained −5 % at B=8 and −1.5 % at B=16 (process ABAB, noisy).
  * But prefill peak memory rose: +2.0 GB at 8K, +3.2 GB at 32K (16.6 → 19.8 GB), and 32K was
    slower. MLX keeps every input of a command buffer alive until the buffer completes.
  * Unsafe for the 35B at long context. Not adopted.
  * The ops limit alone (`MLX_MAX_OPS_PER_BUFFER=200`) leaves the peak unchanged. Its decode gain
    was not measured; prefill time was inconclusive.
* **One batched greedy argmax**: no gain (greedy is 0.5 ms of GPU at B=16).

## Remaining gap / open items

* **Host gap at B=16.** The slack probe at the defaults still shows that each ms of host time
  between steps adds ~1 ms at B=16. The cause is MLX's 10-buffer lookahead, and the batch has
  many small per-row attention buffers.
  * What is left on the host: ~4 ms of Python build + consume per step.
  * Levers: a leaner per-row KV update (≈1 ms), or fewer kernels per row (see next item).
* **Per-row decode attention** is 16–27 % of the step at 8K (B=8/16), about 2× its bandwidth
  floor. A multi-row decode-attention Metal kernel that reads each row's own KV buffer would cut
  this and the kernel count.
* **Serialized prefill dominates c=16 end-to-end.** The fix there is the V2 piggyback design.
