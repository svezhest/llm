# Worker changes: continuous batching (src/worker/worker.py)

Status 2026-10-11: implemented M0–M4 of `design/batching.md` in the worker. CPU tests
pass (34). GPU verification done on the 9B (results in `results/batching_gpu.md`, §7 below);
35B / 27B smoke tests in the same file.

## 1. Files

| file | change |
|---|---|
| `src/worker/worker.py` | `_Scheduler`, `_Req`, `_Row`, `RowsKV` + `_rows_attention`, GDN stacking helpers, memory formulas. The handler is rewritten (`Engine.generate`), and `Engine._run` / `_run_spec` move into the scheduler (`_prefill`, `_spec`, `_iterate`, `_step`, `_sample`). `_Control` is split in two halves. Also: `materialize_images` cleanup, `_prepare` → `_prepare_vision`, `Health.max_batch`, `Usage.queue_ms` / `batch_mean`, gRPC pool size, the A1/A2 RoPE fixes. |
| `src/worker/tests/` (new) | `conftest.py` (tiny random qwen3_5 / qwen3_5_moe models on the CPU), `test_rows.py`, `test_scheduler.py`, `test_handler.py`: 33 tests, ~7 s. |
| `src/worker/pyproject.toml`, `uv.lock` | new `dev` dependency group with `pytest>=9.1.1` (lock adds pytest, pluggy, iniconfig only). Installed with `uv sync --inexact`, so nothing else in the venv changed. |
| `src/proto/worker.proto` | not touched by me (the parent added `max_batch`, `queue_ms`, `batch_mean`). |

The pre-change worker is saved at `scratchpad/orig_worker/worker.py`. It is the reference for the identity tests.

Run the tests:
```
cd src/worker && .venv/bin/python -m pytest tests -q      # or: uv run python -m pytest tests -q
```

## 2. Architecture

* **Threads.**
  * The gRPC handler thread (`Engine.generate`) does the following, all without MLX:
    * render the template;
    * tokenize (text);
    * `_limit`;
    * build `_Control` (the bad words are tokenized here), `_Guide` (llguidance matcher) and `ThinkingBudgetCriteria`;
    * submit a `_Req`;
    * turn the streamed token ids into events: detokenizer, `_GenState`, `_StopFilter`, tool parsing.
  * The `_Scheduler` thread is the only thread that creates or evaluates MLX arrays after load:
    * vision `prepare_inputs`;
    * `_Penalties`, samplers, `_Control.build()` masks;
    * prompt-cache lookup / store;
    * prefill;
    * decode.
* **Scheduling.** One `tick()` makes one decision:
  1. drop cancelled waiting requests;
  2. admit the queue head if it fits;
  3. advance the current prefill by one slab, if decode's time share is used up or there are no rows;
  4. otherwise run one decode step for all rows.
* **Decode at B=1.** A row decoding alone runs the plain library call on its own caches, with an explicit position:
  ```
  lm(y[None], cache=row.cache, position_ids=[[pos + delta]])
  ```
* **Decode at B≥2.** One forward of `(B,1)`:
  * a `RowsKV` (the rows' own KVCache / QuantizedKVCache) per full-attention layer;
  * the stacked GDN state per linear layer;
  * `position_ids (B,1)`.
* **Attention override.** `Qwen3_5Attention.__call__` is overridden once at load (this also covers the MoE model, which uses the same class). It acts only for a `RowsKV` cache:
  * the projections, norms and RoPE run batched;
  * the cache update and SDPA run per row, through the same calls as B=1.

  A model that is not qwen3_5 gets `max_batch=1` and a warning.
* **Per-row post-processing.** The order is the one `_run` used: penalties → `_Control` mask → `maybe_quantize_kv_cache(row.cache)` → guide / sampler. Each row keeps its own sampler key, so seeds stay per request.
* **Step-ahead (`async_eval`)** is kept: the next forward is queued before the current tokens are read. A batch switches to **sync mode** while it holds any row with a grammar (`_Guide`) or a multi-token bad word (`ctl.sync`). In sync mode it reads, consumes, then steps.
* **Thinking budget.** The forced `\n</think>` token replaces that row's next input (`apply_forced_token`), exactly as before.
* **Join and leave.**
  * 1→2 rows: the GDN state is stacked. A row's own `ArraysCache` is emptied at that point, so no stale copy survives.
  * Rows leave by a gather. 2→1: the survivor gets its state slices back and returns to the plain call.
  * A 1-token prompt (no state yet) gets the library's zero state.
  * Before every batched step the code checks that the stacked state's batch size matches the row count. The library would otherwise silently reset to zeros on a mismatch.
* **Prefill.**
  * One request at a time.
  * Today's slab loop runs unchanged: same `_bounds` cuts, quantize, checkpoints, snapshot, `clear_cache` per slab.
  * It yields after each slab except the last.
  * With rows running, decode gets `t·(1-s)/s` seconds after a slab that took `t` seconds (`QWEN_BATCH_PREFILL_SHARE=s`).

## 3. New env knobs

| var | default | meaning |
|---|---|---|
| `QWEN_MAX_BATCH` | `16` | Rows decoded together. `1` = the old one-at-a-time behaviour, run through the new code. Health reports the effective value: 1 for a non-qwen3_5 model, and 1 (with a warning) when the budget cannot hold two typical rows (2K prompt + `QWEN_BATCH_RESERVE_TOKENS`). Memory admission limits it further per request. **Measured:** MLX on M1 runs a per-row qmv below 10 rows for the 4096-wide projections (below 6 for wider layers), then a flat-cost qmm. 9B ms/step: 41 / 51 / 90 / 127 / 133 at B = 1 / 2 / 4 / 8 / 16, so 16 rows cost about the same as 8, and 16 measured ≥ 8 at every concurrency. The 35B pays off only from ~16 concurrent requests. (Round 1 shipped 8.) |
| `QWEN_BATCH_PREFILL_SHARE` | `0.5` | Prefill's share of the time **when the batch is full** (0.05–1). With B of max_batch rows decoding, prefill gets `s_B = 1 − (1−s)·B/max_batch` (a lone row yields ~94% to a waiting prompt at max 8). At 1 the prompt prefills straight through. Changed after the GPU memory test: with a fixed 50/50 split and 8K-token prompts, one decoding row took half the GPU from the next prefill and the batch never formed (batch_mean 1.0, slower than serial). |
| `QWEN_BATCH_MEM_GB` | `0` (derived) | Admission budget (GiB) for the running rows' KV + recurrent state. Derived (`_batch_budget`) as: `cap − active memory after load (weights + drafter) − prefill reserve − 1 GiB − QWEN_PROMPT_CACHE_GB (if the cache is on) − QWEN_CACHE_LIMIT_GB (if QWEN_CLEAR_CACHE=0)`, never below 0. The terms: <br>• `cap` = `QWEN_WIRED_LIMIT_GB` (clamped to the kernel max), or the device's recommended working set: macOS reports the `iogpu.wired_limit_mb` sysctl there once it is raised (28 GiB here), else its default share of RAM; <br>• prefill reserve = `QWEN_PREFILL_BUDGET × q_heads × 2 B + 0.4 GiB` (≈2.4 GiB for the 9B and 35B: 16 q heads). <br>The headroom outside the budget (prefill reserve + 1 GiB) also absorbs rows that outgrow their reservation. Logged at load. |
| `QWEN_GEN_SLOTS` | `max(2·max_batch, 8)` | Generate RPCs accepted at once, running or waiting. One more is refused with `RESOURCE_EXHAUSTED`, which the gateway answers with 503. The gRPC pool is slots + 8 threads, so Check / Cancel / Health always run. `maximum_concurrent_rpcs` = slots + 16 is only a backstop. (Replaces round 1's `QWEN_GRPC_THREADS`.) |
| `QWEN_BATCH_RESERVE_TOKENS` | `4096` | Generation tokens a row reserves at admission: `prompt + min(max_tokens, this)`. Rows are re-accounted at their real length every step, in 1024-token increments. A row that outgrows its reservation keeps running, but while the batch is over budget nothing new is admitted. |

Unchanged knobs keep their meaning: `QWEN_KV_BITS`, `QWEN_PREFILL_STEP`/`_BUDGET`, `QWEN_PROMPT_CACHE*`, `QWEN_CLEAR_CACHE`, `QWEN_CACHE_LIMIT_GB`, `QWEN_DRAFT*`, `QWEN_JSON_MAX_WS`, and the sampler defaults.

## 4. Behaviour notes

* **Admission.**
  * The queue is FIFO with no overtaking.
  * The head is admitted when all of these hold:
    * `rows < max_batch`;
    * `Σreserved + reserve(head) ≤ budget`;
    * `active memory + reserve(head) + prefill reserve ≤ cap` (a live guard against estimate errors).
  * An **empty batch always admits** the head. That is the old behaviour: `MAX_CONTEXT` validation bounds a lone request.
  * A request that does not fit waits in the worker queue. It already holds its gateway slot, so it is never a 503 from the worker. It never OOMs the batch.
* **Reservation.**
  * `reserve(r) = 1.05 · kv(prompt + max_tokens) + 2 · recurrent state + 8 MB`. The state counts twice because the step-ahead holds the queued step's output state beside its input (GPU: 8 rows × 8.3K tokens reserved 2.77 GB with 1× state but used ~3.9 GB above the weights; 2× gives ~3.6 GB).
  * `kv(n)` = `n·fp16` up to `quantized_kv_start` (5000), else `5000·fp16 + n·q`. The first term is the transient fp16 copy while `to_quantized` runs.
  * Per token per row: 9B fp16 32 KiB; 27B q8 34 KiB; 35B q4 5.6 KiB.
  * Recurrent state per row: 9B 51.5 MB, 27B 154 MB, 35B 64 MB. These are unit-tested against real cache `nbytes`.
* **Requests without `max_tokens`** (round 2) reserve only `prompt + QWEN_BATCH_RESERVE_TOKENS`, not the rest of the 128K context, so they batch like any other request. Their real length is re-accounted every step.
  * **Trade-off:** rows admitted with a short reservation can grow together past the budget.
    * Growth is allowed, and further admissions wait meanwhile.
    * The headroom kept outside the budget (prefill reserve + 1 GiB) and the live memory check at admission are what prevent an OOM.
    * An OOM is still possible only if the running rows together outgrow that headroom. As an example on the 9B: 16 rows that each run 4K tokens past their reservation are ≈2 GiB of fp16 KV.
  * `GATEWAY_DEFAULT_MAX_TOKENS` still bounds such a row's length.
  * Such a request reserves ~4 GB of fp16 KV on the 9B. In practice it runs alone or nearly alone.
  * Set `GATEWAY_DEFAULT_MAX_TOKENS` (e.g. 4096–8192) to batch clients that omit `max_tokens`.
  * The gateway sends that value in `Params.default_max_tokens`, which `_limit` already uses.
* **Exactness.**
  * A request that decodes alone (B=1) runs today's library call on its own caches. The explicit position is bit-identical to the computed one (CPU-tested on dense and MoE). Prefill and post-processing are the same code in the same order, even when its prefill overlapped other rows' decode.
  * At B>1 a row differs from solo only by the kernels' row-count-dependent rounding (qmv vs qmm at ≥6 rows; MoE sorted gather at B·top_k ≥ 64). This is the same class as the DFlash verify noise. Under load, greedy answers can therefore differ run to run; `QWEN_MAX_BATCH=1` restores determinism.
  * On the CPU (fp32) batched decode equals solo to ~1e-6 and the token streams are identical in every test.
* **DFlash rule (simplest correct).**
  * Speculative decoding runs only for an eligible request (greedy or `QWEN_DRAFT_SAMPLING`, text, no thinking budget, no grammar or token options) that is admitted with **nothing else running or waiting**.
  * When another request arrives and `max_batch > 1`, the DFlash row is **demoted at the next round boundary**:
    * the target cache then holds every token before the last emitted one, `b`;
    * the row becomes a normal batch row whose next input is `b`;
    * `b` is already emitted and counted (`first=True`);
    * it is never re-promoted.
  * With `QWEN_MAX_BATCH=1` it is never demoted, which is the old behaviour.
  * CPU-tested with a fake drafter: the output equals plain greedy, both solo and with demotion mid-answer.
* **Vision.**
  * Round 2: the image work runs on the **handler thread** (`Engine._vision_inputs`), so one slow image URL or a large decode cannot stall the decoding batch.
    * Images are loaded (path, URL or data URI, converted to RGB) and run through the processor: mlx-vlm's `prepare_inputs` minus MLX. The qwen3-vl processor is numpy throughout and only wraps its results as constant arrays, which come back to numpy.
    * The scheduler converts them exactly as `prepare_inputs` does. The output was tested equal to `prepare_inputs` with the real ornith10 processor.
    * The exact prompt length is now known before queueing, so `context_length_exceeded` arrives before the queue. Errors are `prompt_count_failed` or `context_length_exceeded`, as before.
  * Prefill is serialized like any other.
  * Decode is batched. The row's mRoPE delta is captured after its prefill; positions are `pos + delta`.
  * CPU-tested with a synthetic image layout: batched == solo, and multi-slab == one slab.
* **Grammar rows.** `response_format` and constrained tools are batched in sync mode. The guide masks each row separately (`_Guide.sample`, same as B=1).
* **Errors.**
  * An exception while setting up, sampling or consuming a row (for example "grammar: token rejected") fails that row only. The batch continues.
  * An exception in the batched forward or read fails every row in the batch. The scheduler then resets and goes on with the queue.
  * A failed prefill fails only that request.
* **Cancel.**
  * Sources: the Cancel RPC; `context.add_callback` (a dropped stream frees its row even without the gateway's Cancel); the handler's `finally`; a stop-string hit (the handler sets cancel so the scheduler evicts the row at its next step).
  * Where it takes effect: a waiting request is dropped at the next tick; a prefill between slabs; a decode row at the next step.
* **Events.**
  * `Start` is now sent at once, before queueing. The gateway only reads `model` from it.
  * `Usage.ttft_ms` and `tps` are measured from the start of prefill, as before. The worker-side wait is the new `Usage.queue_ms` (arrival at the worker → prefill start).
  * `Usage.batch_mean` is the mean batch size over the decode steps the request took part in; DFlash rounds count as 1.
* **Memory reporting.**
  * `Usage.peak_gb` and the log's `peak=` / `active=` / `pool=` are captured **on the scheduler when the request ends**, before the next admission can reset the counter (round 2).
    * The value is the MLX peak since the batch was last empty. The peak is reset when a request is admitted to an empty batch.
    * With concurrent rows this is the **process** peak during the request, not the request's own share. For serialized traffic it equals the old per-request value.
    * After a stop-string hit or a cancel, the handler waits for the row's eviction (at most one step) so that the numbers are the scheduler's.
    * `Usage.batch_mean` is taken at the same moment.
  * `wired_limit` is entered once per busy period and left when the scheduler goes idle.
  * `mx.clear_cache()` runs: after every prefill slab; when a row finishes; every 256 batch steps; at idle.
* **Logs.**
  * Per request, as before, plus `queue_ms=` and `batch=`.
  * A scheduler line every 10 s while busy:
    ```
    batch: rows= waiting= prefilling= steps/s= mean_rows= reserved=x/budget GB active=
    ```
  * At load:
    ```
    batching: max_batch, KV budget, cap, prefill reserve, KV B/token, recurrent state MB/row, prefill share, gRPC threads
    ```
* **Out-of-vocab ids** (added on the coordinator's request): `_render` (shared by Check and Generate) rejects `allowed_token_ids` / `stop_token_ids` ≥ the logits width with `invalid_request` (400, before queueing): they index vocab-sized masks and MLX indexing has no bounds check. Bad-word ids come from the tokenizer and are always in range. Tested.
* **Log line** order: `... cached=N peak=..GB active=..GB pool=..GB queue_ms=N batch=X`. The new fields are appended so `scripts/bench_concurrency.py`'s parser keeps matching; its regex now also reads them (optional group).
* **Streaming of space-free output** (pre-existing, seen on GPU): mlx-vlm's BPE detokenizer flushes a word only at the next token that starts with a space, so compact JSON arrives in one piece at the end (client TTFT 15 s for one schema row whose worker TTFT was 0.46 s). Unchanged from the old worker.
* **ultrareview fixes (worker side).**
  * **#4.** `materialize_images` unlinks the files it already wrote when a later image fails. The path is also recorded before the write, so a failing `b64decode` cannot leak its own file. Tested.
  * **#2.** The bad words' `tokenizer.encode` runs on the handler thread. The vocab masks are built on the scheduler at admission; this is cheap, and the gateway caps the list sizes. Nothing sits under a global lock any more.
* **Appendix A bugs: both confirmed on CPU by logic and fixed.** Both are deterministic position bugs, independent of the device. The GPU script measures their size on the real models.
  * **A1** (DFlash first target call after prefill).
    * With `_rope_deltas=None` / `_position_ids=None`, the last prompt token was rotated as **position 0**. Tiny model: identical to explicit position 0; |Δlogit| 0.27–0.31 vs P−1.
    * Fix: `_spec` sets `_rope_deltas = zeros`, `_position_ids = None`.
    * Regression test: the demoted row's prompt keys must equal a plain prefill's (2.63 without the fix, <1e-4 with it).
    * **DFlash outputs therefore differ from the old worker.**
  * **A2** (vision prompt longer than one slab).
    * From slab 2 on, image tokens got `offset + delta`. Tiny model: |Δlogit| 0.05–0.08 vs one slab; 1e-6 with the fix.
    * Fix: vision slabs pass `position_ids = _position_ids[:, :, pos:pos+n]`.
    * Regression test in `test_vision_rows`.
    * **Vision prompts longer than one slab (2048, or 512 with `QWEN_PREFILL_STEP=512`) therefore differ from the old worker.**

## 5. CPU verification done (no lease)

* **`pytest tests`: 33 passed.**
  * `test_rows.py`:
    * RowsKV batched logits == per-row solo for dense and MoE, fp32 / fp16 activations, 8/4-bit weights, fp16 / q8 / q4 KV with the KV crossing `quantized_kv_start` mid-batch;
    * join, leave, split back to B=1;
    * a 1-token prompt;
    * a non-zero mRoPE delta row;
    * explicit == computed positions (bit-exact);
    * KV / recurrent bytes vs real `nbytes`, and the reservation formula across 5000.
  * `test_scheduler.py`, on fp32 tiny models (dense + MoE). Every request's token stream is identical whether it runs alone (`max_batch=1`) or batched with staggered arrivals, for:
    * greedy, seeded sampling with top_p/top_k, presence / frequency / repetition penalties;
    * max_tokens 0 / 1 / 17;
    * a 1-token prompt;
    * q8 / q4 KV crossing the start mid-batch;
    * stop ids, ignore_eos (shown=False), min_tokens, allowed_token_ids, bad words;
    * sync-mode rows, plus a row whose grammar fails (error isolation);
    * thinking-budget forced tokens;
    * time-shared prefill (share 0.3).

    Also covered in this file:
    * cancel while waiting, mid-prefill and mid-decode; reservations return to 0;
    * admission: the max_batch cap, budget refusal, FIFO with no overtaking, the empty batch always admitting, the live memory guard;
    * stats: steps, batch_mean, queue_ms;
    * prompt-cache hits while batching equal a cold run;
    * vision rows (multi-slab == one slab; delta row in a batch == solo);
    * DFlash with a fake drafter (exact, demotion, A1 keys);
    * the real scheduler thread with concurrent submitters.
  * `test_handler.py`: the real `Engine.generate` on a scripted scheduler stream:
    * reasoning split and usage;
    * a stop string split across tokens (it also evicts the row);
    * ignored EOS, min_tokens and stop strings;
    * cancel and error events;
    * tool-call parsing;
    * a context-length reject before queueing;
    * `_StopFilter`, the `_Control` halves, `materialize_images` cleanup.
* **CPU end-to-end** (`scratchpad/gpu_tests/cpu_worker.py`).
  * Setup: the real worker process (gRPC server, handler, scheduler thread, real Ornith tokenizer, chat template, llguidance vocab) with a tiny random 8-bit language model of the real vocabulary.
  * **Old worker vs new worker, the 15-case identity corpus: 15/15 byte-identical.** The corpus covers greedy, seeded, penalties, thinking ± budget, stop strings, min_tokens + ignore_eos, bad words, json and tools. json and tools-required fail identically on the CPU (`llguidance.mlx` is Metal-only).
  * Harness suites against the new worker, all PASS: conc 8, cancel (2 Cancel RPC + 1 dropped stream), stops, mixed max_tokens, mixed thinking + budget.
  * pcache: 7/7 HITs while 7 conversations ran batched.
  * `agree.py` and `rope_checks.py a1` smoke-ran on the CPU too.

## 6. GPU test matrix (to run with the lease)

All commands run from `src/worker`. `SP` is the scratchpad. The worker listens on 127.0.0.1:50051. It is stopped with plain `kill`, and the next run waits for `worker serving`. `W` is shorthand for the harness:
```
SP=<scratchpad>
W="PYTHONPATH=. .venv/bin/python $SP/gpu_tests/wtest.py"
NINE="QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0"
```
The user gateways on :8080/:8081 reconnect to :50051. My request ids are `corpus-*`, `seq-*`, `p*`, `c*`, `m*`, so any foreign request id in the worker log shows interference.

1. **Reference outputs from the old worker** (before any new-worker run):
   ```
   env $NINE PYTHONPATH=. uv run python $SP/orig_worker/worker.py > $SP/results/w_old.log 2>&1 &
   $W corpus --out $SP/results/corpus_old.json
   $W seq --n 16 --max-tokens 256 --out $SP/results/seq16_old.json
   ```
2. **B=1 identity, new worker:**
   * `QWEN_MAX_BATCH=1`: run `corpus` → `corpus_mb1.json` and `seq --n 16` → `seq16_mb1.json`.
   * Default (8): repeat both, giving `corpus_mb8.json` and `seq16_mb8.json`.
   * Then compare:
     ```
     $W compare corpus_old.json corpus_mb1.json
     ```
     and the same for `corpus_mb8`. Expect byte-identical. All corpus prompts are shorter than one slab and none uses DFlash, so the A1/A2 fixes do not apply.
3. **Agreement under batching:**
   * `$W conc --n {2,4,8} --ref seq16_mb8.json` (and `--n 16` with `QWEN_MAX_BATCH=16`). Reports aggregate tok/s, TTFT, tps, queue_ms, batch_mean, identical answers and first divergence.
   * Teacher-forced: `env $NINE PYTHONPATH=. uv run python $SP/gpu_tests/agree.py` (B = 2/4/8/16). Reports top-1 agreement and KL(solo‖batched). Expect ≥99% and ≤1e-3.
4. **Cancel mid-batch:** `$W cancel --ref seq16_mb8.json`. Three rows are cancelled; the others finish; Health is ok. Then check the worker log for `active=` back at its baseline.
5. **Stops, mixed max_tokens (1/7/64/512), mixed thinking + budget 32, grammar (4 JSON + 4 free):** `$W stops`, `$W mixedmax`, `$W think`, `$W grammar`. Each prints PASS or FAIL. For grammar, compare the free rows' tps with the conc run to get the sync-mode cost.
6. **Prompt cache while batching:** restart with `QWEN_PROMPT_CACHE=1`, then `$W pcache`. Expect 7 HIT lines, and TTFT ~0.2 s vs cold.
7. **Memory at B=8 with ~8K-token prompts:**
   * `$W memory --n 8`: peak and active from Usage and the log.
   * Then restart with `QWEN_BATCH_MEM_GB=1.5` and repeat: rows must wait (log `waiting=`), with no OOM and no 503.
8. **RoPE fixes, magnitude on the real models** (both already fixed in the worker):
   ```
   env $NINE PYTHONPATH=. uv run python $SP/gpu_tests/rope_checks.py a1
   QWEN_MODEL=../../weights/ornith10-9b-mlx-8bit QWEN_KV_BITS=0 PYTHONPATH=. uv run python $SP/gpu_tests/rope_checks.py a2
   ```
9. **DFlash:**
   * Restart with `QWEN_DRAFT=../../weights/qwen35-9b-dflash`.
   * Run `corpus` and compare the code0–4 answers with `corpus_mb8.json`. Expect identical up to fp noise; the A1 fix moves DFlash closer to the plain loop.
   * Run `$W conc --n 2`: the log must show `demoted`, and both answers must be correct.
10. **503 overflow** (needs the gateway agent's build on :8090 with `MAX_QUEUE=4 GATEWAY_MAX_INFLIGHT=2`): 10 concurrent curl requests. Exactly 6 get 503 and the rest complete.
11. **27B / 35B smoke:**
    * 27B: `QWEN_KV_BITS=8 QWEN_WIRED_LIMIT_GB=27` plus `sudo sysctl iogpu.wired_limit_mb=28672` (ask the user).
    * 35B: `QWEN_KV_BITS=4 QWEN_WIRED_LIMIT_GB=27`.
    * For each: `corpus` (identity vs the old worker on that model, if time allows) and `conc --n 4`, then read the KV budget and admission in the log.

Results go to `scratchpad/results/batching_gpu.md` with the exact env per run.

## 7. GPU results (measured 2026-10-11; full tables in `results/batching_gpu.md`)

* **Identity at B=1** vs the pre-change worker: byte-identical.
  * 9B, `QWEN_MAX_BATCH=1`: 15/15 corpus cases and 16/16 sequential prompts.
  * 9B, `QWEN_MAX_BATCH=8` with sequential requests: 15/15.
  * 35B q4: 15/15 + 4/4.
  * 27B q8 (`QWEN_PREFILL_STEP=512`, wired 28 GB): 15/15 + 4/4.
* **Teacher-forced agreement, 9B:**
  * B=2 and B=4 bit-identical;
  * B=8 100% top-1, KL 8.5e-6;
  * B=16 99.8% top-1, KL 1.1e-5.
* **35B:** 4 concurrent answers were byte-identical to solo.
* **Decode ceiling, 9B** (the microbench equals the worker, so the scheduler overhead is not measurable): ×1.62 / 1.82 / 2.59 / 4.95 at B = 2 / 4 / 8 / 16.
* **Through the gateway**, 600-token prompts:
  * Old worker: 22–25.5 tok/s at any concurrency.
  * New, `QWEN_MAX_BATCH=8`: 29 / 32 / 40 tok/s at c = 2 / 4 / 8.
  * New, `QWEN_MAX_BATCH=16`: 56 tok/s at c=16 (agg/probe 2.57).
  * With 1024-token answers: c=8 46 tok/s (MB=8); c=16 **81 tok/s** (MB=16, agg/probe 3.37).
  * c=1: no regression (agg/probe 0.86–1.00, against 0.79–1.14 for the old worker).
* **What caps the gain at 256 tokens:** prefill waves. Arrivals synchronize, and their prefills (1.2–1.7 s each) run back to back. The next lever is the design's V2 piggyback of decode on prefill slabs.
* **Feature checks in a batch, all pass:** cancel (RPC and dropped stream), stops, mixed max_tokens, mixed thinking + budget, json_schema rows + free rows, 503 overflow (exactly 6 of 10), prompt cache (7/7 HITs), DFlash (solo exact, demotion at a round boundary).
* **Memory, 8 rows × 8K tokens:** peak 16.15 GB with 11.6 GB of weights. A small `QWEN_BATCH_MEM_GB` makes requests wait with no OOM.
* **Appendix A on the real models:**
  * A1 (DFlash): |Δlogit| 0.88, fixed exactly.
  * A2 (vision prompt over several slabs): 0.30, down to 0.05 (fp16 slab noise) with the fix.
* **Changed after the GPU runs:**
  * the prefill share scales with batch occupancy;
  * the reservation counts the recurrent state twice;
  * Check rejects out-of-vocab ids;
  * the log field order changed;
  * the `QWEN_MAX_BATCH` rationale was corrected.
* **Left running:** the new worker (9B, `QWEN_MAX_BATCH=8`) on :50051 and the gateway on :8090, both at nice 0.

## 8. Round 2: review fixes (`results/review_worker.md`)

| finding | change |
|---|---|
| M1 gRPC capacity | `QWEN_GEN_SLOTS` (default `max(2·max_batch, 8)`) bounded semaphore in `Generate`. Over the limit: `RESOURCE_EXHAUSTED` (gateway 503), logged. Pool = slots + 8 threads; `maximum_concurrent_rpcs` = slots + 16 as a backstop. The server lives in `serve()`. |
| M2 image work on the MLX thread | `Engine._vision_inputs` on the handler thread (load, decode, resize, processor → numpy). The scheduler only wraps arrays and runs the vision tower and embeddings in prefill. |
| M3 peak race | Stats snapshot in `_Scheduler._end` on the scheduler thread; `_done` and the log read `r.peak/active/pool/batch_mean`. |
| M4 robustness | A failed decode step fails its rows only. `SCHED_MAX_FAILURES` (3) consecutive failed steps, or an exception escaping the loop, close the scheduler: everything queued or running gets `Error(worker_fault)`, `submit` refuses at once, and Health reports `ready=false`. `alive()` = thread alive, not closed, and no tick stuck for `SCHED_STALL_S` (300 s) while busy. Handlers wait with a poll (`EVENT_POLL_S`, 5 s) and end the stream with `Error(worker_fault)` if the scheduler is not alive. `_idle` clears `self.wired` before `__exit__`. The scheduler thread **never returns**: when it does, MLX tears down its thread-local stream, and that segfaulted the process in a test. After closing, it parks. |
| L1 failed join | `_gdn_stack` builds every layer before emptying the rows' caches, and `_join` swaps the new stacked state in only on success. Tested: a failed stack fails the new row only, and the running rows' answers equal solo. |
| L2 collateral prefill failure | A step failure no longer touches the prefilling request (tested). |
| L3 Shutdown | `_Scheduler.stop()`: after the current tick everything left gets `Error(shutdown)`, followed by `mx.synchronize`. `main` calls it before `server.stop`. |
| L4 zero budget | The budget is never negative. The effective max_batch drops to 1 (with a warning) when the budget holds fewer than two typical rows, and Health reports it. |
| L5 pool not in budget | With `QWEN_CLEAR_CACHE=0` the budget subtracts `QWEN_CACHE_LIMIT_GB`, and the live guard adds `mx.get_cache_memory()`. |
| L6 DFlash demotion | Demote only when the waiting head is not cancelled and would be admitted next to the row (`_fits(head, 1)`). Tested: a head over budget leaves DFlash running to the end. |
| L7 `_unregister` | Removes the entry only if it is still the same Event. |
| L8 reservation | `QWEN_BATCH_RESERVE_TOKENS` (4096), per-step re-accounting, growth blocks admissions (tested; trade-off in §4). |
| nits | <ul><li>Pool clearing: when a row ends, only if the batch is now empty or the row was long (> 8K tokens); also every 256 steps and after every prefill slab, as before.</li><li>Units: every `*_GB` knob and the batching logs are GiB; `Usage.peak_gb` and the request log stay 1e9 (parser compatibility).</li><li>`_bytes` uses `>=` for the quantization start.</li><li>`batch_mean` is a snapshot.</li><li>Check refuses out-of-vocab ids (from round 1's GPU phase).</li></ul> |
| default | `QWEN_MAX_BATCH=16`. |

**Tests:** 47 in `src/worker/tests/`, ~8 s on the CPU. New in this round:
* `test_robustness.py`:
  * a failed decode step fails its rows only, while a mid-prefill request and later requests complete equal to solo;
  * give-up after consecutive failures (worker_fault to all, later requests refused, `alive()` false);
  * Shutdown;
  * gRPC slots (RESOURCE_EXHAUSTED past the slots, Check / Health under 1 s while all slots stream, slots released);
  * budget derivation and the effective max_batch;
  * reservation growth blocks admission;
  * the peak snapshot;
  * handler ↔ real scheduler thread: a stop string evicts the row, and its stats come from the scheduler;
  * the handler on a dead scheduler;
  * a failed join;
  * handler-side vision inputs equal `prepare_inputs` with the real processor files.
* `test_scheduler.py`:
  * thinking budget + grammar on one row in a batch;
  * DFlash not demoted for an inadmissible head.

**To re-verify on the GPU** (the lease is needed):
1. The identity corpus at `QWEN_MAX_BATCH=1` and at 16 (sequential requests) vs the old worker.
2. Feature suites: cancel, stops, mixedmax, think, grammar, pcache, DFlash demotion.
3. A vision request: its output equals round 1 (the image path moved threads), and an image URL during a running batch must not stall it.
4. The 503 path through the new gateway on :8090: `MAX_QUEUE` overflow, and RESOURCE_EXHAUSTED past `QWEN_GEN_SLOTS`.
5. `bench_concurrency` c=1..16 at the new default.
6. The memory test (8 rows × 8K tokens), with the 4096-token reservation cap.

**GPU re-verification of round 2 (2026-10-12): all PASS.** Details in `results/batching_gpu.md` §9.
* **Identity:** 15/15 at max_batch 1 and at 16; 16/16 sequential prompts.
* **Feature suites:** all pass. In a batch of 4 the grammar rows were byte-identical to solo.
* **Vision (ornith10):** old vs new 2/2 identical. A 6 s image URL did not stall the batch (largest gap 2.4 s, from the vision prefill).
* **503 paths:** the gateway queue gave 4 × 200 / 6 × 503; the worker slots (`QWEN_GEN_SLOTS=3`) gave 3 × 200 / 7 × 503.
* **Memory, 8 × 8K tokens:** peak 17.47 GB. Without max_tokens all 8 batched under the 4096-token reservation cap, reserving 3.95 GiB against ~3.8 GiB used.
* **No code change was needed.** Only the test client changed: the grammar cap went from 200 to 500, and a `memory-nomax` suite was added.
* **Throughput bench:** deferred until the GPU is quiet.

## 9. Round 3: second review (`results/review_worker_r2.md`), CPU only

| finding | change |
|---|---|
| H1 false stall after idle | `beat` is replaced by `tick_t0`, set when a tick starts and cleared when it ends. `alive()` is False only while one tick has run more than `SCHED_STALL_S`; idle never counts. A stuck tick does not close anything: it recovers when the tick ends. Test: `test_stall_detector` (idle 2× the limit → alive; a stuck tick → not alive, then alive again). |
| M1 growth past the headroom | `_regrow`: if re-accounting would push the batch past budget + prefill reserve while other rows run, the growing row ends with `length` (logged). A lone row is never cut. The last 1 GiB stays slack. Tests: `test_rows_past_the_headroom_are_cut` (headroom 0 → one row cut, the other runs to the end; ample headroom → no cut), and `test_reservation_grows_and_blocks_admission` now has a budget that fits 3 initial reservations but not 2 grown rows + 1. |
| M2 permanent give-up | Close after 3 consecutive failed steps with several rows (or with none); a lone row's failure is no strike. Then `on_fault`: the Engine exits with 70 after 1 s, after the errors went out, so launchd restarts it. Tests: `on_fault` called on give-up; `test_lone_row_failures_are_no_strikes`. |
| M3 tokenizer race | The image processor runs with `padding=False`. Values, dtypes and shapes are equal to `prepare_inputs`; the test now checks dtypes and runs the scheduler's own `_vision_arrays`. The handler no longer sets `pad_token`. |
| L1 doc | CLAUDE.md now says the handler builds constant MLX arrays through the processor's `to_mlx`. |
| L2 | `_close` calls `mx.synchronize(generation_stream)`. |
| L3 | SIGTERM → the same graceful stop as Shutdown / Ctrl-C. `stop()` returns False if the tick does not end within 10 s, and then `main` exits(1) instead of finalizing under MLX. |
| L4 | `maximum_concurrent_rpcs` = slots + 64; the docs no longer claim Check/Health can never be refused. `QWEN_GEN_SLOTS` is clamped to ≥ 1, with a warning when it is below max_batch. |
| L5 | A request whose cancel flag is set before the image work, or before submit, ends as cancelled without any image work or queueing (test). |
| cap | `_batch_cap()` (`QWEN_WIRED_LIMIT_GB` clamped to the device working set). Test: `test_cap_derivation`. |
| tests | <ul><li>`test_peak_snapshot_belongs_to_the_request`: a 1500-token request followed by an 8-token one; each keeps its own peak (`a.peak > b.peak`).</li><li>`test_failed_first_join_keeps_the_running_row`: `mx.concatenate` fails on layer 2 of a 1→2 stack, the joining row errors, and the running row's answer equals solo.</li><li>`test_grpc_slot_freed_by_a_client_cancel`.</li></ul> |
| docs | CLAUDE.md KV paragraph (35B q4 5.6 KiB/token ~0.7 GiB; the 9B runs fp16 32 KiB/token ~4 GiB, q8 would be 17 KiB ~2.1 GiB; 27B q8 34 KiB ~4.25 GiB), the same numbers in the worker.py comments and the README KV math, the MLX-thread wording, stall and give-up semantics, `_batch_budget` / `_batch_cap` names, the slot default, the `check` docstring, BACKLOG `Semaphore(1)`, the README recurrent state shown as 2× per row, the README Resilience note (exit 70 → KeepAlive; SIGTERM graceful), the test command (`--ignore tests/test_mrqmm.py`: test_mrqmm runs Metal kernels), and `QWEN_MRQMM` env rows (they were still missing). |

**Tests:** 54 pass, ~12 s (CPU, `--ignore tests/test_mrqmm.py`).

**Not done:**
* M1's refinement: choosing the longest or newest row to cut, and pausing a running prefill while over budget. The simple rule cuts the row that grows.
* A test of `main`'s Shutdown-before-`server.stop` ordering.
* Not run here: `test_mrqmm.py` (Metal) and any GPU re-check of these changes.
