# Review round 2: worker fixes (M1–M4, L1–L8), docs, test_robustness

Read-only review, 2026-10-12. Nothing was run: no pytest, no model, no builds.

**Method.** I read `src/worker/worker.py` against `results/review_worker.md` and `design/worker_changes.md` §8. I also read transformers 5.10.2 / mlx-vlm 0.6.2 source in the worker venv, the gateway's Health and error handling, and `deploy/launchd`.

**Moving target.** `worker.py` changed while I was reviewing (2839 → 2857 lines, mtime 02:01): `QWEN_MRQMM` / `mrqmm.install` were added at :570-578 and :2224-2229. I did not review that change. Line numbers below are from the current file.

## Ranked findings

### H1. High. After 5 minutes idle, the first request is killed with a false `worker_fault`, and Health flaps to not ready (worker.py:1438-1442, 1411-1415, 2403-2411)

**Cause.**
* `_run` sets `self.beat` just before `cv.wait()`.
* After `submit` wakes it, the loop leaves the wait without refreshing `beat`.
* `alive()` is `not (busy and now - beat > SCHED_STALL_S)`. So after any idle period over 300 s, the scheduler counts as "stuck" from the moment the next request is queued until its **first tick ends**.

**Scenario.**
1. The worker has been idle for more than 5 minutes, for example overnight.
2. A request arrives. Its first tick is `_busy` + `_admit` + the first prefill slab, which also includes the whole vision tower and any first-use kernel compiles.
3. If that tick takes longer than `EVENT_POLL_S` (5 s), the handler's `get(timeout=5)` expires. `alive()` returns False, and the handler returns `Error(worker_fault, "the scheduler is not running")`.
4. Its `finally` then cancels the row.

**When the first tick exceeds 5 s.** These are estimates from CLAUDE.md's prefill numbers:
* the 27B at a 2048-token slab: dense and about 3× the 9B, roughly 6–10 s;
* any large image (the vision tower runs inside the first slab);
* the 9B or 35B with `QWEN_PREFILL_STEP` raised.

**Health.** Any Health poll during that tick reports `ready=false`. The gateway sets `qwen_worker_up=0` (worker_client.rs:74-78), so alerts flap.

**Not caught.** No test touches `beat` or `SCHED_STALL_S`.

**Fix (one line).** Refresh the heartbeat when the scheduler wakes:
```python
                        while not (...):
                            self._idle(); self.beat = time.time(); self.cv.wait()
                        self.beat = time.time()          # woke up: the stall clock starts now
```
A cleaner alternative: record `self.tick_t0 = time.time()` at the start of each tick (0 when idle), and let `alive()` check `tick_t0 and now - tick_t0 > SCHED_STALL_S`.

Add a test: set `beat = time.time() - 1000`, submit a request, and assert that `alive()` stays True while the first tick runs.

### M1. Medium; High for long thinking or agentic loads. Rows that outgrow their reservation can OOM the whole process. This path was introduced by the L8 fix (worker.py:1545-1553, 1555-1562)

**Cause.**
* Rows reserve `prompt + min(max_tokens, 4096)`.
* `GATEWAY_DEFAULT_MAX_TOKENS` defaults to 0, so a request without `max_tokens` gets `max_tokens` = the rest of the 128K context (`_limit`).
* `_regrow` only adds to the books. While `reserved > budget` it blocks *admissions*, but nothing limits the growth of rows already running.
* The only protection is the headroom outside the budget: prefill reserve (2.4 GiB) + 1 GiB. While an admitted prefill is running, that shrinks to the 1 GiB.

**Example: 9B, 28 GiB cap, budget 13.8 GiB.**
1. An agentic client sends 25K-token prompts without `max_tokens`. Each reserves (25K + 4K) × 32 KiB × 1.05 + 98 MiB ≈ 1.05 GiB, so 13 rows are admitted.
2. Each row then thinks for about 12K tokens, 8K past its reservation. That is 13 × 8K × 32 KiB ≈ 3.3 GiB of growth, which uses up the whole headroom.
3. Active memory reaches the cap. On Metal this ends in a command-buffer OOM, which aborts the process, so every row is lost.

Round 1 reserved the full context, so this was impossible before. The old serialized worker was bounded by MAX_CONTEXT alone.

**Cheap fix (5 lines in `_regrow`).** Truncate one row instead of OOMing all of them:
```python
        if len(self.rows) > 1 and self.reserved + need - row.reserved > self.budget + self.prefill_reserve:
            log.warning("id=%s: batch KV past its headroom, row ended at %d tokens", row.req.id, row.n)
            self._finish(row, "length"); return
```
This keeps the last 1 GiB as slack. A refinement: end the longest or newest row instead, and skip `_advance` of a running prefill while `reserved > budget`. Document the behaviour, and recommend `GATEWAY_DEFAULT_MAX_TOKENS` more loudly.

### M2. Low-Medium. Giving up is permanent while the process keeps running, so launchd never restarts it, and one bad request retried by its client can trigger it (worker.py:1446-1456, 2077)

**What happens after 3 failed steps.**
* After 3 consecutive failed decode steps the scheduler closes and parks.
* The process stays up with Health `ready=false`, and every request gets `worker_fault` immediately.
* `deploy/launchd/com.qwen.worker.plist` has `KeepAlive true`, but that only restarts a process that exits. `watchdog.sh` watches only the gateway. So recovery needs a human.

**How it can be triggered.**
* `failures` is reset only by a completed `_iterate`, never by a successful prefill.
* On an otherwise idle worker, one request whose B=1 decode forward raises deterministically counts one strike per attempt.
* OpenAI SDKs retry 5xx twice by default, which makes 3 attempts and therefore a permanent outage.

**What does not trigger it.** A single transient allocation failure: it costs one strike, and the next successful step resets the count. A failure in admission or prefill only fails that request (`_admit` / `_advance` catch it). A real Metal command-buffer OOM aborts the process inside MLX's completion handler anyway; the strike counter only ever sees Python-level errors.

**Fix.** After `_close("worker_fault")`, exit the process: for example a timer thread that calls `os._exit(70)` about 1 s after the errors are queued, so launchd restarts the worker. Also, do not count a failed step with B == 1 as a strike: it is that request's own error, like a `_fail` in `_sample`.

### M3. Low-Medium; pre-existing in kind, more likely now. Tokenizer "Already borrowed" race between the image processor and other handlers' `encode` (worker.py:2396)

**Cause.**
* `process_inputs` passes `padding=True` to the processor. transformers 5.10.2 `set_truncation_and_padding` (tokenization_utils_tokenizers.py:782-850) then calls `self._tokenizer.enable_padding(...)` on the **shared** Rust tokenizer.
* Every later text `encode` with padding off calls `no_padding()`.
* tokenizers holds a borrow of the tokenizer while `encode_batch` runs with the GIL released. A concurrent mutating call then raises `RuntimeError: Already borrowed` (huggingface/tokenizers#537).

**Who is affected.** Vision requests now run the processor on handler threads while other handlers tokenize (`Check`, `Generate`, `_Control` bad words, `ThinkingBudgetCriteria`). Under mixed concurrent traffic the result is an intermittent `prompt_count_failed`, or a Check that returns `internal` (HTTP 500). I found this by reading the code; it was not reproduced.

**Fix.** Pass `padding=False` to `process_inputs_with_fallback`. It is a single prompt, and the test's `prepare_inputs` reference already uses `padding=False`. The padding state is then never set, and nothing mutates the tokenizer. Alternatively, put one lock around tokenizer and processor calls on handler threads.

### L1. Low. The handler thread does create MLX arrays, contrary to the documented invariant (worker.py:2396-2400; CLAUDE.md:202-205)

* `process_inputs_with_fallback` defaults to `return_tensors="mlx"`, so the HF processor builds `mx.array`s on the handler thread, and `np.array(v)` reads them back.
* These are constants: allocator calls under its mutex, no graph, no command encoder. So this is very likely safe, and vision passed on the GPU.
* But CLAUDE.md states "MLX arrays are created/evaluated only on the scheduler thread", a rule that came from a segfault.

**Fix.** Either correct the docs, or request `return_tensors="np"`. In the second case check dtypes: python lists become int64 numpy where `mx.array(list)` gave int32. The equality test does not check dtypes (see the tests section).

### L2. Low. `_close` synchronizes the wrong stream (worker.py:1486)

* `_close` runs in `_run`'s `finally`, which is outside `with mx.stream(generation_stream)`. So `mx.synchronize()` waits on the default stream.
* The real wait comes from `_idle()` → `wired_limit.__exit__`, which synchronizes `generation_stream`, and only when `self.wired` is set.
* **Fix:** `mx.synchronize(generation_stream)`.

### L3. Low. The graceful shutdown path is rarely taken (worker.py:2846-2853)

* Only the Shutdown RPC and SIGINT reach `sched.stop()`. README and CLAUDE.md stop the worker with plain `kill` (SIGTERM). There is no handler for it, so in-flight rows get a transport error, not `Error(shutdown)`.
* If a tick runs longer than `stop()`'s 10 s timeout, `main` returns and finalization starts while the scheduler is still inside MLX.
* **Fix:** `signal.signal(signal.SIGTERM, lambda *_: stop_event.set())`. After a timed-out `stop()`, call `os._exit` instead of normal finalization.

### L4. Low. `maximum_concurrent_rpcs = slots + 16` does not keep Health and Cancel from being refused (worker.py:2814 comment, CLAUDE.md:230)

* The gateway runs `Check` for every admitted request before it waits for an in-flight slot (MAX_QUEUE 32 per gateway). So with every Generate slot streaming, a burst of more than 16 Checks across gateways reaches the cap.
* grpc then refuses Health with RESOURCE_EXHAUSTED; the gateway counts that as down (worker_client.rs:60-66) and refuses Cancel.
* **Fix:** a much larger backstop, such as `slots + 64`. Threads stay at slots + 8; the cap only bounds queued RPCs.

### L5. Low. A cancelled request still does its image work and is still submitted (worker.py:2428-2480)

* There is no `cancel_ev.is_set()` check before `_vision_inputs`, which can take up to the 10 s URL timeout, or before `submit`. A dropped stream keeps holding its gRPC slot that long.
* The result is correct: the request is dropped from `waiting` at the next tick.

## Docs vs code: mismatches

| # | where | says | code / fact |
|---|---|---|---|
| 1 | CLAUDE.md:170-171 | 35B q4 "~0.75 GB @128K; q8 would blow the 32 GB budget" | q4 0.75 GB is right (5.6 KiB/token). But q8 is ~1.4 GB, which cannot blow anything, so the sentence no longer makes sense after the figure was fixed. |
| 2 | CLAUDE.md:171-172 | "9B → q8 (~8 GB KV @ 128K, ~18 GB total" | The same paragraph's "q8 ≈ 16 KB/token (9B)" gives ~2.1 GB, and worker.py:369-370 says "~2 GB". The 9B also actually runs fp16 KV (`QWEN_KV_BITS=0`, 32 KiB/token, ~4 GiB; env table, README:327). |
| 3 | CLAUDE.md:174 (also worker.py:371-372) | 27B q8 "32 KB/token", "~3.9 GB @128K" | `_kv_token_bytes`: 34 KiB/token, as README:327 says, so ~4.25 GiB @128K |
| 4 | CLAUDE.md:202-205 | MLX arrays created/evaluated only on the scheduler thread; image processor is "numpy" | The processor returns `mx.array` on the handler thread (L1). |
| 5 | CLAUDE.md:230; worker.py:1115, 2813 | "Check/Cancel/Health always run" / "a Health poll is never refused" | Not guaranteed (L4). |
| 6 | CLAUDE.md:231; worker.py:1365-1369 docstring | "a tick stuck 300 s **close[s]** the scheduler … new requests are refused" | A stuck tick only makes `alive()` False: handlers give up after 5 s, and `submit` still accepts. Nothing is closed, and it recovers if the tick finishes. |
| 7 | CLAUDE.md:237, README.md:72, BACKLOG.md:148 | `uv run --group dev pytest -q`: "CPU, tiny random models, ~10 s" | `pytest -q` also collects `tests/test_mrqmm.py`. It switches to the GPU and runs Metal kernels; it is skipped only without Metal. That contradicts "CPU" and the GPU-lease rule. Use `pytest -q --ignore tests/test_mrqmm.py` or a marker. (mrqmm is work in progress.) |
| 8 | worker.py:1100, 1110 | "see Engine._batch_memory" | The function is the module-level `_batch_budget` (worker.py:1174). |
| 9 | worker.py:1111 | "QWEN_GEN_SLOTS (2*max_batch)" | The default is `max(2*max_batch, 8)` (serve, 2809). README and CLAUDE.md say this correctly. |
| 10 | worker.py:2361-2364 (`check` docstring) | image prompts are counted "only when the scheduler runs them" | Generate counts them on the handler thread before queueing; Check still does not count them. |
| 11 | BACKLOG.md:116 | "single in-flight `Semaphore(1)`" | The in-flight limit is now `max_batch` (P8 line 148). Annotate this item. |
| 12 | CLAUDE.md / README env tables | `QWEN_MRQMM` is missing | worker.py:578, default on (added during this review). |
| 13 | README.md:326-327 | "twice the row's recurrent state (9B: … + 49 MiB" | 49 MiB is one copy; the reservation adds 98 MiB. Ambiguous. |

Consistent with the code:
* `QWEN_MAX_BATCH` 16, `QWEN_BATCH_RESERVE_TOKENS` 4096, `QWEN_BATCH_PREFILL_SHARE` 0.5 with the `1-(1-s)·B/max` formula, `QWEN_BATCH_MEM_GB` (0 = derived, every term listed), `QWEN_GEN_SLOTS` (README and CLAUDE.md), `GATEWAY_MAX_INFLIGHT`, `GATEWAY_CHECK_TIMEOUT_S` 30;
* the 503 semantics (queue full, Check timeout or DeadlineExceeded, RESOURCE_EXHAUSTED → 503, or an error frame with code 503 mid-stream; api.rs:147, 200, 352);
* "9B at 28 GiB: 13.8 GiB" (matches `r2/w_mb16.log`), and the BACKLOG figures of 81 vs ~23 tok/s (match `bench/summary.md`).

## Tests (`tests/test_robustness.py`): which claims are really tested

**Real tests, not tautological:**
* `test_failed_step_fails_its_rows_only` (it re-implements `_run`'s except clause rather than running `_run`);
* `test_scheduler_gives_up_and_refuses`;
* `test_grpc_generation_slots` (real grpc server, abort path, Check and Health latency);
* `test_handler_with_scheduler_thread`;
* `test_handler_survives_a_dead_scheduler`;
* `test_dflash_not_demoted_for_a_request_that_cannot_join` (test_scheduler.py:389).

**Weak or not testing their claim:**
* **`test_reservation_grows_and_blocks_admission` (:191-210) does not test growth blocking.**
  * The budget is `2·bytes(10) + bytes(10)/2`, so `c` cannot join `a` + `b` even with **no** growth: the initial reservations already exceed the budget.
  * Only the second half of the claim (re-accounting raises `reserved` above `two`) is shown. `c.t0 > max(a.t0, b.t0)` is plain FIFO.
  * **Fix:** a budget for 3 short rows, with `c` arriving after `a` and `b` have outgrown their reservations.
* **`test_peak_snapshot_belongs_to_the_request` (:213-220) is tautological.**
  * `peak > 0` and `peak >= active` hold for any implementation, including the old handler-side read. Running it single-threaded through `run()` cannot show the race.
  * **Fix:** record `mx.get_peak_memory()` in `on_tick` just before B's admission and assert `a.peak` equals it. Or give A a much larger prompt and assert `a.peak > b.peak` with `max_batch=1`.
* **`test_failed_join_leaves_the_batch_intact` (:293-313)** injects into `_gdn_concat`, the third row joining. L1's actual scenario is `_gdn_stack` failing **mid-loop in the 1→2 join**, which used to empty the surviving row's state. That case is not exercised. Make `mx.concatenate` fail on the second linear layer during a 1→2 join.
* **`test_shutdown`** covers `_Scheduler.stop()` only. It does not cover the Shutdown RPC, or `main`'s stop-before-`server.stop` ordering.
* **`test_grpc_generation_slots`** does not cover a slot coming back after a **client cancel mid-stream**: the generator close → `finally` path, which is the one that matters in production.
* **`test_budget_derivation`** restates `_batch_budget`'s arithmetic. The cap derivation in `Engine.load` (device working set vs `QWEN_WIRED_LIMIT_GB`, with the clamp) is untested.
* **`test_vision_inputs_equal_prepare_inputs`**:
  * `mx.array_equal` ignores dtype, so a dtype drift would pass;
  * the docstring says "+ the scheduler's conversion", but `_prefill`'s conversion is never run; the test does its own `mx.array`;
  * it skips without the ornith10 weights dir.
* **Missing:** the stall detector and heartbeat (H1 would have failed such a test), and the effect of M1's headroom.

## Nits
* `_vision_inputs` sets `tok.pad_token` from handler threads (racy, but harmless).
* `QWEN_GEN_SLOTS` < 0 crashes at startup (`BoundedSemaphore(-1)`), and there is no warning when the slots are fewer than `max_batch`.
* `mrqmm.install` runs before `fit_max_batch()` can force `max_batch` to 1. This is harmless, because B ≤ 2 never takes it.
* Still open from round 1: `_prompt_cache` scans `_bounds`, `im_start` and `_lcp` in Python on the scheduler thread, tens of ms per admission at 128K ids.
* `_fail_rows` also runs for tick exceptions that are not decode failures (`_busy`, `_fits`, `_log`), and fails healthy rows. This is unlikely.

## Verified OK
* **Generate slot (M1).**
  * `acquire(blocking=False)` comes before `try`, so the `context.abort` path never releases a permit it did not take.
  * `abort` inside the generator raises at the first `next()`, and grpc sends RESOURCE_EXHAUSTED; the real-grpc test covers this.
  * The `finally` release covers a normal end, exceptions and GeneratorExit when the stream is dropped.
  * Slots and the pool size come from the *fitted* `engine.max_batch`, because `serve()` runs after `load()`.
* **Images (M2).**
  * The exact vision token count is known before `_limit` and `submit`, so `context_length_exceeded` and `prompt_count_failed` arrive before queueing.
  * Processor and URL errors become `prompt_count_failed`. Temp files are unlinked in `finally`, after the pixels are decoded; the scheduler never opens files.
  * `materialize_images` cleans up its own files on failure.
  * The scheduler wraps the numpy arrays exactly as `prepare_inputs` does and drops `req.vision`.
  * A cancel during conversion or prefill is honoured at the next slab.
* **Peak snapshot (M3).** `_end` takes it on every way a row can end: finish, fail, a cancel during prefill or decode, and a stop-string hit through `_drain`. It is written before `out.put`, and the queue orders the write before the handler's read. A request cancelled while waiting reports 0, which is correct: it never ran.
* **Failure handling (M4).**
  * `_fail_rows` leaves `task` / `task_row` alone and resets `reserved` to the task's reservation.
  * Prefill and setup errors do not count as strikes.
  * `_idle` clears `wired` before `__exit__`.
  * The loop catches BaseException and then parks.
  * `submit` after a close answers immediately. `_close` ends rows, `task_row` and waiting requests.
  * No legitimate tick comes near 300 s: slabs are bounded by `_step` / PREFILL_BUDGET, a 16-row step takes ~0.13 s, and DFlash runs one round per tick. Only H1's stale beat trips it.
* **Atomic join (L1).** `_gdn_stack` builds every layer before it empties anything, and `_join` assigns `gdn` and `rows` only after success.
* **L2.** A failed step no longer touches the prefill.
* **Shutdown (L3).** `stop()` → `_close("shutdown")` runs before `server.stop`. Late submits are refused. The thread parks and never exits, and `_close` drops the row arrays on that thread.
* **Budget (L4, L5).**
  * `cap` = min(QWEN_WIRED_LIMIT_GB, device working set), consistent with the clamp at load. The budget is never negative.
  * With CLEAR_CACHE=0 the pool limit is subtracted, and the live guard adds `get_cache_memory`.
  * When the budget is below two typical rows, `fit_max_batch` sets 1, and Health and the slot count follow.
* **DFlash demotion (L6).** It happens only when the head is not cancelled and `_fits(head, 1)` holds, and never at `max_batch` 1.
* **L7.** `_unregister` checks that the entry is its own.
* **FIFO.** It is strict: `_admit` only looks at the head, cancelled heads are dropped every tick, and an empty batch always admits. So nothing starves; the price is head-of-line blocking.
* **Clear-cache cadence.** The pool is cleared when a row longer than 8K ends, when the batch empties, every 256 steps, after each prefill slab, and at idle.
* **`batch_mean`.** It is a snapshot taken in `_end`.
