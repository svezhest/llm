# Review: continuous-batching worker (`src/worker/worker.py` diff + `src/worker/tests/`)

Read-only review, 2026-10-11. Nothing was run: no pytest, no model, no builds.
Method: the full `_Scheduler` / `Engine` / handler code read line by line against
`scratchpad/orig_worker/worker.py` (`_run`, `_run_spec`, `generate`), and against
mlx-vlm 0.6.2 (`qwen3_5/language.py` `LanguageModel.__call__`, `Qwen3_5Model.__call__`,
the attention and GDN layers, `get_input_embeddings`, `wired_limit`, `maybe_quantize_kv_cache`).
I also checked the gateway's handling of RESOURCE_EXHAUSTED, Check timeouts and request ids.

Line numbers refer to the current `src/worker/worker.py`.

**Summary.** The batched decode core holds up under every scenario I traced:
* stacking, gathering and splitting the recurrent state;
* the step-ahead pipeline with rows joining and leaving;
* switching between sync and step-ahead mode;
* explicit positions, including prompt-cache hits, vision deltas and forced thinking-budget tokens;
* the KV quantization point;
* DFlash demotion.

I found no correctness bug in a token stream. The real findings are in four areas:
* robustness and lifecycle;
* gRPC capacity;
* the batch stalling on image work done on the scheduler thread;
* memory and peak reporting.

## Ranked findings

### M1. Medium. No limit on concurrent Generates: several gateways exhaust the gRPC pool (worker.py:2623-2629, 2570-2577)
* **The pool.** `ThreadPoolExecutor(max_workers=2*max_batch+8)` is 24 threads at max_batch 8. Every Generate holds one thread for its whole life, including the time it waits in the scheduler queue. There is no `maximum_concurrent_rpcs` and no per-RPC guard, so the worker never returns RESOURCE_EXHAUSTED.
* **Scenario: the BRIEF's own setup.**
  * Three gateways (:8080, :8081, :8090) share one worker. Each follows `Health.max_batch` as its in-flight limit (pool.rs:67).
  * That gives 3 × 8 = 24 streaming Generates, exactly the pool size. Check, Health and Cancel then queue inside grpc's executor:
    * Check waits behind the generations until `GATEWAY_CHECK_TIMEOUT_S` (30 s), and every new request on every gateway gets a 503;
    * Health polls time out (worker_client.rs:68);
    * Cancel RPCs are delayed. Dropped streams still free their rows through `add_callback`.
  * With `QWEN_MAX_BATCH=16` it is 48 Generates against 40 threads. The extra Generates sit in grpc's unbounded executor queue, so `queue_ms` cannot see them.
* **Fix.** Bound Generates and reject the excess. The gateway already maps RESOURCE_EXHAUSTED to 503 (`is_busy`, api.rs:200 and :261). Sketch:
  ```python
  GEN_SLOTS = int(os.environ.get("QWEN_GEN_SLOTS", "0") or 0) or 2 * engine.max_batch
  # WorkerServicer.__init__: self.slots = threading.BoundedSemaphore(GEN_SLOTS)
  def Generate(self, request, context):
      if not self.slots.acquire(blocking=False):
          context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED,
                        f"worker busy: {GEN_SLOTS} generations in flight")
      try:
          yield from self.engine.generate(request, context)
      finally:
          self.slots.release()          # also runs when grpc closes the generator on cancel
  # server: ThreadPoolExecutor(max_workers=GEN_SLOTS + 8),
  #         maximum_concurrent_rpcs=GEN_SLOTS + 16   (backstop only; keep it above GEN_SLOTS
  #         so that Health is never refused: the gateway marks a refused Health as down)
  ```
  * This keeps 8 threads that Check, Health and Cancel can always use.
  * Log the refusal and count it in a metric.
  * The design's risk table (§6.3) asked for "a test with 2× max_batch concurrent streams + Check latency". No such test exists.

### M2. Medium. Image download and preprocessing run on the scheduler thread and stall every decoding row (worker.py:1415-1426, 2206-2227)
* **What runs there.** `_admit` calls `eng._prepare_vision`, which calls mlx-vlm `prepare_inputs`. That function:
  * runs `process_image` / `load_image`;
  * for `http(s)://` images, which pass through `materialize_images` unchanged, makes a `requests.get(..., timeout=10)` call;
  * decodes the image with PIL and resizes it;
  * runs the HF image processor.
* **When it runs.** At the queue head, inside `tick()`. That is also true when the batch is full and the request will not be admitted yet: the preparation runs before the capacity check.
* **Scenario.** A vision request with a slow image URL arrives while 8 rows are streaming. All 8 freeze for up to 10 s or more (the timeout is per socket operation). A 4096×2645 data-URI image still costs hundreds of ms of CPU work holding the GIL. In the old worker this blocked only the serialized queue; now it stalls rows mid-answer.
* **Fix.**
  * Move the non-MLX half to the handler thread: download URLs to temp files in `materialize_images`, and open and resize the PIL images there. Ideally also run the processor with `return_tensors="np"` there.
  * The scheduler then only wraps the arrays with `mx.array` (`_prepare_vision` keeps the token count, `_limit` and `req.vision`).
  * At minimum, run `_prepare_vision` only after the capacity check passes. As written, the pixel arrays of a prepared head that does not fit also sit in memory, outside any reservation, while it waits.

### M3. Medium (observability; this mission reads `peak=`). `peak_gb` and the log's `peak=` race with the next admission's `reset_peak_memory` (worker.py:1432-1433, 2382, 2557)
* **The race.**
  * The peak is reset on the scheduler thread when a request is admitted to an empty batch.
  * It is read on the handler thread after "done" has arrived, after detokenizer finalize, split and tool parsing. It is read twice: once for the log and once for `Usage`.
  * Under serialized backlog (every `QWEN_MAX_BATCH=1` run, or any queue that drains to zero rows), the scheduler finishes A, then admits B and resets the peak, typically within milliseconds and often before A's handler thread gets the GIL.
* **Effect.**
  * A reports a peak lower than its real one, or the peak of B's early prefill.
  * A's log `peak=` and its `Usage.peak_gb` can differ.
  * BRIEF measurements that take `peak=` from the worker log are affected.
* **Fix.** Snapshot on the scheduler thread in `_finish` / `_fail` (`req.peak = mx.get_peak_memory()`) before anything can reset it, and use `r.peak` in the handler. Do the same for `active=` and `pool=` if they should be the values at finish.

### M4. Low-Medium. The scheduler thread can die or spin, and handlers then hang with no timeout while Health says ready (worker.py:1351-1378, 1792-1797, 2339)
* **Hang.**
  * `r.out.get()` has no timeout or liveness check.
  * `_run` catches `Exception` around `tick()`. Anything raised inside the `except` block, from `_fail_all` → `mx.clear_cache()`, or a BaseException, ends the daemon thread.
  * From then on every Generate blocks forever: `submit` still appends, and Health still reports `ready=True`.
* **Spin.**
  * A deterministic exception that recurs every tick turns into a 100%-CPU loop of `_fail_all`. Two examples:
    * `_busy()` → `wired_limit`;
    * `_idle()`, where `self.wired.__exit__` raises before `self.wired = None` (1374-1376), so every pass retries it.
  * `_fail_all` never fails `self.waiting`, so queued requests hang forever.
* **Fix.**
  * Set `self.wired = None` before calling `__exit__` (try/finally).
  * In `_run`, count consecutive failures. After a few, fail every waiting request with `Error(kind="worker_fault")`.
  * Make the loop catch BaseException except during interpreter shutdown.
  * Expose `sched_alive = thread.is_alive()` and report `ready=False` in Health when it is dead.
  * In the handler, use `r.out.get(timeout=5)` in a loop that yields `Error("worker_fault")` if the scheduler thread is dead.

### L1. Low. A failed join can leave the surviving row with an emptied recurrent state, and the library silently zero-fills it (worker.py:1188-1204, 1755-1761, 1464-1472)
* **What happens.** `_gdn_stack` empties each row's own `ArraysCache` (`c.cache = [None]*…`) as it walks the layers. It builds `self.gdn` only at the end.
* **When it bites.** If anything raises mid-loop, for example an eager `concatenate` shape or dtype error inside the 1→2 `_join` at the end of `_prefill`:
  * `_advance` fails only the new row;
  * `rows[0]` keeps decoding on the B=1 path with `None` states in the layers already walked;
  * `Qwen3_5GatedDeltaNet` restarts those from zeros with no error, so the answer turns to garbage without any error;
  * the "state out of step" guard (1812) only covers B≥2.
* **Fix.** Build every stacked array first, then empty the rows' caches. Or make `_join` failures go through `_fail_all`.

### L2. Low. A failure in a batched forward also fails the request that is prefilling (worker.py:1792-1795)
* `_fail_all` fails `self.task_row` as well. A transient error in the decode step therefore kills an innocent mid-prefill request.
* Design §2.11 says only "every row in the step" should get the error.
* **Fix.** Fail `self.rows` only, and keep `task` / `task_row` (its caches are separate). Or document the current behaviour.

### L3. Low. `Shutdown` does not do what design §2.11 says (worker.py:2596-2599, 2634-2640)
* The design says rows get `Error(kind="shutdown")`. In fact `Shutdown` only sets `stop_event`.
* Then `server.stop(grace=2)` cancels the RPCs, and the context callbacks set the cancel flags. The daemon scheduler is still running MLX when the interpreter exits: its last command buffer is in flight and teardown can abort.
* Handler threads stay blocked in `get()` until the scheduler answers. ThreadPoolExecutor joins its workers at exit, so a dead scheduler (M4) turns shutdown into a hang.
* **Fix.**
  * Add `_Scheduler.stop()`: set a flag, fail rows, task and waiting with `shutdown`, call `mx.synchronize()`, and leave the loop.
  * Call it from `Shutdown` / `main`'s `finally` before `server.stop`.

### L4. Low. The derived budget can be 0 while Health advertises max_batch 8 (worker.py:2074-2076, 2588)
* **Scenario.** The 35B without `QWEN_WIRED_LIMIT_GB`:
  * cap ≈ 21.8 GB;
  * weights ≈ 17 GB;
  * prefill reserve 2.5 GB, slack 1 GB, prompt cache 3 GB if it is on.
  * The budget clamps to 0, so every second request waits for an empty batch.
* **Effect.** The gateways still put 8 in flight per gateway. That costs gRPC threads (M1) and makes `queue_ms` large, while the worker is effectively serial.
* **Fix.**
  * Warn at load when the budget is below one typical row, for example `_reserve` of 4K tokens.
  * Report an effective `max_batch` in Health (1 when the budget is below that row).

### L5. Low. With `QWEN_CLEAR_CACHE=0` the buffer pool is not in the budget (worker.py:2074-2075)
* With `CLEAR_CACHE=0` the pool keeps up to `QWEN_CACHE_LIMIT_GB` (3 GB) resident on top of active memory. The budget leaves only 1 GB of slack.
* The live guard (1430) looks at `get_active_memory()` only. So active + pool can pass the cap by about 2 GB at full reservation.
* **Fix.** When `not CLEAR_CACHE`, subtract `CACHE_LIMIT_GB` from the budget, or add `mx.get_cache_memory()` to the live guard.

### L6. Low (perf). DFlash is demoted even when the waiting head cannot be admitted (worker.py:1708-1715)
* **Scenario.** Demotion checks only `self.waiting and max_batch > 1`. Two cases waste it:
  * The head may not fit next to the demoted row, for example a request with no max_tokens and a 4 GB reservation on the 9B. It then waits for the row to finish anyway, and the row has lost DFlash's ×1.5 for nothing.
  * The head may be a cancelled request that has not been dropped yet. The same loss follows.
* **Fix.** Demote only when the head is not cancelled and `self.reserved + self._reserve(head) <= self.budget`. A vision head has `ids=None`, so for it fall back to the current rule.

### L7. Low. `_unregister` pops by id without checking the owner (worker.py:2089-2097)
* **Scenario.** Two live requests with the same id: the second `_register` overwrites the first's event. The first's `finally` then pops the second's entry, so the second can no longer be cancelled by RPC.
* The gateway sends UUID-suffixed ids (api.rs:107-110), so this is latent. The code was the same before the change, but concurrency now makes it reachable for any other client.
* **Fix.** `if self._cancels.get(req_id) is ev: del self._cancels[req_id]`.

### L8. Low. The requests that most need batching cannot batch by default (worker.py:2181-2187, 1402-1411)
This is documented, not a bug. Clients that omit `max_tokens` reserve the rest of the context: about 4.2 GB of fp16 KV on the 9B, against a budget of about 6.7 GB. So by default at most one such row runs, plus small ones.

`GATEWAY_DEFAULT_MAX_TOKENS` defaults to 0, so this applies to typical chat UIs. Two options:
* make the recommendation louder: a load-time log line when `default_max_tokens` is unset and the budget allows fewer than 2 rows;
* or implement the planned V2 chunked reservation.

## Nits

* **Units.** `_batch_memory` computes in GiB (`1024**3`), but the logs print `/1e9`. `QWEN_BATCH_MEM_GB=1.5` is logged as "1.61 GB". The same mix appears in the `reserved=` log line.
* **Off-by-one.** `_reserve` uses `n > start`, but `maybe_quantize_kv_cache` uses `offset >= start`. The difference is negligible.
* **Perf.** When `CLEAR_CACHE` is on (the default), the pool is cleared after every finished row (1781, 1395-1399) and after every prefill slab while rows decode. With high churn at B=16, each clear forces the next decode step to allocate fresh Metal buffers. Consider clearing on finish only when the batch is empty or a long row left; keep the 256-step clear.
* **Racy stats.** `_Req.steps` / `rows_sum` are written as a tuple assignment on the scheduler (1823) and read by the handler after a handler-side stop hit, while the row may still be stepping. `batch_mean` can be off by one step. Only cosmetic.
* **`test_stats` counts the wasted step.** It asserts `steps == max_tokens + 1`, so `batch_mean` includes the look-ahead step whose token is never read. This is fine but undocumented in `Usage`.
* **Python scans on the scheduler thread.** `_prompt_cache` does `_bounds`, the `im_start` scan and `_lcp` over all entries in Python, now on the scheduler thread, about 10 ms at 128K ids. That stalls decode once per admission. Moving it to the handler (pure Python over `req.ids`) would be free.
* **`check()` return type.** It is annotated `-> str` but returns a tuple. This predates the change.
* **`ttft_ms` sentinel.** `if ttft_ms == 0` re-measures when the first token lands within 1 ms of `t0`. This predates the change.
* **Prompt count of a cancelled vision request.** A vision request cancelled before preparation reports `prompt_tokens=0` in its cancelled `Done`.
* **Mixed token dtypes.** `row.y` is int32 for the prompt token and the DFlash `b`, and uint32 from samplers. `mx.concatenate` promotes the mix to int64. This is harmless (embedding gather), but a cast to `mx.uint32` at join would keep the graph uniform.

## Tests: what they show and what they miss

**Not tautological.** The core claim tests compare a batched run with staggered arrivals against the same requests run one at a time:
* `test_batched_tokens_equal_solo`, `test_kv_quantized_mid_batch`, `test_sync_mode_and_row_error`, `test_vision_rows`;
* each also asserts that batching actually happened (`max(sizes) >= 3`, mixed quantized rows in a batch, a delta row inside a batch).

Other tests check against independent ground truth:
* `test_rows_equal_solo` compares logits with the library's own B=1 call;
* `test_explicit_positions_bit_identical` checks the B=1 identity claim bit for bit;
* the DFlash test checks the A1 keys against a fresh prefill;
* `test_memory_formulas` checks bytes per token and per row against real `nbytes`. Its `_reserve` half restates the formula, which is tautological for the arithmetic.

**Missing:**
* The batched-forward failure path: `_fail_all`, the scheduler continuing with the waiting queue, `reserved == 0` afterwards, and the collateral prefill failure (L2). The scheduler thread surviving an exception is also untested.
* A `_sample` exception in **step-ahead** mode, where the row fails inside `_step` with its forward already queued. Only the sync-mode `consume` failure is covered.
* `Usage.peak_gb` / reset semantics (M3), and Shutdown (L3).
* gRPC capacity: 2× max_batch streams plus Check latency (M1). `context.add_callback` on a dropped stream is covered only by the scratchpad e2e harness, not by `tests/`.
* Thinking budget together with a grammar row, in sync mode with a forced token.
* `_batch_memory` derivation (cap, prompt-cache GB, the clamp to 0).
* **No unit test checks B=1 against the old worker.** It is verified only by the scratchpad CPU/GPU corpus (15/15 byte-identical), not by `tests/`. A small test that runs `orig_worker._run` and the new scheduler at `max_batch=1` on the tiny model would lock it in.
* **Handler tests never exercise the scheduler/handler hand-off.** The fake scheduler enqueues the whole script synchronously, so the stop-hit test only checks that the cancel flag is set, not that the row is evicted. Eviction on cancel is covered separately in `test_cancel_waiting_prefill_decode`.

## Verified OK (traced, no issue found)

* **Thread ownership.**
  * After load, MLX arrays are created only on the scheduler thread: `make_sampler`, `_Penalties`, `_Control.build`, vision `prepare_inputs`, prompt-cache lookup, restore and store, prefill, decode.
  * `_Guide.__init__` is numpy and llguidance only; `ThinkingBudgetCriteria.__init__` tokenizes only.
  * Handler threads call only the allocator counters (`get_peak/active/cache_memory`).
  * `_PrefixCache` is touched only by the scheduler.
* **Handoff.**
  * `waiting` is appended under `cv`. `popleft`, `[0]` and `remove` happen only on the scheduler; `remove` runs under `cv`.
  * The idle `cv.wait()` predicate is checked under the lock, so no wake-up is lost.
  * Cancels do not need a notify: the scheduler never sleeps while anything is waiting or running.
  * Every request path ends in exactly one terminal event; `_finish` / `_fail` are idempotent through `row.done`.
* **Cancel sources.**
  * The Cancel RPC, `add_callback` and the handler's `finally` set the flag; a stop-string hit sets it too.
  * The scheduler acts on it in three places: waiting (each tick), prefill (each slab, `_prefill` and `_spec`) and decode (each `_iterate`).
  * Reservations are released exactly once.
* **Step-ahead with joins and leaves.**
  * A departed row's queued token is dropped (`if not r.done`), and its state is dropped by the gather in `_compact`. `gdn` order always equals `rows` order: join appends and concatenates at the end, leave gathers with the same `keep`.
  * A newly joined row's `y` is its own last prompt token, or for a demoted DFlash row its emitted `b`, with `first=True`. So it never receives a stale token, and that token is not emitted, not penalty-counted and not counted toward `min_tokens`.
  * Stacking at 1→2 takes the solo row's state including its in-flight step, which matches its pending `y`.
  * 2→1 hands the survivor slices of the stacked state.
* **Mode switching.** Sync and step-ahead can switch mid-batch in either direction. In sync mode, `guide.consume` / `ctl.history` run before the next `_step`, as in `_run`'s non-ahead order. Forced thinking tokens patch the row's next `y` after its step is queued, as in the old loop.
* **Positions.**
  * Decode always passes explicit `(B,1)` positions, so `LanguageModel.__call__` skips the `_rope_deltas` / `_position_ids` branch entirely and decode never writes model state.
  * Prefill alone mutates that state, and only one prefill runs at a time.
  * Vision slabs slice `lm._position_ids`, which no decode step can change between slabs (the A2 fix holds under interleaving).
  * After a prompt-cache hit, `row.pos` = `len(ids)-1`; vision adds `delta` from `_rope_deltas` after prefill; a demoted DFlash row takes `pos` from the cache offset after rollback.
  * A 1-token text prompt after a vision request is now correct; the old code inherited the stale delta.
* **Library internals with a `RowsKV`.**
  * `RowsKV.offset` is read only for positions (skipped) and the mask. With L=1, `create_attention_mask` returns None, and none of the left-padding or batch-cache branches trigger.
  * The GDN batch-size reset is guarded (1812).
  * `_gdn_zero` matches the library's zero init: conv in the hidden dtype, state fp32.
* **KV quantization while batched.** `quantize(row.cache)` runs per row after the forward is queued, as in `_run`. The per-entry `offset >= start` in mlx-lm 0.31.3 never touches `ArraysCache`. RowsKV is rebuilt from `r.cache` every step, so the swapped-in `QuantizedKVCache` is used from the next step. The reservation's bytes per token match `QuantizedKVCache` (fp16 scale and bias per 64-element group).
* **Per-row sampling state.** Each row keeps its own sampler closure and key (seed per request), its own penalty counts (added from the fed `y`, skipped when `first`), its own `_Control` counter, `_Guide` and `ThinkingBudgetCriteria`.
* **`QWEN_MAX_BATCH=1` against the old `_run` / `_run_spec`, compared line by line.** The same:
  * slab cuts, quantize, checkpoints, snapshot point, and order of penalties → mask → quantize → guide/sampler;
  * the step-ahead and sync schedules, and `max_tokens` 0 / 1 handling;
  * stop and skip handling, and `n` counting skips.
  * DFlash is never demoted at `max_batch=1`.

  The differences:
  * the first decode feeds the token id instead of `inputs_embeds[:, -1:]` (identical gather);
  * positions are explicit (CPU-verified bit-exact);
  * vision `extra` kwargs are not passed to the first decode step (unused once `cache_offset > 0`);
  * the A1 and A2 fixes are intentional;
  * `wired_limit` is entered once per busy period instead of per request;
  * `clear_cache` timing is global (no numerical effect);
  * the peak is reset per empty batch (see M3).
* **Handler parity with the old `generate`.**
  * The `_GenState` split, `state.token` counting (ignored EOS counted, not shown) and `_StopFilter` (gated on `gen_tokens > min_tokens`, `include_stop_str_in_output`) are the same.
  * The `<tool_call>` cut without tools, the buffered tool answer with `parse_tool_calls` / `tool_choice none` / `parallel_tool_calls` false / the finish→`tool_calls` rules, and the final detokenizer segment through the stop filter are also the same.
  * Error events cover render, encode, `_limit`, vision prepare, setup, prefill, sampling and the batched step.
  * `Done(cancelled)` is sent for the RPC and stream-drop paths.
  * Every handler gets its own detokenizer: `make_streaming_detokenizer` copies and `reset()` rebinds the lists, and `tokenmap` is shared read-only.
* **The `_Control` split.** `active` is equivalent to the old expression. The masks are built on the scheduler, and the out-of-vocab check runs in `_render`, so Check and Generate both reject. The proto ids are uint32, so negatives cannot occur.
* **Admission.**
  * It is FIFO with no overtaking, and an empty batch always admits, so no request starves.
  * The task row is outside `rows` but inside `reserved`, and `_admit` never runs while a task exists. So `len(rows) < max_batch` and `reserved` stay consistent.
  * `_fail_all` resets `reserved` to 0, consistent with dropping every row and the task.
* **`mx.clear_cache()` with rows alive.** It frees only unreferenced pool buffers; in-flight command buffers keep theirs. It is safe, and the cost is only allocation churn (see the nits).
* **Prompt cache while batching.** Entries are stored at the end of a row's prefill, from new cache objects and new lists (`_copy_cache`). `_gdn_stack` rebinds the row's own list, never the entry's. KV slice views are copy-on-write under KVCache's in-place `update_and_fetch`. A request's prefill starts after the previous prefill stored its entry, so it hits that entry.
