# Review: prompt cache phase 1 (uncommitted diff, 2026-10-12)

Scope: `git diff` of the tracked files plus `src/worker/tests/test_prompt_cache.py`. This was a
read-only, CPU-only review. Line numbers refer to the working tree of `src/worker/worker.py`.

Commands run:

* `uv run --group dev pytest -q --ignore tests/test_mrqmm.py --ignore tests/test_pfqmm.py`:
  83 passed.
* `cargo test -p qwen-gateway`: 13 passed.
* Scratch-only tools, nothing written to the repo:
  * a bitwise invariant harness;
  * three mutation runs against copies of the worker;
  * a slab-sensitivity probe.

**Hard invariant: holds.**

* **By reasoning.** Every position that can become a resume point is a `_bounds` slab end of the
  prompt whose KV the entry holds. It is also a slab end of the prompt that resumes:
  * a bound at `i` depends only on `ids[:i+1]`;
  * every inherited, absorbed or end position `p` is `< len(shared prefix)` or is checked against
    `ends`.
* **By harness.** For every entry after every request, and every position `best()` could return
  (checkpoints plus the end, at or above the quantized floor), `restore(e, p)` was compared bitwise
  with the entry of a cold run of `e.ids[:p] + [x]`. Workload:
  * two interleaved 8- and 6-turn conversations, plus regenerates and "older prompt again";
  * fp16 KV and q8 KV (`kv_start=40`);
  * `CKPTS` = 12, 4 and 3;
  * 3 seeds.

  About 3,600 comparisons, **0 mismatches**.
* **The check has teeth.** The CPU path is deterministic and sensitive to slab shapes: the same
  64-token prompt cut at {32} vs {16, 32, 48} differs in 13 of 16 cache arrays.

The findings are about memory and admission policy and about test strength, not about exactness.

---

## Findings (ranked)

### 1. Medium: `_fits` keeps the entry with the longest LCP, not the one the request would resume from, and waits for a row to end to save as little as 256 tokens (worker.py:2066–2078, `closest` 815)

What the code does:

* `keep = pc.closest(req.ids)` picks the max `_lcp`.
* It requires `common >= PREFILL_MIN_STEP`.
* If evicting everything else is not enough, `_fits` returns False. The request then waits for a
  running row to finish.

Failure scenarios:

* **27B/35B (q8/q4 KV, floor 5000).**
  * An entry for a 9K prompt (quantized) shares a 3K system prompt with the waiting request.
  * LCP is 3K, so the entry is kept.
  * But `best()` gives 0 for it: 3K is below `quantized_kv_start`.
  * The request waits for a row to end (up to that row's whole generation: minutes for a 4K
    answer at 15 tok/s), then misses anyway.
* **9B, thinned or sparse checkpoints.** The kept entry's deepest usable checkpoint is far below
  the LCP, or a sibling entry has a deeper one at a slightly shorter LCP, and the sibling gets
  evicted.
* **In general.** Waiting is worth it only when the resume saves more prefill time than the
  expected wait. 256–2K tokens is 0.5–4 s of prefill. The wait can be a whole generation, and
  with closed-loop waves all rows end together.

Fix: pick `keep` the way `lookup` would, and require a real depth.

```python
# _Scheduler._fits
if self.reserved + need + pc.bytes > self.budget:
    ends = getattr(req, "_ends", None)
    if ends is None:                       # memoized: _fits runs every tick while blocked
        ends = req._ends = set(_bounds(req.ids, self.eng._im_start)) | {len(req.ids) - 1}
    keep = max(pc.entries, key=lambda e: e.best(req.ids, ends)[0], default=None)
    depth = keep.best(req.ids, ends)[0] if keep is not None else 0
    if depth < KEEP_MIN_TOKENS:            # e.g. 2048, or a fraction of len(req.ids)
        keep = None
```

`_Req` is a dataclass, so add the field rather than setting an ad-hoc attribute.

### 2. Medium-low: `_fits` evicts other entries even when that cannot admit the request (worker.py:2066–2076)

What happens:

* The loop evicts LRU entries (all but `keep`) until it fits *or* only `keep` is left.
* If `reserved + need + bytes(keep alone) > budget`, it then returns False anyway.
* Every other conversation's entry has been dropped for nothing. Their next turns re-prefill their
  whole history: 10K tokens is about 20 s.
* The same holds when the later `live + need + prefill_reserve <= cap` check fails after the
  evictions.

Fix: evict only when that will be enough.

```python
if self.reserved + need + pc.bytes > self.budget:
    floor = pc.bytes_of([keep]) if keep is not None else 0  # _recount over a subset
    if self.reserved + need + floor > self.budget:
        return False                                        # waiting either way: evict nothing
    while ...: pc.evict(keep)
```

Ideally the `live` check is also done *before* evicting, using `live - (pc.bytes - floor)` as the
estimate.

### 3. Medium-low: a store is never checked against the batch budget (worker.py:873–880, snapshot 2184)

The removed 3 GiB carve-out used to be the slack for entries added beside full rows. Now:

* Admission checks `reserved + need + pc.bytes <= budget`.
* But the entry stored right after that prefill (≈ prompt KV + up to 12 × 51.5 MB of
  checkpoints) is added unconditionally, up to the soft cap, or above it when it is the newest
  entry alone.
* The row's first decode step then copy-on-writes its KV away from the buffer the entry pins. So
  the bytes are real: rows + cache exceed the budget by up to one entry.
* Nothing corrects that until a row passes a 1024-token growth step (`_regrow`) or another
  admission happens.

Size of the overshoot:

* 8K: ≈ 0.6 GB, absorbed by the 1 GiB slack.
* 32K: ≈ 1.6 GB.
* 128K: ≈ 4.6 GB on the 9B. That is past prefill reserve + slack (3.4 GiB on the 9B) while a
  prefill uses that reserve, so OOM territory.

Fix: give `store` a limit from the scheduler.

```python
# snapshot
e, covered = pc.store(..., limit=self.budget + self.prefill_reserve - self.reserved)
# store: after the soft-cap loop
while len(self.entries) > 1 and self.bytes > limit:
    self.evict(keep=new)
if self.bytes > limit:          # the newest alone does not fit beside the rows: do not keep it
    self.entries.remove(new); self._recount(); log.info(...)
```

This is the design review's N5 hazard. It is made larger by dropping the carve-out.

### 4. Low: `_regrow` can evict the in-flight prefill's parent; memory is not freed while `pc.bytes` says it is (worker.py:2038–2044, 2184)

* `_regrow` runs from `_iterate` between the slabs of a running prefill task, so `_fits` is not
  the only evictor while a prefill is in progress.
* Correctness is fine: the `snapshot` closure holds `parent` and reads only `parent.ckpts` and
  `parent.cache`.
* But the parent's KV plus its checkpoints stay allocated until that prefill ends, for minutes on
  a 32K–120K cold remainder.
* `_regrow` meanwhile lets the growing rows take that memory, because accounting says it was
  freed.

Fix (either):

* Record `row.parent` in `_prompt_cache` and call `pc.evict(keep=self.task_row.parent)` in
  `_regrow`.
* Better: have the closure capture only what `store` needs, so the parent's KV is not pinned:

  ```python
  inherit = {p: s for p, s in parent.ckpts.items() if p <= reused}
  if reused == len(parent.ids):
      inherit[reused] = _recurrent_states(parent.cache)
  ```

  Pass `inherit` instead of `parent`.

The docs line "Eviction is safe mid-use (a restore holds its own views)" (CLAUDE.md, `evict`
docstring) should add "memory is released when the user finishes".

### 5. Low: quantized entries keep, and now *protect*, checkpoints below `quantized_kv_start` that `best()` can never return (worker.py:2171–2173, 862–870)

On the 27B/35B, a cold 8K prompt takes its checkpoints at the system end, 2048 and 4096. Its entry
is quantized, so `best()` filters every one of them (`low = 5000`, line 772).

Effects:

* That is unique memory (one GDN state each) and up to 3–4 of the 12 slots.
* With `protect`, the system-end one can no longer be thinned at all.
* Absorbed or inherited sub-floor checkpoints of quantized entries are dead weight the same way.

Fix:

* In `_prompt_cache`, drop `b < kv_start` from `keep` and `protect` when `kv_bits` is set.
* In `store`, `if quantized: ckpts = {p: s for p, s in ckpts.items() if p >= KV_KWARGS["quantized_kv_start"]}`
  before `_thin`.
* The floor entry (finding 1 of the design) is unaffected.

Related, pre-existing: on quantized models a shared system prompt under 5000 tokens hits across
conversations only if some conversation's first request was itself under 5000 tokens (the floor
entry). Worth a line in the docs.

### 6. Low (test strength): the "answers as a cold run" assertions compare greedy token ids, which cannot see a wrong slab cut (tests/test_prompt_cache.py, every `== cold(...)`)

How this was checked:

* Mutation: absorb the old END state unconditionally (drop `len(e.ids) in ends`, worker.py:867).
  The repeat of p1 then resumes at a position that is not a slab end of p2 (`reused 32` vs the
  correct 28).
* The `run(...) == cold(...)` token assertion still **passes**.
* Only the structural assert `L not in new.ckpts` catches it.

Also verified:

* A bitwise comparison of the stored entry against a cold run's entry catches it (`False` under the
  mutation, `True` on the real code).
* `_cache_arrays` + `mx.array_equal` over the entry is enough.

Suggested helper, used after every hit in these tests:

```python
def same_entry(a, b):
    x, y = worker._cache_arrays(a.cache), worker._cache_arrays(b.cache)
    return len(x) == len(y) and all(u.shape == v.shape and mx.array_equal(u, v).item()
                                    for u, v in zip(x, y))
```

Better still, port the harness loop: every `best()`-able position of every entry vs a cold prefill
of `ids[:p]`. It ran in about 10 min over the full matrix; one config takes about 40 s.

### 7. Low: item 2 moves the decode wait after admission, which `queue_ms` does not see (worker.py:2090, 2278)

* After a prefill's last slab, the next request is admitted at once (`req.t0`, `queue_ms`
  measured), but its prefill starts only at `resume`, up to t·(1−s)/s later.
  * For a cold 120K prefill that is ≈ 9 s at s = 0.5.
  * Before this change, `resume` had normally expired by then.
* So `queue_ms` ("wait from arrival to the start of prefill", proto comment) under-reports by that
  much.
* This is intended scheduling; only the metric is off. Either set `req.t0` / `queue_ms` on the
  first `_advance` of the task, or reword the proto comment.
* Expected cost, as designed: in a 16-hit burst the last hit's TTFT grows by about Σ of the
  earlier hits' slab times at s = 0.5 (≈ 16 × 0.35 s ≈ 5–6 s). Worth stating in the bench note.

---

## Verified OK

* **Absorption (item 1).** The rule is "any entry whose ids are a prefix of the new entry's ids".
  * Its checkpoints `p < len(e.ids)` are bounds of the new prompt, since `ids[:p+1]` is shared.
  * Its END state is taken only if `len(e.ids)` is in `ends` and below `len(ids)`; `ends`
    contains `len(prompt)-1 = len(ids)`, which the `<` excludes.
  * `ckpts.pop(len(ids))` drops the redundant end checkpoint.
  * Transitive inheritance over many turns stays valid by induction (harness: 8-turn chains,
    caps 3/4/12).
  * The exact-repeat path (no slab, `at = len(ids)`) absorbs the parent with no END checkpoint.
* **Parent inheritance.** `p <= at` with `at ∈ ends`. The parent end state at `at == len(parent.ids)`
  is the same last slab `[b_last, at)` in both prompts.
* **Quantized floor.**
  * Legacy `maybe_quantize_kv_cache` quantizes at `offset >= start` after each slab, so a
    quantized resume at `p >= start` matches the cold conversion point.
  * Per-token groups make `upto=p` slicing exact.
  * New-quantized/old-fp16 is not absorbed. The reverse cannot occur, because a longer prompt is
    at least as quantized.
* **`protect`.** It only filters thinning and never creates a resume point. `_thin` never drops
  the deepest. When everything but the deepest is protected it falls back to the span rule.
  `_thin({5:1}, 0)` terminates.
* **Cancelled or failed mid-prefill.**
  * A cancel is checked before each slab, then `return`, with no `on_prefilled`.
  * An exception in a slab never reaches `on_prefilled`.
  * A cancel during the last slab stores a complete, valid prefill.
  * `store` is atomic up to `_Entry(...)`: `self.entries` is reassigned only after the entry is
    built. A failure in the try/except (item 4) leaves the cache unchanged; the test covers it.
* **`_copy_cache` sharing.** KV gets fresh slice objects. The row's first slab or decode step
  concatenates, or copy-on-writes a non-donatable shared buffer. GDN lists are new and their
  arrays are replaced whole (`_gdn_split` assigns slices). No live cache shares an mx.array object
  that something writes with `a[i] = v`.
* **`_fits` and in-flight restores.** `_fits` runs only from `_admit`, and `_admit` only when
  `self.task is None` (tick, line ~1992). So no restore is in flight then. (`_regrow` is the
  exception: finding 4.)
* **Item 2.**
  * The empty batch is unaffected: admission is not gated by `resume`, and `_advance` runs when
    `not self.rows`.
  * `PREFILL_SHARE` is clamped to ≥ 0.05, so `s > 0` even now that `_slab_done` sees
    `len(rows) == max_batch` after the join. With share 0 this would have divided by zero.
  * No race with `req.cancel`: cancels of joined rows are handled in `_iterate`.
  * `_spec` needs no charge: it is lone, so `resume` is irrelevant. On demotion the next
    admission's first slab runs immediately, as before.
  * Mutation (the charge removed) fails `test_decode_runs_between_short_prefills`.
* **`used` timestamps.** They are set at creation and on a `lookup` hit; ties go to the older list
  position. Fine.
* **`bytes` recount cost.** About 8 KV views + 48 end-state arrays + ≤ 12 × 48 checkpoint arrays,
  so ≈ 650 ids per 9B entry. One set pass per store or evict is ~1 ms at 10 entries. Fine.
* **Thrash under 16 × 8K with the 3 GiB cap.** Budget 15.75 GiB on this machine (log: 12.75 + the
  former 3 GiB carve-out).
  * Each row reserves (8.3K + 4096) × 32 KiB × 1.05 + 2 × 49 MiB ≈ 0.50 GiB, so 16 rows ≈ 8.0 GiB.
  * With 3 GiB of cache, 11 GiB < 15.75. No `_fits` or `_regrow` evictions; only the store-time LRU
    runs, as before.
  * Admission evictions start around 16 × 17K prompts. The cache then settles to about one entry,
    the kept one: one evict per store, which is LRU doing its job rather than a loop.
* **Proto.** `Usage.cached_tokens = 9` (1–8 taken). `HealthReply` 6–10 (1–5 taken). proto3 default
  0 means an old worker yields `cached_tokens: 0`, and an old gateway ignores the new fields.
* **Gateway.**
  * `usage_json` carries `prompt_tokens_details.cached_tokens`.
  * It is used by the stream usage chunk (`include_usage`), the non-stream response and the
    journal (`acc.response` for streams and non-streams).
  * `on_done` is called from both paths. The counter is described in `metrics.rs`.
  * Unit test `usage_reports_cached_tokens` passes.
* **Worker.** `cached_tokens = r.reused` (0 for vision or a miss; `len(ids)-1` for an exact repeat,
  as vLLM would report). Health reads only ints and `len(list)`.
* **`bench_concurrency.py`.** `rx_store` still matches the extended store line (prefix match). The
  new test's `RX_STORE` pins the full format.
* **Mutation runs** on copies:
  * Old covered rule restored: 4 tests fail.
  * `protect` dropped: 2 tests fail.
  * Item-2 charge removed: 1 test fails.

## Nits

* `closest()` ties go to the oldest entry by list order, not LRU (moot if finding 1 is fixed).
* `_slab_done` after `_join` counts the just-joined row in `s`. The intermediate-slab charges
  don't count the prefilling row. That is slightly more decode share than the per-slab rule; fine,
  but say so in the comment.
* Health's `prompt_cache_*` counters reach only a gateway `debug!` line on change. Export them as
  `qwen_worker_prompt_cache_{entries,bytes}` gauges and `…_{hits,misses,evictions}` counters, or
  they are invisible at `RUST_LOG=info`.
* vLLM v0.30 returns `"prompt_tokens_details": null` unless `--enable-prompt-tokens-details` is
  set. Always emitting the object matches OpenAI rather than vLLM's default. That is harmless, but
  worth a README line given "behaviour reference = vLLM".
* `bench_concurrency.py --prefix distinct` (the default) moved the filler from the user to the
  system message and changed the calibration seed. Results are not directly comparable with
  `final_bench.md` runs; note this in the docstring or `docs/perf/README.md`.
* `_fits` recomputes `closest()` (an LCP per entry) every tick while the head is blocked. This is
  cheap after the evictions (one entry left), but it is the place to memoize (see finding 1).
* `pc.bytes` counts KV views at their sliced size. The underlying buffer (rounded up by the 256-step
  growth) is ≤ 8 MB larger per entry. This is negligible; no change needed.
