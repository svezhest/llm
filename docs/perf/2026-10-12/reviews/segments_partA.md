# Review: prompt cache Part A, segment store + phase-1 findings (uncommitted diff, 2026-10-12)

Scope: `git diff` of `src/worker/worker.py`, `tests/conftest.py`, `tests/test_prompt_cache.py`, the new
`tests/test_segments.py`, and CLAUDE.md / README / BACKLOG. Out of scope: `cascade.py`, `test_cascade.py`
and the docs/perf files. Line numbers refer to the working tree (worker.py md5 `7fdb2b87…`).

This was a read-only, CPU-only review. Nothing in the repo was changed except this file.

* **CPU suite**: `uv run --group dev pytest -q --ignore tests/test_mrqmm.py --ignore tests/test_pfqmm.py
  --ignore tests/test_cascade.py` gave **98 passed** (133 s).
* **Mutation runs** used scratch copies of the worker plus `test_segments.py` and `test_prompt_cache.py`:

  | mutation | result |
  |---|---|
  | M1: `_Seg.copy_from` keeps plain views of the row (the HEAD semantics) | 1 test fails (`test_store_copies_out_of_the_row`) |
  | M2: restore without the pre-sized tail | 1 fails (`test_restore_layout`) |
  | M3: parent pieces reused even when the row's prefill quantized them | 3 fail (crash on mixed dims + bitwise) |
  | M4: segments copied as zeros (a common-mode bug in store and in the `Cold` reference) | 11 fail, including both `check_cold` params |
  | M5: an absorbed END state taken without `in ends` (phase-1 review's mutation) | 5 fail, including `test_entries_equal_cold_bitwise[fp16]` |

* **A scratch test confirmed finding 1** (the store wipe): `test_scratch_drop.py`, in the scratchpad.

**The invariant (hit == cold, bitwise) holds**, by reasoning and by the tests. The findings concern
memory policy, plus one leftover pin.

---

## Findings (ranked)

### 1. Medium-low: a store that drops the new entry has already absorbed the conversation's previous entry and evicted the others in vain (worker.py:1002–1042)

**What the code does.** `store` takes these steps in order:

1. It removes the `covered` entries (absorbed), at 1022.
2. It evicts every other entry LRU while `bytes > limit`, at 1031–1035.
3. Only then does it find that `new` alone is over `limit`, and drops it, at 1036–1041.

**Result.** The cache can end up empty although its state before the store fitted the limit.

**Reproduced (CPU).** A cache holds `a0` (turn 1 of chat A) and `o` (another chat). The headroom is
set to exactly `bytes(a0) + bytes(o)`. Turn 2 of A then stores:

* `a0` is absorbed;
* `o` is evicted;
* the new entry is dropped.

Outcome: **0 entries** and `evictions=1`. Before the store, `a0 + o` fitted the limit.

**Effect.** The scenario is a busy batch (many rows near the budget) with a long conversation. The
next turn of that chat re-prefills its whole history, and so does every other conversation evicted
in vain. This is finding 2 of the phase-1 review, now inside `store`.

**Fix.** Decide before mutating `self.entries`:

```python
new = _Entry(...)
if limit is not None and self.bytes_of([new]) > limit:
    # do not absorb; drop new (its own segment goes with it)
    while self.entries and self.bytes > limit and self.evict():   # only what the old state needs
        pass
    log.info("... not kept ..."); return None, 0
# else: absorb, soft-cap loop, then the limit loop as now (new fits alone, so it stays)
```

The check can also run before `_Seg.copy_from`, using an estimate (finding 4).

`test_store_bounded_by_the_budget` pins the current wipe in its dropped case (`pc.evictions == 3`,
`entries == []`). Its fill entries are over the room anyway, so it would pass the fixed code
unchanged. Add the a0/o case above as a test.

### 2. Low-medium (policy): `_fits` still makes the head of the queue wait for a row to end to save a resume of only `PREFILL_STEP` tokens (worker.py:2238–2248)

**What the code does.** When the `peek` entry is at least `PREFILL_STEP` deep and
`reserved + need + floor > budget`, `_fits` returns False. Evicting that entry too would admit the
request now, since `reserved + need <= budget` was checked above.

**Cost of the wait.** FIFO without overtaking means the whole queue waits until enough rows end:
* that can be a full generation (minutes for a 4K answer at 15 tok/s on the 27B);
* `GATEWAY_DEFAULT_MAX_TOKENS=0` (rest of context) has no bound at all.

**What the wait saves.**
* `PREFILL_STEP` is 2048 by default, about 3 s of 9B prefill.
* It is 512 on the 35B run recipe (`QWEN_PREFILL_STEP=512`), about 1 s.

**The threshold.** Since it is the slab knob, the threshold means different things per model. The
phase-1 finding asked for a depth worth the wait.

**Fix (pick one):**
* a token threshold of its own (e.g. 8K, or a fraction of `len(req.ids)`);
* a bounded wait: remember when the head first blocked; after N s, evict the keep entry too and
  admit cold;
* keep only when some running row is near its end.

`test_admission_keeps_the_entry_it_would_resume_from` pins the wait. It would need a variant.

### 3. Low: a DFlash row pins its parent's segments and the prefill's checkpoint states for its whole decode (worker.py:2525, 2575)

**What happens.**
* `_spec` keeps `on_prefilled` (the `snapshot` closure) in its generator frame until the DFlash run
  ends.
* The closure holds `inherit` (the parent's pieces, i.e. its KV segments) and `ckpts`.
* `row.parent` is cleared at the snapshot, so nothing protects the parent any more. Two paths evict:
  * `_regrow` (`keep=()` for this row);
  * `_fits(head, 1)` in the demotion check.
* Either can evict the parent or the new entry. `pc.bytes` then drops, but the segments stay
  allocated until the DFlash generation ends.

The same holds for up to 12 × 51.5 MB of checkpoint states. `_prefill` is unaffected: its frame dies
right after the join.

This is the residual of phase-1 finding 4. CLAUDE.md's "A restore holds copies, so eviction frees
memory at once" is not true for this window.

**Fix.** In `snapshot`, add `nonlocal inherit` and set `inherit = None` (and `ckpts.clear()` is not
needed: the entry owns the states) in a `finally`. Or, in `_spec`, set
`on_prefilled = on_ckpt = None` right after the call.

### 4. Low: `store` materializes the new segment before checking `limit` (worker.py:1002 vs 1031)

**What happens.** For a cold 64K–120K prompt finishing beside a full batch, `_Seg.copy_from` first
allocates the whole `[0, L)` copy (2–3.9 GB on the 9B) and evaluates it. Only then can `limit` drop
it.

**How tight it is.** The headroom left at that moment is `limit` (which includes the idle prefill
reserve) plus 1 GiB of slack.

**Compared with HEAD.** It is strictly better: HEAD's view became a permanent COW copy on the first
decode write, while here the copy is only transient. But the spike can still pass the slack exactly
in the case finding 3 of the phase-1 review was about.

**Fix.** Estimate before copying:
* KV bytes per token: from the row's arrays (`nbytes / capacity`) × `(L − lo)`;
* plus the new states;
* plus `bytes_of` of the inherited pieces.

If that is over `limit`, skip the copy (and the absorb, finding 1).

### 5. Low (metrics semantics): the worker's `ttft_ms` now counts from prefill start, and the proto does not say so (worker.py:2306–2310, 3354; worker.proto:74)

**The change.** `queue_ms` is now what the proto comment says: arrival → the start of prefill. Before,
it under-reported by the decode-share wait. `ttft_ms` (no proto comment) moved with `t0`:

* it was admission → first token;
* it is now first slab → first token.

So `queue_ms + ttft_ms` is still arrival → first token, and the gateway's `qwen_ttft_seconds` (its
own clock, from handler entry) is unaffected. **The change is acceptable.**

**What to fix.**
* Add `// from the start of prefill (queue_ms is the wait before it)` to `ttft_ms` in
  `worker.proto`.
* Note that `w_ttft_ms` columns (`scripts/perf/final/pcache_tables.py`, `bench_concurrency.py`) are
  not comparable with `prompt_cache_bench.md` for requests admitted behind a prefill.

Nothing else reads `t0`. The 300 s watchdog uses `tick_t0`, and the error paths send no Usage. A
request failing in `_admit` setup keeps `queue_ms = 0`, but it only ever yields an Error.

---

## Verified OK

* **Resume positions are unchanged from phase 1.**
  * `best()`: `tops` includes the end only if `end is not None`, which Part A always sets. `low` is
    now per entry from the scheduler's `kv` dict. HEAD read the global `KV_KWARGS`, so with
    `kv_start=40` the tests' quantized entries had a floor of 5000: a latent test-only bug, now fixed.
  * Inherited checkpoints are `p <= at`. `inherit.end` is used only at `at == len(parent.ids)` and
    `at < L`.
  * Absorbed END requires `len(e.ids) in ends`. M5 shows `check_cold` now catches its removal (it
    did not by token ids).
  * `ckpts.pop(L)` is kept.
  * The quantized filter `p >= kv_start` applies to own, inherited and absorbed checkpoints alike.
    `_prompt_cache` filters `keep`/`protect` only when `end >= start`, which is exactly when the
    prefill's last slab quantizes (`maybe_quantize_kv_cache`: `offset >= start` after every slab).
* **Chain and pieces.**
  * `cut(at)` clips at `at`. A straddled parent segment is kept whole, so its dead tail is counted
    in `bytes`.
  * In a chain, `[0, at)` is the parent's bits, and the row's `[0, at)` is a copy of exactly them.
  * When the row's prefill quantized an fp16 parent, `inherit.quantized != quantized`, so all of
    `[0, L)` is copied. The reverse cannot occur: a quantized row stays quantized.
  * Exact repeat (`L == at`): the chain is the parent's pieces; no copy is made. The end-state
    arrays are the parent's own objects, through restore's `list(next(states))`, and are counted
    once.
* **Restore layout.**
  * `KVCache.update_and_fetch` grows only if `prev + n > keys.shape[2]`. The old `[:at]` view had
    `shape == at`, so the first update built `at + round_up(n, 256)`; restore builds the same.
  * `QuantizedKVCache`'s `expand_quant` uses the same formula, with zeros of the component's dtype.
  * The token axis is −2 for packed uint32, scales and biases alike (`(B, H, T, D·bits/32)` and
    `(B, H, T, D/g)`).
  * Buffer capacity never reaches the bits: attention reads `[:offset]`, and `to_quantized`
    quantizes the zero tail per token. Cold and hit capacities already differed on HEAD.
  * `_bytes` (×1.05) is unchanged, and `pc.bytes` is now exact (no 256-rounded views).
* **`first`.** `_slab(reused, len(ids)-1-reused, set(bounds))` equals the first `n` in both
  `_prefill` (`inputs_embeds.shape[1]-1`) and `_spec` (`ctx.shape[1]`). With 0 left it is 1
  (decode, or a DFlash verify of ≤ 16 tokens, inside the 256 tail).
* **No shared buffers.**
  * Restore evaluates the concatenate into a fresh buffer held only by the row's cache object. The
    piece slices are temporaries.
  * `_own_slice` always writes a new buffer. A 1-token piece is a view of its own 2-token buffer.
  * `test_store_copies_out_of_the_row` checks no overlap plus an unchanged pointer after the first
    decode write. M1 (views) fails it, so it would fail on HEAD.
  * `test_restore_layout` checks the first slab writes in place.
* **Refcounting.** Nothing points back: entries → chain → segs, and `_Inherit` → pieces. There are
  no cycles, and segments die with the last chain (A-T4 and A-T5, with weakrefs). `bytes` dedups by
  array id, with quantized tuples flattened in `_Seg.arrays`.
* **`_fits`.**
  * `rows == 0 → True` comes first, so the empty batch always admits.
  * The floor check evicts nothing when that cannot help. The live estimate before evicting is
    `live − (bytes − floor)`, which is exact for unique buffers when no prefill is in flight
    (`_admit` runs only with `task is None`).
  * `peek` has no side effects. Ties go the same way as in `lookup`, so the kept entry is the one
    `lookup` resumes from.
* **`_regrow`.** `keep=(task.parent,)` is guarded against `None`. `row.parent` is cleared in
  `snapshot` (before the store) and in `_end`, which covers cancel mid-prefill, a failed slab, a
  failed setup and finish. With the parent kept, a growing row over the headroom is cut instead.
  That is conservative and correct.
* **`queue_ms` / `t0`.** These are set on the first `_advance` (`row.started`), in both the `_prefill`
  and `_spec` paths. `test_queue_ms_counts_until_the_prefill_starts` uses a fake clock and checks
  both the delayed case and the empty-batch case.
* **`_Req.bounds` memo.** `_bounds` depends only on `ids` and on knobs frozen at load. Images give
  `[]`.
* **Tests.**
  * `check_cold` compares `view()` (KV cut to offset, plus the GDN states) with dtype, shape and
    `mx.array_equal`. It runs at every `positions(e)` of every entry after every request (A-T1:
    fp16 + q8, chains asserted).
  * M4 shows the check is not tautological, even though `Cold` goes through store+restore: a
    common-mode KV corruption changes the hit's GDN states and fails it.
  * The 30-turn test bounds dead tails, `≤ i·(PREFILL_STEP + len(GEN))`.

---

## Nits

* `restore` hardcodes `range(3)` for quantized tuples (worker.py:938–939). Use `len(x[0])`. It only
  matters if a non-affine KV scheme ever lands, and the store's try/except would catch it.
* `_Seg.nbytes` counts a 1-token segment as 1 token while it holds a 2-token buffer. This is
  negligible, but the docstring claims "every buffer once".
* `test_chat_is_one_chain`'s `pc.bytes == per_token·(len+dead) + states` is close to a tautology:
  with one entry, `states` is defined as `sum(e.arrays) − segments`. It checks only that segment
  bytes are per-token uniform. The real content is the `dead` bound and the final `same(...)`.
* `Cold` builds the reference through store+restore. To rule out common mode completely, capture
  the cold row's live caches at `snapshot` (cut to offset) and compare against those.
* The integrated `first` (the `_prompt_cache` computation) is untested. With `small` (slab 8) every
  first slab fits the 256 tail, so `first = 1` would pass every test. This is a perf-only property
  (one extra concat).
* With `QWEN_CLEAR_CACHE=0`, `_fits`'s pre-eviction live estimate counts evicted bytes as freed, but
  they stay in the pool (`get_cache_memory`). So it can evict and then fail the post-check. This is
  a non-default config.
* A dropped new entry is not counted in `evictions`, so Health shows no trace except the info log.
* **Docs.**
  * "16 siblings of an 8K prompt: ~1.38 GB for all 16" (CLAUDE.md, README "~1.4 GB", BACKLOG) is
    the plan's estimate (cascade_plan A.7), not a measurement. Mark it "est." until the GPU
    validation run reports it.
  * CLAUDE.md "A restore holds copies, so eviction frees memory at once" needs "(except while a
    prefill / DFlash run that resumed from it is in flight)" (finding 3).
  * The CLAUDE.md env table `QWEN_BATCH_MEM_GB` still says "− prompt cache". The code no longer
    subtracts it (pre-existing, phase 1).
  * The store bound rule, the `peek` ≥ `QWEN_PREFILL_STEP` rule and the `queue_ms` note match the
    code.
