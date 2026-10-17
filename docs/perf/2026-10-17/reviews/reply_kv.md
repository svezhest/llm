# Review: reply KV in the prompt cache (`QWEN_PROMPT_CACHE_REPLY`)

Date 2026-10-17. Reviewed the uncommitted patch (worker.py, replay.py, README, CLAUDE.md), `src/worker/tests/test_reply_kv.py`
and `docs/perf/2026-10-17/design/reply_kv.md`. CPU only: no worker started, no Metal tests run. Ran on the CPU:
`tests/test_reply_kv.py` (14 passed, 7 s), two mutants of it, one eviction probe, and the design's numerics probe,
which reproduces the doc's numbers. Scratch files are in
`/tmp/scratch/review_reply/`
(`probe/test_probe.py`, `m1/`, `m2/`).

Verdict: the token and position bookkeeping is right in every finish path I traced. The knob-off path is byte-identical.
One real bug: the new store path runs *during* a prefill and can evict that prefill's parent (MED). The rest is LOW.

## Findings

### 1. MED — a reply store evicts the running prefill's parent entry (accounting under-count, `_regrow` invariant broken)

`worker.py:1221` and `:1225` (`store`'s eviction loops, `keep=(new,)`) are called from `_store_reply` (`:3262`).

* **What changed.** Before this patch a store happened only at the end of a prefill. There was no other prefill
  running, and the storing row had already dropped its `parent`. `_regrow` was the only eviction that could run
  between two slabs of a prefill, and it spares that prefill's parent explicitly (`:2578`; its docstring: "never the
  running prefill's parent ... its pieces stay allocated until that prefill stores, whatever the count says").
* **Why the new path is exposed.** `_store_reply` runs from `_drop_done` inside `_iterate`, and `tick()` interleaves
  `_iterate` with the slabs of `task_row`. So a reply store regularly runs while a prefill is mid-way.
* **Why the parent goes first.**
  * `_wanted` spares the task's entry only while `not task.started` (`:2681`).
  * The parent was hit by that prefill's lookup, so it is the most recently used entry.
  * When every entry is live, `victim` evicts the most recently used one first.
* **Probe.** `probe/test_probe.py`, CPU, tiny model:
  1. Conversation B gets its reply entry.
  2. Row D decodes while C (B's next turn, a multi-slab prefill that hit B's entry) is paused between slabs.
  3. D ends, and its reply store evicts C's parent. The log shows `id=d prompt cache reply kept ... (1 entries, 1.9 MB total)`
     while C is still prefilling from it.
  4. C then stores fine, because its `_Inherit` still holds the pieces.
* **Effect.** Correctness is unaffected. But `pc.bytes` drops by the parent's whole KV chain, and those buffers stay
  alive in C's `_Inherit` until C stores. The headroom checks (`_regrow`, `_tidy_choice`, `_charge_fits`, the next
  store's `limit`) then over-commit by up to that size. On the 9B with fp16 KV that is 32 KiB per token: ~0.5 GB at
  16K, 2 GB at 64K, so it can exceed the 1 GiB slack. The eviction also frees almost nothing real, yet it costs an
  eviction and the parent's checkpoints above the resume point.
* **Fix.** In the reply store, keep `task_row.parent` the way `_regrow` does: pass an extra `keep` into `store`, e.g.
  `store(..., spare=(task.parent,))` merged into both evict loops' `keep`. Add a test like the probe that asserts the
  parent stays while the prefill runs.

### 2. LOW — at B = 1 the stored end state can be a view into the previous 2-row stack

`_reply_cache` `:3220` (B = 1 branch, `list(c.cache)`) together with `_gdn_split` `:2225` (`a[row:row + 1]`, a view).

* **Scenario.**
  1. B = 2. Row X stops at the end of iteration j.
  2. `_drop_done` → `_gdn_split` gives row Y slices of the 2-row stack.
  3. Y's `n == max_tokens` was reached in iteration j, so at the start of iteration j + 1 Y ends with `length`
     before any B = 1 step has replaced its state.
  4. Y's reply entry then stores views: the end state pins the whole 2-row buffers (2 × 49 MB on the 9B) while
     `nbytes` counts 49 MB.
* **Also possible** after a demotion or tidy that leaves one row. It is rare, about 1 / (reply length) per 2 → 1
  transition.
* **Fix.** Gather at B = 1 too (`a[mx.array([0])]`, the same 49 MB copy that B ≥ 2 already pays), or gather only
  when the row's state came from `_gdn_split`.
* **Test gap.** `test_batched_replies_match_solo` checks only `a.shape[0] == 1` (test file `:287`). Mutant m1, which
  replaces the B ≥ 2 gather with a slice `a[i:i+1]`, passes all 14 tests. Asserting that the end state does not
  share a buffer is hard from Python. A memory probe could catch it: `mx.get_active_memory()` after dropping the
  batch, with the entry alive.

### 3. LOW — a burst of finishing rows queues 2 command buffers per row between step k and k + 1 (verify on GPU)

`_Seg.copy_from(sync=False)` `:851` and `_store_reply`'s `mx.async_eval(new.end)` `:3272`.

* Each reply store makes two separate `async_eval` calls, and each one ends in a `gpu::finalize`/commit.
* A closed-loop wave of 16 rows finishing in the same iteration therefore puts ~32 small command buffers between
  step k and step k + 1. CLAUDE.md notes MLX keeps only ~10 command buffers in flight.
* If the queue is bounded, `async_eval` blocks the host until step k drains. That is the step-ahead bubble the
  design avoids, once per wave.
* **Fix (cheap).** Collect every finished row's copies and end-state arrays in `_drop_done` and issue **one**
  `mx.async_eval` after the loop. Also fold the end state into the copy's `async_eval`. That way, if the second call
  raises, an entry with an unevaluated end state is never left published.
* Plan item 2 (step_prof A/B) should include a wave where all 16 rows end on the same step, not only 8 rows with
  max_tokens 50.

### 4. LOW — the prompt-part absorb can move checkpoints at positions ≥ the new entry's length

`store` `:1179` (covered via `e.reply`), then `ckpts.pop(L, None)` `:1191` pops only `== L`.

* A reply entry's checkpoints are normally all below `e.reply`. A reply store can absorb a prefix entry longer than
  `e.reply`, though: a `continue_final_message` style prompt that ended inside the same assistant reply. Its own
  slab-end checkpoints then lie inside the reply.
* A later regenerate of the original prompt covers that reply entry through its prompt part. It inherits
  checkpoints `> L`: they are unreachable (`best` needs `p ≤ common ≤ len(ids)`), but `_thin` never drops the
  deepest, so the state arrays stay pinned.
* **Fix.** In the absorb loop, take only `p < L` (or `p ≤ e.reply` for prompt-part covers).

### 5. LOW — "stop strings store nothing extra" is not guaranteed

README / design / CLAUDE.md.

* The handler sets `cancel` only once its `_StopFilter` sees the string. The scheduler may already have ended the
  row with `stop`/`length` (handler lag, or a stop string inside the last tokens).
* The reply is then stored with tokens past the stop string. The client's history holds the truncated text, so the
  next turn falls back, and the stale entry is absorbed by the prompt-part rule.
* Harmless. Fix the wording ("usually") or record `req.p.stop` on the row and skip when it is non-empty.

### 6. LOW — thinking-on requests pay for a reply that is almost never reusable

* With thinking on (the template default), clients rarely send `reasoning_content` back, so the next prompt diverges
  at the reply start. The design calls this "no gain, no loss", but there is a loss.
* The reasoning KV is copied, at 32 KiB per token on the 9B: 128 MB for a 4K-token reasoning. It sits in the cache
  until the next turn's store absorbs it, which raises eviction pressure.
* The prompt-end state is dropped too. A regenerate then re-prefills from the last message start, where today it is
  a full hit.
* Consider skipping rows whose fed tokens contain `</think>` (the request had reasoning), or document it next to
  the knob.

### 7. LOW — observability

* The reply's over-headroom "not kept" uses the shared `prompt cache: entry of N tokens ... not kept` message.
  `scripts/perf/agentic/worker_log.py` therefore counts it in `E["not_kept"]` together with prefill stores, so the
  two cannot be told apart.
* `id=… prompt cache reply not kept: its prompt entry is gone` is parsed by nothing.
* The new `prompt cache reply kept:` line does not collide with `RX_STORE` or with
  `docs/perf/2026-10-16/analyze_worker_log.py`'s `startswith("prompt cache store")` (checked), so the
  HIT/MISS → store prefill-busy intervals are unaffected.
* The `+d decoded` field counts the prompt's last token too (d = generated + 1).
* Suggest a `reply_kept` / `reply_skipped` counter in `worker_log.py`.

### 8. LOW — replay `--own-replies` uses only `message.content`

`replay.py:162`. For traces with the tools API, the parsed `tool_calls` are not carried into the history, and the
recorded message's `tool_calls` stay. Fine for AppWorld (ReAct text). Note it in `--help`, or copy `tool_calls`
too.

## Suspects checked and fine

* **Step-ahead placement.**
  * Every `_drop_done` runs after step k was queued and before step k + 1 is built:
    * end of `_iterate`;
    * start of `_iterate` (cancel / length rows);
    * the sync branch, where step k was already read;
    * `_join`, where the slab's `mx.eval` drained the stream.
  * `_Seg.estimate`, `_layout`, `_Entry.index` and `bytes_of` only read shapes and `nbytes`. Nothing evaluates, and
    no new op lands on a step's outputs after the next step is queued.
  * The GDN gather `a[idx]` and the KV `_own_slice`s are lazy and `async_eval`'d. They queue behind step k, so the
    later `y.item()` waits on step k's event only.
  * There is no donation hazard: the lazy copies hold references to the step-k outputs. No later step writes the
    dropped row's KV, and `_gdn_take` builds new arrays.
* **Feed accounting.** `n = pos − len(prompt)` matches the KV length in every path:
  * step-ahead `stop`: the EOS was fed, `len(fed) == n`;
  * sync `stop`: `len(fed) == n + 1`, sliced by `fed[:n]`;
  * `length` in both modes: the extra step feeds the m-th token, so `len(fed) == n == m`;
  * `max_tokens = 1`;
  * `_regrow` cut, ahead (`len(fed) < n` → skipped) and sync (stored, consistent);
  * forced thinking tokens: replaced `y`, read and fed;
  * `ignore_eos`: the skipped EOS is fed and in `fed`; the text lacks it → fallback;
  * `min_tokens`, single bad words: no change;
  * multi-token bad words: sync;
  * the `_iterate` exception path: `_fail` leaves `why = None` → skipped.

  The offset check (`:3256`) catches the rest.
* **Gates.**
  * `why` is set only by `_finish`. `_fail`, `_fail_rows` and `_close` never reach `_drop_done` with stop/length.
  * DFlash: `row.spec`, also after demotion; demoted rows are `first`, so the DFlash tokens are never in `fed`.
  * Vision: `row.fed` is None.
  * Grouped rows: `_drop_done` runs before `_cascade_tidy` in `_compact`, so a member that ended still has
    `row.group` set. Its `_GroupRef`/None caches fail the `keys` check too.
  * Released rows have contiguous own `KVCache`s (`release` concatenates) at offset P + S.
* **Quantized KV.**
  * Past the start: pieces inherited, `low = kv_start ≤ R`.
  * Crossing during decode: `inherit.quantized != quantized` → copy `[0, pos)`, and the fp16 prompt entry stays as
    the floor (excluded from `covered`).
  * The quantized tuples go through `_own_slice` / `estimate` per part.
* **Resuming at a non-slab end.**
  * `_bounds` depends only on the prefix, so the next prompt's slab ends match the cold ones.
  * The first slab runs `[R, min(R + _step(R), next bound))` and the later slabs reach the next bound. With the 9B
    defaults `_step` is constant 2048, so they reach it in one slab. For the 27B/35B the budget can shrink the step,
    so it can take several slabs, but always ends on a cut.
  * Checkpoints are taken only at cuts.
  * `restore`'s zero tail matches `update_and_fetch`'s rule whatever `at` is (it rounds the first update, not
    `at`), and `first` uses the same `_slab(R, ...)` as the prefill.
  * `best`, `peek` and `_wanted` all go through `_Entry.best`, so admission sees reply hits. The `_peek` memo is
    invalidated by the store's `_recount()` version bump.
  * The resumed store adds no checkpoint at R (`at in ends`). It absorbs the reply entry as a prefix and drops its
    end state (R ∉ ends).
* **Regenerate / sibling / retokenized.**
  * The prompt-part absorb moves only checkpoints `< e.reply`, which are slab ends of the shared prefix and
    therefore valid for the new prompt (but see 4).
  * The absorbed end is guarded by `prefix(e)`.
  * Concurrent same-prompt rows: the later prompt store absorbs the earlier one, which then logs "entry gone". The
    newest reply wins.
* **Memory.**
  * The new entry shares the prompt chain, and `pc.bytes` counts each buffer once.
  * The prompt-end state is freed with the absorbed entry. The net change is +reply KV: the end state is replaced,
    not added.
  * The budget check runs before any copy. "Not kept" returns before `entries` changes and counts one eviction
    (tested).
  * `tick=False` keeps 2 ticks per turn, and `new.seq` / `used` are refreshed to the reply time. That is intended,
    and MRU-by-decode-end is a reasonable "next turn furthest away" proxy.
* **Knob off.** `row.fed` stays None, so `_drop_done` does no reply work and `_iterate` does no append. `e.reply` is
  None, so the `best` and `covered` changes are no-ops. `at in ends` is always true, because a resume point comes
  from `best` over the same `ends` set. `tick=True`, `sync=True`. The only extra work is `row.why` being set:
  byte-identical.
* **replay.py.** `own[len(call["messages"])]` is the next call's index of this reply (phase 0: histories are
  append-only). `own` is per episode and task. The role guard is in place. `--own-replies` with `--ignore-eos on` is
  rejected after the analyze-only early return. With `ignore_eos` off the reply ids equal the text's
  re-tokenization (phase 0: 1344/1344).
* **Numerics doc.** Re-ran `reply_kv_numerics_probe.py`. T = 1 is bitwise. At T = 2, fp32 conv differs by 6e-7 and
  the state by 2.3e-9. 8-bit: conv 0, state 4.4e-6, KV 3–6e-3. This matches the design doc.
* **Tests.**
  * The 14 new tests pass.
  * Mutant m1 (gather → slice) survives: see 2.
  * Mutant m2 (dropping the `at in ends` guard) survives. It is benign, since a checkpoint at R is unreachable or
    still valid, so this is not a gap that matters.
  * Not testable on the CPU: `async_eval` vs `eval` (plan item 2), mrqmm, cascade-released rows.
  * No test covers finding 1.

## Round 2 (fixes re-checked, same day)

Scope: the "Fixes" section of the design doc and the current `git diff` (worker.py, replay.py, both log parsers) plus
`tests/test_reply_kv.py`. CPU only: no worker started, no Metal tests run.

The 8 round-1 findings are fixed as described. Two new findings: one MED (the eviction order), and one LOW that
comes from the `async_eval` fix. Two LOW nits on `spare`.

### R2-1. MED — reply entries become the preferred eviction victim (the MRU order is inverted)

`store` (`new = _Entry(...)`: `used = time.time()`, then `new.seq = self.seq`), as called from `_store_reply`.

* **What the order relies on.** When every entry is live, `victim` evicts the most recently `used` one. The
  rationale: its next turn is the furthest away.
* **Knob off.** `used` is the turn's prefill time. Of the agent episodes, one still decoding has a later `used` than
  one in its tool phase, so the decoding one goes first. That is correct: its next turn is after its decode and its
  tool call.
* **Knob on.** The reply store builds a new entry with `used` = the end of decode. A conversation that just finished
  and is now waiting on its tool output (next lookup imminent) becomes the MRU. It is evicted before conversations
  still decoding: the opposite of the intended order. A whole entry is evicted, prompt part included, because the
  prompt entry was absorbed.
* **Same pass.** When several rows end in one `_drop_done`, each later reply store's MRU victim is the reply entry
  stored a moment earlier in that pass. Its arrays are then still in `kept`, so the single `async_eval` copies data
  that is already evicted.
* **Impact.** It shows only under cache pressure, but that is the target load (AppWorld, `QWEN_PROMPT_CACHE_GB=6`,
  8 episodes).
* **Fix.** A reply store continues the same turn, so it should not refresh recency. Carry the absorbed prompt
  entry's clock: `new.used, new.seq = e.used, e.seq`. Pass them in, or set them in `_store_reply` before any
  eviction can see `new`; the victim is only chosen inside `store`, so it has to be done there, e.g. a
  `clock=(used, seq)` argument.
* **Correction.** In round 1 I called MRU-by-decode-end "a reasonable proxy". It is not.
* **Test.** Two conversations: A decoding, B finished with its reply entry stored. A third store must evict and
  should pick A's entry, not B's.

### R2-2. LOW — if the single `async_eval` fails, the prompt entries are lost too

`_drop_done` (the `except` after `mx.async_eval([... e.arrays])`).

* The failure path removes the reply entries from `pc.entries` and recounts, so no lazy entry stays published.
* But each reply store already absorbed its prompt entry (`covered`) and may have evicted others. Those are not
  restored. The conversation then loses its whole entry, not just the extension: the opposite of the "not kept →
  prompt entry intact" rule that the in-store failure path keeps (`store`'s `except` restores `covered`).
* No `evictions` count is added either.
* This needs an encode-time failure (a Metal allocation): rare, and it costs only the cache.
* **Fix.** `_store_reply` returns `(new, covered)`. On failure, put back the covered entries that are still not in
  `pc.entries`, then recount.
* Do not "fix" it by making the copies synchronous.

### R2-3. LOW — `spare` can leave a reply entry published over the batch headroom without a log line; the soft-cap warning text is now inaccurate

`store`'s two eviction loops (`keep=(new, *spare)`).

* **Before.** The loops ended only at `len(entries) == 1`. For the limit loop that state was proven to fit: the
  pre-check `new.nbytes + extra ≤ limit`.
* **Now.** With a spared entry they can stop at `[spare, new]`, with `bytes + extra > limit`. The entry is
  published anyway, and nothing is logged for `limit`.
* The soft-cap overrun still logs, but as "the newest entry alone … is over", which is not what happened.
* The excess is real memory either way: the spared pieces are pinned by the running prefill. The overrun is bounded
  by the reply entry's own new bytes (its segment + end state − the absorbed prompt-end state, ≈ +reply KV).
* **Fix (both).** Check `bytes_of([new, *spare]) + extra > limit` up front, on the "not kept" path (nothing
  changes). Say "beside the running prefill's entry" in the warning when spare blocked the loop.

**Interaction with keep / wanted: correct.**
* `spare` only removes entries from `left`; `wanted` still orders what remains.
* `wanted()` is still called once and lazily.
* A spared entry that the store covers is absorbed (it is not in `rest`), so `keep` is moot then.
* With `spare=()`, both loops are the old ones: `evict(keep=(new,))` cannot return False while
  `len(entries) > 1`. The prefill store is unchanged.

### Checked, fine

* **(2) Owned end state.**
  * `_reply_cache` gathers `a[idx]` in both paths: `[i]` out of the stack, `[0]` out of the row's own (possibly
    `_gdn_split`-view) arrays. Both make a new buffer, and `None` is kept.
* **(2) Step-ahead.**
  * `copy_from(sync=False)` builds only lazy `_own_slice`s.
  * Every finishing row's store runs first. Then one `mx.async_eval` over `kept` entries' `arrays`. The already
    evaluated prompt pieces and checkpoints in there are no-ops.
  * This still happens after step k is queued and before step k + 1 is built, at every `_drop_done` call site. No
    `mx.eval` is reachable on the reply path: `restore` is not called, and `copy_from` is sync only for prefill
    stores.
* **(2) Lazy arrays read before evaluation.**
  * Later stores in the same pass read only `nbytes` / shapes of lazy arrays (`_Seg.__init__`, `_Entry.index`,
    `bytes_of`, `_Seg.estimate` on the *row's* cache), so nothing forces them early.
  * A later restore (a later tick) finds them evaluated or evaluates them.
  * Dropped entries leave no other strong references: `row.entry` is cleared, and `_peek` memos are weakrefs plus a
    membership check.
* **(3) `p < L`.** It is a no-op for full-prefix covers, where every `p < len(e.ids) ≤ L`. It keeps prompt-part
  covers reachable. Inherited checkpoints are `≤ at ≤ L`, and `pop(L)` stays.
* **(4) Reasoning skip.**
  * The rule mirrors the engine's own detection (worker.py `:4350`): the last of `<think>`/`</think>` in the prompt
    is an open one, or any fed token is a think id.
  * Thinking off: the generation prompt `<think>\n\n</think>\n\n` ends on the close tag, and history turns render
    closed. The answer holds no think ids, so it stores. Test: `test_thinking_off_replies_are_stored`.
  * Thinking on: the prompt pre-opens `<think>` → skipped, and the prompt entry and its end state stay.
  * A forced `</think>` (thinking budget) is in `fed` → skipped.
  * A model that leaks think tokens with thinking off is skipped too, which is harmless.
  * Engines without `_think` (tests) store.
* **(5) Log prefixes and parsers.**
  * The reply over-headroom line now starts `prompt cache reply not kept: over the batch's headroom` (no `id=`).
    Both parsers match it before their `prompt cache:` / `id=` branches.
  * The prefill "not kept" (`prompt cache: entry of …`) stays in `not_kept`.
  * `id=… prompt cache reply kept:` / `… not kept: its prompt entry is gone` / `… not kept: reasoning in the reply`
    each match exactly one counter.
  * `RX_STORE` and `startswith("prompt cache store")` still never match reply lines, so the HIT/MISS → store
    prefill-busy intervals are unchanged.
  * The `+g generated` field is `n`, the generated count.
  * The "row's cache does not match" line is a WARNING and is counted by neither parser. That is fine: it should
    never happen.
* **(8) replay.py.** `mine()` carries our `content` and `tool_calls` and drops the recorded ones. Our tool-call ids
  differ from the recorded `tool` messages' `tool_call_id`. The Qwen template does not render ids, so that does not
  matter here.

**CPU suite** (`--ignore tests/test_mrqmm.py --ignore tests/test_pfqmm.py --ignore tests/test_rattn.py`, run once):
**247 passed, 2 skipped** (6:04), as the fix agent reported.

## Round 3 (R2-1..R2-3 fixes re-checked)

Scope: the design doc's "Round 2" fixes table and the current worker.py (`store` `:1113`–`:1280`, `_drop_done`
`:3214`–`:3256`, `_store_reply` `:3330`–`:3352`). CPU only.

**Nothing above LOW. No new findings.** One cosmetic nit, at the end.

### (1) `clock=` in `store`: correct

* **Applied before any eviction.** It is set right after `_Entry(...)` (`:1212`), before:
  * the `stays` / up-front limit check;
  * the `wanted()` call;
  * the first `self.entries = rest + [new]`.

  So no `victim`/`stale` call ever sees the end-of-decode time.
* **Prefill stores** (`clock=None`): `new.seq = self.seq` and `used = time.time()` from `__init__`, exactly as
  before.
* **Return value.** It changed from `(entry, n)` to `(entry, covered list)`. Its only two callers (the prefill
  store at `:2896` and `_store_reply` at `:3336`) now log `len(covered)`. No test or script calls
  `_PrefixCache.store` directly.
* **Stale rule.** The reply entry carries the prompt entry's `(used, seq)`, which is exactly the clock the prompt
  entry would have had at that moment with the knob off. Its tick age (`self.seq - e.seq` against
  `2 × len(entries)`) and its idle time (`QWEN_PROMPT_CACHE_STALE_S`) are therefore identical to knob-off:
  * The reply entry is never stale "at once" unless the knob-off prompt entry would be stale too. That happens
    after a decode longer than a round of the other conversations' turns, and then it is the right call.
  * `tick=False` keeps a turn at 2 ticks.
  * A sibling lookup that hit the prompt entry during the decode refreshed `e.used/e.seq`, and the reply inherits
    that, which is also right.
  * The next turn's lookup refreshes the reply entry like any hit.
* **Order.** Several replies in one pass now order among themselves by their prompts' times, and a still-decoding
  conversation's entry (prefilled later) is the most recently used. That is the knob-off order, and R2-1 is gone.

### (2) The `async_eval` failure path: correct

* **No half entry.** Every `kept` reply entry leaves `pc.entries`.
* **Restore.** `back` = the absorbed entries that are not themselves `kept` reply entries. This covers the case
  where a later reply store in the same pass absorbed an earlier reply entry: that one is dropped, not restored.
  `back` is re-added unless already present.
  * Absorbed entries are not mutated by `store`: it works on a copy of the checkpoint dict and only refers to the
    pieces. So the restored prompt entries are whole, with their own end state, `used` and `seq`. They are
    evaluated buffers, not lazy ones.
* **Bytes.** `pc._recount()` with `added=True` bumps `version`, so waiting requests' `_peek` memos re-scan; they
  must, because entries came back. Restored ⊆ pre-pass entries, so the bytes never exceed the pre-pass total.
* **Counter.** `evictions += len(kept)`, matching how an over-headroom "not kept" is counted.
* **Accepted limit.** Entries the reply stores evicted to make room stay gone, as in `store`'s own failure path.
* **Nit.** The failure line `prompt cache reply not kept: queueing the copies failed …` is logged at ERROR level
  with no `id=`. It starts with the same prefix as the over-headroom line but continues differently, so neither
  parser counts it: `worker_log.py` matches only the "over the batch's headroom" prefix, and
  `analyze_worker_log.py` reads INFO lines only. That is fine for an error that should never happen; grep for it.

### (3) Up-front "not kept" with `stays`, and the warning text: correct

* `stays` = spared entries that are in the cache and are not absorbed by this store.
  `bytes_of([new, *stays]) + extra > limit` → not kept, and nothing has changed yet: nothing absorbed, evicted or
  copied, and the counter and log are as before. The return is `(None, [])`.
* **Without `spare`** (every prefill store): `bytes_of([new])` is `new.nbytes`, because both dedup arrays by id. The
  old check, byte for byte.
* **The guarantee holds.** The limit loop evicts everything except `new` and `spare`. It can stop at most at
  `[new, *stays]`, whose bytes the up-front check proved `≤ limit − extra`. A published entry is therefore never
  over the batch headroom, which closes R2-3. Spared entries that were absorbed are simply not in `entries`, so
  keeping them is a no-op.
* **Warning text.** `evict` returns False only when every remaining entry is in `keep=(new, *spare)`. So the
  soft-cap overrun warning saying "beside the running prefill's entry" when `len(entries) > 1` is accurate, and
  "alone" covers the rest. The not-kept logs append "beside the running prefill's entry" only when `stays` is
  non-empty.
* **Parsers.** The new not-kept text keeps the `prompt cache reply not kept: over the batch's headroom` prefix, so
  both parsers still match it.

**CPU suite** (`--ignore tests/test_mrqmm.py --ignore tests/test_pfqmm.py --ignore tests/test_rattn.py`, run once):
**251 passed, 2 skipped** (4:44).
