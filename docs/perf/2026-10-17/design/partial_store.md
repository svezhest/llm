# Prompt cache: store on cancel mid-prefill (A) + chain policy for long fresh entries (B)

2026-10-17. Motivation: `../REPORT-appworld-run-1017.md` (the AppWorld run of 17.10 on the 9B).

- **115K episode.** A request for a 114 502-token prompt was cancelled by the client's 600 s
  timeout while it was still prefilling. Nothing was stored, so the next attempt prefilled all
  of it again (~8 min each; the report lost 2 × 8 min).
- **50K+ MISS entries.** After a MISS, the new entry absorbs nothing and so kept the cap policy:
  12 GDN checkpoints of ~49 MB each beside 1.6 GB of KV, about 2.3–2.5 GB per entry. Only 3–4
  such entries fit in a 6 GB cache.

CPU-only work. Nothing was measured on the GPU. The numbers below come from the scheduler
simulation (`docs/perf/2026-10-16/chain_ckpts_sim.py`, 9B byte scaling).

## A. Store on cancel mid-prefill

**What.** Take a text prefill that ends early for one of two reasons:

- it was cancelled (the Cancel RPC, a dropped stream, a client timeout);
- it failed outside the forward, so its caches still hold exactly the prefix.

If it completed at least one slab past its resume point, it now stores an entry for the
prefix it holds:

- `ids = prompt[:pos]`, where `pos` is the end of the last completed slab;
- the KV over `[0, pos)`;
- the row's recurrent state at `pos` as the end state;
- the checkpoints taken so far.

It goes through the normal `_PrefixCache.store`: the same budget limit and "not kept" path,
the same absorption, chain/fresh policy, thinning and quantized floor. The `_Inherit` is
released in `snapshot`'s `finally`, as for a full store. A retry of the prompt resumes at `pos`.

**Where (src/worker/worker.py):**

- `:631`: module comment (the policy).
- `:939` `_Entry.partial` / `.chained`: a partial entry, and whether an entry absorbed an older turn.
- `:1142` `_PrefixCache.store(..., fresh=, partial=)`.
- `:1231`: `chained = any(not e.partial or e.chained for e in covered)`. The chain policy
  needs a real older turn. Absorbing a cancelled prefix of the same prompt is not a chain, so
  the retry keeps the policy its prompt would get on a cold run.
- `:2942` `snapshot(cache, upto=None, why=None)` / `:2961` `store(cache, upto, why)` in
  `_Scheduler._prompt_cache`. With `upto` set, the stored ids are `ids[:upto]`. It checks that
  `upto` is one of the prompt's `ends` and that every KV cache's `offset == upto`, and stores
  nothing otherwise. `upto == len - 1` (a failure after the last slab) is a normal full entry,
  not a partial one.
- `:3059–3103` `_prefill`. `intact` is True whenever the caches hold exactly `prompt[:pos]`: it
  is cleared before the forward and set again after the slab's `mx.eval` and `pos += n`.
  - A cancel at the loop top `break`s out, then `on_prefilled(cache, pos, "cancelled")` runs
    if `pos > start`, then `_finish(row, "cancelled")`.
  - An `Exception` while `intact` (`_slab_done`, `on_ckpt`, `clear_cache`) runs
    `on_prefilled(cache, pos, "failed")` and re-raises, so the request fails as before.
  - `GeneratorExit` (the scheduler closing) is not an `Exception`, so nothing is stored then.
  - Vision prompts are skipped (`pixel_values is not None`), and `use_cache` is off for them anyway.
- `_spec` (DFlash) is unchanged: a cancel there stores nothing. The spec asked for this; a lone
  DFlash prefill would be easy to add later.
- Log: the normal store line with ` (cancelled at pos/len)` or ` (failed at pos/len)` appended
  at the **end**. `scripts/perf/agentic/worker_log.py` and `scripts/bench_concurrency.py` match
  `prompt cache store: (\d+) tokens, …`, and `docs/perf/2026-10-16/analyze_worker_log.py` uses
  `startswith("prompt cache store")`. A `prompt cache store (cancelled…):` prefix would have
  broken the first two regexes. A partial entry that is not kept logs the store's own
  `prompt cache: entry of … not kept` line (counted as `not_kept` by worker_log.py) plus
  `id=… prompt cache: the cancelled prefill's prefix (pos/len tokens) was not kept`.

**Invariants.**

1. **Bits = cold.** `pos` is a slab end of the prompt: every slab of `_prefill` ends on a
   `_bounds` cut, because `_slab` stops at the next cut and `_bounds` puts a cut at least every
   `_step`. The store checks it anyway. `_bounds` depends only on the tokens before each cut,
   so a cold prefill of `prompt[:pos] + [x]` cuts `[0, pos)` the same way. The entry's KV, its
   checkpoints (all ≤ `pos`, taken at slab ends) and its end state are therefore the cold bits,
   and `check_cold` confirms it at every resumable position. A retry resumes at `pos`, which is
   in its `ends`, and continues on the same cuts, so its answer equals the cold run's.
2. **Exactly the cached length.** The full store's entry is `prompt[:-1]` (the last token is the
   first decode input). The partial entry is `prompt[:pos]` with `pos ≤ len - 2`. In both cases
   the cache holds `len(ids)` tokens, `end` is the state at `len(ids)`, and `restore(e, len(ids))`
   takes the end state. `restore`'s zeros tail is sized by the retry's first slab as always.
3. **Never in a decode step's window.** The partial store runs inside the prefill generator, so
   inside `_advance`, exactly where the full store runs. In a tick, `_advance` and `_iterate`
   are exclusive. Inside `_iterate` nothing runs between `_step` (queue step k+1) and the
   `r.y.item()` reads. The store's `mx.eval` of the segment copies waits on the queue like a
   prefill slab's own eval; it adds no op on any step output. The test checks the call stack:
   `_advance`, not `_iterate`.
4. **Budget.** Same `limit = budget + prefill_reserve - reserved`, with the cancelled row still
   reserved. That is conservative: while the copy is made, the row's KV is still alive.
5. **Chain policy is unaffected by a cancel.** Under the chain and fresh policies the retry's
   checkpoints equal the uncancelled run's (the filter drops `pos`; tested for a long fresh
   prompt, `test_fresh_divergence_point_survives_a_cancel`). Under the cap policy the retry
   has the uncancelled candidates plus `pos` (the partial entry's end state becomes a
   checkpoint), thinned to the cap. Below the cap that is "uncancelled ∪ {pos}" (tested for
   MISS / HIT / chain). Past the cap `pos` takes part in the thinning, so the set is only a
   subset of "uncancelled ∪ {pos}" (the closest pair's left one can differ). It is still a set
   of valid checkpoints, so the bits are unaffected (review finding 3).
6. **Quantized KV.** A prefix below `quantized_kv_start` has not been quantized yet, so it is a
   plain fp16 entry with the cold run's fp16 bits; the quantized retry keeps it as the floor
   entry (existing rule). A prefix past the start is a quantized entry with `low = start`.
   Both are tested.

## B. Chain policy for long fresh entries

**What.** `QWEN_PROMPT_CACHE_CHAIN_FRESH_TOKENS` (default 16384, 0 = off) sets a length
threshold. A fresh entry is one that absorbed no chain: a MISS, a new episode, or a sibling hit
that absorbs nothing. If it has at least that many tokens (`len(prompt) - 1`; for a partial
entry, the whole prompt's length, so the partial entry and its retry follow one policy), it
keeps the chain set:

- the end of the system block;
- the last `CHAIN_MARKS` message starts;
- the tail strides;
- the strides inside the first user message;
- **plus the divergence point** from the closest other entry. This prompt's own partial
  entries are ignored, and their `diverge` sets are carried over (review finding 1);
- **plus the strides inside the system block**, everything before the first user message.
  They are unprotected, so the cap thins them (review finding 2).

The last two items go beyond the spec.
- The divergence point is the one checkpoint a fresh entry learns from the cache. It costs at
  most one state and usually coincides with the end of the system block;
  `test_long_fresh_entry_keeps_the_divergence_point` covers the case where it does not.
- The system-block strides serve a new session whose long system prompt (tools, date, cwd)
  differs near its end. Without them it would re-prefill from 0.
Shorter fresh entries keep the cap policy. `CHAIN_MARKS = -1` still turns everything off.

**Where.** `_prompt_cache` builds `diverge` (`:2865–2884`) and `fresh` (`:2918–2929`), and
`store` applies `policy = chain if chained else fresh`.

**Number.** `SIM_FRESH=1 python docs/perf/2026-10-16/chain_ckpts_sim.py` (production knobs) on
an AppWorld-shaped fresh entry of 50 148 tokens:

| `CHAIN_FRESH_TOKENS` | checkpoints | 9B MB (KV + states) | states MB |
|---|---|---|---|
| 0 (before) | 12 | 2280 | 637 |
| 16384 (first patch: no system-block strides) | 5 | 1937 | 294 |
| 16384 (after the review fixes) | 7 | 2035 | 392 |

That is −245 MB (−11 %) per 50K MISS entry. The 2 extra states are the strides at 2048 and
4096 in the 4.7K AppWorld system block. KV dominates (1.64 GB), so this does not let a
6 GB cache hold noticeably more 50K entries. (The report's 2.5 GB was for 54–58K prompts.)
The simulation's chain rows are identical to the 2026-10-16 run.

## Tests

`src/worker/tests/test_partial_store.py` (25 tests, ~90 s on CPU; the 20K test takes ~27 s):

- `test_cancel_stores_the_prefix_and_the_retry_resumes_there[miss|hit|chain]`: entry ids
  `prompt[:pos]` at the 3rd slab end, end state, `partial` / `chained`, log line (RX_STORE +
  suffix), `check_cold`; the retry resumes at `pos`, output == cold, the partial entry is
  absorbed (`absorbed=1`), and the checkpoints equal the uncancelled run's (∪ `{pos}`; below
  the cap).
- `test_cancel_before_the_first_slab_stores_nothing[miss|hit]`.
- `test_partial_store_is_bounded_by_the_budget`: kept (others evicted), or not kept with the
  cache unchanged (one eviction counted).
- `test_partial_store_beside_decoding_rows`: decode and prefill ticks alternate; the decoding
  row's output == solo, the store runs from `_advance` (call stack), and the retry == cold.
- `test_dflash_prefill_cancel_stores_nothing` (+ the same prompt on the plain path stores).
- `test_failure_outside_the_forward_stores_the_prefix`: a `_slab_done` failure stores
  (`failed at`); a failure inside the forward (`quantize`) never tries.
- `test_quantized_kv_partial[below-start|quantized]`.
- `test_long_fresh_entry_takes_the_chain_policy`: at the threshold, exactly the chain set; one
  token short of it, and with the knob at 0, the cap set (12, with deep strides). `check_cold`,
  then the next turn == cold.
- `test_long_fresh_entry_keeps_the_divergence_point`.
- `test_20k_fresh_entry_at_production_knobs`: 20K AppWorld-shaped prompt, the chain set plus
  the 2 system-block strides (≤ 8) vs 12, `check_cold` at every kept position. A 4K prompt keeps the cap set. At these knobs the
  two sets coincide for 4K: the 4096 tail covers every stride.

Mutation checks (by hand):

- `chained = bool(covered)` fails the miss/hit cases.
- Dropping the `intact` guard fails the failure test, which asserts that the store is never
  attempted rather than relying on the offset check.

Full CPU suite: see the summary.

## Fixes after review (docs/perf/2026-10-17/reviews/partial_store.md)

Line numbers are for `src/worker/worker.py` after the fixes. Tests are in
`tests/test_partial_store.py`, under the `review 2026-10-17` header.

1. **MED: the divergence point was lost after a cancel and retry.**
   - Fix (`:2865–2884`): the retry's split counted its own partial entry, so it was `pos`. The
     divergence point is now `below(other)`, where `other` is the deepest common prefix with
     every entry except this prompt's own partial entries (prefixes of it); `want` gets
     `other` too.
   - A partial entry keeps the divergence points its prompt was planned with
     (`_Entry.diverge`, `:940`, set at `:2992`). The retry adds them, so it keeps them even if
     the sibling was evicted in between.
   - Test: `test_fresh_divergence_point_survives_a_cancel[kept|evicted]`, the reviewer's
     repro. The retry's checkpoints equal the uncancelled run's exactly, the answer equals
     cold, and `check_cold` passes.
2. **MED: long fresh entries dropped the system block's strides.**
   - Fix (`:2928`): `sysb` holds the strides before the first user message (by role token;
     tests: `starts[1]`). It is part of `fresh[0]`, not `fresh[1]`, so the cap thins it like the
     document strides. The chain policy (absorbed turns) is unchanged; see open item 4.
   - Test: `test_long_system_block_sibling[0|64]` (the reviewer's scratch case). The sibling
     resumes at the last kept stride before the divergence: 128 with the fresh policy (96 under
     the cap policy, 0 before the fix), as cold.
   - `test_20k…` now expects the 2 system-block strides; the simulated 50K entry has 7 states,
     2035 MB.
3. **LOW: invariant 5 wording.** Invariant 5 above now says "thinned to the cap" for the cap
   policy and exact equality for the chain and fresh policies; so does the docstring of
   `test_cancel_stores_the_prefix_and_the_retry_resumes_there`.
4. **LOW: a snapshot exception could change the outcome.**
   - Fix: `_Scheduler._store_prefix` (`:3114`) wraps the early-stop `on_prefilled` in
     `try/except Exception: log.exception`. Both early-stop paths call it (`:3095` failure,
     `:3099` cancel), so a cancel stays `Done(cancelled)` and a failure keeps its own error.
   - Test: `test_prefix_store_failure_never_changes_the_outcome[cancel|failure]`, with a
     raising snapshot.
5. **LOW: the guard was untested.** The guard is `store`'s `ends`/offset check (`:2972`).
   - Test: `test_prefix_store_guard[not-a-slab-end|offset]` calls the real snapshot with
     `pos + 1` (not a slab end) and with the next slab end (an offset mismatch). Nothing is
     stored and the warning is logged.
6. **LOW: policy notes.** Recorded in the open list below.

Mutants (`scratchpad/pstore/mut.py`) on the new tests are all caught:
- ignore own partials off;
- `diverge` not carried;
- `sysb` empty;
- the `_store_prefix` except narrowed;
- the guard removed.

### Round 2 (review "## Round 2": nothing above LOW)

- **R2-1 / R2-3: a carried divergence point past the cancel point was never taken.**
  - The case: the prefix ends before d (`pos < d`, so the partial entry holds no state at d)
    and the sibling is evicted before the retry. The carried point only widened the fresh
    filter, so nothing took a checkpoint at d.
  - Fix (`:2884`): `want |= set(o.diverge)` for each of the prompt's own partial entries.
    `keep` maps the points through `below`, so only this prompt's slab ends are taken.
  - Test: `test_divergence_point_past_the_cancel[kept|evicted]`, the reviewer's scratch case
    (stride 64, a sibling diverging without a mark at 146, cancel after one slab). The retry's
    checkpoints equal the uncancelled run's exactly (`[13, 146, 304, …]`), the answer equals
    cold, and `check_cold` passes. Without the line, `[evicted]` fails.
- **R2-2: nothing pinned that the system-block strides are thinned before `fixed`.**
  - Test: `test_system_block_strides_thin_before_the_marks` (cap 5, stride 32, a 150-token
    system block, the last three starts a few tokens apart). The entry keeps all 4 `fixed`
    positions plus one system-block stride. The `sysb_protected` mutant
    (`fresh[1] = fixed | sysb`) fails it: with everything protected, `_thin` falls back to
    the closest pairs, which are the marks.
- Mutants (`scratchpad/pstore/mut2.py`): no `want |=` (caught by `[evicted]`) and
  `sysb_protected` (caught by `thin_before_the_marks`).

## Open

1. **DFlash excludes exactly the 115K case.** A lone greedy long prompt with nothing else
   running or waiting takes `_spec`, and a cancel there stores nothing. On 2026-10-17 DFlash
   was off and rows/waiting were ≠ 0, so the run was unaffected. Adding it means calling
   `on_prefilled(cache, pos, "cancelled")` in `_spec`'s slab loop. The KV there comes from
   `lm.model` with `capture_layer_ids`; this needs a check that its bits equal `_prefill`'s
   (the full DFlash store already assumes they do).
2. **A partial entry is speculative.** Its store goes through the normal limit and soft cap,
   so a 3.7 GB prefix of a 115K prompt can evict live entries of other episodes for a retry
   that may never come. It is protected only while the retry waits (`_wanted`); after that it
   is the most recently used live entry and goes first, but the evictions it caused are done.
   Option: a partial store evicts only stale entries, and is "not kept" otherwise.
3. **`QWEN_PROMPT_CACHE_KEEP_WAIT_S` does not scale with depth (pre-existing).** It is 10 s,
   sized for an 8K hit. A queued retry with a 110K hit that cannot fit beside running rows
   waits at most 10 s, then evicts the partial entry and re-prefills cold (~8 min). Option:
   scale the wait with the depth (e.g. 10 % of the prefill the hit saves).
4. **The chain policy (absorbed turns) still drops the system block's strides.** A new
   session differing near the end of a long system prompt finds them only in a fresh or
   cap-policy entry. Extending `sysb` to chains changes the 2026-10-16 chain numbers (+1–2
   states per AppWorld entry), so it was not done here.
5. **The cancel tick does not charge `_slab_done`** for the store's copy time. This is
   negligible.
