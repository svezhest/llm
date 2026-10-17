# Review: partial store on cancel (A) + chain policy for long fresh entries (B)

2026-10-17. Patch = `git diff f91a207` in worktree `agent-a38526f89553f2b05`
(src/worker/worker.py, tests/test_partial_store.py, README, CLAUDE.md, chain_ckpts_sim.py,
design/partial_store.md). CPU only. Line numbers are the worktree's worker.py.

The CPU suite passes: 264 passed, 3 skipped (406 s). Scratch tests and mutants are in
`/tmp/scratch/review_partial/`
(`test_zz_review.py`, `mutate.py`).

**Verdict.** I found no correctness bug in A: no path breaks hit == cold, stores a
half-updated cache, stores twice, or leaks an event or a reservation. B has two weaknesses in
which checkpoints it keeps. Neither changes the output bits; each costs resume depth.

## Findings

### 1. MED: a retry of a long fresh prompt loses the divergence point and keeps the cancel point
`worker.py:2827` (`split` from `lookup`), `:2901-2903` (`at = below(split)`).

**Scenario.** A fresh entry (≥ `CHAIN_FRESH_TOKENS`) has a sibling, so its divergence point
is s0. The prompt is cancelled at pos > below(s0), then retried.

1. The partial entry P keeps below(s0). Its `fresh` set was built with split = s0.
2. The retry's lookup returns split = max LCP over all entries. P is now one of them, so
   split = pos and `at = below(pos) = pos`.
3. The retry's store absorbs P, then filters with `chain[0] | {pos}`. This drops below(s0),
   the checkpoint that B exists to keep.
4. It keeps pos, which is an arbitrary mid-prompt slab end (~49 MB on the 9B).

**Confirmed** (scratch `test_fresh_divergence_point_lost_after_cancel`, small knobs,
FRESH = 64):

| run | checkpoints |
|---|---|
| uncancelled | `[13, 29, 106, 304, 351, 383, 391]` |
| cancel + retry | `[13, 29, 122, 304, 351, 383, 391]` |

Here 106 = below(s0) and 122 = pos.

This breaks design invariant 5 ("the uncancelled run's entry, or that plus pos unless the
chain filter drops it"): the fresh filter adds pos itself. Even without a sibling, every long
fresh retry keeps one useless state at pos. On the 115K episode (MISS, split 313) the retried
entry keeps a ~110K checkpoint for nothing.

No test covers it: `test_partial_store` runs every cancel case below the fresh threshold, so
under the cap policy.

**Fix.** Choose one:
- Compute the divergence point for `fresh` while ignoring partial entries whose ids are a
  prefix of this prompt (this prompt's own cancelled attempts). For example, give `_scan` an
  `ignore` predicate, or compute a second `split` in `_prompt_cache`.
- Store `at` on the partial entry, and use `parent.at` when `parent.partial` and
  `ids[:len(parent.ids)] == parent.ids`.

Then add the scratch test, with the assertion that `ref.ckpts == retry.ckpts`.

### 2. MED (policy, default on): long fresh entries drop every checkpoint inside the system block
`worker.py:2883-2903`. The `fresh` set is fixed | tail | doc | {at}:
- `fixed` holds only the system block's end;
- `doc` holds only the strides of the first *user* message.

**Scenario.** A fresh entry of ≥ 16K tokens whose system block is long (agentic clients:
15–20K of system prompt and tools). It is followed by a new session whose system block
differs near its end (date, cwd, env inside the system prompt). The new session finds no
checkpoint before the divergence, so it is a full re-prefill. The cap policy had a stride
every 2048 there.

**Confirmed** (scratch `test_long_system_block_sibling`): the sibling resumes at 96 with
FRESH = 0 and at 0 with FRESH = 64. It answers as cold either way.

Only the first sibling pays this: its own entry then gets `at`. Even so, it is ~35 s of 9B
prefill for a 20K prompt, on a knob that is on by default.

**Fix.** Choose one:
- Treat the system block like `doc`: keep its strides, unprotected, so the cap thins them.
  For AppWorld this adds at most 2 states (4.7K system prompt).
- Or document it in the README/CLAUDE.md knob rows ("strides inside the system block are
  dropped") and lower the default.

### 3. LOW: invariant 5 is not exact under the cap policy
`worker.py:1238` (`_thin`), design note invariant 5, and the docstring of
`test_cancel_stores_the_prefix_and_the_retry_resumes_there`.

Below the fresh threshold, the retry's checkpoint set is uncancelled ∪ {pos} *before*
thinning. When there are ≥ 12 candidates, the extra pos makes `_thin` drop a different
checkpoint (the closest pair's left one). That happens at roughly 12–16K prompts at the
production knobs (7 strides, 4 marks, the split, and now pos). The result is still a set of
valid checkpoints, so the bits are fine, but `set(ref) <= set(full)` would fail there.

**Fix.** Word the invariant as "∪ {pos}, thinned to the cap". Or, at the retry's store, drop
the absorbed partial's end state when it is not in `keep`/`want`: it is not a checkpoint any
cold run would take.

### 4. LOW: an exception in `snapshot` outside `pc.store`'s try changes the outcome
`worker.py:2920-2950`, `:3063-3073`.

The `ends`/offset pre-check, the log formatting and `weakref.ref(e)` run outside the try.

- **Cancel path:** an exception there propagates out of the generator to `_advance` and
  `_fail`. A cancelled request then ends with an `Error` event instead of
  `Done(cancelled)`. The gateway tolerates this (the client is gone), but the metrics and
  the log see an error.
- **Failure path:** the store's exception replaces the original one in `_fail`'s message.
  The original stays chained in the traceback only.

None of these can plausibly raise today.

**Fix.** In both early-stop paths, wrap `on_prefilled(cache, pos, why)` in
`try/except Exception: log.exception(...)`. "Prefix not stored" must never change how the
request ends.

### 5. LOW: the offset / `ends` guard is unreachable and untested
`worker.py:2946-2950`. The mutant that removes the guard (`d_drop_guard`) passes all of
`test_partial_store`. The guard is fine as defence in depth, because `intact` and the slab
rule make it unreachable. If it should be tested, call `snapshot(cache, upto=L±1)` directly
in a unit test and assert that nothing is stored and the warning is logged.

### 6. LOW (policy notes, no code bug)
- **DFlash is excluded.** A lone greedy long prompt is exactly the 115K case when nothing
  else runs (`_spec_ok` = no rows and nothing waiting), and `_spec` still stores nothing on
  cancel. This is documented. On the 2026-10-17 run DFlash was not active (rows and
  waiting ≠ 0).
- **A partial entry is speculative.** Its store goes through the normal limit and soft cap,
  so a 3.7 GB 115K prefix can evict live entries of other episodes for a retry that may
  never come. It is protected only while the retry is waiting (`_wanted`). After that it is
  the MRU live entry and goes first. That ordering is right, but the evictions it caused are
  already done. Consider letting a partial store evict only stale entries.
- **`KEEP_WAIT` does not scale with depth (pre-existing).** A queued retry with a 110K hit
  that cannot fit beside running rows waits at most `KEEP_WAIT_S` = 10 s, then evicts the
  partial and re-prefills cold, about 8 minutes. That cap was sized for an 8K hit.
- **The cancel tick does not call `_slab_done`.** The store's copy time is not charged to
  prefill's share, which is negligible.

## Suspects checked and fine

**hit == cold**
- `_bounds` decides a cut at i from ids[i] and `last` only, with i < pos. So any prompt
  sharing ids[:pos] (including prompt[:pos] + [x], which is what `Cold` builds) has the same
  cuts below pos.
- The slab ending at pos has the same shape in both runs (pos − c_k ≤ `_step(c_k)`).
- pos is always a cut, even after a resume at a reply end that is not a cut: the first slab
  stops at the next cut.
- A partial entry resumes only where pos ∈ that prompt's `ends` (`best`).
- Its carried checkpoints are slab ends ≤ pos (`keep` uses `below`, inherited ones are
  ≤ reused), and `ckpts.pop(L)` drops the one at pos.
- `restore`'s zeros tail is sized by the retry's first slab, as for any entry.
- `check_cold` passes in every scratch scenario, including the two below.

**Cancel path**
- The store runs inside `_advance`'s `next()`; a tick runs `_advance` or `_iterate`, never
  both. The only op it adds is the segment copy's `mx.eval`, which waits on the queued step
  like a prefill slab does and adds nothing on a step output. The spy test confirms the call
  stack.
- No double store: the last slab never yields, so a late cancel is seen in `_iterate` after
  the full store, and `_reply_cache` skips `cancelled`.
- `row.parent` is cleared at the top of `snapshot`; `inherit` and `ckpts` are released in its
  `finally` (or with the generator frame when `pos == start`).
- Each store bumps `version`, so the retry's `_peek` memo is refreshed.
- `queue_ms`, `ttft_ms` and `cached_tokens` (`req.reused`) are unchanged.
- The handler yields only `_done(cancelled)`; cancelled requests log no `finish=` line.
- The retry is admitted only after `task` is None, so after the store, and it finds the
  entry.

**Failure path**
- `intact` is False from just before the forward until after `mx.eval` and `pos += n`. So a
  failure in the forward or in `quantize` (whose cache objects may already be swapped)
  never stores.
- GeneratorExit and KeyboardInterrupt are not `Exception`, so nothing is stored then.
- A `pc.store` failure restores the absorbed entries, and the caller logs it.

**Quantized KV**
- Below the start: an fp16 entry with only its end state (`keep` was filtered to ≥ start),
  kept as the floor under the quantized retry (`covered` excludes it).
- Above the start: `low = start`. Quantization runs per group along head_dim, so the
  quantized bits do not depend on which slab quantized them.

**Vision, DFlash, cascade**
- Vision is guarded twice (`use_cache`, and `pixel_values` in `_prefill`).
- `_spec` is unchanged, and its own test passes.
- A retry has `row.cached = reused > 0`, the same cascade eligibility as a cold run that
  stores.

**Flags**
- `chained = any(not e.partial or e.chained)` holds through absorb chains:
  - P1 → P2 → retry stays unchained;
  - E1 → P (chained) → retry is chained.
- Reply stores cover their non-partial prompt entry, so the reply-entry behaviour is
  unchanged.
- A prefill-cancelled row never joins `rows`, so `_drop_done` and `_store_reply` never see
  its `fed` or `entry`.
- With `CHAIN_MARKS = -1`, both `chain` and `fresh` are None, so the cap policy applies.

**Scratch scenarios** (both pass `check_cold`)
- A partial entry that is a prefix of an existing longer entry (a sibling cancelled
  before the shared part ends): stored as its own entry (duplicate KV, harmless), the
  retry resumes there and equals cold.
- Reply KV on, then a cancel, then a retry: the partial absorbs the reply entry, the retry
  absorbs the partial, and `reserved` returns to 0.

**B**
- The threshold `0 < F <= end` (end = len − 1) matches the docs. `test_..._chain_policy`
  pins `<=` (mutant `b_threshold_lt` is caught).
- `at` is always a `_bounds` cut < end, and `want.add(split)` ensures it is taken.

**Sim arithmetic**: 50 148 × 32 KiB = 1643 MB; + 13 × 49 = 2280 MB; + 6 × 49 = 1937 MB.
The chain rows are unaffected, because the fresh entries there are first turns under 16K.

**Log parsers**
- The store line keeps the `prompt cache store: N tokens, …` prefix with the suffix at the
  end. `RX_STORE` in worker_log.py and bench_concurrency.py, and `startswith` in
  analyze_worker_log.py, all still match.
- The `id=… the cancelled prefill's prefix … was not kept` line is not counted twice:
  worker_log counts only the store's own `prompt cache: entry … not kept`.
- Cancelled requests are excluded from `done`, so the store and prefill statistics are not
  skewed.

**Mutants** (`mutate.py`, each run on `test_partial_store`). Every one is caught except
`d_drop_guard` (finding 5):

| mutant | caught by |
|---|---|
| drop `or e.chained` | the [chain] case |
| `<` threshold | the chain-policy test |
| drop `{at}` | the divergence test |
| `partial=False` | the [miss] case |
| no checkpoints in the partial | the [miss] case |
| drop the `intact` guard | the failure test |

## Round 2 (after the fixes)

Same worktree, CPU only. The CPU suite passes: **272 passed, 3 skipped** (317 s). Scratch
tests and mutants are in the same scratchpad: `w2/`, `test_zz_r2.py`, `mutate2.py`.

**Verdict: nothing above LOW.** Fixes 1–5 work. There is one new LOW gap in the
`diverge` carry (R2-1), and two LOW test gaps.

### R2-1. LOW: a carried divergence point past `pos` is lost when the sibling is gone
`worker.py:2873-2880`.

**What happens.**
- `o.diverge` is merged into the retry's `diverge`, which feeds only the fresh *filter*
  (`fresh[0]`).
- It is not added to `want`, so the retry never *takes* a checkpoint there.
- When the cancel came before the divergence point (d > pos), P does not hold d.
- If the sibling was evicted before the retry, then `other` = 0 and `below(other)` is None.
- The retry's prefill passes d without taking a state, so d is lost.

**Confirmed** (scratch `test_diverge_beyond_pos_sparse`, stride 64, sibling diverging
without a mark at 146, cancelled at 138):

| case | checkpoints |
|---|---|
| uncancelled | `[13, 146, 304, …]` |
| retry, sibling kept | `[13, 146, 304, …]` |
| retry, sibling evicted | `[13, 304, …]` |

The output and `check_cold` are fine. `test_fresh_divergence_point_survives_a_cancel[evicted]`
does not catch this, because there d < pos and P already holds it.

**Fix.** Also put the carried points in `want`: `want |= set(o.diverge)` for `o in own`.
`keep` maps them through `below`, so only this prompt's slab ends can be taken. Add the
scratch case as `[evicted-past-pos]`.

### Checked and fine

**(1) `other` / `diverge`**
- **A stale `diverge` position cannot become a checkpoint.** `diverge` only widens the
  filter `policy[0]`. Checkpoints still come only from `keep` (this prompt's `below` →
  `bounds`), from the parent (≤ reused) or from absorbed prefix entries, and all of those
  are slab ends of this prompt. A position from another prompt that matches no existing
  checkpoint has no effect.
- **The carry chains.** For P1 → P2 → retry, P2's `diverge` includes P1's through `own`.
- **Absorb.** P's checkpoint at d moves to the retry, which keeps it because d is in the
  filter.
- **Thinning.** d is unprotected, the same as `at` before the fix.
- **Quantized KV.** Points below the start are dropped by `store`.
- **Clearing.** `diverge` is set only on partial entries (`:2987`) and dies with the entry
  when it is absorbed. Entries built by a full store keep `frozenset()`.
- **A partial of another conversation that is a prefix of this prompt** counts as "own",
  for example a sibling cancelled inside the shared system block. Then its LCP is left out
  of `other` and its `diverge` widens the filter. The cost is at most one extra valid
  checkpoint, or a slightly shallower `other`. It is harmless.
- **The fallback.** `default=0` (only own entries) gives `below(0)` = None, so no point is
  added.

**(2) `sysb`**
- `sysb` is in `fresh[0]` only. `fresh[1]` stays `fixed`, so `_thin` (whose candidates
  exclude `protect`) drops the system-block strides before the system-block end or the last
  starts.
- `chain` is unchanged (fixed | tail | doc), so the 2026-10-16 chain rows of the sim are
  unaffected.
- The new 50K figure checks out: 1643 MB of KV + 8 × 49 MB of states = 2035 MB (7
  checkpoints + the end state).
- When there is no user message, `k` is None and `u0` is 0, so `sysb` is empty: the
  behaviour from before the fix.

**(3) `_store_prefix`**
- **The failure path keeps its own error.** It swallows only the store's exception, and the
  bare `raise` in the outer `except` re-raises the original
  (`test_prefix_store_failure_never_changes_the_outcome[failure]`).
- **A cancel stays a cancel.** `_finish(row, "cancelled")` runs after it, so the request
  ends with `Done(cancelled)`.
- **No double release.** `snapshot`'s `finally` runs once per call and clears `inherit` and
  `ckpts`. `row.parent = None` is its first statement, so a raise cannot skip it.

**(4) Guard test.** `test_prefix_store_guard[not-a-slab-end|offset]` covers both halves of
the guard. Both removing the guard and keeping only its `ends` half are caught.

**Mutants** (`mutate2.py`). Every one is caught except `sysb_protected`:

| mutant | result |
|---|---|
| own partials not ignored | caught: `survives_a_cancel[kept]` |
| `diverge` not carried | caught: `[evicted]` |
| `diverge` not set | caught: `[kept]` |
| `sysb` empty | caught: `long_system_block_sibling[64]` |
| `except` narrowed | caught: `[cancel]` |
| re-raise in `_store_prefix` | caught: `[cancel]` |
| guard off | caught: `[not-a-slab-end]` |
| guard keeps only the `ends` half | caught: `[offset]` |
| `sysb_protected` (`fresh[1] = fixed \| sysb`) | **survives** |

### R2-2. LOW (test gap): nothing pins "`sysb` is thinned before `fixed`"
The code is right, but the `sysb_protected` mutant survives. Add a fresh entry whose
system-block strides plus `fixed` exceed the cap, and assert that every `fixed` position
survives `_thin`.

### R2-3. LOW (test gap): `[evicted]` covers only d < pos
See R2-1.
