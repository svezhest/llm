# Review: chain checkpoint policy (A) + eviction order (B)

Patch: `cpu_agent_AB.patch` (uncommitted diff of `src/worker/worker.py`, `tests/test_prompt_cache.py`,
`tests/test_segments.py`, README, CLAUDE.md). Reviewed on the working tree (identical to the patch).
CPU only; nothing ran on the GPU. Line numbers are `src/worker/worker.py` of the working tree.

CPU suite (`uv run --group dev pytest -q --ignore tests/test_mrqmm.py --ignore tests/test_pfqmm.py
--ignore tests/test_rattn.py`): **212 passed, 2 skipped** (4 min 29 s).

Throwaway experiments (not part of the repo):
`/tmp/scratch/review_AB/`
(`test_review_ab.py`: findings 1–3 reproduced; `time_wanted.py`: finding 4; `mut.py` + `mut/*`: the
mutation runs below).

## Findings

### 1. HIGH — `_wanted` iterates `self.waiting` without the lock: a concurrent submit raises, and the raise fails the whole batch

`worker.py:2584` (`for req in self.waiting:` in `_Scheduler._wanted`).

`self.waiting` is a `collections.deque` that gRPC handler threads append to (`submit`, 2332, under
`self.cv`). Every other scheduler-side *iteration* of it takes `self.cv` (`tick`, 2430: "handlers
append meanwhile"); `_admit` only indexes / pops, which is safe. `_wanted` iterates it bare, and each
step calls `_peek` → `pc.peek` → `_lcp` over long id lists (milliseconds after any store, finding 4),
so the interpreter switches threads mid-iteration (5 ms switch interval) and an `append` from a handler bumps the deque's state →
`RuntimeError: deque mutated during iteration` on the next step. Reproduced deterministically
(`test_wanted_iterates_a_deque_others_append_to`: a submit between two peeks).

It is reached exactly under load (arrivals) with the cache in the way (evictions):

- from `_fits` (admission, 2555) and `_regrow` (decode growth, 2493): the exception leaves `tick()`;
  `_run` logs "batch step failed", `_fail_rows(e)` fails **every running row**, and with several
  rows it counts as a strike — three in a row and the worker closes with `worker_fault` and exits 70;
- from `store` (via `wanted=self._wanted`, 2756): caught by `_prompt_cache.store`'s `except`, but
  it leaves a half-built entry in the cache (finding 2).

Fix: snapshot the queue under the lock, e.g.
`with self.cv: reqs = list(self.waiting)` and iterate `reqs` (or `list(self.waiting)`, which copies
in C without releasing the GIL — still take the lock for consistency with `tick`).

### 2. HIGH — an exception in `store`'s eviction loops leaves a half-built entry: later hits restore a too-short KV (silent wrong output) or crash

`worker.py:1139–1148` vs the rollback `try` at 1149–1157.

`store` publishes `new` (`self.entries = … + [new]`, 1139) while `new.chain` is still only the
parent's pieces `[0, lo)`; the segment for `[lo, L)` is attached after the two eviction loops. The
rollback (`except BaseException:` → covered entries back, `new` out) wraps only the copy. Before the
patch `evict` could not raise; now each eviction may call `wanted()` → `_Scheduler._wanted` →
`_peek`/`pc.peek`, which can raise (finding 1, or any bug/MemoryError there). Then `new` stays in the
cache with no KV past `lo`, the covered (absorbed) entries are already gone, and `bytes`/`nbytes`
undercount it. Its checkpoints and end state are real, so `best()` happily resumes from it:

- fresh entry (`lo = 0`, chain `[]`): every later hit → `restore` → `IndexError: list index out of
  range` (the request fails; the next turn / regenerate of that conversation fails every time until
  the entry is evicted). Reproduced: `test_store_raising_in_eviction_leaves_a_half_entry`.
- resumed entry (`lo = reused`): `restore(e, at > lo)` concatenates only `[0, lo)` + the zeros tail
  and sets `offset = at` — **silently wrong output**, finish `length`, no error. Reproduced:
  `test_half_entry_with_a_parent` (regenerate of turn 2: `[479, 349, 208, …]` vs cold
  `[479, 369, 373, …]`).

This violates the "hit == cold bitwise" invariant. Fix (both, belt and braces):
(a) compute `wanted` *before* mutating anything (it is only needed if an eviction will happen:
`extra`, `budget`, `limit` and `bytes_of(entries − covered) + new.nbytes` are all known before 1139,
so evaluate the callable eagerly when `bytes_of(...) + extra > min(budget, limit)`), or make
`victim`/`wanted` failures non-fatal (`try: wanted = wanted() except Exception: log; wanted = ()`);
(b) extend the rollback `try` to cover 1139–1159 (restore `covered`, drop `new`, `_recount()`),
so no exception between publish and `new.index()` can leave a half entry. A test: a `wanted`
callable that raises during `store` must leave `pc.entries` either unchanged or fully built
(`check_cold` passes).

### 3. MED — a head that gives up its hit evicts *other* waiting requests' entries first, in vain

`worker.py:2541–2556` (`_fits`, the give-up branch) with `victim` (1009–1025).

When only the head's own entry H keeps it out (`blocked(floor_H)` and not `blocked(0)`) and the
head gives up (`depth < KEEP_TOKENS` or the wait expired), `keep = ()` and the loop evicts in
`victim` order. H is the head's resume entry, so `_wanted()` lists it: it is now "wanted" like the
second-in-line's entry S and goes *last*, after every unwanted entry and in MRU/stale order among the
wanted. But `blocked(floor_H)` already proved H must go (it alone is over the budget), and once H is
gone the rest may fit. Reproduced (`test_give_up_evicts_others_wanted_entry_first`): H 120 tokens,
S 20 tokens (S the MRU), budget = rows + need + (H+S)/2: S is evicted first (useless), then H —
both gone, where evicting H alone admits the head. Pre-patch LRU evicted H (older) first and kept S,
so this is a regression for the second request in line (and for every unwanted entry that goes
before H).

Fix: on give-up, evict H first explicitly (`pc.entries` remove via `pc.evict(keep=[x for x in
pc.entries if x is not e])`, then the normal loop), or pass `wanted` minus `e` so H sorts as the
first unwanted victim. Add the repro as a test.

### 4. LOW — `_wanted` cost: a cold re-peek of the whole queue per store under pressure, and twice per store

`worker.py:1140/1159` (`_recount()` bumps `version` twice per kept store), `_peek` 2563–2577.

Every store bumps `pc.version` (at publish *and* again after the copy, 1159 — that second bump is
not an add: the entry's ids/checkpoints/end did not change, so no peek can change). After a bump the
next eviction (`store`'s own, then `_fits` / `_regrow` on the next tick) re-peeks every waiting
request against every entry: `_lcp` is a slicing binary search, O(L log L) per pair. Measured
(`time_wanted.py`, pure Python, 8 entries of 40K ids sharing a 6K system prefix): **302 ms for 32
waiting requests** (~9.4 ms per request), i.e. ~75 / 150 ms at 8 / 16 waiting — on the scheduler
thread, while every decode row waits; under eviction pressure that is paid up to twice per store.
Only when evictions happen, and the gateway keeps the worker's queue near `max_batch`, so LOW; but
it is avoidable:

- `_recount(added=False)` for the post-copy recount at 1159 (halves it; the rollback path at 1156
  must keep `added=True`);
- cap `_wanted` to the first `max_batch` waiting requests (the ones that can be admitted before the
  next store anyway);
- optionally, `_lcp` on a per-request memo of `(entry → lcp)` (entries' ids are immutable).

### 5. LOW — staleness clock advances only on stores: with no text stores (vision traffic, a quiet cache) old entries stay "live" and MRU evicts the newest

`worker.py:1022` (`self.seq - e.seq > len(self.entries)`), `seq` incremented only in `store` (1081).

Vision requests never store, and lookups (hits) only refresh `e.seq` without advancing the clock. So
after a burst of text conversations followed by a vision-only period (or any long gap), every entry
is still "within the round" however old it is, and `_regrow`/`_fits` evictions for the vision rows
take the **most recently used** entry first — the opposite of what an idle cache wants. Same with
few entries and slow traffic: "the last round" means "the last N stores", not time.

Fix: advance the clock on every text lookup too (`lookup`: `self.seq += 1` — one per request, hit or
miss, so a chain turn costs two ticks; scale the rule to `> 2 * len(entries)`), and/or treat an
entry as stale when `time.time() - e.used` exceeds a few minutes.

### 6. LOW — chain policy also fires on a *repeat* of a fresh prompt and on the first follow-up of a long-document conversation; deep strides lost for sibling conversations

`worker.py:1117` (`if covered and chain is not None`).

`covered` is "any entry whose ids are a prefix of the new one", not "the previous turn": an exact
repeat (regenerate, an identical request in a burst) and the first follow-up turn both trigger the
chain policy. For a fresh entry whose first user message is long (a 30K-token document + question),
the next turn / repeat drops every stride checkpoint inside that message (only the system-block end,
the last 3 message starts and the strides in the last 4096 tokens stay). A *different* conversation
over the same document whose question diverges more than 4K tokens before the end then resumes at
the end of the system block instead of a stride ~2K below the divergence: up to the whole document
re-prefilled (minutes at 32K+). Pre-patch the absorbing entry kept up to 12 of those strides. The
test `test_fresh_entries_keep_the_cap_policy` asserts this drop as intended, and the bits stay
correct — it is a policy trade-off, but README/CLAUDE.md present the deep strides as "dead weight"
without this caveat.

Options: apply the chain filter only from the second absorption on (the covered entry itself
absorbed something: carry a `chained` flag on `_Entry`), or keep the strides of the *first* user
message (`starts[1]..starts[2]`, the shared-document region) in `chain[0]`, or at least document the
case.

### 7. LOW — the admitted-but-not-started request's entry is unprotected

`worker.py:2489–2493` (`_regrow`'s `keep`), `_wanted` 2580.

Between `_admit` (the request leaves `waiting`) and its first slab in `_advance` (with rows running
it waits for decode's share, `resume`), its lookup has not run, so `task_row.parent` is `None` and the
request is not in `waiting`: `_regrow` can evict the very entry it is about to resume from — not even
"wanted"-last. Pre-existing (the old LRU had the same hole), but the new "wanted go last" order is
the natural place to close it: in `_wanted`, include `self.task_row.req` while `not
self.task_row.started` (or let `_regrow` keep its `_peek` entry).

### 8. LOW — tests that do not catch the mutations they suggest

Mutation runs against `tests/test_prompt_cache.py` + `tests/test_segments.py` (copies of the worker
in `review_AB/mut/<name>`, `-x`):

| mutant | change | result |
|---|---|---|
| `noprotect` | drop `protect = chain[1]` | **survives** (39 passed) |
| `noversion` | `_recount` never bumps `pc.version` | **survives** (39 passed) |
| `nodepth` | `_wanted` without `depth >= PREFILL_STEP` | **survives** (39 passed) |
| `nowanted` | `victim` ignores `wanted` | caught (`test_victim_order`) |
| `nopresence` | `_peek` memo skips the presence check | caught (`test_admission_spares_…`) |
| `nomemo` | `_peek` never uses the memo | caught (`test_admission_spares_…`, asserts memo identity) |
| `lru` | live entries LRU instead of MRU | caught (`test_admission_evicts_the_cache_on_demand`) |
| `chainall` | chain filter also on fresh entries | caught (`test_fresh_entries_keep_the_cap_policy[chain]`) |

- `noversion` matters most: with no bump on add, a waiting request whose memo says "no hit" (or a
  shallow entry) keeps that answer after a store creates a deeper entry for it, so `_fits` would not
  keep that entry and `_wanted` would not protect it. Nothing tests the add path of the memo. A test:
  memoize a waiting request's peek, store an entry it would resume from (deeper), assert `_peek`
  returns the new one and `_wanted()` lists it.

- `protect = chain[1]` is never exercised: with the test knobs a chain keeps ≤ 4 marks + ≤ 3 tail
  strides < `PROMPT_CACHE_CKPTS` (12), so `_thin` never runs on a chain entry. A test with
  `PROMPT_CACHE_CKPTS=4` and a long tail (`CHAIN_TAIL` large) would check that thinning drops tail
  strides, never `fixed`.
- The quantized filter of `chain` in `_prompt_cache` (`chain = tuple({b ≥ start} …)`, 2729) is
  redundant: `store` filters the checkpoints by `kv_start` after the chain filter, and `protect`
  entries below the start are never present. Harmless; no test can catch its removal.
- `nodepth`: no test has a waiting request with a shallow (< one slab) hit, so the `wanted`
  threshold is unchecked.

### 9. LOW — stale "LRU" wording left behind

- `worker.py:913` `_PrefixCache` docstring "The entries, LRU."; `worker.py:625–629` header comment
  ("LRU-evicted at store", "an admission evicts LRU entries"); `_regrow` docstring 2471 ("LRU
  entries evicted"); `_fits` docstring 2514 ("its LRU entries are evicted").
- `CLAUDE.md:444` (Continuous batching invariants): "give way first (LRU evicted on demand)".
- CLAUDE.md "`_fits` … spares the entry `peek` says … (memoized `req.bounds`)": now also memoized
  `req.peeked` (`_peek`).

## Suspects checked, no issue

- **Checkpoint validity under the chain filter.** `chain[0]` is built only from `below(...)` of
  *this* prompt's `bounds` (slab ends, `< end`), and the filter only *removes* keys from a dict whose
  entries were already valid (own checkpoints at slab ends, inherited ones `≤ reused`, absorbed ones
  at slab ends of a shared prefix). A subset of valid checkpoints stays valid; `fixed` cannot hold a
  non-slab-end. `inherit.end` placed at `at` is dropped or kept, never moved. Byte-identity tests
  (`check_cold` in the chain tests) pass.
- **Quantized `low` filter.** `store` still filters `ckpts` by `kv_start` *after* the chain filter;
  `covered` never contains an unquantized entry under a quantized one, so the fp16 floor entry is
  untouched by the chain policy (`test_quantized_floor_entry_survives` passes).
- **`_Inherit` and dropped parent checkpoints.** The in-flight prefill already restored from the
  parent; its `_Inherit` carries its own copies of the pieces / checkpoint dict, and only one prefill
  runs at a time, so no other store can thin a parent under it. `_regrow` still keeps
  `task.parent`; `_fits` only runs with no task.
- **`_thin` protect semantics.** `protect = chain[1]` (= `fixed`, quantized-filtered) is a subset of
  the remaining keys or absent ones; `_thin` falls back to dropping protected ones only when nothing
  else is left (CHAIN_MARKS > CKPTS), never the deepest. With defaults a chain holds ≤ 7 < 12, so
  `_thin` is a no-op there (see 8).
- **`CHAIN_MARKS = 0`** handled (`starts[-0:]` would be the whole list; the code special-cases it).
  `-1` = off restores the fresh policy everywhere (`test_fresh_entries_keep_the_cap_policy[off]`).
- **Peek memo correctness.** `_scan` returns the *first* entry with the strictly deepest position,
  so removing any other entry cannot change the result, and a missing hit (`None`) cannot appear by
  removal; every add (publish, absorb, rollback) goes through `_recount()` with `added=True`; the
  memo re-validates presence with identity, and `_Entry` supports weakrefs (no `__slots__`). An
  absorb is an add (new entry published, version bumped) — a waiting request whose memo pointed at
  the absorbed entry re-peeks. Correct, only over-invalidating (finding 4).
- **`victim` with an empty cache.** `evict` returns `False` before calling `victim` when `left` is
  empty; `store`'s loops guard `len(entries) > 1` with `new` kept, so `left` is non-empty.
- **`seq` overflow / ordering.** Python ints; `seq` only grows. Lookups set `hit.seq = self.seq`
  (no increment), ties broken by `used`. A "not kept" store still advances the clock (counts as a
  round step) — consistent with the docs. The lazy `wanted` memo inside `store` may list an entry
  that the same store already evicted (harmless) and miss a request's *next* best entry after its
  wanted one went (minor; only within one store).
- **KEEP / KEEP_WAIT interplay** (other than 3): unchanged logic; `_peek` returns exactly what
  `pc.peek` did; `keep_since` semantics unchanged; the head's kept entry is excluded from `left`.
- **`_fits` / `_regrow` never evict the running prefill's parent**: `_regrow` passes the same
  `keep`; `wanted` only reorders within `left`.
- **Log format.** `states=%.1f MB` is appended after `pieces=`; every parser matches a prefix:
  `scripts/bench_concurrency.py:179`, `scripts/perf/agentic/worker_log.py:22`,
  `docs/perf/2026-10-16/analyze_worker_log.py:27` (`startswith("prompt cache store")`),
  `tests/test_prompt_cache.py` `RX_STORE`. The new eviction line is DEBUG and does not collide with
  the INFO "entries evicted to admit/for the growth" lines `analyze_worker_log.py` counts.
- **`state_bytes`** counts every array once (ckpt states shared between entries / with `end`); the
  test's `(len(ckpts)+1) * one` holds because the end state is always fresh arrays.
- **README / CLAUDE.md vs code** (besides 9): chain contents, defaults (3 / 4096, `-1` off), the
  fresh/absorbing split, victim order (wanted last, stale `pc.seq − e.seq > len(entries)` LRU, else
  MRU), `version` bumping on adds, and the `states=` log field all match the code. The 932 MB
  (README, measured handoff) vs 1012 MB (CLAUDE.md, simulation) figures describe different sources;
  both are labelled.

## Round 2

Re-review of the fixes (`chain_evict_fixes.md`; diff vs HEAD `cpu_agent_AB_fixed.patch`). Line
numbers are `src/worker/worker.py` of the working tree after the fixes. CPU only.

CPU suite (same command): **222 passed, 2 skipped** (3 min 55 s).

Round-1 repros re-run on the fixed code (`review_AB/test_review_ab.py`): all four assertions of the
bugs now fail for the right reason — `_wanted` no longer raises on a concurrent submit; a raising
`wanted` during a store leaves a fully built entry (KV over all 45 / 115 tokens, the regenerate
answers exactly as cold: `[479, 369, 373, 209, 464, 250]`); the give-up evicts only H (S stays,
`evictions == 1`).

### R2-1. MED — the vision clock ticks (fix for 5) bring back the round-robin worst case under mixed traffic

`worker.py:2697` (`eng._pcache.seq += 2` for a vision request), with the rule at 1032
(`self.seq - e.seq > 2 * len(self.entries)`).

The clock now counts *requests*, not rounds of the entries' own conversations. Any traffic that does
not touch the entries — vision requests by construction — lengthens the gap between two turns of
the same episode in ticks without adding entries, so live episodes cross `2 × len(entries)` and
become "stale" while their next turn is still coming; stale entries go LRU-first, and LRU is exactly
the pathology the order was built to avoid. Reproduced (`review_AB/test_round2.py`: the repo's
`test_round_robin_episodes_keep_hitting` setup, 8 episodes, a cap of 7, with `pc.seq += 2` per
simulated vision request between text turns):

| vision requests per text turn | misses after round 1 (of 32) |
|---|---|
| 0 | 5 |
| 1 | **28** (= plain LRU) |
| 2 | 28 |

So an agentic text load sharing the worker with an equal rate of image requests loses the whole gain
of B. Before the fix vision did not tick (the store-only clock), so this is introduced by the fix;
the review's finding 5 (an idle / vision-only period keeps old entries "live") is real but the cure
should not age entries that are still in an active round.

Fix: drop the vision ticks (and keep the lookup ticks, which only scale the rule); handle idleness
with wall-clock age instead: stale also when `time.time() - e.used > PROMPT_CACHE_STALE_S` (a knob,
e.g. 120–300 s, well above an agent's inter-turn gap, i.e. tool latency + other episodes' turns).
Optionally a rank-based round instead of ticks: stale iff at least `len(entries)` *other* entries
were stored or hit since `e` was (traffic that touches no entry then never ages anything). Add the
vision-interleaved round-robin above as a test.

### R2-2. LOW — `doc` strides assume a system message

`worker.py:2758` (`starts[1] <= b < starts[2]`), and `fixed`'s `starts[1:2]` (pre-existing).

`starts[1]` is "the end of the system block" only when the prompt has a system message. Without one
(`starts[0]` = the first user message), `fixed` protects the first assistant start and `doc` takes
the strides of the first *assistant reply*, while the first user message (the shared document of
option (i)) gets none. The bits stay valid (all positions are slab ends of this prompt); it only
misses the intent. Fix: locate the first user message from the template's role token after
`<|im_start|>` (or: the first start whose following token is not the system role), or document
"with a system message".

### R2-3. LOW — `evict`'s callable `wanted` is still unguarded on the `_fits` / `_regrow` path

`worker.py:1049`.

`store` now evaluates `wanted` eagerly and treats its failure as `()`, but `_fits` (2577) and
`_regrow` (2511) still pass `self._wanted` to `evict`, which calls it bare; an exception there
leaves `tick()` and fails every running row (finding 1's consequence). With the queue copied under
the lock the only remaining sources are bugs in `_peek` / `pc.peek`, so this is defence in depth:
wrap the call in `evict` the way `store` does (`except Exception: log; wanted = ()`).

### Checked in round 2, no issue

- **(1) Lock and lock order.** `self.cv` is `threading.Condition()`, whose default lock is an
  `RLock` (re-entrant on the scheduler thread). No caller of `_wanted` holds `cv`: `tick` releases it
  after the cancelled-request scan, before `_admit`/`_fits`/`_advance`/`_iterate`; `_run` releases it
  before `tick`; `_close` holds it only around its own copy; handlers hold it only to append. The
  copy is `list(islice(...))` under the lock, and the peeks run outside it (no handler is blocked by
  an `_lcp`). No MLX work happens inside `_wanted`, so the test calling it from a second thread is
  safe.
- **(2) Eager `wanted` + rollback.** The eager condition `bytes_of(rest + [new]) + extra >
  min(budget, limit)` is exactly the loops' entry condition after the publish (`self.bytes` =
  `bytes_of(rest + [new])`). `wanted` is computed before anything is mutated, so a failure costs only
  the order; peeking then still sees the `covered` entries and not `new` (harmless: covered ones are
  dropped from the cache anyway, `new` is kept, and the publish bumps `version` so those memos are
  redone). The `try` covers publish → evictions → copy → `index()`. On any exception: `new` is
  removed, `covered` come back (appended last: only the tie order of `_scan` changes, and the
  `_recount()` there bumps `version`), `bytes` is recounted; entries evicted before the raise stay
  gone and counted in `evictions` (their memos fail the presence check; their buffers are freed with
  the last reference, as with any eviction). `seq` keeps the store's tick — a clock, not state. The
  "not kept" early return mutates nothing but `seq` and `evictions` (as before). Covered entries can
  never be evicted inside the loop (they are not in `entries` then).
- **(3) Give-up evicts H first.** `_fits` only runs with no task (`_admit` requires `task is None`),
  so there is no running prefill's parent to protect; `keep` is already `()` on that branch; the
  `evict(keep=[all but H])` call removes exactly H, after `blocked(0)` was checked with the
  pre-eviction `live`. It also eases a latent case (pre-existing, not fully fixed): when
  `blocked(floor_H)` came from the live/cap term alone, the budget-only loop evicts nothing; now at
  least H goes, but if the cap term needs more than H, `_fits` still returns False with the other
  entries kept (the loop never looks at the cap term) — the head then waits for a row to end.
- **(4) `_recount(added=False)`.** Only two call sites pass it: `evict` (pure removal) and the
  post-copy recount in `store` (only `new.chain` / `arrays` / `nbytes` change; `best()` reads `ids`,
  `ckpts`, `end`, `low`, all fixed at construction). Absorption (covered removed, checkpoints moved
  into `new`) is part of the publish, whose `_recount()` bumps; the rollback bumps too. No path
  changes a present entry's `ids`/`ckpts`/`end` after it is published.
- **(5) Vision ticks — placement and side effects.** Once per vision prefill, in `_prompt_cache`
  (the first `next()` of the prefill; vision never takes DFlash, so no double count), guarded by
  `PROMPT_CACHE`; it touches neither `Usage` (`reused`, `cached_tokens`) nor any log line nor
  `hits`/`misses`. The placement is right; the policy is not (R2-1).
- **(6) Option (i) validity and `_thin`.** `doc` is a subset of `strides` = `below(...)` over this
  prompt's `bounds`, so every position is a slab end of the new prompt (`< end`); the store filter
  still only removes keys of an already valid dict, and quantized entries are still filtered by
  `kv_start` in `store` after the chain filter (the removed pre-filter was redundant). `protect =
  chain[1] = fixed`; `doc` and `tail` are unprotected, so `_thin` drops them first and drops a
  `fixed` position only when nothing unprotected is left but the deepest
  (`test_chain_thinning_drops_strides_never_the_marks` covers it). A huge first message briefly holds
  all its stride states before `_thin` — the row's prefill took them anyway (`want` has every
  stride).
- **(7) Not-started `task_row` in `_wanted`.** `row.started` is set in `_advance` immediately before
  the `next()` that runs `_prompt_cache` (the lookup), so the window it covers is exactly admission →
  lookup; after the lookup `_regrow` keeps `row.parent` instead. `task_row` is cleared on
  `StopIteration`, on a prefill failure and in `_close`; a DFlash run keeps it but with `started`
  true, so it is never listed after its request began, and `store` (called from inside the running
  prefill) never lists its own row. Vision task rows are skipped like vision waiting requests.
- **Docs.** README / CLAUDE.md describe the lock, the `max_batch` scope, the clock (`2 ×
  len(entries)`, vision ticks), the eager `wanted` and the widened rollback, the give-up order and
  the first-user-message strides as coded.

## Round 3

Narrow re-review of the round-2 fixes (`chain_evict_fixes.md` "## Round 2"). CPU only. Line numbers
are `src/worker/worker.py` of the working tree. CPU suite: started, result not available when this
section was written (expected 227 passed, 2 skipped; see the caller's summary).

Nothing above LOW.

### R3-1. LOW — entries of *running* requests are not exempt from the wall-clock rule

`worker.py:1040–1049` (`stale`). The exemption covers only entries a *waiting* request would resume
from. A conversation whose row is still decoding (long answers at batch load: ~12 tok/s per row at
c=16, so a 4K answer runs ~340 s) has its entry age past 300 s while its next turn is certain to
come, and it goes LRU-first as "stale". In most such loads the tick rule marks it stale anyway (16
rows' traffic), so the wall clock rarely changes the outcome; noted, no action required. If wanted:
treat `e.used` as refreshed while a row whose `row.parent`/stored entry is `e` is running, or set
`e.used` at the row's end.

### R3-2. LOW — the role token assumes `<|im_start|>user` tokenizes the same alone and in context

`worker.py:3846–3850`. `_im_user` comes from `encode("<|im_start|>user")`; the template emits
`<|im_start|>user\n`. On the Qwen tokenizer this is `[248045, 846, 198]` (checked by the fix agent),
so it matches. A tokenizer that merges `user\n` would give a non-matching id: `k` is then None and
`doc` is empty (no fallback, since `user` is not None) — safe (fewer checkpoints only), silent.
A one-line load log of `_im_user` (or a check against `encode("<|im_start|>user\n")[1]`) would make
it visible.

### Checked, no issue

- **(1) `stale` / `QWEN_PROMPT_CACHE_STALE_S`.** Tick rule unchanged (`> 2 × len(entries)`); wall
  clock is a second, OR-ed condition, off at 0; `victim` takes one `now` per call and passes the
  wanted flag; wanted entries are exempt only from the wall clock (the tick rule still ages them).
  The debug line calls `stale(e, now)` without the flag but prints "wanted" first when `w`, so the
  label is right. `time` is the module's own, so the tests' `SimpleNamespace(time=…)` drives it
  (`test_victim_order` pins it; the stubs' old `used` would otherwise be idle).
- **(2) Vision ticks removed.** The only `seq` increments left are `lookup` (968) and `store`
  (1112); `_prompt_cache` no longer touches the clock. My round-2 repro (8 episodes, cap 7, one
  vision request per turn) is now covered by the repo test with real vision requests.
- **(3) First user message by role.** `k` iterates `starts[:-1]`, so `ids[s + 1]` and
  `starts[k + 1]` always exist. No user message → `k` None → `doc` empty. A user message as the
  first message → `[starts[0], starts[1])`. Tool messages render as `<|im_start|>user` +
  `<tool_response>` (template line 129), so a tool response that precedes every real user message
  would be taken as "the document" — harmless, those are slab ends of this prompt like any stride.
  The `starts[1]..starts[2]` fallback only runs when the role is unknown (test engines). `doc` is
  still a subset of `strides` (slab ends of this prompt), unprotected; `fixed` still protected.
  `fixed`'s `starts[1:2]` without a system message is pre-existing and documented as such.
- **(4) Guarded `wanted` in `evict`.** `except Exception` → log + `()`, the same as `store`;
  `BaseException` still propagates; `victim` itself cannot raise on valid entries.
