# Fixes for the chain / eviction review (`chain_evict.md`)

CPU only; nothing ran on the GPU, no worker was started. Line numbers are
`src/worker/worker.py` of the working tree after the fixes (another agent edits other
regions of the file concurrently, so they drift). Tests in `src/worker/tests/`.

CPU suite (`uv run --group dev pytest -q --ignore tests/test_mrqmm.py --ignore
tests/test_pfqmm.py --ignore tests/test_rattn.py`): **222 passed, 2 skipped** (3 min 47 s;
baseline 212 passed, 2 skipped: +10 new tests). The repo's `chain_ckpts_sim.py` re-run on
the final code: 11.4K 1012 → 669 MB, 41K 1995 → 1652 MB (unchanged from before the fix).

## Per finding

### 1. HIGH — `_wanted` iterated `self.waiting` without the lock

- `_Scheduler._wanted` (2601–2621): the queue is copied under `self.cv`
  (`list(itertools.islice(self.waiting, self.max_batch))`, 2609–2610) and the copy is
  iterated; `import itertools` added.
- Test: `test_segments.py::test_wanted_copies_the_queue_under_the_lock` — the review's
  repro (a submit between two peeks) no longer raises, and `_wanted` blocks while another
  thread holds `cv` (the copy really is taken under the lock).

### 2. HIGH — half-built entry on an exception in `store`'s eviction loops

Both parts:

- (a) `store` (1143–1154): a callable `wanted` is evaluated once, **before anything is
  mutated**, and only when an eviction is coming (`rest` non-empty and
  `bytes_of(rest + [new]) + extra > min(budget, limit)` — the exact condition the two
  loops test right after publishing). Its failure is logged and treated as `()` (it only
  costs the order). The lazy memo wrapper is gone.
- (b) the rollback `try` (1155–1174) now covers everything from the publish
  (`self.entries = rest + [new]`) through both eviction loops, the copy and `new.index()`:
  the new entry goes, the absorbed ones come back, `_recount()` (an add: memos re-peek).
- Tests: `test_segments.py::test_store_survives_a_failing_wanted[fresh|resumed]` (the
  review's two repros inverted: a raising `wanted` during a store that evicts → the entry
  is fully built, KV over all its tokens, the regenerate resumes from its end and answers
  as cold, `check_cold` passes), and `test_store_failing_while_evicting_rolls_back`
  (`victim` raising inside the loop → no entry for the new prompt, the absorbed turn is
  back, bytes recounted, the regenerate answers as cold, `check_cold` passes).

### 3. MED — give-up evicted other waiting requests' entries first

- `_fits` (2529–2583): when the head gives up its hit (`first, keep, floor = keep[0], (),
  0`, 2570), that entry is evicted first (2574–2575: `pc.evict(keep=[every other
  entry])`), after the `blocked(0)` check (so the live estimate is still the one taken
  before any eviction), then the normal `victim`-order loop runs. Docstring updated.
- Test: `test_segments.py::test_give_up_evicts_the_heads_entry_first` (the review's repro:
  now only H goes, S — the second-in-line's, most recent entry — stays,
  `evictions == 1`).

### 4. LOW — `_wanted` cost

- `store`: the post-copy recount is `_recount(added=False)` (1175); the rollback keeps
  `added=True`. A kept store bumps `pc.version` once.
- `_wanted` peeks only the first `max_batch` waiting requests (2610).
- (The optional per-request `_lcp` memo was not done.)
- Tests: `test_peek_memo_follows_an_added_entry` (`pc.version == v + 1` after a kept
  store) and `test_wanted_copies_the_queue_under_the_lock` (max_batch 2, three waiting
  requests with entries: only the first two are spared).

### 5. LOW — staleness clock advanced only on stores

- `_PrefixCache.lookup` (963–973): `self.seq += 1` on every text lookup (hit or miss); a
  turn is now 2 ticks, so the rule is `self.seq - e.seq > 2 * len(self.entries)` in
  `victim` (1032) and the debug line of `evict` (1050–1054, "ticks old").
- Beyond the brief, because the review's headline case is vision traffic (which never
  looks up): `_prompt_cache` adds 2 ticks for a vision request (2696–2697), so a
  vision-only period ages the text entries like text turns would.
- Test: `test_segments.py::test_lookups_advance_the_round_clock` (two entries; five hits
  on A and no use of B: B turns stale and is the victim; with the store-only clock both
  stayed live and A, the most recent, went). `test_victim_order` rescaled to the new rule.
- `evict_sim.py` models the clock: `python3 evict_sim.py clock` runs LRU / MRU at 3–10 GB
  with the store-only clock vs lookup + store ticks (2× threshold, hits refresh `stored`
  as `e.seq` does). The results are **identical** at every budget (e.g. MRU 5 GB 483 done
  / 36.7 % misses, 6 GB 1024 / 15.5 %, 8 GB 3253 / 3.4 %; LRU 304 / 630 / 2886): in that
  load every request looks up and stores, so the doubled clock and doubled threshold
  cancel. The documented conclusion (+13–60 % served at 5–8 GB vs LRU) stands.
- Docs: README eviction-order paragraph, CLAUDE.md prompt-cache paragraph, `victim`
  docstring.

### 6. LOW — chain policy on the first absorption / a repeat

Measured with `docs/perf/2026-10-16/chain_ckpts_sim.py` (30 AppWorld-shaped turns,
production knobs, 9B bytes), both options behind a temporary switch (removed), plus a
long-document variant (`SIM_USER0=8000`: the first user message an 8K document; the
switch is now an env var of the sim):

| first user message | entry | policy off | chain (before) | (i) + first-message strides | (ii) filter from the 2nd absorption |
|---|---|---|---|---|---|
| 300 tokens (AppWorld) | turn 2, 5.8K | 484 MB | 484 | 484 | 484 |
| | 11.4K | 1012 | **669** | **669** | 669 |
| | 26.3K | 1506 | 1163 | 1163 | 1163 |
| 8K document | turn 2, 13.5K | 883 | 687 | 785 (6 states) | 883 (8) |
| | 19.1K | 1264 | 921 | 1068 (8) | 921 |
| | 34.0K | 1758 | 1415 | 1562 (8) | 1415 |

Resume positions (`reused`) identical in every variant. **Option (i) taken**: the 11K
AppWorld entry stays at 669 MB (≤ 700 MB; AppWorld's first user message holds no stride,
so (i) costs nothing there). It is also the only option that fixes the review's case
past turn 2: (ii) keeps the document strides only until the third turn absorbs the
second. Cost of (i): +2–4 states (+100–200 MB) per entry whose first user message is a
long document, for the entry's life (thinned by the cap 12, unprotected).

- `_prompt_cache` (2742–2759): `strides` is a list now; `doc` = the strides with
  `starts[1] <= b < starts[2]` (the first user message) joins `chain[0]`, not `chain[1]`
  (so thinning drops them before any mark). Comments, the header block (608–621),
  README, CLAUDE.md (prompt-cache paragraph + env table) document the trade-off.
- Tests: `test_prompt_cache.py::test_fresh_entries_keep_the_cap_policy[chain|off]`
  rewritten — 4 turns, the first two user messages 120 tokens: under the chain the
  first message's strides survive, the second message's are gone, and a sibling
  conversation diverging 100 tokens into the document resumes at the deepest document
  stride below it (`> below(starts[1]) + 64`), answers as cold; `chain_parts` returns
  the document strides too, and `test_chain_keeps_the_last_boundaries_and_the_tail` /
  `test_chain_repeat_of_an_older_turn` include them (the older turn now resumes at the
  deepest document stride, else the system block).

### 7. LOW — admitted-but-not-started request's entry unprotected

- `_wanted` (2611–2613): the admitted request whose prefill has not started
  (`task_row` with `started == False`) is peeked first, so `_regrow` evicts its entry
  last (its lookup has not run, `task_row.parent` is still `None`). `_regrow`'s
  docstring says so.
- Test: `test_wanted_copies_the_queue_under_the_lock` (a not-started `task_row` puts its
  entry first; once started it is no longer listed: `row.parent` is kept instead).

### 8. LOW — tests that did not catch mutations

- `test_prompt_cache.py::test_chain_thinning_drops_strides_never_the_marks[5|7]`:
  `PROMPT_CACHE_CKPTS` 5 / 7, `CHAIN_TAIL` 400, `CHAIN_MARKS` 4 (one more mark than the
  fresh policy protects, so `protect = chain[1]` is distinguishable from the fresh
  `protect`): over 10 turns the chain entry holds ≤ cap checkpoints, all 5 marks every
  turn, thinning really ran (≥ 5 turns with more candidates than the cap), answers and
  `check_cold` as cold.
- Coordinator addendum: `test_peek_memo_follows_an_added_entry` (a waiting request's
  no-hit memo is redone after a store adds an entry it resumes from, a slab deep, and
  that entry becomes wanted) and `test_wanted_needs_a_slab_deep_hit` (a 6-token hit is
  not spared, a slab-deep one is).
- The quantized pre-filter of `chain` in `_prompt_cache` is removed (2768 comment):
  `store` drops checkpoints below `kv_start` after the chain filter, and protected
  positions that are absent never matter to `_thin`. Proof: the quantized tests
  (`test_quantized_floor_entry_survives`, `test_quantized_entries_have_no_subfloor_checkpoints`,
  `test_quantized_child_of_fp16_parent`, `test_entries_equal_cold_bitwise[q8]`) pass.

Mutation runs (copies of `src/worker` in the scratchpad, `-k` the new / touched tests):

| mutant | caught by |
|---|---|
| `_wanted` iterates `self.waiting` bare | `test_wanted_copies_the_queue_under_the_lock` |
| no `max_batch` cap | same |
| no not-started `task_row` | same |
| no depth threshold in `_wanted` | `test_wanted_needs_a_slab_deep_hit` |
| `pc.version` never bumped on an add | `test_peek_memo_follows_an_added_entry`, `test_store_failing_while_evicting_rolls_back` |
| post-copy `_recount()` bumps again | `test_peek_memo_follows_an_added_entry` |
| `wanted` failure fatal | `test_store_survives_a_failing_wanted[fresh, resumed]` |
| `wanted` not asked before the publish (lazy) | same |
| rollback covers only the copy (budget loop outside) | `test_store_failing_while_evicting_rolls_back` |
| give-up does not evict H first | `test_give_up_evicts_the_heads_entry_first` |
| lookup does not tick | `test_lookups_advance_the_round_clock` |
| no first-message strides in the chain | `test_fresh_entries_keep_the_cap_policy[chain]` |
| `protect = chain[1]` removed | `test_chain_thinning_drops_strides_never_the_marks[5, 7]` |

### 9. LOW — stale "LRU" wording

- worker.py: header block (630–637: evicted at store in `victim` order, "an admission
  evicts entries on demand"), `_PrefixCache` docstring (921), `peek` docstring (960: no
  used / seq touch), `_regrow` docstring (2487–2490), `_fits` docstring (2531–2540).
- CLAUDE.md: Continuous-batching invariant ("give way first (evicted on demand, `victim`
  order)"), the `_fits` sentence ("memoized `req.bounds`, `req.peeked`"), the eviction
  order (`_wanted` scope, the lock, the clock), the give-up order, the store's eager
  `wanted` + extended rollback.
- README: the eviction-order paragraph (scope of "waiting", give-up first, clock), the
  vision sentence ("never read or store the cache; they only advance its round clock").

## Round 2

CPU suite after round 2: **227 passed, 2 skipped** (4 min 8 s; +5 tests over round 1).

### R2-1 MED — vision ticks made live episodes stale

- Dropped the 2 vision ticks in `_prompt_cache` (round-1 line 2696–2697). Text lookups still
  tick (`lookup`, 965) and the tick rule stays `> 2 * len(entries)`.
- Wall-clock age is now a second staleness condition: `QWEN_PROMPT_CACHE_STALE_S`
  (`PROMPT_CACHE_STALE_S`, 657; default 300, 0 = off). The rule lives in
  `_PrefixCache.stale(e, now, wanted)` (1040), used by `victim` (one `time.time()` per
  call, 1031) and by `evict`'s debug line (which now also prints the idle seconds).
  `time` is the module's own, so the tests' injected clock drives it.
- **Deviation from the brief: entries a waiting request wants are exempt from the
  wall-clock condition.** The tick rule still applies to them. Reason: `evict_sim.py
  clock` now also models wall-clock age (`stale_s`). Without the exemption, 300 s cut the
  requests served at 4–6 GB by 6–12 %, and 120 s by 12–32 %:

  | GB | MRU, ticks only | + 300 s | + 120 s |
  |---|---|---|---|
  | 4 | 309 done / 61.4 % misses | 296 / 63.8 % | 273 / 73.5 % |
  | 5 | 483 / 36.7 % | 424 / 42.0 % | 347 / 53.0 % |
  | 6 | 1024 / 15.5 % | 914 / 18.9 % | 700 / 26.9 % |

  The cause is queue waits above 300 s, under which every queued episode's entry looks
  idle. Among wanted entries, "stale → LRU" then evicts the one at the head of the queue.
  With the exemption, 300 s and 120 s give results identical to ticks only at 3–10 GB
  (120 s is 1082 vs 1024 done at 6 GB). The "+13–60 % vs LRU" conclusion stands.
  - Caveat: the worker only sees its own queue (and `_wanted` only the first `max_batch`
    requests in it). Episodes queued in the gateway past 300 s can still age out; the sim
    does not model this.
- Tests (`test_segments.py`):
  - `test_round_robin_with_vision_traffic_keeps_hitting` (989): the round-robin set-up
    with a real vision request (`vision_ids` / `vision_inputs`) after every text turn.
    Misses after round 1 are ≤ 5 of 32. The mutant that puts the vision ticks back fails
    it (28).
  - `test_idle_entries_go_first` (1019): A is left idle for 400 s on a fake clock, B and C
    are stored after. A is still live by ticks. The store that evicts takes A with 300 s
    and C (the MRU) with 0. A wanted entry is never idle: with all entries wanted the MRU
    goes.
  - `test_victim_order` now pins the clock (its stubs' `used` of 100 s would otherwise
    all be idle).
- Docs: README eviction-order paragraph, vision sentence ("never touch the cache (nor
  tick its round clock)"), env table row. CLAUDE.md eviction order, sim result, env
  table. `victim` / `stale` docstrings, worker header comment. `evict_sim.py` gained the
  `stale_s` parameter (wanted-exempt) and docstring.

### R2-2 LOW — document strides assumed a system message

- `Engine.load` (3846–3850) sets `_im_user` = the token after `<|im_start|>` in
  `<|im_start|>user`: 846 on the Ornith/Qwen tokenizer (checked: `[248045, 846, 198, …]`).
  It is None when that is not a single token.
- `_prompt_cache` (2775–2782): the first user message is the first start whose next
  token is `_im_user`, up to the next start. If the role is unknown (`None`: the test
  engines), it falls back to `starts[1]..starts[2]`.
- Unchanged (pre-existing, not part of R2-2): `fixed`'s `starts[1:2]` ("end of the
  system block") is still the second message start without a system message.
- Test: `test_prompt_cache.py::test_document_strides_follow_the_user_role[no-system|system]`
  (304). It uses role stand-in tokens and a 120-token document as the first user message.
  The first reply's strides are not kept (they lie outside the tail and the marks). A
  sibling conversation diverging 100 tokens into the document resumes at the deepest
  document stride, answers as cold, and `check_cold` passes. The mutant that ignores the
  role (`user = None`) fails `[no-system]`.

### R2-3 LOW — `evict`'s callable `wanted` unguarded

- `evict` (1062–1069): the call is wrapped like `store`'s (log "entries are unknown",
  `wanted = ()`), so a failing peek on the `_fits` / `_regrow` path costs only the order.
- Test: `test_segments.py::test_evict_survives_a_failing_wanted` (1047). The unguarded
  mutant fails it.

Round-2 mutants (same harness): vision ticks back, wall clock off, wall clock without the
wanted exemption, `evict` unguarded, role ignored. All are caught by the tests above.
