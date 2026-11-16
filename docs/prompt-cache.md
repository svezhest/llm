# Prompt cache

Current conclusions (the state of all topics: [MAP.md](MAP.md)):

- Multi-entry prefix cache: KV in shared immutable segments, GDN states checkpointed at slab
  ends; a hit resumes bit-identical to a cold prefill (reply KV aside).
- Eviction: entries a waiting request wants last, stale ones first (LRU), else MRU.
- Reply KV is on by default (−21 % prefill tokens on AppWorld) and not bit-exact; `=0` restores exactness.
- The disk tier is the biggest AppWorld lever but off by default (needs a directory); MOVE
  (one KV copy) is for a single stream, e.g. the 27B at long context.
- Never share mx.array objects between copies; never trim with the library.

- **Prompt cache** (`QWEN_PROMPT_CACHE=1`): multi-slot prefix cache
  (`_PrefixCache`/`_Entry`). KV is sliceable to any prefix; the GDN recurrent
  state is not, so `_prefill`/`_spec` checkpoint it (~52 MB, fp32 32×128×128 ×
  24 layers on the 9B) at slab ends forced onto `cuts`: every 2048 tokens,
  `<|im_start|>` boundaries (end of system block, last two messages) and the
  divergence point from the closest entry. Hit = deepest checkpoint ≤ longest
  common prefix with any entry. **Segments** (`_Seg`, since 2026-10-12): an
  entry's KV is a chain of pieces `(seg, lo, hi)` over immutable per-FA-layer
  buffers of its own; `store` copies only `[at, len)` out of the row (the parent's
  pieces cover `[0, at)`; all of `[0, len)` when the row's prefill quantized an
  fp16 parent's part) and never keeps a view of the row (its first decode write
  stays in place; phase 1 pinned the whole buffer and the row COW-copied it);
  `restore(e, at, first)` concatenates the pieces + a zeros tail of
  round_up(first slab, 256) = the buffer the first `update_and_fetch` built from
  the old `[:at]` view (bit-identical, tested). Bytes count every buffer once; a
  segment dies with the last entry holding it. 16 siblings of an 8K prompt:
  ~1.38 GB for all 16 by the plan's estimate (cascade_plan A.7; not yet measured —
  phase 1 measured 6 kept in 3.18 GB); a chat keeps one chain (+ the
  dead tail of each resumed turn, ≤ a slab + the reply opener). A checkpoint at p
  is valid for an entry only where p is a slab end of *that entry's* prompt (its
  KV over [0, p) was cut there). An older entry whose ids are a prefix of the new
  one (the previous turn) is absorbed at store: its checkpoints move over (shared
  objects), its end state only if that position is a slab end of the new prompt
  (usually not: an exact repeat of the older prompt then resumes at its deepest
  checkpoint), and it is dropped — except an unquantized entry under a quantized
  new one (27B/35B: the only resume point below `quantized_kv_start`). Thinning
  to `QWEN_PROMPT_CACHE_CKPTS` (`_thin`) never drops the end of the system block
  (`starts[1]`) or the last three `<|im_start|>` (mapped to slab ends, inherited
  ones too); strides and older turns' boundaries go first, closest pair's left.

## Chain policy and eviction

  **Chain policy** (2026-10-16, AppWorld handoff: an 11K agentic entry was 932 MB,
  ~60 % GDN states): an entry that absorbs an older one (`covered`) keeps only
  `chain[0]` = the end of the system block + the last `QWEN_PROMPT_CACHE_CHAIN_MARKS`
  (3) message starts + the stride checkpoints ≥ end − `QWEN_PROMPT_CACHE_CHAIN_TAIL`
  (4096) + the strides inside the first user message (the first `<|im_start|>user` block,
  by `Engine._im_user`; unknown role → `starts[1]..starts[2]`; a shared
  document: a sibling conversation diverging inside it resumes near the divergence, not
  at the system block), thinned with `chain[1]` (the marks) protected; + its end state.
  Trade-off (review 2026-10-16 #6, `chain_ckpts_sim.py` with `SIM_USER0`): free for
  AppWorld (300-token first message: 11.4K entry 669 MB either way), +2–4 states
  (+100–200 MB) per entry for an 8K first message (13.5K 687 → 785 MB, 34K 1415 → 1562);
  the alternative (chain filter only from the 2nd absorption) was rejected: it keeps
  nothing of the document past turn 3. The quantized floor is applied by `store` (the
  chain is not pre-filtered). Computed in
  `_prompt_cache` (`chain`), applied in `store`; `CHAIN_MARKS=-1` = off. Fresh entries
  (no chain absorbed) keep the cap-12 policy below `QWEN_PROMPT_CACHE_CHAIN_FRESH_TOKENS`
  (16384, 0 = never; length of the whole prompt), from there `fresh` = `chain` + the
  divergence point from the closest *other* entry (`diverge`: the prompt's own partial
  entries are ignored, their `_Entry.diverge` carried) + the strides inside the system
  block (before the first user message, unprotected) (2026-10-17: a 50K MISS entry
  12 → 7 states, 2.28 → 2.04 GB simulated). Byte-identical (a subset of valid checkpoints).
  Simulated through the scheduler (CPU, production knobs, AppWorld-shaped turns,
  9B bytes, `docs/perf/2026-10-16/chain_ckpts_sim.py`): 11.4K 1012 → 669 MB (12 → 5
  states), 41K 1995 → 1652 MB, resume
  positions unchanged; an older turn repeated resumes at its last message start
  (previous turn) or the system block. Store log `states=` = checkpoints + end MB.
  `QWEN_PROMPT_CACHE_GB` (3) is a soft cap (at store, the newest stays, alone
  if over with a warning); **eviction order** (`_PrefixCache.victim`, every eviction:
  store, `_fits`, `_regrow`): entries a waiting request would resume from ≥ a slab
  deep (`_Scheduler._wanted`: the admitted request whose prefill has not started + the
  first `max_batch` waiting, the queue copied under `cv`; peeks memoized per request in
  `req.peeked`: `pc.version` bumps once per kept store (publish / rollback), an eviction
  invalidates only the memos that pointed at it; weakref)
  last; stale ones (`_PrefixCache.stale`: `pc.seq − e.seq > 2 × len(entries)`, the clock
  ticking at every text lookup and store — no hit / store of it within a round of turns;
  vision does not tick: 2 ticks per vision request made live episodes stale, 28/32
  misses with one image per text turn, review round 2 — or, unless wanted, unused for
  `QWEN_PROMPT_CACHE_STALE_S` (300 s): an idle cache) first, LRU; all live → MRU (round-robin episodes: the latest
  one's next turn is the furthest; LRU evicted the next in line, and its re-prefill
  the one after: 28/32 misses vs 5 in `test_round_robin_episodes_keep_hitting`;
  event sim of the AppWorld load +13–60 % requests served at 5–8 GB, "largest
  first" 3–10 % worse than MRU: `docs/perf/2026-10-16/evict_sim.py`; `evict_sim.py clock`:
  the lookup+store clock gives identical results, and so does + 300 s / 120 s wall-clock
  age when wanted entries are exempt — without the exemption 300 s cost 6–12 % of the
  requests served at 4–6 GB, where queue waits pass 300 s); the cache's live bytes (`_PrefixCache.bytes`: every
  buffer once) count against the batch budget: a store first sizes the new entry
  (its segment from the row's shapes, `_Seg.estimate`, exact) and, if it alone is
  over budget + prefill reserve − reserved, drops it before anything changes —
  nothing absorbed, evicted or copied, logged "not kept", counted in `evictions`;
  else it absorbs, evicts entries (victim order) down to that headroom, then copies `[at, len)`
  (`wanted` is asked before anything changes and only when an eviction is coming; its
  failure is logged and costs the order only; anything raising between the publish and
  the segment rolls back: the new entry out, the absorbed ones back — never a
  half-built entry, whose hits would restore a too-short KV);
  `_fits` evicts on demand but spares the entry `peek` says the request would resume
  from when that is ≥ `QWEN_PREFILL_STEP` deep (memoized `req.bounds`, `req.peeked`), and evicts
  nothing when even that would not admit it (budget or live check). When that entry
  alone keeps the head out, the head waits for a row to end only for a hit ≥
  `QWEN_PROMPT_CACHE_KEEP_TOKENS` (8192) and at most `QWEN_PROMPT_CACHE_KEEP_WAIT_S`
  (10 s, from its first refusal: `req.keep_since`); then, or at once for a shallower
  hit, the entry goes too, first (logged "gives up its N-token hit"). `_regrow` never evicts
  the running prefill's parent (`row.parent`; the snapshot closure holds an
  `_Inherit`, not the entry, and drops it and the checkpoint dict right after its
  store — a DFlash run keeps the closure for its whole decode). Quantized entries keep
  no checkpoint below `quantized_kv_start` (never resumable; `_Entry.low`). A
  restore holds copies, so eviction frees memory at once, except the pieces an
  in-flight prefill resumed from (its `_Inherit`, released at its store). A failed store is
  logged, the request goes on. Store log: `… MB (n entries, X MB total) reused=r
  gap=split−reused absorbed=k pieces=c states=S MB`; `Usage.cached_tokens` = reused; Health has
  entries / bytes / hits / misses / evictions. Never trim with the library; never share mx.array
  *objects* between cache copies (`a[i] = v` rebinds the Python object in place
  — slice to get a new one, buffers are copy-on-write). On a hit set
  `language_model._rope_deltas` to zeros (text mRoPE continues from the cache
  offset; a previous vision request leaves its own delta there). Text-only.
  7.5K shared system prompt, new conversation: TTFT 14.4 → 0.20 s, byte-identical.

## Partial store

  **Partial store (2026-10-17)**: a text `_prefill` that stops early — cancel at the slab
  loop's top, or an exception outside the forward (`intact`: after the slab's eval) —
  with ≥ 1 slab past `reused` calls `snapshot(cache, upto=pos, why)`: ids = prompt[:pos],
  pos the last completed slab end (∈ `ends`, offsets checked), the row's state = end
  state, checkpoints so far, same `pc.store` (limit, absorb, chain/fresh, `_Inherit`
  released); `_Entry.partial` (absorbing only partial entries is no chain: `chained` =
  any non-partial or chained covered entry), logged as the normal store line +
  ` (cancelled at pos/len)`. Runs in `_advance` like the full store. DFlash (`_spec`)
  prefills too since 2026-11-11; vision: nothing. Retry resumes at pos, bit-identical to cold. A store exception there
  is logged (`_store_prefix`), never changes the outcome. `docs/perf/2026-10-17/
  design/partial_store.md`, `tests/test_partial_store.py`.

## Reply KV

  **Reply KV (`QWEN_PROMPT_CACHE_REPLY`, default 1 since the 2026-10-17 GPU gate;
  `0` = off)**: at a row's end
  (`stop`/`length`; not cancel (stop strings usually; a late cancel after stop/length
  stores, harmlessly), errors, vision, rows still in a cascade group (logged
  `id=… prompt cache reply skipped: in a cascade group …`), replies that generated a
  reasoning section — `_GenState.for_prompt(ids, _think)`: the prompt open in `<think>`,
  the handler's own split; a think id inside the content with thinking off is content,
  stored) `_drop_done` → `_reply_cache` + `_store_reply` extend the row's prompt
  entry (`row.entry`, weakref; gone = nothing) to ids = prompt + `row.fed[:pos − len]`
  (tokens as `_iterate` read them; the step-ahead fed the EOS it stops on, a sync row
  did not, a `length` row fed its last generated token too: entry = prompt +
  completion) through `pc.store` (`inherit` = its
  whole chain, so only `[len−1, pos)` is copied; budget checks as any store; `reply=` →
  `_Entry.reply`, whose end `best()` accepts though it is no slab end; `tick=False`:
  a turn stays 2 clock ticks; `sync=False`: copies left lazy, `_drop_done` queues every
  finishing row's copies + end states in ONE `async_eval` — no host wait between steps;
  `spare=(task_row.parent,)`: never evicts the running prefill's parent, as `_regrow`,
  and "not kept" when it does not fit beside it; `clock=` the prompt entry's
  (used, seq): a reply keeps its prefill's recency, else a just-finished conversation
  — next turn imminent — would be the MRU victim). A failed pass `async_eval` drops the
  reply entries and restores the prompt entries they absorbed.
  The GDN end state is always a gather `a[[i]]` (own buffer; at B = 1 too: right after a
  2 → 1 shrink the row's state is `_gdn_split`'s view of the old stack). Absorbed
  checkpoints only `p < L`. The prompt-end state is dropped (a regenerate resumes at the last
  message start); a reply entry whose prompt part a new store covers is absorbed (the
  reply re-tokenized differently / regenerated). Phase 0 (AppWorld traces, CPU): 100 %
  of 1344 replies re-render to the same ids, −20.8 % prefill tokens.
  **Numerics**: reply KV from decode kernels + the next slab starting at the reply end
  → a hit through a reply ≠ cold prefill bitwise (CPU: GDN recurrence of a T-slab vs T
  steps differs from layer 0): the batched-decode numerics class — GPU gate (9B, hit
  through a reply vs cold, teacher-forced 862 positions) top-1 99.88 %, KL 5.8e-5;
  greedy answers can depend on cache contents. `QWEN_PROMPT_CACHE_REPLY=0` is the way
  back (bit-exact, `_bounds` determinism). Gate: −21 % prefill tokens / +6.5 % req/min
  on the AppWorld replay, no visible decode cost. Tests `tests/test_reply_kv.py`; design
  `docs/perf/2026-10-17/design/reply_kv.md`, gate `docs/perf/2026-10-17/reply_kv_gpu.md`.

## One KV copy (MOVE)

  **One KV copy (`QWEN_PROMPT_CACHE_MOVE`, 2026-11-11, off by default; `_move_on`: max_batch 1, no disk tier; quantized
  KV since 2026-11-15: only quantized entries move, an fp16 one (< `quantized_kv_start`) is copied, a row still fp16
  at its prefill's end stores by copy, an fp16 parent of a quantizing row inherits no pieces; parity tests
  `tests/test_kvq8.py`)**: `_prompt_cache` takes the conversation's own entry (`taken`: its ids are a prefix of the prompt or the
  prompt covers its reply; a sibling's entry only when a copy does not fit beside it, logged) with
  `restore(move=True)`: the entry leaves `pc.entries`, its exclusive segments are released layer by layer as the
  row's buffers are built, the KV-less entry stays as the lease's `husk` (`_Own`: husk / at / lo / L / end / store).
  `snapshot()` at the prefill's end only saves the end states (`lease.end`); the row's end goes through `_settle`
  (`_drop_done`, `_spec_reply`, the max_tokens = 0 path): `_Seg.take` turns the row's own buffers into the segment
  (views; own slices when the buffers run a KVCache step past what is kept), `lease.store` = the deferred prompt
  store with `seg=` / `moved=husk`, then `_store_reply(seg=whole.sub(L, pos))` over the same buffers. Early stops:
  `_settle_prefix` (`unmove` when nothing was prefilled past the entry, else the partial store); a failed forward:
  `_lost` (nothing evaluated or stored: the arrays hang on the failed graph). Checkpoints of a running prefill are
  thinned live to `QWEN_PROMPT_CACHE_CKPTS` (they are the row's, uncounted, until the store). `_Own.close()` at
  every exit: `store` is a closure over the lease and the row (a cycle that kept a dead row's KV and states until
  the cyclic GC). Parity with the copies (answers, reused, cache contents bit for bit) + no-cycle check per request:
  `tests/test_pcache_move.py`. README "One KV copy for a single stream".
