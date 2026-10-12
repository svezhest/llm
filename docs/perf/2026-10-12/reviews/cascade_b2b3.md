# Review: cascade B2 + B3 (scheduler wiring, `QWEN_CASCADE`, default off)

**Scope.** The uncommitted diff on HEAD 66a7c17:

* `src/worker/worker.py` (+495 lines), `src/proto/worker.proto`, `src/gateway/src/worker_client.rs`;
* CLAUDE.md, README, BACKLOG, and `cascade_plan.md` "As built";
* the new `src/worker/tests/test_cascade_sched.py`.

**Context read.**

* `cascade_plan.md` Part B and `reviews/cascade_b1.md`;
* `src/worker/cascade.py`;
* the mlx-vlm 0.7.6 `qwen3_5/language.py`, for how `cache.offset` and the masks are used.

**How it was reviewed.** Read-only, CPU only.

* Suite: `uv run --group dev pytest -q --ignore tests/test_mrqmm.py --ignore tests/test_pfqmm.py` gives
  **177 passed, 2 skipped** (153 s).
* Scratch experiments ran in the session scratchpad. They were scaled-down CPU runs through
  `conftest`'s tiny models and the real `_Scheduler`. Nothing in the repo was written except this file.

## Findings (ranked)

### 1. HIGH (when on): a reference cycle keeps the group's fp32 arrays alive after it dissolves or fails

The leak is gigabytes of fp32 prefix and stacks, held until Python's cyclic GC runs.

**Where.**

* `worker.py:1593-1604`: `_GroupRef` holds strong references to `group` and `row`.
* `_cascade_tidy` at `:3145-3147`: a finished member keeps its `_GroupRef`s, and only `r.group` is
  cleared.
* `_fail_rows` at `:3242-3243` and `_close` at `:2305`: `self.groups = []` while `g.rows` and the rows'
  `_GroupRef`s still point at each other.

**Mechanism.**

* `row.cache[i] = _GroupRef(g, row)` makes a cycle `row → row.cache → _GroupRef → row`.
* Each `_GroupRef` also points at `g`, and so at `g.pg`: the fp32 prefix plus every layer's stacks.

**Scenario 1: normal end of a group.**

* Every member finishes. Tidy sets `r.group = None` and `g.rows = []`.
* Each done row is now unreachable but still in its own cycle, and that cycle holds `g`.
* The prefix stays alive: 0.5 GiB at 8K, 2 GiB at 32K. So do the stacks: up to ~4.5 GiB at 16 × 4352.
* Meanwhile `self.reserved` has already dropped the charge.

**Scenario 2: `_fail_rows`.** `g.rows ↔ row.group` is a cycle too.

**How long the arrays stay alive.** The rows and groups lived for many ticks, so they sit in GC
generation 2. Python 3.12's generation-2 pass runs only when pending long-lived objects exceed 25 % of
all long-lived objects. In a process holding the whole model's module tree, that can take minutes.

* Admissions trust `reserved`, so the next formation or long prefill can run into real memory that
  is still held: OOM on 32 GB.
* B4's peak and memory numbers would be skewed too.

**Verified** (scratch `cycle.py` / `cycle2.py`, with `gc.disable()` so that only refcounting acts):

* After a normal wave, `weakref(pg)()` is still alive. It is freed by `gc.collect()`.
* After a failed grouped step plus `_fail_rows`, the same.
* No test checks this. `test_failed_step_drops_groups` checks accounting only.

**Fix** (any one breaks the cycle; do the first two):

* `_GroupRef` needs only `row.pos`. Hold `weakref.ref(row)` (`_Row` has no `__slots__`) and drop
  `group`, or keep `group` as a weakref too.
* In `_cascade_tidy`, for done members: `for i in self.fa_layers: r.cache[i] = None`.
* In `_fail_rows` and `_close`, before dropping the groups: `for g in self.groups: for r in g.rows:
  r.group = None` and `g.rows = []`.
* Add a test: run a wave with `gc.disable()`, take `weakref.ref(g.pg)` at build, and assert it is dead
  after dissolution and after `_fail_rows`.

### 2. MEDIUM: the memory rule is not the charge, and formation can raise `reserved`

**Where.**

* `_cascade_loses` at `:1583`, used at `:3036`, `:3065` and `:3134`;
* `_group_bytes` at `:2978-2991`.

**The mismatch.** `_cascade_loses` compares the idealised `2P + 2nS` with `n(P + S)`. The real charge
`_group_bytes` adds terms that the rule ignores:

* chunk padding;
* ×1.05 on the stack;
* score transients for all FA layers (`:2989`);
* the permanent "moves" term (`:2990`): one layer's old + new stack, 1.06 GiB at 16 × 4352.

The docs state "the group would not take more KV than their own caches". That is not what the code
checks.

**Numbers.** 9B formulas, prompt = P + 150, KV part only (members keep their GDN in both cases):

| P | reserve | n | rule forms? | group charge | rows' KV charge | Δ reserved |
|---|---|---|---|---|---|---|
| 8K | 4096 | 5 | yes | 2.35 GiB | 1.99 | **+0.36** |
| 8K | 4096 | 8 | yes | 3.42 | 3.19 | **+0.23** |
| 8K | 4096 | 16 | yes | 6.28 | 6.38 | −0.10 |
| 4K | 4096 | 3–16 | **never** | — | — | — |
| 4K / 8K | 256 | 4–16 | yes | | | −0.08 … −3.1 |
| 32K | 4096 | 3–16 | yes | | | −0.35 … −10.8 |

**What this means.**

* With the default `QWEN_BATCH_RESERVE_TOKENS` 4096 and the gateway's unbounded `max_tokens` (reserve
  = len + 4096), `QWEN_CASCADE_MIN_PREFIX=4096` is dead.
* A group forms only at P ≳ 5.0K (16 rows) or ≳ 5.8K (8 rows).
* An 8K group of 5–8 rows *raises* `reserved`. Formation is never refused, so this can push
  `reserved` past the budget. Then:
  * admissions stall;
  * the next `_regrow` past the headroom ends a row with `length`.

**Fix.**

* Decide on the real numbers, the ones the code already computes. Let `g_b = _group_bytes(P, cands)`
  and `rows_b = Σ(r.reserved) − n·_member_bytes()`.
  * Form or keep the group only if `g_b ≤ rows_b`.
  * Alternatively, allow `g_b − rows_b ≤ self.budget − self.reserved`: trading admission headroom for
    decode speed, but never past the budget.
* `test_memory_rule` should compare against `_group_bytes`, not against the idealised formula.

**On the proposed variant** (real suffix lengths for formation, reservation for the charge): it is
sound only with the budget guard above.

* With real lengths, P = 4K / 8 rows would form and add +1.0 GiB of reservation. At 16 rows, +1.7 GiB.
* Without the guard, that can overshoot the budget silently.
* It does make 4K prefixes groupable. So: real-length rule + guard + reservation charge.

**Optional.** The `scores` term charges all 8 FA layers' transients, but only about one layer's are
alive at a time. The `moves` term is charged permanently for a transient. Both are conservative and
eat most of the 16-row win at 8K (−0.10 GiB). B4's measured peak can tell whether to keep them.

### 3. MEDIUM-LOW: a lazy `add` pins the joining row's whole contiguous cache, uncharged

**Where.** `_group_add` at `:3095-3110`, through `cascade.PrefixGroup.add(evaluate=False)`.

**What happens.**

* The lazy `_stack([fa], …)` slices the row's KVCache buffers.
* Until the next decode step evaluates the concat, the graph keeps the row's whole (P + S) fp16 cache
  alive: ~1.07 GB at 32K on the 9B.
* Meanwhile `row.reserved` has already dropped to `_member_bytes()`.
* Hit prefills can run back to back without a decode step in between (`resume` passes during a short
  hit slab plus the next `_admit`). The concats then chain, and each pins its row.

**Verified.** In scratch `lazyadd.py` on the CPU, a 33 MiB row stayed fully resident after
`add()` + `del`, and was freed only after `mx.eval(g.sk, g.sv)`.

**Fix.**

* In `PrefixGroup.add`, `mx.eval(new_k, new_v)` (the S-token fp32 copies, small) before the lazy
  concat. That frees the row's buffers at once and keeps the stack copy lazy (finding 1 of B1).
* Alternative: an eager `add` from `_join`. After a slab's `mx.eval` the in-order stream is already
  drained, so it does not stall. But it pays an all-layer old + new stack transient.

### 4. MEDIUM-LOW: re-accounting on release is never checked, and the P/2 leave raises memory

**Where.** `_cascade_tidy` at `:3141-3165`; plan B.1, "the memory effect is neutral or better".

**What happens.** A member that leaves at S ≈ P/2:

* gets a contiguous (P + S) fp16 cache, 1.5·P·32 KiB, evaluated now;
* gives up a stack slot of `cap`·64 KiB (≈ P·32 KiB) only when the next step consumes the lazy `take`.

So the net change is about +0.5·P·32 KiB per leaver, not neutral.

**Reservation side.**

* Δ ≈ (P − S_res)·32 KiB·1.05: about +96 MiB at 8K and +0.5 GiB at 32K per leaving row.
* It is added unconditionally, past `budget + prefill_reserve` if need be.
* `set_cascade(False)` releases every member at once. With 16 × 32K that is ~16 × 1.1 GB of fp16
  caches next to the live fp32 group: an OOM, but in tools only.

**Fix.**

* Release a P/2 leaver only if `self.reserved + Δ ≤ self.budget + self.prefill_reserve`. Otherwise
  keep it grouped: the rule is a speed heuristic, and grouping stays correct past P/2.
* Have `set_cascade(False)` refuse, or stage, when Σ `_bytes(out)` does not fit.
* Correct plan B.1.
* Dissolves by "memory" or "< 3 members" are fine. They involve few rows, and their per-row total is
  ≤ the group's charge by construction.

### 5. LOW: the eager `release` in `_iterate`'s tidy costs one step its overlap; `abort_step` does useless work

**`release`.**

* `release` evaluates (`cascade.py` default). From `_compact` in `_iterate` it waits for the step in
  flight, so the GPU idles while the host builds the next step: once per leave or dissolve event.
* From `_join`'s `_compact` it is free (the slab eval has drained the stream).
* This is acceptable because it is rare and documented in `cascade.py`, but the worker docstring
  should say so. CLAUDE.md's "never put a new op on a step's outputs" is about per-step reads.
* Every other per-step addition is lazy: `take`, `add`, `begin_step`'s `grow`, `_upos` and the mask.

**`abort_step`.**

* `_grouped_forward` (`:3205`) calls `abort_step`, which runs `mx.eval` over the stacks and re-runs
  the aborted graph.
* `_fail_rows` drops every group immediately afterwards.
* So `abort_step` is wasted work on the error path, with a second chance to raise there. Drop it
  (B1 finding 4 allowed that), or keep it with a comment.

### 6. LOW (perf): every grouped step pays an output take

**Where.** `_group_build` at `:3075/3077`, via `cands = [row] + others` (`:3054`).

**Why.** The joining row comes first in `g.rows` but last in `self.rows`. So `order` in `_plan`
(`:3187`) is never sorted.

**Cost.** Every grouped step uploads `mx.array(argsort)` and runs a gather per FA layer
(`_cascade_rows:1700`). In the CPU wave (scratch `cycle3.py`), all 31 of 31 grouped steps took it.

**Fix.**

* Order the members by batch index at build. The source can stay the joining row:
  `PrefixGroup.build(source=row.cache, members=sorted …)`.
* Adds append at the end in both lists, and `take` keeps the order, so `inv` stays None.
* Nit: when `idx == arange(B)`, pass `queries` / `keys` / `values` without the gather.

### 7. LOW: `QWEN_CASCADE` turns on for "false" or "off"

`_cascade_knobs` at `:1571` uses `not in ("", "0")`. The default-off knobs use `== "1"`
(`PROMPT_CACHE`, `DRAFT_SAMPLING`). For an experimental, inexact feature, use `== "1"`.

### 8. LOW: a failing tidy or join blames the wrong request; `_group_add` is not atomic

**Wrong request.** An exception from `_cascade_tidy` inside `_join`'s leading `_compact()` (a
`release` OOM) goes to `_advance`, which fails the *joining* request although its prefill succeeded.
`_gdn_take` already had the same pattern before this diff.

**`_group_add`.** It mutates `pg` (`g.pg.add`, `:3098`) before `g.rows.append` and `_group_bytes`. An
exception in between would leave `pg.n != len(g.rows)`. The next `_plan` would then raise and fail
the whole batch. That is unlikely, but cheap to avoid: compute the charge first, then `add`, then
assign.

### 9. Test gaps (MEDIUM for coverage, `tests/test_cascade_sched.py`)

**(a) The production path is never run at scheduler level.** The `casc` fixture sets
`CASCADE_CHECK=True`, and every scheduler test inherits it.

* So the default `CHECK=0` path is never exercised there.
* The checks also evaluate the rows' caches (`_same_prefix`, `prefix_equal`). That syncs and can hide
  laziness effects such as finding 3.
* A regression in P selection becomes "row refused with a warning" instead of "wrong prefix bits".
* Fix: parametrize the wave and mixed-batch tests over CHECK on and off.

**(b) Token equality on fp32 cannot see small numeric errors.**

* Consider a wrong-slot or zero-key dilution inside a 150-token prefix. It moves the attention output
  by ~1/(P + S), and greedy argmax on a random model may not flip in 20 tokens.
* The logits-level test (`test_grouped_logits`) covers the kernel, not the scheduler's
  plan / add / take / grow / step-ahead path.
* Fix: capture the grouped rows' per-step logits in the wave test (hook `_logits`) and compare them
  with the serial run at ≤ 1e-5 on `dense`.

**(c) `test_memory_rule` is tautological.** It re-derives the same idealised inequality and never
compares it with `_group_bytes` (finding 2).

**(d) No leak test** (finding 1).

**(e) Not covered:**

* two groups in one batch;
* a member that resumed *deeper* than P (a second-turn hit joining a system-prompt group, so its
  suffix includes restored tokens);
* P = the joining row's `len − 1` end, which is a bound of the others;
* cancelling a grouped row mid-batch;
* `_regrow`'s "ended at" for a grouped row;
* an exception inside `_cascade_join` being swallowed (the row continues ungrouped);
* a thinking-budget forced token in a group;
* two back-to-back joins with no decode step between (finding 3).

**(f) Default off is adequately covered.** `test_off_is_today` (plus the 160 other tests running
with the feature off) and inspection (see Verified OK) suffice. Counting MLX ops is not needed.

### 10. Docs and nits

* **Plan B.3, "Donation".** It still says the previous step's read has finished "because this step
  depends on the previous step's token". B1 finding 5 asked for a correction; the module docstring
  has the right mechanism.
* **Plan B.1.** The "neutral or better" claim is wrong (finding 4).
* **README and CLAUDE.md** describe the rule as "would take more KV than the members' own caches".
  That is approximate (finding 2).
* **Dead code.** `_PrefixGroup.key` (`:1618`, a `hash(tuple(ids))` over up to 32K ids) is never
  used.
* **Redundant condition.** The "batch of ≤ 2" dissolve is subsumed by "< 3 members", since members
  are ≥ 3 implies B ≥ 3. Harmless.
* **`set_cascade`.** Its docstring should say it must be called on the scheduler thread (it builds
  and evaluates MLX). It has no non-test callers yet.
* **`row.cached` is not what makes the bits equal.** `_bounds` does: a cold row that never stored an
  entry holds the same `[0, P)` bits. The gate is harmless and conservative. The same holds for the
  DFlash exclusion, which is stricter than B.6.

## Verified OK

**1. Default off is bitwise today.**

* With `QWEN_CASCADE=0`, `self.cascade` is False and `self.groups` is always `[]`.
* `_step` builds `plan = None`, and `RowsKV(caches, i, None)` adds only a Python int.
* `_rows_attention`'s `plan is None` branch builds the same ops. The old one-liner
  `concatenate(...).transpose(...).reshape(...)` was split across statements, giving an identical
  graph.
* `_compact`'s restructure is equivalent, and `_regrow`'s `g is None` branch is the old code.
* `_join`, `row.cached`, `row.spec` and the B=1 `r.group` check are Python only.
* No MLX op or eval is added at B ≥ 2 or B = 1. `fa_layers` is read at `__init__` from `is_linear`,
  which exists on both qwen3_5 and qwen3_5_moe.

**2. P selection is sound, given hit == cold.**

* `_bounds` decides a cut at i from ids[0..i]. So:
  * for rows with LCP ≥ P, every bound below P agrees;
  * P itself depends on ids[P], so it is checked per row (`P in _ends(r)`, `_fits_prefix`).
* The slab partition of [0, P) is therefore identical: a hit resumes at a bound, and the following
  slabs follow the bounds.
* That holds for the joining row's `len − 1` end too, since its last slab starts at the shared last
  bound.
* A row whose prompt diverges inside the last slab before P has LCP < P and is excluded.
* The prefix source is always the joining row's own evaluated cache. The producer and every other
  member contribute only `[P, offset)`.

**3. Positions and masks.**

* `_GroupRef.offset` is read only through `RowsKV.offset`, from `LanguageModel`'s `cache_offset`.
  There it is compared with 0, and position slicing happens only when `position_ids` is longer than
  the input. The worker passes explicit `(B, 1)` positions.
* The mask is None at L = 1, and `_rows_attention` ignores it anyway.
* `r.pos`, `cache.offset` and `pg.lens` stay in step (`check_lens`, Python, every step).
* GDN is untouched.
* The ungrouped rows of a mixed batch are bit-identical (`test_grouped_logits`, rows 3..5).

**4. Lifecycle.**

* Rows that finish during a step stay in the group until the next `_compact`, then `take` drops them.
* Cancels go the same way.
* Thinking-budget forced tokens change only `r.y`.
* Sync mode for grammar rows is tested.
* Two groups are handled by `_plan` and `inv`; the code is correct but untested.
* A spread of `max_tokens` is tested (6 / 8 / 30 / 30).
* `_regrow` for a grouped row grows `g.reserved`, and "ended at" uses the group-level headroom.

**5. Step protocol.**

* `begin_step` runs before the graph is built; `end_step` runs after it is built and before
  sampling and `async_eval`. Both only touch Python state and build lazy small arrays.
* `_plan`'s argsort is host numpy plus one small upload.
* The per-step host cost at B = 16 is a few dict and list passes, plus one `mx.array` per group.
* `grow` (every 256 steps) is lazy, and its copy is ~8 GiB of traffic per 256 steps at 16 × 4K:
  negligible.
* Step-ahead plus groups is correct on CPU: every scheduler test runs ahead mode.

**6. Failures.**

* A host error in the build goes through `abort_step`, then `_fail_rows` drops every group.
* An eval-time error surfaces at the next `r.y.item()`, and `_fail_rows` fails all rows, as for
  today's failed step. The 3-strike counting is unchanged.
* `_cascade_join` exceptions are logged with `log.exception`, and the row continues ungrouped.
* `PrefixGroup.build` and `add` run before any scheduler state changes (except finding 8's
  ordering).
* `_close` drops the groups.

**7. Accounting invariants.**

* `check_accounting` (in the tests): `reserved = ungrouped rows + members' _member_bytes + Σ g.reserved
  + the prefilling row`, and `g.reserved == _group_bytes(g.P, g.rows)` at every tick.
* `_end` subtracts the member part, and tidy recomputes the group's charge.
* `_fits` sees the group charges.

**8. Proto, gateway and Health.**

* Fields 11 and 12 are added. The generated `worker_pb2` accepts them (checked).
* The gateway logs them at debug level on change.
* Health reads plain ints that the scheduler keeps current.

**9. `fit_max_batch` and the load log.**

* `fit_max_batch` turns cascade off below 3.
* The load log line states "on", or why it is off.
* The env rows in CLAUDE.md and README match `_cascade_knobs`.

## For B4 (GPU)

* Run `QWEN_CASCADE_CHECK=1` with an entry produced by a DFlash prefill. `_spec` calls `lm.model`
  directly; hit == cold for those entries is assumed, not tested here.
* Check that a released cache (capacity P + S + 256, not a multiple of 256) keeps per-row SDPA bits
  on Metal. That was CPU-verified in B1.
* Donation under step-ahead in the real worker (B1 finding 5).
* Measure the formation, add and release transients against `_group_bytes`, after fixing findings 1
  and 3, or the peak numbers will include the pinned arrays.
