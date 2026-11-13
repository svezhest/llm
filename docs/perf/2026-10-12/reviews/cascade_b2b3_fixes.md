# Re-review: the fixes for cascade B2 + B3 (`reviews/cascade_b2b3.md` findings 1–10)

**Scope.** The uncommitted diff on HEAD ee00db4, checked against the original review:

* `src/worker/worker.py`, `src/worker/cascade.py`;
* `tests/test_cascade_sched.py`, `tests/test_cascade.py`;
* CLAUDE.md, README, BACKLOG, `cascade_plan.md` "As built".

**How it was reviewed.** Read-only, CPU only.

* Suite: `uv run --group dev pytest -q --ignore tests/test_mrqmm.py --ignore tests/test_pfqmm.py` gives
  **198 passed, 2 skipped** (141 s). It was run once.
* Two scratch CPU scripts ran in the session scratchpad. Nothing in the repo was written except this
  file.
  * `quirk.py` shows that an fp32 slice update covering the whole zeros buffer aliases its source:
    the source buffer is not freed after `del`. With S + 1 slots the buffer is freed. fp16 sources
    are always freed.
  * `slice.py` shows that `cascade.attention` on `q[1:4]` / `k[1:4]` is bitwise equal to the gather
    `q[[1, 2, 3]]` on the CPU.
* The byte numbers below use the 9B formulas: `_group_bytes`, `_bytes`, and `_prefill_reserve` =
  2.39 GiB.

## Findings (ranked)

### A. MEDIUM (when on): the release check tests the state *after* the release, not the peak, so a multi-row optional release can pass the headroom by the group's own size

**Where.**

* `worker.py:3267-3270`: `_tidy_choice.fits`;
* `worker.py:3299`: `_tidy_group`.

**Mechanism.**

* `fits` credits the group's shrink. Its delta includes `_group_bytes(stay) − g.reserved`, or
  `− g.reserved` for a full dissolve.
* `_tidy_group`, however, builds and **evaluates** every `release` while the whole group is alive.
  The `take` is lazy, so the old stacks also stay alive until the next step.
* So the real transient is `pre-state + Σ new fp16 caches`. The check tests `pre-state − group +
  Σ caches`, which is lower by the group's live arrays.

**Where it bites.** A release of several rows in one tidy:

* a memory dissolve, which releases every live member at once;
* several P/2 leavers in the same tick;
* a leave that drops the group below 3 members.

**Numbers** (9B, reservation = real + 1024 after `_regrow`):

| case | group live | fp16 caches | over the check |
|---|---|---|---|
| 8K × 8, memory dissolve at S = 3584 | 2.38 GiB | 2.94 GiB | **+2.0 GiB** |
| 32K × 8, memory dissolve at S = 15360 | 9.6 GiB | 11.8 GiB | **+8.8 GiB** |
| 16 × 8K, all crossing P/2 in one tick | | | +3.8 GiB |

* The check already spends the prefill reserve: headroom = budget + prefill reserve. Only the 1 GiB
  slack is left beyond it.
* So both memory-dissolve cases can pass the cap when the batch is near its headroom.
* The 32K case passes the check whenever the others hold ≲ 0.7 GiB. Its pre-state charge
  (12.8 GiB) is admissible on a 27 GiB cap.
* The P/2 case is usually staggered by the serialized joins, so it is lower.
* The docs claim "optional releases are now bounded by the headroom" (`cascade_plan.md:940`, README,
  CLAUDE.md). That holds for the post-state only.

**Fix.**

* Gate on the transient. Do not credit the group in `fits`: `self.reserved + Σ(_bytes(r.tokens) −
  r.reserved for r in out) (+ pc.bytes, see B) ≤ headroom`.
* Stage memory dissolves (and P/2 leaves) like `set_cascade(False)`: one member per tidy, cheapest
  first, with the last fewer-than-3 going together.
* Each staged `take` frees that member's slot at the next step. The per-tick peak is then one cache
  on top of the live group.
* Update `test_release_waits_for_headroom`. It codifies the post-state check.

### B. MEDIUM (when on): both guards ignore the prompt cache, which the scheduler counts inside the budget

**Where.**

* `worker.py:3066`: `_charge_fits` checks `reserved + delta ≤ budget`;
* `worker.py:3267-3270`: `fits` checks `reserved + delta ≤ budget + prefill_reserve`.

**The rest of the scheduler.**

* Admission (`_fits`) requires `reserved + need + pc.bytes ≤ budget`.
* `_regrow` requires `rows + pc.bytes ≤ headroom`, and evicts LRU entries first (keeping `task.parent`).

**Why the guard is a no-op.**

* Cascade requires `QWEN_PROMPT_CACHE=1`, and its members resumed from cached entries. So `pc.bytes`
  is typically 1–3 GiB (soft cap 3) exactly when groups form.
* With admissions filled to the budget (`reserved + pc = budget`), the guard lets a formation add up
  to `pc.bytes`.
* That covers every positive delta in finding 2's table (+0.23 … +1.69 GiB). The original finding's
  overshoot therefore still happens in the realistic case.
* `test_budget_guard` sets `budget = reserved + delta + room` and asserts `reserved ≤ budget`, so it
  encodes the pc-blind form.

**Fix.**

* `_charge_fits`: `delta <= 0 or self.reserved + delta + self.eng._pcache.bytes <= self.budget`.
  Refuse rather than evict: grouping is optional, and evicting the shared system-prompt entry costs
  future TTFT.
* `fits`: add `pc.bytes`.
* For forced releases, evict as `_regrow` does: `while … pc.evict(keep)`.
* Adjust both tests.

### C. LOW-MEDIUM (when on): a forced release of 2 rows is unchecked, and at long P it passes the reservation

**Where.** `worker.py:3261`; caller's question (4).

**Mechanism.** A forced release happens when no live member is left or fewer than 3 are. Its peak is
`pre-state + 2 × (P + S + 256) × 32 KiB`, with the group still alive. The group's charge absorbs part
of that: score transients, `moves`, the reservation slack in the stack, and 64 MiB.

**Overshoot past `reserved`** (S = 200):

| case | overshoot |
|---|---|
| 16 × 32K, 14 finish in one tick, default 4096 reserve | −3.9 GiB (covered) |
| 16 × 32K, same, `max_tokens` ≈ 200 (s_res 400) | +1.06 GiB |
| 3 × 32K, s_res 400 | +1.8 GiB |
| 3 × 64K, s_res 400 | +3.7 GiB |
| 3 × 96K, s_res 400 | +5.6 GiB |

* `reserved` may sit anywhere up to the headroom, since `_regrow` admits growth to it. Only the 1 GiB
  slack is outside.
* So 3-member groups over ≳ 32K prefixes with short answers can OOM on dissolve. One example is
  three questions on one long document.
* The forced release itself is unavoidable: B ≤ 2 must be the per-row path, and the members' KV lives
  only in the group. The accounting after the release is honest.

**Fix.**

* Charge the dissolve floor in `_group_bytes`: `max(terms, prefix32 + stacks + 2 × int((P + s) ×
  kv_fp16 × 1.05))`. That is, a group must always be able to turn back into 2 contiguous rows.
* At 3 × 96K this makes the group lose to its rows (12 vs 9 GiB), which is the honest answer. At
  16 rows it still wins by far.
* Alternatively, release the two rows sequentially, dropping `kc` / `vc` after the first one. That
  peaks at ~3P instead of 4P fp16-tokens.

### D. LOW: `_GroupRef.offset` dereferences a dead weakref without a check

**Where.** `worker.py:1606`, `self._row().pos`.

Unreachable today: a `_GroupRef` lives only in its own row's `cache`, and unlinked entries become
None. Nit: raise a clear `RuntimeError` instead of `AttributeError: 'NoneType'`.

## Fix-by-fix verification

1. **Cycles: OK.**
   * `_GroupRef` has `__slots__` with a weakref to the row and no group.
   * `_unlink` / `_unlink_row` are used in:
     * `_tidy_group` (done rows; released rows `group = None`; `g.rows = stay`);
     * the tidy-failure path;
     * `_drop_groups`, from `_fail_rows` and `_close`.
   * Nothing else keeps a group:
     * `_StepPlan` / `RowsKV` / `plan` are locals of `_step`, not stored on the scheduler.
     * `cascade.compiled` closures capture only `scale`; arrays are inputs.
     * `fits` / `without` are closures of a returning frame, not self-referencing.
     * `_PrefixGroup.ids` is a slice copy.
     * The MLX graph holds arrays, not Python objects.
   * Exceptions:
     * `_fail` and `_end` take `str(e)`.
     * `log.exception` records go to `basicConfig`'s StreamHandler, which retains nothing.
     * `except … as e` unbinds at block end, so the traceback frames (`_tidy_group`'s `caches`,
       `_grouped_forward`'s `plan`) die then.
   * Test: `test_dropped_group_freed_without_gc` (gc off; dissolve, failed step, close) is a real
     regression test.
   * Not leak-tested: the tidy-failure drop. Under `caplog` the record would hold the traceback
     anyway.
2. **Rule and guard: formulas consistent; guard incomplete (B).**
   * `_group_wins`: `_group_bytes(P, rows, {r: pos + 1})` ≤ Σ `_bytes(pos + 1)` − n·`_member_bytes()`.
     That equals Σ `int(n·kv·1.05)`, because `_bytes` = that + 2·gdn_row + 8 MiB = that + member.
     Consistent.
   * The break-evens in the docstring match `test_memory_rule_bytes`.
   * The delta at form and at join equals exactly what `_group_build` / `_group_add` move into
     `reserved`.
   * The guard is applied at form and at join.
   * `_regrow` uses the ordinary headroom path (pc-aware, ends the row), not `_charge_fits`. That is
     consistent with ungrouped rows.
   * A refused join leaves the row ungrouped with `reserved = _bytes(tokens)`, untouched. The
     accounting is correct.
   * Side effects of a refused join (nit): the row then falls through to formation, which can build
     a second group at the same P from other refused rows and duplicate the prefix. A refused row is
     never retried for a join.
3. **`add` with S + 1 slots: OK and robust.**
   * The S + 1 buffer is sized independently of `cap`, so the update is always partial.
   * When `cap == S + 1`, `_pad` returns the evaluated private buffer, and the lazy concat copies it.
   * S = 0 gives zeros(1).
   * The quirk is reproduced on the CPU (fp32 aliasing).
   * The values written are unchanged, so `test_cascade`'s bitwise claims hold.
   * The new `mx.eval(own)` from `_join` is cheap: the slab eval has drained the stream.
   * The new transient (S × 64 KiB, S ≤ P/2) is below the pinned (P + S) × 32 KiB it replaces.
   * `test_lazy_add_frees_the_row_at_once` covers F16 and F32.
4. **Forced vs optional: logic right; peak and accounting as in A, B and C.**
   * Done rows always leave, even when every optional release is refused (`len(stay) !=
     len(g.rows)`).
   * Staging for `set_cascade(False)` is correct.
   * The 16 × 32K case asked about is fine with default reservations. 14 finishing leaves 2 forced
     releases, −3.9 GiB vs the charge. It is +1.06 GiB at `max_tokens` ≈ 200 (C).
5. **No `abort_step`: OK.**
   * A host error in `_plan` / `begin_step` / the forward / `end_step` propagates through `tick` to
     `_run`, then `_fail_rows`.
   * Every row fails, grouped or not, as for today's failed step.
   * Every group is dropped, including any left `_pending`. No survivor reads a half-written stack.
   * Per-row sampling failures after `end_step` are individual `_fail`s. The slot was written for all
     members, so the lens and positions agree.
6. **Slice plan: OK.**
   * `_plan` emits a slice only for ascending consecutive indices.
   * The whole-batch shortcut skips a one-array concatenate, which is an exact copy anyway.
   * Slice vs gather is bitwise equal on the CPU (scratch). Metal is unverified: add a check to B4,
     or a CPU assert in `test_grouped_logits`.
   * Nit: `order = single + groups` means any ungrouped row placed after a group forces `inv`. So
     does a join after ungrouped rows, which also makes the group a gather. The cost is a (B, Hq, 1,
     D) gather per FA layer: negligible.
7. **Knob: OK.** `== "1"` for `QWEN_CASCADE` and `_CHECK`. `test_knobs` covers "false/off/no/true/2".
8. **Atomic add and per-group tidy failure: OK.**
   * The charge is computed before `pg.add`, and `pg.add` assigns last.
   * `_cascade_tidy` catches per group:
     * it fails the live members;
     * `reserved −= g.reserved`, while `_end` already removed the member parts;
     * it calls `_unlink`.
   * `_compact` re-tidies until no failure. The joining request goes on, as tested.
   * Nit: the handler is not idempotent. An exception raised after `_tidy_group`'s dissolve branch
     has run `self.groups.remove(g)` / `reserved −= old` would remove the group twice and subtract
     twice. Only `log.info` follows there, so this is theoretical. Guard with `if g in self.groups`.
9. **Tests: meaningful.**
   * The wave test's `0 < d ≤ 1e-5` compares per-step logits with the same arrival schedule and
     cascade off.
     * The `> 0` proves grouped steps ran: the fp32 cascade differs by ~1e-7.
     * A diluted or shifted prefix moves the output by ~1/(P + S) ≈ 6e-3 relative, far above 1e-5.
   * `CASCADE_CHECK` is parametrized on wave, mixed and back-to-back.
   * The leak test with gc off is real (see 1).
   * The new edge tests cover review gap 9(e) entirely.
   * Gaps: no test of the release transient (A) or of `pc.bytes` in the guards (B).
   * Two tests encode the current, flawed checks: `test_budget_guard` and
     `test_release_waits_for_headroom`.
10. **Docs: corrected, except for two overclaims.**
    * Plan B.1 and B.3 are corrected, and the README / CLAUDE.md rule text matches `_group_wins`.
    * Two overclaims remain:
      * "A group never pushes the batch past its budget" / "never pushes `reserved` past the budget"
        (`cascade_plan.md:901`, `worker.py:2173`, CLAUDE.md:456) is false while `pc.bytes > 0` (B).
      * "optional releases are now bounded by the headroom" is true for the post-state only (A).

## Default off (`QWEN_CASCADE=0`): re-verified, no new MLX op or eval

* `_compact` is `_drop_done` (the old body) plus `while self.groups and …`, which is skipped.
* `_join` returns at `if self.cascade`.
* `_step`: `plan = None` (Python); `RowsKV(…, i, None)`.
* `_rows_attention`'s None branch is the old loop. The old one-line concatenate → transpose → reshape
  is split, giving the same graph.
* `_regrow`'s `g is None` branch is the old code.
* `_fail_rows` and `_close` gain `_drop_groups()`, which is Python.
* `_fail` logs `str(e)`: the same text.
* The B = 1 `r.group` check is Python. `row.cached` and `row.spec` are Python.
* Health uses `getattr` ints.
* Module imports:
  * `cascade` creates no arrays at import (dtype constants and `math.inf` only);
  * `KVCache` is a class import.
* `fa_layers` uses `is_linear`, already required elsewhere (`_kv_token_bytes`, `_install_rows_attention`).
