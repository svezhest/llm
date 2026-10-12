# Prompt cache phase 2: segment store (A) + cascade decode attention (B): implementation plan

Date 2026-10-12. Read-only design, CPU only. Line numbers refer to the **working tree** of
`src/worker/worker.py` (phase-1 diff applied, 3461 lines), mlx 0.32.3 / mlx-vlm 0.7.6 in
`src/worker/.venv`. CPU suite at the time of writing: `83 passed` (`uv run --group dev pytest -q
--ignore tests/test_mrqmm.py --ignore tests/test_pfqmm.py`, 39 s).

Inputs: `prompt_cache_opt.md` (proposals 1, 2, 3; §D1/§D2), `prompt_cache_opt_2.md` (N1–N5, item-1
refinement "group by token prefix"), `cascade_microbench.md` + `scripts/perf/microbench/microbench_cascade.py`
(`b32c`, pitfalls), `reviews/prompt_cache_phase1.md` (findings 1–7), `prompt_cache_bench.md`,
`step_profile.md`. Model numbers are for the 9B (Ornith-1.5 8-bit, fp16 KV): 8 full-attention (FA)
layers × 4 KV heads × head_dim 256 = **32 KiB/token** fp16 (64 KiB fp32); one GDN state 51.5 MB.

Three facts were checked on the CPU for this plan (scratch scripts, nothing in the repo):

* **[CPU] `mx.contiguous` of a slice copies.** An axis-2 slice of a (1, 4, cap, 256) KV buffer, and also
  a row-contiguous axis-1 slice (the GDN conv-state case of `prompt_cache_opt_2.md` open question 1).
  The source buffer is freed once the source is deleted. So a segment cut out of a row's cache is a
  fresh buffer and does not pin the row's buffer. Re-check this once on Metal: test A-T5 does.
* **[CPU] The restore layout does not change bits.** Today a restore hands out a view `[:at]`, and
  the first update concatenates `[view, zeros(ceil(n1/256)·256)]`. A restore that concatenates 2
  segments plus that zeros tail gives the same buffer shape, the same KV bits after the suffix
  prefill, and the same next-token logits (dense 8-bit fp16 tiny model).
* **[CPU] The `b32c` formulation is generic in (Hq, Hkv, D).** It was checked at (4, 2, 64), the tiny
  test model, and (16, 4, 256), with a padded and masked prefix, ragged suffixes, and 1e4 garbage
  in unused suffix slots. rms rel error vs an fp32 reference was 2.0–2.1e-4, the output was finite,
  and the GQA layout matched per-row SDPA.

---

## Summary

* **Part A (byte-identical).**
  * Entries become chains of immutable KV **segments**: own buffers, shared between entries.
  * A store copies only `[at, len)` out of the row, so the row's buffer is no longer pinned and
    the COW copy on its first decode write goes away.
  * A restore concatenates the pieces once, into exactly the buffer today's first update would have
    built.
  * Accounting counts unique buffers.
  * Review findings 1–5 are folded in.
  * Result: 16 siblings of an 8K prompt fit in **1.38 GB, all 16 kept**. Today 6 are kept in 3.18 GB
    charged.
  * Size: ~170 lines in the worker, ~250 lines of tests.
* **Part B (numerics change for grouped rows at B ≥ 3 only).** Rows sharing a token prefix `[0, P)`
  form a `_PrefixGroup` once ≥ `QWEN_CASCADE_MIN_ROWS` (8) of them share a prefix of
  ≥ `QWEN_CASCADE_MIN_PREFIX` (4096) tokens.
  * The group holds one **fp32** copy of the prefix, chunked by 1024 (padded with zeros, plus an
    additive mask). Its members' suffixes sit in one stacked fp32 buffer per FA layer.
  * Attention per layer: ungrouped rows keep today's per-row path, bit-identical. Each group runs
    the microbench's `b32c` under `mx.compile`. Outputs are put back in row order with one take.
  * Groups dissolve whenever B ≤ 2 or a group drops to 1 member, so B ≤ 2 stays today's code path.
  * Expected gains:
    * 16 × 8K shared: decode 239 → ~290–310 tok/s.
    * 16 × 32K shared: becomes admittable (rows' KV 17 GB → ~2.7 GB), ~×1.7 per step.
    * 8 × 32K: +35–50 %.
  * Rollback: `QWEN_CASCADE=0`.
  * Size: ~400 lines in the worker, ~350 lines of tests, ~250 lines of GPU tools.
* **Deviations from the brief**, argued below:
  * (B2) The suffix is a **stacked per-group fp32 buffer** (the measured `b32u` layout), not a
    per-row KVCache.
  * (B4) The fp32 prefix is built from **the forming row's own cache**: same bits as the entry's
    segments, and always alive.
  * (B5) Group bytes are charged **per group, on the max member reservation**: a ragged stack
    would otherwise be under-charged.

---

## Part A: segment-sharing store

### A.0 What changes and what does not

| | today (phase 1) | Part A |
|---|---|---|
| entry KV | `_copy_cache(cache)`: views of the hit row's whole private buffer (`store` 870) | chain of pieces `(seg, lo, hi)`. Parent pieces are shared up to `at`; one new segment `[at, L)` is copied out of the row |
| row's first decode write after a store | COW copy of the whole buffer (the entry pins it) | in place (nothing pins the row's buffer) |
| restore | `_copy_cache(..., upto=at)`: views `[:at]`, copied by the first `update_and_fetch` | `mx.concatenate(pieces + [zeros(round_up(first, 256))])`, `offset = at`. One copy, the same buffer shape |
| entry end state | inside `e.cache` (ArraysCache objects) | `e.end` (list of state lists), or None |
| `bytes` | unique arrays by `id()` (`_recount` 786) | unchanged mechanism. Segments are own buffers, so it is exact |
| absorb / covered rule, `protect` / `_thin`, quantized floor in `best()` | 833–880, 736–748, 772 | unchanged, plus finding 5 (sub-floor checkpoints dropped on quantized entries) |

The KV bits, GDN states, slab cuts and resume positions are all unchanged, so every output stays
**byte-identical**. That includes the hit == cold invariant.

### A.1 Data structures (replace `_Entry` 751–774; new `_Seg` above it)

```python
class _Seg:
    """KV of tokens [a, b) of every full-attention layer, in buffers of its own
    (copied out of a row's cache, never a view of one). kv[j] = (k, v) of the
    j-th FA layer in cache order: fp16 (1, Hkv, b-a, D), or for quantized KV
    (packed, scales, biases) tuples. Immutable; shared by every entry whose
    chain covers it; freed when the last referrer goes (plain refcounting)."""
    __slots__ = ("a", "b", "kv", "arrays", "nbytes", "__weakref__")   # weakref: tests

    @classmethod
    def copy_from(cls, cache, a, b):   # one mx.contiguous slice per array, then mx.eval
        ...


class _Inherit(typing.NamedTuple):    # what a prefill needs of its parent (finding 4)
    pieces: list                      # [(seg, lo, hi)], cut at `at`
    ckpts: dict                       # parent.ckpts restricted to p <= at
    end: list | None                  # parent.end when at == len(parent.ids), else None
    quantized: bool


class _Entry:
    def __init__(self, ids, chain, ckpts, end, layout, quantized):
        self.ids = ids                # prompt[:-1]
        self.chain = chain            # [(seg, lo, hi)]: lo/hi absolute, [seg.a, seg.b) ⊇ [lo, hi),
                                      # chain[0].lo == 0, contiguous, chain[-1].hi == len(ids)
        self.ckpts = ckpts            # pos -> recurrent states (shared objects), pos < len(ids)
        self.end = end                # recurrent states at len(ids), or None (A3 may drop it)
        self.layout = layout          # per layer: ("kv",), ("q", group_size, bits) or ("arr", size)
        self.quantized = quantized
        self.used = time.time()
        segs = {id(s): s for s, _, _ in chain}
        self.arrays = [a for s in segs.values() for a in s.arrays] \
            + [a for st in ([end] if end else []) for layer in st for a in layer if a is not None] \
            + _state_arrays(ckpts)
        self.nbytes = sum(a.nbytes for a in {id(a): a for a in self.arrays}.values())

    def cut(self, at) -> list:        # pieces covering [0, at)
        return [(s, lo, min(hi, at)) for s, lo, hi in self.chain if lo < at]

    def best(self, ids, ends): ...    # as today (765–775); the end position only if self.end is not None
```

* **A straddled parent piece is kept whole.** A parent piece with `lo < at < hi` keeps its segment
  whole (the bytes are the parent's anyway) and is cut logically to `hi = at`. A3 below trims dead
  tails.
* **`layout`.** `_layout(cache)` records each layer's class and params, so `restore` can rebuild
  caches without `lm`. It uses the same constructors as `_copy_cache` today (667–696). Keep
  `_copy_cache` for the B ≥ 2 tooling and tests; `restore` no longer uses it.

### A.2 `_PrefixCache` API (777–880)

```python
def peek(self, ids, ends) -> tuple[_Entry | None, int]:
    """lookup() without side effects (no `used`, no hit/miss counters): for _fits."""

def lookup(self, ids, ends): ...         # unchanged (795)

def restore(self, entry, at, first=1):
    """Caches for a row resuming at `at`. Each FA layer: one concatenate of the
    entry's pieces over [0, at) plus a zeros tail of round_up(first, 256)
    tokens (`first` = the first update's token count: the first slab, or 1 when
    no slab runs). That is exactly the buffer today's first update_and_fetch
    built from the [:at] view (cache.py 513-522), so the first update writes in
    place and nothing ever holds a view of a segment. GDN: entry.end at
    len(ids), else entry.ckpts[at] (fresh lists, shared arrays, as today)."""

def store(self, ids, cache, ckpts, inherit=None, at=0, ends=(), protect=(), limit=None):
    """-> (entry or None, absorbed count). See A.3."""

def bytes_of(self, entries) -> int:      # unique bytes of a subset (finding 2's floor)
def evict(self, keep=()) -> bool:        # keep: a collection of entries (was one entry)
```

* **`closest()`** (815) becomes unused after finding 1. Delete it.
* **Restore for quantized KV.** Each of the 3 components is concatenated and gets a zeros tail
  (uint32 packed, fp16 scales and biases), as `QuantizedKVCache.update_and_fetch`'s `expand_quant`
  would build it (`cache.py` 337–378).

### A.3 `store` (833–880), step by step

1. `quantized = _quantized(cache)`. `L = len(ids)`.
2. **Reuse the parent's pieces** only if `inherit is not None and inherit.quantized == quantized`:
   `pieces, lo = inherit.pieces, at`. Otherwise `pieces, lo = [], 0`.
   * Case: an fp16 parent below `quantized_kv_start` and a new row quantized by its prefill. Then
     `to_quantized` re-encoded `[0, at)` in the row, so the parent's fp16 pieces are not this
     entry's bits. Copy the whole `[0, L)` out of the row.
   * The reverse case cannot occur (see the review's "Quantized floor").
3. `seg = _Seg.copy_from(cache, lo, L) if L > lo else None`. `L == lo` is the exact-repeat case: no
   slab ran, and the chain is the parent's pieces up to L. `chain = pieces + [(seg, lo, L)]`.
4. **Checkpoint inheritance**, as today (848–853):
   * `ckpts.setdefault(p, s)` for `inherit.ckpts`;
   * `ckpts[at] = inherit.end` when `inherit.end is not None and at < L`.
5. **Covered / absorb**: unchanged (859–866), except that an absorbed entry's end state is `e.end`
   (skip it when None) instead of `_recurrent_states(e.cache)`.
6. **Finding 5**: `if quantized: ckpts = {p: s for p, s in ckpts.items() if p >= KV_KWARGS["quantized_kv_start"]}`.
   Read the floor from the scheduler's kv dict: pass `kv_start` in, because tests use `kv_start=40`.
7. `ckpts.pop(L, None)`, then `_thin(ckpts, PROMPT_CACHE_CKPTS, protect)`.
8. `end = _recurrent_states(cache)`, then `new = _Entry(ids, chain, ckpts, end, _layout(cache), quantized)`.
   `mx.eval(seg.arrays)` runs inside `copy_from`.
9. `self.entries = [non-covered] + [new]`, then `_recount()`, then the soft-cap LRU loop as today
   (`evict(keep=(new,))`).
10. **Finding 3**:
    ```python
    if limit is not None:
        while len(self.entries) > 1 and self.bytes > limit and self.evict(keep=(new,)):
            pass
        if self.bytes > limit:          # the newest alone does not fit beside the rows
            self.entries.remove(new); self._recount()
            log.info("prompt cache: entry of %d tokens (%.2f GiB) not kept: over the batch headroom", ...)
            return None, len(covered)
    ```
    `snapshot` logs the store line only when an entry is returned. Keep the `RX_STORE` format (the
    test and `bench_concurrency.py` parse it).

### A.4 `_prompt_cache` (2128–2195) and `_Row` / `_Req`

* **`_Req` (1741)**: add `bounds: list | None = None`, memoized `_bounds(ids, im_start)`. It is
  shared by `_fits` (finding 1) and `_prompt_cache`. Add a helper `_ends(req)` →
  `set(bounds) | {len(ids)-1}`.
* **`_Row.__init__` (1785)**: add `self.parent = None` (the entry the in-flight prefill resumed
  from, for finding 4) and `self.group = None` (Part B).
* **On a hit**:
  ```python
  first = _slab(reused, len(ids) - 1 - reused, set(bounds)) if len(ids) - 1 > reused else 1
  cache = pc.restore(parent, reused, first)
  inherit = _Inherit(parent.cut(reused), {p: s for p, s in parent.ckpts.items() if p <= reused},
                     parent.end if reused == len(parent.ids) else None, parent.quantized)
  row.parent = parent
  ```
  `first` is exactly the `n` that `_prefill` (2245) and `_spec` (2342) compute for their first slab.
* **`snapshot`** (2179–2193):
  * call `pc.store(ids[:-1], cache, ckpts, inherit, reused, ends, protect,
    limit=self.budget + self.prefill_reserve - self.reserved)`;
  * then `row.parent = None`;
  * the closure no longer captures `parent` (finding 4, second half).
* **Finding 5 in the plan**: when `self.kv["kv_bits"]` and `len(ids) - 1 >= kv_start`, drop
  `b < kv_start` from `keep` and `protect` (2170–2173).

### A.5 Admission and growth (`_fits` 2054–2078, `_regrow` 2022–2052): findings 1, 2, 4

```python
def _fits(self, req, rows):
    if rows == 0: return True
    need = self._reserve(req)
    if rows >= self.max_batch or self.reserved + need > self.budget: return False
    pc = self.eng._pcache
    live = mx.get_active_memory() + (0 if CLEAR_CACHE else mx.get_cache_memory())
    if self.reserved + need + pc.bytes > self.budget:
        keep = ()
        if PROMPT_CACHE and not req.images:                          # finding 1
            e, depth = pc.peek(req.ids, _ends(req))
            if e is not None and depth >= PREFILL_STEP:              # a real depth, not 256 tokens
                keep = (e,)
        floor = pc.bytes_of(keep)
        if (self.reserved + need + floor > self.budget               # finding 2: evict nothing
                or live - (pc.bytes - floor) + need + self.prefill_reserve > self.cap):
            return False
        n = 0
        while self.reserved + need + pc.bytes > self.budget and pc.evict(keep):
            n += 1
        ...log...
        live = mx.get_active_memory() + ...                          # evictions free memory now
    return live + need + self.prefill_reserve <= self.cap
```

* **Keep threshold.** `PREFILL_STEP` (2048 by default; 8 in the `small` test fixture) stands in for
  the review's `KEEP_MIN_TOKENS`: one full slab, ~2–4 s of prefill. No new knob.
* **`_regrow`, finding 4.** Protect the in-flight prefill's parent:
  `keep = (self.task_row.parent,) if self.task_row is not None and self.task_row.parent is not None else ()`,
  then `pc.evict(keep)` (2039).
* **Part B also changes `_regrow`'s byte math** (B.5). Land A first; B adds the group branch.

### A.6 A3 (optional, byte-identical): dead-tail trim and end-state pressure

* **Trim on absorb.** At store, when a piece `(s, lo, hi)` of the new chain has `hi < s.b` and no
  surviving entry other than `new` references `s` (scan `self.entries`), replace it with
  `_Seg.copy_from` of `[lo, hi)`. The source is `s` itself: slice, `mx.contiguous`. That costs one
  copy of that segment's used part, typically the previous turn (~650 tokens, 21 MB). The first
  segment (system + turn 1, ~8K tokens, 268 MB) is copied once, at turn 2.
* **End states under pressure.** In the soft-cap / `limit` / `_fits` loops, first set `e.end = None`
  on LRU entries, oldest first, then evict entries. Keep end states otherwise: they serve
  *regenerate* (an exact repeat, 0-token prefill instead of ≤ 256).

### A.7 Memory estimates

**16 siblings of one 8K system prompt** (`prompt_cache_bench` A1: cold 8336 tokens with ckpts
{2048, 4096, 6144, 8192}; 15 hits at 8192 with ~150-token suffixes):

| | today (phase 1, 3 GiB soft cap) | Part A |
|---|---|---|
| prefix KV | one full copy per entry, 272 MB each | 1 segment `[0, 8336)`: 273 MB |
| sibling KV | (inside the above) | 15 × ~150 tokens × 32 KiB = 74 MB |
| checkpoints (4) | 206 MB, shared objects (phase 1 counts them once) | 206 MB |
| end states | 51.5 MB each | 16 × 51.5 = 824 MB (A3: dropped first under pressure) |
| **entries kept / bytes** | **6 / 3.18 GB** (bench log: 6 / 3185 MB) | **16 / 1.38 GB** (0.55 GB without the end states) |
| row's first decode write | COW of 272 MB per row | in place |

**30 chat turns** (7.5K system prompt, users 30–150 tokens, answers 200–900 tokens, ≈ 650 tokens per
turn; turn 30 ≈ 26.5K tokens; one entry per conversation since N1):

| | today (phase 1) | Part A | Part A + A3 |
|---|---|---|---|
| entry KV | 0.87 GB (a view of the row's whole buffer) | ~30 pieces, 0.87 GB + dead tails (the per-turn gap is 4–157 tokens: 0.07–0.15 GB) | 0.87 GB |
| checkpoints + end | ≤ 12 × 51.5 + 51.5 = 0.67 GB | same | same |
| copy per turn at store | 0.87 GB (COW on the first decode write) | ~21 MB (`[at, L)` only) | + ~21 MB trim |
| k conversations on one system prompt | k × full KV | the 7.5K system segment (245 MB) is shared: −(k−1) × 245 MB | same |

For one growing conversation Part A saves copies, not resident bytes. Its memory wins are fan-out
(siblings, agents on a shared tool prompt) and several conversations on one system prompt. The
checkpoint budget (N4, not in this plan) dominates the per-conversation bytes.

### A.8 Tests (CPU, `tests/test_prompt_cache.py`)

Helpers:

```python
def view(caches):          # every array cut to the cache's offset (restore pre-sizes a zeros tail)
    out = []
    for c in caches:
        if hasattr(c, "cache"): out += [a for a in c.cache if a is not None]
        else: out += [x[..., :c.offset, :] for x in (*(c.keys if isinstance(c.keys, (tuple, list)) else [c.keys]),
                                                     *(c.values if isinstance(c.values, (tuple, list)) else [c.values]))]
    return out

def same(a, b):
    x, y = view(a), view(b)
    return len(x) == len(y) and all(u.shape == v.shape and mx.array_equal(u, v).item() for u, v in zip(x, y))
```

| # | test | proves |
|---|---|---|
| A-T1 | `test_entries_equal_cold_bitwise`: the reviewer's harness (`scratchpad/pc/inv_small.py`), ported. Two interleaved conversations (8 and 6 turns) + regenerate + an older prompt again. After every request, for every entry and every position `p` that `best()` can return (checkpoints, plus the end when `end` is set, ≥ the quantized floor): `same(pc.restore(e, p), cold_entry_restore(e.ids[:p] + [x], p))`. fp16 and q8 (`kv_start=40`), `CKPTS=4`, 1 seed (≤ ~40 s); the full matrix is a `-m slow` param | hit == cold for every resumable position, chains included |
| A-T2 | `test_store_shares_prefix_segments`: 4 siblings over a 40-token system prompt (`small`). One shared segment + 3 own ones; each own segment spans `[at, L)` (`k.shape[2] == b - a`); `pc.bytes == Σ unique arrays`; `e.nbytes` counts the shared segment | sharing, exact accounting |
| A-T3 | `test_restore_layout`: `restore(e, at, first=n)` gives `keys.shape[2] == at + round_up(n, 256)` and `offset == at`. After the first slab the buffer object is the same (written in place). A 3-piece chain restores bitwise equal to a 1-piece one | the "one copy, same buffer" claim |
| A-T4 | `test_eviction_frees_unshared_segments`: `weakref` to each `_Seg`. Evict the parent while a child references its piece: the parent's segment stays. Evict the child: collected (`gc.collect()`). `pc.bytes == recount` after every step | refcount eviction |
| A-T5 | `test_restore_never_views_a_segment`: restore, evict every entry, `gc.collect()`: every segment's weakref is dead while the restored row cache lives; `mx.get_active_memory()` dropped by at least the segment bytes (CPU; on Metal in the GPU smoke) | nothing pins a segment buffer |
| A-T6 | extend `test_thinning_keeps_the_last_boundaries_over_many_turns` (30 turns): one entry; `len(e.chain)` grows by ≤ 1 per turn; `pc.bytes <= KV(len) + dead-tail bound + ckpts + end`; the last turn == cold. With A3, no dead tail | the chat case |
| A-T7 | `test_quantized_child_of_fp16_parent`: parent < `kv_start`, child ≥: the child's chain does not reuse the parent's segment; A-T1 passes for both | step 2's rule |
| A-T8 | finding 1: a quantized-floor layout where the longest-LCP entry has `best()` = 0 and another has a deep checkpoint: `_fits` keeps the deep one, evicts the other, and the request hits | review 1 |
| A-T9 | finding 2: `keep` alone over budget: `_fits` → False and `pc.evictions` unchanged | review 2 |
| A-T10 | finding 3: `limit` below the new entry: LRU others evicted first; the new entry is dropped when it alone exceeds `limit`; the request's tokens are unchanged | review 3 |
| A-T11 | finding 4: a multi-slab prefill resumed from E, plus a growing row that needs evictions (`_regrow`) between its slabs: E stays, others go | review 4 |
| A-T12 | finding 5: a quantized entry holds no checkpoint below `kv_start` | review 5 |

Existing tests that change:

* `test_admission_keeps_the_entry_it_would_resume_from` (`target.nbytes` still works; the keep now
  comes from `peek`);
* `test_turns_absorb_the_previous_entry` (`pc.bytes == pc.entries[0].nbytes` holds);
* anything touching `e.cache`: grep `\.cache\b` in tests.

---

## Part B: cascade attention for rows sharing a prefix

### B.1 When a row is grouped

New knobs, next to `MAX_BATCH` (1326):

```python
CASCADE = os.environ.get("QWEN_CASCADE", "0") != "0"            # rollback switch; "1" after the B4 gate
CASCADE_MIN_ROWS = int(os.environ.get("QWEN_CASCADE_MIN_ROWS", "8"))       # 0 = off; 1..2 -> 3
CASCADE_MIN_PREFIX = int(os.environ.get("QWEN_CASCADE_MIN_PREFIX", "4096")) # 0 = off
CASCADE_CHECK = os.environ.get("QWEN_CASCADE_CHECK", "0") == "1"  # verify members' prefix bits (tests: on)
CASCADE_CHUNK = 1024          # prefix chunk (microbench: 1024 best of 512..4096); tests monkeypatch 16
CASCADE_STEP = 256            # suffix stack growth; tests monkeypatch 16
```

The scheduler enables it (`self.cascade`, in `__init__` 1839) only when all of these hold:
`CASCADE and CASCADE_MIN_ROWS and CASCADE_MIN_PREFIX and PROMPT_CACHE and max_batch >= 3 and
self.kv["kv_bits"] is None`. Engine.load logs "cascade: on (min rows R, min prefix P)" or the
reason it is off.

**A row is eligible for a group with prefix `ids[:P]`** (`_Scheduler._eligible(row, P, ids)`) when
all of these hold:

* `row.group is None`, `not row.done`, `not row.req.images` and `row.delta == 0`. Vision KV is not
  a function of the token ids.
* `len(row.req.ids) - 1 >= P`: the prefix lies inside the prompt, never in generated tokens.
* `row.req.ids[:P] == ids`. A C-level list compare, ~0.1 ms at 32K.
* Every FA cache is a plain `KVCache` (`type(c) is KVCache`) with `c.offset == row.pos`.
* `row.pos - P <= P // 2`. Its suffix is not comparable to the prefix (microbench: "do not group
  rows whose own suffix is comparable to the prefix").
* With `CASCADE_CHECK`, its FA keys/values `[0, P)` equal the group's fp16 prefix
  (`mx.array_equal`). If they do not, log a warning and leave the row out.

**Why grouping by the token prefix is sound.**

* `_bounds` is a pure function of the prefix, so every row whose prompt shares `[0, P)` with
  `P` a slab end holds the same KV bits there. This is the hit == cold invariant, checked by
  ~3,600 bitwise comparisons in the review.
* That covers rows resumed from different entries, the cold producer, and a DFlash-demoted row
  (whose prefill uses the same `_slab` cuts and `_prompt_cache`).
* `P` is the resume position `row.req.reused` of a resumed row. That is always a slab end (`best()`
  filters on `ends`).
* `CASCADE_CHECK` turns the invariant into an assertion in tests and in the validation run.

**Formation and joining** (`_Scheduler._cascade_join(row)`, called at the end of `_join` 2464, after
`self.rows.append(row)` and `_regrow`):

1. Skip if `not self.cascade` or `len(self.rows) < 3`.
2. **Existing group.** For each group, deepest `P` first: if `_eligible(row, g.P, g.ids)`, add the
   row to it (`_group_add`) and return. The row's suffix is `[g.P, pos)`, even when its own resume
   point was deeper.
3. **New group.** `P = row.req.reused`. If `P >= CASCADE_MIN_PREFIX`:
   `cands = [r for r in self.rows if _eligible(r, P, row.req.ids[:P])]`. This includes rows that
   joined earlier as ungrouped, such as the wave's cold producer. If
   `len(cands) >= max(3, CASCADE_MIN_ROWS)`, build the group from `cands` (`_group_build`).

**Decision: convert earlier rows, do not wait for an "open group".**

* In the 16-wave, the group forms when the 8th row joins. Rows 1–7 hold full 8K copies until then:
  the cold row 1 holds one from its own prefill, rows 2–7 from their restores.
* Converting them frees 7 × 268 MB (7 × 1.07 GB at 32K) and makes them group members for the rest
  of the wave.
* Without conversion, 16 × 32K stays unadmittable.
* Conversion is a numerics change for those rows from that step on: the same class as grouping
  itself, gated by B4.

**Dissolution** (`_cascade_tidy`, called at the end of `_compact` 2477 on every call, also when no
row ended):

* **All groups, when `len(self.rows) <= 2`.** This keeps the CLAUDE.md invariant "B ≤ 2 always
  bit-exact": the B=1 / B=2 path is today's code.
* **A group with 1 member left.**
* **A member whose suffix grew past `g.P // 2`** leaves its group, converted back.

The memory effect is neutral or better, since a fp32 prefix is P × 64 KiB:

* converting back ≤ 2 members costs ≤ 2 × P × 32 KiB;
* a member leaving at S = P/2 frees S × 64 KiB of suffix = P × 32 KiB.

There is no other hysteresis. A group needs ≥ 8 rows to form and dissolves only at 1 member or
B ≤ 2. It never re-forms except on a join with ≥ 8 candidates.

### B.2 Data structures (new, after `_gdn_split` 1734)

```python
class _GroupRef:
    """Stands in for a grouped row's FA KVCache in row.cache: the KV lives in the group."""
    def __init__(self, group, row): self.group, self.row = group, row
    @property
    def offset(self): return self.row.pos          # RowsKV.offset reads it; masks are None at L=1


class _PrefixGroup:
    key: tuple                  # (P, hash(tuple(ids)))
    ids: list                   # prompt ids [0, P)
    P: int; c: int; Ppad: int   # c = ceil(P / CASCADE_CHUNK); Ppad = c * CASCADE_CHUNK
    kc: dict[int, mx.array]     # FA layer i -> (Hkv, c, Pc, D) fp32 (view of a contiguous (1,Hkv,Ppad,D))
    vc: dict[int, mx.array]
    pmask: mx.array | None      # (1, c, 1, Pc) fp32: 0 for t < P, -inf for padding; None if P % Pc == 0
    sk: dict[int, mx.array]     # (n, Hkv, cap, D) fp32 suffix stack, zeros past each row's length
    sv: dict[int, mx.array]
    cap: int                    # suffix capacity, a multiple of CASCADE_STEP
    rows: list                  # members, in stack order (the refcount: dissolve at 1)
    kv_dtype: mx.Dtype          # the rows' KV dtype (fp16; fp32 on the CPU test model)
    reserved: int               # bytes charged in _Scheduler.reserved (B.5)
    nbytes: int                 # kc+vc+sk+sv actual bytes (tests: <= reserved)
```

**Decision: one stacked fp32 suffix per group per layer, not a per-row suffix KVCache.**

* The stacked layout is what the microbench measured (`b32`/`b32u`): ~20 kernels per layer
  independent of B, and one `put_along_axis` per K and V for all members. The host build fell from
  1.9 ms to 0.4 ms per step at B=16.
* A per-row suffix would cost ~5–7 kernels per row (manual fp32 scores, LSE, softmax, P·V, casts).
  That is today's per-row kernel count, and it loses the host saving.
* The price is ragged waste: every member's slot is `cap` wide. B.5 charges for it, and the
  `P // 2` leave rule bounds it.
* Storage stays fp32 (64 KiB/token). An fp16 stack would need an fp32 cast of the whole stack
  every layer and step: ~0.8 GB of traffic per step at 16 × 512, which eats a large part of the
  13–16 ms saving. Keep fp16 storage + cast as a measured option for later.

**Helpers** (pure functions over cache lists, so that `agree_cascade.py` can call them too):

```python
def _group_build(lm_fa: list[int], members: list[list], P: int, ids: list, rows: list) -> _PrefixGroup
def _group_add(g, row_cache: list, row) -> None          # concat one padded suffix onto the stack (copy)
def _group_take(g, keep: list[int]) -> None              # members kept, stack re-taken, cap trimmed
def _group_release(g, j: int) -> dict[int, KVCache]      # member j back to contiguous KVCaches
def _group_grow(g, need: int) -> None                    # cap -> round_up(need, CASCADE_STEP), zeros concat
```

All five build every new array first and assign them to `g` only at the end. A failure therefore
leaves the group and the rows as they were, the `_gdn_stack` pattern (`_join` 2464).

**Shapes in `_group_build`** (9B, `P=8192`, 16 members, all ops lazy, one `mx.eval` at the end):

* **Prefix, from `members[0]`** (decision below), per FA layer `i`:

  ```
  k = cache[i].keys[..., :P, :]                                    (1, 4, 8192, 256) fp16 view
  k = concat([k, zeros((1, 4, Ppad-P, 256), fp16)], 2) if Ppad > P
  kc[i] = k.astype(fp32).reshape(4, c, 1024, 256)                  (4, 8, 1024, 256) fp32, 33.5 MB
  ```
  The same for `vc`.
* **Suffix of member j**, `S_j = pos_j - P`:

  ```
  cache[i].keys[..., P:pos_j, :].astype(fp32)                      (1, 4, S_j, 256)
  ```
  padded with zeros to `cap = round_up(max_j S_j + 1, CASCADE_STEP)`, then
  `sk[i] = concat over j` → `(n, 4, cap, 256)` fp32. The zeros matter (pitfall: unused slots must
  be finite).
* **Member caches.** Set `row.cache[i] = _GroupRef(g, row)` for every FA layer `i`, and
  `row.group = g`. The rows' old KV buffers are freed when the in-flight step that reads them
  completes.
* **Prefix source: the forming row's own cache.**
  * Its `[0, P)` bits equal the entry's segments: the row's cache was restored from them, or, for
    the producer, the segments were copied from it.
  * It is contiguous and always alive. The entry may have been evicted between the row's prefill
    and the formation, and its chain would need a concatenation anyway.
  * Either way the copy is built **once per group, from one source**. `CASCADE_CHECK` verifies the
    other members against it.
* **`_group_release(g, j)`**, per FA layer: `keys = concat([kc[i].reshape(1,4,Ppad,256)[..., :P, :].astype(kv_dtype),
  sk[i][j:j+1, :, :S_j, :].astype(kv_dtype), zeros((1,4,CASCADE_STEP,256), kv_dtype)], 2)`,
  `offset = P + S_j`.
  * fp16 → fp32 → fp16 is exact, so the prefix bits are those of the entry and of a never-grouped
    row.
  * The suffix bits are what grouped decode wrote.
  * The buffer capacity does not affect bits [CPU-verified above].

### B.3 `_rows_attention` and `RowsKV` (1338–1378)

```python
class RowsKV:
    def __init__(self, caches, layer=None, plan=None):
        self.caches, self.layer, self.plan = caches, layer, plan


def _rows_attention(self, x, mask=None, cache=None, position_ids=None, position_embeddings=None, **kw):
    ...                                                  # unchanged up to _prepare_projected_qkv
    plan = cache.plan
    if plan is None:                                     # no group in this step: today's loop, as is
        outs = [... per-row update_and_fetch + _Q35.scaled_dot_product_attention ...]
        out = mx.concatenate(outs, axis=0)
    else:
        parts = []
        for b in plan.single:                            # ungrouped rows: the identical per-row calls
            c = cache.caches[b]
            k, v = c.update_and_fetch(keys[b:b + 1], values[b:b + 1])
            parts.append(_Q35.scaled_dot_product_attention(queries[b:b + 1], k, v, cache=c,
                                                           scale=self.scale, mask=None))
        i = cache.layer
        for gs in plan.groups:
            g = gs.group
            g.sk[i] = mx.put_along_axis(g.sk[i], gs.upos, keys[gs.idx].astype(mx.float32), axis=2)
            g.sv[i] = mx.put_along_axis(g.sv[i], gs.upos, values[gs.idx].astype(mx.float32), axis=2)
            parts.append(_cascade_fn(self.scale, g.pmask is not None)(
                queries[gs.idx], g.kc[i], g.vc[i], *((g.pmask,) if g.pmask is not None else ()),
                g.sk[i], g.sv[i], gs.smask))
        out = mx.concatenate(parts, axis=0)[plan.inv]   # back to batch row order: one take
    out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(out * mx.sigmoid(gate))
```

* The **per-step plan** is built once per `_step`, in `_Scheduler._plan(rows)`, and shared by all
  8 FA layers. `_step` passes `RowsKV([r.cache[i] for r in rows], i, plan)` (2541).

  ```python
  class _StepPlan(NamedTuple): single: list[int]; groups: list[_GroupStep]; inv: mx.array
  class _GroupStep(NamedTuple): group: _PrefixGroup; idx: mx.array; upos: mx.array; smask: mx.array
  ```
  * `idx`: (B_g,) int32, the members' positions in `rows`. A missing member is a bookkeeping bug:
    raise RuntimeError, like the GDN stacking check at 2540.
  * `lens = [r.pos - g.P for r in g.rows]`: the write index of this step's token. `row.pos` is
    incremented after the forward is built (2550).
  * `_group_grow(g, max(lens) + 1)` when needed.
  * `upos = broadcast_to(array(lens)[:, None, None, None], (B_g, Hkv, 1, D))` int32.
  * `smask = where(arange(cap)[None] < array(lens)[:, None] + 1, 0, -inf).astype(fp32).reshape(B_g, 1, 1, cap)`:
    the just-written token is attended.
  * `inv = argsort(single + Σ idx)` as an int32 mx.array.
  * The plan is None when `self.groups` is empty: zero cost and zero change.
* **`_cascade(q, kc, vc, pmask, sk, sv, smask, scale)`** ports `cascade(..., dt=float32)` from
  `microbench_cascade.py` (lines 109–137), generic in (Hq, Hkv, D). Shapes for the 9B at
  `B_g=16, P=8192, cap=512`:

  ```
  q              (16, 16, 1, 256) fp16      gathered queries after RoPE
  qg = (q.astype(f32) * scale).reshape(16, 4, 4, 256)       head h -> kv h // 4 (contiguous repeats)
  qp = qg.transpose(1,0,2,3).reshape(4, 1, 64, 256)         row index b*4 + rep
  s  = qp @ kc.swapaxes(-1,-2) [+ pmask]   (4, 8, 64, 1024) fp32   8.4 MB (33.5 MB at 32K)
  lse_c = logsumexp(s, -1, keepdims)       (4, 8, 64, 1)
  o_c = softmax(s, precise=True) @ vc      (4, 8, 64, 256)
  lse_p = logsumexp(lse_c, 1, keepdims)    (4, 1, 64, 1)
  o_p = sum(exp(lse_c - lse_p) * o_c, 1)   (4, 1, 64, 256)
  ss = qg @ sk.swapaxes(-1,-2) + smask     (16, 4, 4, 512) fp32
  lse_s, o_s = ... same two ops ...         (16, 4, 4, 1), (16, 4, 4, 256) -> prefix layout (4, 1, 64, ·)
  m = max(lse_p, lse_s); wp, ws = exp(lse_p - m), exp(lse_s - m)
  out = (wp*o_p + ws*o_s) / (wp + ws)      (4, 1, 64, 256) fp32
  -> reshape(4, 16, 4, 256).transpose(1,0,2,3).reshape(16, 16, 1, 256).astype(q.dtype)
  ```
* **`_cascade_fn(scale, masked)`** returns `mx.compile(functools.partial(_cascade, ...))`, cached
  in a module dict. It retraces when B_g, c or cap changes: on joins and leaves, and every
  `CASCADE_STEP` steps. That is expected (microbench pitfall).
* **What stays byte-identical in a step with groups:**
  * every ungrouped row, because its calls are identical and concat + take are exact copies;
  * the projections, norms and RoPE, batched over all B rows with mrqmm as today;
  * GDN layers;
  * sampling.

  Test B-T4 asserts that the ungrouped rows' logits are bitwise equal with and without a group in
  the same batch.
* **Donation.** Each `g.sk[i]` and `g.sv[i]` is updated exactly once per step and referenced only
  by the group dict. The previous step's attention read of it is finished before this step's
  `put_along_axis` runs, because this step depends on the previous step's token. So the update is
  in place, like `KVCache`'s slice update today. If a profile ever shows a full stack copy per step
  (+8 ms in microbench run 1), look for an extra live reference.

### B.4 Group lifecycle in the scheduler

| hook (current line) | change |
|---|---|
| `__init__` 1839 | `self.groups = []`; `self.cascade = ...` (B.1) |
| `_join` 2464 | after `_regrow(row)`: `self._cascade_join(row)` in try/except. On error, log it and leave the row ungrouped: the helpers are atomic |
| `_compact` 2477 | after the row filter: each group drops its finished members (`_group_take`), then `self._cascade_tidy()` (B.1 dissolution), which also runs when no row ended. Dissolution runs before the B=1 step: `_gdn_split` (2484) only touches GDN layers, and `_cascade_tidy` gives the last row its KVCaches back |
| `_step` 2526 | `plan = self._plan(rows)` if `self.groups`; `RowsKV(..., i, plan)`. Defensive: at `B == 1`, a grouped row is a bug: raise |
| `_end` 2489 | unchanged (`self.reserved -= row.reserved` = the member part); the group charge is recomputed by `_compact` → `_cascade_tidy` |
| `_fail_rows` 2514 | `self.groups = []` (the arrays go with the rows) |
| `_close` 1946 | `self.groups = []` |
| `_log` 2656 | add `groups=%d grouped=%d` |
| tools | `_Scheduler.set_cascade(on)`: off dissolves every group now; on runs a regroup pass over the running rows (the same formation rule, keyed by each row's `req.reused`). For step_prof's A/B |

**The 16-wave, step by step** (`prompt_cache_bench` A1):

1. Row 1 (cold, reused 0) joins at B=1. No group (P = 0).
2. Rows 2–7 (reused 8192) join. Fewer than 8 candidates.
3. Row 8 joins. The 8 candidates are rows 1–8; row 1 shares `ids[:8192]` and its suffix of ~144
   tokens is ≤ 4096. The group forms:
   * an fp32 prefix of 537 MB from row 8's cache;
   * 8 suffix copies of ~150 tokens;
   * 8 × 268 MB of row buffers freed once the in-flight step lands.
4. Rows 9–16 join the group directly (`_group_add`, a stack concat).
5. The 16 rows decode as one group (`plan.single == []`).
6. As rows finish: `_group_take`. At B=2 the group dissolves, and the last 2 rows get contiguous
   caches back.

**Transient at formation.** The fp32 prefix (2 × P × 32 KiB: 0.54 GB at 8K, 2.1 GB at 32K) exists
before the members' buffers are released. Formation runs at the end of a prefill tick: that slab's
activations are freed (`mx.clear_cache` at 2263), so the prefill reserve is idle at that moment.
The peak stays within cap. Note it in `Usage.peak_gb`'s docstring: the process peak now includes
it.

### B.5 Accounting (`_bytes` 2005, `_regrow` 2022, `_reserve`)

New helpers:

```python
def _member_bytes(self):                       # a grouped row's own part: GDN state only
    return 2 * self.gdn_row + (8 << 20)

def _group_bytes(self, g, tokens=None):
    """Prefix once (fp32) + the stack charged at the max member reservation
    (every slot is `cap` wide) + the step transient. `tokens`: {row: tokens}
    overrides, for _regrow's what-if."""
    kv32 = 2 * self.kv_fp16                                    # bytes/token, all FA layers, fp32
    s_res = max((tokens or {}).get(r, r.tokens) - g.P for r in g.rows)
    s_res = -(-s_res // CASCADE_STEP) * CASCADE_STEP
    n_fa = len(g.kc)
    transient = n_fa * 128 * len(g.rows) * (g.Ppad + s_res) + (64 << 20)
    return g.Ppad * kv32 + int(len(g.rows) * s_res * kv32 * 1.05) + transient
```

* **The transient term** is the per-layer score and probability buffers: `s` and `p`, each
  `(Hkv, c, rep·B_g, Pc)` fp32 = 64·B_g·Ppad bytes for the 9B layout (4 × 4 heads × 4 bytes), plus
  the same for the suffix scores. Microbench check:
  * 8K/B=16: 8 × 128 × 16 × 8192 + 64 MB = 198 MB, measured 0.17–0.21 GB;
  * 32K/B=16: 601 MB, measured 0.36–0.61 GB.

  For other head layouts the factor is `4 × Hq` bytes per (row, token) per layer; compute it from
  `lm.args`, not as a literal 128.
* **Formation**: `self.reserved += g.reserved + Σ_m (member_bytes − m.reserved)` with
  `m.reserved = member_bytes`. **Release / dissolve**: the reverse, with `row.reserved = self._bytes(row.tokens)`.
  This is a re-accounting, never refused (as `_regrow`'s overshoot).
* **`_compact` → `_cascade_tidy`** recomputes `g.reserved` after `_group_take`.
* **`_regrow(row)`** (finding 4 change from A.5 kept):
  ```python
  tokens = min(row.pos + 1024, len(row.req.ids) + row.req.max_tokens + 1)
  if row.group is None: delta = self._bytes(tokens) - row.reserved
  else:                 delta = self._group_bytes(row.group, {row: tokens}) - row.group.reserved
  rows = self.reserved + delta          # then the same headroom / cache-eviction / "ended at" logic
  ...
  row.tokens = tokens; (row.group.reserved if row.group else row.reserved) += delta; self.reserved += delta
  ```
  * The "ended at … tokens" path is unchanged. `_finish(row, "length")` ends a grouped row like
    any other, and `_compact` removes it from the group.
* **Admission (`_fits`) is unchanged.** A request that will join a group still reserves its full
  prompt: its prefill holds a contiguous `[0, len)` cache until `_join`. The re-charge at
  formation or `_group_add` returns the difference.
* **Numbers.** `prompt_cache_bench` A1 shape: 16 × (8.34K prompt + 256 tokens), P = 8192.

  | | today (reserved / real) | cascade (reserved / real) |
  |---|---|---|
  | 16 × 8K, 256 tokens | 16 × 0.40 GiB = 6.4 GiB / 4.5 GB KV | prefix 0.54 GB + stack 16 × 512 × 64 KiB = 0.54 GB + transient 0.2 GB → **1.3 GB** / 1.1 GB KV |
  | 16 × 32K, 128 tokens | 16 × 1.22 GB = 19.5 GB → **only ~12 rows fit** in the 15.75 GiB budget | 2.15 + 0.54 + 0.6 GB → **3.3 GB**; all 16 rows admitted (each needs its full 1.22 GB only while it prefills) |
  | 16 rows, 4096 reserved generation tokens each, 8K | 16 × 0.53 GiB | 0.54 + 16 × 4.2K × 64 KiB × 1.05 = 4.5 GB + 0.2 |

  * The fp32 stack doubles the per-token cost of *generated* tokens. That is the price of `b32`.
  * The 0.54 GB fp32 prefix per 8K group (2.1 GB at 32K) is the price of the speed-up.
  * An fp16 prefix was rejected by the microbench (`b16`: 4–10× the error on peaked scores).
    Bringing it back needs an fp32-accumulating Metal kernel (later).

### B.6 Interactions

* **DFlash (`_spec` 2280)** is never grouped while speculative: the row is not in `self.rows`. A
  demoted row joins through `_join` and is eligible like any other: its KV has the same cuts, and
  `CASCADE_CHECK` covers it.
* **Vision rows** are never cached (`use_cache` at 2134), and B.1 excludes them by images / delta.
* **Grammar and sync-mode rows** are fine: `_step` builds the same plan in sync mode. Thinking
  budget forced tokens change only `r.y`.
* **Quantized KV (27B q8, 35B q4)**: no groups in this phase (`kv_bits is None` gate).
  * A quantized prefix would need dequantizing into the fp32 copy. That is easy, but
    `self.quantize(row.cache)` (2586) would also have to convert a group's fp32 stack at
    `quantized_kv_start`, a new code path with its own exactness question.
  * The payoff is also smaller there: 27B batching gains ×1.4–1.9, and the 35B MoE is dominated by
    expert reads.
* **`QWEN_MAX_BATCH=1`** (or < 3): `self.cascade` is False. **Prompt cache off**: no row has
  `reused > 0`, so no key, so no group, and `self.cascade` is False anyway.
* **mrqmm, pfqmm, GDN fuse** are untouched. Projections stay batched over all B rows; prefill never
  sees groups.
* **Prompt-cache store** runs in `snapshot` before `_join`, while the row's cache is still
  contiguous. Part A segments never alias group arrays: the group copies, and the entry copies.
* **`Usage.peak_gb`**: no code change. It now includes the group's fp32 prefix, the formation
  transient and the step transient.
* **Determinism (CLAUDE.md caveat to add).** At B ≥ 3, whether a row is grouped depends on the
  prompt cache's contents and on arrival order, so a greedy answer can depend on them. Restore
  determinism with `QWEN_MAX_BATCH=1` or `QWEN_CASCADE=0`.

### B.7 Exactness gate (GPU, B4)

* **Tool**: port `scratchpad/gpu_tests/agree.py` (it is not in the repo) to
  `scripts/perf/prof/agree_cascade.py`. Load the Engine like `agree.py`.
* **Prompts**: one shared system prompt (8K and 32K filler, as `bench_concurrency.py --prefix
  shared`) plus B distinct user tasks.
* **Copies**: each prompt is prefilled cold with the worker's `_bounds` cuts three times:
  * a solo copy;
  * a batch copy grouped with `_group_build` at `P` = the deepest common cut;
  * a control batch copy with no group.
* **Run**: teacher-forced for 128 steps on the solo greedy stream, at B ∈ {3, 8, 16} (16 only at 8K).
* **Report**: top-1 agreement and KL mean / p99 / max for cascade-vs-solo, control-vs-solo and
  cascade-vs-control.
* **Pass** (all at 8K and 32K):
  * cascade-vs-solo top-1 **≥ 99.7 % at B=16 and B=8**;
  * KL mean **≤ 1.5e-5** (the mrqmm baseline is 9.7e-6–1.1e-5);
  * cascade-vs-control KL mean ≤ control-vs-solo KL mean, i.e. grouping adds no more noise than
    mrqmm did.
* **Per-layer criterion** (the microbench's): capture one step's per-FA-layer attention output.
  * For the grouped rows, the rms relative error vs an fp32 reference (per-row
    `mx.fast.scaled_dot_product_attention` on fp32 copies of q/K/V) must be ≤ 2.3e-4.
  * It must also be ≤ the per-row fp16 kernel's own error in ≥ 95 % of (layer, step) pairs.
    Microbench values: `b32c` 2.03–2.12e-4, `a` 2.70–2.95e-4.
* **GPU unit test**: `tests/test_cascade_gpu.py`, skipped without Metal, like `test_mrqmm.py`. It
  checks `_cascade` at 9B shapes (B 16, P 8192 and 8150 (padded), S 64/512): the rms criterion, a
  finite output, and the GQA order against per-row SDPA.
* **Corpus**: `wtest.py` 15/15 byte-identical at `QWEN_MAX_BATCH=1` and with `QWEN_CASCADE=0`
  (expected trivially).

### B.8 Tests (CPU, tiny models; `tests/test_cascade.py`)

All tests run with `small`-like monkeypatches: `CASCADE=True, CASCADE_MIN_ROWS=3, CASCADE_MIN_PREFIX=16,
CASCADE_CHUNK=16, CASCADE_STEP=16, CASCADE_CHECK=True`, `PROMPT_CACHE=True, PREFILL_STEP=8,
PREFILL_MIN_STEP=4`. The tiny model is Hq 4, Hkv 2, D 64 (rep 2), which exercises the generic GQA
code.

| # | test | proves |
|---|---|---|
| B-T1 | `test_cascade_matches_rows` (function level): (Hq,Hkv,D) ∈ {(4,2,64),(16,4,256)}; P ∈ {48 (= 3 chunks), 40 and 70 (padded)}; B ∈ {3,5}; flat and peaked q. vs fp32 per-row SDPA: rms rel ≤ 3e-4 and ≤ the fp16 per-row kernel's own error; max \|cascade − per-row fp16\| ≤ 4e-3 | the formulation, padding+mask, GQA layout (random per-head data: a wrong order fails) |
| B-T2 | `test_cascade_unused_slots`: suffix slots past each length filled with 1e4 vs zeros → bitwise equal outputs; padded prefix slots likewise | the mask makes unused slots weightless |
| B-T3 | `test_grouped_rows_close_to_solo` (test_rows.py style, logits level): `dense` fp32 (tol 1e-4) and `dense_fused` (2e-2). 5 rows share a 40-token prefix (P=32, and P=36 padded), built with `_group_build`; 6 steps with the plan vs solo B=1 decode; top-1 agree on every step; then `_group_release` → KV `[0,P)` bitwise equal to the original prefix, `[P,pos)` equal to the stack's fp16 cast | B2 plumbing, release exactness |
| B-T4 | `test_ungrouped_rows_unchanged_by_groups`: the same 6 rows, same B. Run 1 has rows 0–2 grouped, run 2 has no plan: the logits of rows 3–5 are `mx.array_equal` | today's path untouched for ungrouped rows |
| B-T5 | `test_cascade_wave` (scheduler, `dense`): 6 requests on a 40-token system prompt arriving one per tick. The group forms at the 3rd join and converts the cold row (spy `_group_build`: 3 members incl. reused==0); later rows `_group_add`; tokens == solo (`serial`); at the end `sched.groups == []`, `sched.reserved == 0` | formation, conversion in, joining, dissolve on finish |
| B-T6 | `test_cascade_off_is_today`: `QWEN_CASCADE=0`, `MIN_ROWS=0`, `MIN_PREFIX=0` (3 runs): `_plan` never builds, `_cascade` never called, tokens equal | rollback switch |
| B-T7 | `test_cascade_dissolves_at_two_rows`: 4 grouped rows with max_tokens 6/8/30/30. When 2 remain, `groups == []` and every later step takes the per-row path (spy). The last two finish with the right lengths | B ≤ 2 invariant, conversion out mid-generation |
| B-T8 | `test_cascade_member_leaves_on_long_suffix`: one member's max_tokens drives its suffix past P//2: it is released, the others stay grouped, tokens continue | leave rule |
| B-T9 | `test_cascade_accounting`: after formation, `reserved == Σ member_bytes + g.reserved` and `g.nbytes <= g.reserved`. A member's `_regrow` raises `g.reserved` by the formula. After every end, `reserved == 0` | B.5 |
| B-T10 | `test_cascade_suffix_growth`: generation passes `cap` twice (`CASCADE_STEP=16`): the stack grows, tokens equal | `_group_grow`, plan rebuild |
| B-T11 | `test_cascade_excludes_vision_quantized_dflash`: a vision row with the same leading tokens stays ungrouped; `kv_bits=8` → `sched.cascade is False`; a DFlash row is not grouped while speculative (FakeDraft from test_scheduler) | B.6 |
| B-T12 | `test_cascade_failed_step_drops_groups`: monkeypatch `_cascade` to raise once → the rows fail, `groups == []`, a following request runs | failure path |
| B-T13 | `test_cascade_check_rejects_mismatch`: corrupt one candidate's `[0,P)` key → that row is not grouped (warning logged), the others are | `CASCADE_CHECK` |

### B.9 Benchmark plan (GPU lease, `setopt NO_BG_NICE`, nothing else on the GPU)

1. **In-process A/B** (`scripts/perf/prof/step_prof.py`):
   * add `--ctx 8k-shared` / `32k-shared`: B prompts on one system prompt, `QWEN_PROMPT_CACHE=1`;
   * add a variant pair `cascade1,cascade0` that toggles `sched.set_cascade()`;
   * `--ab cascade1,cascade0 --ab-steps 16`, dropping 2 steps after each swap. A swap converts all
     rows: ~15 ms at 16 × 8K.
   * Runs: B = 8 and 16 at 8K; B = 8 at 32K. 16 × 32K does not fit with the cascade off, so report
     cascade-on alone and compare with the derived ~110 ms.
   * **Expect**: 8K/B16 step 67–80 → 52–66 ms; 8K/B8 −3–4 ms; 32K/B8 −16–20 ms (×1.35–1.5).
2. **End to end** (the `run_pcache.zsh` phases, worker restarted per variant, interleaved
   on/off/on/off, `--probe 64 --cooldown 30`):
   * **A1** (16 × 8K shared, c=16, 256 tokens): decode aggregate **239 → ~290–310 tok/s**, peak
     17.7 → ~14.5–15 GB, TTFT unchanged (±drift).
   * **A5** (8 × 32K, 128 tokens): 107 → ~145–160 tok/s.
   * **New A6** (16 × 32K shared, 128 tokens):
     * cascade off: count the rows admitted at once (expect ~12; log `rows=` / `waiting=`);
     * cascade on: 16 rows, ~×1.7 per step vs a derived B=16 32K step.
   * **Regression levels**: A2 (16 distinct 8K) and the final_bench 1K c=16 level must be unchanged
     (no groups form).
3. Record the worker's `groups=` log fields, `Usage.peak_gb`, and group formation and dissolution
   lines. Report in `docs/perf/<date>/cascade_bench.md`.

### B.10 Risks and rollback

| risk | mitigation |
|---|---|
| Numerics at B ≥ 3 (the gate fails, or long-suffix rows drift) | B.7 gate before the default flips. `QWEN_CASCADE=0` restores today's code path exactly (`_plan` returns None, `_cascade_join` returns at once) |
| GQA or row-order bug (plausible-looking output) | B-T1 and B-T3 compare against per-row SDPA with random per-head data; B-T4 checks the order of ungrouped rows; `CASCADE_CHECK` in the validation run |
| Suffix grows (long answers) → the suffix part is slower than today (microbench S=2048 at P=1K is a loss) | `P // 2` leave rule; the gain at 8K/B16/S=2048 is still −11 ms. Later: a suffix kernel with LSE (`sdpa_vector` via `metal_kernel`) or the sink trick (`b3`) |
| Ragged stack waste | charged at the max member reservation (B.5) and trimmed on every `_group_take` |
| `mx.compile` retrace per (B_g, c, cap); its trace cache grows over a long uptime | traces are small. If host time shows it, compile only the merge or bucket cap. Measure the retrace cost once in B4 |
| Donation lost → full stack copy per step | B.3 rule (one update per layer per step, no extra references). The step_prof graph dump (`--dot`) shows SliceUpdate vs a Copy |
| Formation transient +2·P·32 KiB (2.1 GB at 32K) | the prefill reserve is idle at join time (B.4). If the peak is ever tight, build the prefix, `mx.eval`, then release members one by one |
| Step transient (0.6 GB at 32K × 16) | inside `g.reserved` (B.5) |
| Cache contents change B ≥ 3 greedy answers | documented (B.6). `QWEN_MAX_BATCH=1` / `QWEN_CASCADE=0` |
| Scheduler bookkeeping (membership, `_GroupRef` in row.cache) | atomic helpers; RuntimeError on an inconsistent plan (fails the step's rows like any step error; 3 strikes close the scheduler as today); B-T12 |

---

## Implementation steps (each ≤ ~250K tokens of agent work; land and test in order)

| step | scope | lines (worker / tests) | its tests prove |
|---|---|---|---|
| **A1** | `_Seg`, `_Inherit`, new `_Entry`, `_layout`, `_PrefixCache.peek/restore(first)/store(inherit, limit)/bytes_of/evict(keep=())`; `_prompt_cache` (first slab, inherit, `row.parent`); `snapshot` passes `limit` | ~140 / ~160 | A-T1..A-T7: byte-identical entries vs cold, sharing, layout, refcounted eviction, no pinned views, the 30-turn chain; all 83 existing tests pass |
| **A2** | findings 1–5: `_Req.bounds` + `_ends`, `_fits` (peek keep, floor, live pre-check), store `limit` (wired in A1, policy tested here), `_regrow` protect, quantized sub-floor checkpoints | ~40 / ~120 | A-T8..A-T12 |
| A3 (optional) | trim-on-absorb, end-state drop under pressure | ~40 / ~60 | A-T6 with no dead tail; eviction order end states → entries |
| **B1** | knobs; `_cascade` + `_cascade_fn` (compile cache); `tests/test_cascade_gpu.py` (Metal, skipped on CPU) | ~80 / ~120 | B-T1, B-T2 on CPU; the GPU rms criterion when run under the lease |
| **B2** | `_GroupRef`, `_PrefixGroup`, `_group_build/_add/_take/_release/_grow`; `RowsKV(layer, plan)`; `_StepPlan` + `_rows_attention` branch | ~150 / ~120 | B-T3, B-T4: logits-level agreement, release exactness, ungrouped rows bit-identical |
| **B3** | scheduler: `_cascade_join`, `_cascade_tidy`, `_plan`, `_eligible`, accounting (`_member_bytes`, `_group_bytes`, `_regrow` branch), `_fail_rows`/`_close`, `set_cascade`, logs, Engine.load line; default `QWEN_CASCADE=0`; docs (CLAUDE.md env table + batching invariants + determinism caveat, README env) | ~170 / ~200 | B-T5..B-T13; full CPU suite |
| **B4** | GPU: `scripts/perf/prof/agree_cascade.py`, step_prof `--ctx *-shared` + `cascade1,cascade0`, the B.9 bench; gate B.7. On pass: default `QWEN_CASCADE=1`, CLAUDE.md perf notes | ~250 scripts / — | the exactness gate and the gains |

Order: A1 → A2 → (A3) → B1 → B2 → B3 → B4. Part A can ship alone. B1 and B2 change nothing at
run time (no caller builds a plan until B3). B3 ships with the switch off. Only B4 flips it.

---

## Open points and blockers

1. **No blocker for Part A.**
   * The restore-layout equality and "contiguous copies" were checked on the CPU only.
   * A-T5 re-checks the second on Metal in the GPU smoke.
   * The first is covered on the GPU by the existing hit == cold corpus check: a hit's first-token
     logprob bitwise equals the cold one's.
2. **`agree.py` is not in the repo** (session scratchpad, `gpu_tests/agree.py`). B4 must port it.
   It also depends on `wtest.py`'s task list.
3. **Brief deviations** (B.2 stacked suffix, B.2 prefix source, B.5 group-level charge) are design
   choices, not blockers. If the per-row suffix KVCache is preferred anyway, B2/B3 get simpler
   (no `_group_add`/`_take` stack copies, no ragged waste), at ~5–7 more kernels per grouped row
   and per layer. Measure it as `b32r` in the microbench first.
4. **16 × 32K cannot be A/B'd in one process.** Cascade-off does not fit. Compare it with the
   derived step and with A6's admission count.
5. **The unaligned-prefix choice.**
   * Padding + mask (recommended): +0.1 ms per step at 8K/S64, up to +2.3 ms at 32K/S64 (`b32m`).
     It costs 64 MiB of fp32 padding per group at most.
   * Rounding P down to a 1024 multiple puts ≤ 1023 tokens into *every* member's fp32 suffix:
     64 MiB per row (1 GiB at B=16), plus ~+2.5–3 ms of suffix GEMM at B=16. The suffix is the
     weak, M=4-padded part.
   * So padding wins on memory always, and on time at 8K.
   * If the 32K mask cost shows in B4, the cheaper unmeasured variant is a separate unpadded tail
     chunk (its own partial + LSE in the merge, ~4 extra kernels).
