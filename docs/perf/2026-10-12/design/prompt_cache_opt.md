# Prompt cache under continuous batching: what to optimize

Date 2026-10-12. Read-only analysis, CPU only (nothing ran on the GPU). Line numbers refer to
`src/worker/worker.py` at b5cccb7, and to mlx-vlm 0.7.6 / mlx 0.32.3 in `src/worker/.venv`.
Model numbers are for the 9B (Ornith-1.5, fp16 KV): 8 full-attention layers × 4 KV heads × head_dim
256 → **32 KiB/token**, so an 8K prefix is **268 MB** of KV. One GDN state (24 layers × fp32
32×128×128 + conv) is **51.5 MB**.

## Summary

Today the prompt cache saves prefill and nothing else. Every hit gets a private copy of the
prefix. `restore` (759–762) hands out slices, and the row's first `update_and_fetch` concatenates
them into a new buffer (`cache.py` 513–528). Every stored sibling entry is then another full copy,
because it is a view of the hit row's own buffer (`store` → `_copy_cache(cache)`, 779). On top of
that, each entry is charged again for the checkpoints it shares with its parent (728, 765–768).

In the benchmark (16 × one shared 8K system prompt), this means:

* All 16 TTFTs land at about T_cold + 15·t_hit ≈ 17.2 + 15 × (0.15–0.3) ≈ **20–21 s**. Compare
  ~k × 17 s with 16 distinct prefixes. This is within ~10 % of the floor.
* Decode tok/s and row memory are **the same as with distinct prefixes**. Every row reads its own
  268 MB of prefix KV every step (16 × 268 MB = 4.3 GB per step at B=16). Per step_profile.md this
  is ~24 ms of a ~89 ms step.
* The cache keeps at most ~6 of the 16 entries. Each is charged ~523 MB against the 3 GiB budget
  but really holds ~317 MB.

The one lever that moves decode is **grouped shared-prefix ("cascade") attention**:

* Rows that resumed from the same prefix keep a two-segment KV: one shared, read-only prefix per
  group plus their own small suffix.
* Attention computes all of a group's queries against the prefix in one GEMM (the prefix is read
  once) and merges that with the per-row suffix part through an explicit max/sum.
* The same change gives the memory sharing (rows' KV 4.3 GB → ~0.7 GB). Estimated gain: **+25–30 %
  decode tok/s at 8K, roughly ×1.5–2 at 32K**.
* The cost: rows in a group at B ≥ 3 stop being bitwise equal to a lone row. This is the same
  class of deviation the mrqmm kernel already introduced at B ≥ 3. B ≤ 2 stays exact by
  materializing.

A cheap, byte-identical prerequisite is a **segment-sharing store**: immutable KV segments shared
between entries, with unique-bytes accounting. It holds all 16 siblings in ~1.35 GB (0.5 GB without
end states) instead of 6 entries in 3 GB. It changes nothing measurable in this benchmark beyond
memory.

The remaining items are small fixes:

* checkpoint thinning drops the system-prompt checkpoint first (a bug for prompts over 24K);
* elastic cache budget;
* the quantized-KV floor on the 27B/35B.

Coalescing concurrent cold misses is not worth it.

## Ranked proposals

Gains are for the parallel benchmark: 16 concurrent requests, 8K shared prefix, 9B fp16 KV, B=16,
short answers.

| # | proposal | TTFT | memory | decode tok/s | code / risk | bit-exactness |
|---|---|---|---|---|---|---|
| 1 | **Grouped shared-prefix decode attention** (two-segment rows, one group prefix, fp32 cascade with explicit max/sum merge) | = | rows' KV 4.3 GB → ~0.7 GB (fp32 prefix 0.54 GB + 16 suffixes); 16 × 32K becomes admittable (today ~11 rows fit) | **+25–30 % at 8K** (attention ~24 → ~5 ms of an ~89 ms step); ~×1.5–2 at 32K; ≤ +6 % at 1K | ~300 lines worker + ~200 tests; medium-high (scheduler transitions, accounting, numerics) | B ≤ 2: unchanged (materialized). B ≥ 3 grouped rows: attention summation order changes (fp32), expected ≥ 99.7 % top-1 / KL ~1e-5 like mrqmm. Gate on `agree.py`; knob to turn it off |
| 2 | **Segment-sharing store** (entries = chains of immutable KV segments + shared checkpoints; unique-bytes accounting; leaf end-states dropped first) | = (hits already happen) | cache: all 16 siblings in ~1.35 GB (0.53 GB without end states) vs 6 entries / 3.1 GB charged / 2.1 GB real | = | ~120 lines + ~100 tests; low | **byte-identical** (same KV bits and GDN states; restore is still one copy) |
| 3 | **Elastic cache budget + suffix-only reservations** (cache bytes counted in `_fits`, LRU-evicted on demand instead of a static 3 GiB cut in `_batch_budget`; grouped rows reserve the suffix only, the group reserves the prefix once) | = here (no admission pressure at 16 × 8K) | +3 GiB batch budget when the cache is idle or useless (distinct prompts) | = | ~60 lines; low | none (admission timing only) |
| 4 | **Thinning keeps message-boundary checkpoints** (bug fix, 770–773) | = here (4 checkpoints < 12); for prompts > 24K, a new conversation hits at the system boundary (~0.2 s) instead of up to 2047 tokens below it (≤ ~3.6 s) | = | = | ~10 lines; low | byte-identical |
| 5 | **Quantized-KV floor** (27B/35B: do not let the "covered" rule drop the last entry that can resume below `quantized_kv_start`; optionally keep the fp16 head segment) | n/a for the 9B; 27B/35B system prompts < 5000 tokens keep hitting | +~100–320 MB per kept fp16 head | = | ~30 lines; low | byte-identical |
| 6 | Batched (packed) suffix prefill for a burst of hits | −1–2 s for the 16th request (~5–10 %), less on the mean | = | = | high (varlen GDN, same as the "V2 piggyback") | needs a bitwise check of packed vs solo slabs |
| 7 | Partial entries from an in-flight cold prefill | 0 (FIFO + serialized prefill; the prefix is computed once anyway); helps only when the producer is cancelled mid-prefill | = | = | medium | byte-identical |
| 8 | Faster lookup host code (numpy LCP, closed-form `_bounds`) | −2–5 ms per admission at 8K, −50–100 ms at 128K with 6 entries | = | = | low | byte-identical |
| 9 | Remove COW / growth copies | 0 | transient −268 MB per row | ~+0.1 % (0.1 ms/step) | comes free with 1/2 | — |

Items 1 + 2 + 3 form one coherent design (§D1–D3). Items 4 and 5 are independent bug fixes. Items
6–9 are not worth doing now.

---

## 1. Memory: what the cache holds today, and a shared-segment store

### 1.1 Today, line by line

* **Restore makes a private copy.** `restore` (759–762) returns `_copy_cache(..., upto=at)`, which
  slices `c.keys[..., :at, :]` (680–683). The slab loop's first `update_and_fetch` then finds
  `prev + n > keys.shape[2]`, because the slice is exactly `at` long. It concatenates
  `[slice, zeros(ceil(n/256)·256)]` (`cache.py` 513–522) into a fresh buffer. Each hit row
  therefore owns a full 268 MB copy of the prefix. That is right for today's attention, which needs
  one contiguous K/V per row.
* **Storing makes another copy.** At the end of the prefill, `store(ids[:-1], cache, ...)` (2052 →
  764) builds `_Entry(ids, _copy_cache(cache), ckpts)` (779). These are views of the row's live
  buffer: the private copy, prefix + suffix.
  * The entry now pins that buffer, so `is_donatable()` is false (`mlx/include/mlx/array.h` 304:
    `array_desc_.use_count()==1 && data.use_count()==1`).
  * The row's first decode write (`self.keys[..., prev:offset, :] = keys`, `cache.py` 528) is
    therefore a SliceUpdate that **copies the whole buffer** (copy-on-write).
  * Result: the entry keeps the old 268 MB buffer and the row has a new one. Sixteen siblings over
    one system prompt mean 16 × 268 MB of identical prefix KV in the cache. Only the budget stops
    it.
* **Each entry also holds its end state.** `_cache_arrays` (694–702) includes the ArraysCache
  contents, so every entry carries its end GDN state (51.5 MB). In this benchmark that state serves
  only an exact repeat of the same prompt (§6.4).
* **Inherited checkpoints are counted again.** `store` inherits the parent's checkpoints ≤ `at`
  (765–768). They are the same array objects, but `_Entry.nbytes` (728) sums them again, and the
  eviction loop (781) sums `e.nbytes` over entries.
  * In the benchmark the first request stores checkpoints {2048, 4096, 6144, s_user} (§6.3),
    206 MB. Every sibling inherits all four.
  * Charged per sibling: 268 (KV) + 51.5 (end state) + 206 (shared checkpoints) ≈ **523 MB**.
    Really new per sibling: ~317 MB.
  * 3 GiB = 3.22 GB holds **6 entries** (3.14 GB charged, ~2.1 GB real). Each later store evicts
    one.
  * The first request's entry survives. `lookup` (747–757) refreshes `used` only on the first
    entry with the deepest position, and all siblings tie at s_user, so the oldest one wins.

### 1.2 Design: shared immutable segments (§D2)

`_copy_cache`'s rule (docstring 661–667) forbids sharing **array objects**, not buffers. Slices are
new objects over the same buffer, and MLX copies on write when a shared buffer is written (no
donation). So sharing a buffer between entries, and between an entry and a restore, is safe.

Two consequences shape the design:

* **A view pins its whole buffer.** A segment that should outlive its source must be a *fresh*
  buffer: `mx.contiguous(c.keys[..., a:b, :])`. A slice on axis 2 of a (1, 4, cap, 256) array is
  non-contiguous, so this copies. Byte accounting must count buffers, not views.
* **Rows must never write into a segment.** They don't. Restore still produces a private buffer
  (one copy, as today), and the cascade groups of §D1 only read segments.

Structure (the existing API stays: `lookup`, `restore`, `store`):

```python
class _Seg:                      # KV of tokens [a, b) for every full-attention layer; own buffer
    a: int; b: int
    kv: dict[int, tuple]         # layer -> (k, v) arrays, or ((packed, scales, biases), ...) for quantized KV
    nbytes: int

class _Entry:
    ids: list[int]
    chain: list[_Seg]            # chain[0].a == 0, contiguous, chain[-1].b == len(ids)
    ckpts: dict[int, list]       # pos -> GDN states (shared array objects, as today)
    end: list | None             # GDN states at len(ids); None once dropped under pressure
    used: float
```

* **`store`**:
  * The new entry's chain is the parent's chain up to `at`, plus **one new segment
    `[at, len(ids))` copied out of the row's cache** (3.3 MB for a 100-token suffix). It is never a
    view, so the row's buffer is not pinned and the COW copy on its first decode write disappears.
  * Cold store (no parent): one segment `[0, len)`, also a copy (268 MB, ~1.6 ms). Today's view
    costs the same COW copy one step later.
  * A parent segment that straddles `at` is kept whole: it is shared, and its tail is just unused by
    this entry. Splitting by copy is optional.
  * The thinning and "covered" rules stay as they are (thinning fixed per §4).
* **`restore(entry, at, n_first)`**: per full-attention layer,
  `keys = mx.concatenate([pieces of chain up to at] + [zeros(round_up(n_first, 256))], axis=2)`,
  with `offset = at`.
  * That is exactly one copy, the same one today's first `update_and_fetch` makes. The zeros tail
    makes the first update write in place.
  * The K/V bits are identical, and GDN states come from `ckpts[at]` / `end` as today, so the output
    stays **byte-identical**. A restore at `len(ids)` uses `end`.
* **Accounting**: `bytes = Σ unique segments + Σ unique checkpoint arrays (by id) + Σ end states`.
  Recompute this on store and evict. It is O(entries × chain), which is tiny.
* **Eviction** (a tree in all but name; the chains form a radix tree whose nodes are segments):
  1. Drop `end` states of the LRU entries first. An entry without `end` still serves every interior
     checkpoint. Only an exact repeat loses its 0-token resume and falls back to the deepest
     checkpoint.
  2. Then evict LRU entries. Bytes freed = segments, checkpoints and end states referenced by no
     other entry. Leaves go first, because an interior segment is referenced by its descendants and
     frees nothing until they are gone.
  3. Keep "the newest always stays".
* **Lookup** stays a scan over entries for now (§4.3 shows it is cheap). A real radix walk becomes
  worthwhile only with dozens of entries.

### 1.3 Memory for 16 siblings of an 8K prefix (s_user ≈ 8000, ~100-token questions)

| | today (3 GiB budget) | segment store |
|---|---|---|
| shared prefix KV | 1 copy per entry (268 MB each) | 1 segment: 265 MB |
| per-sibling KV | (included above) | 15 × 3.3 MB = 49 MB |
| shared checkpoints (4) | charged 206 MB per entry, held once | 206 MB once |
| end states | 51.5 MB each | 16 × 51.5 = 824 MB (dropped first under pressure) |
| **entries kept / real bytes** | **6 / ~2.1 GB (3.14 GB charged)** | **16 / ~1.35 GB (~0.53 GB without end states)** |

For the benchmark this changes nothing measurable: all 15 hits happen while the first entry is
alive. It matters for revisits, and for multi-conversation agents with a shared 10–30K tool prompt.

---

## 2. Decode-time sharing: grouped (cascade) attention

### 2.1 What a step reads today

`_rows_attention` (1258–1281) runs `c.update_and_fetch(...)` plus the library SDPA per row (1277).
At B=16 with an 8K prefix, each step reads 16 × 268 MB = 4.3 GB of KV: a 10.7 ms floor at
400 GB/s.

step_profile.md (d) measured the per-row update + SDPA at 27 % of the step at 8K/B=16, ~24 ms. That
puts the step at ≈ 89 ms, a derived figure, and the attention at ~2× the bandwidth floor. Of the
Python build, 1.1 ms is the 128 per-row `update_and_fetch` calls.

### 2.2 Why "a view into the shared buffer" alone cannot work

`KVCache.update_and_fetch` grows the buffer with `mx.concatenate([self.keys, zeros])` (`cache.py`
522) and writes with a SliceUpdate (528).

* A row whose `keys` is a view of the shared prefix either concatenates (a full private copy) or,
  if the shared buffer had spare capacity, writes into it. The buffer is shared, so it is not
  donatable, and MLX copies it.
* Either way the row ends up with a private full copy.
* Sharing memory therefore needs **two segments per row**: the read-only prefix and the row's own
  suffix cache.
* With two segments, one SDPA over a contiguous K/V no longer exists. Attention must be computed
  over [prefix | own suffix] and merged. That is the cascade structure, so memory sharing and the
  attention speed-up come together.

### 2.3 The merge without an LSE from SDPA

`mx.fast.scaled_dot_product_attention` returns no log-sum-exp, so the merge must be manual. For one
group with G rows and shared prefix P, per full-attention layer:

```
q_g  = queries[rows of g]                      # (G, 16, 1, 256)
     -> reshape/transpose to (1, 4, 4G, 256)   # GQA: 4 q heads per KV head; 4G = 64 at G=16
sp   = (q_g · scale) @ Kpᵀ                     # (1, 4, 4G, P)    ← the prefix is read ONCE
ss   = per-row scores over the suffix stack    # (G, 4, 4, S_cap), positions ≥ len_i masked to -inf
m    = max(max(sp), max(ss))                   # per (row, head)
out  = (exp(sp-m) @ Vp  +  exp(ss-m) @ Vs) / (Σexp(sp-m) + Σexp(ss-m))
```

* **Numerics.** MLX's decode kernel (`sdpa_vector`) accumulates scores and output in fp32. A manual
  fp16 matmul would round the scores to fp16, which is the precision of MLX's *unfused* prefill
  SDPA, not of decode. The recommended choice is to **keep the group prefix in fp32** (one
  536 MB copy per 8K group instead of 16 × 268 MB) and do the whole cascade in fp32, casting to
  fp16 at the end.
  * Apple GPUs run fp32 FMA at the fp16 rate, so this costs only bandwidth: 536 MB → ~1.6 ms.
  * The only difference from a lone row is then the summation order, i.e. a ≤ 1-ulp fp16 output
    difference now and then. That is a smaller perturbation than mrqmm's fp16-output rounding
    (rms rel 2.9e-4), which was accepted at 99.7–100 % top-1 and KL ~1e-5.
  * If `agree.py` shows fp16 is equally good, keep the prefix in fp16 and halve the memory.
* **Cost at 8K, G=16.**
  * QKᵀ and PV are each 2·64·8192·256·4 ≈ 1.07 GFLOP per layer, so 17 GFLOP per step. At ~6–8
    TFLOPS that is 2.1–2.9 ms.
  * Bytes: 536 MB (fp32) once, ~1.6 ms, overlapped.
  * Scores (1, 4, 64, 8192) fp32 = 8 MB per layer, a few passes, < 1 ms total.
  * The suffix part is tiny (S ≤ a few hundred).
  * **Total ≈ 4–6 ms vs ~24 ms**, so the step drops from ~89 to ~70 ms: **−20–23 %, +25–30 %
    tok/s**.
  * At 32K the prefix share grows 4×: today's per-row attention alone would be ~96 ms per step
    against ~16–20 ms grouped, which gives ~×1.5–2 decode.
  * At 1K the gain is ≤ 6 %, since attention is 7.5 % of the step.
* **Kernel count.** If the suffixes live in one stacked `(G, 4, S_cap, 256)` buffer with a lengths
  vector, like the stacked GDN state, the update is one scatter per layer for all rows. The whole
  group's attention is then ~12–15 kernels per layer instead of today's ~4–6 per row (64–96 per
  layer at B=16). This also cuts the host build: the step_profile slack probe shows each host ms
  costs ~1 ms at B=16.
* **Rejected variants.**
  * *Masked single SDPA over one group buffer* `[prefix | slot_1 | … | slot_G]`: exact softmax and
    no merge, but every row's queries also score every other row's slots. Compute is ×2 at 512-token
    slots and ×9 at 4K generations, and slot overflow forces a re-layout copy of the prefix.
  * *Appending a dummy key to recover Z from SDPA*: needs head_dim 257 (no fused kernel) and
    underflows in fp16.
* **Later: a Metal kernel** (`mx.fast.metal_kernel`, as mrqmm/pfqmm did). Flash-decoding over the
  fp16 prefix in fixed key blocks for all 4G queries (simdgroup MMA, fp32 partials), plus the suffix
  blocks, plus a fixed-order merge.
  * It reaches the ~1 ms bandwidth floor without the fp32 copy.
  * It could serve non-grouped rows too: the "multi-row decode-attention kernel" step_profile.md
    names as the next lever.
  * With absolute-position key blocks it could even make segmentation bit-invariant, if all decode
    rows (B=1 included) used it. That would be a one-time output change for B=1.
  * ~300 lines of Metal. Do it after the MLX-ops version has measured the gain.

### 2.4 Bit-exactness

* **Prefill is untouched.** A hit row still restores a private contiguous cache (one copy), and its
  suffix prefill runs on it exactly as today. Only after `on_prefilled` (2129) is the row split into
  prefix reference + suffix copy, and the temporary buffer freed. Hit == cold for every cache bit.
* **B ≤ 2 stays exact.** No group exists then: rows are materialized (prefix + suffix concatenated
  into a plain `KVCache`, one copy, ≤ 2 × 268 MB) and decode through today's per-row path (qmv per
  row, library SDPA). This is the CLAUDE.md invariant "B ≤ 2 always bit-exact".
* **B ≥ 3 grouped rows** differ from solo by the attention summation order, on top of mrqmm. This
  is the same acceptance class: validate with `agree.py`, teacher-forced at B=3/8/16 over an 8K
  shared prefix, with targets ≥ 99.7 % top-1 and KL ≤ 1e-5.
* **New determinism caveat.** At B ≥ 3, whether a row is grouped depends on whether it hit the
  cache, so cache contents can now change a B ≥ 3 greedy answer. That joins the existing "under
  load a greedy answer can differ run to run". `QWEN_MAX_BATCH=1` or the new knob restores
  determinism.

### 2.5 What it needs from the scheduler

* **Group membership.**
  * A row that resumed with `reused ≥ GROUP_MIN` (e.g. 1024) joins the group keyed by
    `(id(first segment), P = reused)`.
  * The **producer** of the entry (the cold row 1, still running) holds the same bits for `[0, P)`,
    because they were computed in its own prefill with the same cuts. When the first sibling
    arrives, it is moved into the group too: its own prefix part is dropped and its suffix copied.
  * To find it, record `entry.producer = row` at store, or compare `row.req.ids[:P]` over the
    running rows: O(B·P) list compares, ~1 ms.
* **Positions**: unchanged. Positions are explicit (`row.pos + row.delta`, 2390 and the B ≥ 2
  path), and `_rope_deltas` is never read at decode.
* **GDN**: per row, stacked as today; nothing changes.
* **Excluded**: DFlash, which is always a lone row; vision rows, which are never cached; and
  quantized KV (27B/35B) at first, because a quantized prefix would need `quantized_matmul` for QKᵀ
  as the library's quantized SDPA does. That can be a later extension.

---

## 3. Concurrent cold miss

### 3.1 What happens today when 16 arrive at once

1. Request 1 is admitted (`_admit`, 1954). The others wait FIFO, and `_admit` runs only while
   `task is None` (1894).
2. Request 1 prefills cold: **17.2 s** at 8.15K (final_bench.md). Rows = 0, so `_slab_done` gives
   no decode share. It stores the entry and joins.
3. Each later tick admits the next request. `time.time() >= self.resume` (1896) holds because
   `resume` was last set by request 1's second-to-last slab (2127), and a single-slab suffix prefill
   never calls `_slab_done`. So each hit's prefill runs to completion in that tick: restore + one
   ~100-token slab at offset 8K + store, **t_hit ≈ 0.15–0.3 s**. The README's 7.5K hit is 0.20 s
   end to end.
4. **No decode step runs between the 15 hits.** Decode starts only once the queue is empty.
5. So **every** request's first token comes at ≈ 17.2 + 15·t_hit + one step ≈ **19.5–21.8 s**
   (p50 ≈ p95), including request 1. Its first token is produced by the first decode step after the
   burst.

### 3.2 Options

* **The floor.** The prefix must be prefilled once (17.2 s). The 15 suffixes are compute-bound
  already at ~100 rows, so packing them into one slab saves only the small-M inefficiency: about
  1500 × 1.76 ms ≈ 2.6 s vs 15 × t_hit ≈ 2.3–4.5 s. **Today is within ~1–2 s (5–10 %) of the
  floor.**
* **(a) Partial entries.** Waiting requests resume from the producer's checkpoints as they are
  produced. Prefill is serialized and the queue is FIFO, so a waiting request could only run *after*
  the producer anyway, and recomputing the prefix in parallel would be pure waste. Gain 0. The only
  use: a producer cancelled mid-prefill would leave its checkpoints for the next request.
* **(b) Prefill once and fan out.** That is what already happens, minus a wait that does not exist
  (the hits run back to back). Packing the fan-out suffixes is item 6: 1–2 s for the last request,
  but it needs the varlen-GDN "V2 piggyback" machinery.
* **(c) Do nothing.** Recommended.
* **One fairness note.** The burst starves decode for ~15·t_hit (2–4.5 s). That is harmless here,
  since all TTFTs are equal. With long-running rows already decoding, a burst of hits stalls them
  just as a long prefill does. If ITL under hit bursts ever matters, call `_slab_done` after every
  completed prefill and not only before a yield. That is a one-line change, and it does not affect
  outputs.

Distinct prefixes, for contrast: request k waits for k−1 cold prefills, interleaved with a decode
share of `1-(1-s)·B/16` (2060–2066). Expected TTFT p50 is roughly 8 × 17 s × ~1.2 ≈ 150–170 s. The
shared/distinct TTFT gap is therefore ~8×. The decode gap is 0 today.

---

## 4. Cost of the hit path

### 4.1 Full-buffer copies per request

Each copy is 268 MB read + 268 MB written, ~1.6 ms at ~330 GB/s:

| copy | where | per request |
|---|---|---|
| restore → first suffix slab | `cache.py` 522 (slice + zeros concat) | 1 |
| COW after store pins the row's buffer | 779 + `array.h` 304 + `cache.py` 528 | 1 |
| growth by 256 tokens | `cache.py` 513–522 (concat of the whole buffer) | ceil(answer/256): 1 for 256-token answers |

Per request that is ~3 copies, ~5 ms of GPU. At B=16 in steady state it is 16 rows × 1 growth copy
per 256 steps × 1.6 ms ≈ **0.1 ms per step, ~0.1 %**. Negligible. The transient memory (old + new
buffer during a concat, +268 MB per row) is absorbed by the 1.05 factor and the slack in `_bytes`
(1907–1916). §D2 removes the COW copy, and §D1 makes growth copies small (suffix only).

### 4.2 `mx.eval(arrs)` in `_Entry.__init__` (727)

There is no hidden sync. Each slab already ends with `mx.eval([c.state for c in cache])` (2118), so
the entry's arrays are views of evaluated buffers. The eval runs no kernels that wait behind
anything new.

### 4.3 Python host work, measured on CPU (pure Python copy of the same code, random ids)

| prompt | `_bounds` | `_lcp` per entry (mismatch → binary search) | `starts` scan |
|---|---|---|---|
| 8K | 1.6 ms | 0.6 ms | 0.2 ms |
| 32K | 6.2 ms | 3.0 ms | 0.8 ms |
| 128K | 18.5 ms | 13.9 ms | 3.1 ms |

* `store`'s covered check (776–778) adds one list compare per entry.
* Per admission at 8K with 6 entries: **~6 ms**, ×1.3–2 under gRPC load (step_profile (a)).
* At 128K: ~100–150 ms. That runs on the scheduler thread, but the queued decode step overlaps
  part of it.
* Fix (item 8) if it ever shows up: numpy `argmax(a != b)` for the LCP, and a closed-form
  `_bounds` when `_step` is constant (the fixed fp16 slab) that iterates only over `<|im_start|>`
  positions.

---

## 5. Admission accounting

* **`_reserve` is correct today.** `_tokens` (1918–1919) counts `len(req.ids)` even on a hit. That
  is right: the hit row owns a full private copy (§1.1).
* **With §D1 it must change.**
  * Do the lookup at admission. That is safe because the cache changes only inside a prefill
    (`store`), there is one prefill at a time, and `_admit` → `_prompt_cache` of the same row
    happens with no other prefill in between.
  * Pass the hit along to `_prompt_cache`.
  * Reserve `(len(ids) − P) + min(max_tokens, RESERVE_TOKENS)` tokens for a row that will be
    grouped. It still needs the full prompt *during its prefill*, because the temporary private
    buffer exists until the split, so reserve the full amount and shrink it at the split in `_join`.
  * The group reserves its prefix once (fp32: 2 × P × 32 KiB). That also covers materializing ≤ 2
    rows when the group dissolves, since the fp32 copy is freed first.
* **`_batch_budget` (1305–1319) subtracts the whole `PROMPT_CACHE_GB` statically.** Proposal
  (item 3):
  * `budget = cap − active − prefill_reserve − 1 GiB (− pool)`, without the cache term.
  * `_fits` (1945–1952) checks `reserved + cache.bytes + need ≤ budget`. On failure it evicts
    unpinned cache entries LRU-first until it fits or the cache is empty, then re-checks.
  * `PROMPT_CACHE_GB` stays the cache's own ceiling.
  * The live check (`mx.get_active_memory()`) already includes the cache's arrays, so it needs no
    change.
  * Segments referenced by a running group are **pinned**: not evictable, and still counted in
    `cache.bytes`, so they are never counted twice.
  * Gain: +3 GiB of batch budget whenever the cache holds less (always with distinct prompts). In
    this benchmark there is no admission pressure either way: 16 × (8.45K tokens × 32 KiB × 1.05 +
    2 × 51.5 MB) ≈ 6.4 GB against a ~12 GiB budget.

---

## 6. Eviction and policy

* **6.1 LRU.** LRU over whole entries ignores sharing (§1.1). With segments: drop end states first,
  then LRU entries, leaves first by construction. Bytes freed = unreferenced items only.
* **6.2 "Covered" rule (775–778).** It is correct as a dedup, since the new entry holds the old one
  as a prefix with a checkpoint at its end. It keeps working on chains. On quantized models it can
  drop the only entry able to resume below `quantized_kv_start` (§7.4).
* **6.3 Checkpoint placement for the benchmark prompt** (`<|im_start|>system\n` + filler +
  `<|im_end|>\n<|im_start|>user\n` + question + generation prompt):
  * `_bounds` (831–846) cuts at every 2048 tokens. With the fixed fp16 slab `_step` is 2048 at
    every offset, because `PREFILL_BUDGET // pos ≥ 2048` up to 32K and the fused branch from 32K.
  * It also cuts at the user `<|im_start|>` (s_user) if `s_user − last cut ≥ PREFILL_MIN_STEP`
    (256).
  * Request 1's `keep` (2029–2043) includes `starts[1:2] = [s_user]`, mapped to the closest cut
    ≤ s_user.
  * Every sibling shares tokens through `<|im_start|>user\n`, so `lcp ≥ s_user + 3`. Cuts depend
    only on the tokens before them, so the sibling's `ends` contain the same cut.
  * **A hit at s_user is guaranteed.** If `s_user mod 2048 < 256`, the hit lands at the 2048
    multiple below instead (≤ 255 tokens re-prefilled). This assumes the 16 system prompts render
    to identical tokens.
  * The prompt has 4 checkpoints (or 5 if s_user > 8192), fewer than 12, so no thinning.
* **6.4 Why resume only at slab ends (`best`, 731–740).** The suffix after the resume point must be
  cut into the same slabs as a cold run. Kernels round differently by slab shape (split-K, SDPA key
  blocks; `_bounds` docstring), and the GDN state exists only at slab ends anyway. That keeps
  "hit == cold" bitwise.
  * The entry end (`len(ids)`, the end state) serves only an exact repeat. A next turn diverges
    before it in thinking mode: history renders the assistant turn without the pre-opened
    `<think>` the stored prompt ends with. It resumes at the checkpoint mapped from
    `starts[-3:]`, near the last messages.
  * Hence "drop end states first".
* **6.5 Thinning bug (770–773).** The rule drops `ps[i]` with the smallest `ps[i+1] − ps[i−1]`.
  Stride checkpoints have span 4096. A message-boundary checkpoint m between k·2048 and (k+1)·2048
  has span exactly 2048, and its neighbours have spans in (2048, 4096).
  * So **the system-prompt checkpoint is the first one dropped** once a prompt has > 12 checkpoints
    (> ~24K tokens). The same holds for `split` and the `starts[-3:]` checkpoints.
  * New conversations then resume at the stride checkpoint below: up to 2047 tokens of re-prefill
    (≤ ~3.6 s instead of ~0.2 s), unless a shorter entry still holds the boundary.
  * Fix (item 4): tag checkpoints from `starts`/`split` as protected and thin only stride
    checkpoints. Byte-identical.

---

## 7. Other findings

1. **Thread safety: OK.** `eng._pcache` is touched only in `_prompt_cache` (2011–2056) on the
   scheduler thread (and in `Engine.__init__`, 2540). No handler or Check path reads it.
2. **`lm._rope_deltas` on a hit (2017): OK under batching.**
   * Decode always passes explicit positions (B=1: 2390; B ≥ 2: the `pos` array in `_step`), so the
     zeros left there are never read at decode.
   * A text miss starts at cache offset 0, where `LanguageModel.__call__` recomputes positions and
     resets the delta to zeros for text (`language.py` 1651–1687). A vision request resets on
     `pixel_values`.
   * `_spec` sets the zeros itself (2193).
3. **DFlash after a hit loses drafter context (speed only).** `_spec` feeds the drafter the
   captured hidden states of the slabs it prefilled (2196–2220). On a hit that is the suffix only,
   e.g. ~100 tokens instead of the window of 4095 (`sliding_window` 4096). Acceptance after a hit is
   probably lower, like the known "first rounds have no context". Greedy output is unaffected (the
   target verifies). Worth one measurement. A fix would need ~4095 × 5 layers × 4096 × 2 B ≈ 170 MB
   of hidden states per entry, so it is probably not worth it. DFlash is never on with 16
   concurrent requests anyway (`_spec_ok`, 1978–1989).
4. **Quantized KV (27B/35B).**
   * `best` refuses positions below `quantized_kv_start` (5000) for a quantized entry (738). This
     is correct: the cold run has fp16 KV there, and `to_quantized` quantizes the whole cache once
     `offset ≥ 5000` (mlx-vlm `generate/common.py` 203–212).
   * But a system prompt < 5000 tokens can then be shared only through an entry whose prompt
     stayed < 5000. When the "covered" rule (776–778) replaces such an entry with its quantized
     continuation (non-thinking continuations, where `len(e.ids)` is a checkpoint of the new
     entry), the < 5000 resume point is gone.
   * Fix (item 5): skip the covered rule when the old entry is unquantized and the new one is
     quantized. Optionally, in §D2, keep the fp16 head segment `[0, last cut < 5000)` captured at
     the checkpoint hook before quantization: ~100 MB on the 35B, ~320 MB on the 27B.
   * Quantization is per token (groups along head_dim), so resuming a quantized cut ≥ 5000 is
     exact.
5. **Vision**: excluded on both lookup and store (2008). Fine.
6. **Tests**: only `test_prompt_cache_while_batching` (`tests/test_scheduler.py` 257) covers the
   cache: 3 requests, outputs == cold, `reused > 0`. Nothing unit-tests `_PrefixCache`: eviction,
   thinning, quantized floor, nbytes, the covered rule.
7. **The benchmark must set `QWEN_PROMPT_CACHE=1`.** It is off by default, and final_bench.md ran
   with it off.

---

## D1. Detailed design: grouped shared-prefix decode attention (item 1)

**Knob.** `QWEN_PREFIX_GROUP` (default 0 until `agree.py` passes, then 1); `QWEN_PREFIX_GROUP_MIN`
(1024 tokens).

**Data.**

```python
class _Group:
    key: tuple                         # (id(segment), P)
    P: int
    kp: dict[int, mx.array]            # FA layer -> (1, 4, P, 256) fp32 (or fp16), read-only
    vp: dict[int, mx.array]
    rows: list[_Row]                   # members, in batch order
    sk: dict[int, mx.array]            # (G, 4, S_cap, 256) fp16 suffix stack (grows by 256, like KVCache)
    sv: dict[int, mx.array]
    lens: list[int]                    # suffix length per member
    reserved: int                      # bytes of kp/vp, counted once
```

A row in a group has `row.group` set and no full-attention `KVCache` objects. GDN stays in
`self.gdn`.

**Attention.** In `_rows_attention`, the cache argument becomes a `GroupedKV(groups, singles)`
alongside `RowsKV`. Projections, norms and RoPE stay batched as now. Then:

* for each group, the §2.3 computation;
* for each single row, today's per-row `update_and_fetch` + SDPA;
* results go back into row order, then `concatenate` / `o_proj` as now.

The suffix write is one scatter per layer (`sk[g_idx, :, lens, :] = k`). Growth past `S_cap` is a
concat of the small stack.

**Scheduler.**

* **`_join(row)`** (2320): if B ≥ 3 after the join, the row resumed with `reused ≥ GROUP_MIN`, and
  KV is fp16:
  * find or create the group;
  * create it from the entry's segment (fp32 copy of `[0, P)`) and pull in the producer row if it
    is running;
  * append the row's `[P, pos)` to the suffix stack and drop its full KV.
* **`_compact`** (2333): drop finished members from their groups. When B ≤ 2, **dissolve** every
  group: materialize each member into a plain `KVCache` (`concat(kp.astype(fp16), suffix)` + zeros
  capacity), then free kp/vp. A group with 1 member at B ≥ 3 stays (no copy). Add hysteresis so
  that B oscillating 2↔3 does not thrash: re-group only on a join.
* **`_regrow`/`_bytes`**: a grouped row is accounted on suffix tokens. The group's `reserved` is
  added once to `self.reserved` and removed at dissolve. `_fits` uses the post-split reservation for
  rows that will be grouped (§5).
* **`_fail_rows`/`_close`**: drop groups with the rows.
* **The prefix-cache segment of a live group is pinned** (§5).

**Stays byte-identical:**

* every prefill (cold and hit);
* every step at B ≤ 2;
* `QWEN_PREFIX_GROUP=0`;
* DFlash, vision and quantized-KV models;
* all non-grouped rows at B ≥ 3, which take the same per-row path and the same mrqmm as today.

**Changes:** grouped rows at B ≥ 3, by the summation order only.

**Tests.**

* *CPU (tiny models, `tests/test_rows.py` style):*
  * `test_group_attention_matches_rows`: random q/K/V, ragged suffix lengths; the group output
    against per-row library SDPA within fp16 tolerance, and masked positions contributing exactly
    0.
  * `test_grouped_batch_equals_solo` (dense + dense_fused, 4 requests over a shared system prompt,
    max_batch 4): tokens equal solo within the tolerance class `test_rows_equal_solo` uses for
    B ≥ 3, and exactly equal with `QWEN_PREFIX_GROUP=0`.
  * `test_group_dissolve_at_b2`: rows finish until B=2. Members are materialized, decode continues
    error-free, `offset`/`pos` stay consistent, and the materialized K/V equal `concat(prefix,
    suffix)` bitwise.
  * `test_group_producer_joins`: the cold row is pulled into the group when the first sibling
    joins.
  * `test_group_accounting`: `reserved` = suffix-only rows + the group prefix once, and back to 0
    at the end.
* *GPU (lease):*
  * `agree.py` at B=3/8/16 over an 8K shared prefix: ≥ 99.7 % top-1, KL ≤ 1e-5; try fp16 vs fp32
    prefix.
  * `step_prof.py --ab group0,group1` at 1K/8K/32K shared.
  * `bench_concurrency` shared-prefix level.

## D2. Detailed design: segment-sharing store (item 2)

See §1.2. Code: `_Seg`, `_Entry.chain/end`, `_PrefixCache.store/restore/_bytes/_evict`, and
`_prompt_cache` passing the first slab length to `restore`. `_copy_cache` stays for the
quantized-tuple handling.

**Tests (CPU).**

* `test_store_shares_prefix_segment`: three siblings. `cache.bytes` = one prefix + 3 suffixes +
  shared checkpoints once + 3 end states. Each new segment's `nbytes` = its token range (it is a
  fresh buffer, not a view).
* `test_restore_bits_equal_cold`: chains of 1/2/3 segments (system → turn 1 → turn 2) and a
  restore at an interior checkpoint. Restored K/V equal the cold cache at that position
  (`mx.array_equal`), GDN states identical, outputs equal.
* `test_evict_end_states_then_leaves`: a tiny `PROMPT_CACHE_GB`. End states go first, then leaves.
  A shared segment survives while referenced.
* Extend `test_prompt_cache_while_batching` to 8 siblings under eviction pressure.
* With kv_bits=8 and kv_start small: the quantized floor and the covered-rule fix (item 5).
* `test_thinning_keeps_boundaries` (item 4).
* GPU: the README's five TTFT scenarios plus the bitwise first-token logprob check, as for the
  original cache.

## D3. Detailed design: elastic budget (item 3)

`_batch_budget` (1305) drops the `PROMPT_CACHE_GB` term. `_fits` (1945) becomes:

```python
need = self._reserve(req)
while self.reserved + cache.bytes + need > self.budget and cache.evict_one(): pass
```

`evict_one` returns False when only pinned items remain. The live check is unchanged.

Tests: update `test_budget_derivation` (`tests/test_robustness.py` 240), plus a test where a full
cache yields to an admission and outputs are unchanged.

## Open questions for the benchmark

1. **Shared vs distinct, decode.** Is decode tok/s per request at B=16 the same in both cases
   (predicted: yes, ±drift)? This confirms there is no decode-side sharing today. Use the worker's
   `batch: steps/s` lines at steady B=16 and compare with a 1K-context run at the same B to
   calibrate the ~24 ms attention share. That number sets item 1's gain.
2. **Shared TTFT shape.** Are all 16 TTFTs ≈ T_cold + 15·t_hit (p50 ≈ p95 ≈ 20–22 s, request 1
   included)? Are there 15 `prompt cache HIT` lines with `reused` = s_user (or the 2048 multiple
   below)? What is t_hit (each hit's prefill in the log)? It bounds item 6.
3. **Cache occupancy.** Expect `store` log totals near 3.1 GB charged with ~6 entries and
   evictions from the 7th store on (§1.1). `mx.get_active_memory` should show ~2.1 GB actually
   held.
4. **Peak memory.** Expected: shared ≈ distinct (rows own full copies). A visibly lower shared peak
   would mean MLX shares more than this analysis assumes.
5. **Distinct case with the cache on vs off.** Predicted no difference beyond noise: stores are
   views plus one COW copy, and the budget has no pressure at 16 × 8K. Any visible difference
   would point at an overlooked host cost.
6. **If DFlash is ever measured with the cache:** acceptance after a hit vs cold (§7.3).
