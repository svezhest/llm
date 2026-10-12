# Review: cascade B1 (`src/worker/cascade.py`, `src/worker/tests/test_cascade.py`)

Scope: step B1 of `design/cascade_plan.md` (Part B). The implementation goes beyond the plan's
B1. It also contains B2's group data structure as a module class (`PrefixGroup`: build / add /
take / grow / release / begin_step / append / end_step / abort_step). This review was read-only;
the experiments ran on scratch copies in the session scratchpad.

* CPU suite: `uv run --group dev pytest -q tests/test_cascade.py` gives **44 passed, 9 skipped**
  (GPU-only tests skipped).
* GPU variant: **not run**. Another agent's CPU pytest runs were active throughout
  (`pgrep -fl pytest`), so the rule "nothing else running" was never met. Every GPU-dependent item
  below is marked "GPU unverified".

## Findings (ranked)

### 1. MEDIUM: `take()` / `add()` evaluate eagerly by default, which holds the old and new stacks of every layer at once and forces a host sync

`cascade.py:397-416` (`take`, `evaluate=True`, `mx.eval(sk, sv)` at :413-414 while `self.sk/sv`
still hold the old arrays until :415), and `cascade.py:369-390` (`add`, same pattern at :385-387).

**Failure scenario.** Late in a wave, rows finish with long suffixes. Example: 9B, 15 rows kept
of 16, cap 4352.

* Each `take` builds a fresh contiguous stack for all 8 layers. That is about 4 GiB (16 × 4352 ×
  64 KiB = 4.25 GiB before the take).
* The old 4.25 GiB stays alive until `mx.eval` returns, so the peak rises by about 4 GiB.
* Nothing in plan B.5 accounts for this. `_group_bytes` charges the stack once, plus score
  transients.
* Measured on a scaled-down CPU copy (8 rows, 8 layers, suffix 1000):
  * `take(evaluate=True)`: peak +448 MiB, the size of the whole new stack;
  * `take(evaluate=False)`, with the next step's appends consuming it layer by layer: peak +48 MiB.
* The eager `mx.eval` in `_compact` also waits for the in-flight step k, because `take` reads
  `sk_k`. That loses the step-ahead overlap on every row exit. `release(evaluate=True)` has the
  same sync. It is acceptable there because release is rare.

**Fix.**

* B3 should call `take(..., evaluate=False)` and `add(..., evaluate=False)`. The lazy
  `contiguous(take(...))` is then consumed by the next step's `put_along_axis` of each layer: the
  old stack is referenced only by the graph, the result is donatable, and the transient is about
  one layer.
* Alternatively, flip the defaults to lazy.
* Either way, add a transient term for take/add/grow to B.5. `grow` is already lazy in
  `begin_step`, but at cap 4096 → 4352 its per-layer old K+V is about 0.5 GiB alive during the
  concat; the 5 % slack in `_group_bytes` is 0.2 GiB.
* The class docstring's atomicity promise ("an exception leaves the group as it was") holds only
  at build time. With lazy evaluation, an OOM surfaces in the step, and the plan's `_fail_rows`
  already drops all groups in that case.

### 2. MEDIUM: nothing enforces "append exactly once per layer, before attention, before end_step"

`cascade.py:486-504`: `append`, `attention` (:215-230) and `end_step`.

**Failure scenario.** A B2/B3 bug skips `append` for a layer, for example a layer-key mismatch or
an early `continue`. Then:

* `attention` reads slot `S_j`, which the mask attends (`+1` inside a step) but which still holds
  zeros;
* `end_step` advances `lens`;
* a zero key gets score 0 and weight `exp(0 - m)` with a zero value, so it silently dilutes that
  row's attention in every later step;
* there is no NaN, and `assert_unused_zero` cannot see it, because the slot is "used".

Calling `append` twice is harmless, but `attention` before `append` has the same effect as a
skipped append for that layer.

**Fix.** Track a per-step set `self._appended`, reset in `begin_step`:

* `append` adds the layer and raises on a duplicate;
* `attention` raises when `_pending` is set and the layer is not in the set;
* `end_step` raises when `_appended != set(self.layers)`.

This is pure Python and costs nothing.

### 3. MEDIUM-LOW: `put_along_axis` has no bounds check; the "index < cap" invariant has two sources of truth and is not asserted

`cascade.py:476-484` (`begin_step`), `:418-429` (`grow`), `:495-496`.

**Failure scenario.** An out-of-range write index is not an error in MLX.

* Mutating `_upos` to `lens + 1` or `grow` to an off-by-one cap killed the CPU pytest process
  with **SIGBUS**, an out-of-bounds scatter.
* On Metal it would silently overwrite a neighbouring buffer: another layer's KV, weights, or the
  GDN state.
* Today the code is safe by construction, because `begin_step` grows to `max(lens) + 1`.
* However:
  * `grow()` is public and not step-guarded. A mid-step `grow` also resets `_upos` to None
    (`_changed`, :429), and the next `append` then fails obscurely.
  * `lens` is a second copy of `row.pos - P`, which plan B.3 computed from `row.pos` in `_plan`.

**Fix.**

* At the end of `begin_step`, assert with Python ints:
  `max(self.lens) < self.cap and all(a.shape[2] == self.cap for a in (*self.sk.values(), *self.sv.values()))`.
* Add `self._no_step("grow")` to `grow`. `begin_step` calls `grow` before it sets `_pending`, so
  this is safe.
* In B2/B3 `_plan`, assert `g.lens == [r.pos - g.P for r in g.rows]`.

### 4. LOW: `abort_step` chains onto the aborted step's lazy graph

`cascade.py:506-516`: zeros are `put_along_axis`'d on top of the `sk[i]` that `append` already
rebound to a lazy scatter of the failed step's keys.

**Failure scenario.** Suppose the build failed because of an eval-time error inside the graph that
the keys depend on: a synchronous `.item()` or OOM during the build, or a grammar mask eval. Then:

* every later step's stack still depends on that poisoned graph, so every step fails (strikes);
* even in the benign case, the aborted forward up to the last appended layer runs again on the
  next eval, which is wasted GPU time.

A snapshot of the pre-step arrays would fix it, but it would also kill donation: it adds a second
reference, and every step would copy the stack.

**Fix.**

* Document that `abort_step` is only for pure-Python build errors.
* Or `mx.eval` the stacks inside `abort_step`, and have the caller drop the group if that raises.
* The plan's `_fail_rows → self.groups = []` already covers the scheduler path, so `abort_step`
  may end up unused in B3. In that case, say so.

### 5. LOW (perf; verify in B4): donation under the scheduler's step-ahead. Correct, but plan B.3's reasoning is wrong and no test covers it

Plan B.3, "Donation": "the previous step's attention read of it is finished before this step's
`put_along_axis` runs, because this step depends on the previous step's token".

**Why that reasoning is wrong.** Donation is decided at **encode** time on the host:
`array::is_donatable()` is `array_desc_.use_count() == 1 && data.use_count() == 1` (`mlx/array.h:304`).
In `_iterate`, step k+1 is built and `async_eval`'d while step k is still running on the GPU, so
GPU ordering says nothing about whether the scatter can donate.

**Why the code is still correct.**

* MLX's Metal eval retains each input's `Data` shared_ptr in the command buffer's completion
  handler until that buffer completes.
* While step k's buffer that read `sk_k` is in flight, `sk_k` is therefore not donatable, and the
  scatter **copies**: a perf loss, never corruption.
* This is the same mechanism that protects today's `KVCache` slice updates.

**When it costs speed.** Only when step k+1's layer-i scatter is encoded before step k's layer-i
reader completes. With MLX's cap of ≤ 10 in-flight command buffers and a step that spans well over
10 buffers, that reader has normally completed.

**Verification so far.**

* CPU experiment (scratch `exp/ahead.py`: B 8, P 1K, 4 layers, 30 steps, `async_eval` of the step
  output, then build the next step before reading):
  * outputs identical to a fully synchronous run;
  * peak above base +7 MiB in both modes, less than one layer's 8 MiB stack, so no per-step copy.
* GPU unverified. `test_gpu_timing_smoke` syncs every step (`mx.eval(..., g.sk.values())`, test
  :527), so it cannot show this.

**Fix.**

* Add a test that queues step k+1 before evaluating step k, and compare it with the synchronous
  run. CPU is enough for correctness.
* Assert `peak - base < one layer's stack` in the GPU smoke.
* In B4 step_prof, compare step-ahead vs sync step time.
* Correct the B.3 wording.

### 6. LOW (design input for B3): small groups use more memory than ungrouped rows

| | bytes |
|---|---|
| grouped | P·64 KiB + B·S·64 KiB (fp32 prefix and fp32 suffix, 9B) |
| per row | B·(P + S)·32 KiB |

* Break-even is at S = P·(B−2)/B: 0.875·P at B = 16, but P/3 at B = 3.
* The `P // 2` leave rule therefore keeps a 3- to 7-row group in the memory-losing band; with a
  long tail of rows a group shrinks toward 3 rows.
* Speed is also weakest there: the suffix GEMM has M = rep, and there are few rows to amortise the
  prefix.
* Accounting stays correct (B.5 charges it), so this costs efficiency, not an OOM.

**Suggestion.** Make the dissolve or leave rule memory-aware (`_group_bytes(g) > Σ _bytes(row)`),
or raise the dissolve threshold from 2 rows.

### 7. LOW: `append` accepts any dtype

`cascade.py:495-496`.

`release`'s bit-exactness claim (:451-454, module docstring :55-59) assumes every stored value was
a `kv_dtype` value. If keys ever arrive as fp32, for example after a RoPE change that promotes,
the fp32 stack keeps unrounded values that `release` then rounds.

* The per-row path would have rounded on write, and then attended the rounded values, so the
  grouped and per-row paths would diverge.
* `release` stays deterministic.

**Fix.** Raise unless `keys.dtype == values.dtype == self.kv_dtype`, or store
`keys.astype(kv_dtype).astype(F32)`.

### 8. Test gaps (`tests/test_cascade.py`)

* **No step-ahead / `async_eval` test** (see 5).
* **Release-then-continue** (:313-316) checks only `k.shape[2]` after one `update_and_fetch`.
  * I checked it in scratch (`exp/release_cont.py`) on CPU: P 300; suffixes 0 / 9 / 37 / 255 /
    256; 300 further steps crossing capacity on both caches.
  * The library SDPA on the released cache is bitwise equal to the original row's at every step.
  * Worth pinning as a test, also on GPU (strides differ: capacity is P + S + step, not a multiple
    of 256). GPU unverified.
* **Compile eviction safety.** The docstring claim at :95-96 was "not checked" by the author. I
  checked it in scratch (`exp/funid.py`):
  * setup: `MAX_TRACES = 1`, 400 calls alternating 4 different `scale`s;
  * the closure lambdas reused only 3 addresses, so `fun_id` collided repeatedly, yet every result
    matched the fp32 reference;
  * so the evicted wrapper's trace is erased, not reused.
  * Worth a test: alternating scales under `MAX_TRACES = 1`.
* **Mutation check** (scratch `mut/mutate.py`, each mutation on a copy of `cascade.py` against
  the unmodified tests). All were caught:

  | mutation | tests failed |
  |---|---|
  | smask without +1 | 4 |
  | end_step +0 | 8 |
  | end_step +2 | 8 |
  | write index +1 | SIGBUS |
  | wrong GQA order | 34 |
  | no empty-suffix guard | 5 |
  | pmask dropped | 24 |
  | release suffix zeroed | 3 |
  | abort no-op | 1 |
  | non-zero stack fill | 3 |
  | take without cap trim | 1 |
  | grow off-by-one | SIGBUS |
  | scale ×1.001 | 34 |

  So the mask boundary and an `end_step` off-by-one are pinned. They are caught through
  `test_decode_steps_match_rows` and `test_empty_suffix_and_exact_chunk_multiple`, where the new
  token is the whole suffix of the S = 0 rows. The out-of-bounds cases are caught only by a crash
  (see 3).
* `test_gpu_rms_criterion_9b` asserts `e_c <= e_t` for every case, while plan B.7 asks for
  ≥ 95 % of pairs. Fine as long as the margin holds (microbench 2.1e-4 vs 2.8e-4), but it may
  flake.

## Verified OK

* **GQA.** MLX maps q head h to kv head `h / gqa_factor` (`sdpa_vector.h:63`).
  * `reshape(B, Hkv, rep, D)` (:155) gives kv head major.
  * `transpose(1,0,2,3).reshape(Hkv, 1, B·rep, D)` puts row `b·rep + r` (:156).
  * The inverse at :173 is exact.
  * The suffix path stays in `(B, Hkv, rep, ·)`, and its LSE and output are moved to the prefix
    layout the same way.
  * A wrong order fails 34 tests.
* **Scale.** It is applied to fp32 q before the matmul, as the kernel does
  (`sdpa_vector.h:85`, `scale * float(q)`). A Python float times an fp32 array rounds to the same
  float32 scale.
* **Output dtype.** The output is cast once from fp32 to `q.dtype` (:174). fp16, fp32 and bf16
  were all tested.
* **Empty suffix (outside a step) is exactly prefix-only.**
  * `lse_s = −inf`, `o_s → 0` (:167), `m = lse_p`, `wp = exp(0) = 1`, `ws = 0`, so
    `out = (o_p + 0) / 1 = o_p` bitwise.
  * On Metal, `logsumexp` of an all-−inf row returns −inf, not NaN. `maxval` starts at
    `finite_min`, so `isinf` is false and `log(0) + finite_min = −inf` (`logsumexp.h:39, 70`).
    The `== −inf` guard therefore also fires on the GPU (GPU run unverified).
  * Inside a step every suffix has ≥ 1 key, so `lse_s` is finite.
* **No all-padding chunk.** `Ppad = ceil(P/Pc)·Pc`, so the last chunk has ≥ 1 real key and every
  `lse_c` is finite. `exp(−inf − m) = 0` handles the padded tail, and the zeros in the padding
  make 0 × V finite. No NaN path exists unless K/V are themselves inf or NaN, and then the per-row
  path would be NaN too.
* **Overflow and underflow.**
  * fp32 scores ≤ ~7e10 even at fp16-max inputs.
  * `logsumexp` and `softmax` subtract the max.
  * The merge weights lie in (0, 1] with one of them equal to 1, so the denominator is ≥ 1.
  * Underflow of the smaller weight is the correct limit.
* **Mask boundary.** The suffix mask attends `[0, S_j]`, which includes the slot `append` just
  wrote. This matches `update_and_fetch` followed by SDPA over P + S + 1 keys.
* **Holders copy only `[P, offset)` and `[0, P)`.**
  * Padding and spare slots are zeros.
  * `take` preserves the zeros, `grow` pads with zeros, and `abort_step` re-zeroes the slot.
  * NaN past the source offsets never reaches the group (test :242).
* **Release.** `release` is bit-exact for fp16, fp32 and bf16, also after grouped steps across
  growth.
* **Retrace bound.**
  * The shape key is (B_g, c, cap). Over one 16-row group decoding 4K tokens: ≤ 17 caps
    (256…4352) plus ≤ 14 sizes (16 → 3), about 31 keys over its life (cap grows while B_g
    shrinks, so the keys follow a path, not a product).
  * Host trace cost per new key: about 0.17 ms after a one-time 28 ms warm-up (CPU,
    `exp/trace_cost.py`).
  * LRU 64 is ample.
  * `shapeless=True` is not usable here (the `B·rep` reshapes).
* **Memory, 9B** (8 FA layers, Hkv 4, D 256; fp32 K+V = 64 KiB/token):

  | item | size |
  |---|---|
  | prefix, 8K | 512 MiB |
  | prefix, 32K | 2 GiB |
  | stacks, B_g 16, cap 4352 | 16 × 4352 × 64 KiB = **4.25 GiB** (272 MiB per K per layer) |
  | per-layer step transient, 8K / B 16 | `s` and `p` 8 MiB each, `o_c` 2 MiB, suffix `ss` + `p` about 8.5 MiB at cap 4352: about 27 MiB per layer |
  | per-layer step transient, 32K | `s` and `p` 32 MiB each |
  | per-row today, same rows | 16 × 12.3K × 32 KiB = 6 GiB |

  The docstring's "~0.2 GB per 8-layer step" is conservative. `nbytes` is exact for the arrays it
  lists; it omits `_smask` (n × cap × 4 B, under 0.3 MB).

## Nits

* **Compile cache key** (`_key`, :181-183) uses `str(mx.default_device())`, not the stream. The
  worker traces and runs everything under `mx.stream(generation_stream)` on one thread, so this
  is fine. Be aware of it for tools that call `attention` outside that stream.
* **Plan text is out of date.** The plan's B1 put the knobs in `worker.py` and the GPU tests in
  `tests/test_cascade_gpu.py`. The module instead takes `chunk` / `step` parameters, and the GPU
  tests live in `test_cascade.py` behind `CASCADE_GPU=1`. Update B.2 and B.3 so that B2 does not
  build a second copy of `upos` / `smask` / `lens`: the group owns them now. `_PrefixGroup`'s
  `key` / `ids` / `rows` / `reserved` must wrap this class.
* **Unused attributes.** `prefix_kv` and `nbytes` are fine. `cascade.traces()` reads `_FNS`
  without the lock (harmless).
* **`abort_step`** writes zeros into every layer, including layers never appended: harmless, 16
  scatters.
* **Module docstring :100-103.** Also mention `release(evaluate=False)` and `take(evaluate=False)`:
  a lazy slice of `sk[i]` pins the stack and costs the next append its donation.
