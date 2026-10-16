# Review: ragged multi-row decode attention (`rattn`)

16.10.2026. Reviewed: `src/worker/rattn.py`, `src/worker/tests/test_rattn.py`,
`scripts/perf/microbench/microbench_rattn.py`, the `scripts/perf/prof/step_prof.py` diff and
`docs/perf/2026-10-16/rattn_microbench.md` (+ `rattn_raw/`). References: `_rows_attention` / `RowsKV`
(`worker.py` 1753-1801), MLX 0.32.3 `kernels/sdpa_vector.h`, `cascade.py`.

The review ran on the CPU only and read the code statically. Nothing ran on the GPU, and the Metal tests were
not run. The only thing executed was the host-side item table (`rattn._meta`), on the CPU: its contents for edge
lengths, and its cost.

**Verdict.** I found no correctness bug in the kernel. Indexing, the GQA mapping, the softmax merge and the chunk
and row boundaries are all right. The findings below are about the test gating (one MED), two things to settle
before integration (one MED, plus a LOW about the launch split), input checks the code does not make, robustness
nits, and how strong the measurements are.

## Findings

### 1. MED: `test_rattn.py` runs on the GPU in the routine CPU suite

`tests/test_rattn.py:13` is gated only on `mx.metal.is_available()`, which is true on this Mac. The documented
CPU suite is `pytest -q --ignore tests/test_mrqmm.py`, and it ignores only mrqmm. So every routine run now
launches Metal kernels (and allocates ~1 GB of random KV in the 16-row case) without the GPU lease, and possibly
while a worker or a measurement is running. `test_pfqmm.py` is the house pattern: its Metal tests run only with
`PFQMM_GPU=1`.

**Fix:** gate the module on `RATTN_GPU=1 and mx.metal.is_available()`, the way `test_pfqmm.py:224` does. Or add
`--ignore tests/test_rattn.py` to the documented CPU command when the kernel lands.

### 2. MED (before integration): short contexts and small B are unmeasured, and the chunk size is fixed

`CHUNK = 512` (`rattn.py:42`) gives one threadgroup per (512 keys × kv head). Two examples:

* B=3 rows of 1K keys: 6 items × 4 kv heads = 24 threadgroups of 128 threads, on a 32-core GPU.
* B=8 rows of 600 keys: 16 items, so 64 threadgroups. Each simdgroup walks 128 keys serially.

MLX's per-row 2-pass always spreads a row over 64 blocks. All the numbers in the report are at 5–33K.

The report's integration plan (option 2: `QWEN_RATTN` for B ≥ 3 and fp16 KVCache rows, no length condition)
would send every batched step through rattn, including the 1K chat case, which has not been measured.

**Fix:** before integrating, run the microbench at 1K / 4K × B = 3, 4, 8, 16. Then either pick the chunk per
launch from the total work, for example the largest power of two in [64, 512] that still gives at least ~4 ×
cores threadgroups, or gate on the total number of keys.

### 3. LOW: `MAX_ROWS = 8` is not the buffer limit

Pass 1 binds `q, items, rows` + 2n K/V + `po, pml` = 2n + 5 buffers. No `*_shape` / `*_strides` names appear in
the source, so MLX adds no extra buffers. With 31 buffers, n ≤ 13.

The 8 comes from bad arithmetic: the module docstring (`rattn.py:25-26`), the comment at `:44` and the report
(line 87, "at most 8 rows (2·8 + 5 buffers)") all state the bound as 31, but 2·8 + 5 is 21. As a result, B = 9..13
pays a second launch pair plus a concatenate for nothing, every FA layer of every step.

**Fix:** set `MAX_ROWS = 13` and split into balanced launches: `k = ceil(B / 13)` launches of `ceil(B / k)` rows.
B=16 then still runs as 8 + 8, and B ≤ 13 is one launch.

### 4. LOW: the input contract is documented but not checked

* **`lens[i] == 0`:** the row gets no items, so pass 2 computes `0/0` and writes NaN (`rattn.py:141`).
* **`lens[i] > cap_i`:** the kernel reads past the head's region, into the next head or past the buffer.
* **`len(lens) != B`:** `_meta`'s `zip` truncates silently, and pass 2 then reads `rows[4*row..]` past the end of
  the table.
* **Unchecked shapes and dtypes:** `ks` / `vs` dtype, `ks[i].shape[3] == D`, and `HQ % HKV == 0` are not checked.
  `G = HQ // HKV` floors silently.
* **Kernel cache is keyed by name only:** MLX caches a custom kernel's compiled library by its name
  (`custom_kernel_<name>`), and the input dtypes are not part of that name. So a bf16 K/V sent after an fp16 call
  with the same name would bind to the `half` kernel and give garbage silently, not a compile error. Today the
  `q.dtype` check makes this unlikely, because all rows share the model's dtype. It is still a trap.

**Fix:** check `1 <= n <= cap` and the lengths inside `_meta`. It runs only on a cache miss, once per step; I
measured 8 µs (B=8 × 600), 61 µs (B=8 ragged 5–33K) and 373 µs (B=8 × 128K) per miss on the CPU. In `attention()`,
also check the `ks` / `vs` dtype and shapes.

### 5. LOW: the online softmax starts from `-INFINITY` and depends on `fast::exp2(-inf) == 0`

`m[g] = -INFINITY` (`rattn.py:76`), so the first key computes `f = fast::exp2(-inf - sc)` (`:89`).

* The `fast::` functions do not promise IEEE results for special values: the "safe" `math_mode` that keeps
  `exp(-inf) == 0` does not apply to an explicit `fast::exp2`.
* MLX's `sdpa_vector` starts from `Limits<U>::finite_min` for exactly this reason (`sdpa_vector.h:91, 255, 387`).
* It works on M1 Max: the 17 Metal tests pass and check for NaN. It may not hold on another GPU family or compiler.

**Fix:** start from `-FLT_MAX` (finite), or special-case the first key. Keep the empty-run guard at `:116`.

### 6. LOW: kernel names do not identify the source

MLX reuses a compiled library by name, so names have to be unique per source.

* Pass 1 (`rattn.py:162`) is named `..._{abs(hash(scale)) % 10**8}`. Two scales that collide modulo 10^8 would
  silently reuse each other's `SCALE2`.
* Pass 2 (`:166`) is named `rattn_p2_g{G}_h{HQ}`, but its source also carries `SCALE2` and `NS` through
  `_consts`. They are unused today, so this is harmless now. It becomes a stale-library bug as soon as pass 2 uses
  either of them.

**Fix:** append a short hash of the full source string to both names, for example `hashlib.sha1(src)[:12]`.

### 7. LOW (perf, not a bug): threadgroup memory limits occupancy

`o_sh[NS*G*D]` is fp32, which is 16 KB per 128-thread threadgroup at (NS 4, G 4) and at (2, 8). That limits how
many threadgroups can be resident per core, which matters for a kernel that is bound by memory bandwidth.

**Possible fix:** merge one query head at a time. Each simdgroup writes head g into an `NS*D` buffer (4 KB), with
a barrier, and simdgroup `g % NS` reduces it. Measure it with the "kernel headroom" item of the report.

### 8. LOW (measurement): the numbers are not as solid as the report presents them

* **In-step A/B:** `rattn_raw/p2_instep.log` has a single round (round 0, n = 70 per variant, alternating blocks
  of 10 steps). At B=8, the −2.3 ms is inside both interquartile ranges: base 62.6–67.3, rattn 59.6–66.2.
  CLAUDE.md asks for medians over at least 3 rounds. The no-go conclusion at B=8 does not depend on this, since
  even the ~7 ms ceiling is below the gate. But the B=16 −13.9 ms that motivates option 2 should be repeated over
  ≥ 3 rounds before anyone acts on it.
* **Microbench rounds:** mb1 and mb2 ran 2 rounds each, not 3.
* **Microbench settings not recorded:** `RB_EVAL_LAYER`, `RB_SWEEP` and `RB_VARS` are written to neither the
  logs nor the JSON (`microbench_rattn.py:149, 179`). That mb1 ran with `RB_EVAL_LAYER=0` (the "pipelined" table)
  can only be inferred from its +5 GB copy artifact.

**Fix:** dump the settings into the JSON and the first log line.

### 9. LOW (microbench bias, slightly against rattn)

In the `rattn` variant with appends (`microbench_rattn.py:78, 92`), one KVCache serves all 8 simulated layers, so
the lengths change at every layer. `_meta` is therefore rebuilt 8 times per step, against once per step in the
model.

I measured about 61 µs per rebuild at B=8 ragged, so about 0.5 ms per step at B=8 and about 1 ms at B=16 (two
launches). With mb2's sync per layer, this adds straight to the "append + rattn" column (21.6 / 39.3 ms), which is
about 2–3 %. The `rattn_k` columns are not affected, because their lengths are constant.

### 10. LOW (tool only): `step_prof.py` rough edges

* `attn_variant` (`step_prof.py:284-310`) takes the `rattn` / `nosdpa` path for any `RowsKV` cache without groups.
  With quantized KV (27B/35B, or a row below `quantized_kv_start`), `c.keys` is a tuple and the step crashes. Add
  an `isinstance(c, KVCache)` check, or document that these variants are 9B / `QWEN_KV_BITS=0` only.
* `--stagger` (`:557-559`) waits on `while not eng.sched.rows` with no exit. If the first request fails before it
  is admitted, the profiler hangs.

## Integration notes

None of these is a bug in the kernel, but each matters when it goes into `_rows_attention`.

* **What to pass:** pass `c.keys` / `c.values` after `update_and_fetch`, with `lens = c.offset`, which includes
  the new token, as `step_prof` and the microbench do. Passing the `[..., :L, :]` views would still be correct, but
  `metal_kernel`'s default `ensure_row_contiguous=True` would copy each row's whole KV at every layer and step. That
  is a silent performance cliff, and the Python API cannot check contiguity.
  * Every path that produces a live row cache yields contiguous buffers: KVCache growth (concatenate), the
    prompt-cache `restore` (`build` concatenates, `worker.py:990-1016`) and `cascade.release` (concatenates,
    `cascade.py:602-604`).
  * `_copy_cache` makes views (`worker.py:740`), but it has no callers. Even if it were called, `offset == cap`,
    so the first append regrows into a contiguous buffer.
* **Rows to gate on:** gate per row on a plain `KVCache` with fp16 dtype.
  * With `QWEN_KV_BITS=8/4`, a batch can mix fp16 rows (below `quantized_kv_start` = 5000) with quantized rows.
  * Grouped rows keep `_cascade_rows`. Only `plan.single` could go through rattn, with the existing `inv` take.
  * Keep B ≤ 2 on the per-row path, which is bit-exact today.
  * DFlash (T > 1) never reaches `RowsKV`. `attention()` raises on `q.shape[2] != 1` anyway.
* **Compile stall:** the first use of each launch size n compiles two Metal libraries on the scheduler thread in
  the middle of decode. The report already plans a warm-up at load; it needs n = 1..8, or 1..13 after finding 3.

## Suspects checked and found fine

* **Chunk and row boundaries:**
  * `range(0, n, chunk)` never produces an empty item. On the CPU, the table is right for [1, 2, 3],
    [511, 512, 513, 1541] and [1024].
  * The per-simdgroup split `sub = ceil(len / NS)` with `s0` and `s1` clamped to `end` gives empty runs for short
    items. Their pointer is computed but never dereferenced.
  * Lengths below 512, non-multiples of 512 and length 1 are all handled.
* **Addressing:**
  * The head stride uses `cap` from the row table (`kvh*cap + t`), not `len`, so buffers with spare capacity
    are right.
  * Offsets are computed in `size_t`.
  * Lane loads are 8 halves (two `half4`) at `lid*8`, inside D = 256 and 16-byte aligned. The `q[i:i+8]` slices
    for the split are row-contiguous with an offset: no copy and no misalignment.
* **GQA mapping:** query head `kvh*G + g` reads kv head `kvh`. That is MLX's `kv_head = q_head / gqa_factor`
  (`sdpa_vector.h:63`).
* **Softmax:**
  * `q` is pre-scaled once by `scale·log2 e` in fp32. MLX also scales q in fp32.
  * The online update runs in the log2 domain.
  * The merge inside pass 1 weights empty runs by 0 (`:116`).
  * Every non-empty item has `l ≥ 1`, so `1/L` is safe.
  * Pass 2 weights the normalized partials by `l_c·2^(m_c−M)`, which is the exact log-sum-exp merge. It is the
    same math as cascade.py's (max, sum-exp) merge.
  * fp16 rounding happens once, at the output, which is the same point as MLX's `2pass_2`
    (`o / sum → T`). MLX keeps its partials in T (fp16), so the report's precision claim holds.
* **Buffers:** 21 at n = 8, within 31 (see finding 3 for the real maximum).
* **Grid and threadgroups:**
  * Pass 1: the grid is in threads, `32·ns·n_items × HKV`, with a threadgroup of `32·ns`, so one threadgroup per
    (item, kv head). The `NS` constant equals the launched threadgroup size, since both come from the same `ns`.
  * The barrier is unconditional.
  * `simd_sum` runs with the same trip count in every lane of a simdgroup.
  * Pass 2: grid `D·B × HQ`, threadgroup `D`.
  * The `ns` clamp keeps threadgroup memory ≤ 16 KB for G ≤ 16. With G = 6 (the 27B layout), ns = 2 and the
    `g += NS` merge loop covers every head.
* **No per-step recompilation:**
  * No `mx.compile` is involved.
  * Libraries are keyed by (n, G, HQ, scale, ns).
  * Lengths and capacities reach the kernel through buffers.
  * `_meta` is cached per (lens, caps) and shared by the 8 FA layers of a step. Capacities are re-read from the
    real shapes at every call, so layers with different capacities would still be right.
  * Host cost in the real step: `fwd_cpu` goes from 2.73 to 2.86 ms at B=8 and from 3.38 to 3.57 ms at B=16
    (`p2_instep.log`).
* **Append order:** `step_prof` and the microbench append before attending, and `lens = offset` includes the
  new token, which is what SDPA on the fetched view does.
* **Tests:**
  * The references are right: an fp32 softmax, and MLX's per-row fast SDPA on the same fp16 views, which is the
    worker's call for a KVCache.
  * Garbage K = 1000 and V = NaN past each row's length, with a spare capacity of 37, would catch over-reads and
    head-stride errors.
  * Lengths: 1, 2, 3, C−1, C, C+1, and non-multiples of C.
  * Batch sizes: 3, 4, 8, 11 (8 + 3) and 16 (8 + 8). GQA 4 and 8. Flat and peaky q.
  * The tolerance `max(1.05·e_mlx, 4e-4)` is tight enough: one key dropped at L = 5000 under flat q moves the
    output by about 1/√L ≈ 1.4e-2 relative.
  * Gaps (LOW): B = 1 and 2, and a 1-row tail launch (B = 9). G = 6 (Hq 24 / Hkv 4). Real `KVCache` objects after
    `update_and_fetch`; the microbench's numerics cover that last one.
* **Microbench method:**
  * It syncs once per step (`mx.eval` per step), with 2 warm-up steps per variant and round, after the numerics
    pass has compiled the kernels.
  * The variant order is reversed every round.
  * The rows grow by at most ~1.9K tokens per configuration, which stays within the 4096-token reserve, so no
    buffer regrowth happens inside the timed steps.
  * `rows_k` (views) against `rattn_k` (whole buffers) compares like with like.
