# Review: pfqmm mixed last tile (`…_mix`, `_use` = M ≥ 64 ∧ MLX qmm_t)

Date 2026-10-17. Scope: `git diff -- src/worker/pfqmm.py src/worker/tests/test_pfqmm.py` (uncommitted),
`scripts/perf/microbench/microbench_pfqmm_m.py`, `scripts/perf/prof/ab_pfqmm_mix.py`,
`docs/perf/2026-10-17/pfqmm_short_slabs.md` + `raw/pfqmm_m_*`. CPU only: no worker, no Metal; only
`tests/test_pfqmm.py` without `PFQMM_GPU` was run (29 passed, 19 skipped). The MLX dispatch was read from
the saved mlx 0.32.3 `mlx/backend/metal/quantized.cpp` (scratchpad `src_mlx/v0.32.3/q/quantized.cpp`)
and the wheel's `kernels/quantized.h`.

**Verdict: no HIGH or MED bug.** The kernel change is correct for every M % 64 class, the 64-row path's
bits are unchanged, and `_mlx_uses_qmm` matches MLX's qmv / split-K / qmm_t dispatch exactly for g64
at any bit width and for the 9B, 27B and 35B shapes. The findings below are about test coverage,
reproducibility and docs.

## Findings

1. **LOW — the GPU bit-identity test exercises the mixed tile on one shape only, and never the new
   M = 64…127 range.** `tests/test_pfqmm.py:261-276`. MLX runs qmm_t from M = 257 on (4096, 1024) and
   from M = 385 on (5120, 640). So on (4096, 1024) only 288 (a full 32-row last tile), 640 and 2048
   run. On (5120, 640) only 640 and 2048 run, both full 64-row tiles: no mixed or ragged tile at all.
   M = 64 is in the list but is skipped on all three shapes, because all three run split-K there.
   Every last-tile kind × every bit width runs only through (4096, 4096). The production row counts
   the patch newly enables (M = 64…127 on gate/up, qkv/q and lm_head, which run qmm_t from 32–33 rows)
   are never compared on the GPU. The code paths are the same as at 128–191, so the risk is small.
   But `pfqmm_short_slabs.md:144` says the test "checks … M = 64, 96, …", and on the GPU M = 64 never
   runs. Fix: add a shape with N ≥ 8192, e.g. (4096, 12288), where MLX runs qmm_t from 32 rows. For
   each (bits, shape), assert how many M were actually compared, so a skipped list cannot pass
   silently. For (5120, 640), add tails past 384, e.g. 392 / 416 / 424.

2. **LOW — `test_use_every_qmm_row_count` restates `_use`'s body. It cannot catch a wrong
   `_mlx_uses_qmm`.** `tests/test_pfqmm.py:50-61`. The test computes
   `want = M >= MIN_ROWS and _mlx_uses_qmm(...)`, which is the same expression as `pfqmm.py:245`. It
   guards against a padding rule coming back (useful). The docstring says it checks that every
   qmm_t row count takes the kernel, but the mirror itself is never checked against MLX in tests. The
   only empirical check is the microbench `--check` lines in `raw/pfqmm_m_*.log`. I checked them all:
   every row predicted qmm_t is bit-identical, and every row predicted split-K differs. Fix (GPU): pin
   the boundary on real MLX. For (4096, 1024), assert `pfqmm.pfqmm == quantized_matmul` at M = 257 /
   264 and `!=` at M = 256. Do the same for (4096, 4096) at 65 / 64. An mlx bump that moves a
   threshold then fails the test instead of only the load self-test, which checks one module at
   ≥ 256 rows.

3. **LOW — the A/B script's explanation of the "reference from round 1" workaround is imprecise.
   The workaround does not hide a kernel difference.** `scripts/perf/prof/ab_pfqmm_mix.py:5-6, 78-80`;
   `pfqmm_short_slabs.md:157-159`. mlx-vlm 0.7.6 `qwen3_5/language.py:1650-1680`: with
   `position_ids=None` and `self._rope_deltas is None`, the call takes the `get_rope_index(inputs)`
   branch even at cache offset 12288. That rotates the slab as positions 0…L−1 and then sets
   `_rope_deltas = 0`. Every later call takes the `cache_offset + delta` branch, which is correct, at
   any slab length. So only the **first call in the process** is wrong, not "the first call at a new
   slab length". This fits the data. In `raw/pfqmm_m_ab_slab.log` the only offset is 12288, and the
   process's first call was mlx's round 0 at slab 416. Only the 416 reference was bad, and old and new
   both "DIFFER" from it. In `ab_slab2` offset 0 runs first, so the artefact could not occur at all:
   all 24 states are bit-identical. This is the same trap CLAUDE.md describes ("On a hit set
   `language_model._rope_deltas` to zeros"). Fix: before every call set
   `lm._rope_deltas = mx.zeros((1, 1), mx.int32)` and `lm._position_ids = None`, as the worker does,
   or pass `position_ids`. Then compare every round, round 0 included. Also correct the comment and the
   doc sentence. Related gap: the in-model check compares cache states only (`skip_logits=True`). The
   last layer's MLP (gate/up/down) and the final norm feed no cache, so their bits are not checked in
   the model, only in the microbench. Comparing the returned hidden state (`return_hidden`) would
   close this.

4. **LOW — the microbench and its aggregation are stale or not in the repo.**
   `scripts/perf/microbench/microbench_pfqmm_m.py:58` still labels `"pf64"` "pfqmm today". Lines
   315-316 hard-code the **old** rule (`M >= 128 and ceil64 <= 1.06·ceil32`) as `pfqmm_use`. But since
   the patch, the `pfqmm` variant runs the mixed kernel. A re-run aggregated with `summ.py` would
   charge the new kernel at MLX's time wherever the old rule said no, and report "pfqmm" ≈ old. The
   existing tables were not affected: sweep1/2/3 and small ran before the patch, since `pfqmm.py.orig`
   (= HEAD) was saved at 06:02:44, after sweep3 at 06:00:54 and small at 06:01:59. The swizzle run
   (06:15) used the mixed kernel only at M where the old rule also said yes. The slab-weighted
   aggregation behind every table in the doc lives only in the scratchpad (`summ.py`), and so does the
   copy of `quantized.cpp` the dispatch claims rest on. Neither can be reproduced from the repo. Fix:
   compute `pfqmm_use` with `pfqmm._use` (or record both rules). Commit `summ.py` next to the
   microbench, or as `raw/pfqmm_m_summ.py`. Cite the upstream file and tag (mlx v0.32.3
   `mlx/backend/metal/quantized.cpp`: `get_qmv_batch_limit` l.89, `qmm` l.1042, `qmm_splitk`
   l.1152-1200, `QuantizedMatmul::eval_gpu` l.1868) or save the excerpt in `raw/`.

5. **LOW — the doc's mix table mixes runs, and the low-M gains are noisier than stated.**
   `pfqmm_short_slabs.md:17, 104-125`. The text says rows 32–112 come from `pfqmm_m_small`. But the
   M = 64 (145.1 / 134.3 ms) and M = 96 (215.0 / 194.1, ×1.107) rows are from `sweep3`. `small` gives
   64 ×1.084, **96 ×1.060** and 128 ×0.991 vs old. That explains the "×1.06–1.19" in Results against
   the table's ×1.107. Where old and mix run the same code, the per-shape cells spread 0.93–1.05
   (sweep1 pfqmm / pf64) and 0.95–1.03 (sweep3). So the +3.4–5.5 % at 544 / 800 rests on the
   slab-weighted sum of one 7-round run. Only 600 was confirmed in the model (×1.043). The rest is
   in line with what I recomputed from the raw JSON: 160 ×1.144, 288 ×1.194, 400 ×1.169, 416 ×1.179,
   432 ×0.998, 2048 ×0.986. Fix: label each row with its run, and state the range as ×1.06–1.19 with
   the reason.

6. **LOW — stale "≥ 128 rows" text elsewhere.** `CLAUDE.md:150` ("taken only where MLX runs qmm_t,
   ≥128 rows"), `CLAUDE.md:739` (the `QWEN_PFQMM` row) and `src/worker/worker.py:758` (">= 128 rows").
   Update them to "≥ 64 rows, mixed 32-row last tile" when committing. Note also that the :50051 worker
   was restarted at 06:16 on the uncommitted patch: `worker_pfqmm_restart.log` shows
   `tested … M=256/264/288/296`, `>= 64 rows`.

## Checked and fine

- **Grid with a mixed last row.** `pfqmm.py:202`: `grid=(N/32·64, ceil(M/64)·2)`, `threadgroup=(64,2)`,
  so N/32 × ceil(M/64) threadgroups, unchanged. With r = M − y_row of the last tile ∈ (0, 64]:
  - r = 64 (M % 64 == 0): `r <= 32` is false, so the 64 path runs with `num_els = 64`: unsafe loads,
    full store, as before.
  - r = 32 (M % 64 == 32): the 32 path runs with `num_els = 32 = BM`: unsafe loads of rows
    y_row…y_row+31, all < M, and `store_result`.
  - r ∈ (0, 32): the 32 path runs `load_safe` / `store_result_safe(short2(BN, num_els))`.
  - r ∈ (32, 64): the 64 path runs the ragged path as before.

  Only the last tile row can have r ≤ 32. The tile's rows 32…63 are ≥ M, so nothing is dropped. No
  off-by-one. If M < 64 were ever passed (`_use` excludes it), M ≤ 32 → one 32-row tile and
  33…63 → one ragged 64-row tile, both correct.
- **Tile origin and loop bounds in the 32 path.** `y_row` is a multiple of 64 (it is also a 32-row
  tile origin). x, w, scales, biases and y are offset exactly as MLX's `qmm_t_impl`
  (`quantized.h:1238-1250`). The K loop is `k < K` with `K_eff = K`, as MLX's `qmm_t` wrapper passes
  it (`quantized.h:1949-1963`). The `BM/2` template is `BlockMMA<T,T,32,32,32,2,2,false,true,40,40>`
  plus `BlockLoader<T,32,32,40,1,128>` and `QuantizedBlockLoader<T,32,32,40,1,128,gs,bits>`: the same
  types as MLX's 32×32×32 qmm_t.
- **Template flags.** MLX has no `aligned_M` template parameter. The M tail is the runtime
  `num_els < BM` branch, which pfqmm copies. `aligned_N`: MLX picks `alN_true` for N % 32 == 0
  (`quantized.cpp:1096`), and `supported()` requires N % 32 == 0. With `aligned_N` true, MLX's
  `!aligned_N && num_outs < BN` branches are dead and `store_result_safe(short2(num_outs = BN, …))`
  equals pfqmm's `short2(BN, num_els)`.
- **Accumulation order and bits.** BM only changes TM (fragments per simdgroup) and which thread loads
  what. Each output element still accumulates the same K/32 tiles × 4 MMAs of 8 in the same order, in
  the fp32 `BlockMMA` accumulators, rounded to fp16 once at store. Both paths use 4 simdgroups / 128
  threads, and MLX's (32, 2, 2) vs pfqmm's (64, 2, 1) layout only renumbers simdgroups. The branch is
  uniform per threadgroup, so the barriers inside both arms are safe. `Xs` / `Ws` are sized for
  BM = 64, and the 32 path uses a prefix of them.
- **64-row path unchanged.** The only edit is that `y_row` / `y_col` are computed in the kernel body
  (`int(tgid.y)·64`) instead of `tid.y·BM` inside the function: same values. The body is otherwise
  identical to HEAD (`pfqmm.py.orig` == HEAD). The new kernel name also avoids `metal_kernel`'s
  by-name cache serving the old source in a process that built the old one.
- **`_mlx_uses_qmm` vs mlx 0.32.3.** The source path, line by line:
  - `eval_gpu` (l.1868+): non-batched M = x.size/K. qmv applies only below
    `get_qmv_batch_limit` (≤ 33 on every architecture branch, l.89-135). From there, transpose with
    B == 1 always goes to `qmm_splitk`.
  - `qmm_splitk` (l.1152-1200): tiles = ⌈N/32⌉·⌈M/32⌉, `split = max(1, 512/tiles)`, capped at
    K / k_align with `k_align = max(gs, 32)` = 64 for g64, then decremented while
    K % (split·64) ≠ 0. It runs `qmm` when split ≤ 1.
  - `qmm` (l.1042): the NAX branch is off on M1, then qmm_t 32×32 runs.

  The mirror is a literal transcription, independent of the bit width (bits never enter it), so it
  holds for 3/4/5/6/8. It is monotone in M: checked for M = 32…5000 on all shapes below, so the
  extra self-test tails never change `_test_rows`' pick. Thresholds (first qmm_t M):

  | model | shape (K → N) | first qmm_t M |
  |---|---|---:|
  | 9B | gate/up 4096 → 12288 | 32 |
  | 9B | qkv/q 4096 → 8192 | 33 |
  | 9B | z/out/o 4096 → 4096 | 65 |
  | 9B | down 12288 → 4096 | 65 |
  | 9B | k/v 4096 → 1024 | 257 |
  | 9B | b/a 4096 → 32 | never |
  | 9B | lm_head 4096 → 248320 | 32 |
  | 27B | 5120 → 17408 | 32 |
  | 27B | 17408 → 5120 | 33 |
  | 27B | 5120 → 1024 | 257 |
  | 35B | router 2048 → 256 | 1025 |
  | 35B | 2048 → 4096 | 65 |

  Empirically, every `check` line in `raw/pfqmm_m_{sweep1,sweep2,sweep3,small}.log` agrees with the
  prediction in both directions.
- **Load self-test.** `TEST_TAILS = (0, 8, 32, 40)` covers all four code variants on the first
  qualifying module: full 64 tiles, a 32-path ragged tile (`load_safe`), a 32-path full tile
  (`load_unsafe`) and a 64-path ragged tile. The kernel is instantiated per (bits, gs) only (K, N are
  runtime), so one module per key is enough. "untested" and "failed" keys still stay off: install's
  logic is unchanged, and `test_install_untested_key_stays_off` and `test_install_per_key` pass. Cost
  on the 9B (lm_head, 256…296 rows): 4 × (ours + reference) GEMMs of ≤ 0.6 TFLOP, i.e. < 1 s, plus up
  to ~150 MB transient output each, freed by the `clear_cache()` after install. The 35B router case
  tests 1088 / 1096 / 1120 / 1128, all qmm_t.
- **CPU dispatch tests.** `test_use_every_qmm_row_count` really enumerates M = 1…1199 × 6 shapes
  through `_use`, which is what `_call` (`pfqmm.py:252`) dispatches on. The `test_use_rows` cases are
  right: (K = 256, N = 4096) at M = 64 has 256 tiles, so split 2 → split-K, False; at 96, 384 tiles →
  qmm, True.
- **GPU tests listed (read, not run).** On (4096, 4096) every tail kind runs: 1 / 8 / 31 / 32 rows on
  the 32 path and 33 / 40 / 63 on the 64 path. `test_tiny_model_prefill` at 2112 / 2120 / 2152 gives
  full / 8-row 32-path / 40-row 64-path last tiles (2120 is a ragged 32-path tile, not "a 32-row" full
  one). The router test expects the 4 tails.
- **Other callers are still excluded.** DFlash verify (1, T ≤ 16, K) and batched decode (B, 1, K)
  (`M != shape[-2]`) are never taken.
- **Microbench method.** Each round times every (shape, M, variant) with the variant order rotated per
  round (`microbench_pfqmm_m.py:287-289`). Reported values are medians over rounds (7 / 9 / 15), as
  stated. The A/B alternates variants with the order reversed every other round, discards round 0,
  and takes medians of 6. The A/B's "old" is the microbench `pf64` ((128,1,1) threadgroups, early
  return), not the literal previous kernel. It has the same per-element math, and sweep1 shows
  pfqmm / pf64 at a median of 1.006 on the same M, so it is a fair stand-in.
