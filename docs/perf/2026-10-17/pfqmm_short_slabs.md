# pfqmm on short prefill slabs (M = 64…1024), 9B 8-bit

Date 2026-10-17, M1 Max 32 GB, mlx 0.32.3. GPU lease held, the :50051 worker stopped for the runs
(restarted afterwards). Question: can the 8-bit prefill GEMMs of short slabs (the agentic load's
~400–860-token suffixes) run faster and stay bit-identical to `mx.quantized_matmul`?

**Verdict: yes for half of all short row counts, no for the rest.** The rate per row is not the
problem: for M a multiple of 64, pfqmm's 64-row tile runs as fast at M = 128 as at M = 2048. The
loss was the dispatch. The old rule "64-row padding ≤ 1.06 × the 32-row padding, M ≥ 128" sent every
M in (64k, 64k + 32] below ~512 rows back to `quantized_matmul`, which is half of all row counts
there, and M < 128 always went back. The fix is a **mixed last tile**: the last row of threadgroups
runs MLX's own 32-row tile when ≤ 32 rows are left. The padding is then never larger than MLX's, so
the kernel takes every M ≥ 64 where MLX runs qmm_t. It is bit-identical (the k order per output
element is unchanged; checked on the real weights and in tests).

Results:
- GEMM time per slab (slab-weighted) at M = 64…416 that the old rule left on MLX: **×1.06–1.19**.
  The low end is M = 96 in `small` (×1.060; `sweep3` gave ×1.107 for the same M). At 160–416 it is
  ×1.14–1.19.
- M = 544 / 600 / 800 (old kernel with a half-empty last tile): +3.4–5.5 % slab-weighted, from one
  7-round run. That is within the per-cell noise: where old and mix run the same code, single
  (shape, M) cells spread ×0.93–1.05. Only 600 is confirmed in the model (×1.043).
- Multiples of 64, or a remainder > 32 (e.g. 432, 512, 2048): unchanged, same code path.
- In-model slab A/B, against the old pfqmm, over three runs: **416 tokens ×1.07–1.25, 288 tokens
  ×1.055–1.22, 600 tokens ×1.02–1.04**. 432 and 2048 are unchanged. Cache states and the final
  hidden state are bit-identical on every call.

The other tile and raster candidates found nothing that wins ≥ 5 % at M ≤ 512 over the 64×32×32
tile. Details below.

The machine ran hot during this study: quantized_matmul at ~6.2 TFLOPS against ~8.1 when cool, and
the agentic runs see the same throttling. Ratios come from interleaved runs inside one process.
Compare absolute TFLOPS within a table only.

## Dispatch facts (mlx 0.32.3 source, `mlx/backend/metal/quantized.cpp`)

Source: https://raw.githubusercontent.com/ml-explore/mlx/v0.32.3/mlx/backend/metal/quantized.cpp
(sha256 `36b79607…5982`). The excerpts used (`get_qmv_batch_limit` l.89, `qmm` l.1042,
`qmm_splitk` l.1152, `QuantizedMatmul::eval_gpu` l.1868) are in
`raw/pfqmm_m_mlx_quantized_cpp_excerpt.txt`.

- `QuantizedMatmul::eval_gpu` runs qmv below `get_qmv_batch_limit` (≤ 33 rows on any device). From
  there, a non-batched transposed call goes to `qmm_splitk`: `split = min(max(1, 512 / tiles32),
  K / 64)`, decreased until K % (split × 64) == 0. When split ≤ 1 it runs `qmm` (qmm_t, BM = BN =
  BK = 32, 2×2 simdgroups). On M1 there is no NAX path. `pfqmm._mlx_uses_qmm` mirrors this exactly:
  every variant was bit-identical wherever it predicts qmm_t and differed wherever it predicts
  split-K (checked on all shapes × all M below).
- 9B shapes at short M: k/v (N = 1024) runs split-K up to M = 256. z / out / o (N = 4096) and down
  run split-K at M ≤ 64. gate/up, qkv/q and lm_head run qmm_t from 32 / 33 / 32 rows. The GDN b/a (N = 32) always
  run split-K. Split-K calls stay on MLX: their fp16 partial sums plus a strided reduce would have to
  be copied bit for bit, for ≤ 3 ms of k/v per slab.
- The old pfqmm was taken at M ≥ 128 where MLX runs qmm_t and `ceil64(M) ≤ 1.06 · ceil32(M)`. For
  k < 8 that excludes all M = 64k + r with r ∈ (0, 32] (e.g. 160, 288, 400, 416, 480). Those calls
  ran `quantized_matmul` at its 32×32 rate.

## Microbench (`scripts/perf/microbench/microbench_pfqmm_m.py`)

Real 9B weights (Ornith-1.5 8-bit, fp16 scales), cycling 3–6 layers' copies per shape. The variants
are MLX's qmm_t code (`QuantizedBlockLoader` + steel `BlockMMA` from the wheel's headers) with other
BM×BK×BN, WM×WN or a steel swizzle (`sN` = 2^N M-tiles per N column), and `mix` = the new kernel.
Each round times every (shape, M, variant) in a rotated order. Values are medians of 7 rounds (15
for the swizzle check). "Slab-weighted" = Σ over the shapes of calls per 32-layer slab × time
(gate/up 64, down 32, qkv/q 32, z/out/o 56, k/v 16). A call where MLX runs split-K is charged at
MLX's time for every variant, and `pfqmm` is charged at MLX's time where its `_use` said no.

### Sweep 1: tile candidates, slab-weighted GEMM speedup vs quantized_matmul (`raw/pfqmm_m_sweep1.*`)

| variant | M=64 | 128 | 256 | 432 | 512 | 768 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|---|
| pfqmm (64×32×32, 2×2; old rule) | 1.000 | 1.188 | 1.210 | 1.179 | 1.203 | 1.180 | 1.182 | 1.226 |
| 32×32×32 via metal_kernel (= MLX's tile) | 0.996 | 1.001 | 1.002 | 0.981 | 1.004 | 0.979 | 0.969 | 0.969 |
| 32×32×64 | 0.993 | 0.967 | 0.993 | 0.942 | 0.966 | 0.942 | 0.944 | 1.022 |
| 32×32×128 | 0.975 | 0.910 | 0.936 | 0.903 | 0.907 | 0.899 | 0.875 | 0.991 |
| 32×64×32 (BK 64) | 1.067 | 1.141 | 1.186 | 1.124 | 1.133 | 1.093 | 1.063 | 1.092 |
| 64×64×32 (BK 64) | 1.053 | 1.099 | 1.133 | 1.119 | 1.128 | 1.109 | 1.116 | 1.243 |
| 64×32×64 | 1.087 | 1.161 | 1.199 | 1.159 | 1.197 | 1.170 | 1.176 | 1.325 |
| 64×32×32, 4×1 simdgroups | 1.099 | 1.171 | 1.205 | 1.165 | 1.196 | 1.143 | 1.167 | 1.310 |
| 64×32×32, 1×4 | 1.088 | 1.154 | 1.163 | 1.134 | 1.172 | 1.117 | 1.130 | 1.295 |
| 128×32×32, 4×2 (256 threads) | 0.688 | 1.162 | 1.194 | 1.023 | 1.193 | 1.177 | 1.232 | 1.356 |
| 64×32×64, 2×4 (256 threads) | 1.124 | 1.190 | 1.220 | 1.201 | 1.222 | 1.200 | 1.235 | 1.381 |
| 64×32×32 swizzle s1 | 1.134 | 1.257 | 1.281 | 1.217 | 1.271 | 1.244 | 1.279 | 1.388 |
| 64×32×32 swizzle s2 | 1.125 | 1.191 | 1.246 | 1.192 | 1.236 | 1.226 | 1.259 | 1.371 |
| 32×32×32, 4×1 | 1.026 | 0.999 | 0.971 | 0.975 | 0.974 | 0.957 | 0.999 | 1.015 |

All variants were bit-identical wherever MLX runs
qmm_t (`check` lines in the log). Per-row rate: for every shape, pfqmm runs at the same TFLOPS from
M = 128 to 2048 (6.9–7.4 TF in this hot state). Short slabs are not slower per row on full tiles.

- Smaller tiles (32-row, BN 64/128) lose: x is re-read per N-tile and W is dequantized per 32 rows.
- BK 64 loses (twice the threadgroup memory, lower occupancy).
- 128-row tiles lose at short or ragged M (432: ×1.02, 64: ×0.69).
- 8-simdgroup tiles and swizzles looked +2–6 % over pfqmm here. A dedicated re-check (15 rounds,
  below) shrank that to noise.

### Swizzle re-check (15 rounds, `raw/pfqmm_m_swizzle.*`), slab-weighted, vs pfqmm

| M | 256 | 432 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| 64×32×32 s1 | 1.010 | 1.002 | 1.007 | 1.010 | 1.007 |
| mix s1 | 1.011 | 0.994 | 1.006 | 1.008 | 1.001 |
| 64×32×64, 2×4, s1 | 0.946 | 0.938 | 0.946 | 0.951 | 0.948 |

Swizzle: ≤ 1 %, rejected. Sweep 2 (`raw/pfqmm_m_sweep2.*`: s1/s2/s3, 64×32×64 s1, 32×32×32 s1–s3,
128×32×32 s1, M = 64…2048 + 288) agrees: the best full-tile candidate is within +0–2.9 % of pfqmm at
every M ≥ 128. The only real gap is at M = 288 (+4–7 % for any swizzled 32-row tile vs MLX), because
the old rule left that M on MLX.

### The mixed last tile (`mix`; `raw/pfqmm_m_sweep3.*`, `raw/pfqmm_m_small.*`)

Slab-weighted GEMM ms per 32-layer slab. "old" = pfqmm as dispatched before (MLX where its rule said
no). gate/up TFLOPS columns are mlx / old / mix.

| M | M % 64 | old rule took it | run | mlx ms | old ms | mix ms | mix vs mlx | mix vs old | gate/up TF |
|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| 32 | 32 | no (MLX) | small | 69.9 | 69.9 | 71.4 | ×0.979 | ×0.979 | 6.80 / 6.80 / 6.48 |
| 48 | 48 | no (MLX) | small | 139.7 | 139.7 | 133.5 | ×1.047 | ×1.047 | 5.02 / 5.02 / 5.50 |
| 64 | 0 | no (MLX) | sweep3 | 145.1 | 145.1 | 134.3 | ×1.080 | **×1.080** | 6.38 / 6.38 / 7.21 |
| 64 | 0 | no (MLX) | small | 133.4 | 133.4 | 123.0 | ×1.084 | **×1.084** | 6.78 / 6.78 / 7.79 |
| 80 | 16 | no (MLX) | small | 192.1 | 192.1 | 191.1 | ×1.005 | ×1.005 | 5.74 / 5.74 / 6.03 |
| 96 | 32 | no (MLX) | sweep3 | 215.0 | 215.0 | 194.1 | ×1.107 | **×1.107** | 6.23 / 6.23 / 7.05 |
| 96 | 32 | no (MLX) | small | 194.7 | 194.7 | 183.8 | ×1.060 | **×1.060** | 6.96 / 6.96 / 7.58 |
| 112 | 48 | no (MLX) | small | 269.6 | 269.6 | 242.9 | ×1.110 | **×1.110** | 6.03 / 6.03 / 6.75 |
| 128 | 0 | yes | sweep3 | 288.2 | 240.0 | 235.9 | ×1.222 | ×1.017 | 6.45 / 7.59 / 7.80 |
| 128 | 0 | yes | small | 258.6 | 230.3 | 232.4 | ×1.113 | ×0.991 | 7.01 / 7.95 / 7.89 |
| 160 | 32 | no (MLX) | sweep3 | 353.4 | 353.4 | 308.8 | ×1.144 | **×1.144** | 6.48 / 6.48 / 7.43 |
| 288 | 32 | no (MLX) | sweep3 | 654.2 | 654.2 | 548.1 | ×1.194 | **×1.194** | 6.38 / 6.38 / 7.31 |
| 400 | 16 | no (MLX) | sweep3 | 920.4 | 920.4 | 787.2 | ×1.169 | **×1.169** | 6.11 / 6.11 / 7.25 |
| 416 | 32 | no (MLX) | sweep3 | 921.7 | 921.7 | 781.6 | ×1.179 | **×1.179** | 6.31 / 6.31 / 7.50 |
| 432 | 48 | yes | sweep3 | 1006.7 | 837.7 | 839.2 | ×1.200 | ×0.998 | 6.17 / 7.28 / 7.22 |
| 512 | 0 | yes | sweep3 | 1143.1 | 945.3 | 941.0 | ×1.215 | ×1.005 | 6.47 / 7.60 / 7.70 |
| 544 | 32 | yes | sweep3 | 1191.3 | 1075.6 | 1019.6 | ×1.168 | ×1.055 | 6.44 / 6.99 / 7.51 |
| 600 | 24 | yes | sweep3 | 1402.3 | 1194.1 | 1151.4 | ×1.218 | ×1.037 | 6.35 / 6.96 / 7.34 |
| 800 | 32 | yes | sweep3 | 1777.7 | 1548.4 | 1497.5 | ×1.187 | ×1.034 | 6.40 / 7.14 / 7.50 |
| 2048 | 0 | yes | sweep3 | 4756.2 | 3832.2 | 3885.3 | ×1.224 | ×0.986 | 6.20 / 7.51 / 7.36 |

Runs: `sweep3` = `raw/pfqmm_m_sweep3.*` (7 rounds); `small` = `raw/pfqmm_m_small.*` (9 rounds, a
few minutes later and cooler). M = 64 / 96 / 128 were in both, and both values are shown. Compare
absolute ms only within one run. Noise: where old and mix run the same code (multiples of 64, or a
remainder > 32), the slab-weighted sums agree within ±1.5 % (128: ×1.017 / ×0.991, 432 ×0.998, 512
×1.005, 2048 ×0.986). Single (shape, M) cells spread ×0.93–1.05. Gains of a few percent from one run
(544 / 600 / 800) are therefore indicative only. Below 64 rows the gain is ≤ 5 % (32 rows loses,
since it is MLX's own tile plus a metal_kernel launch), so the floor is `MIN_ROWS = 64`. If row
counts are spread evenly, the expected GEMM saving is ~7 % for slabs of 128–512 rows (half of them
gain ~15 %) and ~2 % for 512–1024. `scripts/perf/microbench/summ_pfqmm_m.py` recomputes every
table from the JSON.

## Change (`src/worker/pfqmm.py`, `tests/test_pfqmm.py`; not committed)

- Kernel: `pf_qmm_t` now takes the tile origin. The kernel source runs `pf_qmm_t<BM/2>` (MLX's
  32×32×32, 2×2 tile: TM = TN = 2) for the last row of threadgroups when `M − y_row ≤ 32`, and
  `pf_qmm_t<64>` otherwise. The grid, threadgroup size and threadgroup memory are unchanged. The
  kernel is renamed `…_mix`.
- `_use`: `M ≥ MIN_ROWS (64)` and MLX runs qmm_t. The 1.06 padding rule is gone.
- Load self-test (`_selftest` / `_test_rows`): M, M + 8, M + 32, M + 40 (`TEST_TAILS`) cover full
  64-row tiles, a ragged and a full last 32-row tile, and a ragged last 64-row tile. All four must be
  qmm_t rows in MLX. 9B: `pfqmm 8-bit g64: tested, 249 modules on (lm_head K=4096 N=248320
  M=256/264/288/296)`.
- CPU tests: new `_use` cases (32/63 below the floor, 64 split-K, 96/200/288/300 now taken).
  `test_use_every_qmm_row_count` checks `_use == (M ≥ 64 and MLX qmm_t)` for M = 1…1199 on all 9B
  shapes. `_test_rows` now checks all tails.
- GPU tests (`PFQMM_GPU=1`):
  - `test_kernel_bit_identical`: bits 3/4/5/6/8 × (4096, 4096), (4096, 1024), (5120, 640), (4096,
    12288) at M = 64, 72, 96, 100, 127, 128, 129, 159, 160, 161, 191, 200, 232, 256, 288, 392, 416,
    424, 448, 640, 2048. Every shape covers every last-tile kind. The test asserts the number of M
    actually compared per shape: 20 / 7 / 6 / 21, so a list that MLX's split-K skips cannot pass
    silently. (4096, 12288) covers M = 64…127.
  - `test_mlx_dispatch_boundary`: `_mlx_uses_qmm` pinned on real MLX. The kernel equals
    quantized_matmul at 257 / 264 and differs at 256 on (4096, 1024); likewise at 65 / 72 vs 64 on
    (4096, 4096).
  - `test_tiny_model_prefill` at 2112 / 2120 / 2152 rows: full tiles / a ragged 32-row last tile /
    a ragged 64-row one.
  - The 35B-router self-test expects the 4 tails.
- Verification (after the review fixes): `PFQMM_GPU=1 pytest tests/test_pfqmm.py` 55 passed. CPU
  suite (`--ignore test_mrqmm/test_pfqmm/test_rattn`) 279 passed, 2 skipped. `tests/test_pfqmm.py`
  on CPU: 29 passed, 26 skipped.
- Review: `docs/perf/2026-10-17/reviews/pfqmm_mix.md` (no HIGH/MED, 6 LOW, all addressed).

## In-model slab A/B (`scripts/perf/prof/ab_pfqmm_mix.py`)

One worker slab (`lm(..., skip_logits=True, return_hidden=True)` + eval) on a synthetic cache. The
variants are mlx / old (the previous kernel and rule) / new, alternated with the order reversed per
round: 6 timed rounds after a warm-up, medians.

Before every call the rope state is reset as the worker does on a cache hit (`_rope_deltas` = 0,
`_position_ids` = None). Every call, round 0 included, is compared bitwise with the first call:
the post-slab cache state plus the final hidden state. The hidden state covers the last layer's MLP
and the final norm, which feed no cache.

Main run (`raw/pfqmm_m_ab_slab3.*`, after the review fixes):

| offset | slab (M % 64) | mlx ms | old ms | new ms | new vs old | new vs mlx | equal calls |
|---:|---|---:|---:|---:|---:|---:|---|
| 12288 | 416 (32) | 869.0 | 907.7 | 816.2 | **×1.112** | ×1.065 | 21/21 |
| 12288 | 288 (32) | 645.1 | 641.3 | 586.4 | **×1.094** | ×1.100 | 21/21 |
| 12288 | 600 (24) | 1415.6 | 1240.1 | 1207.6 | ×1.027 | ×1.172 | 21/21 |
| 12288 | 432 (48) | 1070.1 | 918.9 | 919.7 | ×0.999 | ×1.163 | 21/21 |
| 12288 | 2048 (0) | 5080.1 | 4285.4 | 4276.2 | ×1.002 | ×1.188 | 21/21 |
| 0 | 416 (32) | 968.8 | 989.7 | 793.8 | **×1.247** | ×1.221 | 21/21 |
| 0 | 288 (32) | 687.1 | 687.2 | 563.4 | **×1.220** | ×1.219 | 21/21 |
| 0 | 600 (24) | 1458.4 | 1222.8 | 1201.8 | ×1.017 | ×1.214 | 21/21 |
| 0 | 432 (48) | 1055.4 | 865.6 | 860.1 | ×1.006 | ×1.227 | 21/21 |
| 0 | 2048 (0) | 5090.9 | 4261.9 | 4220.8 | ×1.010 | ×1.206 | 21/21 |

Earlier runs, cache states only, reference from round 1:
- `raw/pfqmm_m_ab_slab2.*`, new vs old: offset 0 gave 416 ×1.096, 288 ×1.077, 432 ×0.977, 2048
  ×1.011. Offset 12288 gave 416 ×1.086, 288 ×1.055, 432 ×1.026, 2048 ×1.015.
- `raw/pfqmm_m_ab_slab.*`, offset 12288: 416 ×1.070, 400 ×1.151, 288 ×1.155, 432 ×0.998, 600
  ×1.043, 2048 ×1.002.

In that first run the 416 reference "differed" from every other call. That was the process's first
call: with `_rope_deltas` None, mlx-vlm rotates the slab from position 0 whatever the cache offset,
and every later call continues from the offset. The reset above removes it, and slab3 starts at
offset 12288 with all calls equal.

The gain on the remainder-≤ 32 slabs scatters (×1.07–1.25) with the machine's state, more than
the GEMM share alone predicts. Take the low end (~×1.07–1.10) as the safe figure.

**The task's example slab (432 tokens at 12K) does not change**: 432 = 6 × 64 + 48 already took the
64-row kernel with a 16-row pad. Slabs whose remainder is ≤ 32 gain.

## Not done / open

- Split-K calls (k/v at M ≤ 256, everything at M ≤ 64 except gate/up and qkv): bit-identity would
  need MLX's fp16 partials and its strided reduce copied exactly. Worth ≤ 3 ms per slab.
- A remainder in (32, 64) still pads to 64 rows (≤ 31 rows, e.g. 432 → 448: 3.6 %). A 16- or
  48-row tile needs another x loader (steel's `BlockLoader` needs BK % (BM·BK/threads) == 0). The
  gain is ≤ 3.6 % on those M.
- Not measured: the 27B / 35B (3/4/5/6-bit) end to end. The GPU kernel test covers those bit widths
  bit-identically, and the dispatch is the same code.
