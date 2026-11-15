"""Multi-row 8-bit quantized matmul for batched decode on Apple GPUs (M1/M2).

mx.quantized_matmul runs one qmv per row below ~6 rows and a qmm (32-row
tiles, weights dequantized through threadgroup memory) above: on M1 an 8-bit
decode step costs x1.9 at 4 rows and a flat x2.6 from 6 to 32 rows. This
kernel reads each weight once for up to 16 rows of x with simdgroup MMAs:

    y[m, n] = sum_k x[m, k] * (s[n, g] * q[n, k] + b[n, g])     (fp32 accumulate)

computed as C^T = W . x^T. Each simdgroup owns 8*NA output features (NA 8x8
A fragments) and all M (padded to 8 or 16) rows; the KS simdgroups of a
threadgroup split K (64-wide groups, interleaved) and add their partial sums
through threadgroup memory once at the end. No threadgroup memory and no
barrier in the K loop:
  * The k order inside a 64-group is permuted (the same for A and B, so the
    dot product is unchanged): logical k c (0..7) of MMA step st (0..7) is
    physical k 8*c + st.
  * A (weights) is built in registers: per 64-group each thread loads ONE
    16-byte uint4 per fragment (8 rows x 64 bytes per simdgroup = 8 full
    cache lines); its two A elements of step st are byte st%4 of words st/4
    and 2 + st/4. Two weights become a half2 with the 0x6400 trick (mask,
    insert, or, subtract 1024) and one half2 fma with the group's scale and
    bias (fp16 rounding of the dequantized weight, as MLX's qmm loader).
  * B (x^T) is read straight from device memory (x is tiny, L1/L2-resident):
    per 64-group a thread's B elements of each of its two x rows are one
    16-byte uint4 (8 consecutive halves). This order costs a few more integer
    ops on A than one where both A elements sit in one word, but needs a
    quarter of the x load instructions: equal at 8 rows, -12% at 16.
Rows m >= M0 of the padded M are computed from row 0 and never stored.

Used only for 8-bit affine, group size 64, fp16 x/scales/biases, 3 <= rows
<= 16 of a decode-shaped input (..., 1, K), K % 64 == 0, N % 8 == 0, N >= 512
(MIN_N); every other call goes to mx.quantized_matmul unchanged (M <= 2 stays
bit-identical to a solo row).

5 bits (QWEN_MRQMM5=1, default 0; the 27B, whose every linear is 5-bit g64):
the same kernel with another weight side (_WSIDE[5]): a 64-group of a row is
40 bytes with the fields crossing byte boundaries, a thread loads the 3 words
that hold its 16 weights and regroups them once per group so that an MMA
step's two A elements are one shift and one mask. Everything else (x side,
MMAs, K split, reduction, fp16 dequantized weight, fp32 accumulation, the
row and N rules above) is shared; the 8-bit kernel's source is unchanged.
Measured 2026-11-10 on the real 27B weights (M1 Max, chains of distinct
layers, scripts/perf/dflash2/kern5_bench.py; raw: docs/perf/2026-11-10/raw/
kern5_*.json), the linears of one 27B forward (353 matmuls incl. lm_head):
a solo row (qmv) 55 ms; quantized_matmul 84 / 121 / 159 / 250 / 249 / 254 ms
at 2 / 3 / 4 / 6 / 8 / 16 rows (x1.5 / 2.2 / 2.9 / 4.6 / 4.5 / 4.5); this
kernel 81 / 83 / 86 / 89 ms at 3 / 4 / 6 / 8 rows (x1.48-1.61) and 124 / 126
ms at 12 / 16 (x2.2); lm_head alone 11.6 -> 4.0 ms at 8 rows. mlx-vlm's exact
verify kernels: 117 ms at 3 rows, 1.6 s at 8. At 8 rows it runs ~4.7 TFLOPS
(16 rows: 6.7), i.e. MMA-bound like the 8-bit kernel at 16 rows: the unpack
is not the cost (random 4-bit weights of the same shapes: 76 ms at 8 rows,
x1.64 of their own 46.5 ms solo row; a 4-bit build would gain its 4/5 of
bytes, not a cheaper kernel). Numerics on every real shape, rows 3..16: rms
rel error vs an fp32 reference 2.3-2.4e-4, never above quantized_matmul's at
the same rows (2.3e-4..1.4e-3) and 4-6x below a solo qmv row's (0.9-1.4e-3);
rows are NOT bit-equal to a solo qmv row (15-23 % of the elements are), but a
row's bits are the same at every row count 3..16 and whatever sits beside it
(KS 8 in both buckets, _config).
4 bits (QWEN_MRQMM4=1, default 0 until a GPU gate; the lmstudio-community
4-bit 27B, every linear 4-bit g64): the weight side _WSIDE[4] (a 64-group is
8 aligned words; a thread's 16 weights are 2 of them, regrouped once per group
into 16-bit lanes, so step st is one shift and one mask), dispatched exactly
where 5 bits is (the knobs are per width: a model with 4-bit layers next to
others, e.g. the 35B's 4-bit shared expert, is not pulled in by
QWEN_MRQMM5). Ungated, so it is self-tested on the GPU before it is
dispatched (`_selftest`, once per process on the first 4-bit weights offered:
install's first covered-size module, else whatever `supported` is asked
about): the unpack exactly (unit scales, one-hot x at every in-group offset,
both row buckets) and real-scale rows 3 / 8 / 9 / 16 within an fp32
reference's tolerance and no worse than quantized_matmul; a failure (or an
error) leaves every 4-bit call on quantized_matmul, logged. Random 4-bit
weights of the 27B's shapes (2026-11-10): 76 ms at 8 rows, x1.64 of their
own solo row (46.5 ms) -- the kernel is MMA-bound there like the 5-bit one,
so a 4-bit build gains its smaller weights only (solo row / plain decode).
6 bits (QWEN_MRQMM6=1, default 0 until a GPU gate; mixed-precision builds:
oQ4e's lm_head 248320 x 5120 and its last layer's k/v): the weight side
_WSIDE[6] (a thread's 16 weights are 3 aligned words of the 12-word group,
regrouped once per group into 4 words of 16-bit lanes; step st is one shift
and one mask), same config, dispatch and load self-test as 4 bits. Without
it a DFlash verify block streams a 6-bit lm_head once per token (mlx-vlm's
exact verify kernels take 4 / 5 / 8 bits only; 6 falls to per-token qmv) and
the drafter's head runs quantized_matmul's qmm. Chosen over re-quantizing the
head to 8 bits at load: no memory, no second rounding, and the solo-row
decode (which shares the head) reads 6 bits, not 8.

Batch-invariant mode (set_invariant, the worker's QWEN_BATCH_INVARIANT=1): row m
of the output is then the same bits whatever the number of rows (1..16) and
whichever rows sit beside it. The order of the sum for one output (n, m) is
fixed by K and KS only: simdgroup ks accumulates its groups g = ks, ks + KS, ...
(8 MMA steps of 8 k each, the k permutation the same for every row) into one
fp32 accumulator, and the reduction adds simdgroups 1..KS-1 to simdgroup 0's
sum in that order. NA (which simdgroup / fragment owns n) and MB (which 8-row
bucket holds m) only place the element: the MMA result of (n, m) reads A row n
and B column m alone (padded columns m >= M are computed from row 0 and never
stored), the dequantized weight is a per-n fma, and the accumulators of
different (i, mb) never meet. So the mode fixes KS per K (8, halved until it
divides K/64: _config) for both buckets, keeps NA free, and takes 1..16 rows
and every 8-bit module whatever its N (MIN_N is a speed rule; tests/
test_batch_invariant_metal.py checks it at the 9B shapes).

    import mrqmm; mrqmm.install(model)        # QWEN_MRQMM=0 disables

The GDN decode in-projections (qkv, z, b, a as one matmul over concatenated
weights) are fused by the worker (`worker._install_gdn_fuse`, QWEN_GDN_FUSE),
which calls `mrqmm` for 3..16 rows itself; mlx-vlm 0.7 has no fused hook of
its own (0.6.2's `_decode_quantized_linears_fused`, which this module used to
route, is gone).
"""
from __future__ import annotations

import logging
import math
import os
import threading

import mlx.core as mx
import mlx.nn as nn

log = logging.getLogger("worker.mrqmm")

MIN_ROWS = 3
MAX_ROWS = 16
# Narrower layers stay on quantized_matmul: launch-bound, no gain (35B MoE
# router N=256: x0.96-1.03 at 3..16 rows), and the routing top-k keeps the
# library's numerics. The worker's fused GDN in-projection is judged by its
# total N.
MIN_N = 512
BITS = 8
GROUP = 64
# Bit widths the kernel exists for. 8 is always dispatched; 5 (the 27B 5-bit)
# only with QWEN_MRQMM5=1, 4 (the 27B 4-bit) only with QWEN_MRQMM4=1 and 6
# (oQ4e's lm_head / k/v) only with QWEN_MRQMM6=1
# (default 0 both: every such call stays on quantized_matmul, the worker is
# unchanged bit for bit), never in batch-invariant mode (its guarantee is gated
# on the 8-bit kernel only: 5- / 4- / 6-bit modules stay "uncovered" there,
# max_batch 2 as before). One knob per width: each width's kernel is gated on
# its own, and a model mixing widths (the 35B: 4-bit shared expert) takes only
# the widths asked for.
KERNEL_BITS = (8, 5, 4, 6)
KNOBS = {5: "QWEN_MRQMM5", 4: "QWEN_MRQMM4", 6: "QWEN_MRQMM6"}
# Widths self-tested on the GPU before their first dispatch (`_selftest`): 4
# and 6 (new, no GPU gate yet). 5 is not (gated by tests/test_mrqmm.py and the
# 2026-11-10 runs; adding it here would change its load, not its numbers).
SELFTEST_BITS = (4, 6)
_TESTED = {}                  # bits -> (passed, detail): once per process


def knob(bits: int) -> bool:
    """QWEN_MRQMM5 / QWEN_MRQMM4 / QWEN_MRQMM6 for 5 / 4 / 6 bits (read at
    every call: tests set them)."""
    return bits in KNOBS and os.environ.get(KNOBS[bits], "0") == "1"


def bits5() -> bool:
    return knob(5)


def bits4() -> bool:
    return knob(4)


def bits6() -> bool:
    return knob(6)


def _bits_on(bits: int) -> bool:
    """Bit widths install / supported dispatch to the kernel now (a width in
    SELFTEST_BITS also needs its self-test passed: `supported`)."""
    return bits == BITS or (not INVARIANT and knob(bits))

_HEADER = """
#include <metal_simdgroup_matrix>
using namespace metal;
"""

# Constants (prepended per kernel, see _get_kernel): K, N, MB (M/8: 1 or 2),
# NA (A frags per simdgroup), KS (simdgroups splitting K); T = half.
# Input `m0` = int32[1] valid rows. The @W...@ marks are the weight side, per
# bit width (_WSIDE, _src): which bytes of a row's 64-group a thread loads and
# how its two A elements of MMA step st come out of them as a uint `q` holding
# them in 16-bit lanes; everything else is shared.
_SRC_T = r"""
    constexpr int G = K / 64;
    threadgroup float2 red[(KS > 1 ? KS - 1 : 1) * NA * MB * 32];
    const uint lid = thread_index_in_simdgroup;
    const int ks = int(simdgroup_index_in_threadgroup);
    const int n0 = int(threadgroup_position_in_grid.x) * (8 * NA);
    const int M0 = m0[0];
    const short qid = short(lid) / 4;
    const short fm = (qid & 4) + ((short(lid) / 2) % 4);
    const short fn = (qid & 2) * 2 + (short(lid) % 2) * 2;

@WDECL@    const device T* sp[NA];
    const device T* bp[NA];
    bool act[NA];
    for (int i = 0; i < NA; ++i) {
      act[i] = n0 + 8 * i < N;
      const int r = act[i] ? n0 + 8 * i + fm : fm;
@WPTR@      sp[i] = scales + size_t(r) * G;
      bp[i] = biases + size_t(r) * G;
    }
    // x: thread's B elements of step st are x[8mb+fn+e][g*64 + 8*fm + st], i.e.
    //    half st%2 of word st/2 of uint4 fm of the group (8 uint4)
    const device uint4* xp[MB][2];
    for (int mb = 0; mb < MB; ++mb)
      for (int e = 0; e < 2; ++e) {
        const int m = 8 * mb + fn + e;
        xp[mb][e] = (const device uint4*)(x + size_t(m < M0 ? m : 0) * K) + fm;
      }

    simdgroup_matrix<float, 8, 8> C[NA][MB];
    for (int i = 0; i < NA; ++i)
      for (int mb = 0; mb < MB; ++mb) C[i][mb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    // K loop over 64-groups, the KS simdgroups interleaved (adjacent groups of a
    // row are read by sibling simdgroups at the same time). Software pipelining
    // (loading the next group's weights before this group's MMAs), 2-4 groups
    // per iteration were measured slower or equal:
    // the extra registers cost more occupancy than the latency they hide.
    for (int g = ks; g < G; g += KS) {
      uint4 xw[MB][2];
      for (int mb = 0; mb < MB; ++mb)
        for (int e = 0; e < 2; ++e) xw[mb][e] = xp[mb][e][8 * g];
@WQDECL@      vec<T, 2> s2[NA], b2[NA];
      for (int i = 0; i < NA; ++i) {
@WLOAD@        s2[i] = vec<T, 2>(sp[i][g]);
        b2[i] = vec<T, 2>(bp[i][g]);
      }
#pragma unroll
      for (int st = 0; st < 8; ++st) {
        simdgroup_matrix<T, 8, 8> B[MB];
        for (int mb = 0; mb < MB; ++mb) {
          // B elements (x[8mb+fn][k], x[8mb+fn+1][k]): half st%2 of word st/2 of each row
          const uint u0 = xw[mb][0][st / 2], u1 = xw[mb][1][st / 2];
          const uint u = (st % 2) ? insert_bits(u1, u0 >> 16, 0u, 16u) : insert_bits(u0, u1, 16u, 16u);
          const vec<T, 2> h = as_type<vec<T, 2>>(u);
          B[mb].thread_elements()[0] = h.x;
          B[mb].thread_elements()[1] = h.y;
        }
        for (int i = 0; i < NA; ++i) {
@WSTEP@          const vec<T, 2> a = fma(as_type<vec<T, 2>>(q | 0x64006400u) - vec<T, 2>(1024), s2[i], b2[i]);
          simdgroup_matrix<T, 8, 8> A;
          A.thread_elements()[0] = a.x;
          A.thread_elements()[1] = a.y;
          for (int mb = 0; mb < MB; ++mb) simdgroup_multiply_accumulate(C[i][mb], A, B[mb], C[i][mb]);
        }
      }
    }

    // thread holds C[i][mb] at (n = n0+8i+fm, m = 8mb+fn+{0,1})
    float2 acc[NA][MB];
    for (int i = 0; i < NA; ++i)
      for (int mb = 0; mb < MB; ++mb) {
        auto e = C[i][mb].thread_elements();
        acc[i][mb] = float2(e[0], e[1]);
      }
    if (KS > 1) {
      if (ks > 0)
        for (int i = 0; i < NA; ++i)
          for (int mb = 0; mb < MB; ++mb) red[(((ks - 1) * NA + i) * MB + mb) * 32 + lid] = acc[i][mb];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (ks > 0) return;
      for (int s = 0; s < KS - 1; ++s)
        for (int i = 0; i < NA; ++i)
          for (int mb = 0; mb < MB; ++mb) acc[i][mb] += red[((s * NA + i) * MB + mb) * 32 + lid];
    }
    for (int i = 0; i < NA; ++i) {
      if (!act[i]) continue;
      const int n = n0 + 8 * i + fm;
      for (int mb = 0; mb < MB; ++mb) {
        const int m = 8 * mb + fn;
        if (m < M0) y[size_t(m) * N + n] = T(acc[i][mb].x);
        if (m + 1 < M0) y[size_t(m + 1) * N + n] = T(acc[i][mb].y);
      }
    }
"""

# The weight side of _SRC_T. 8 bits is the kernel as it has always been
# (tests/test_mrqmm_bits.py pins the rendered source by hash). 5 bits
# (the 27B): a row's 64-group is 40 bytes = 10 words, weight k at bit 5k, LSB
# first across byte boundaries (MLX's packing, kernels/quantized.h
# `dequantize`). The thread's 16 weights k in [8fn, 8fn + 16) are the 80 bits
# from byte 5fn of the group: 3 words starting at word {0, 2, 5, 7}[fn/2], the
# field at bit 0 or 16 of them (the 4 threads of a row read 12 of the 10 words:
# words 2 and 7 twice). Once per group they are regrouped into 3 words of two
# 16-bit lanes, the low lane from weights 0..7 (bits 0..39) and the high lane
# from weights 8..15 (bits 40..79), word j holding bits [15j, 15j + 16) of each
# 40-bit half: weights 3j..3j+2 sit at lane bits 0, 5, 10, so step st is one
# shift and one mask (cheaper per step than the 8-bit byte shuffle; the
# regrouping costs ~25 integer ops per group against 8 x 3 saved).
_WSIDE = {
    8: dict(
        WDECL="""    // weights: thread reads bytes [16*(fn/2), +16) of each 64-byte group of row n0+8i+fm
    const device uint4* wp[NA];
""",
        WPTR="""      wp[i] = (const device uint4*)((const device uint8_t*)w + size_t(r) * K) + fn / 2;
""",
        WQDECL="""      uint4 wq[NA];
""",
        WLOAD="""        wq[i] = wp[i][4 * g];
""",
        WSTEP="""          // A elements (k = 8fn + st, 8fn + 8 + st): byte st%4 of words st/4 and 2 + st/4
          // -> half2 (0x6400 | q) - 1024 = q, exact
          const uint w0 = wq[i][st / 4], w1 = wq[i][2 + st / 4];
          const uint m0 = ((st % 2) ? (w0 >> 8) : w0) & 0x00ff00ffu;
          const uint m1 = ((st % 2) ? (w1 >> 8) : w1) & 0x00ff00ffu;
          const uint q = ((st / 2) % 2) ? insert_bits(m1, m0 >> 16, 0u, 16u) : insert_bits(m0, m1, 16u, 16u);
"""),
    5: dict(
        WDECL="""    // weights (5-bit): thread reads words [(5*fn)/4, +3) of each 10-word group of row n0+8i+fm;
    // its 80 bits (16 weights from k = 8fn) start at bit 16 there when fn/2 is odd
    const device uint* wp[NA];
    const bool wodd = (fn / 2) % 2 != 0;
""",
        WPTR="""      wp[i] = (const device uint*)w + size_t(r) * (K / 32 * 5) + (5 * fn) / 4;
""",
        WQDECL="""      uint3 wq[NA];
""",
        WLOAD="""        {
          const device uint* p = wp[i] + 10 * g;
          const uint a0 = p[0], a1 = p[1], a2 = p[2];
          // the 80 bits from bit 0: v0 | v1 << 32 | v2 << 64
          const uint v0 = wodd ? (a0 >> 16) | (a1 << 16) : a0;
          const uint v1 = wodd ? (a1 >> 16) | (a2 << 16) : a1;
          const uint v2 = wodd ? a2 >> 16 : a2;
          // 16-bit lanes (low: bits 0..39, high: bits 40..79), word j = bits [15j, 15j+16) of each
          wq[i] = uint3(insert_bits(v0, v1 >> 8, 16u, 16u),
                        insert_bits(v0 >> 15, (v1 >> 23) | (v2 << 9), 16u, 16u),
                        insert_bits((v0 >> 30) | (v1 << 2), v2 >> 6, 16u, 16u));
        }
""",
        WSTEP="""          // A elements (k = 8fn + st, 8fn + 8 + st): bits [5*(st%3), +5) of both lanes of word st/3
          // -> half2 (0x6400 | q) - 1024 = q, exact
          const uint q = (wq[i][st / 3] >> uint(5 * (st % 3))) & 0x001f001fu;
"""),
    # 4 bits: a 64-group is 32 bytes, the thread's 16 weights are 2 aligned
    # words (nibbles k = 8fn.. in the first, 8fn + 8.. in the second), regrouped
    # into 2 words of two 16-bit lanes of 4 nibbles each
    4: dict(
        WDECL="""    // weights (4-bit): thread reads words [fn, fn + 2) of each 8-word group of row n0+8i+fm
    const device uint2* wp[NA];
""",
        WPTR="""      wp[i] = (const device uint2*)(w + size_t(r) * (K / 8)) + fn / 2;
""",
        WQDECL="""      uint2 wq[NA];
""",
        WLOAD="""        {
          const uint2 a = wp[i][4 * g];
          wq[i] = uint2(insert_bits(a.x, a.y, 16u, 16u), insert_bits(a.y, a.x >> 16, 0u, 16u));
        }
""",
        WSTEP="""          // A elements (k = 8fn + st, 8fn + 8 + st): nibble st%4 of both lanes of word st/4
          // -> half2 (0x6400 | q) - 1024 = q, exact
          const uint q = (wq[i][st / 4] >> uint(4 * (st % 4))) & 0x000f000fu;
"""),
    # 6 bits (oQ4e's lm_head and last k/v): a 64-group is 48 bytes = 12 words,
    # weight k at bit 6k, LSB first across bytes (MLX: 4 values per 3 bytes,
    # kernels/quantized.h `dequantize`). The thread's 16 weights k in
    # [8fn, 8fn + 16) are bits [48fn, 48fn + 96): 3 ALIGNED words from word
    # 3 * (fn / 2) (fn is even). Regrouped once per group into 4 words of two
    # 16-bit lanes, word j holding weights 2j, 2j + 1 (bits [12j, 12j + 12)) of
    # the low (weights 0..7) / high (8..15) 48-bit half: step st is one shift
    # (0 or 6) and one mask. Bits 12..15 of a lane are left unmasked: after the
    # shift they land in bits 6..9 of the lane, cleared by the step's mask.
    6: dict(
        WDECL="""    // weights (6-bit): thread reads words [3*(fn/2), +3) of each 12-word group of row n0+8i+fm
    const device uint* wp[NA];
""",
        WPTR="""      wp[i] = (const device uint*)w + size_t(r) * (K / 64 * 12) + 3 * (fn / 2);
""",
        WQDECL="""      uint4 wq[NA];
""",
        WLOAD="""        {
          const device uint* p = wp[i] + 12 * g;
          const uint a0 = p[0], a1 = p[1], a2 = p[2];
          // low half: bits 0..47 = a0 | a1[0:16]; high half: bits 48..95 = a1[16:32] | a2
          const uint h0 = (a1 >> 16) | (a2 << 16), h1 = a2 >> 16;
          wq[i] = uint4(insert_bits(a0, h0, 16u, 16u),
                        insert_bits(a0 >> 12, h0 >> 12, 16u, 16u),
                        insert_bits((a0 >> 24) | (a1 << 8), (h0 >> 24) | (h1 << 8), 16u, 16u),
                        insert_bits(a1 >> 4, h1 >> 4, 16u, 16u));
        }
""",
        WSTEP="""          // A elements (k = 8fn + st, 8fn + 8 + st): bits [6*(st%2), +6) of both lanes of word st/2
          // -> half2 (0x6400 | q) - 1024 = q, exact
          const uint q = (wq[i][st / 2] >> uint(6 * (st % 2))) & 0x003f003fu;
"""),
}


def _src(bits: int) -> str:
    """The kernel source for a bit width (8: the text it has always had)."""
    s = _SRC_T
    for k, v in _WSIDE[bits].items():
        assert s.count(f"@{k}@") == 1, k
        s = s.replace(f"@{k}@", v)
    return s


_SRC = _src(BITS)             # the 8-bit kernel's source

# The wide variant (mrqmm_wide, the fp32 DFlash2 drafter: QWEN_DRAFT_MRQMM):
# the same kernel, fp32 output scaled per row on the way out (y = acc *
# rs[m]); fp16 x is the caller's row-prescaled input (power-of-two scale, so
# the un-scale is exact).
_STORE = ("if (m < M0) y[size_t(m) * N + n] = T(acc[i][mb].x);",
          "if (m + 1 < M0) y[size_t(m + 1) * N + n] = T(acc[i][mb].y);")
_STORE_WIDE = ("if (m < M0) y[size_t(m) * N + n] = acc[i][mb].x * rs[m];",
               "if (m + 1 < M0) y[size_t(m + 1) * N + n] = acc[i][mb].y * rs[m + 1];")


def _src_wide(bits: int) -> str:
    s = _src(bits)
    for a, b in zip(_STORE, _STORE_WIDE):
        assert s.count(a) == 1, a
        s = s.replace(a, b)
    return s


# Batch-invariant mode (set_invariant): one K split for every row count, 1..16
# rows and any N take the kernel. Off = the speed rules above, bit for bit.
INVARIANT = False
INVARIANT_KS = 8
INVARIANT_NA16 = 2            # NA at 9..16 rows (placement only: free to retune)

_KERNELS = {}                 # (K, N, MB, NA, KS, bits[, "wide"]) -> compiled metal_kernel
_CALLS = {}                   # (x.shape, x.dtype, w.shape, cfg) -> (kernel, kwargs, m0 array)
_LOCK = threading.Lock()      # cache misses only (warm-up at load vs the scheduler thread)


def _config(K: int, N: int, MB: int, bits: int = BITS) -> tuple[int, int]:
    """(NA, KS): A fragments (8 output features each) per simdgroup and
    simdgroups splitting K, per row bucket. Measured on M1 Max (results/mrqmm.md):
    8 rows -> 2 x 8, 16 rows -> 4 x 4 (more MMAs per x fragment); a narrow
    layer keeps NA 2 so that it still spreads over enough simdgroups.
    5 / 4 / 6 bits (27B shapes, 2026-11-10; 6 not swept yet): 8 rows -> 4 x 8 (4-11 % under 2 x 8;
    the kernel is MMA-bound there, ~4.7 TFLOPS), 16 rows -> 2 x 8 (within 1 %
    of 4 x 4), so KS is 8 in both buckets and a row's bits are the same at
    every row count 3..16. 2 x 16 at 8 rows is another 6-7 % faster for
    K <= 6144 (slower at K = 17408), but 512 threads per threadgroup is over
    some of these pipelines' limit (448 with NA 4): not taken."""
    na, ks = (2, 8) if MB == 1 else (4, 4)
    if bits != BITS:
        na, ks = (4, 8) if MB == 1 else (2, 8)
    if INVARIANT:
        # the sum order of (n, m) depends on KS (and K) alone, NA / MB only
        # place it: KS 8 as the 1..8-row bucket (unchanged there), NA 2 at
        # 9..16 rows (2 x 8 within 3 % of 4 x 4 at 16 rows, results/mrqmm.md
        # config sweep). NA can be retuned without changing a bit.
        na, ks = (2 if MB == 1 else INVARIANT_NA16), INVARIANT_KS
    if N < 4096 or N % (8 * na):
        na = 2 if N % 16 == 0 else 1
    while ks > 1 and (K // GROUP) % ks:
        ks //= 2          # not needed for correctness (groups are strided), keeps KS | G
    return na, ks


def _get_kernel(K, N, MB, na, ks, bits=BITS, wide=False):
    """One metal_kernel per shape/config with the constants baked into the
    source: template arguments cost ~6 us of host time per call (mlx 0.31),
    a plain call ~1.5 us. wide: the fp32-output variant (_src_wide)."""
    key = (K, N, MB, na, ks, bits) + (("wide",) if wide else ())
    k = _KERNELS.get(key)
    if k is None:
        consts = (f"using T = half;\n    constexpr int K = {K}, N = {N}, MB = {MB}, NA = {na}, "
                  f"KS = {ks};\n")
        k = _KERNELS[key] = mx.fast.metal_kernel(
            name=f"mrqmm{bits}{'w' if wide else ''}_g64_k{K}_n{N}_mb{MB}_a{na}_s{ks}",
            input_names=["x", "w", "scales", "biases", "m0"] + (["rs"] if wide else []),
            output_names=["y"],
            source=consts + (_src_wide(bits) if wide else _src(bits)), header=_HEADER)
    return k


def _call(K, N, rows, cfg=None, bits=BITS, wide=False):
    MB = 1 if rows <= 8 else 2
    na, ks = cfg or _config(K, N, MB, bits)
    tiles = -(-N // (8 * na))
    kw = dict(grid=(32 * ks * tiles, 1, 1), threadgroup=(32 * ks, 1, 1),
              output_shapes=[(rows, N)], output_dtypes=[mx.float32 if wide else mx.float16])
    if wide:
        return _get_kernel(K, N, MB, na, ks, bits, True), kw
    return _get_kernel(K, N, MB, na, ks, bits), kw


def _bits_of(K: int, words: int):
    """The bit width of a packed weight row of `words` uint32 holding K
    values (the kernel's widths only), else None."""
    for bits in KERNEL_BITS:
        if words * 32 == K * bits:
            return bits
    return None


def _layout_ok(w: mx.array, scales: mx.array, biases, bits: int, group_size: int, mode="affine") -> bool:
    """The packing / dtypes / shapes the kernel of `bits` takes (whatever the
    knobs say)."""
    if bits not in KERNEL_BITS or group_size != GROUP or mode != "affine" or biases is None:
        return False
    if w.dtype != mx.uint32 or w.ndim != 2 or scales.dtype != mx.float16 or biases.dtype != mx.float16:
        return False
    N, K = w.shape[0], w.shape[1] * 32 // bits
    return (N % 8 == 0 and K % GROUP == 0 and w.shape[1] * 32 == K * bits
            and scales.shape == (N, K // GROUP))


def supported(w: mx.array, scales: mx.array, biases, bits: int, group_size: int, mode="affine") -> bool:
    """Weights the kernel can take (any row count): 8-bit, 5-bit with
    QWEN_MRQMM5=1, 4- / 6-bit with QWEN_MRQMM4 / 6=1 (`_bits_on`) once its self-test
    passed (run here on these weights if no weights of that width were
    tested yet: `_tested`)."""
    return (_bits_on(bits) and _layout_ok(w, scales, biases, bits, group_size, mode)
            and _tested(bits, w, scales, biases))


def _test_device():
    """The device of the self-test (the kernel's: tests on the CPU suite
    replace it, and the kernel, by CPU stand-ins)."""
    return mx.gpu


def _tested(bits: int, w, scales, biases, where: str = "") -> bool:
    """A width outside SELFTEST_BITS: True. Else the self-test's verdict,
    running it (logged) on these weights the first time."""
    if bits not in SELFTEST_BITS:
        return True
    r = _TESTED.get(bits)
    if r is None:
        N, K = w.shape[0], scales.shape[-1] * GROUP
        try:
            with mx.stream(_test_device()):
                why = _selftest(w, scales, biases, bits)
        except Exception as e:  # noqa: BLE001 — compile / launch error: stay on quantized_matmul
            why = f"{type(e).__name__}: {str(e)[:300]}"
        what = f"{where + ' ' if where else ''}K={K} N={N}"
        r = _TESTED[bits] = (why is None, what if why is None else f"{what}: {why}")
        if why is None:
            log.info("mrqmm %d-bit g%d self-test passed (%s): %s on the kernel", bits, GROUP, r[1], KNOBS[bits])
        else:
            log.warning("mrqmm %d-bit g%d self-test FAILED (%s): every %d-bit call stays on quantized_matmul",
                        bits, GROUP, r[1], bits)
    return r[0]


SELFTEST_ROWS = (3, 8, 9, 16)     # real-scale rows: both buckets, their edges
SELFTEST_SLICE = 256              # output rows compared at each end of N (fp32 reference)


def _selftest(w, scales, biases, bits: int) -> str | None:
    """None when the kernel is right on these weights, else why not:
      1. the unpack, exactly: unit scales, zero biases, one-hot x rows, so
         y[m, n] must be the packed integer q[n, k_m] itself (sums of zeros
         and one q: exact in fp32 and fp16). 8- and 16-row calls (both row
         buckets) put the ones at every in-group offset 0..63, spread over
         the groups (every simdgroup of the K split), all N outputs checked
         against mx.dequantize of those groups;
      2. the real scales / biases at SELFTEST_ROWS rows of a random x: finite,
         and on the first / last SELFTEST_SLICE outputs within fp16 rounding
         of an fp32 reference (max |d| <= 1.5e-3 max|ref| + 1e-4) and no
         worse than mx.quantized_matmul at the same rows + one fp16 ulp (the
         rules of tests/test_mrqmm.py's check_close).
    Runs on the current default stream (`_tested` picks it). ~30 kernel
    calls over the full weights (each reads them once)."""
    N, K = w.shape[0], scales.shape[-1] * GROUP
    G, wpg = K // GROUP, GROUP * bits // 32          # uint32 words per 64-group
    one, zero = mx.ones_like(scales), mx.zeros_like(biases)
    cols = {}

    def col(k):                                      # q[:, k], fp32 (N,)
        g = k // GROUP
        if g not in cols:
            cols[g] = mx.dequantize(w[:, wpg * g:wpg * (g + 1)], one[:, g:g + 1], zero[:, g:g + 1],
                                    group_size=GROUP, bits=bits).astype(mx.float32)
        return cols[g][:, k % GROUP]
    for rows in (8, 16):
        for j in range(GROUP // rows):
            ks = [GROUP * ((7 * m + j) % G) + rows * j + m for m in range(rows)]
            x = (mx.arange(K)[None, :] == mx.array(ks)[:, None]).astype(mx.float16).reshape(rows, 1, K)
            y = mrqmm(x, w, one, zero).reshape(rows, N).astype(mx.float32)
            want = mx.stack([col(k) for k in ks])
            if not mx.array_equal(y, want).item():
                bad = mx.argmax((y != want).any(axis=1).astype(mx.int32)).item()
                return (f"unpack: {rows} rows, one-hot k={ks[bad]} (offset {ks[bad] % GROUP} of group "
                        f"{ks[bad] // GROUP}) differs from the packed weights")
        cols.clear()
    spans = sorted({(0, min(N, SELFTEST_SLICE)), (max(0, N - SELFTEST_SLICE), N)})
    for rows in SELFTEST_ROWS:
        x = (mx.random.normal((rows, 1, K), key=mx.random.key(rows)) * 0.5).astype(mx.float16)
        y = mrqmm(x, w, scales, biases).reshape(rows, N).astype(mx.float32)
        if not mx.isfinite(y).all().item():
            return f"{rows} rows: non-finite output"
        xf = x.reshape(rows, K).astype(mx.float32)
        for lo, hi in spans:
            deq = mx.dequantize(w[lo:hi], scales[lo:hi].astype(mx.float32), biases[lo:hi].astype(mx.float32),
                                group_size=GROUP, bits=bits)
            ref = xf @ deq.T
            q = mx.quantized_matmul(x, w[lo:hi], scales[lo:hi], biases[lo:hi], transpose=True,
                                    group_size=GROUP, bits=bits).reshape(rows, hi - lo).astype(mx.float32)
            mref = mx.abs(ref).max().item()
            err = mx.abs(y[:, lo:hi] - ref).max().item()
            qerr = mx.abs(q - ref).max().item()
            ulp = 2.0 ** (math.ceil(math.log2(max(mref, 1e-6))) - 11)
            if err > 1.5e-3 * mref + 1e-4 or err > qerr + ulp:
                return (f"{rows} rows, outputs [{lo}, {hi}): max |d| {err:.3g} vs fp32 (|ref| max {mref:.3g}), "
                        f"quantized_matmul's {qerr:.3g}")
    return None


def mrqmm(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array, cfg=None) -> mx.array:
    """x[..., K] (fp16, 1..16 rows) @ dequant(w)^T for 8-bit (or 5- / 4- / 6-bit)
    g64 affine weights -> [..., N]. The bit width is the packed row's: w.shape[1]
    words for x's K values (8: K / 4, 5: K * 5 / 32, 4: K / 8; whatever
    QWEN_MRQMM5 / 4 say: the knobs are the dispatch's, `supported`). Checks only the x side (rows,
    dtype, K) once per x / w shape: call `supported` on the weights first. x
    is made row-contiguous by metal_kernel (a strided x is copied); a
    contiguous view starts on a row, so its 16-byte loads stay aligned
    (K % 64 == 0)."""
    N = w.shape[0]
    key = (x.shape, x.dtype, w.shape, cfg)      # w.shape: a hit skips the K check below
    c = _CALLS.get(key)
    if c is None:
        K = x.shape[-1]
        rows = x.size // K
        bits = _bits_of(K, w.shape[1])
        if not (1 <= rows <= MAX_ROWS and x.dtype == mx.float16 and bits is not None
                and K % GROUP == 0 and scales.shape[-1] * GROUP == K):
            raise ValueError(f"mrqmm: x {x.shape} {x.dtype} vs w {w.shape}")
        with _LOCK:
            kernel, kw = _call(K, N, rows, cfg, bits)
            kw["output_shapes"] = [(*x.shape[:-1], N)]
            c = _CALLS[key] = (kernel, kw, mx.array([rows], dtype=mx.int32))
    kernel, kw, m0 = c
    return kernel(inputs=[x, w, scales, biases, m0], **kw)[0]


_WIDE_CALLS = {}              # (x.shape, w.shape, cfg) -> (kernel, kwargs, m0 array)


def mrqmm_wide(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array, rs: mx.array,
               cfg=None) -> mx.array:
    """mrqmm with an fp32 output scaled per row: y[m] = (x[m] @ dequant(w)^T)
    * rs[m]. x fp16 (..., K), 1..16 rows; rs fp32 with one value per row (the
    inverse of the caller's power-of-two prescale: _drafter_linear). The sums
    are the fp32 accumulators of mrqmm, never rounded to fp16."""
    N = w.shape[0]
    key = (x.shape, w.shape, cfg)
    c = _WIDE_CALLS.get(key)
    if c is None:
        K = x.shape[-1]
        rows = x.size // K
        bits = _bits_of(K, w.shape[1])
        if not (1 <= rows <= MAX_ROWS and x.dtype == mx.float16 and bits is not None
                and K % GROUP == 0 and scales.shape[-1] * GROUP == K):
            raise ValueError(f"mrqmm_wide: x {x.shape} {x.dtype} vs w {w.shape}")
        with _LOCK:
            kernel, kw = _call(K, N, rows, cfg, bits, wide=True)
            kw["output_shapes"] = [(*x.shape[:-1], N)]
            c = _WIDE_CALLS[key] = (kernel, kw, mx.array([rows], dtype=mx.int32))
    kernel, kw, m0 = c
    return kernel(inputs=[x, w, scales, biases, m0, rs], **kw)[0]


def _use(x: mx.array, n: int | None = None) -> bool:
    """Decode-shaped fp16 input with MIN_ROWS..MAX_ROWS rows for a layer of n
    outputs (>= MIN_N; None: vetted by the caller); batch-invariant:
    1..MAX_ROWS rows, any n (an install in that mode swapped narrow layers
    too, so toggling the mode off restores the speed rules exactly)."""
    if x.dtype != mx.float16 or x.ndim < 2 or x.shape[-2] != 1:
        return False
    rows = x.size // x.shape[-1]
    if INVARIANT:
        return 1 <= rows <= MAX_ROWS
    return (n is None or n >= MIN_N) and MIN_ROWS <= rows <= MAX_ROWS


def set_invariant(on: bool):
    """Batch-invariant mode on / off (the worker at load, before install;
    tools toggle it between steps on the scheduler thread). The call cache
    holds configs: cleared."""
    global INVARIANT
    with _LOCK:
        INVARIANT = bool(on)
        _CALLS.clear()
        _WIDE_CALLS.clear()


class MRQuantizedLinear(nn.QuantizedLinear):
    """nn.QuantizedLinear whose decode-shaped 3..16-row calls take mrqmm.
    Installed by swapping __class__ (parameters and state are untouched)."""

    def __call__(self, x):
        if not _use(x, self["weight"].shape[0]):
            return super().__call__(x)
        y = mrqmm(x, self["weight"], self["scales"], self["biases"])
        if "bias" in self:
            y = y + self["bias"]
        return y


def _module_ok(m) -> bool:
    """Weights the kernel takes (8-bit, or 5- / 4- / 6-bit with QWEN_MRQMM5 / 6 /
    QWEN_MRQMM4 = 1, 4-bit after its self-test; affine g64, fp16 scales and
    biases)."""
    return (type(m) in (nn.QuantizedLinear, MRQuantizedLinear)
            and supported(m["weight"], m["scales"], m.get("biases"), m.bits, m.group_size,
                          getattr(m, "mode", "affine")))


def enabled() -> bool:
    return os.environ.get("QWEN_MRQMM", "1") != "0" and mx.metal.is_available()


def install(model, warmup: bool = True) -> int:
    """Swap every supported nn.QuantizedLinear of `model` (8-bit, with
    QWEN_MRQMM5 / 4 / 6 = 1 and not batch-invariant also 5- / 4- / 6-bit;
    g64, fp16 scales/biases, N >= MIN_N; any N in batch-invariant mode) to
    MRQuantizedLinear; every other module is left alone. A width in
    SELFTEST_BITS is self-tested first on its first module of N >= MIN_N
    (else its first module) and left alone if that fails. Logs the coverage
    once. Returns the number of modules covered (0 = no-op). The GDN decode
    in-projections fused into one matmul are the worker's
    (`_install_gdn_fuse`, after this; it calls `warm` for its shapes)."""
    if not enabled():
        log.info("mrqmm: off (QWEN_MRQMM=0 or no Metal)")
        return 0
    for bits in SELFTEST_BITS:
        if bits in _TESTED or not _bits_on(bits):
            continue
        mods = [(n, m) for n, m in model.named_modules()
                if type(m) in (nn.QuantizedLinear, MRQuantizedLinear) and m.bits == bits
                and _layout_ok(m["weight"], m["scales"], m.get("biases"), bits, m.group_size,
                               getattr(m, "mode", "affine"))]
        pick = next(((n, m) for n, m in mods if m.weight.shape[0] >= MIN_N), mods[0] if mods else None)
        if pick is not None:
            n, m = pick
            _tested(bits, m["weight"], m["scales"], m["biases"], where=n)
    covered, skipped = [], {}
    for name, m in model.named_modules():
        if not isinstance(m, nn.QuantizedLinear):
            continue
        if _module_ok(m) and (INVARIANT or m.weight.shape[0] >= MIN_N):
            m.__class__ = MRQuantizedLinear
            covered.append((name, m))
        elif type(m) is nn.QuantizedLinear:
            why = "" if not _bits_on(m.bits) or m.group_size != GROUP else (
                " N % 8" if m.weight.shape[0] % 8 else " not fp16" if m.scales.dtype != mx.float16
                else f" N < {MIN_N}" if m.weight.shape[0] < MIN_N
                else " self-test failed" if not _TESTED.get(m.bits, (True,))[0] else "")
            k = f"{m.bits}-bit g{m.group_size}{why}"
            skipped[k] = skipped.get(k, 0) + 1
    if not covered:
        log.info("mrqmm: no supported module, all on quantized_matmul (%s)",
                 ", ".join(f"{k} x{v}" for k, v in skipped.items()) or "nothing quantized")
        return 0
    kinds = {}
    for name, _ in covered:
        k = ".".join(p for p in name.split(".") if not p.isdigit())
        for pre in ("language_model.", "model.layers.", "layers."):
            k = k[len(pre):] if k.startswith(pre) else k
        kinds[k] = kinds.get(k, 0) + 1
    widths = sorted({m.bits for _, m in covered})
    log.info("mrqmm: %d modules on the multi-row kernel%s for %d..%d decode rows (%s); "
             "left on quantized_matmul: %s", len(covered),
             "" if widths == [BITS] else
             f" ({'/'.join(map(str, widths))}-bit, {', '.join(KNOBS[b] for b in widths if b in KNOBS)})",
             1 if INVARIANT else MIN_ROWS, MAX_ROWS,
             ", ".join(f"{k} x{v}" for k, v in sorted(kinds.items())),
             ", ".join(f"{k} x{v}" for k, v in skipped.items()) or "none")
    if warmup:
        shapes = {}
        for _, m in covered:
            shapes.setdefault((m.bits, m.weight.shape), (m.weight, m.scales, m.biases))
        for w in shapes.values():
            warm(*w)
    return len(covered)


def uncovered(model, skip=()) -> dict:
    """Batch-invariant coverage: the matmul modules of `model` a decode step
    runs outside the kernel, by class -> count. Their rows keep MLX's own
    row-count rules (qmv per row below its threshold, then qmm: B <= 2 is
    still exact there, larger B is not): 5-bit / 4-bit / 6-bit layers (27B,
    35B shared expert / lm_head), N % 8 layers (35B shared_expert_gate),
    unquantized nn.Linear, the MoE experts (gather_qmm; SwitchGLU also sorts
    by expert from B * top_k >= 64 rows), a tied lm_head (embed_tokens.as_linear).
    `skip`: ids of modules a decode step never calls (the worker's fused GDN
    in-projections)."""
    out = {}

    def add(k):
        out[k] = out.get(k, 0) + 1
    for _, m in model.named_modules():
        t = type(m)
        if t is MRQuantizedLinear or id(m) in skip:
            continue
        if t is nn.QuantizedLinear:
            why = ("" if (m.bits, m.group_size) != (BITS, GROUP) else
                   " N % 8" if m.weight.shape[0] % 8 else
                   " not fp16" if m.scales.dtype != mx.float16 else
                   " mrqmm off" if not enabled() else "")
            add(f"QuantizedLinear {m.bits}-bit g{m.group_size}{why}")
        elif t is nn.Linear:
            add("Linear (unquantized)")
        elif "Switch" in t.__name__ and hasattr(m, "weight"):
            add(f"{t.__name__} (MoE experts)")
    args = getattr(model, "args", None)
    if getattr(args, "tie_word_embeddings", False):
        add("tied lm_head (embed_tokens.as_linear)")
    return out


def warm(w: mx.array, scales: mx.array, biases: mx.array):
    """Compile the kernels of this weight shape for both row buckets (3..8 and
    9..16 rows) now, not on the first batched decode step (~60 ms of Metal
    compile each, on the scheduler thread). Zero inputs: only the shapes
    matter (K from the scales: one per 64 values, whatever the bit width)."""
    K = scales.shape[-1] * GROUP
    with mx.stream(mx.gpu):
        mx.eval([mrqmm(mx.zeros((rows, 1, K), mx.float16), w, scales, biases)
                 for rows in (MIN_ROWS, MAX_ROWS)])


def uninstall(model) -> int:
    """Undo install on `model` (its modules go back to nn.QuantizedLinear)."""
    n = 0
    for _, m in model.named_modules():
        if type(m) is MRQuantizedLinear:
            m.__class__ = nn.QuantizedLinear
            n += 1
    return n
