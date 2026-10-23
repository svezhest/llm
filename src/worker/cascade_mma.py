"""Cascade prefix attention on simdgroup MMA over an fp16 prefix (QWEN_CASCADE_KERNEL).

The prefix half of `cascade.cascade` -- the B_g rows of a group attending the
shared [0, P) of one full-attention layer -- as two Metal kernels that read
the prefix in fp16 straight from a (1, Hkv, capP, D) KV buffer (the donor
row's own KVCache buffer, aliased: cascade.PrefixGroup.build(kernel=True)),
instead of an fp32 chunked copy with (Hkv, c, B_g*rep, Pc) fp32 scores and
probabilities materialized per layer.

The rep = Hq // Hkv query heads of every row that map to kv head h are
stacked into M0 = B_g * rep query rows (row b * rep + r, as cascade's `qp`):
16 rows x G4 = 64 query rows per kv head, so a decode step over a shared
prefix is a small prefill-shaped attention per kv head (M0 queries x P keys,
no mask) -- the shape MLX's steel attention kernel is built for, except that
4 kv heads x 2 query blocks alone would occupy 8 threadgroups of a 32-core
GPU, hence the split over the keys.

  pass 1 (`cmma_p1`, grid items x Hkv x ceil(M0 / BQ) threadgroups of WM x WN
         simdgroups): MLX's steel attention loop (mlx 0.32.3 kernels/steel/
         attn/kernels/steel_attention.h, `attention`: the same loaders, MMA
         tiles, WN = 2 score exchange and online softmax, from the wheel's
         headers) over ONE chunk of the prefix (split-KV / flash-decoding:
         `items` = ceil(P / chunk) chunks), writing the chunk's normalized
         output o / l (fp32), its row max m (log2 domain) and sum l instead of
         the final fp16 output. Q (BQ x D) and each K / V block (BK keys) are
         staged in threadgroup memory; S = Q K^T is half x half MMA with fp32
         accumulation (the products are exact: 11 + 11 bits < 24), scaled by
         scale * log2 e in fp32; P V multiplies the fp32 probabilities by half
         V fragments with fp32 accumulation (P is not rounded to fp16). A
         partial last block loads zeros past the chunk and masks those scores;
         nothing past P is ever read (a donor buffer holds its own suffix there).
  pass 2 (grid M0 x Hkv threadgroups of D threads): the flash-decoding merge
         of rattn.py's pass 2 in fp32: o = sum_c w_c o_c / sum_c w_c, w_c =
         l_c 2^(m_c - M), and the natural-log LSE (M + log2 L) ln 2.

The caller (`cascade.attention`) then runs the suffix part and the prefix /
suffix merge with today's mx ops (cascade.merge, under mx.compile): the
members' suffixes stay in the fp32 stack, so append / take / release and
their bit-exactness are unchanged. The suffixes are not in this kernel: they
are short (<= P // 2 by the leave rule, ~150 tokens at formation), today's
suffix GEMMs cost little next to an 8K-32K prefix, and moving them in would
turn the stack into fp16 with per-row item tables (append, take, grow,
release, the masks) -- a second step once the prefix part is measured
(docs/perf/2026-10-24/design/cascade_mma.md).

Work split: `chunk` (a multiple of BK) is the largest power of two in
[MIN_CHUNK, MAX_CHUNK] that still gives items x Hkv x query blocks >=
TARGET_TG threadgroups. Everything shape-dependent (M0, P, capP, chunk,
items) is a runtime input (`meta`, cached per shape: one per group shape and
step, shared by every layer), so one compiled pipeline per model (D, Hkv,
scale) serves every group size and prefix length.

Numerics vs today's fp32 prefix: the same products (q fp16 -> exact; the scale
1/sqrt(D), a power of two for D = 64 / 256, applied in fp32 after the dot
product instead of before: exact either way), summed in another order (8 x 8
MMA trees, a WN = 2 split of D, the online rescale per BK block, the chunk
merge), exp2 with log2 e folded into the scale, no fp16 rounding before the
final output: the class of today's cascade (rms rel ~2.1e-4 against an fp32
reference on the 9B layout, dominated by the output's fp16 rounding; the
kernel's fp32 outputs alone are ~1e-6 from the fp32 reference,
tests/test_cascade_mma.py).

    o_p, lse_p = cascade_mma.prefix_attention(q, k, v, P, scale)
    # q (B, Hq, 1, D) fp16; k, v (1, Hkv, capP, D) fp16 row-contiguous, P <= capP
    # -> o_p (Hkv, B * rep, D) fp32, lse_p (Hkv, B * rep) fp32

On a CPU default device (the worker's CPU tests) `prefix_attention` computes
the same quantities with mx ops in fp32 (`reference`), so the scheduler's
kernel-mode code runs unchanged there; the Metal kernel is tested in
tests/test_cascade_mma.py (skipped without Metal).
"""
from __future__ import annotations

import math
import os
import re
import threading

import mlx.core as mx

F32 = mx.float32
MIN_CHUNK = 256
MAX_CHUNK = 2048
TARGET_TG = 256       # pass-1 threadgroups the chunk size aims for (32 cores)
HEAD_DIMS = (64, 128, 256)   # the tested ones (tests/test_cascade_mma.py)


def tiles(D: int):
    """(BQ, BK, WM, WN) per head_dim: MLX's D = 256 attention tile (BQ 32, BK
    16, 4 x 2 simdgroups); a 16-row tile for D <= 128 (the tests' head_dim 64:
    the WN = 2 score exchange must fit the K / V threadgroup buffer)."""
    return (32, 16, 4, 2) if D > 128 else (16, 16, 2, 2)


# The ops of steel_attention.h
_OPS = r"""
struct MaxOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return metal::max(x, y); }
};
struct SumOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x + y; }
};
struct MulOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x * y; }
};
struct ExpSubOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return fast::exp2(x - y); }
};
struct DivOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x / y; }
};
"""

_IMPL = r"""
// Adapted from mlx 0.32.3 kernels/steel/attn/kernels/steel_attention.h
// (`attention`, MIT): no mask / causal / sinks, WN = 2, Q in threadgroup
// memory; Q is (Hkv, M0, BD) row-major, K / V are (Hkv, capP, BD) and this
// threadgroup attends keys [start, end) of its chunk; the output is the
// chunk's normalized o (fp32) + its m and l.
template <typename T, int BQ, int BK, int BD, int WM, int WN>
METAL_FUNC void cmma_p1(
    const device T* Q, const device T* K, const device T* V,
    device float* po, device float* pml,
    const int M0, const int P, const int capP, const int chunk, const int HKV_,
    const float scale,
    threadgroup T* Q_smem, threadgroup T* KV_smem,
    uint simd_lane_id, uint simd_group_id, uint3 tid) {
  using namespace mlx::steel;
  using AccumType = float;
  constexpr int BV = BD;
  const int item = int(tid.x), h = int(tid.y);
  const int q0 = int(tid.z) * BQ;
  const int q_rows = min(BQ, M0 - q0);
  const int start = item * chunk;
  const int kL = min(P, start + chunk) - start;
  const int NK = (kL + BK - 1) / BK;
  const int NK_aligned = kL / BK;
  const int kL_rem = kL - NK_aligned * BK;

  Q += (size_t(h) * M0 + q0) * BD;
  K += (size_t(h) * capP + start) * BD;
  V += (size_t(h) * capP + start) * BD;

  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);
  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BV + padV;
  constexpr short tgp_mem_0 = (BK + padK) * (BD);
  constexpr short tgp_mem_1 = BK * (BV + padV);
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;

  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  using QBlockLoader = BlockLoaderT<T, BQ, BD, LDQ_tgp, 1, 1, WM * WN * 32>;
  using KBlockLoader = BlockLoaderT<T, BK, BD, 1, LDK_tgp, 0, WM * WN * 32>;   // K transposed
  using VBlockLoader = BlockLoaderT<T, BK, BV, LDV_tgp, 1, 0, WM * WN * 32>;
  QBlockLoader loader_q(Q, BD, Qs, simd_group_id, simd_lane_id);
  KBlockLoader loader_k(K, BD, Ks, simd_group_id, simd_lane_id);
  VBlockLoader loader_v(V, BD, Vs, simd_group_id, simd_lane_id);

  constexpr short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;
  using MMAFrag_in_t = BaseMMAFrag<T, kFragSize, kFragSize>;
  static_assert(WN == 2, "WN = 2 only");
  constexpr int TQ = BQ / (WM * kFragSize);
  static_assert(TQ == 1, "one 8-row fragment per simdgroup");
  constexpr int TK = BK / kFragSize;
  constexpr int TD = BD / kFragSize;
  constexpr int TDh = TD / WN;
  constexpr int BVh = BV / WN;
  constexpr int TVh = BV / (kFragSize * WN);

  MMATile<T, TQ, 1, MMAFrag_in_t> Qtile;
  MMATile<T, 1, TK, MMAFrag_in_t> Ktile;
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<T, 1, 1, MMAFrag_in_t> Vtile;
  MMATile<AccumType, TQ, TVh, MMAFrag_acc_t> Otile;
  Otile.clear();

  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short row_group = simd_group_id / WN;
  const short d_half = simd_group_id % WN;
  const short tm = kFragSize * TQ * row_group;
  const short Qs_offset = (tm + sm) * LDQ_tgp + d_half * (BD / WN) + sn;
  const short Ks_offset = d_half * (BD / WN) * LDK_tgp + sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + d_half * BVh + sn;
  constexpr short Qs_tile_stride = kFragSize;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (q_rows < BQ) {
    loader_q.load_safe(short2(BD, q_rows));     // rows past M0: zeros, never stored
  } else {
    loader_q.load_unsafe();
  }

  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;
  AccumType max_score[kRowsPT];
  AccumType sum_score[kRowsPT] = {0};
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) max_score[i] = Limits<AccumType>::finite_min;

  for (int kb = 0; kb < NK; kb++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kb == NK_aligned) {
      loader_k.load_safe(short2(BD, kL_rem));
    } else {
      loader_k.load_unsafe();
    }
    Stile.clear();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TDh; dd++) {
      simdgroup_barrier(mem_flags::mem_none);
      Qtile.template load<T, 1, 1, LDQ_tgp, 1>(&Qs[Qs_offset + dd * Qs_tile_stride]);
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(&Ks[Ks_offset + dd * Ks_tile_stride]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(Stile, Qtile, Ktile, Stile);
    }
    {   // WN = 2: each simdgroup summed half of D; exchange the partial scores
      constexpr int s_xchg_stride = decltype(Stile)::kElemsPerTile * 32;
      constexpr int s_xchg_size = WM * WN * s_xchg_stride;
      static_assert(s_xchg_size * sizeof(AccumType) <= tgp_mem_s * sizeof(T),
                    "The score exchange must fit in KV_smem");
      threadgroup_barrier(mem_flags::mem_threadgroup);
      threadgroup AccumType* s_xchg = reinterpret_cast<threadgroup AccumType*>(KV_smem);
      threadgroup AccumType* slot = s_xchg + (row_group * WN + d_half) * s_xchg_stride;
      const short base = short(simd_lane_id) * decltype(Stile)::kElemsPerTile;
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) slot[base + ii] = Stile.elems()[ii];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const threadgroup AccumType* peer = s_xchg + (row_group * WN + 1 - d_half) * s_xchg_stride;
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) Stile.elems()[ii] += peer[base + ii];
    }
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) Stile.elems()[ii] *= scale;
    if (kb == NK_aligned) {     // keys past the chunk's end
      using stile_t = decltype(Stile);
      constexpr auto neg_inf = Limits<AccumType>::finite_min;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          short col_pos = sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((col_pos + jj) >= kL_rem) Stile.frag_at(i, j)[jj] = neg_inf;
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kb == NK_aligned) {
      loader_v.load_safe(short2(BV, kL_rem));   // zeros past the end: 0 * NaN never happens
    } else {
      loader_v.load_unsafe();
    }
    AccumType new_max[kRowsPT];
    AccumType factor[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) new_max[i] = max_score[i];
    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) factor[i] = fast::exp2(max_score[i] - new_max[i]);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) max_score[i] = new_max[i];
    AccumType sum_score_tmp[kRowsPT] = {0};
    Stile.template row_reduce<SumOp>(sum_score_tmp);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    Otile.template row_bin_op<MulOp>(factor);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TVh; id++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        simdgroup_barrier(mem_flags::mem_none);
        Vtile.template load<T, 1, 1, LDV_tgp, 1>(&Vs[Vs_offset + ik * kFragSize * LDV_tgp + id * kFragSize]);
        simdgroup_barrier(mem_flags::mem_none);
        MMAFrag_acc_t::mma(Otile.frag_at(0, id), Stile.frag_at(0, ik), Vtile.frag_at(0, 0),
                           Otile.frag_at(0, id));
      }
    }
    loader_k.next();
    loader_v.next();
  }

  Otile.template row_bin_op<DivOp>(sum_score);
  const int row = tm + sm;
  const size_t p0 = (size_t(item) * HKV_ + h) * M0 + q0;
  device float* O = po + (p0 + row) * BD + d_half * BVh + sn;
  if (q_rows < BQ) {
    const short2 dims = short2(BVh - sn, q_rows - row);
    if (dims.x > 0 && dims.y > 0) Otile.template store_safe<float, 1, 1>(O, BD, dims);
  } else {
    Otile.template store<float, 1, 1>(O, BD);
  }
  if (sn == 0 && d_half == 0 && row < q_rows) {
    pml[2 * (p0 + row)] = max_score[0];
    pml[2 * (p0 + row) + 1] = sum_score[0];
  }
}
"""

# constants prepended (_consts): D, HKV, BQ, BK, WM, WN, SCALE2 (scale * log2 e)
# meta (int32): M0 (query rows), P (prefix keys), capP (slots per kv head), chunk, items
_P1 = r"""
    constexpr int LDQ = D + 8;
    constexpr int KVS = (BK + 8) * D > BK * (D + 8) ? (BK + 8) * D : BK * (D + 8);
    threadgroup half Q_smem[BQ * LDQ];
    threadgroup half KV_smem[KVS];
    cmma_p1<half, BQ, BK, D, WM, WN>(q, k, v, po, pml, meta[0], meta[1], meta[2], meta[3], HKV,
        SCALE2, Q_smem, KV_smem, thread_index_in_simdgroup, simdgroup_index_in_threadgroup,
        threadgroup_position_in_grid);
"""

_P2 = r"""
    const int r = int(threadgroup_position_in_grid.x);
    const int h = int(threadgroup_position_in_grid.y);
    const int d = int(thread_position_in_threadgroup.x);
    const int M0 = meta[0], n = meta[4];
    float M = -INFINITY;
    for (int c = 0; c < n; ++c) M = max(M, pml[2 * ((size_t(c) * HKV + h) * M0 + r)]);
    float L = 0.0f, acc = 0.0f;
    for (int c = 0; c < n; ++c) {
      const size_t p = (size_t(c) * HKV + h) * M0 + r;
      const float w = pml[2 * p + 1] * fast::exp2(pml[2 * p] - M);
      L += w;
      acc += w * po[p * D + d];
    }
    out[(size_t(h) * M0 + r) * D + d] = acc / L;
    if (d == 0) lse[size_t(h) * M0 + r] = (M + log2(L)) * M_LN2_F;
"""

_HEADER = None
_KERNELS: dict = {}
_META: dict = {}          # (M0, P, capP, chunk) -> int32 meta: one per group shape, all layers
_LOCK = threading.Lock()


def _header() -> str:
    """The wheel's steel attention loader + MMA tile headers, includes
    inlined (the runtime Metal compiler has no include path; metal_kernel
    prepends kernels/utils.h itself, so it is skipped), + the ops and
    `cmma_p1`."""
    global _HEADER
    if _HEADER is None:
        root = os.path.join(os.path.dirname(mx.__file__), "include")
        seen = set()

        def inline(rel):
            if rel in seen:
                return ""
            seen.add(rel)
            out = []
            with open(os.path.join(root, rel)) as f:
                for line in f.read().splitlines():
                    m = re.match(r'\s*#include\s+"(.+)"', line)
                    if m:
                        out.append(inline(m.group(1)))
                    elif line.strip() != "#pragma once":
                        out.append(line)
            return "\n".join(out)

        k = "mlx/backend/metal/kernels/"
        inline(k + "utils.h")
        _HEADER = ("\n".join(inline(k + f) for f in ("steel/attn/loader.h", "steel/attn/mma.h"))
                   + _OPS + _IMPL)
    return _HEADER


def _consts(D, hkv, scale):
    BQ, BK, WM, WN = tiles(D)
    return (f"constexpr int D = {D}, HKV = {hkv}, BQ = {BQ}, BK = {BK}, WM = {WM}, WN = {WN};\n"
            f"    constexpr float SCALE2 = {scale * math.log2(math.e)!r}f;\n")


def _kernels(D, hkv, scale):
    """The two pipelines of one model layout (compiled on first use, ~0.1-0.3 s
    of Metal compile; `warm` does it at load)."""
    key = (D, hkv, float(scale))
    k = _KERNELS.get(key)
    if k is None:
        with _LOCK:
            k = _KERNELS.get(key)
            if k is None:
                c = _consts(D, hkv, scale)
                # the exact scale in the name: MLX caches libraries by kernel name
                tag = f"d{D}_k{hkv}_s{float(scale).hex().replace('.', '_').replace('-', 'm').replace('+', 'p')}"
                p1 = mx.fast.metal_kernel(
                    name=f"cmma_p1_{tag}", input_names=["q", "k", "v", "meta"],
                    output_names=["po", "pml"], source=c + _P1, header=_header())
                p2 = mx.fast.metal_kernel(
                    name=f"cmma_p2_{tag}", input_names=["po", "pml", "meta"],
                    output_names=["out", "lse"], source=c + _P2)
                k = _KERNELS[key] = (p1, p2)
    return k


def items(P: int, chunk: int) -> int:
    return -(-P // chunk)


def pick_chunk(P: int, hkv: int, blocks: int) -> int:
    """Keys per pass-1 item: the largest power of two in [MIN_CHUNK,
    MAX_CHUNK] that keeps items x hkv x query blocks >= TARGET_TG (MIN_CHUNK
    when even that gives fewer)."""
    c = MAX_CHUNK
    while c > MIN_CHUNK and items(P, c) * hkv * blocks < TARGET_TG:
        c //= 2
    return c


def partial_bytes(P: int, rows: int, hq: int, D: int) -> int:
    """Upper bound of one layer's pass-1 partials + pass-2 output for `rows`
    group rows (the worker charges it with the group's transients): every
    default chunk is >= MIN_CHUNK keys."""
    return (items(P, MIN_CHUNK) + 1) * rows * hq * (D + 2) * 4


def _meta(m0, P, cap, chunk):
    key = (m0, P, cap, chunk)
    hit = _META.get(key)
    if hit is None:
        hit = mx.array([m0, P, cap, chunk, items(P, chunk)], dtype=mx.int32)
        with _LOCK:
            if len(_META) >= 64:
                _META.pop(next(iter(_META)))
            _META[key] = hit
    return hit


def use_kernel() -> bool:
    """The Metal kernel runs on a GPU default device; else `reference`."""
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def reference(q, k, v, P, scale):
    """The kernel's outputs with mx ops in fp32 (the CPU path; the GPU tests'
    reference): o_p (Hkv, B * rep, D), lse_p (Hkv, B * rep)."""
    B, Hq, _, D = q.shape
    Hkv = k.shape[1]
    rep = Hq // Hkv
    qp = q.astype(F32).reshape(B, Hkv, rep, D).transpose(1, 0, 2, 3).reshape(Hkv, B * rep, D)
    kk = k[0, :, :P, :].astype(F32)
    vv = v[0, :, :P, :].astype(F32)
    s = (qp * scale) @ kk.swapaxes(-1, -2)                                  # (Hkv, B*rep, P)
    lse = mx.logsumexp(s, axis=-1)
    o = mx.softmax(s, axis=-1, precise=True) @ vv
    return o, lse


def check(q, k, v, P):
    """Shapes / dtypes the kernel takes; raises ValueError otherwise."""
    if q.ndim != 4 or q.shape[2] != 1 or k.ndim != 4 or k.shape[0] != 1 or v.shape != k.shape:
        raise ValueError(f"cascade_mma: q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)}")
    B, Hq, _, D = q.shape
    Hkv = k.shape[1]
    if Hq % Hkv or k.shape[3] != D or D not in HEAD_DIMS or not 1 <= P <= k.shape[2]:
        raise ValueError(f"cascade_mma: {Hq} q / {Hkv} kv heads, head_dim {D} / {k.shape[3]}, "
                         f"P {P} of {k.shape[2]} slots")
    if q.dtype != mx.float16 or k.dtype != mx.float16 or v.dtype != mx.float16:
        raise ValueError(f"cascade_mma: fp16 only (q {q.dtype}, k {k.dtype}, v {v.dtype})")


def prefix_attention(q, k, v, P, scale, chunk=None):
    """q (B, Hq, 1, D) fp16, k / v (1, Hkv, capP, D) fp16 buffers whose
    [0, P) are the prefix -> (o_p (Hkv, B * rep, D) fp32, lse_p (Hkv, B * rep)
    fp32, natural log). Nothing past P is read. `chunk` (a multiple of 16):
    the microbench's override of pick_chunk."""
    check(q, k, v, P)
    if not use_kernel():
        return reference(q, k, v, P, scale)
    B, Hq, _, D = q.shape
    Hkv, cap = k.shape[1], k.shape[2]
    rep = Hq // Hkv
    m0 = B * rep
    BQ, BK, WM, WN = tiles(D)
    blocks = -(-m0 // BQ)
    chunk = chunk or pick_chunk(P, Hkv, blocks)
    if chunk % BK:
        raise ValueError(f"cascade_mma: chunk {chunk} not a multiple of {BK}")
    n = items(P, chunk)
    meta = _meta(m0, P, cap, chunk)
    p1, p2 = _kernels(D, Hkv, scale)
    # (Hkv, M0, D): kv head major, row b * rep + r (metal_kernel copies it
    # row-contiguous: 128 KiB at 16 rows on the 9B)
    qs = q.reshape(B, Hkv, rep, D).transpose(1, 0, 2, 3).reshape(Hkv, m0, D)
    tg = 32 * WM * WN
    po, pml = p1(inputs=[qs, k, v, meta], grid=(tg * n, Hkv, blocks), threadgroup=(tg, 1, 1),
                 output_shapes=[(n * Hkv * m0 * D,), (n * Hkv * m0 * 2,)],
                 output_dtypes=[F32, F32])
    out, lse = p2(inputs=[po, pml, meta], grid=(D * m0, Hkv, 1), threadgroup=(D, 1, 1),
                  output_shapes=[(Hkv, m0, D), (Hkv, m0)], output_dtypes=[F32, F32])
    return out, lse


def warm(D: int, hkv: int, hq: int, scale: float):
    """Compile both pipelines now (the scheduler thread's first group would
    otherwise pay the Metal compile). Tiny zero inputs: only the layout
    matters (the pipelines take every M0 / P / capP at run time)."""
    if not use_kernel():
        return
    q = mx.zeros((1, hq, 1, D), mx.float16)
    kv = mx.zeros((1, hkv, 16, D), mx.float16)
    mx.eval(prefix_attention(q, kv, kv, 16, scale))
