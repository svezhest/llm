"""One-pass attention of a DFlash verify block over a long fp16 KV (QWEN_VERIFY_ATT=2).

A verify block is the last T positions of a lone row (T <= MAX_T; 8 with the
27B's DFlash2 drafter): query i attends keys [0, n - T + i + 1) of the row's
KVCache. mlx-vlm runs T one-query SDPA calls per full-attention layer, i.e.
reads the whole KV T times; QWEN_VERIFY_ATT=1 (kernels._block_attention) packs
32 // G queries per call into MLX's vector kernel (5 + 3 on the 27B, G = 6), so
the KV is still read twice a round, at ~80 GB/s (60K keys: 8.1 ms a layer vs
2.9 for one decode query). Here the G * T query rows of each kv head (48 on the
27B at T = 8) attend the KV together, read once:

  pass 1 (grid items x query blocks x Hkv threadgroups, items = key chunks,
         query blocks of BQ = 16 rows; the blocks of one chunk are adjacent
         threadgroups, so their K / V reads meet in the GPU's caches):
         MLX's steel attention loop (cascade_mma.cmma_p1 = mlx 0.32.3
         steel_attention.h `attention`: loaders, 8x8 simdgroup MMA, WN = 2
         score exchange, online softmax in the log2 domain) over one chunk of
         keys, with the block's causal tail -- row r = g * T + t (query head
         h * G + g, block position t) masks keys >= n - T + t + 1 -- writing the
         chunk's normalized fp32 output, its max m and sum l (split-KV /
         flash-decoding). Q stays in registers (QREG), K / V blocks of BK keys
         are staged in threadgroup memory. S = Q K^T is fp16 x fp16 MMA with
         fp32 accumulation, P V multiplies the fp32 probabilities by fp16 V
         fragments, fp32 accumulation. Small threadgroups win (4 simdgroups,
         BK = 16): see tile_for.
  pass 2 (grid G * T x Hkv threadgroups of D threads): o = sum_c w_c o_c /
         sum_c w_c, w_c = l_c 2^(m_c - M) -> fp16, written as (1, Hq, T, D).

K / V are read where they are: the KVCache buffers (1, Hkv, cap, D) are
passed as they are (ensure_row_contiguous=False; head / token strides from
the arrays, the head axis contiguous in its last dimension as every KVCache
buffer is), so nothing is copied and nothing past n is read. Numerics: the
class of the cascade kernel / MLX's fused attention (fp32 everywhere, no fp16
partials -- MLX's 2-pass vector kernel keeps fp16 partials), not the bits of
the per-query calls; tests/test_vatt.py and the load-time self-test
(kernels._probe_verify_attention) compare against an fp32 reference.

    out = vatt.attention(q, k, v, n, scale)
    # q (1, Hq, T, D) fp16 (the block's queries after RoPE), k / v (1, Hkv,
    # cap, D) fp16 buffers whose [0, n) are valid -> (1, Hq, T, D) fp16

On a CPU default device `attention` computes the same with mx ops in fp32
(`reference`), so the worker's CPU tests run the path unchanged.
"""
from __future__ import annotations

import math
import os
import threading

import mlx.core as mx

import cascade_mma as _cm

F32 = mx.float32
D_OK = (256,)         # the WN = 2 score exchange fits KV_smem only for D = 256 at these tiles
MAX_T = 16            # block queries per call
MIN_CHUNK = 256
MAX_CHUNK = 4096
TARGET_TG = int(os.environ.get("VATT_TARGET_TG") or "256")   # pass-1 threadgroups aimed at
# (BQ, BK, WM, WN) override for the microbench: "BQxBK" (WM = BQ / 8, WN = 2), "BQxBKw1" (WN = 1:
# a simdgroup's 8 rows over the whole head_dim, no score exchange)
TILE = os.environ.get("VATT_TILE", "")
BK_MUL = int(os.environ.get("VATT_BK_MUL") or "1")           # > 1: BK = BK_MUL * BQ

_IMPL = r"""
// Adapted from mlx 0.32.3 kernels/steel/attn/kernels/steel_attention.h
// (`attention`, MIT) via cascade_mma.cmma_p1: WN = 2, Q in registers, K / V
// strided buffers (head stride, token stride), a causal tail per query row
// (row r < M0 sees keys < N - TQN + 1 + r % TQN), keys [start, start + kL)
// of one chunk; writes the chunk's normalized o (fp32) + m and l.
template <typename T, int BQ, int BK, int BD, int WM, int WN>
METAL_FUNC void vatt_p1(
    const device T* Q, const device T* K, const device T* V,
    device float* po, device float* pml,
    const int M0, const int N, const int TQN, const int chunk, const int blocks, const int HKV_,
    const size_t ksh, const int kst, const size_t vsh, const int vst,
    const float scale,
    threadgroup T* KV_smem,
    uint simd_lane_id, uint simd_group_id, uint3 tid) {
  using namespace mlx::steel;
  using AccumType = float;
  constexpr int BV = BD;
  const int item = int(tid.x) / blocks, qb = int(tid.x) % blocks, h = int(tid.y);
  const int q0 = qb * BQ;
  const int q_rows = min(BQ, M0 - q0);
  const int start = item * chunk;
  const int kL = min(N, start + chunk) - start;
  const int NK = (kL + BK - 1) / BK;
  const int NK_aligned = kL / BK;
  const int kL_rem = kL - NK_aligned * BK;
  const int lim0 = N - TQN + 1;                 // row t = 0 sees keys < lim0

  Q += (size_t(h) * M0 + q0) * BD;
  K += size_t(h) * ksh + size_t(start) * kst;
  V += size_t(h) * vsh + size_t(start) * vst;

  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BV + padV;
  constexpr short tgp_mem_0 = (BK + padK) * (BD);
  constexpr short tgp_mem_1 = BK * (BV + padV);
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;

  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  using KBlockLoader = BlockLoaderT<T, BK, BD, 1, LDK_tgp, 0, WM * WN * 32>;   // K transposed
  using VBlockLoader = BlockLoaderT<T, BK, BV, LDV_tgp, 1, 0, WM * WN * 32>;
  KBlockLoader loader_k(K, kst, Ks, simd_group_id, simd_lane_id);
  VBlockLoader loader_v(V, vst, Vs, simd_group_id, simd_lane_id);

  constexpr short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;
  using MMAFrag_in_t = BaseMMAFrag<T, kFragSize, kFragSize>;
  static_assert(WN == 1 || WN == 2, "WN = 1 or 2");
  constexpr int TQ = BQ / (WM * kFragSize);
  static_assert(TQ >= 1 && BQ == WM * kFragSize * TQ, "TQ 8-row fragments per simdgroup");
  constexpr int TK = BK / kFragSize;
  constexpr int TD = BD / kFragSize;
  constexpr int TDh = TD / WN;
  constexpr int BVh = BV / WN;
  constexpr int TVh = BV / (kFragSize * WN);

  MMATile<T, TQ, 1, MMAFrag_in_t> Qtiles[TDh];
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
  const short Ks_offset = d_half * (BD / WN) * LDK_tgp + sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + d_half * BVh + sn;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  STEEL_PRAGMA_UNROLL
  for (short dd = 0; dd < TDh; dd++) {
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      const int r = tm + sm + iq * kFragSize;
      const device T* q_src = Q + r * BD + d_half * (BD / WN) + dd * kFragSize + sn;
      if (q_rows < BQ) {
        MMAFrag_in_t::load_safe(Qtiles[dd].frag_at(iq, 0), q_src, BD, Int<1>{},
                                q_rows - r, (BD / WN) - (dd * kFragSize + sn), Int<0>{}, Int<0>{});
      } else {
        MMAFrag_in_t::load(Qtiles[dd].frag_at(iq, 0), q_src, BD, Int<1>{});
      }
    }
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
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(&Ks[Ks_offset + dd * Ks_tile_stride]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(Stile, Qtiles[dd], Ktile, Stile);
    }
    if constexpr (WN == 2) {   // each simdgroup summed half of D; exchange the partial scores
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
    const int kbase = start + kb * BK;
    if (kb == NK_aligned || kbase + BK > lim0) {     // past the chunk / the block's causal tail
      using stile_t = decltype(Stile);
      constexpr auto neg_inf = Limits<AccumType>::finite_min;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int r = q0 + tm + sm + i * kFragSize;
        const int lim = min(r < M0 ? lim0 + r % TQN : N, start + kL) - kbase;
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          short col_pos = sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((col_pos + jj) >= lim) Stile.frag_at(i, j)[jj] = neg_inf;
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kb == NK_aligned) {
      loader_v.load_safe(short2(BV, kL_rem));   // zeros past the chunk: 0 * NaN never happens
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
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++)
          MMAFrag_acc_t::mma(Otile.frag_at(iq, id), Stile.frag_at(iq, ik), Vtile.frag_at(0, 0),
                             Otile.frag_at(iq, id));
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
  if (sn == 0 && d_half == 0) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      const int ri = row + i * kFragSize;
      if (ri < q_rows) {
        pml[2 * (p0 + ri)] = max_score[i];
        pml[2 * (p0 + ri) + 1] = sum_score[i];
      }
    }
  }
}
"""

# constants prepended (_consts): D, HKV, BQ, BK, WM, WN, SCALE2 (scale * log2 e)
# meta (int32): M0 (= G * T query rows per kv head), N (keys), T, chunk, items, blocks
_P1 = r"""
    constexpr int KVS = (BK + 8) * D > BK * (D + 8) ? (BK + 8) * D : BK * (D + 8);
    threadgroup half KV_smem[KVS];
    vatt_p1<half, BQ, BK, D, WM, WN>(q, k, v, po, pml, meta[0], meta[1], meta[2], meta[3], meta[5], HKV,
        size_t(k_strides[1]), int(k_strides[2]), size_t(v_strides[1]), int(v_strides[2]),
        SCALE2, KV_smem, thread_index_in_simdgroup, simdgroup_index_in_threadgroup,
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
    out[(size_t(h) * M0 + r) * D + d] = half(acc / L);
"""

_HEADER = None
_KERNELS: dict = {}
_META: dict = {}
_LOCK = threading.Lock()


def _header() -> str:
    global _HEADER
    if _HEADER is None:
        _HEADER = _cm.steel_headers() + _cm._OPS + _IMPL
    return _HEADER


def tile_for(m0: int):
    """(BQ, BK, WM, WN, blocks) for M0 query rows per kv head: query blocks of
    BQ = 16 rows (8 when M0 <= 8: a decode step's G rows), WM = BQ / 8
    simdgroup rows x WN = 2, BK = 16 keys; the blocks of one chunk are
    adjacent threadgroups (their K / V reads meet in the cache). Measured
    2026-11-16 at 60K keys, 27B layout, T = 8 (raw/sweep): 16x16 2.41-2.52 ms
    a layer; 16x32 2.50; 24x24 2.62; 48x24 2.81-3.08; 48x48 3.08-3.27; 32x32
    4.23; 8x16 3.31; WN = 1 tiles 2.76-8.5."""
    wn = 2
    if TILE:
        t = TILE.split("w")
        bq, bk = (int(x) for x in t[0].split("x"))
        wn = int(t[1]) if len(t) > 1 else 2
    else:
        bq = 8 if m0 <= 8 else 16
        bk = max(16, BK_MUL * bq) if BK_MUL > 1 else 16
    blocks = -(-m0 // bq)
    return bq, bk, bq // 8, wn, blocks


def _loader_ok(bk, D, tgp):
    n_reads, rem = divmod(bk * D, tgp)
    if rem or not n_reads or D % n_reads:
        return False
    tcols = D // n_reads
    return tgp % tcols == 0 and bk % (tgp // tcols) == 0


def _consts(D, hkv, scale, bq, bk, wm, wn):
    return (f"constexpr int D = {D}, HKV = {hkv}, BQ = {bq}, BK = {bk}, WM = {wm}, WN = {wn};\n"
            f"    constexpr float SCALE2 = {scale * math.log2(math.e)!r}f;\n")


def _kernels(D, hkv, scale, bq, bk, wm, wn):
    key = (D, hkv, float(scale), bq, bk, wm, wn)
    k = _KERNELS.get(key)
    if k is None:
        with _LOCK:
            k = _KERNELS.get(key)
            if k is None:
                if not _loader_ok(bk, D, 32 * wm * wn):
                    raise ValueError(f"vatt: tile {bq}x{bk} does not fit the loaders")
                c = _consts(D, hkv, scale, bq, bk, wm, wn)
                tag = (f"q{bq}k{bk}w{wm}x{wn}_d{D}_h{hkv}_s"
                       + float(scale).hex().replace('.', '_').replace('-', 'm').replace('+', 'p'))
                p1 = mx.fast.metal_kernel(
                    name=f"vatt_p1_{tag}", input_names=["q", "k", "v", "meta"],
                    output_names=["po", "pml"], source=c + _P1, header=_header(),
                    ensure_row_contiguous=False)
                p2 = mx.fast.metal_kernel(
                    name=f"vatt_p2_d{D}_h{hkv}", input_names=["po", "pml", "meta"],
                    output_names=["out"], source=c + _P2)
                k = _KERNELS[key] = (p1, p2)
    return k


def pick_chunk(n: int, hkv: int, blocks: int, bk: int) -> int:
    """Keys per pass-1 item: the largest power of two in [MIN_CHUNK, MAX_CHUNK]
    that keeps items x Hkv x blocks >= TARGET_TG, rounded up to a multiple of BK."""
    c = MAX_CHUNK
    while c > MIN_CHUNK and -(-n // c) * hkv * blocks < TARGET_TG:
        c //= 2
    return -(-c // bk) * bk


def _meta(m0, n, T, chunk, blocks):
    key = (m0, n, T, chunk, blocks)
    hit = _META.get(key)
    if hit is None:
        hit = mx.array([m0, n, T, chunk, -(-n // chunk), blocks], dtype=mx.int32)
        with _LOCK:
            if len(_META) >= 64:
                _META.pop(next(iter(_META)))
            _META[key] = hit
    return hit


def use_kernel() -> bool:
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def check(q, k, v, n):
    if q.ndim != 4 or q.shape[0] != 1 or k.ndim != 4 or k.shape[0] != 1 or v.shape != k.shape:
        raise ValueError(f"vatt: q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)}")
    _, Hq, T, D = q.shape
    Hkv = k.shape[1]
    if Hq % Hkv or k.shape[3] != D or D not in D_OK or not 1 <= T <= MAX_T or not T <= n <= k.shape[2]:
        raise ValueError(f"vatt: {Hq} q / {Hkv} kv heads, head_dim {D} / {k.shape[3]}, T {T}, "
                         f"n {n} of {k.shape[2]} slots")
    if q.dtype != mx.float16 or k.dtype != mx.float16 or v.dtype != mx.float16:
        raise ValueError(f"vatt: fp16 only (q {q.dtype}, k {k.dtype}, v {v.dtype})")


def reference(q, k, v, n, scale):
    """fp32 mx ops: the kernel's output before its fp16 rounding (the CPU path
    and the tests' reference). -> (1, Hq, T, D) fp32."""
    _, Hq, T, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    qq = q.astype(F32).reshape(Hkv, G * T, D)
    kk = k[0, :, :n].astype(F32)
    vv = v[0, :, :n].astype(F32)
    s = (qq * scale) @ kk.swapaxes(-1, -2)                         # (Hkv, G*T, n)
    lim = (n - T + 1 + mx.arange(T))                               # row g*T+t sees keys < lim[t]
    mask = mx.arange(n)[None, :] < mx.tile(lim, (G,))[:, None]     # (G*T, n)
    s = mx.where(mask[None], s, -mx.inf)
    return (mx.softmax(s, axis=-1, precise=True) @ vv).reshape(1, Hq, T, D)


def attention(q, k, v, n, scale, chunk=None):
    """q (1, Hq, T, D) fp16 (T <= MAX_T), k / v (1, Hkv, cap, D) fp16 buffers
    (last axis contiguous), keys [0, n) valid; query t attends [0, n - T + t + 1)
    -> (1, Hq, T, D) fp16. `chunk`: the microbench's override (a multiple of BK)."""
    check(q, k, v, n)
    if not use_kernel():
        return reference(q, k, v, n, scale).astype(mx.float16)
    _, Hq, T, D = q.shape
    Hkv = k.shape[1]
    m0 = Hq // Hkv * T
    bq, bk, wm, wn, blocks = tile_for(m0)
    chunk = chunk or pick_chunk(n, Hkv, blocks, bk)
    if chunk % bk:
        raise ValueError(f"vatt: chunk {chunk} not a multiple of {bk}")
    items = -(-n // chunk)
    meta = _meta(m0, n, T, chunk, blocks)
    p1, p2 = _kernels(D, Hkv, scale, bq, bk, wm, wn)
    tg = 32 * wm * wn
    qf = q.reshape(-1)               # (Hkv, M0, D) row-major (a copy when q is a transposed view)
    po, pml = p1(inputs=[qf, k, v, meta], grid=(tg * items * blocks, Hkv, 1), threadgroup=(tg, 1, 1),
                 output_shapes=[(items * Hkv * m0 * D,), (items * Hkv * m0 * 2,)],
                 output_dtypes=[F32, F32])
    return p2(inputs=[po, pml, meta], grid=(D * m0, Hkv, 1), threadgroup=(D, 1, 1),
              output_shapes=[(1, Hq, T, D)], output_dtypes=[mx.float16])[0]


def partial_bytes(n: int, hq: int, T: int, D: int = 256) -> int:
    """Upper bound of one call's pass-1 partials (every chunk >= MIN_CHUNK keys)."""
    return (-(-n // MIN_CHUNK) + 1) * hq * T * (D + 2) * 4
