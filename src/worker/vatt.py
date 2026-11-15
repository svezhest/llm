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
SPLIT = os.environ.get("VATT_SPLIT", "0") == "1"   # K / V / exchange in separate smem regions
BK_MUL = int(os.environ.get("VATT_BK_MUL") or "1")           # > 1: BK = BK_MUL * BQ
Q8_KMAP = os.environ.get("VATT_Q8_KMAP", "c")   # q8 K loader: "c" = threads over a key's units (coalesced reads), "k" = over keys

_IMPL = r"""
// Adapted from mlx 0.32.3 kernels/steel/attn/kernels/steel_attention.h
// (`attention`, MIT) via cascade_mma.cmma_p1: WN = 2, Q in registers, K / V
// strided buffers (head stride, token stride), a causal tail per query row
// (row r < M0 sees keys < N - TQN + 1 + r % TQN), keys [start, start + kL)
// of one chunk; writes the chunk's normalized o (fp32) + m and l.
template <typename T, int BQ, int BK, int BD, int WM, int WN, bool SPLIT, typename KVSRC>
METAL_FUNC void vatt_p1(
    const device T* Q, thread const KVSRC& kv,
    device float* po, device float* pml,
    const int M0, const int N, const int TQN, const int chunk, const int blocks, const int HKV_,
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

  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BV + padV;
  constexpr short tgp_mem_0 = (BK + padK) * (BD);
  constexpr short tgp_mem_1 = BK * (BV + padV);
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;

  // SPLIT: K, V and the score exchange in their own regions (K and V of a
  // block loaded together: 3 barriers a block instead of 6)
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = SPLIT ? KV_smem + tgp_mem_0 : KV_smem;
  threadgroup float* Xs = SPLIT ? reinterpret_cast<threadgroup float*>(KV_smem + tgp_mem_0 + tgp_mem_1)
                                : reinterpret_cast<threadgroup float*>(KV_smem);

  // K transposed (Ks[d * LDK_tgp + key]), V as it is (Vs[key * LDV_tgp + d]), both fp16 in
  // threadgroup memory whatever the source (KVF16: copied; KVQ8: dequantized on the way)
  auto loader_k = kv.template kloader<BK, BD, LDK_tgp, WM * WN * 32>(h, start, Ks, simd_group_id, simd_lane_id);
  auto loader_v = kv.template vloader<BK, BV, LDV_tgp, WM * WN * 32>(h, start, Vs, simd_group_id, simd_lane_id);

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
      if constexpr (SPLIT) loader_v.load_safe(short2(BV, kL_rem));
    } else {
      loader_k.load_unsafe();
      if constexpr (SPLIT) loader_v.load_unsafe();
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
      static_assert(SPLIT || s_xchg_size * sizeof(AccumType) <= tgp_mem_s * sizeof(T),
                    "The score exchange must fit in KV_smem");
      if constexpr (!SPLIT) threadgroup_barrier(mem_flags::mem_threadgroup);
      threadgroup AccumType* s_xchg = Xs;
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
    if constexpr (!SPLIT) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (kb == NK_aligned) {
        loader_v.load_safe(short2(BV, kL_rem));   // zeros past the chunk: 0 * NaN never happens
      } else {
        loader_v.load_unsafe();
      }
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
    if constexpr (!SPLIT) threadgroup_barrier(mem_flags::mem_threadgroup);
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

// fp16 buffers (1, Hkv, cap, D): the steel loaders, as before
template <typename T>
struct KVF16 {
  const device T* k;
  const device T* v;
  size_t ksh, vsh;              // head strides
  int kst, vst;                 // token strides
  template <short BK, short BD, short LD, short TGP>
  METAL_FUNC mlx::steel::BlockLoaderT<T, BK, BD, 1, LD, 0, TGP> kloader(
      int h, int start, threadgroup T* dst, uint sg, uint ln) const thread {
    return mlx::steel::BlockLoaderT<T, BK, BD, 1, LD, 0, TGP>(
        k + size_t(h) * ksh + size_t(start) * kst, kst, dst, sg, ln);
  }
  template <short BK, short BD, short LD, short TGP>
  METAL_FUNC mlx::steel::BlockLoaderT<T, BK, BD, LD, 1, 0, TGP> vloader(
      int h, int start, threadgroup T* dst, uint sg, uint ln) const thread {
    return mlx::steel::BlockLoaderT<T, BK, BD, LD, 1, 0, TGP>(
        v + size_t(h) * vsh + size_t(start) * vst, vst, dst, sg, ln);
  }
};

// A block of BK keys of an 8-bit affine QuantizedKVCache (mx.quantize: 4 values a uint32,
// lowest byte first; one fp16 scale and bias per 64 values) -> fp16 threadgroup memory,
// dequantized in registers: w = scale * q + bias (one fma, rounded to fp16 once). A unit is
// 16 values of one key (a uint4 of packed words, a quarter of a group); consecutive threads
// take consecutive units of one key (16 threads read its 256 contiguous bytes). K is stored
// transposed (TRANS: dst[d * LD + key], scalar stores), V as it is (half4 stores).
// VATT_Q8_KMAP=k: K's threads over consecutive keys instead (fewer smem bank conflicts,
// uncoalesced reads): measured slower (27B layout, 80K keys: decode 1.23 vs 0.89 ms a layer,
// verify-8 3.63 vs 3.42; raw/vatt_q8/sweep2).
template <short BK, short BD, bool TRANS, short LD, short TGP>
struct Q8Loader {
  STEEL_CONST short CPK = BD / 16;                  // units per key
  STEEL_CONST short PASSES = BK * CPK / TGP;
  STEEL_CONST bool KF = TRANS && VATT_Q8_KFAST;      // consecutive threads on consecutive keys
  static_assert(PASSES * TGP == BK * CPK, "a block of units must split evenly over the threads");
  static_assert(KF ? TGP % BK == 0 : TGP % CPK == 0, "thread to unit mapping");
  const device uint32_t* w;
  const device half* s;
  const device half* b;
  const int wt, st, bt;
  threadgroup half* dst;
  const short key0, c0;

  METAL_FUNC Q8Loader(const device uint32_t* w_, const device half* s_, const device half* b_,
                      int wt_, int st_, int bt_, threadgroup half* dst_, uint sg, uint ln) thread
      : w(w_), s(s_), b(b_), wt(wt_), st(st_), bt(bt_), dst(dst_),
        key0(KF ? short((sg * 32 + ln) % BK) : short((sg * 32 + ln) / CPK)),
        c0(KF ? short((sg * 32 + ln) / BK) : short((sg * 32 + ln) % CPK)) {}

  METAL_FUNC void unit(short p, bool valid) const thread {
    const short key = KF ? key0 : short(key0 + p * (TGP / CPK));
    const short c = KF ? short(c0 + p * (TGP / BK)) : c0;
    half v[16];
    if (valid) {
      const uint4 q = *(const device uint4*)(w + key * wt + c * 4);
      const half2 sc = half2(s[key * st + c / 4]);
      const half2 bi = half2(b[key * bt + c / 4]);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 4; i++) {
        const uint x = q[i];
        const half2 lo = fma(as_type<half2>((x & 0x00ff00ffu) | 0x64006400u) - half2(1024.0h), sc, bi);
        const half2 hi = fma(as_type<half2>(((x >> 8) & 0x00ff00ffu) | 0x64006400u) - half2(1024.0h), sc, bi);
        v[4 * i] = lo.x; v[4 * i + 1] = hi.x; v[4 * i + 2] = lo.y; v[4 * i + 3] = hi.y;
      }
    } else {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 16; i++) v[i] = half(0);
    }
    if (TRANS) {
      STEEL_PRAGMA_UNROLL
      for (short d = 0; d < 16; d++) dst[(c * 16 + d) * LD + key] = v[d];
    } else {
      threadgroup half4* o = (threadgroup half4*)(dst + key * LD + c * 16);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 4; i++) o[i] = half4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);
    }
  }
  METAL_FUNC void load_unsafe() const thread {
    STEEL_PRAGMA_UNROLL
    for (short p = 0; p < PASSES; p++) unit(p, true);
  }
  METAL_FUNC void load_safe(short2 dims) const thread {     // dims.y keys valid; zeros past them
    STEEL_PRAGMA_UNROLL
    for (short p = 0; p < PASSES; p++) unit(p, (KF ? key0 : short(key0 + p * (TGP / CPK))) < dims.y);
  }
  METAL_FUNC void next() thread {
    w += BK * wt;
    s += BK * st;
    b += BK * bt;
  }
};

struct KVQ8 {
  const device uint32_t* kw;
  const device half* ks;
  const device half* kb;
  const device uint32_t* vw;
  const device half* vs;
  const device half* vb;
  size_t kwh, ksh, kbh, vwh, vsh, vbh;      // head strides
  int kwt, kst, kbt, vwt, vst, vbt;         // token strides
  template <short BK, short BD, short LD, short TGP>
  METAL_FUNC Q8Loader<BK, BD, true, LD, TGP> kloader(
      int h, int start, threadgroup half* dst, uint sg, uint ln) const thread {
    return Q8Loader<BK, BD, true, LD, TGP>(
        kw + size_t(h) * kwh + size_t(start) * kwt, ks + size_t(h) * ksh + size_t(start) * kst,
        kb + size_t(h) * kbh + size_t(start) * kbt, kwt, kst, kbt, dst, sg, ln);
  }
  template <short BK, short BD, short LD, short TGP>
  METAL_FUNC Q8Loader<BK, BD, false, LD, TGP> vloader(
      int h, int start, threadgroup half* dst, uint sg, uint ln) const thread {
    return Q8Loader<BK, BD, false, LD, TGP>(
        vw + size_t(h) * vwh + size_t(start) * vwt, vs + size_t(h) * vsh + size_t(start) * vst,
        vb + size_t(h) * vbh + size_t(start) * vbt, vwt, vst, vbt, dst, sg, ln);
  }
};
"""

# constants prepended (_consts): D, HKV, BQ, BK, WM, WN, SCALE2 (scale * log2 e)
# meta (int32): M0 (= G * T query rows per kv head), N (keys), T, chunk, items, blocks
_P1 = r"""
    constexpr int KVS = SPLIT ? (BK + 8) * D + BK * (D + 8) + 2 * WM * WN * (BQ / (8 * WM)) * BK * 8
                              : ((BK + 8) * D > BK * (D + 8) ? (BK + 8) * D : BK * (D + 8));
    threadgroup half KV_smem[KVS];
    KVF16<half> kv;
    kv.k = k; kv.v = v;
    kv.ksh = size_t(k_strides[1]); kv.kst = int(k_strides[2]);
    kv.vsh = size_t(v_strides[1]); kv.vst = int(v_strides[2]);
    vatt_p1<half, BQ, BK, D, WM, WN, SPLIT>(q, kv, po, pml, meta[0], meta[1], meta[2], meta[3], meta[5], HKV,
        SCALE2, KV_smem, thread_index_in_simdgroup, simdgroup_index_in_threadgroup,
        threadgroup_position_in_grid);
"""

# the same over an 8-bit QuantizedKVCache's (packed, scales, biases) buffers
_P1Q = _P1.split("    threadgroup half KV_smem[KVS];")[0] + r"""    threadgroup half KV_smem[KVS];
    KVQ8 kv;
    kv.kw = kw; kv.ks = ks; kv.kb = kb; kv.vw = vw; kv.vs = vs; kv.vb = vb;
    kv.kwh = size_t(kw_strides[1]); kv.kwt = int(kw_strides[2]);
    kv.ksh = size_t(ks_strides[1]); kv.kst = int(ks_strides[2]);
    kv.kbh = size_t(kb_strides[1]); kv.kbt = int(kb_strides[2]);
    kv.vwh = size_t(vw_strides[1]); kv.vwt = int(vw_strides[2]);
    kv.vsh = size_t(vs_strides[1]); kv.vst = int(vs_strides[2]);
    kv.vbh = size_t(vb_strides[1]); kv.vbt = int(vb_strides[2]);
    vatt_p1<half, BQ, BK, D, WM, WN, SPLIT>(q, kv, po, pml, meta[0], meta[1], meta[2], meta[3], meta[5], HKV,
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
        _HEADER = (_cm.steel_headers() + _cm._OPS
                   + f"#define VATT_Q8_KFAST {'false' if Q8_KMAP == 'c' else 'true'}\n" + _IMPL)
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
            f"    constexpr bool SPLIT = {'true' if SPLIT else 'false'};\n"
            f"    constexpr float SCALE2 = {scale * math.log2(math.e)!r}f;\n")


def _kernels(D, hkv, scale, bq, bk, wm, wn, q8=False):
    key = (D, hkv, float(scale), bq, bk, wm, wn, q8)
    k = _KERNELS.get(key)
    if k is None:
        with _LOCK:
            k = _KERNELS.get(key)
            if k is None:
                tgp = 32 * wm * wn
                if not (_q8_loader_ok(bk, D, tgp) if q8 else _loader_ok(bk, D, tgp)):
                    raise ValueError(f"vatt: tile {bq}x{bk} does not fit the loaders")
                c = _consts(D, hkv, scale, bq, bk, wm, wn)
                tag = (f"q{bq}k{bk}w{wm}x{wn}{'s' if SPLIT else ''}_d{D}_h{hkv}_s"
                       + float(scale).hex().replace('.', '_').replace('-', 'm').replace('+', 'p'))
                p1 = mx.fast.metal_kernel(
                    name=f"vatt{'q8' if q8 else ''}_p1_{tag}",
                    input_names=(["q", "kw", "ks", "kb", "vw", "vs", "vb", "meta"] if q8
                                 else ["q", "k", "v", "meta"]),
                    output_names=["po", "pml"], source=c + (_P1Q if q8 else _P1), header=_header(),
                    ensure_row_contiguous=False)
                p2 = _KERNELS.get(("p2", D, hkv))
                if p2 is None:
                    p2 = _KERNELS[("p2", D, hkv)] = mx.fast.metal_kernel(
                        name=f"vatt_p2_d{D}_h{hkv}", input_names=["po", "pml", "meta"],
                        output_names=["out"], source=c + _P2)
                k = _KERNELS[key] = (p1, p2)
    return k


def _q8_loader_ok(bk, D, tgp):
    units = bk * D // 16
    return D % 64 == 0 and units % tgp == 0 and tgp % bk == 0 and tgp % (D // 16) == 0


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
    return _run(q, [k, v], k.shape[1], n, scale, chunk, False)


def _run(q, kv_inputs, Hkv, n, scale, chunk, q8):
    _, Hq, T, D = q.shape
    m0 = Hq // Hkv * T
    bq, bk, wm, wn, blocks = tile_for(m0)
    chunk = chunk or pick_chunk(n, Hkv, blocks, bk)
    if chunk % bk:
        raise ValueError(f"vatt: chunk {chunk} not a multiple of {bk}")
    items = -(-n // chunk)
    meta = _meta(m0, n, T, chunk, blocks)
    p1, p2 = _kernels(D, Hkv, scale, bq, bk, wm, wn, q8)
    tg = 32 * wm * wn
    qf = q.reshape(-1)               # (Hkv, M0, D) row-major (a copy when q is a transposed view)
    po, pml = p1(inputs=[qf, *kv_inputs, meta], grid=(tg * items * blocks, Hkv, 1), threadgroup=(tg, 1, 1),
                 output_shapes=[(items * Hkv * m0 * D,), (items * Hkv * m0 * 2,)],
                 output_dtypes=[F32, F32])
    return p2(inputs=[po, pml, meta], grid=(D * m0, Hkv, 1), threadgroup=(D, 1, 1),
              output_shapes=[(1, Hq, T, D)], output_dtypes=[mx.float16])[0]


# ---- 8-bit quantized KV (QWEN_VATT_Q8): a QuantizedKVCache's own buffers ----

Q8_BITS = (8,)
Q8_GROUP = 64


def check_q8(q, kq, vq, n, group_size, bits):
    """kq / vq = (packed uint32 (1, Hkv, cap, D * bits / 32), scales, biases
    fp16 (1, Hkv, cap, D / group_size)) as a QuantizedKVCache holds them."""
    if bits not in Q8_BITS or group_size != Q8_GROUP:
        raise ValueError(f"vatt: {bits}-bit KV in groups of {group_size} (8 / 64 only)")
    if len(kq) != 3 or len(vq) != 3:
        raise ValueError("vatt: quantized K / V must be (packed, scales, biases)")
    if q.ndim != 4 or q.shape[0] != 1 or q.dtype != mx.float16:
        raise ValueError(f"vatt: q {tuple(q.shape)} {q.dtype}")
    _, Hq, T, D = q.shape
    w = kq[0]
    Hkv, cap = w.shape[1], w.shape[2]
    want = ((1, Hkv, cap, D * bits // 32), (1, Hkv, cap, D // group_size), (1, Hkv, cap, D // group_size))
    for x in (kq, vq):
        if tuple(tuple(a.shape) for a in x) != want:
            raise ValueError(f"vatt: quantized KV shapes {[tuple(a.shape) for a in x]}, want {want}")
        if x[0].dtype != mx.uint32 or x[1].dtype != mx.float16 or x[2].dtype != mx.float16:
            raise ValueError(f"vatt: quantized KV dtypes {[a.dtype for a in x]} (uint32 + fp16 scales only)")
    if Hq % Hkv or D not in D_OK or not 1 <= T <= MAX_T or not T <= n <= cap:
        raise ValueError(f"vatt: {Hq} q / {Hkv} kv heads, head_dim {D}, T {T}, n {n} of {cap} slots")


def dequantized(x, n, group_size=Q8_GROUP, bits=8):
    """(packed, scales, biases) -> fp16 (1, Hkv, n, D) (mx.dequantize of [0, n))."""
    return mx.dequantize(*(a[..., :n, :] for a in x), group_size=group_size, bits=bits)


def reference_q8(q, kq, vq, n, scale, group_size=Q8_GROUP, bits=8):
    """fp32 reference over the dequantized KV -> (1, Hq, T, D) fp32."""
    return reference(q, dequantized(kq, n, group_size, bits), dequantized(vq, n, group_size, bits), n, scale)


def attention_q8(q, kq, vq, n, scale, group_size=Q8_GROUP, bits=8, chunk=None):
    """`attention` over an 8-bit QuantizedKVCache: kq / vq the cache's
    (packed, scales, biases) buffers (1, Hkv, cap, ...), keys [0, n) valid,
    read in place and dequantized in registers into the kernel's fp16
    threadgroup tiles (the rest of the kernel is the fp16 one)."""
    check_q8(q, kq, vq, n, group_size, bits)
    if not use_kernel():
        return reference_q8(q, kq, vq, n, scale, group_size, bits).astype(mx.float16)
    return _run(q, [*kq, *vq], kq[0].shape[1], n, scale, chunk, True)


def partial_bytes(n: int, hq: int, T: int, D: int = 256) -> int:
    """Upper bound of one call's pass-1 partials (every chunk >= MIN_CHUNK keys)."""
    return (-(-n // MIN_CHUNK) + 1) * hq * T * (D + 2) * 4
