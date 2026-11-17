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

8-bit quantized KV (QWEN_VATT_Q8, `attention_q8`): the same two passes over a
QuantizedKVCache's (packed, scales, biases) buffers. Pass 1 is one of three
kernels (QWEN_VATT_Q8_KERNEL): "tile" = vatt_p1 with Q8Loader dequantizing
into its threadgroup tiles (every 16-row query block of a chunk converts its
key tiles again); "reg" = vattq_p1, K / V fragments built in registers from
the packed bytes as mrqmm.py builds its weights (no tiles, one barrier a key
block, one conversion for the MB row blocks of a simdgroup), the same fp16
values; "regf" = reg with the MMAs on the codes, scale and bias applied to
the sums (the affine values unrounded: `reference_q8(exact=True)`). reg and
regf were written 2026-11-17 without a GPU: not compiled or measured until
docs/perf/2026-11-16/vatt_q8.md's last section is filled.
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
# microbench only, WRONG numbers (attention_q8(timing=) per call; a kernel per value): 1 = no
# scale / bias (no reads of them, no fma), 2 = 1 without the offset subtraction, 3 = no KV reads
# at all (constants: the MMAs, the tile stores / fragment builds and the merge alone)
Q8_TIMING = int(os.environ.get("VATT_Q8_TIMING") or "0")
# q8 pass 1 (QWEN_VATT_Q8_KERNEL; attention_q8(kernel=) per call): "tile" = Q8Loader into the fp16
# kernel's threadgroup tiles (every 16-row query block converts its key tiles); "reg" = vattq_p1,
# fragments built in registers, one conversion for the MB row blocks of a simdgroup, the same fp16
# values; "regf" = reg with scale and bias taken out of the sums (the affine values unrounded)
Q8_KERNELS = ("tile", "reg", "regf")
Q8_KERNEL = os.environ.get("QWEN_VATT_Q8_KERNEL") or "tile"
# "reg" tiles (MB 8-row query blocks a simdgroup, WN simdgroups splitting head_dim, KF 8-key
# fragments a key block) for a decode step (<= 8 query rows a kv head) / a verify block; the
# microbench reassigns them. Unmeasured guesses (2026-11-17: written without a GPU): decode =
# today's 2 simdgroups x 8 rows; verify = 16 rows a threadgroup as the tile kernel, one group of
# 64 dims a simdgroup (16 O fragments in registers, as the tile kernel holds)
Q8R_DECODE = tuple(int(x) for x in (os.environ.get("VATT_Q8R_DECODE") or "1x2x1").split("x"))
Q8R_VERIFY = tuple(int(x) for x in (os.environ.get("VATT_Q8R_VERIFY") or "2x4x1").split("x"))
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
    if (valid && VATT_Q8_TIMING == 3) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 16; i++) v[i] = half(0.01h * half(c + 1));
    } else if (valid && VATT_Q8_TIMING != 0) {
      const uint4 q = *(const device uint4*)(w + key * wt + c * 4);
      const half2 off = VATT_Q8_TIMING == 2 ? half2(0.0h) : half2(1024.0h);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 4; i++) {
        const uint x = q[i];
        const half2 lo = as_type<half2>((x & 0x00ff00ffu) | 0x64006400u) - off;
        const half2 hi = as_type<half2>(((x >> 8) & 0x00ff00ffu) | 0x64006400u) - off;
        v[4 * i] = lo.x; v[4 * i + 1] = hi.x; v[4 * i + 2] = lo.y; v[4 * i + 3] = hi.y;
      }
    } else if (valid) {
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

// Pass 1 over an 8-bit QuantizedKVCache with no threadgroup tiles (QWEN_VATT_Q8_KERNEL=reg /
// regf): every MMA fragment of K and V is built in registers from the packed bytes, as
// mrqmm.py builds its weights. A threadgroup = WN simdgroups x 8 * MB query rows, key blocks
// of 8 * KF keys; a simdgroup owns GS = 4 / WN of the 4 groups of 64 dims (scores: its
// partial sum over them, exchanged once per key block -- WN = 1: no exchange, no barrier;
// values: its 64 * GS output dims). Per 8 keys of a block and group g, MMA step st = 0..7
// takes the dims d = 64 g + 8 c + st of fragment column / row c:
//   S (rows x keys) += Q frag (row fm; d-slots fn, fn + 1: registers) x K^T frag (d-slot fm;
//     keys fn, fn + 1: byte st of the 8 bytes at 64 g + 8 fm of each key, one uint2 a key
//     and group -> 16-bit lanes once per group, a shift and a mask per step)
//   O[g, st] (rows x d-slots) += P frag (rows x keys) x V frag (key fm; d-slots fn, fn + 1:
//     byte st of the low / high half of the uint4 at 64 g + 8 fn of the key)
// K's fragment is built once per step for all MB row blocks of the simdgroup, V's too.
// FACT = false ("reg"): dequantized as the tile loader does, (0x6400 | byte) - 1024 and one
// fp16 fma with the group's scale and bias: the same fp16 values, summed in another order.
// FACT = true ("regf"): the MMAs run on the centred codes c = byte - 128 (exact in fp16);
// scale and bias come out of the sums, in fp32, once per (8 keys, group):
//   S += s[k, g] * (q_g . c[k, g]) + (b[k, g] + 128 s[k, g]) * sum(q_g)
//   O_g += (P s[., g]) c[., g] + sum_k P[r, k] (b[k, g] + 128 s[k, g])     (the last per row)
// i.e. attention over the affine values s * code + b unrounded -- not mx.dequantize's fp16
// values (reference_q8(exact=True) is its reference). Keys past N are never read (the index
// is clamped; their scores are masked, so the clamped key's values get weight 0). The
// exchange slots alternate between two buffers: one barrier a block. Same split-KV outputs
// as vatt_p1. VATT_Q8_TIMING (microbench, wrong numbers): 1 = no scale / bias, 2 = 1 without
// the offset subtraction, 3 = no KV reads.
template <int MB, int WN, int KF, bool FACT>
METAL_FUNC void vattq_p1(
    const device half* Q, thread const KVQ8& kv,
    device float* po, device float* pml,
    const int M0, const int N, const int TQN, const int chunk, const int blocks, const int HKV_,
    const float scale,
    threadgroup float2* X,
    const uint lid, const uint sg, const uint3 tid) {
  using namespace mlx::steel;
  constexpr int BD = 256, GS = 4 / WN, NS = 8 * GS, BQ = 8 * MB, BK = 8 * KF;
  constexpr bool FA = FACT && VATT_Q8_TIMING == 0;
  static_assert(WN == 1 || WN == 2 || WN == 4, "WN = 1, 2 or 4");
  const int item = int(tid.x) / blocks, qb = int(tid.x) % blocks, h = int(tid.y);
  const int q0 = qb * BQ;
  const int start = item * chunk;
  const int kend = min(N, start + chunk);
  const int lim0 = N - TQN + 1;                 // row t = 0 sees keys < lim0
  const short qid = short(lid) / 4;
  const short fm = (qid & 4) + ((short(lid) / 2) % 4);      // fragment row
  const short fn = (qid & 2) * 2 + (short(lid) % 2) * 2;    // fragment columns fn, fn + 1
  const int g0 = int(sg) * GS;
  // the code as fp16: (0x6400 | byte) = 1024 + byte, minus OFF
  const half2 OFF = half2(VATT_Q8_TIMING == 2 ? 0.0h : (FA ? 1152.0h : 1024.0h));

  const device uint8_t* kw = (const device uint8_t*)(kv.kw + size_t(h) * kv.kwh);
  const device uint8_t* vw = (const device uint8_t*)(kv.vw + size_t(h) * kv.vwh);
  const device half* ks = kv.ks + size_t(h) * kv.ksh;
  const device half* kb = kv.kb + size_t(h) * kv.kbh;
  const device half* vs = kv.vs + size_t(h) * kv.vsh;
  const device half* vb = kv.vb + size_t(h) * kv.vbh;
  const size_t kwt = size_t(kv.kwt) * 4, vwt = size_t(kv.vwt) * 4;     // token strides, bytes

  half2 qf[MB][NS];                             // Q fragments: step i = 8 * gi + st
  float qs[MB][GS];                             // FACT: sum of the row's q over group gi
  STEEL_PRAGMA_UNROLL
  for (int mb = 0; mb < MB; ++mb) {
    const int r = q0 + 8 * mb + fm;
    const device half* qr = Q + (size_t(h) * M0 + min(r, M0 - 1)) * BD + 64 * g0 + 8 * fn;
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < NS; ++i) {
      const int o = 64 * (i / 8) + (i % 8);
      qf[mb][i] = r < M0 ? half2(qr[o], qr[o + 8]) : half2(0.0h);
    }
    STEEL_PRAGMA_UNROLL
    for (int gi = 0; gi < GS; ++gi) {
      qs[mb][gi] = 0.0f;
      if (FA) {                                 // Q frag x ones: the row's sum in every column
        simdgroup_matrix<float, 8, 8> C = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<half, 8, 8> I = make_filled_simdgroup_matrix<half, 8, 8>(1.0h);
        STEEL_PRAGMA_UNROLL
        for (int st = 0; st < 8; ++st) {
          simdgroup_matrix<half, 8, 8> A;
          A.thread_elements()[0] = qf[mb][8 * gi + st].x;
          A.thread_elements()[1] = qf[mb][8 * gi + st].y;
          simdgroup_multiply_accumulate(C, A, I, C);
        }
        qs[mb][gi] = C.thread_elements()[0];
      }
    }
  }

  simdgroup_matrix<float, 8, 8> O[MB][NS];
  float ob[MB][GS];                             // FACT: sum_k P[row, k] b'[k, g]
  float max_score[MB], sum_score[MB];
  STEEL_PRAGMA_UNROLL
  for (int mb = 0; mb < MB; ++mb) {
    max_score[mb] = Limits<float>::finite_min;
    sum_score[mb] = 0.0f;
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < NS; ++i) O[mb][i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    STEEL_PRAGMA_UNROLL
    for (int gi = 0; gi < GS; ++gi) ob[mb][gi] = 0.0f;
  }

  int par = 0;
  for (int kbase = start; kbase < kend; kbase += BK, par ^= 1) {
    float2 s[MB][KF];                           // this simdgroup's partial scores
    STEEL_PRAGMA_UNROLL
    for (int kf = 0; kf < KF; ++kf) {
      const size_t ka = size_t(min(kbase + 8 * kf + fn, N - 1)), kc = size_t(min(kbase + 8 * kf + fn + 1, N - 1));
      simdgroup_matrix<float, 8, 8> S[MB];
      STEEL_PRAGMA_UNROLL
      for (int mb = 0; mb < MB; ++mb) {
        S[mb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        s[mb][kf] = float2(0.0f);
      }
      STEEL_PRAGMA_UNROLL
      for (int gi = 0; gi < GS; ++gi) {
        const int g = g0 + gi;
#if VATT_Q8_TIMING == 3
        const uint2 a = uint2(0x01020304u + uint(fm)), c = uint2(0x04030201u + uint(fn));
#else
        const uint2 a = *(const device uint2*)(kw + ka * kwt + 64 * g + 8 * fm);
        const uint2 c = *(const device uint2*)(kw + kc * kwt + 64 * g + 8 * fm);
#endif
        // 16-bit lanes (key fn, key fn + 1), word j = bytes 2j, 2j + 1 of each
        const uint4 w = uint4(insert_bits(a.x, c.x, 16u, 16u), insert_bits(c.x, a.x >> 16, 0u, 16u),
                              insert_bits(a.y, c.y, 16u, 16u), insert_bits(c.y, a.y >> 16, 0u, 16u));
#if VATT_Q8_TIMING == 0
        const half2 sc = half2(ks[ka * kv.kst + g], ks[kc * kv.kst + g]);
        const half2 bi = half2(kb[ka * kv.kbt + g], kb[kc * kv.kbt + g]);
#else
        const half2 sc = half2(0.01h), bi = half2(0.0h);
#endif
        STEEL_PRAGMA_UNROLL
        for (int st = 0; st < 8; ++st) {
          const uint u = (w[st / 2] >> uint(8 * (st % 2))) & 0x00ff00ffu;
          const half2 cd = as_type<half2>(u | 0x64006400u) - OFF;
#if VATT_Q8_TIMING == 0
          const half2 e = FA ? cd : fma(cd, sc, bi);
#else
          const half2 e = cd * sc;
#endif
          simdgroup_matrix<half, 8, 8> B;
          B.thread_elements()[0] = e.x;
          B.thread_elements()[1] = e.y;
          STEEL_PRAGMA_UNROLL
          for (int mb = 0; mb < MB; ++mb) {
            simdgroup_matrix<half, 8, 8> A;
            A.thread_elements()[0] = qf[mb][8 * gi + st].x;
            A.thread_elements()[1] = qf[mb][8 * gi + st].y;
            simdgroup_multiply_accumulate(S[mb], A, B, S[mb]);
          }
        }
        if (FA) {                               // the group's sums -> scores, restart the sums
          const float2 fs = float2(sc), fb = fma(float2(128.0f), fs, float2(bi));
          STEEL_PRAGMA_UNROLL
          for (int mb = 0; mb < MB; ++mb) {
            const float2 t = float2(S[mb].thread_elements()[0], S[mb].thread_elements()[1]);
            s[mb][kf] += fma(t, fs, qs[mb][gi] * fb);
            S[mb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
          }
        }
      }
      if (!FA) {
        STEEL_PRAGMA_UNROLL
        for (int mb = 0; mb < MB; ++mb)
          s[mb][kf] = float2(S[mb].thread_elements()[0], S[mb].thread_elements()[1]);
      }
    }

    if (WN > 1) {     // every simdgroup adds the WN partial scores in the same order: the same bits in all
      threadgroup float2* xs = X + par * (WN * MB * KF * 32);
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < MB * KF; ++j) xs[(int(sg) * MB * KF + j) * 32 + lid] = s[j / KF][j % KF];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < MB * KF; ++j) {
        float2 t = xs[j * 32 + lid];
        STEEL_PRAGMA_UNROLL
        for (int p = 1; p < WN; ++p) t += xs[(p * MB * KF + j) * 32 + lid];
        s[j / KF][j % KF] = t;
      }
    }
    STEEL_PRAGMA_UNROLL
    for (int j = 0; j < MB * KF; ++j) s[j / KF][j % KF] *= scale;
    if (kbase + BK > min(lim0, kend)) {           // past the chunk / the block's causal tail
      STEEL_PRAGMA_UNROLL
      for (int mb = 0; mb < MB; ++mb) {
        const int r = q0 + 8 * mb + fm;
        const int lim = min(r < M0 ? lim0 + r % TQN : N, kend) - kbase;
        STEEL_PRAGMA_UNROLL
        for (int kf = 0; kf < KF; ++kf) {
          if (8 * kf + fn >= lim) s[mb][kf].x = Limits<float>::finite_min;
          if (8 * kf + fn + 1 >= lim) s[mb][kf].y = Limits<float>::finite_min;
        }
      }
    }

    float2 pr[MB][KF];                            // probabilities: row fm; keys fn, fn + 1
    STEEL_PRAGMA_UNROLL
    for (int mb = 0; mb < MB; ++mb) {             // online softmax, log2 domain (as vatt_p1)
      float t = max(s[mb][0].x, s[mb][0].y);
      STEEL_PRAGMA_UNROLL
      for (int kf = 1; kf < KF; ++kf) t = max(t, max(s[mb][kf].x, s[mb][kf].y));
      t = max(t, simd_shuffle_xor(t, ushort(1)));
      t = max(t, simd_shuffle_xor(t, ushort(8)));
      const float new_max = max(max_score[mb], t);
      const float factor = fast::exp2(max_score[mb] - new_max);
      max_score[mb] = new_max;
      float u = 0.0f;
      STEEL_PRAGMA_UNROLL
      for (int kf = 0; kf < KF; ++kf) {
        pr[mb][kf] = fast::exp2(s[mb][kf] - new_max);
        u += pr[mb][kf].x + pr[mb][kf].y;
      }
      u += simd_shuffle_xor(u, ushort(1));
      u += simd_shuffle_xor(u, ushort(8));
      sum_score[mb] = sum_score[mb] * factor + u;
      if (factor != 1.0f) {
        STEEL_PRAGMA_UNROLL
        for (int i = 0; i < NS; ++i) {
          O[mb][i].thread_elements()[0] *= factor;
          O[mb][i].thread_elements()[1] *= factor;
        }
        STEEL_PRAGMA_UNROLL
        for (int gi = 0; gi < GS; ++gi) ob[mb][gi] *= factor;
      }
    }

    STEEL_PRAGMA_UNROLL
    for (int kf = 0; kf < KF; ++kf) {
      const size_t kk = size_t(min(kbase + 8 * kf + fm, N - 1));
      // FACT: the P fragment's keys (its columns), for their scales and biases
      const size_t ka = size_t(min(kbase + 8 * kf + fn, N - 1)), kc = size_t(min(kbase + 8 * kf + fn + 1, N - 1));
      STEEL_PRAGMA_UNROLL
      for (int gi = 0; gi < GS; ++gi) {
        const int g = g0 + gi;
#if VATT_Q8_TIMING == 3
        const uint4 a = uint4(0x01020304u + uint(fm), 0x04030201u, 0x01010101u, 0x02020202u + uint(fn));
#else
        const uint4 a = *(const device uint4*)(vw + kk * vwt + 64 * g + 8 * fn);
#endif
        // 16-bit lanes (d-slot fn, d-slot fn + 1) = the low / high 8 bytes, word j = bytes 2j, 2j + 1
        const uint4 w = uint4(insert_bits(a.x, a.z, 16u, 16u), insert_bits(a.z, a.x >> 16, 0u, 16u),
                              insert_bits(a.y, a.w, 16u, 16u), insert_bits(a.w, a.y >> 16, 0u, 16u));
#if VATT_Q8_TIMING == 0
        const half2 sc = FA ? half2(1.0h) : half2(vs[kk * kv.vst + g]);
        const half2 bi = FA ? half2(0.0h) : half2(vb[kk * kv.vbt + g]);
#else
        const half2 sc = half2(0.01h), bi = half2(0.0h);
#endif
        simdgroup_matrix<float, 8, 8> P[MB];      // FACT: each key's column times its scale
        float2 fs = float2(1.0f);
        if (FA) {
          fs = float2(float(vs[ka * kv.vst + g]), float(vs[kc * kv.vst + g]));
          const float2 fb = fma(float2(128.0f), fs, float2(float(vb[ka * kv.vbt + g]), float(vb[kc * kv.vbt + g])));
          STEEL_PRAGMA_UNROLL
          for (int mb = 0; mb < MB; ++mb) {
            float t = pr[mb][kf].x * fb.x + pr[mb][kf].y * fb.y;
            t += simd_shuffle_xor(t, ushort(1));
            t += simd_shuffle_xor(t, ushort(8));
            ob[mb][gi] += t;
          }
        }
        STEEL_PRAGMA_UNROLL
        for (int mb = 0; mb < MB; ++mb) {
          P[mb].thread_elements()[0] = pr[mb][kf].x * fs.x;
          P[mb].thread_elements()[1] = pr[mb][kf].y * fs.y;
        }
        STEEL_PRAGMA_UNROLL
        for (int st = 0; st < 8; ++st) {
          const uint u = (w[st / 2] >> uint(8 * (st % 2))) & 0x00ff00ffu;
          const half2 cd = as_type<half2>(u | 0x64006400u) - OFF;
#if VATT_Q8_TIMING == 0
          const half2 e = FA ? cd : fma(cd, sc, bi);
#else
          const half2 e = cd * sc;
#endif
          simdgroup_matrix<half, 8, 8> B;
          B.thread_elements()[0] = e.x;
          B.thread_elements()[1] = e.y;
          STEEL_PRAGMA_UNROLL
          for (int mb = 0; mb < MB; ++mb)
            simdgroup_multiply_accumulate(O[mb][8 * gi + st], P[mb], B, O[mb][8 * gi + st]);
        }
      }
    }
  }

  const size_t p0 = (size_t(item) * HKV_ + h) * M0 + q0;
  STEEL_PRAGMA_UNROLL
  for (int mb = 0; mb < MB; ++mb) {
    const int r = 8 * mb + fm;
    if (q0 + r < M0) {
      device float* o = po + (p0 + r) * BD + 64 * g0 + 8 * fn;
      STEEL_PRAGMA_UNROLL
      for (int i = 0; i < NS; ++i) {              // step i holds dims 64 gi + 8 (fn, fn + 1) + st
        const int d = 64 * (i / 8) + (i % 8);
        o[d] = (O[mb][i].thread_elements()[0] + ob[mb][i / 8]) / sum_score[mb];
        o[d + 8] = (O[mb][i].thread_elements()[1] + ob[mb][i / 8]) / sum_score[mb];
      }
      if (fn == 0 && sg == 0) {
        pml[2 * (p0 + r)] = max_score[mb];
        pml[2 * (p0 + r) + 1] = sum_score[mb];
      }
    }
  }
}
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

# q8 "reg" / "regf": constants D, HKV, MB, WN, KF, FACT, SCALE2; the same inputs / outputs / meta as _P1Q
_P1R = r"""
    threadgroup float2 X[2 * WN * MB * KF * 32];
""" + _P1Q.split("    threadgroup half KV_smem[KVS];")[1].split("    vatt_p1<")[0] + r"""    vattq_p1<MB, WN, KF, FACT>(q, kv, po, pml, meta[0], meta[1], meta[2], meta[3], meta[5], HKV, SCALE2, X,
        thread_index_in_simdgroup, simdgroup_index_in_threadgroup, threadgroup_position_in_grid);
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

_HEADER: dict = {}
_KERNELS: dict = {}
_META: dict = {}
_LOCK = threading.Lock()


def _header(timing: int = 0) -> str:
    """The kernels' header; `timing` = the q8 loaders' VATT_Q8_TIMING (0 for every real kernel)."""
    if timing not in _HEADER:
        _HEADER[timing] = (_cm.steel_headers() + _cm._OPS
                           + f"#define VATT_Q8_KFAST {'false' if Q8_KMAP == 'c' else 'true'}\n"
                           + f"#define VATT_Q8_TIMING {timing}\n" + _IMPL)
    return _HEADER[timing]


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


def _kernels(D, hkv, scale, bq, bk, wm, wn, q8=False, timing=0):
    key = (D, hkv, float(scale), bq, bk, wm, wn, q8, timing)
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
                    name=f"vatt{'q8' if q8 else ''}_p1_{tag}{f'_t{timing}' if timing else ''}",
                    input_names=(["q", "kw", "ks", "kb", "vw", "vs", "vb", "meta"] if q8
                                 else ["q", "k", "v", "meta"]),
                    output_names=["po", "pml"], source=c + (_P1Q if q8 else _P1), header=_header(timing),
                    ensure_row_contiguous=False)
                p2 = _KERNELS.get(("p2", D, hkv))
                if p2 is None:
                    p2 = _KERNELS[("p2", D, hkv)] = mx.fast.metal_kernel(
                        name=f"vatt_p2_d{D}_h{hkv}", input_names=["po", "pml", "meta"],
                        output_names=["out"], source=c + _P2)
                k = _KERNELS[key] = (p1, p2)
    return k


def _kernels_reg(D, hkv, scale, mb, wn, kf, fact=False, timing=0):
    key = ("reg", D, hkv, float(scale), mb, wn, kf, fact, timing)
    k = _KERNELS.get(key)
    if k is None:
        with _LOCK:
            k = _KERNELS.get(key)
            if k is None:
                if D != 256 or wn not in (1, 2, 4) or not 1 <= mb <= 6 or kf not in (1, 2):
                    raise ValueError(f"vatt: q8 reg tile {mb}x{wn}x{kf} at head_dim {D}")
                c = (f"constexpr int D = {D}, HKV = {hkv}, MB = {mb}, WN = {wn}, KF = {kf};\n"
                     f"    constexpr bool FACT = {'true' if fact else 'false'};\n"
                     f"    constexpr float SCALE2 = {scale * math.log2(math.e)!r}f;\n")
                tag = (f"m{mb}w{wn}k{kf}{'f' if fact else ''}{f't{timing}' if timing else ''}_d{D}_h{hkv}_s"
                       + float(scale).hex().replace('.', '_').replace('-', 'm').replace('+', 'p'))
                p1 = mx.fast.metal_kernel(
                    name=f"vattq8r_p1_{tag}", input_names=["q", "kw", "ks", "kb", "vw", "vs", "vb", "meta"],
                    output_names=["po", "pml"], source=c + _P1R, header=_header(timing),
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


def _run(q, kv_inputs, Hkv, n, scale, chunk, q8, kernel="tile", timing=0):
    _, Hq, T, D = q.shape
    m0 = Hq // Hkv * T
    reg = q8 and kernel != "tile"
    if reg:                          # 8 * MB query rows x 8 * KF keys a block, WN simdgroups
        mb, wn, kf = Q8R_DECODE if m0 <= 8 else Q8R_VERIFY
        bk, blocks, tg = 8 * kf, -(-m0 // (8 * mb)), 32 * wn
    else:
        bq, bk, wm, wn, blocks = tile_for(m0)
        tg = 32 * wm * wn
    chunk = chunk or pick_chunk(n, Hkv, blocks, bk)
    if chunk % bk:
        raise ValueError(f"vatt: chunk {chunk} not a multiple of {bk}")
    items = -(-n // chunk)
    meta = _meta(m0, n, T, chunk, blocks)
    p1, p2 = (_kernels_reg(D, Hkv, scale, mb, wn, kf, kernel == "regf", timing) if reg
              else _kernels(D, Hkv, scale, bq, bk, wm, wn, q8, timing))
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


def reference_q8(q, kq, vq, n, scale, group_size=Q8_GROUP, bits=8, exact=False):
    """fp32 reference over the dequantized KV -> (1, Hq, T, D) fp32. `exact`: over the
    affine values scale * code + bias in fp32 instead of mx.dequantize's fp16 ones."""
    if exact:
        kq, vq = ((x[0], x[1].astype(F32), x[2].astype(F32)) for x in (kq, vq))
    return reference(q, dequantized(kq, n, group_size, bits), dequantized(vq, n, group_size, bits), n, scale)


def attention_q8(q, kq, vq, n, scale, group_size=Q8_GROUP, bits=8, chunk=None, kernel=None, timing=None):
    """`attention` over an 8-bit QuantizedKVCache: kq / vq the cache's
    (packed, scales, biases) buffers (1, Hkv, cap, ...), keys [0, n) valid,
    read in place. `kernel` (None = Q8_KERNEL): "tile" dequantizes in registers
    into the fp16 kernel's threadgroup tiles, "reg" straight into the MMA
    fragments (vattq_p1), "regf" runs the MMAs on the codes with scale and
    bias taken out of the sums (the affine values unrounded: not
    mx.dequantize's fp16 ones). `timing` (None = Q8_TIMING): the microbench's
    cost-split variants, wrong numbers."""
    check_q8(q, kq, vq, n, group_size, bits)
    kernel = kernel or Q8_KERNEL
    if kernel not in Q8_KERNELS:
        raise ValueError(f"vatt: q8 kernel {kernel!r} ({' / '.join(Q8_KERNELS)})")
    if not use_kernel():
        return reference_q8(q, kq, vq, n, scale, group_size, bits, exact=kernel == "regf").astype(mx.float16)
    return _run(q, [*kq, *vq], kq[0].shape[1], n, scale, chunk, True, kernel,
                Q8_TIMING if timing is None else timing)


def partial_bytes(n: int, hq: int, T: int, D: int = 256) -> int:
    """Upper bound of one call's pass-1 partials (every chunk >= MIN_CHUNK keys)."""
    return (-(-n // MIN_CHUNK) + 1) * hq * T * (D + 2) * 4
