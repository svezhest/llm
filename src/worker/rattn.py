"""Ragged multi-row decode attention for batched decode (fp16 KV, L = 1).

Today a batched decode step runs, per full-attention layer and per row,
`update_and_fetch` + mx.fast.scaled_dot_product_attention on (1, Hq, 1, D)
vs the row's own (1, Hkv, L_i, D) KV: 2 kernels per row (MLX's 2-pass
sdpa_vector; on M1 Max with GQA 4 it runs 64 key blocks whatever L is, each
of its 4 simdgroups -- one per query head -- loading the same keys).

This kernel attends ALL rows of the step in one launch (+ a tiny merge):
each row keeps its own K/V buffers (the KVCache arrays, passed as separate
kernel inputs kb0..kbn, vb0..vbn, selected by row index in MSL; no copy into a
stacked buffer) and its own length. Work is split into chunks of CHUNK keys
per (row, kv head) -- a host-built item table, so a 33K row and a 5K row
cost what their keys cost -- and each simdgroup of a chunk's threadgroup
computes all G query heads of its kv head over a contiguous run of keys
(K/V read once, as MLX's sdpa_vector_2pass_1_gqa does for G 8/12/16):

  pass 1 (items x Hkv threadgroups of NS simdgroups): per item and query head
         the softmax-normalized partial output o (fp32), its max m and sum l
         (log2 domain: q is pre-scaled by scale * log2 e);
  pass 2 (rows x Hq threadgroups of D threads): o = sum_c w_c o_c / sum_c w_c,
         w_c = l_c 2^(m_c - M) -> fp16.

fp32 accumulation throughout (MLX keeps its partials in fp16). Metal allows
31 buffers per kernel, so one launch takes at most MAX_ROWS rows (2 x rows +
5 buffers); a bigger batch is split into launches of <= MAX_ROWS rows.

    out = rattn.attention(q, ks, vs, lens, scale)   # q (B, Hq, 1, D) fp16

`ks[i]` / `vs[i]`: (1, Hkv, cap_i, D) fp16 row-contiguous (the KVCache
buffers after update_and_fetch, not the [..., :L, :] views), `lens[i]` the
valid length (keys [0, lens[i]) are attended; nothing is masked inside).
"""
from __future__ import annotations

import math
import threading

import mlx.core as mx

D = 256
CHUNK = 512          # keys per work item
NS = 4               # simdgroups per threadgroup (split an item's keys)
MAX_ROWS = 8         # rows per launch (buffer limit: 2*rows + 5 <= 31)

_HEADER = """
#include <metal_simdgroup>
using namespace metal;
"""

# constants prepended: G (q heads per kv head), HQ, D, NS, SCALE2 (scale * log2 e)
_P1 = r"""
    constexpr int PT = D / 32;                       // elements per lane
    const uint lid = thread_index_in_simdgroup;
    const int s = int(simdgroup_index_in_threadgroup);
    const int item = int(threadgroup_position_in_grid.x);
    const int kvh = int(threadgroup_position_in_grid.y);
    const int row = items[4 * item], start = items[4 * item + 1], end = items[4 * item + 2];
    const int cap = rows[4 * row + 1];
    const device half* kp;
    const device half* vp;
    switch (row) {
%SWITCH%
      default: kp = kb0; vp = vb0;
    }
    const int sub = (end - start + NS - 1) / NS;
    const int s0 = min(end, start + s * sub), s1 = min(end, s0 + sub);
    const size_t base = (size_t(kvh) * cap + s0) * D + lid * PT;
    const device half4* k4 = (const device half4*)(kp + base);
    const device half4* v4 = (const device half4*)(vp + base);

    float qr[G][PT], o[G][PT], m[G], l[G];
    for (int g = 0; g < G; ++g) {
      const device half* qp = q + (size_t(row) * HQ + kvh * G + g) * D + lid * PT;
      for (int i = 0; i < PT; ++i) { qr[g][i] = SCALE2 * float(qp[i]); o[g][i] = 0.0f; }
      m[g] = -INFINITY; l[g] = 0.0f;
    }
    for (int t = s0; t < s1; ++t) {
      const float4 ka = float4(k4[0]), kb = float4(k4[1]);
      float sc[G];
      for (int g = 0; g < G; ++g) {
        float d = qr[g][0] * ka.x + qr[g][1] * ka.y + qr[g][2] * ka.z + qr[g][3] * ka.w
                + qr[g][4] * kb.x + qr[g][5] * kb.y + qr[g][6] * kb.z + qr[g][7] * kb.w;
        sc[g] = simd_sum(d);
      }
      const float4 va = float4(v4[0]), vb = float4(v4[1]);
      for (int g = 0; g < G; ++g) {
        const float mn = max(m[g], sc[g]);
        const float f = fast::exp2(m[g] - mn), e = fast::exp2(sc[g] - mn);
        m[g] = mn;
        l[g] = l[g] * f + e;
        o[g][0] = o[g][0] * f + e * va.x; o[g][1] = o[g][1] * f + e * va.y;
        o[g][2] = o[g][2] * f + e * va.z; o[g][3] = o[g][3] * f + e * va.w;
        o[g][4] = o[g][4] * f + e * vb.x; o[g][5] = o[g][5] * f + e * vb.y;
        o[g][6] = o[g][6] * f + e * vb.z; o[g][7] = o[g][7] * f + e * vb.w;
      }
      k4 += D / 4;
      v4 += D / 4;
    }

    // merge the NS simdgroups of this item (threadgroup memory), one query head per simdgroup
    threadgroup float o_sh[NS * G * D];
    threadgroup float m_sh[NS * G], l_sh[NS * G];
    for (int g = 0; g < G; ++g) {
      for (int i = 0; i < PT; ++i) o_sh[(s * G + g) * D + lid * PT + i] = o[g][i];
      if (lid == 0) { m_sh[s * G + g] = m[g]; l_sh[s * G + g] = l[g]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = s; g < G; g += NS) {
      float M = -INFINITY;
      for (int j = 0; j < NS; ++j) M = max(M, m_sh[j * G + g]);
      float L = 0.0f, acc[PT];
      for (int i = 0; i < PT; ++i) acc[i] = 0.0f;
      for (int j = 0; j < NS; ++j) {
        const float mj = m_sh[j * G + g];
        const float w = mj == -INFINITY ? 0.0f : fast::exp2(mj - M);    // an empty run: l = 0, o = 0
        L += l_sh[j * G + g] * w;
        for (int i = 0; i < PT; ++i) acc[i] += w * o_sh[(j * G + g) * D + lid * PT + i];
      }
      const size_t p = size_t(item) * HQ + kvh * G + g;
      const float inv = 1.0f / L;
      for (int i = 0; i < PT; ++i) po[p * D + lid * PT + i] = acc[i] * inv;
      if (lid == 0) { pml[2 * p] = M; pml[2 * p + 1] = L; }
    }
"""

_P2 = r"""
    const int row = int(threadgroup_position_in_grid.x);
    const int h = int(threadgroup_position_in_grid.y);
    const int d = int(thread_position_in_threadgroup.x);
    const int first = rows[4 * row + 2], n = rows[4 * row + 3];
    float M = -INFINITY;
    for (int c = 0; c < n; ++c) M = max(M, pml[2 * ((size_t(first) + c) * HQ + h)]);
    float L = 0.0f, acc = 0.0f;
    for (int c = 0; c < n; ++c) {
      const size_t p = (size_t(first) + c) * HQ + h;
      const float w = pml[2 * p + 1] * fast::exp2(pml[2 * p] - M);
      L += w;
      acc += w * po[p * D + d];
    }
    out[(size_t(row) * HQ + h) * D + d] = half(acc / L);
"""

_KERNELS = {}
_LOCK = threading.Lock()
_META = {}            # (lens, caps, chunk) -> (items, rows, n_items): reused by every layer of a step


def _consts(G, HQ, scale, ns):
    return (f"constexpr int G = {G}, HQ = {HQ}, D = {D}, NS = {ns};\n"
            f"    constexpr float SCALE2 = {scale * math.log2(math.e)!r}f;\n")


def _kernels(n, G, HQ, scale, ns):
    key = (n, G, HQ, scale, ns)
    k = _KERNELS.get(key)
    if k is None:
        with _LOCK:
            sw = "\n".join(f"      case {i}: kp = kb{i}; vp = vb{i}; break;" for i in range(n))
            src = _consts(G, HQ, scale, ns) + _P1.replace("%SWITCH%", sw)
            p1 = mx.fast.metal_kernel(
                name=f"rattn_p1_r{n}_g{G}_h{HQ}_s{ns}_{abs(hash(scale)) % 10**8}",
                input_names=["q", "items", "rows"] + [f"kb{i}" for i in range(n)] + [f"vb{i}" for i in range(n)],
                output_names=["po", "pml"], source=src, header=_HEADER)
            p2 = mx.fast.metal_kernel(
                name=f"rattn_p2_g{G}_h{HQ}", input_names=["po", "pml", "rows"], output_names=["out"],
                source=_consts(G, HQ, scale, ns) + _P2, header=_HEADER)
            k = _KERNELS[key] = (p1, p2)
    return k


def _meta(lens, caps, chunk):
    key = (lens, caps, chunk)
    hit = _META.get(key)
    if hit is not None:
        return hit
    items, rows = [], []
    for r, (n, cap) in enumerate(zip(lens, caps)):
        first = len(items) // 4
        for s in range(0, n, chunk):
            items += [r, s, min(n, s + chunk), 0]
        rows += [n, cap, first, len(items) // 4 - first]
    val = (mx.array(items, dtype=mx.int32), mx.array(rows, dtype=mx.int32), len(items) // 4)
    if len(_META) >= 8:                 # a step's launches (<= 2 at B 16) + the previous step's
        _META.pop(next(iter(_META)))
    _META[key] = val
    return val


def _launch(q, ks, vs, lens, scale, chunk, ns):
    B, HQ, _, d = q.shape
    HKV = ks[0].shape[1]
    G = HQ // HKV
    caps = tuple(k.shape[2] for k in ks)
    items, rows, n_items = _meta(tuple(lens), caps, chunk)
    p1, p2 = _kernels(B, G, HQ, scale, ns)
    po, pml = p1(inputs=[q, items, rows, *ks, *vs],
                 grid=(32 * ns * n_items, HKV, 1), threadgroup=(32 * ns, 1, 1),
                 output_shapes=[(n_items * HQ * D,), (n_items * HQ * 2,)],
                 output_dtypes=[mx.float32, mx.float32])
    return p2(inputs=[po, pml, rows], grid=(D * B, HQ, 1), threadgroup=(D, 1, 1),
              output_shapes=[(B, HQ, 1, D)], output_dtypes=[mx.float16])[0]


def attention(q, ks, vs, lens, scale, chunk=CHUNK, ns=NS, max_rows=MAX_ROWS):
    """q (B, Hq, 1, D) fp16; ks/vs: B arrays (1, Hkv, cap_i, D) fp16; lens: B
    ints (1 <= lens[i] <= cap_i) -> (B, Hq, 1, D) fp16."""
    B = q.shape[0]
    if q.dtype != mx.float16 or q.shape[2] != 1 or q.shape[3] != D or len(ks) != B or len(vs) != B:
        raise ValueError(f"rattn: q {q.shape} {q.dtype}, {len(ks)} rows")
    ns = max(1, min(ns, 16 // max(1, q.shape[1] // ks[0].shape[1])))  # NS * G * D floats of threadgroup memory <= 16 KB
    if B <= max_rows:
        return _launch(q, ks, vs, lens, scale, chunk, ns)
    parts = [_launch(q[i:i + max_rows], ks[i:i + max_rows], vs[i:i + max_rows], lens[i:i + max_rows],
                     scale, chunk, ns) for i in range(0, B, max_rows)]
    return mx.concatenate(parts, axis=0)
