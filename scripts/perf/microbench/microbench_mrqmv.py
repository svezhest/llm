"""PROTOTYPE: 8-bit affine multi-row qmv (one weight read shared by M <= 16 rows).

Question: can a custom kernel beat mx.quantized_matmul (qmv per row below the M1
batch limit, qmm_splitk above it: ~x2.7 at M>=6) for batched decode on M1?

Kernel (mx.fast.metal_kernel): like MLX's qmv, each simdgroup owns R output
rows, each thread 8 consecutive K values per 256-wide block. Per block the
threadgroup first stages x[M, 256] in threadgroup memory (one device read per
threadgroup instead of one per simdgroup), then every thread dequantizes its
R x 8 weights ONCE into registers and runs them against all M rows:
  acc[t][r] += scale_r * dot(wq_r, x_t) + bias_r * sum(x_t)     (fp32)
and simd_sum's at the end. Variant "dev" skips the threadgroup staging and
reads x straight from device memory (cache) per row.

Timed like design/microbench_qmm.py: chains of distinct weight copies (> 256 MB,
defeats the 48 MB SLC), per-matmul median. Correctness: max |diff| against
mx.quantized_matmul and against an fp32 dequantize -> matmul reference.

  <venv>/bin/python -I microbench_mrqmv.py --label new [--rows 1,2,4,8,12,16]
NEEDS THE GPU LEASE.
"""
import argparse
import json
import os
import statistics
import time

import mlx.core as mx

REPO = "/Users/user/Projects/llm"

SRC = r"""
    constexpr int SIMD = 32;
    constexpr int VPT = 8;                  // values per thread per block
    constexpr int BLOCK = VPT * SIMD;       // 256 K values per block
    constexpr int NT = NSG * SIMD;
    uint tg = threadgroup_position_in_grid.y;
    uint sg = simdgroup_index_in_threadgroup;
    uint lid = thread_index_in_simdgroup;
    uint tid = sg * SIMD + lid;
    const int out_row = (int(tg) * NSG + int(sg)) * R;
    const int Kg = K / GS;

    float acc[M][R];
    for (int t = 0; t < M; ++t)
      for (int r = 0; r < R; ++r) acc[t][r] = 0.0f;

    const device uint8_t* wb = (const device uint8_t*)w + out_row * K + lid * VPT;
    const device T* sb = scales + out_row * Kg;
    const device T* bb = biases + out_row * Kg;
#if XTG
    threadgroup T xs[M * BLOCK];
#endif

    for (int k0 = 0; k0 < K; k0 += BLOCK) {
#if XTG
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int i = int(tid) * 4; i < M * BLOCK; i += NT * 4) {
        int t = i / BLOCK, j = i % BLOCK;
        *((threadgroup vec<T, 4>*)(xs + i)) = *((const device vec<T, 4>*)(x + t * K + k0 + j));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
      float wv[R][VPT];
      float s[R], b[R];
      const int g = (k0 + int(lid) * VPT) / GS;
      for (int r = 0; r < R; ++r) {
        uchar4 lo = *((const device uchar4*)(wb + r * K + k0));
        uchar4 hi = *((const device uchar4*)(wb + r * K + k0 + 4));
        wv[r][0] = lo.x; wv[r][1] = lo.y; wv[r][2] = lo.z; wv[r][3] = lo.w;
        wv[r][4] = hi.x; wv[r][5] = hi.y; wv[r][6] = hi.z; wv[r][7] = hi.w;
        s[r] = float(sb[r * Kg + g]);
        b[r] = float(bb[r * Kg + g]);
      }
      for (int t = 0; t < M; ++t) {
#if XTG
        const threadgroup T* xp = xs + t * BLOCK + lid * VPT;
        vec<T, 4> xa = *((const threadgroup vec<T, 4>*)xp);
        vec<T, 4> xb = *((const threadgroup vec<T, 4>*)(xp + 4));
#else
        const device T* xp = x + t * K + k0 + lid * VPT;
        vec<T, 4> xa = *((const device vec<T, 4>*)xp);
        vec<T, 4> xb = *((const device vec<T, 4>*)(xp + 4));
#endif
        float4 fa = float4(xa), fb = float4(xb);
        float xsum = fa.x + fa.y + fa.z + fa.w + fb.x + fb.y + fb.z + fb.w;
        for (int r = 0; r < R; ++r) {
          float d = wv[r][0] * fa.x + wv[r][1] * fa.y + wv[r][2] * fa.z + wv[r][3] * fa.w
                  + wv[r][4] * fb.x + wv[r][5] * fb.y + wv[r][6] * fb.z + wv[r][7] * fb.w;
          acc[t][r] += s[r] * d + b[r] * xsum;
        }
      }
    }
    for (int t = 0; t < M; ++t) {
      for (int r = 0; r < R; ++r) {
        float v = simd_sum(acc[t][r]);
        if (lid == 0) y[t * N + out_row + r] = T(v);
      }
    }
"""

_KERNELS = {}


def mrqmv(x, wq, s, b, R=4, NSG=2, xtg=True, gs=64):
    """y[M, N] = x[M, K] @ dequant(wq)^T for 8-bit affine, M <= 16."""
    M, K = x.shape
    N = wq.shape[0]
    assert K % 256 == 0 and N % (R * NSG) == 0 and wq.dtype == mx.uint32
    key = (R, NSG, xtg)
    if key not in _KERNELS:
        _KERNELS[key] = mx.fast.metal_kernel(
            name=f"mrqmv8_r{R}_s{NSG}_{'tg' if xtg else 'dev'}",
            input_names=["x", "w", "scales", "biases"],
            output_names=["y"],
            source=SRC.replace("XTG", "1" if xtg else "0"),
        )
    return _KERNELS[key](
        inputs=[x, wq, s, b],
        template=[("T", x.dtype), ("M", M), ("K", K), ("N", N), ("R", R), ("NSG", NSG), ("GS", gs)],
        grid=(32, NSG * (N // (R * NSG)), 1),
        threadgroup=(32, NSG, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )[0]


MMA_HEADER = """
#include <metal_simdgroup_matrix>
using namespace metal;
"""

# simdgroup-MMA variant: each simdgroup owns BN=32 output features; per 64-wide
# K block it dequantizes its 32x64 weight tile ONCE into threadgroup memory
# (fp16, scale*q+bias like MLX's qmm loader) and multiplies it by the
# M (padded to 8/16) x 64 activation tile with 8x8 simdgroup MMAs (fp32 acc).
# MLX's qmm does the same with a 32-row tile, i.e. 2-4x the MMA work at M=8-16.
MMA_SRC = r"""
    constexpr int BN = 32;
    constexpr int BK = 64;
    constexpr int P = 8;          // row padding (halfs) for the weight tile
    constexpr int MT = M / 8;
    constexpr int NT = BN / 8;
    uint sg = simdgroup_index_in_threadgroup;
    uint lid = thread_index_in_simdgroup;
    uint tid = sg * 32 + lid;
    const int n0 = (int(threadgroup_position_in_grid.y) * NSG + int(sg)) * BN;
    const int Kg = K / GS;
    threadgroup T Xs[M * BK];
    threadgroup T Wsall[NSG * BN * (BK + P)];
    threadgroup T* Ws = Wsall + sg * BN * (BK + P);
    simdgroup_matrix<float, 8, 8> C[MT][NT];
    for (int i = 0; i < MT; ++i)
      for (int j = 0; j < NT; ++j) C[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const device uint8_t* wr = (const device uint8_t*)w + (n0 + int(lid)) * K;
    for (int k0 = 0; k0 < K; k0 += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int i = int(tid) * 4; i < M * BK; i += NSG * 32 * 4) {
        int t = i / BK, j = i % BK;
        *((threadgroup vec<T, 4>*)(Xs + i)) = *((const device vec<T, 4>*)(x + t * K + k0 + j));
      }
      float sc = float(scales[(n0 + int(lid)) * Kg + k0 / GS]);
      float bi = float(biases[(n0 + int(lid)) * Kg + k0 / GS]);
      threadgroup T* wd = Ws + int(lid) * (BK + P);
      for (int j = 0; j < BK; j += 4) {
        uchar4 q = *((const device uchar4*)(wr + k0 + j));
        *((threadgroup vec<T, 4>*)(wd + j)) = vec<T, 4>(float4(q) * sc + bi);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int kk = 0; kk < BK; kk += 8) {
        simdgroup_matrix<T, 8, 8> A[MT];
        simdgroup_matrix<T, 8, 8> B[NT];
        for (int i = 0; i < MT; ++i) simdgroup_load(A[i], Xs + i * 8 * BK + kk, BK);
        for (int j = 0; j < NT; ++j)
          simdgroup_load(B[j], Ws + j * 8 * (BK + P) + kk, BK + P, ulong2(0, 0), true);
        for (int i = 0; i < MT; ++i)
          for (int j = 0; j < NT; ++j) simdgroup_multiply_accumulate(C[i][j], A[i], B[j], C[i][j]);
      }
    }
    const short qid = short(lid) / 4;
    const short fm = (qid & 4) + ((short(lid) / 2) % 4);
    const short fn = (qid & 2) * 2 + (short(lid) % 2) * 2;
    for (int i = 0; i < MT; ++i)
      for (int j = 0; j < NT; ++j) {
        auto e = C[i][j].thread_elements();
        int row = i * 8 + fm;
        int col = n0 + j * 8 + fn;
        y[row * N + col] = T(e[0]);
        y[row * N + col + 1] = T(e[1]);
      }
"""


def mrqmm(x, wq, s, b, NSG=4, gs=64):
    """simdgroup-MMA multi-row 8-bit matmul; pads M to 8 or 16."""
    M0, K = x.shape
    N = wq.shape[0]
    M = 8 if M0 <= 8 else 16
    assert M0 <= 16 and K % 64 == 0 and N % (32 * NSG) == 0
    if M != M0:
        x = mx.concatenate([x, mx.zeros((M - M0, K), x.dtype)])
    key = ("mma", NSG)
    if key not in _KERNELS:
        _KERNELS[key] = mx.fast.metal_kernel(
            name=f"mrqmm8_s{NSG}", input_names=["x", "w", "scales", "biases"], output_names=["y"],
            source=MMA_SRC, header=MMA_HEADER)
    y = _KERNELS[key](
        inputs=[x, wq, s, b],
        template=[("T", x.dtype), ("M", M), ("K", K), ("N", N), ("NSG", NSG), ("GS", gs)],
        grid=(32, NSG * (N // (32 * NSG)), 1), threadgroup=(32, NSG, 1),
        output_shapes=[(M, N)], output_dtypes=[x.dtype])[0]
    return y if M == M0 else y[:M0]


# Register-direct MMA variant ("rmma"): no threadgroup memory for weights.
# C^T[n, m] = W[n, :] . x[m, :] with A = dequantized W frags built straight in
# registers (thread_elements) and B = x^T frags from a threadgroup x tile.
# The k order inside each 64-wide group is permuted so that every thread reads
# 16 CONTIGUOUS weight bytes per 8-row frag (one uint4 load; 8 rows x 64 bytes
# per simdgroup load = 8 full cache lines); x is staged with the same
# permutation, so the dot products are unchanged.
RMMA_SRC = r"""
    constexpr int BK = 64;
    constexpr int XB = 256;        // x columns staged per barrier
    constexpr int MB = M / 8;      // x frags (8 rows of x each)
    uint sg = simdgroup_index_in_threadgroup;
    uint lid = thread_index_in_simdgroup;
    uint tid = sg * 32 + lid;
    const int n0 = (int(threadgroup_position_in_grid.y) * NSG + int(sg)) * (8 * NA);
    const int Kg = K / GS;
    const short qid = short(lid) / 4;
    const short fm = (qid & 4) + ((short(lid) / 2) % 4);
    const short fn = (qid & 2) * 2 + (short(lid) % 2) * 2;
    const short c = fn / 2;                       // which 16-byte quarter of the 64-wide group
    threadgroup T Xs[M * XB];
    simdgroup_matrix<float, 8, 8> C[NA][MB];
    for (int i = 0; i < NA; ++i)
      for (int j = 0; j < MB; ++j) C[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const device uint8_t* wrow[NA];
    for (int i = 0; i < NA; ++i) wrow[i] = (const device uint8_t*)w + (n0 + 8 * i + fm) * K + 16 * c;
    for (int kx = 0; kx < K; kx += XB) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // stage x[:, kx:kx+XB] permuted: slot (s*8 + j) of group g <- k = g*64 + 16*(j/2) + 2*s + (j%2)
      // work item = (m, 16-wide quarter): read 16 contiguous x, scatter 8 pairs
      for (int i = int(tid); i < M * (XB / 16); i += NSG * 32) {
        int m = i / (XB / 16), qq = i % (XB / 16);
        int g = qq / 4, cc = qq % 4;
        const device vec<T, 2>* src = (const device vec<T, 2>*)(x + m * K + kx + 16 * qq);
        threadgroup T* dst = Xs + m * XB + g * BK + 2 * cc;
        for (int st = 0; st < 8; ++st) *((threadgroup vec<T, 2>*)(dst + st * 8)) = src[st];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int g = 0; g < XB / BK; ++g) {
        const int k0 = kx + g * BK;
        uint4 wq[NA];
        float sc[NA], bi[NA];
        for (int i = 0; i < NA; ++i) {
          wq[i] = *((const device uint4*)(wrow[i] + k0));
          sc[i] = float(scales[(n0 + 8 * i + fm) * Kg + k0 / GS]);
          bi[i] = float(biases[(n0 + 8 * i + fm) * Kg + k0 / GS]);
        }
        for (int st = 0; st < 8; ++st) {
          simdgroup_matrix<T, 8, 8> B[MB];
          for (int j = 0; j < MB; ++j)
            simdgroup_load(B[j], Xs + j * 8 * XB + g * BK + st * 8, XB, ulong2(0, 0), true);
          for (int i = 0; i < NA; ++i) {
            uint word = wq[i][st / 2];
            uint sh = (st % 2) * 16;
            simdgroup_matrix<T, 8, 8> A;
            A.thread_elements()[0] = T(float((word >> sh) & 0xff) * sc[i] + bi[i]);
            A.thread_elements()[1] = T(float((word >> (sh + 8)) & 0xff) * sc[i] + bi[i]);
            for (int j = 0; j < MB; ++j) simdgroup_multiply_accumulate(C[i][j], A, B[j], C[i][j]);
          }
        }
      }
    }
    for (int i = 0; i < NA; ++i)
      for (int j = 0; j < MB; ++j) {
        auto e = C[i][j].thread_elements();
        int n = n0 + 8 * i + fm;
        int m = j * 8 + fn;
        y[m * N + n] = T(e[0]);
        y[(m + 1) * N + n] = T(e[1]);
      }
"""


def rmma(x, wq, s, b, NSG=2, NA=4, gs=64):
    M0, K = x.shape
    N = wq.shape[0]
    M = 8 if M0 <= 8 else 16
    assert M0 <= 16 and K % 256 == 0 and N % (8 * NA * NSG) == 0
    if M != M0:
        x = mx.concatenate([x, mx.zeros((M - M0, K), x.dtype)])
    key = ("rmma", NSG, NA)
    if key not in _KERNELS:
        _KERNELS[key] = mx.fast.metal_kernel(
            name=f"rmma8_s{NSG}_a{NA}", input_names=["x", "w", "scales", "biases"], output_names=["y"],
            source=RMMA_SRC, header=MMA_HEADER)
    y = _KERNELS[key](
        inputs=[x, wq, s, b],
        template=[("T", x.dtype), ("M", M), ("K", K), ("N", N), ("NSG", NSG), ("NA", NA), ("GS", gs)],
        grid=(32, NSG * (N // (8 * NA * NSG)), 1), threadgroup=(32, NSG, 1),
        output_shapes=[(M, N)], output_dtypes=[x.dtype])[0]
    return y if M == M0 else y[:M0]


def shapes_9b(tc):
    h = tc["hidden_size"]; nq, nkv = tc["num_attention_heads"], tc["num_key_value_heads"]
    d = tc["head_dim"]; L = tc["num_hidden_layers"]; fa = L // tc.get("full_attention_interval", 4)
    gdn = L - fa
    kd = tc["linear_num_key_heads"] * tc["linear_key_head_dim"]
    vd = tc["linear_num_value_heads"] * tc["linear_value_head_dim"]
    I = tc["intermediate_size"]
    return [
        ("q_proj(+gate)", h, 2 * nq * d, fa), ("k_proj", h, nkv * d, fa), ("v_proj", h, nkv * d, fa),
        ("o_proj", nq * d, h, fa), ("gdn in_proj_qkv", h, 2 * kd + vd, gdn),
        ("gdn in_proj_z", h, vd, gdn), ("gdn out_proj", vd, h, gdn),
        ("mlp gate_proj", h, I, L), ("mlp up_proj", h, I, L), ("mlp down_proj", I, h, L),
        ("lm_head", h, tc["vocab_size"], 1),
    ]


def bench_chain(fns, x, warmup=3, iters=15):
    def chain():
        y = x
        outs = []
        for fn in fns:
            o = fn(y)
            y = x + (o[..., :1].sum() * 0).astype(x.dtype)
            outs.append(o[..., :1])
        return outs
    for _ in range(warmup):
        mx.eval(chain())
    mx.synchronize()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        mx.eval(chain())
        mx.synchronize()
        ts.append((time.perf_counter() - t) / len(fns))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--config", default=f"{REPO}/weights/ornith15-9b-mlx-8bit")
    ap.add_argument("--rows", default="1,2,4,8,12,16")
    ap.add_argument("--variants", default="tg_r4_s2,tg_r2_s4,tg_r4_s4,dev_r4_s2")
    ap.add_argument("--only", help="substring filter on shape names")
    ap.add_argument("--extra-shape", default="mlp gate+up fused:4096:24576:0",
                    help="name:K:N:count (count 0 = reported, not in the step sum)")
    ap.add_argument("--dequant", action="store_true", help="also time dequantize->fp16 matmul")
    ap.add_argument("--verify", action="store_true", help="also mlx-vlm 0.7 verify kernel (M<=8)")
    ap.add_argument("--chain", type=int, default=16)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--json")
    a = ap.parse_args()
    tc = json.load(open(os.path.join(a.config, "config.json")))
    tc = tc.get("text_config", tc)
    shapes = shapes_9b(tc)
    if a.extra_shape:
        n, K, N, c = a.extra_shape.split(":")
        shapes.append((n, int(K), int(N), int(c)))
    if a.only:
        shapes = [s for s in shapes if any(k in s[0] for k in a.only.split(","))]
    rows = [int(r) for r in a.rows.split(",")]
    variants = [v for v in a.variants.split(",") if v]
    vq = None
    if a.verify:
        try:
            from mlx_vlm.models.quantized_verifier import optimized_affine_linear as vq
            import mlx.nn as nn
        except ImportError:
            vq = None
    dt = mx.float16
    print(f"# multi-row 8-bit qmv prototype — {a.label}\n\nmlx {mx.__version__}; per-matmul median "
          f"of {a.iters} x chain {a.chain} distinct weights; µs (× vs quantized_matmul M=1)\n")
    print("| shape | K x N | kernel | maxdiff vs qmm / vs fp32 (M=16) | " + " | ".join(f"M={m}" for m in rows) + " |")
    print("|---|---|---|---|" + "---:|" * len(rows))
    raw, per = [], {}
    mx.random.seed(0)
    for name, K, N, count in shapes:
        wbytes = N * K + 4 * N * (K // 64)
        copies = max(2, min(a.chain, int(256e6 // wbytes) + 1))
        ws = []
        for _ in range(copies):
            wf = mx.random.normal((N, K)).astype(dt) * K ** -0.5
            wq, s, b = mx.quantize(wf, group_size=64, bits=8)
            ws.append((wq, s.astype(dt), b.astype(dt)))
            mx.eval(ws[-1])
        sel = (ws * (a.chain // copies + 1))[: a.chain]
        kern = {"quantized_matmul": [(lambda y, w=w: mx.quantized_matmul(
            y, w[0], w[1], w[2], transpose=True, group_size=64, bits=8)) for w in sel]}
        for v in variants:
            if v.startswith("rmma"):
                _, a_, s2 = v.split("_")
                NA, NSG = int(a_[1:]), int(s2[1:])
                if N % (8 * NA * NSG) == 0:
                    kern[v] = [(lambda y, w=w, NSG=NSG, NA=NA: rmma(y, *w, NSG=NSG, NA=NA)) for w in sel]
                continue
            if v.startswith("mma"):
                NSG = int(v.split("_s")[1])
                if N % (32 * NSG) == 0:
                    kern[v] = [(lambda y, w=w, NSG=NSG: mrqmm(y, *w, NSG=NSG)) for w in sel]
                continue
            xtg, r, s_ = v.split("_")
            R, NSG = int(r[1:]), int(s_[1:])
            if N % (R * NSG):
                continue
            kern[v] = [(lambda y, w=w, R=R, NSG=NSG, xtg=xtg: mrqmv(y, *w, R=R, NSG=NSG, xtg=(xtg == "tg")))
                       for w in sel]
        if a.dequant:
            kern["dequant+fp16 matmul"] = [(lambda y, w=w: y @ mx.dequantize(
                w[0], w[1], w[2], group_size=64, bits=8).T) for w in sel]
        if vq is not None:
            mods = []
            for wq, s, b in ws:
                m_ = nn.QuantizedLinear(64, 8, bias=False, group_size=64, bits=8)
                m_.weight, m_.scales, m_.biases = wq, s, b
                mods.append(m_)
            msel = (mods * (a.chain // copies + 1))[: a.chain]
            kern["vlm verify"] = [(lambda y, m_=m_: vq(m_, y[None])[0]) for m_ in msel]
        # correctness at M=16 (and M=1)
        x16 = (mx.random.normal((16, K)) * 0.5).astype(dt)
        w0 = ws[0]
        ref_q = mx.quantized_matmul(x16, *w0, transpose=True, group_size=64, bits=8).astype(mx.float32)
        ref_f = x16.astype(mx.float32) @ mx.dequantize(w0[0], w0[1].astype(mx.float32),
                                                       w0[2].astype(mx.float32), group_size=64, bits=8).T
        for kname, fl in kern.items():
            md = ""
            if kname not in ("quantized_matmul", "vlm verify"):
                out = fl[0](x16).astype(mx.float32)
                md = (f"{mx.abs(out - ref_q).max().item():.3g} / {mx.abs(out - ref_f).max().item():.3g}")
            elif kname == "quantized_matmul":
                md = f"0 / {mx.abs(ref_q - ref_f).max().item():.3g}"
            ts = {}
            for m in rows:
                if kname == "vlm verify" and m > 8:
                    ts[m] = None
                    continue
                x = (mx.random.normal((m, K)) * 0.5).astype(dt)
                mx.eval(x)
                try:
                    ts[m] = bench_chain(fl, x, iters=a.iters)
                except Exception as e:  # noqa: BLE001
                    ts[m] = None
                    print(f"<!-- {name} {kname} M={m}: {e} -->")
            per[(name, kname)] = (count, ts)
            base = per[(name, "quantized_matmul")][1][1] if 1 in rows else None
            cells = ["" if ts[m] is None else
                     f"{ts[m]*1e6:.0f}" + (f" (×{ts[m]/base:.2f})" if base else "") for m in rows]
            print(f"| {name} | {K}x{N} | {kname} | {md} | " + " | ".join(cells) + " |", flush=True)
            for m in rows:
                raw.append(dict(label=a.label, shape=name, K=K, N=N, count=count, kernel=kname, M=m, sec=ts[m]))
        del ws, sel, kern
        mx.clear_cache()
    print("\n| kernel | M | linears ms/step (9B, counts) | × vs qmm M=1 | vs qmm same M |")
    print("|---|---:|---:|---:|---:|")
    kn_all = []
    for (_, k) in per:
        if k not in kn_all:
            kn_all.append(k)
    def step(k, m):
        tot = 0.0
        for (name, kn), (count, ts) in per.items():
            if kn != k or count == 0:
                continue
            t = ts.get(m)
            if t is None:
                # shape unsupported by this kernel: fall back to quantized_matmul
                t = per[(name, "quantized_matmul")][1].get(m)
                if t is None:
                    return None
            tot += count * t
        return tot
    b1 = step("quantized_matmul", rows[0])
    for k in kn_all:
        for m in rows:
            st = step(k, m)
            sq = step("quantized_matmul", m)
            if st is None or not sq:
                continue
            print(f"| {k} | {m} | {st*1e3:.1f} | {st/b1:.2f} | {st/sq:.2f} |")
    # dispatcher estimate: per shape and M, the fastest kernel measured (excluding dequant)
    names = sorted({n for (n, _) in per})
    for m in rows:
        tot, pick = 0.0, []
        for n in names:
            cand = [(ts.get(m), k) for (nn, k), (count, ts) in per.items()
                    if nn == n and ts.get(m) is not None and not k.startswith("dequant")]
            count = per[(n, "quantized_matmul")][0]
            if not cand or count == 0:
                continue
            t, k = min(cand)
            tot += count * t
            pick.append(f"{n}:{k}")
        sq = step("quantized_matmul", m)
        print(f"| best-of (per shape) | {m} | {tot*1e3:.1f} | {tot/b1:.2f} | {tot/sq:.2f} |")
        print(f"<!-- M={m} picks: {', '.join(pick)} -->")
    if a.json:
        json.dump(raw, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
