#!/usr/bin/env python3
"""Prefill 8-bit qmm by row count M: mx.quantized_matmul vs pfqmm (64x32x32)
vs tile/raster candidates, on the REAL 9B weights (Ornith-1.5 8-bit, fp16
scales/biases). NEEDS THE GPU LEASE.

Shapes (K -> N, calls per 32-layer slab): gate/up 4096->12288 (64), down
12288->4096 (32), GDN qkv / attn q 4096->8192 (24+8), GDN z / out, attn o
4096->4096 (24+24+8), attn k/v 4096->1024 (16). lm_head and the GDN b/a
(N=32, always split-K) are excluded. Each shape cycles over a few layers'
weights (distinct copies, as in the model) so small matrices do not sit in
the SLC.

Variants are MLX's qmm_t code (QuantizedBlockLoader + steel BlockMMA from the
wheel's headers) with other tiles: BMxBKxBN, WMxWN simdgroups, optional
steel-style swizzle (M-tiles grouped 2^s per N column). Every variant keeps
the per-element k order (BK-steps of 8x8 MMAs, fp32 accumulators), so it is
expected bit-identical to quantized_matmul wherever MLX itself runs qmm_t;
--check verifies it (per shape, per M, random fp16 x).

Variant names: `mlx` = mx.quantized_matmul; `pfqmm` = pfqmm.pfqmm as the
worker has it now (since 2026-10-17: 64-row tiles + a 32-row last tile when
<= 32 rows are left, "mix"); `pf64` = 64x32x32 without the mixed last tile
(pfqmm's kernel before 2026-10-17); `mix*` = the mixed kernel, here with an
optional swizzle. Records carry `pfqmm_use` (pfqmm._use: would the worker
take the kernel at this M) and `old_rule` (the pre-2026-10-17 rule, M >= 128
and ceil64(M) <= 1.06 ceil32(M)); `summ_pfqmm_m.py` aggregates the JSON.

Interleaved: every round times every (shape, M, variant), the variant order
rotated per round; medians over rounds.

  cd src/worker && .venv/bin/python ../../scripts/perf/microbench/microbench_pfqmm_m.py \
      --rounds 7 --check --json $PERF_OUT/pfqmm_m.json
  --variants mlx,pf64,32x32x64 ...   --ms 64,128,256,432,512,768,1024,2048
"""
import argparse
import json
import os
import statistics
import sys
import time

import mlx.core as mx

REPO = "/Users/user/Projects/llm"
sys.path.insert(0, os.path.join(REPO, "src/worker"))
import pfqmm  # noqa: E402

MODEL = os.path.join(REPO, "weights/ornith15-9b-mlx-8bit")
PFX = "language_model.model.layers."
# name, K, N, calls per slab, [(layer, module)]
SHAPES = [
    ("gate/up", 4096, 12288, 64, [(0, "mlp.gate_proj"), (0, "mlp.up_proj"), (1, "mlp.gate_proj"),
                                  (1, "mlp.up_proj")]),
    ("down", 12288, 4096, 32, [(0, "mlp.down_proj"), (1, "mlp.down_proj"), (2, "mlp.down_proj")]),
    ("qkv/q", 4096, 8192, 32, [(0, "linear_attn.in_proj_qkv"), (1, "linear_attn.in_proj_qkv"),
                               (3, "self_attn.q_proj"), (2, "linear_attn.in_proj_qkv")]),
    ("z/out/o", 4096, 4096, 56, [(0, "linear_attn.in_proj_z"), (0, "linear_attn.out_proj"),
                                 (3, "self_attn.o_proj"), (1, "linear_attn.in_proj_z"),
                                 (1, "linear_attn.out_proj"), (2, "linear_attn.in_proj_z")]),
    ("k/v", 4096, 1024, 16, [(3, "self_attn.k_proj"), (3, "self_attn.v_proj"), (7, "self_attn.k_proj"),
                             (7, "self_attn.v_proj")]),
]

# tag -> (BM, BK, BN, WM, WN, swizzle_log)
VARIANTS = {
    "pf64": (64, 32, 32, 2, 2, 0),          # pfqmm's kernel before the mixed last tile (2026-10-17)
    "32x32x32": (32, 32, 32, 2, 2, 0),      # MLX's own tile, through metal_kernel
    "32x32x64": (32, 32, 64, 2, 2, 0),
    "32x32x128": (32, 32, 128, 2, 2, 0),
    "32x64x32": (32, 64, 32, 2, 2, 0),
    "64x64x32": (64, 64, 32, 2, 2, 0),
    "64x32x64": (64, 32, 64, 2, 2, 0),
    "64x32x32w41": (64, 32, 32, 4, 1, 0),
    "64x32x32w14": (64, 32, 32, 1, 4, 0),
    "128x32x32w42": (128, 32, 32, 4, 2, 0),
    "64x32x64w24": (64, 32, 64, 2, 4, 0),
    "64x32x32s1": (64, 32, 32, 2, 2, 1),
    "64x32x32s2": (64, 32, 32, 2, 2, 2),
    "32x32x32w41": (32, 32, 32, 4, 1, 0),
    "64x32x32s3": (64, 32, 32, 2, 2, 3),
    "64x32x64s1": (64, 32, 64, 2, 2, 1),
    "64x32x64w24s1": (64, 32, 64, 2, 4, 1),
    "32x32x32s1": (32, 32, 32, 2, 2, 1),
    "32x32x32s2": (32, 32, 32, 2, 2, 2),
    "32x32x32s3": (32, 32, 32, 2, 2, 3),
    "128x32x32w42s1": (128, 32, 32, 4, 2, 1),
    # mixed: 64-row tiles, the last one 32 rows when <= 32 rows are left (MLX's tile there);
    # "mix" is pfqmm's kernel since 2026-10-17
    "mix": (64, 32, 32, 2, 2, 0, 1),
    "mixs1": (64, 32, 32, 2, 2, 1, 1),
    "mixs2": (64, 32, 32, 2, 2, 2, 1),
}

_SRC = r"""
  constexpr int BK_padded = BK + 16 / sizeof(T);
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  const int tx = int(threadgroup_position_in_grid.x) >> SWZ;
  const int ty = (int(threadgroup_position_in_grid.y) << SWZ)
      + (int(threadgroup_position_in_grid.x) & ((1 << SWZ) - 1));
  const int M = dims[0], K = dims[1], N = dims[2];
  const int y_row = ty * BM;
  if (y_row >= M) {
    return;
  }
  if constexpr (MIX) {
    if (M - y_row <= BM / 2) {      // the last tile: half height (MLX's own 32-row tile at BM 64)
      pfg_tile<T, GS, BITS, BM / 2, BK, BN, WM, WN>(w, scales, biases, x, y, Xs, Ws, M, K, N, y_row, tx * BN,
          simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
      return;
    }
  }
  pfg_tile<T, GS, BITS, BM, BK, BN, WM, WN>(w, scales, biases, x, y, Xs, Ws, M, K, N, y_row, tx * BN,
      simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""

# pfqmm._IMPL with WM/WN as template parameters and the tile origin passed in
# (the kernel source picks the tile: swizzle, mixed last tile).
_IMPL = r"""
template <typename T, int group_size, int bits, int BM, int BK, int BN, int WM, int WN>
METAL_FUNC void pfg_tile(
    const device uint32_t* w, const device T* scales, const device T* biases,
    const device T* x, device T* y, threadgroup T* Xs, threadgroup T* Ws,
    int M, int K, int N, int y_row, int y_col, uint simd_gid, uint simd_lid) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using mma_t = mlx::steel::BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t = mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  auto wl = (const device uint8_t*)w;
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;
  const short num_els = min(BM, M - y_row);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);
  if (num_els < BM) {
    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(BK, num_els));
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
      loader_x.next();
      loader_w.next();
    }
  } else {
    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_unsafe();
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
      loader_x.next();
      loader_w.next();
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM) {
    mma_op.store_result_safe(y, N, short2(BN, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}
"""

_K = {}
_DIMS = {}


def kernel(tag, bits=8, gs=64):
    key = (tag, bits, gs)
    if key not in _K:
        BM, BK, BN, WM, WN, SWZ, MIX = (VARIANTS[tag] + (0,))[:7]
        hdr = pfqmm._header()[: -len(pfqmm._IMPL)] + _IMPL      # headers without pfqmm's own impl
        consts = (f"using T = half;\n  constexpr int BM = {BM}, BK = {BK}, BN = {BN}, WM = {WM}, WN = {WN}, "
                  f"SWZ = {SWZ}, MIX = {MIX}, BITS = {bits}, GS = {gs};\n")
        _K[key] = mx.fast.metal_kernel(
            name=f"pfg_b{bits}_g{gs}_{tag.replace('x', '_')}",
            input_names=["w", "scales", "biases", "x", "dims"], output_names=["y"],
            source=consts + _SRC, header=hdr)
    return _K[key]


def run(tag, x, w, s, b, bits=8, gs=64):
    BM, BK, BN, WM, WN, SWZ = VARIANTS[tag][:6]
    K = x.shape[-1]
    N = w.shape[0]
    M = x.size // K
    dims = _DIMS.get((M, K, N))
    if dims is None:
        dims = _DIMS[(M, K, N)] = mx.array([M, K, N], dtype=mx.int32)
    tg = 32 * WM * WN
    tn, tm = N // BN, -(-M // BM)
    gx, gy = tn << SWZ, -(-tm // (1 << SWZ))
    return kernel(tag, bits, gs)(
        inputs=[w, s, b, x, dims], grid=(gx * tg, gy, 1), threadgroup=(tg, 1, 1),
        output_shapes=[(*x.shape[:-1], N)], output_dtypes=[mx.float16])[0]


def call(tag, x, w, s, b):
    if tag == "mlx":
        return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=8)
    if tag == "pfqmm":
        return pfqmm.pfqmm(x, w, s, b, 8, 64)
    return run(tag, x, w, s, b)


def valid(tag, N):
    if tag in ("mlx", "pfqmm"):
        return True
    BM, BK, BN, WM, WN, SWZ = VARIANTS[tag][:6]
    return N % BN == 0 and BM % (8 * WM) == 0 and BN % (8 * WN) == 0


class _Mod(dict):
    """Enough of a QuantizedLinear for pfqmm._use (weight shape, group size)."""
    group_size = 64


def worker_takes(M, K, N):
    m = _Mod(weight=mx.zeros((N, K // 4), mx.uint32))
    return bool(pfqmm._use(m, mx.zeros((1, M, K), mx.float16)))


def old_rule(M, K, N):
    return bool(M >= 128 and -(-M // 64) * 64 <= 1.06 * (-(-M // 32) * 32) and pfqmm._mlx_uses_qmm(M, K, N, 64))


def load_weights():
    keys = {f"{PFX}{l}.{m}" for _, _, _, _, mods in SHAPES for l, m in mods}
    idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
    files = {idx[k + ".weight"] for k in keys}
    out = {}
    for f in sorted(files):
        d = mx.load(os.path.join(MODEL, f))
        for k in keys:
            if k + ".weight" in d:
                out[k] = (d[k + ".weight"], d[k + ".scales"].astype(mx.float16),
                          d[k + ".biases"].astype(mx.float16))
        del d
    mx.eval(list(out.values()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="64,128,256,432,512,768,1024,2048")
    ap.add_argument("--variants", default="mlx,pfqmm," + ",".join(VARIANTS))
    ap.add_argument("--shapes", default=",".join(s[0] for s in SHAPES))
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--target-ms", type=float, default=25.0, help="GPU time per timed batch")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    Ms = [int(v) for v in a.ms.split(",")]
    tags = a.variants.split(",")
    shapes = [s for s in SHAPES if s[0] in a.shapes.split(",")]
    W = load_weights()
    print(f"mlx {mx.__version__}, {mx.device_info().get('device_name')}", flush=True)
    xs = {}
    for name, K, N, cnt, mods in shapes:
        for M in Ms:
            xs[(K, M)] = (mx.random.normal((1, M, K), key=mx.random.key(M * 7 + K)) * 0.5).astype(mx.float16)
    mx.eval(list(xs.values()))

    exact = {}
    for name, K, N, cnt, mods in shapes:
        w, s, b = W[f"{PFX}{mods[0][0]}.{mods[0][1]}"]
        for M in Ms:
            x = xs[(K, M)]
            ref = call("mlx", x, w, s, b)
            for t in tags:
                if t == "mlx" or not valid(t, N):
                    continue
                y = call(t, x, w, s, b)        # compiles; equality only reported with --check
                if a.check:
                    eq = mx.array_equal(y, ref).item()
                    exact[(name, M, t)] = eq
                mx.eval(y)
    if a.check:
        for name, K, N, cnt, mods in shapes:
            for M in Ms:
                qmm = pfqmm._mlx_uses_qmm(M, K, N, 64)
                bad = [t for t in tags if (name, M, t) in exact and not exact[(name, M, t)]]
                print(f"check {name:8s} M={M:5d} mlx={'qmm_t ' if qmm else 'splitK'}: "
                      f"{'all bit-identical' if not bad else 'DIFFER: ' + ','.join(bad)}", flush=True)

    # batch sizes: enough calls to reach target-ms with the mlx kernel at ~8 TFLOPS
    nrep = {}
    for name, K, N, cnt, mods in shapes:
        for M in Ms:
            per = 2.0 * M * N * K / 8e12 * 1e3
            nrep[(name, M)] = max(len(mods), int(round(a.target_ms / per)))

    times = {}
    for r in range(a.rounds):
        t0 = time.time()
        for name, K, N, cnt, mods in shapes:
            ws = [W[f"{PFX}{l}.{m}"] for l, m in mods]
            for M in Ms:
                x = xs[(K, M)]
                n = nrep[(name, M)]
                order = [t for t in tags if valid(t, N)]
                k = r % len(order)
                order = order[k:] + order[:k]
                for t in order:
                    mx.synchronize()
                    t1 = time.perf_counter()
                    outs = [call(t, x, *ws[i % len(ws)]) for i in range(n)]
                    mx.eval(outs)
                    dt = (time.perf_counter() - t1) / n
                    times.setdefault((name, M, t), []).append(dt)
                    del outs
        print(f"round {r + 1}/{a.rounds} {time.time() - t0:.1f} s", flush=True)

    res = []
    for name, K, N, cnt, mods in shapes:
        print(f"\n{name} K={K} N={N} ({cnt} calls/slab)  ms (TFLOPS), x vs mlx", flush=True)
        for M in Ms:
            base = statistics.median(times[(name, M, "mlx")]) if (name, M, "mlx") in times else None
            fl = 2.0 * M * N * K
            cells = []
            for t in tags:
                ts = times.get((name, M, t))
                if not ts:
                    continue
                med = statistics.median(ts)
                rec = dict(shape=name, K=K, N=N, calls=cnt, M=M, variant=t, ms=med * 1e3,
                           tflops=fl / med / 1e12, all_ms=[v * 1e3 for v in ts],
                           mlx_path="qmm_t" if pfqmm._mlx_uses_qmm(M, K, N, 64) else "splitk",
                           pfqmm_use=worker_takes(M, K, N), old_rule=old_rule(M, K, N),
                           exact=exact.get((name, M, t)))
                res.append(rec)
                cells.append(f"{t} {med * 1e3:.3f} ({fl / med / 1e12:.2f}"
                             + (f", x{base / med:.3f}" if base else "") + ")"
                             + ("" if rec["exact"] in (None, True) else " NE"))
            print(f"  M={M:5d} [{'qmm_t' if pfqmm._mlx_uses_qmm(M, K, N, 64) else 'splitK'}] "
                  + " | ".join(cells), flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(mlx=mx.__version__, rounds=a.rounds, results=res), f, indent=1)
        print("wrote", a.json)


if __name__ == "__main__":
    main()
