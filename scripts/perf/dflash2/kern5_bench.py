"""5-bit multi-row matmul (mrqmm, QWEN_MRQMM5) on the real 27B weights: kernel
cost per matmul and the linears of one decode step (GPU lease; no model:
the tensors of a few layers + lm_head straight from the safetensors, ~2.6 GB
at the default --mb 256).

Per shape and row count, interleaved in one process, medians over rounds:

  qmv1     mx.quantized_matmul, ONE row (1, 1, K): the solo decode step's qmv
  mlx      mx.quantized_matmul, rows as (rows, 1, K): today's batched decode
           (per-row qmv below MLX's row limit, then qmm / split-K)
  k5       mrqmm.mrqmm, its default config for the row bucket
  k5:AxS   mrqmm with NA = A fragments per simdgroup, KS = S simdgroups over
           K (--sweep 2x8,4x4,...)
  mlx_t    mx.quantized_matmul on (1, rows, K) (--mlx-t: a DFlash verify
           block above QWEN_VERIFY_EXACT_MAX_T)
  exact    mlx-vlm's exact verify kernels on (1, rows, K) (--exact; rows <= 8:
           what a DFlash verify block takes today, QWEN_VERIFY_EXACT_MAX_T)

Each timed call is a chain over distinct weights (the same projection of
several layers, >= --mb MB per shape; lm_head is one 800 MB copy), so they
stream from DRAM as in a decode step. "step" = sum over shapes x their count
in one 27B forward (gate/up 128, down 64, fused GDN in 48, GDN out 48, q 16,
k/v 32, o 16, lm_head 1); its ratio to the qmv1 step measured in the same
rows block is the number to read (the GPU drifts between blocks).

--random --bits B: random packed weights of the same shapes and counts at
another bit width (4, 5 or 8: the price of a build of the same model at that
width; --bits 5 --random checks that random weights time like the real ones).
Real weights: --bits is the checkpoint's (config.json "quantization"; 4 for
weights/qwen38-27b-mlx-4bit, QWEN_MRQMM4).

  cd src/worker && PYTHONPATH=. python ../../scripts/perf/dflash2/kern5_bench.py \\
      [--weights DIR] [--rounds 5] [--rows 2,3,4,6,8,12,16] [--sweep 2x8,4x4] [--exact] \\
      [--random --bits 4] --out $PERF_OUT/kern5_bench.json
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.getcwd())
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

import mrqmm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=os.path.join(HERE, "..", "..", "..", "weights", "qwen38-27b-mlx-5bit"))
ap.add_argument("--rounds", type=int, default=5)
ap.add_argument("--rows", default="2,3,4,6,8,12,16")
ap.add_argument("--reps", type=int, default=16, help="calls per timed chain (>= the copies of a shape)")
ap.add_argument("--mb", type=int, default=256, help="distinct weights per shape, MB")
ap.add_argument("--sweep", default="", help="extra kernel configs NAxKS, comma separated")
ap.add_argument("--sweep-rows", default="8,16")
ap.add_argument("--only", default="", help="shape names, comma separated")
ap.add_argument("--k5-rows", default="3,4,6,8,12,16", help="rows the kernel is timed at (add 1,2 to look)")
ap.add_argument("--exact", action="store_true")
ap.add_argument("--bits", type=int, default=0, help="random weights' width (real: the checkpoint's)")
ap.add_argument("--random", action="store_true", help="random weights of the 27B's shapes (needed for --bits != 5)")
ap.add_argument("--mlx-t", action="store_true", help="also quantized_matmul on (1, rows, K): a verify block's shape")
ap.add_argument("--out", default=os.path.join(os.environ.get("PERF_OUT", "/tmp/perf"), "kern5_bench.json"))
a = ap.parse_args()
mx.set_default_device(mx.gpu)

GS = 64
if a.random:
    BITS = a.bits or 5
else:
    _q = json.load(open(os.path.join(a.weights, "config.json")))["quantization"]
    BITS = _q["bits"]
    assert _q["group_size"] == GS and a.bits in (0, BITS), (_q, a.bits)
L = "language_model.model.layers.%d."
N_LAYERS = 64
FA = [i for i in range(N_LAYERS) if i % 4 == 3]
GDN = [i for i in range(N_LAYERS) if i % 4 != 3]
ALL = list(range(N_LAYERS))
# (name, layers it exists in, module names (a tuple = concatenated: the worker's fused GDN in-projection),
#  calls per decode step)
SHAPES = [
    ("mlp_gate/up", ALL, ["mlp.gate_proj", "mlp.up_proj"], 128),
    ("mlp_down", ALL, ["mlp.down_proj"], 64),
    ("gdn_in_fused", GDN, [("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b",
                            "linear_attn.in_proj_a")], 48),
    ("gdn_out", GDN, ["linear_attn.out_proj"], 48),
    ("attn_q", FA, ["self_attn.q_proj"], 16),
    ("attn_k/v", FA, ["self_attn.k_proj", "self_attn.v_proj"], 32),
    ("attn_o", FA, ["self_attn.o_proj"], 16),
    ("lm_head", None, ["language_model.lm_head"], 1),
]
if a.only:
    SHAPES = [s for s in SHAPES if s[0] in a.only.split(",")]
ROWS = [int(x) for x in a.rows.split(",")]
K5_ROWS = {int(x) for x in a.k5_rows.split(",")}
SWEEP = [tuple(int(v) for v in c.split("x")) for c in a.sweep.split(",") if c]
SWEEP_ROWS = {int(x) for x in a.sweep_rows.split(",")}

_idx = None if a.random else json.load(open(os.path.join(a.weights, "model.safetensors.index.json")))["weight_map"]
_files = {}
# K, N of the 27B's modules (config: hidden 5120, intermediate 17408, 24 heads x 256 with 4 KV heads,
# GDN 16 key / 48 value heads x 128, vocab 248320), for --random
DIMS = {"mlp.gate_proj": (5120, 17408), "mlp.up_proj": (5120, 17408), "mlp.down_proj": (17408, 5120),
        "linear_attn.in_proj_qkv": (5120, 10240), "linear_attn.in_proj_z": (5120, 6144),
        "linear_attn.in_proj_b": (5120, 48), "linear_attn.in_proj_a": (5120, 48),
        "linear_attn.out_proj": (6144, 5120), "self_attn.q_proj": (5120, 12288),
        "self_attn.k_proj": (5120, 1024), "self_attn.v_proj": (5120, 1024), "self_attn.o_proj": (6144, 5120),
        "language_model.lm_head": (5120, 248320)}
_seed = [0]


def random_linear(name):
    """Random packed words + fp16 scales / biases (the time does not depend on the values)."""
    K, N = DIMS[name]
    _seed[0] += 1
    mx.random.seed(_seed[0])
    w = mx.random.randint(0, 256, (N, K * BITS // 8), dtype=mx.uint8).view(mx.uint32)
    s = (mx.random.uniform(shape=(N, K // GS)) * 2e-3 + 1e-4).astype(mx.float16)
    b = (mx.random.normal((N, K // GS)) * 3e-2).astype(mx.float16)
    return w, s, b


def tensor(name):
    f = _idx[name]
    if f not in _files:
        _files[f] = mx.load(os.path.join(a.weights, f))
    return _files[f][name]


def linear(prefix):
    """(w, scales, biases) of one module, scales / biases cast to fp16 as the
    worker does at load (QWEN_FP16=1; the checkpoint stores bf16)."""
    if a.random:
        return random_linear(prefix.split("layers.")[-1].split(".", 1)[-1] if "layers." in prefix else prefix)
    return (tensor(prefix + ".weight"), tensor(prefix + ".scales").astype(mx.float16),
            tensor(prefix + ".biases").astype(mx.float16))


def load(layers, mods):
    out, size = [], 0
    for i in (layers if layers is not None else [None]):
        for m in mods:
            names = m if isinstance(m, tuple) else (m,)
            parts = [linear((L % i if i is not None else "") + n) for n in names]
            w, s, b = (mx.concatenate([p[j] for p in parts], axis=0) if len(parts) > 1 else parts[0][j]
                       for j in range(3))
            w, s, b = mx.contiguous(w), mx.contiguous(s), mx.contiguous(b)
            mx.eval(w, s, b)
            out.append((w, s, b))
            size += w.nbytes + s.nbytes + b.nbytes
        if size >= a.mb << 20:
            break
    return out, size


def qmm(x, w, s, b):
    return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=GS, bits=BITS)


def module(w, s, b):
    lin = nn.QuantizedLinear(w.shape[1] * 32 // BITS, w.shape[0], bias=False, group_size=GS, bits=BITS)
    lin.weight, lin.scales, lin.biases = w, s, b
    return lin


def timed(fn, ws, x, reps):
    ys = [fn(x, ws[i % len(ws)]) for i in range(reps)]
    mx.eval(ys)                                     # warm (compile, first touch)
    t = time.perf_counter()
    ys = [fn(x, ws[i % len(ws)]) for i in range(reps)]
    mx.eval(ys)
    return 1e6 * (time.perf_counter() - t) / reps


exact_fn = None
if a.exact:
    from mlx_vlm.models.quantized_verifier import optimized_affine_linear as exact_fn

res, meta = {}, {}
for name, layers, mods, cnt in SHAPES:
    ws, size = load(layers, mods)
    N, K = ws[0][0].shape[0], ws[0][0].shape[1] * 32 // BITS
    assert mrqmm._bits_of(K, ws[0][0].shape[1]) == BITS and ws[0][1].dtype == mx.float16
    lins = [module(*w) for w in ws] if exact_fn else None
    reps = max(len(ws), 4 if name == "lm_head" else a.reps)
    meta[name] = dict(K=K, N=N, count=cnt, copies=len(ws), mb=size / 2**20, reps=reps)
    x16 = (mx.random.normal((16, 1, K)) * 2).astype(mx.float16)
    mx.eval(x16)
    for rows in ROWS:
        x = x16[:rows]
        x1 = x16[:1]
        xt = mx.contiguous(x.reshape(1, rows, K))
        mx.eval(xt)
        MB = 1 if rows <= 8 else 2
        fns = {"qmv1": lambda x, w: qmm(x1, *w), "mlx": lambda x, w: qmm(x, *w)}
        if a.mlx_t:
            fns["mlx_t"] = lambda x, w: qmm(xt, *w)
        if rows in K5_ROWS:
            fns["k5"] = lambda x, w: mrqmm.mrqmm(x, *w)
            if rows in SWEEP_ROWS:
                for c in SWEEP:
                    if N % (8 * c[0]) == 0:
                        fns[f"k5:{c[0]}x{c[1]}"] = (lambda c: lambda x, w: mrqmm.mrqmm(x, *w, cfg=c))(c)
        if exact_fn and rows <= 8:
            probe = exact_fn(lins[0], xt)
            if probe is not None:
                by_w = {id(w[0]): lin for w, lin in zip(ws, lins)}
                fns["exact"] = lambda x, w: exact_fn(by_w[id(w[0])], xt)
        order = list(fns)
        samples = {v: [] for v in order}
        for r in range(a.rounds):
            for v in (order if r % 2 == 0 else order[::-1]):
                samples[v].append(timed(fns[v], ws, x, reps))
        res[f"{name}/{rows}"] = {v: dict(median_us=statistics.median(s), samples_us=s) for v, s in samples.items()}
        cfg = mrqmm._config(K, N, MB, BITS)
        print(f"{name:13s} K={K:5d} N={N:6d} rows={rows:2d} " + " ".join(
            f"{v}={statistics.median(s):8.1f}us" for v, s in samples.items())
            + (f"  [k5 = {cfg[0]}x{cfg[1]}]" if "k5" in fns else ""), flush=True)
    del ws, lins
    mx.clear_cache()

step = {}
if not a.only:
    for rows in ROWS:
        variants = set.intersection(*(set(res[f"{n}/{rows}"]) for n, *_ in SHAPES))
        tot = {v: sum(cnt * res[f"{n}/{rows}"][v]["median_us"] for n, _, _, cnt in SHAPES) / 1e3
               for v in variants}
        step[rows] = dict({f"{v}_ms": t for v, t in tot.items()},
                          **{f"{v}_x": t / tot["qmv1"] for v, t in tot.items() if v != "qmv1"})
        print(f"step linears rows={rows:2d}: " + ", ".join(
            f"{v} {tot[v]:7.2f} ms" + ("" if v == "qmv1" else f" (x{tot[v] / tot['qmv1']:.2f})")
            for v in sorted(tot, key=lambda v: (v != "qmv1", v != "mlx", v))), flush=True)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump({"shapes": meta, "per_call": res, "step_linears": step, "args": vars(a),
           "mlx": mx.__version__}, open(a.out, "w"), indent=1)
print("->", a.out)
