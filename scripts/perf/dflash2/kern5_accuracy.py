"""5-bit mrqmm on the real 27B weights: numerics (GPU lease; one GDN layer,
one attention layer and lm_head from the safetensors, ~1.2 GB).

Per shape and row count M (x fp16: normal * 0.5 with outliers):
  * error vs an fp32 reference (x @ dequantize(w, fp32 scales / biases)^T, in
    chunks of output rows): max |d|, rms relative — for the kernel, for
    mx.quantized_matmul at the same M rows and for M solo qmv rows;
  * rows bit-equal to a solo qmv row? (share of equal elements);
  * the kernel's row m at M rows vs the same x row at every other M
    (determinism in the row count: equal inside a row bucket when the bucket's
    K split is fixed, see mrqmm._config) and with other rows beside it.

The bit width is the checkpoint's (config.json "quantization"): 5 for
weights/qwen38-27b-mlx-5bit, 4 for weights/qwen38-27b-mlx-4bit (QWEN_MRQMM4).

  cd src/worker && PYTHONPATH=. python ../../scripts/perf/dflash2/kern5_accuracy.py \\
      [--weights DIR] --out $PERF_OUT/kern5_accuracy.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import mlx.core as mx  # noqa: E402

import mrqmm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=os.path.join(HERE, "..", "..", "..", "weights", "qwen38-27b-mlx-5bit"))
ap.add_argument("--rows", default="3,4,6,8,9,12,16")
ap.add_argument("--gdn-layer", type=int, default=1)
ap.add_argument("--fa-layer", type=int, default=3)
ap.add_argument("--out", default=os.path.join(os.environ.get("PERF_OUT", "/tmp/perf"), "kern5_accuracy.json"))
a = ap.parse_args()
mx.set_default_device(mx.gpu)

_q = json.load(open(os.path.join(a.weights, "config.json")))["quantization"]
BITS, GS = _q["bits"], _q["group_size"]
assert GS == 64 and BITS in (5, 4), _q
ROWS = [int(x) for x in a.rows.split(",")]
_idx = json.load(open(os.path.join(a.weights, "model.safetensors.index.json")))["weight_map"]
_files = {}


def tensor(name):
    f = _idx[name]
    if f not in _files:
        _files[f] = mx.load(os.path.join(a.weights, f))
    return _files[f][name]


def linear(*names):
    parts = [(tensor(n + ".weight"), tensor(n + ".scales").astype(mx.float16),
              tensor(n + ".biases").astype(mx.float16)) for n in names]
    w, s, b = (mx.contiguous(mx.concatenate([p[j] for p in parts], axis=0)) for j in range(3))
    mx.eval(w, s, b)
    return w, s, b


G = f"language_model.model.layers.{a.gdn_layer}."
F = f"language_model.model.layers.{a.fa_layer}."
SHAPES = [
    ("mlp_gate", (G + "mlp.gate_proj",)), ("mlp_up", (G + "mlp.up_proj",)), ("mlp_down", (G + "mlp.down_proj",)),
    ("gdn_in_fused", tuple(G + "linear_attn." + n for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"))),
    ("gdn_out", (G + "linear_attn.out_proj",)),
    ("attn_q", (F + "self_attn.q_proj",)), ("attn_k", (F + "self_attn.k_proj",)),
    ("attn_v", (F + "self_attn.v_proj",)), ("attn_o", (F + "self_attn.o_proj",)),
    ("lm_head", ("language_model.lm_head",)),
]


def qmm(x, w, s, b):
    return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=GS, bits=BITS)


def ref32(x, w, s, b, chunk=16384):
    """fp32 reference, [rows, N]."""
    xf = x.reshape(-1, x.shape[-1]).astype(mx.float32)
    out = []
    for lo in range(0, w.shape[0], chunk):
        deq = mx.dequantize(w[lo:lo + chunk], s[lo:lo + chunk].astype(mx.float32),
                            b[lo:lo + chunk].astype(mx.float32), group_size=GS, bits=BITS)
        y = xf @ deq.T
        mx.eval(y)
        out.append(y)
    return mx.concatenate(out, axis=1)


def err(y, ref):
    d = y.reshape(ref.shape).astype(mx.float32) - ref
    return dict(max_abs=mx.abs(d).max().item(),
                rms_rel=(mx.sqrt((d * d).mean()) / mx.sqrt((ref * ref).mean())).item())


res = {}
worst = dict(k5=0.0, mlx=0.0, qmv=0.0)
for name, mods in SHAPES:
    w, s, b = linear(*mods)
    N, K = w.shape[0], w.shape[1] * 32 // BITS
    mx.random.seed(len(name) + N)
    X = mx.random.normal((16, 1, K)) * 0.5
    X[..., ::211] = X[..., ::211] * 30.0
    X = X.astype(mx.float16)
    Z = (mx.random.normal((16, 1, K)) * 0.5).astype(mx.float16)        # other rows' data
    mx.eval(X, Z)
    ref = ref32(X, w, s, b)
    solo = mx.concatenate([qmm(X[i:i + 1], w, s, b) for i in range(16)]).reshape(16, N)
    mx.eval(ref, solo)
    out = {"K": K, "N": N, "ref_abs_max": mx.abs(ref).max().item(), "rows": {}}
    ys = {}
    for M in ROWS:
        y = mrqmm.mrqmm(X[:M], w, s, b).reshape(M, N)
        q = qmm(X[:M], w, s, b).reshape(M, N)
        mx.eval(y, q)
        ys[M] = y
        MB = 1 if M <= 8 else 2
        e = dict(cfg=list(mrqmm._config(K, N, MB, BITS)), k5=err(y, ref[:M]), mlx=err(q, ref[:M]),
                 qmv=err(solo[:M], ref[:M]),
                 k5_eq_solo_qmv=(y == solo[:M]).astype(mx.float32).mean().item(),
                 mlx_eq_solo_qmv=(q == solo[:M]).astype(mx.float32).mean().item(),
                 k5_max_vs_solo_qmv=mx.abs(y.astype(mx.float32) - solo[:M].astype(mx.float32)).max().item())
        # the same x row with other rows beside it (rows 1.. replaced)
        y2 = mrqmm.mrqmm(mx.concatenate([X[:1], Z[1:M]]), w, s, b).reshape(M, N)
        e["row0_independent_of_neighbours"] = bool(mx.array_equal(y2[0], y[0]).item())
        out["rows"][M] = e
        for k in worst:
            worst[k] = max(worst[k], e[k]["rms_rel"])
    base = ROWS[0]
    out["rows_equal_across_M"] = {
        f"{M}=={base}": bool(mx.array_equal(ys[M][:base], ys[base]).item()) for M in ROWS[1:]}
    res[name] = out
    r8 = out["rows"].get(8) or out["rows"][ROWS[0]]
    print(f"{name:13s} K={K:5d} N={N:6d} rms rel k5 {r8['k5']['rms_rel']:.2e} mlx {r8['mlx']['rms_rel']:.2e} "
          f"qmv {r8['qmv']['rms_rel']:.2e} | max|d| k5 {r8['k5']['max_abs']:.3g} mlx {r8['mlx']['max_abs']:.3g} "
          f"(|ref| max {out['ref_abs_max']:.3g}) | == solo qmv: k5 {r8['k5_eq_solo_qmv']:.1%} "
          f"mlx {r8['mlx_eq_solo_qmv']:.1%} | rows equal across M: {out['rows_equal_across_M']}", flush=True)
    del w, s, b, ref, solo, ys
    mx.clear_cache()
print("worst rms rel over shapes and rows:", {k: f"{v:.2e}" for k, v in worst.items()})
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump({"shapes": res, "worst_rms_rel": worst, "args": vars(a)}, open(a.out, "w"), indent=1)
print("->", a.out)
