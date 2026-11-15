#!/usr/bin/env python3
"""Night plan P2 (bit-exact launch cuts) before writing it: does one 1-row qmv
over concatenated weights beat the separate qmv launches it replaces? Real
weights of one checkpoint (default the 27B 4-bit), fp16 scales as the worker
casts them, decode input (1, 1, K). Per case, every layer's matmuls are queued
in ONE eval (the step's cost shape: no per-call sync), medians over rounds,
cases interleaved:

  mlp   gate_proj + up_proj (separate) vs [gate; up] (one call, split after)
  qkv   q_proj + k_proj + v_proj of the full-attention layers vs their concat

Also checks bit equality of the concatenated call's slices vs the separate
calls (the worker's B <= 2 invariant). NEEDS THE GPU LEASE.

  src/worker/.venv/bin/python scripts/perf/microbench/microbench_concat_qmv.py \
    --weights weights/qwen38-27b-mlx-4bit [--rounds 7] [--json out]
"""
import argparse
import glob
import json
import os
import statistics
import time

import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="weights/qwen38-27b-mlx-4bit")
ap.add_argument("--rounds", type=int, default=7)
ap.add_argument("--reps", type=int, default=4, help="passes over all layers per eval")
ap.add_argument("--json")
a = ap.parse_args()

cfg = json.load(open(os.path.join(a.weights, "config.json")))
q = cfg.get("quantization", cfg.get("quantization_config", {}))
GS, BITS = q.get("group_size", 64), q.get("bits", 4)
W = {}
for f in sorted(glob.glob(os.path.join(a.weights, "*.safetensors"))):
    for k, v in mx.load(f).items():
        if (".mlp.gate_proj." in k or ".mlp.up_proj." in k or ".self_attn.q_proj." in k
                or ".self_attn.k_proj." in k or ".self_attn.v_proj." in k) and "language_model" in k:
            W[k] = v.astype(mx.float16) if v.dtype == mx.bfloat16 else v


def mods(kind):
    out = {}
    for k in W:
        if k.endswith(".weight") and f".{kind}." in k:
            p = k[: -len(".weight")]
            out[p] = (W[p + ".weight"], W[p + ".scales"], W[p + ".biases"])
    return out


def layer_of(p):
    return int(p.split(".layers.")[1].split(".")[0])


def groups(kinds):
    by = {}
    for kind in kinds:
        for p, t in mods(kind).items():
            by.setdefault(layer_of(p), []).append((kind, t))
    order = {k: i for i, k in enumerate(kinds)}
    return [sorted(v, key=lambda x: order[x[0]]) for _, v in sorted(by.items()) if len(v) == len(kinds)]


def qmv(x, t):
    w, s, b = t
    return mx.quantized_matmul(x, w, scales=s, biases=b, transpose=True, group_size=GS, bits=BITS)


res = {}
for name, kinds in (("mlp", ("gate_proj", "up_proj")), ("qkv", ("q_proj", "k_proj", "v_proj"))):
    gs = groups(kinds)
    if not gs:
        print(f"{name}: no layers")
        continue
    cat = []
    for g in gs:
        w, s, b = (mx.concatenate([t[i] for _, t in g], axis=0) for i in range(3))
        cat.append(((w, s, b), [t[0].shape[0] for _, t in g]))
    mx.eval([c[0] for c in cat])
    cuts = [[sum(ns[:i + 1]) for i in range(len(ns) - 1)] for _, ns in cat]
    K = gs[0][0][1][0].shape[1] * 32 // BITS
    x = (mx.random.normal((1, 1, K)) * 0.5).astype(mx.float16)
    mx.eval(x)
    # bit equality of the slices
    eq = True
    for g, (t, ns) in zip(gs, cat):
        y = qmv(x, t)
        o = 0
        for (_, tt), n in zip(g, ns):
            eq &= bool(mx.array_equal(y[..., o:o + n], qmv(x, tt)).item())
            o += n

    def sep():
        return [qmv(x, t) for _ in range(a.reps) for g in gs for _, t in g]

    def one():
        return [mx.split(qmv(x, t), cuts[i], axis=-1) for _ in range(a.reps) for i, (t, _) in enumerate(cat)]

    fns = {"separate": sep, "concat": one}
    for f in fns.values():
        mx.eval(f())
    t = {k: [] for k in fns}
    for r in range(a.rounds):
        for k in (list(fns) if r % 2 == 0 else list(fns)[::-1]):
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval(fns[k]())
            t[k].append((time.perf_counter() - t0) / a.reps * 1e3)
    med = {k: statistics.median(v) for k, v in t.items()}
    nbytes = sum(sum(x.nbytes for x in tt) for g in gs for _, tt in g)
    res[name] = dict(layers=len(gs), launches_saved=len(gs) * (len(kinds) - 1), bit_equal=eq,
                     ms=med, raw=t, gbps={k: nbytes / (v / 1e3) / 1e9 for k, v in med.items()})
    print(f"{name}: {len(gs)} layers, separate {med['separate']:.2f} ms vs concat {med['concat']:.2f} ms "
          f"per step ({med['separate'] - med['concat']:+.2f} ms, {len(gs) * (len(kinds) - 1)} launches fewer); "
          f"{nbytes / 1e9:.2f} GB -> {res[name]['gbps']['separate']:.0f} / {res[name]['gbps']['concat']:.0f} GB/s; "
          f"slices bit-equal: {eq}", flush=True)
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
