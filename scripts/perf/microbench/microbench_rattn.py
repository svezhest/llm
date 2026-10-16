#!/usr/bin/env python3
"""Ragged multi-row decode attention (src/worker/rattn.py) vs today's per-row
loop (update_and_fetch + MLX's fused SDPA per row), synthetic 9B shapes:
Hq 16, Hkv 4, D 256, fp16 KV, L = 1 queries, 8 full-attention layers per
step. NEEDS THE GPU LOCK. Random K/V (one KVCache per row, reused for the 8
"layers": 600 MB+ of KV per layer at B 8 ragged, far beyond the 48 MB SLC).

Variants per config (interleaved per round, medians):
  rows       today: per row update_and_fetch + SDPA, concat (one layer x 8)
  rattn      per row update_and_fetch + ONE rattn launch for all rows
  rows_k     SDPA per row only (no update)        -- kernel cost
  rattn_k    rattn only (no update)
  upd        the per-row update_and_fetch alone
  bw         a plain sum over every row's K and V (the achievable read floor)
Every layer is evaluated before the next (RB_EVAL_LAYER=1, default): the rows'
caches are shared by the 8 simulated layers, and an unevaluated layer would make
the next layer's append copy the whole buffer (a microbench artifact: the
model's layers have their own caches).
Numerics (one layer, two regimes: flat q ~ N(0,1), peaky q x 8): rel rms and
max abs of MLX's SDPA and of rattn vs an fp32 reference (the
microbench_cascade.py criterion: rattn's error <= the per-row kernel's own).

  cd src/worker && PYTHONPATH=. .venv/bin/python ../../scripts/perf/microbench/microbench_rattn.py
  RB_CFG="ragged8=5,8,12,16,20,24,28,33;ragged16=5,8,12,16,20,24,28,33,5,8,12,16,20,24,28,33;u12=12x8"
  RB_STEPS=20 RB_ROUNDS=3 RB_SWEEP="512:4,256:4,1024:4,512:2"  (chunk:ns variants of rattn)
"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.getcwd())
import mlx.core as mx  # noqa: E402
from mlx_vlm.models.base import scaled_dot_product_attention  # noqa: E402
from mlx_vlm.models.cache import KVCache  # noqa: E402

import rattn  # noqa: E402

HQ, HKV, D, LAYERS = 16, 4, 256, 8
SCALE = D ** -0.5
STEPS = int(os.environ.get("RB_STEPS", "20"))
ROUNDS = int(os.environ.get("RB_ROUNDS", "3"))
CFG = os.environ.get("RB_CFG", "ragged8=5,8,12,16,20,24,28,33;u12=12x8;u20=20x8;u30=30x8;"
                     "ragged16=5,8,12,16,20,24,28,33,5,8,12,16,20,24,28,33")
SWEEP = [tuple(int(x) for x in v.split(":")) for v in os.environ.get("RB_SWEEP", "512:4").split(",")]
OUT = os.environ.get("RB_JSON", "")
VARS = os.environ.get("RB_VARS", "rows,rattn,rows_k,rattn_k,upd,bw").split(",")
EVAL_LAYER = os.environ.get("RB_EVAL_LAYER", "1") == "1"


def parse(spec):
    if "x" in spec:
        n, b = spec.split("x")
        return [int(n) * 1000] * int(b)
    return [int(x) * 1000 for x in spec.split(",")]


def make_rows(lens):
    rows = []
    for i, n in enumerate(lens):
        c = KVCache()
        mx.random.seed(100 + i)
        k = (mx.random.normal((1, HKV, n, D)) * 0.5).astype(mx.float16)
        v = (mx.random.normal((1, HKV, n, D)) * 0.5).astype(mx.float16)
        c.update_and_fetch(k, v)
        c.prefix_cache_reserve(n + 4096)     # no regrowth (a whole-buffer concat) inside the timed steps
        mx.eval(c.keys, c.values)
        rows.append(c)
    return rows


def step(rows, variant, q, kn, vn, chunk, ns):
    outs = []
    B = len(rows)
    for _ in range(LAYERS):
        if variant in ("rows", "rattn", "upd"):
            kv = [c.update_and_fetch(kn[i:i + 1], vn[i:i + 1]) for i, c in enumerate(rows)]
        else:
            kv = [(c.keys[..., :c.offset, :], c.values[..., :c.offset, :]) for c in rows]
        if variant == "bw":                   # read every row's K and V once: the achievable floor
            o = mx.concatenate([(c.keys[..., :c.offset, :].sum(axis=2, keepdims=True)
                                 + c.values[..., :c.offset, :].sum(axis=2, keepdims=True)).sum(axis=1, keepdims=True)
                                for c in rows], axis=0)
        elif variant == "upd":                # the per-row appends alone (made a dependency)
            o = mx.concatenate([(k[:, :, -1:, :] + v[:, :, -1:, :]).sum(axis=1, keepdims=True)
                                for k, v in kv], axis=0)
        elif variant.startswith("rows"):
            o = mx.concatenate([scaled_dot_product_attention(q[i:i + 1], k, v, cache=rows[i], scale=SCALE,
                                                             mask=None) for i, (k, v) in enumerate(kv)], axis=0)
        else:
            o = rattn.attention(q, [c.keys for c in rows], [c.values for c in rows], [c.offset for c in rows],
                                SCALE, chunk=chunk, ns=ns)
        outs.append(o)
        if EVAL_LAYER:   # one KVCache per row serves all 8 "layers": without this, layer j+1's append
            mx.eval(o)   # finds layer j's buffer still referenced (its SDPA) and copies the whole buffer
    return outs


def bench(rows, variant, chunk, ns):
    B = len(rows)
    q = mx.random.normal((B, HQ, 1, D)).astype(mx.float16)
    kn = (mx.random.normal((B, HKV, 1, D)) * 0.5).astype(mx.float16)
    vn = (mx.random.normal((B, HKV, 1, D)) * 0.5).astype(mx.float16)
    mx.eval(q, kn, vn)
    for _ in range(2):
        mx.eval(step(rows, variant, q, kn, vn, chunk, ns))
    ts = []
    mx.reset_peak_memory()
    a0 = mx.get_active_memory()
    for _ in range(STEPS):
        t = time.perf_counter()
        mx.eval(step(rows, variant, q, kn, vn, chunk, ns))
        ts.append(time.perf_counter() - t)
    return 1e3 * statistics.median(ts), (mx.get_peak_memory() - a0) / 1e6


def numerics(rows, chunk, ns):
    B = len(rows)
    res = {}
    for regime, mul in (("flat", 1.0), ("peaky", 8.0)):
        mx.random.seed(7)
        q = (mx.random.normal((B, HQ, 1, D)) * mul).astype(mx.float16)
        ks = [c.keys[..., :c.offset, :] for c in rows]
        vs = [c.values[..., :c.offset, :] for c in rows]
        ref = []
        for i in range(B):
            qg = q[i].astype(mx.float32).reshape(HKV, HQ // HKV, D)            # (4, 4, 256)
            s = (qg * SCALE) @ ks[i][0].astype(mx.float32).transpose(0, 2, 1)  # (4, 4, L)
            p = mx.softmax(s, axis=-1)
            ref.append((p @ vs[i][0].astype(mx.float32)).reshape(1, HQ, 1, D))
        ref = mx.concatenate(ref, axis=0)
        mlx_o = mx.concatenate([scaled_dot_product_attention(q[i:i + 1], ks[i], vs[i], cache=rows[i],
                                                             scale=SCALE, mask=None) for i in range(B)], axis=0)
        ra = rattn.attention(q, [c.keys for c in rows], [c.values for c in rows], [c.offset for c in rows],
                             SCALE, chunk=chunk, ns=ns)
        mx.eval(ref, mlx_o, ra)
        rr = mx.sqrt(mx.mean(ref * ref)).item()
        for name, o in (("mlx", mlx_o), ("rattn", ra)):
            d = o.astype(mx.float32) - ref
            res[f"{regime}_{name}"] = dict(rel_rms=mx.sqrt(mx.mean(d * d)).item() / rr,
                                           max_abs=mx.max(mx.abs(d)).item())
        d = ra.astype(mx.float32) - mlx_o.astype(mx.float32)
        res[f"{regime}_rattn_vs_mlx_max"] = mx.max(mx.abs(d)).item()
    return res


def main():
    print(f"mlx {mx.__version__}, device {mx.device_info().get('architecture')}, steps {STEPS}, rounds {ROUNDS}",
          flush=True)
    out = {}
    for spec in CFG.split(";"):
        name, lens = spec.split("=")
        lens = parse(lens)
        rows = make_rows(lens)
        kvgb = sum(c.offset for c in rows) * HKV * D * 2 * 2 * LAYERS / 1e9
        print(f"\n== {name}: B={len(lens)} lens(K)={[n // 1000 for n in lens]}; KV per step {kvgb:.2f} GB "
              f"= {kvgb / 400 * 1e3:.1f} ms at 400 GB/s", flush=True)
        num = numerics(rows, *SWEEP[0])
        for k, v in num.items():
            print(f"  num {k}: {v}", flush=True)
        res = {}
        vlist = [(v, c, n) for v in VARS for (c, n) in (SWEEP if v.startswith("rattn") else [SWEEP[0]])]
        for r in range(ROUNDS):
            for (v, c, n) in (vlist if r % 2 == 0 else vlist[::-1]):
                ms, peak = bench(rows, v, c, n)
                key = v if not v.startswith("rattn") else f"{v}@{c}:{n}"
                res.setdefault(key, []).append((ms, peak))
                print(f"  round {r} {key:16s} {ms:7.2f} ms/step  peak +{peak:.0f} MB", flush=True)
        summ = {k: dict(ms=statistics.median(x[0] for x in v), peak_mb=max(x[1] for x in v)) for k, v in res.items()}
        base = summ["rows"]["ms"] if "rows" in summ else None
        for k, v in summ.items():
            rel = f" ({v['ms'] / base:.3f} of rows)" if base else ""
            print(f"  {name} {k:16s} {v['ms']:7.2f} ms/step{rel}  peak +{v['peak_mb']:.0f} MB", flush=True)
        out[name] = dict(lens=lens, kv_gb=kvgb, numerics=num, timing=summ)
        del rows
        mx.clear_cache()
    if OUT:
        json.dump(out, open(OUT, "w"), indent=1)


main()
