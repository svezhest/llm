"""GPU cost of the worker's sampler, QWEN_SAMPLER_TOPK_FAST off vs on, isolated
(no model): (B, 248320) fp16 logits, the production entry points (B = 1: the
row's sampler; B >= 2: sample_rows, one group), one process, the two variants
interleaved per repetition (ABBA), median ms per step over --reps after a
warm-up. Also the old pieces (topk, argsort) for reference.

  cd src/worker && PYTHONPATH=. .venv/bin/python ../../scripts/perf/sampler_topk/sampler_ab.py \\
    --json $PERF_OUT/sampler_topk/sampler_ab.json
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.getcwd())
import mlx.core as mx  # noqa: E402
import worker as W  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--B", default="1,4,8,16")
ap.add_argument("--reps", type=int, default=40)
ap.add_argument("--json", default="")
a = ap.parse_args()

V = 248320
CFG = {
    "card": dict(temperature=0.6, top_p=0.95, top_k=20),          # Qwen card, non-thinking
    "card-think": dict(temperature=1.0, top_p=0.95, top_k=20),    # thinking card
    "topk50": dict(temperature=0.7, top_k=50),
    "card+minp": dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.05),
}
mx.random.seed(0)
LG = {B: (mx.random.normal((B, V)) * 2.5).astype(mx.float16) for B in map(int, a.B.split(","))}
mx.eval(list(LG.values()))


def make(cfg, B, fast):
    W.SAMPLER_TOPK_FAST = fast
    ss = [W.make_sampler(dict(W.SAMPLING_KWARGS, presence_penalty=0.0, **cfg), seed=i) for i in range(B)]
    W.SAMPLER_TOPK_FAST = False
    lg = LG[B]
    xs = [lg[i:i + 1] for i in range(B)]
    if B == 1:
        return lambda: ss[0](xs[0])
    return lambda: mx.concatenate(W.sample_rows(ss, xs))


def timed(fn):
    t = time.perf_counter()
    mx.eval(fn())
    return 1e3 * (time.perf_counter() - t)


res = {}
for name, cfg in CFG.items():
    for B in LG:
        fns = {"off": make(cfg, B, False), "on": make(cfg, B, True)}
        for f in fns.values():
            for _ in range(5):
                mx.eval(f())
        ts = {"off": [], "on": []}
        for r in range(a.reps):
            for v in (("off", "on") if r % 2 == 0 else ("on", "off")):
                ts[v].append(timed(fns[v]))
        m = {v: statistics.median(x) for v, x in ts.items()}
        res[f"{name}/B{B}"] = dict(m, q={v: statistics.quantiles(x, n=4) for v, x in ts.items()})
        print(f"{name:10s} B={B:2d}  off {m['off']:6.2f} ms  on {m['on']:6.2f} ms  "
              f"x{m['off'] / m['on']:.2f}", flush=True)

# reference pieces at B=16: what the old path pays
x = LG[max(LG)].astype(mx.float32)
for name, fn in {"topk20": lambda: mx.topk(x, 20, axis=-1), "argsort": lambda: mx.argsort(x, axis=-1),
                 "argpartition": lambda: mx.argpartition(x, kth=V - 20, axis=-1),
                 "categorical": lambda: mx.random.categorical(x)}.items():
    for _ in range(5):
        mx.eval(fn())
    t = statistics.median(timed(fn) for _ in range(a.reps))
    res[f"piece/{name}/B{max(LG)}"] = t
    print(f"piece {name:12s} B={max(LG)}: {t:6.2f} ms", flush=True)

if a.json:
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    json.dump(res, open(a.json, "w"), indent=1)
    print("wrote", a.json)
