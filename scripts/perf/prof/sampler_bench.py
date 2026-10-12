"""GPU cost of the worker's per-row samplers on (B, V) logits, isolated:
greedy / penalties / card sampler (temp 0.6, top_p 0.95, top_k 20) / top_p only,
per row (the worker today) and pieces of it. Each config: 30 reps of
build + eval, median ms per step for all B rows.
  cd src/worker; PYTHONPATH=. .venv/bin/python $SP/prof/sampler_bench.py"""
import os, statistics, sys, time
sys.path.insert(0, os.getcwd())
import mlx.core as mx
import worker as W

V = 248320
B = int(os.environ.get("SB_B", "16"))
mx.random.seed(0)
lg = (mx.random.normal((B, V)) * 2.5).astype(mx.float16)
mx.eval(lg)
CFG = {
    "greedy": dict(W.SAMPLING_KWARGS, temperature=0.0, presence_penalty=0.0),
    "card": dict(W.SAMPLING_KWARGS, temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.0),
    "topp": dict(W.SAMPLING_KWARGS, temperature=0.7, top_p=0.9, top_k=0, presence_penalty=0.0),
    "topk": dict(W.SAMPLING_KWARGS, temperature=0.7, top_p=1.0, top_k=20, presence_penalty=0.0),
    "temp": dict(W.SAMPLING_KWARGS, temperature=0.7, top_p=1.0, top_k=0, presence_penalty=0.0),
}


def bench(fn, reps=30):
    ts = []
    for _ in range(5):
        mx.eval(fn())
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t)
    return 1e3 * statistics.median(ts)


def rows_fn(name, pen=False):
    samplers = [W.make_sampler(CFG[name], seed=i) for i in range(B)]
    pens = [W._Penalties(dict(CFG[name], presence_penalty=1.5), [1, 2, 3], V) for _ in range(B)] if pen else None
    toks = mx.array([5], dtype=mx.uint32)

    def f():
        out = []
        for i in range(B):
            x = lg[i:i + 1]
            if pen:
                pens[i].add(toks)
                x = pens[i](x)
            out.append(samplers[i](x))
        return mx.concatenate(out)
    return f


def group_fn(name, pen=False):
    samplers = [W.make_sampler(CFG[name], seed=i) for i in range(B)]
    pens = [W._Penalties(dict(CFG[name], presence_penalty=1.5), [1, 2, 3], V) for _ in range(B)] if pen else None
    toks = mx.array([5], dtype=mx.uint32)

    def f():
        xs = []
        for i in range(B):
            x = lg[i:i + 1]
            if pen:
                pens[i].add(toks)
                x = pens[i](x)
            xs.append(x)
        return mx.concatenate(W.sample_rows(samplers, xs))
    return f


res = {}
with mx.stream(mx.gpu):
    res["empty eval"] = bench(lambda: lg[:, :1] + 0)
    for n in ("greedy", "temp", "topk", "topp", "card"):
        res[f"{n} rows"] = bench(rows_fn(n))
    for n in ("greedy", "temp", "topk", "topp", "card"):
        res[f"{n} sample_rows"] = bench(group_fn(n))
    res["card+pen sample_rows"] = bench(group_fn("card", pen=True))
    res["greedy+pen rows"] = bench(rows_fn("greedy", pen=True))
    res["card+pen rows"] = bench(rows_fn("card", pen=True))
    x = lg.astype(mx.float32)
    res["argsort (B,V) one op"] = bench(lambda: mx.argsort(x, axis=-1))
    res["argsort per row"] = bench(lambda: [mx.argsort(x[i:i + 1], axis=-1) for i in range(B)])
    res["topk20 per row"] = bench(lambda: [mx.topk(x[i:i + 1], 20, axis=-1) for i in range(B)])
    res["topk20 (B,V)"] = bench(lambda: mx.topk(x, 20, axis=-1))
    res["categorical per row"] = bench(lambda: [mx.random.categorical(x[i:i + 1], key=mx.random.key(i)) for i in range(B)])
for k, v in res.items():
    print(f"B={B} {k:24s} {v:7.2f} ms")
