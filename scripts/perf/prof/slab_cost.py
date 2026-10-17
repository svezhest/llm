"""Cost of ONE prefill slab of N tokens at a cache offset (S2b, docs/perf/2026-10-17/slabs.md).

The worker's Engine is loaded exactly as worker.py main() does (Engine.load: fp16 cast, fast
paths, fixed slab, rows attention, pfqmm, mrqmm, GDN fuse); its scheduler thread stays idle
(no request is submitted), the slabs run on this thread on the generation stream. Each trial:
a fresh cache restored like _PrefixCache.restore does (KV [0, off) + a zeros tail of
round_up(N, 256), so update_and_fetch writes in place; random small GDN states), synced; then
the worker's slab body (_Scheduler._prefill: lm(..., skip_logits=True), quantize (no-op for
fp16 KV), mx.eval of the cache states), synced and timed; then mx.clear_cache() (CLEAR_CACHE)
timed apart. N and the offsets are interleaved per round in a rotating order (drift).

  cd src/worker && QWEN_MODEL=/Users/user/Projects/llm/weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 \
    PYTHONPATH=. .venv/bin/python ../../scripts/perf/prof/slab_cost.py --offsets 10240,24576 \
    --ns 5,16,64,128,256,432,512,1024,2048 --rounds 7 --json $PERF_OUT/slabs_cost.json
"""
import argparse, json, os, random, statistics, sys, time
sys.path.insert(0, os.getcwd())
if os.environ.get("SCHED_WORKER_DIR"):          # a frozen copy of src/worker (worker.py, pfqmm.py, ...)
    sys.path.insert(0, os.environ["SCHED_WORKER_DIR"])
import mlx.core as mx
import worker as W

ap = argparse.ArgumentParser()
ap.add_argument("--offsets", default="10240,24576")
ap.add_argument("--ns", default="5,16,64,128,256,432,512,1024,2048")
ap.add_argument("--rounds", type=int, default=7)
ap.add_argument("--warm", type=int, default=1)
ap.add_argument("--json")
a = ap.parse_args()

eng = W.Engine(os.environ["QWEN_MODEL"])
eng.load()
lm = eng.model.language_model
sched = eng.sched
from mlx_vlm.generate.common import generation_stream
from mlx_vlm.models.cache import make_prompt_cache

OFFS = [int(x) for x in a.offsets.split(",")]
NS = [int(x) for x in a.ns.split(",")]
args = lm.args


def base_cache(off, seed):
    """KV pieces [0, off) and GDN states as after a prefill to `off` (random values)."""
    mx.random.seed(seed)
    out = []
    for c in make_prompt_cache(lm):
        if hasattr(c, "cache"):
            conv_dim = args.linear_key_head_dim * args.linear_num_key_heads * 2 + \
                args.linear_value_head_dim * args.linear_num_value_heads
            out.append(("gdn", (0.1 * mx.random.normal((1, args.linear_conv_kernel_dim - 1, conv_dim))).astype(mx.float16),
                        0.01 * mx.random.normal((1, args.linear_num_value_heads, args.linear_value_head_dim,
                                                 args.linear_key_head_dim))))
        else:
            shp = (1, args.num_key_value_heads, off, args.head_dim)
            out.append(("kv", mx.random.normal(shp).astype(mx.float16), mx.random.normal(shp).astype(mx.float16)))
    mx.eval([x for t in out for x in t[1:]])
    return out


def restore(base, off, n):
    """A row's cache resumed at `off` with the first slab of n tokens pre-sized (as restore)."""
    tail = -(-n // 256) * 256
    cache = make_prompt_cache(lm)
    for c, (kind, x, y) in zip(cache, base):
        if kind == "gdn":
            c.cache[0] = mx.array(x)
            c.cache[1] = mx.array(y)
        else:
            z = mx.zeros((1, x.shape[1], tail, x.shape[3]), dtype=x.dtype)
            c.keys = mx.concatenate([x, z], axis=2)
            c.values = mx.concatenate([y, z], axis=2)
            c.offset = off
    mx.eval([t for c in cache for t in (list(c.cache) if hasattr(c, "cache") else [c.keys, c.values])])
    return cache


def one(base, off, n, ids):
    cache = restore(base, off, n)
    inp = mx.array([ids[:n]])
    emb = eng.model.get_input_embeddings(inp, None).inputs_embeds
    mx.eval(emb)
    lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int32)     # as a text prompt-cache hit
    lm._position_ids = None
    mx.synchronize()
    t = time.perf_counter()
    lm(inp, inputs_embeds=emb, cache=cache, skip_logits=True)
    sched.quantize(cache)
    mx.eval([c.state for c in cache])
    mx.synchronize()
    dt = time.perf_counter() - t
    assert cache[lm.model.fa_idx].offset == off + n
    del cache, emb
    t = time.perf_counter()
    mx.clear_cache()
    dc = time.perf_counter() - t
    return dt, dc


random.seed(0)
ids = [random.randrange(1000, 150000) for _ in range(max(NS))]
res = []
with mx.stream(generation_stream):
    bases = {off: base_cache(off, i) for i, off in enumerate(OFFS)}
    for off in OFFS:                       # warm-up / JIT of every shape
        for n in NS:
            for _ in range(a.warm):
                one(bases[off], off, n, ids)
    print(f"warm; peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)
    combos = [(off, n) for n in NS for off in OFFS]
    for r in range(a.rounds):
        k = (r * 5) % len(combos)
        order = combos[k:] + combos[:k]
        if r % 2:
            order = order[::-1]
        for off, n in order:
            dt, dc = one(bases[off], off, n, ids)
            res.append(dict(round=r, off=off, n=n, s=dt, clear_s=dc))
        print(f"round {r} done", flush=True)

print("\n| N | " + " | ".join(f"off {o}: ms (min..max) | ms/token" for o in OFFS) + " |")
print("|---|" + "---|---|" * len(OFFS))
summ = {}
for n in NS:
    cells = []
    for off in OFFS:
        xs = [x["s"] * 1e3 for x in res if x["off"] == off and x["n"] == n]
        m = statistics.median(xs)
        summ[(off, n)] = m
        cells.append(f"{m:.1f} ({min(xs):.1f}..{max(xs):.1f}) | {m / n:.2f}")
    print(f"| {n} | " + " | ".join(cells) + " |")
cl = [x["clear_s"] * 1e3 for x in res]
print(f"\nclear_cache after a slab: median {statistics.median(cl):.1f} ms (max {max(cl):.1f})")
for off in OFFS:
    # marginal cost: least squares over N >= 256; fixed = the fit's intercept
    pts = [(n, summ[(off, n)]) for n in NS if n >= 256]
    mn = sum(p[0] for p in pts) / len(pts); mt = sum(p[1] for p in pts) / len(pts)
    b = sum((p[0] - mn) * (p[1] - mt) for p in pts) / sum((p[0] - mn) ** 2 for p in pts)
    print(f"off {off}: fit N>=256: {mt - b * mn:.1f} ms + {b:.3f} ms/token; N=5 measured {summ.get((off, 5), float('nan')):.1f} ms")
if a.json:
    json.dump(dict(rows=res, offsets=OFFS, ns=NS), open(a.json, "w"), indent=1)
