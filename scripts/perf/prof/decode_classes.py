"""Op-class breakdown of the plain B=1 decode step by removal variants (GPU time).

The worker's model setup (Engine.load: fp16 cast, mrqmm, fused GDN in-projection,
fast paths); a prompt of each --ctx length is prefilled into a fresh cache
(slabs of 2048, the worker's B=1 decode call afterwards: `lm(y, cache,
position_ids)`). Per measurement K decode steps are BUILT lazily (each step's
input = the previous step's argmax, as the worker's step-ahead does) and then
evaluated at once: eval time / K = the GPU time of a step whenever the GPU is
slower than the encoder (the build time is printed apart; a variant whose eval/K
comes near the build/encode cost is host-bound, its number an upper bound).
Variants, alternated per round (order reversed every other round), medians:

  base    the step as the worker runs it
  nolin   every mx.quantized_matmul (linears, fused GDN in-projection, lm_head)
          replaced by a broadcast view of its input (no kernel): base - nolin =
          the matmuls
  noattn  the full-attention SDPA replaced by its queries (+ a dependency on the
          appended K/V: the KV write stays): base - noattn = attention
  norec   the GDN recurrence kernel replaced by v * beta: base - norec = the
          recurrence
  gdnmix  each GDN layer = its fused in-projection + out_proj only (conv, SiLU,
          q/k norms, g/beta, recurrence, gated norm skipped): base - gdnmix =
          what a fused GDN decode kernel (night plan P1) would replace
  bare    nolin + noattn + norec at once: what is left = the small kernels
          (norms, residuals, conv, gates, rope, splits, KV append, embedding,
          argmax)

The caches are put back after every measurement (KV trimmed to the context,
GDN states restored), so every variant runs at the same offset on clean state.

  cd src/worker && QWEN_MODEL=../../weights/qwen38-27b-mlx-4bit QWEN_TEXT_ONLY=1 \
    QWEN_MRQMM4=1 QWEN_KV_BITS=0 QWEN_MAX_BATCH=1 QWEN_WIRED_LIMIT_GB=27 \
    PYTHONPATH=. .venv/bin/python ../../scripts/perf/prof/decode_classes.py \
    --ctx 2048,32768 --K 8 --rounds 5 --json $PERF_OUT/decode_classes.json
NEEDS THE GPU LEASE.
"""
import argparse, collections, json, os, re, statistics, sys, time
sys.path.insert(0, os.getcwd())
import mlx.core as mx
import worker as W

ap = argparse.ArgumentParser()
ap.add_argument("--ctx", default="2048,32768")
ap.add_argument("--K", type=int, default=8, help="steps built per eval")
ap.add_argument("--rounds", type=int, default=5)
ap.add_argument("--variants", default="base,nolin,noattn,norec,gdnmix,bare")
ap.add_argument("--pipe", type=int, default=0,
                help="also N steps of the worker's step-ahead loop (build step k+1, async_eval, then "
                     "step k's .item()) per round: its period vs eval/K = what pipelining loses")
ap.add_argument("--only-pipe", action="store_true", help="skip the removal variants")
ap.add_argument("--json")
a = ap.parse_args()

from mlx_vlm.models.cache import make_prompt_cache, KVCache, ArraysCache  # noqa: E402
from mlx_vlm.models.qwen3_5 import gated_delta as gd, language as q35  # noqa: E402

eng = W.Engine(os.environ.get("QWEN_MODEL"))
eng.load()
lm = eng.model.language_model
tok = getattr(eng.processor, "tokenizer", eng.processor)
print(f"loaded {os.environ.get('QWEN_MODEL')}; mlx {mx.__version__}", flush=True)

# ------------------------------------------------------------------ variants
_QMM = mx.quantized_matmul
_SDPA = q35.scaled_dot_product_attention
_GDK = gd.gated_delta_kernel
_GDN_CALL = q35.Qwen3_5GatedDeltaNet.__call__


def qmm_view(x, w, scales=None, biases=None, transpose=True, group_size=64, bits=4, mode="affine", **kw):
    n = w.shape[0] if transpose else w.shape[1] * 32 // bits
    return mx.broadcast_to(x[..., :1], (*x.shape[:-1], n))


def sdpa_q(queries, keys, values, cache=None, scale=None, mask=None, sinks=None, **kw):
    dep = (keys[:, :, -1:, :] + values[:, :, -1:, :]).sum(axis=1, keepdims=True)
    return queries + 0 * dep


def rec_id(q, k, v, g, beta, state, mask=None):
    return v * beta[..., None], state


def gdn_mix(self, inputs, mask=None, cache=None):
    f = self.__dict__.get("_fused_in")
    if f is not None:
        w, s, b, split, gs, bits, mode, _ = f
        y = mx.quantized_matmul(inputs, w, scales=s, biases=b, transpose=True, group_size=gs,
                                bits=bits, mode=mode)
        z = mx.split(y, split, axis=-1)[1]
    else:
        for n in ("in_proj_qkv", "in_proj_b", "in_proj_a"):
            self[n](inputs)
        z = self.in_proj_z(inputs)
    return self.out_proj(z)


def setv(v):
    names = set(v.split("+")) if v != "bare" else {"nolin", "noattn", "norec"}
    mx.quantized_matmul = qmm_view if "nolin" in names else _QMM
    q35.scaled_dot_product_attention = sdpa_q if "noattn" in names else _SDPA
    gd.gated_delta_kernel = rec_id if "norec" in names else _GDK
    q35.Qwen3_5GatedDeltaNet.__call__ = gdn_mix if "gdnmix" in names else _GDN_CALL


# ------------------------------------------------------------------ prompt
def filler(n_tok):
    words = ("alpha beta gamma delta buffer socket thread mutex queue cache index shard "
             "def class return import yield lambda async await").split()
    text = " ".join(words[(i * 7919) % len(words)] for i in range(int(n_tok * 1.3) + 64))
    ids = tok.encode(text)
    while len(ids) < n_tok:
        ids = ids + ids
    return ids[:n_tok]


cache = make_prompt_cache(lm)
lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int32)
lm._position_ids = None
pos = 0
ctxs = sorted(int(x) for x in a.ctx.split(","))
ids_all = filler(ctxs[-1])


def prefill_to(n):
    global pos
    t = time.time()
    while pos < n:
        m = min(2048, n - pos)
        lm(mx.array([ids_all[pos:pos + m]]), cache=cache)
        mx.eval([c.state for c in cache])
        pos += m
        mx.clear_cache()
    print(f"prefilled to {n} in {time.time() - t:.1f}s", flush=True)


def snapshot():
    return [(c.offset if isinstance(c, KVCache) else list(c.cache)) for c in cache]


def restore(snap):
    for c, s in zip(cache, snap):
        if isinstance(c, KVCache):
            c.trim(c.offset - s)
        else:
            c.cache = list(s)


def count_nodes(y):
    path = os.environ.get("PERF_OUT", "/tmp/perf") + "/_decode_step.dot"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mx.export_to_dot(path, y)
    txt = open(path).read()
    ops = collections.Counter(re.findall(r'label ="?([A-Za-z_]+)', txt))
    return sum(ops.values()), ops.most_common(30)


def measure(v, K, y0):
    setv(v)
    snap = snapshot()
    try:
        mx.synchronize()
        t0 = time.perf_counter()
        y, p = y0, pos
        for _ in range(K):
            out = lm(y[None], cache=cache, position_ids=mx.array([[p]], dtype=mx.int32))
            y = mx.argmax(out.logits[:, -1, :], axis=-1)
            p += 1
        t1 = time.perf_counter()
        mx.eval(y)
        t2 = time.perf_counter()
    finally:
        setv("base")
        restore(snap)
    return (t2 - t1) / K * 1e3, (t1 - t0) / K * 1e3


def pipe(N, y0):
    """The worker's B=1 step-ahead loop without the scheduler: -> (period ms,
    async_eval ms, item wait ms) medians over the steps."""
    snap = snapshot()
    per, ae, wt = [], [], []
    try:
        p = pos

        def step(y, p):
            out = lm(y[None], cache=cache, position_ids=mx.array([[p]], dtype=mx.int32))
            return mx.argmax(out.logits[:, -1, :], axis=-1)
        mx.synchronize()
        cur = step(y0, p)
        mx.async_eval(cur)
        last = time.perf_counter()
        for _ in range(N):
            p += 1
            nxt = step(cur, p)
            t = time.perf_counter()
            mx.async_eval(nxt)
            t1 = time.perf_counter()
            cur.item()
            t2 = time.perf_counter()
            ae.append((t1 - t) * 1e3)
            wt.append((t2 - t1) * 1e3)
            per.append((t2 - last) * 1e3)
            last = t2
            cur = nxt
        cur.item()
    finally:
        restore(snap)
    k = len(per) // 4
    return (statistics.median(per[k:]), statistics.median(ae[k:]), statistics.median(wt[k:]))


res = {}
VARS = a.variants.split(",")
if a.only_pipe:
    VARS = ["base"]
for n in ctxs:
    prefill_to(n)
    y0 = mx.array([ids_all[n - 1]])
    # one step's graph (node classes)
    snap = snapshot()
    out = lm(y0[None], cache=cache, position_ids=mx.array([[pos]], dtype=mx.int32))
    nodes, top = count_nodes(mx.argmax(out.logits[:, -1, :], axis=-1))
    mx.eval(out.logits)
    restore(snap)
    print(f"ctx {n}: one step's graph = {nodes} labelled nodes; top {top[:20]}", flush=True)
    for v in VARS:                          # warm (kernels compiled for every variant)
        measure(v, 2, y0)
    t = {v: [] for v in VARS}
    b = {v: [] for v in VARS}
    pp = []
    for r in range(a.rounds):
        if a.pipe:
            pp.append(pipe(a.pipe, y0))
            print(f"  ctx {n} round {r} pipe    period {pp[-1][0]:7.2f} ms  async_eval {pp[-1][1]:6.2f} "
                  f"item wait {pp[-1][2]:6.2f}", flush=True)
        for v in (VARS if r % 2 == 0 else VARS[::-1]):
            g, h = measure(v, a.K, y0)
            t[v].append(g)
            b[v].append(h)
            print(f"  ctx {n} round {r} {v:7s} eval/K {g:7.2f} ms  build/K {h:5.2f} ms", flush=True)
        mx.clear_cache()
    med = {v: statistics.median(t[v]) for v in VARS}
    bmed = {v: statistics.median(b[v]) for v in VARS}
    d = {}
    if "base" in med:
        for v in VARS:
            if v != "base":
                d[f"base-{v}"] = med["base"] - med[v]
    print(f"ctx {n} medians (ms/step): " + " ".join(f"{v}={med[v]:.2f}" for v in VARS), flush=True)
    print(f"ctx {n} build/K (ms): " + " ".join(f"{v}={bmed[v]:.2f}" for v in VARS), flush=True)
    print(f"ctx {n} deltas (ms): " + " ".join(f"{k}={x:.2f}" for k, x in d.items()), flush=True)
    if all(k in med for k in ("base", "nolin", "noattn", "norec")):
        small = med["nolin"] - (med["base"] - med["noattn"]) - (med["base"] - med["norec"])
        print(f"ctx {n} small kernels (nolin - attn - rec) = {small:.2f} ms; bare = "
              f"{med.get('bare', float('nan')):.2f} ms", flush=True)
    if pp:
        pm = [statistics.median(x[i] for x in pp) for i in range(3)]
        print(f"ctx {n} pipelined step: period {pm[0]:.2f} ms (async_eval {pm[1]:.2f}, item wait {pm[2]:.2f}) "
              f"vs chained base {med['base']:.2f} ms; MLX_MAX_OPS_PER_BUFFER="
              f"{os.environ.get('MLX_MAX_OPS_PER_BUFFER', '')} MLX_MAX_MB_PER_BUFFER="
              f"{os.environ.get('MLX_MAX_MB_PER_BUFFER', '')}", flush=True)
    res[str(n)] = dict(raw=t, build=b, median=med, build_median=bmed, deltas=d, nodes=nodes, top_ops=top,
                       pipe=pp, env={k: v for k, v in os.environ.items() if k.startswith("MLX_")},
                       peak_gib=mx.get_peak_memory() / 2**30)
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
os._exit(0)
