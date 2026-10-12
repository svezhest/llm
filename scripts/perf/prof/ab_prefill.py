"""Full-prompt prefill A/B (in-process, interleaved): the worker's slab loop
(_bounds/_slab cuts, lm(..., skip_logits=True), quantize, eval state,
clear_cache) over N random ids, per variant. Reports wall, tok/s, peak
memory, slab count, and whether the final cache equals the first variant's.

  cd src/worker && QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 \
    .venv/bin/python $SP/prof/ab_prefill.py --lens 8192,32768 --variants old,new --rounds 2
Variants: old = slab shrunk by PREFILL_BUDGET + quantized_matmul, KV grown per
slab; new = the current worker code (fixed fp16 slab, pfqmm, KV reserved);
pf = old slabs + pfqmm; nores = new without the KV reservation."""
import argparse, functools, os, sys, time, random, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
import pf_common as C
import worker as W

ap = argparse.ArgumentParser()
ap.add_argument("--lens", default="8192")
ap.add_argument("--variants", default="old,new")
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--json")
ap.add_argument("--keep", action="store_true", help="compare final caches with the first variant")
ap.add_argument("--hash", action="store_true", help="sha1 of the final cache (after the peak is read)")
a = ap.parse_args()

model, proc = C.load()
lm = model.language_model
import pfqmm
pfqmm.install(lm)
from mlx_vlm.generate.common import maybe_quantize_kv_cache, generation_stream
quantize = functools.partial(maybe_quantize_kv_cache, **{"quantized_kv_start": W.KV_KWARGS["quantized_kv_start"],
                             "kv_group_size": 64, "kv_bits": W.KV_KWARGS["kv_bits"]})
NEW_STEP = W._step


def OLD_STEP(pos):
    return max(W.PREFILL_MIN_STEP, min(W.PREFILL_STEP, W.PREFILL_BUDGET // max(pos, 1)))


RESERVE = [True]


def setv(v):
    W._step = OLD_STEP if v in ("old", "pf") else NEW_STEP
    pfqmm.ACTIVE = v in ("new", "pf", "nores")
    RESERVE[0] = v == "new"


def prefill(ids):
    from mlx_vlm.models.cache import make_prompt_cache
    cache = make_prompt_cache(lm)
    cuts = set(W._bounds(ids, -1))
    inp = mx.array([ids])
    emb = model.get_input_embeddings(inp, None).inputs_embeds
    total = len(ids) if RESERVE[0] else 0
    pos, slabs = 0, 0
    while emb.shape[1] > 1:
        n = W._slab(pos, emb.shape[1] - 1, cuts)
        if total and hasattr(W, "_reserve_kv"):
            total = W._reserve_kv(cache, total, pos + n, lm, emb.dtype, W.KV_KWARGS["kv_bits"])
        lm(inp[:, :n], inputs_embeds=emb[:, :n], cache=cache, skip_logits=True)
        quantize(cache)
        mx.eval([c.state for c in cache])
        pos += n
        slabs += 1
        inp, emb = inp[:, n:], emb[:, n:]
        mx.clear_cache()
    return cache, slabs


random.seed(0)
res = []
for L in [int(x) for x in a.lens.split(",")]:
    ids = [random.randrange(1000, 150000) for _ in range(L)]
    names = a.variants.split(",")
    with mx.stream(generation_stream):
        setv(names[-1]); c, _ = prefill(ids[:3000]); del c        # warm-up / JIT
    mx.clear_cache()
    ref = None
    for r in range(a.rounds):
        for v in (names if r % 2 == 0 else names[::-1]):
            setv(v)
            mx.clear_cache(); mx.reset_peak_memory()
            base = mx.get_active_memory()
            t = time.perf_counter()
            with mx.stream(generation_stream):
                cache, slabs = prefill(ids)
            mx.synchronize()
            dt = time.perf_counter() - t
            peak = mx.get_peak_memory()
            eq = ""
            if a.hash:
                import hashlib, numpy as np
                h = hashlib.sha1()
                for c in cache:
                    for x in (list(c.cache) if hasattr(c, "cache") else [c.keys[..., :c.offset, :], c.values[..., :c.offset, :]]):
                        h.update(np.ascontiguousarray(np.array(x)).tobytes())
                eq = f"  sha1 {h.hexdigest()[:12]}"
            if a.keep:
                arrs = [x for c in cache for x in (list(c.cache) if hasattr(c, 'cache') else [c.keys[..., :c.offset, :], c.values[..., :c.offset, :]])]
                if ref is None:
                    ref = (v, arrs)
                else:
                    same = all(mx.array_equal(x, y).item() for x, y in zip(ref[1], arrs))
                    md = 0.0 if same else max(mx.abs(x.astype(mx.float32) - y.astype(mx.float32)).max().item()
                                               for x, y in zip(ref[1], arrs))
                    bad = [i for i, (x, y) in enumerate(zip(ref[1], arrs)) if not mx.array_equal(x, y).item()]
                    eq = f"  vs {ref[0]}: {'bit-identical' if same else f'DIFF max {md:.3g} arrays {bad[:12]} of {len(arrs)}'}"
                del arrs
            print(f"len {L:6d} round {r} {v:5s}: {dt:7.1f} s  {L/dt:6.1f} tok/s  peak {peak/1e9:5.2f} GB "
                  f"(+{(peak-base)/1e9:.2f} over weights)  slabs {slabs}{eq}", flush=True)
            res.append(dict(len=L, round=r, variant=v, s=dt, peak_gb=peak / 1e9, slabs=slabs))
            del cache
            mx.clear_cache()
    del ref
    mx.clear_cache()
setv("new")
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
