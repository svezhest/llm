"""In-process A/B of one prefill slab at given cache offsets: variants are
alternated A/B/A/B (rounds), medians compared; the post-slab cache state of
each variant is compared bitwise to the first variant's (exactness).

  cd src/worker && QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 \
    .venv/bin/python $SP/prof/ab_slab.py --variants base,pf --offsets 0,8192 --slab 2048 --rounds 4
Variants are defined in VARIANTS (setup functions toggling worker / pfqmm globals)."""
import argparse, os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
import pf_common as C
import worker as W

ap = argparse.ArgumentParser()
ap.add_argument("--variants", default="base,pf")
ap.add_argument("--offsets", default="0,8192")
ap.add_argument("--slab", type=int, default=2048)
ap.add_argument("--rounds", type=int, default=4)
ap.add_argument("--pfqmm", type=int, default=1)
a = ap.parse_args()

model, proc = C.load()
lm = model.language_model
import pfqmm
if a.pfqmm:
    pfqmm.install(lm)
import variants_pf as V          # setup functions, scratchpad/prof/variants_pf.py
VARIANTS = V.make(W, pfqmm, lm)

random.seed(0)
for off in [int(x) for x in a.offsets.split(",")]:
    base = C.synth_cache(lm, off)
    ids = mx.array([[random.randrange(1000, 150000) for _ in range(a.slab)]])
    emb = lm.model.embed_tokens(ids)
    mx.eval(emb)
    names = a.variants.split(",")
    times = {n: [] for n in names}
    states = {}
    for r in range(a.rounds + 1):
        for n in (names if r % 2 == 0 else names[::-1]):
            VARIANTS[n]()
            cache = C.fresh(base)
            mx.synchronize()
            t = time.perf_counter()
            lm(ids, inputs_embeds=emb, cache=cache, skip_logits=True)
            mx.eval([c.state for c in cache])
            dt = time.perf_counter() - t
            if r:                                  # round 0 = warm-up / JIT
                times[n].append(dt)
            if n not in states:
                states[n] = W._cache_arrays(cache)
            del cache
            mx.clear_cache()
    ref = states[names[0]]
    t0 = C.med(times[names[0]])
    print(f"offset {off} slab {a.slab}:", flush=True)
    for n in names:
        eq = all(mx.array_equal(x, y).item() for x, y in zip(ref, states[n]))
        md = max(mx.abs(x.astype(mx.float32) - y.astype(mx.float32)).max().item() for x, y in zip(ref, states[n]))
        tn = C.med(times[n])
        print(f"  {n:12s} {tn*1e3:8.1f} ms  x{t0/tn:.3f} vs {names[0]}  {a.slab/tn:6.0f} tok/s  "
              f"{'bit-identical' if eq else f'DIFF max {md:.3g}'}  runs {' '.join(f'{x*1e3:.0f}' for x in times[n])}",
              flush=True)
    VARIANTS[names[0]]()
    del base, states
    mx.clear_cache()
