"""In-process A/B of one prefill slab (9B), pfqmm's mixed last tile
(2026-10-17) vs the previous pfqmm (64-row tiles only, taken when the 64-row
padding wastes <= 6% over 32-row tiles and M >= 128) vs quantized_matmul.
Variants alternate per round (order reversed every other round, round 0 =
warm-up, not timed); medians. Before every call the model's rope state is
reset as the worker does on a cache hit (`_rope_deltas` = 0, `_position_ids`
= None): left alone, the process's first call (`_rope_deltas` None) rotates
the slab from position 0 whatever the cache offset, and only later calls
continue from it. Every call of every round (round 0 included) is compared
bitwise to the first variant's round-0 call: the post-slab cache state and the
model's final hidden state (`return_hidden`; the last layer's MLP and the
final norm feed no cache).

  cd src/worker && QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 \
    .venv/bin/python ../../scripts/perf/prof/ab_pfqmm_mix.py --offsets 12288 --slabs 416,432,2048 --rounds 6
"""
import argparse, json, os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "microbench"))
import mlx.core as mx
import pf_common as C
import worker as W
import microbench_pfqmm_m as MB

ap = argparse.ArgumentParser()
ap.add_argument("--variants", default="mlx,old,new")
ap.add_argument("--offsets", default="12288")
ap.add_argument("--slabs", default="416,432,2048")
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--json")
a = ap.parse_args()

model, proc = C.load()
lm = model.language_model
import pfqmm
print("pfqmm modules:", pfqmm.install(lm), flush=True)
NEW_USE, NEW_KERNEL = pfqmm._use, pfqmm.pfqmm


def old_use(m, x):
    if x.dtype != mx.float16 or x.ndim < 2:
        return False
    K = x.shape[-1]
    M = x.size // K
    if M != x.shape[-2]:
        return False
    return (M >= 128 and -(-M // 64) * 64 <= 1.06 * (-(-M // 32) * 32)
            and pfqmm._mlx_uses_qmm(M, K, m["weight"].shape[0], m.group_size))


def old_kernel(x, w, s, b, bits, gs):          # the previous kernel: 64x32x32, no mixed tile
    return MB.run("pf64", x, w, s, b, bits, gs)


def setv(v):
    pfqmm.ACTIVE = v != "mlx"
    pfqmm._use, pfqmm.pfqmm = (old_use, old_kernel) if v == "old" else (NEW_USE, NEW_KERNEL)


out = []
random.seed(0)
for off in [int(x) for x in a.offsets.split(",")]:
    base = C.synth_cache(lm, off)
    for slab in [int(x) for x in a.slabs.split(",")]:
        ids = mx.array([[random.randrange(1000, 150000) for _ in range(slab)]])
        emb = lm.model.embed_tokens(ids)
        mx.eval(emb)
        names = a.variants.split(",")
        times = {n: [] for n in names}
        ref = None
        same = {n: [] for n in names}               # per call: cache state and hidden == ref
        for r in range(a.rounds + 1):
            for n in (names if r % 2 == 0 else names[::-1]):
                setv(n)
                cache = C.fresh(base)
                lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int32)
                lm._position_ids = None
                mx.synchronize()
                t = time.perf_counter()
                o = lm(ids, inputs_embeds=emb, cache=cache, skip_logits=True, return_hidden=True)
                h = o.hidden_states[-1]
                mx.eval([c.state for c in cache], h)
                dt = time.perf_counter() - t
                if r:
                    times[n].append(dt)
                got = W._cache_arrays(cache) + [h]
                if ref is None:
                    ref = got
                same[n].append(all(mx.array_equal(x, y).item() for x, y in zip(ref, got)))
                del cache, o, h, got
                mx.clear_cache()
        t0 = C.med(times[names[0]])
        print(f"offset {off} slab {slab} (M % 64 = {slab % 64}):", flush=True)
        for n in names:
            eq = all(same[n])
            tn = C.med(times[n])
            told = C.med(times["old"]) if "old" in times else tn
            print(f"  {n:4s} {tn * 1e3:8.1f} ms  x{t0 / tn:.3f} vs {names[0]}  x{told / tn:.3f} vs old  "
                  f"{slab / tn:6.0f} tok/s  {'bit-identical' if eq else 'DIFFERS'} ({sum(same[n])}/{len(same[n])} calls)  runs "
                  f"{' '.join(f'{x * 1e3:.0f}' for x in times[n])}", flush=True)
            out.append(dict(offset=off, slab=slab, variant=n, ms=tn * 1e3, runs=[x * 1e3 for x in times[n]],
                            bit_identical=eq, calls_equal=sum(same[n]), calls=len(same[n])))
        del ref
        mx.clear_cache()
    setv("new")
    del base
    mx.clear_cache()
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
