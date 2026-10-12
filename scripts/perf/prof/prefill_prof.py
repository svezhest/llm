"""Op-class profile of one prefill slab (in-process, the worker's model setup).

Every op of interest is wrapped: its array inputs are evaluated first (that
time goes to the 'other' bucket = everything not wrapped), then the op runs
and its outputs are evaluated (time -> the op's class). So the slab's wall
time is partitioned completely; each wrapped op adds one GPU sync (~0.1 ms).
Reports per class: ms, % of slab, FLOPs or bytes, achieved rate, roofline time.

  cd src/worker && QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 \
    .venv/bin/python $SP/prof/prefill_prof.py --offsets 0,8192,32768,122880 --slab 2048 --reps 3
"""
import argparse, collections, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
import mlx.nn as nn
import pf_common as C
import worker as W

ap = argparse.ArgumentParser()
ap.add_argument("--offsets", default="0,8192,32768,122880")
ap.add_argument("--slab", default="2048")          # comma list; 'adapt' = worker _step(offset)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--json")
ap.add_argument("--pfqmm", type=int, default=0)
ap.add_argument("--fused", default="cur", help="cur (worker rule) | on (always force_fused for causal q>=512) | off")
a = ap.parse_args()

PEAK_FLOPS = 10.4e12
BW = 400e9

model, proc = C.load()
lm = model.language_model
if a.pfqmm:
    import pfqmm
    pfqmm.install(lm)
from mlx_vlm.models.qwen3_5 import language as q35, gated_delta as gd
from mlx_vlm.models import rope_utils, cache as vcache

P = {"on": False, "mark": 0.0, "depth": 0, "acc": collections.defaultdict(float),
     "flops": collections.defaultdict(float), "bytes": collections.defaultdict(float),
     "n": collections.defaultdict(int)}


def _arrs(x, out):
    if isinstance(x, mx.array):
        out.append(x)
    elif isinstance(x, (list, tuple)):
        for y in x:
            _arrs(y, out)
    elif isinstance(x, dict):
        for y in x.values():
            _arrs(y, out)
    return out


def wrap(fn, cls, flops=None, self_arg=False):
    def w(*args, **kw):
        if not P["on"] or P["depth"]:
            return fn(*args, **kw)
        ins = _arrs(list(args[1:] if self_arg else args), [])
        ins += _arrs(kw, [])
        mx.eval(ins)
        t0 = time.perf_counter()
        P["acc"]["other"] += t0 - P["mark"]
        P["depth"] += 1
        try:
            out = fn(*args, **kw)
        finally:
            P["depth"] -= 1
        outs = _arrs(out, [])
        mx.eval(outs)
        t1 = time.perf_counter()
        k = cls(*args, **kw) if callable(cls) else cls
        P["acc"][k] += t1 - t0
        P["n"][k] += 1
        if flops is not None:
            P["flops"][k] += flops(*args, **kw)
        P["bytes"][k] += sum(x.nbytes for x in ins) + sum(x.nbytes for x in outs)
        P["mark"] = t1
        return out
    return w


# --- names of linears
names = {}
for n, m in lm.named_modules():
    if isinstance(m, nn.QuantizedLinear):
        leaf = n.split(".")[-1]
        names[id(m)] = {"q_proj": "gemm attn q", "k_proj": "gemm attn k/v", "v_proj": "gemm attn k/v",
                        "o_proj": "gemm attn o", "gate_proj": "gemm mlp gate/up", "up_proj": "gemm mlp gate/up",
                        "down_proj": "gemm mlp down", "in_proj_qkv": "gemm gdn in (qkv,z)",
                        "in_proj_z": "gemm gdn in (qkv,z)", "in_proj_b": "gemm gdn in b/a (N=32)",
                        "in_proj_a": "gemm gdn in b/a (N=32)", "out_proj": "gemm gdn out"}.get(leaf, "gemm " + leaf)


def lin_cls(self, x):
    return names.get(id(self), "gemm ?")


def lin_flops(self, x):
    K = x.shape[-1]
    M = x.size // K
    N = self.weight.shape[0]
    return 2.0 * M * N * K


nn.QuantizedLinear.__call__ = wrap(nn.QuantizedLinear.__call__, lin_cls, lin_flops, self_arg=True)


def sdpa_flops(q, k, v, cache=None, scale=None, mask=None, sinks=None):
    B, H, L, D = q.shape
    S = k.shape[2]
    return 4.0 * B * H * L * (S - L / 2) * D


def sdpa_cls(q, k, v, cache=None, scale=None, mask=None, sinks=None):
    fused = (W.SDPA_FUSED and q.shape[2] >= W.SDPA_FUSED_MIN_Q and k.shape[2] >= W.SDPA_FUSED_MIN_KV)
    return "attention sdpa (fused)" if fused else "attention sdpa (default)"


q35.scaled_dot_product_attention = wrap(q35.scaled_dot_product_attention, sdpa_cls, sdpa_flops)
q35.gated_delta_update = wrap(q35.gated_delta_update, "gdn recurrence (+g/beta)")
q35.swiglu = wrap(q35.swiglu, "mlp swiglu")
nn.RMSNorm.__call__ = wrap(nn.RMSNorm.__call__, "rmsnorm", self_arg=True)
q35.Qwen3_5RMSNormGated.__call__ = wrap(q35.Qwen3_5RMSNormGated.__call__, "gdn gated norm", self_arg=True)
nn.Conv1d.__call__ = wrap(nn.Conv1d.__call__, "gdn conv1d+", self_arg=True)
rope_utils.MRoPERotaryEmbedding.apply_rotary = wrap(rope_utils.MRoPERotaryEmbedding.apply_rotary, "rope", self_arg=True)
vcache.KVCache.update_and_fetch = wrap(vcache.KVCache.update_and_fetch, "kv cache update", self_arg=True)

if a.fused == "on":
    W.SDPA_FUSED_MIN_KV = 0
elif a.fused == "off":
    W.SDPA_FUSED = False

import random
random.seed(0)
results = []
for off in [int(x) for x in a.offsets.split(",")]:
    base = C.synth_cache(lm, off)
    for sl in a.slab.split(","):
        n = W._step(off) if sl == "adapt" else int(sl)
        ids = mx.array([[random.randrange(1000, 150000) for _ in range(n)]])
        emb = lm.model.embed_tokens(ids)
        mx.eval(emb)
        # warm-up (JIT)
        C.slab(lm, ids, emb, C.fresh(base))
        mx.clear_cache()
        plain, inst, host = [], [], []
        accs = []
        for r in range(a.reps):
            # un-instrumented slab (the worker's sequence: build, eval, clear_cache)
            cache = C.fresh(base)
            mx.synchronize()
            t0 = time.perf_counter()
            lm(ids, inputs_embeds=emb, cache=cache, skip_logits=True)
            t1 = time.perf_counter()
            mx.eval([c.state for c in cache])
            t2 = time.perf_counter()
            mx.clear_cache()
            t3 = time.perf_counter()
            plain.append(t2 - t0)
            host.append((t1 - t0, t3 - t2))
            del cache
            # instrumented
            cache = C.fresh(base)
            for d in ("acc", "flops", "bytes", "n"):
                P[d].clear()
            mx.synchronize()
            P["on"] = True
            t0 = P["mark"] = time.perf_counter()
            lm(ids, inputs_embeds=emb, cache=cache, skip_logits=True)
            mx.eval([c.state for c in cache])
            t1 = time.perf_counter()
            P["acc"]["other"] += t1 - P["mark"]
            P["on"] = False
            inst.append(t1 - t0)
            accs.append({k: (P["acc"][k], P["flops"][k], P["bytes"][k], P["n"][k]) for k in P["acc"]})
            del cache
            mx.clear_cache()
        keys = sorted({k for d in accs for k in d}, key=lambda k: -C.med([d.get(k, (0,))[0] for d in accs]))
        tp = C.med(plain)
        ti = C.med(inst)
        print(f"\n=== offset {off}  slab {n}: plain {tp*1e3:.0f} ms ({n/tp:.0f} tok/s), instrumented "
              f"{ti*1e3:.0f} ms; host build {C.med([h[0] for h in host])*1e3:.1f} ms, clear_cache "
              f"{C.med([h[1] for h in host])*1e3:.1f} ms", flush=True)
        rows = []
        for k in keys:
            ms = C.med([d.get(k, (0, 0, 0, 0))[0] for d in accs]) * 1e3
            fl = accs[0].get(k, (0, 0, 0, 0))[1]
            by = accs[0].get(k, (0, 0, 0, 0))[2]
            cnt = accs[0].get(k, (0, 0, 0, 0))[3]
            rate = roof = ""
            if fl:
                rate = f"{fl / (ms / 1e3) / 1e12:.2f} TFLOPS"
                roof = f"{fl / PEAK_FLOPS * 1e3:.0f}"
            elif by:
                rate = f"{by / (ms / 1e3) / 1e9:.0f} GB/s"
                roof = f"{by / BW * 1e3:.1f}"
            print(f"  {k:28s} {ms:8.1f} ms {100 * ms / (ti * 1e3):5.1f}%  n={cnt:4d}  "
                  f"{fl / 1e12:7.2f} TF  {by / 1e9:7.2f} GB  {rate:>14s}  roof {roof:>6s} ms", flush=True)
            rows.append(dict(cls=k, ms=ms, flops=fl, bytes=by, n=cnt))
        results.append(dict(offset=off, slab=n, plain_ms=tp * 1e3, inst_ms=ti * 1e3,
                            build_ms=C.med([h[0] for h in host]) * 1e3,
                            clear_ms=C.med([h[1] for h in host]) * 1e3, rows=rows))
    del base
    mx.clear_cache()
if a.json:
    json.dump(results, open(a.json, "w"), indent=1)
