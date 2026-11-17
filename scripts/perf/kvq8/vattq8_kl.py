"""Teacher-forced numerics gate of QWEN_VATT_Q8 (the one-pass attention over an
8-bit QuantizedKVCache, vatt.attention_q8) on the real model, vatt_kl.py's
harness over q8 KV. The context is prefilled as the worker does with
QWEN_KV_BITS=8 (slabs of sizing._slab, quantized once the cache passes
quantized_kv_start, maybe_quantize_kv_cache = the worker's self.quantize).
Then:

  verify blocks  the verify forward (speculative_verify=True, a DFlash round's
                 path) over --blocks blocks of T real tokens, each once per
                 variant on the same cache state (aborted after each), then fed
                 as a plain slab:  lib = the library's quantized one-query calls
                 (_VATT_Q8 off) vs vq8 = the q8 kernel;
  decode steps   --decode one-token forwards (state restored between the two
                 variants): lib = the library's quantized SDPA vs vq8.

A third variant, ref = the exact attention over the same dequantized KV
(vatt.reference_q8: fp32 throughout, rounded to fp16 once), tells which side
of a lib / vq8 disagreement is off: KL(ref || lib) vs KL(ref || vq8).
--modes lib,tile,reg,regf,ref compares the q8 kernels of vatt.py by name in one
run (tile = the threadgroup-tile loader, reg = fragments in registers, regf =
reg with scale and bias out of the sums; vq8 = the one QWEN_VATT_Q8_KERNEL
selects): every kernel against lib and ref, and tile against reg / regf.

Per pair: KL(a || b) mean / p99 / max over every position (fp32 log-softmax
of the logits), top-1 agreement, max |d logprob|, and for every top-1
disagreement the top-1 / top-2 logprob gap of ref there. Gate (night rules):
KL(lib || vq8) mean <= 1e-4, top-1 >= 99.7 %.

  cd src/worker && QWEN_MODEL=../../weights/qwen38-27b-mlx-4bit QWEN_TEXT_ONLY=1 QWEN_MRQMM4=1 \\
    QWEN_KV_BITS=8 QWEN_VATT_Q8=1 QWEN_KVQ_PREFILL_FUSED=1 QWEN_MAX_BATCH=1 QWEN_WIRED_LIMIT_GB=27 \\
    QWEN_CACHE_LIMIT_GB=1 QWEN_SDPA_FUSED_MIN_KV=8192 QWEN_PREFILL_BUDGET=16777216 .venv/bin/python \\
    ../../scripts/perf/kvq8/vattq8_kl.py --tokens 32768 --blocks 64 --decode 128 --json $PERF_OUT/vq8/kl.json
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import worker as W  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "prof"))
from doc_notes import NOTES  # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, default=32768, help="context before the first block")
ap.add_argument("--blocks", type=int, default=64)
ap.add_argument("--T", type=int, default=8)
ap.add_argument("--decode", type=int, default=128)
ap.add_argument("--modes", default="lib,vq8,ref")
ap.add_argument("--json", default="")
a = ap.parse_args()

import mlx.core as mx  # noqa: E402
from mlx_vlm.generate.common import maybe_quantize_kv_cache  # noqa: E402
from mlx_vlm.models.cache import make_prompt_cache  # noqa: E402
from mlx_vlm.speculative.cache_state import abort_speculative_round  # noqa: E402

assert W.KV_KWARGS["kv_bits"] == 8, "QWEN_KV_BITS=8"
eng = W.Engine(os.environ["QWEN_MODEL"])
eng.load()
lm = eng.model.language_model
assert W._VATT_Q8 and W._VATT_Q8_DEC, "QWEN_VATT_Q8=1 self-test did not pass (see the load log)"
START = W.KV_KWARGS["quantized_kv_start"]
tok = getattr(eng.processor, "tokenizer", eng.processor)
text = ""
for f in list(NOTES) + sorted(glob.glob(os.path.join(ROOT, "docs/perf/2026-1*/*.md"))):
    text += open(os.path.join(ROOT, f)).read() + "\n\n"
    if len(text) > 5 * (a.tokens + a.T * a.blocks + a.decode + 64):
        break
ids = tok.encode(text)[:a.tokens + a.T * a.blocks + a.decode]
assert len(ids) == a.tokens + a.T * a.blocks + a.decode, len(ids)
print(f"loaded; {len(ids)} tokens", flush=True)

cache = make_prompt_cache(lm)
inp = mx.array([ids])
lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int32)   # text: mRoPE = the cache offset (as the worker sets it)
lm._position_ids = None


def quantize():
    maybe_quantize_kv_cache(cache, quantized_kv_start=START, kv_group_size=64, kv_bits=8)


t0 = time.time()
pos = 0
while pos < a.tokens:
    n = min(W._slab(pos, a.tokens - pos, set()), a.tokens - pos)
    lm(inp[:, pos:pos + n], cache=cache)
    quantize()
    mx.eval([c.state for c in cache])
    mx.clear_cache()
    pos += n
kinds = {type(c).__name__ for c in cache}
assert "QuantizedKVCache" in kinds, kinds
print(f"prefill {a.tokens} in {time.time() - t0:.0f} s; caches {sorted(kinds)}", flush=True)
lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int32)
FA = [i for i, c in enumerate(cache) if type(c).__name__ == "QuantizedKVCache"]


import vatt  # noqa: E402
KERNEL = vatt.attention_q8


def exact(q, kq, vq, n, scale, group_size=64, bits=8, chunk=None):
    return vatt.reference_q8(q, kq, vq, n, scale, group_size, bits).astype(q.dtype)


def named(kernel):
    def f(q, kq, vq, n, scale, group_size=64, bits=8, chunk=None):
        return KERNEL(q, kq, vq, n, scale, group_size, bits, chunk, kernel=kernel)
    return f


ATT = dict(ref=exact, vq8=KERNEL, lib=KERNEL, tile=named("tile"), reg=named("reg"), regf=named("regf"))


def mode(m):
    """lib / vq8 / tile / reg / regf / ref -> the q8 attention switches (verify and decode alike)."""
    vatt.attention_q8 = ATT[m]
    return m != "lib"


def verify(block, m):
    W._VATT_Q8 = mode(m)
    out = lm(block, cache=cache, speculative_verify=True)
    lp = out.logits[0].astype(mx.float32)
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    mx.eval(lp)
    abort_speculative_round(out.gdn_states)
    mx.eval([c.state for c in cache])
    return lp


MODES = tuple(a.modes.split(","))
assert "lib" in MODES and "ref" in MODES and all(m in ATT for m in MODES), MODES
KERNELS = tuple(m for m in MODES if m not in ("lib", "ref"))
PAIRS = (tuple(("lib", k) for k in KERNELS) + (("ref", "lib"),) + tuple(("ref", k) for k in KERNELS)
         + tuple(("tile", k) for k in KERNELS if k in ("reg", "regf") and "tile" in KERNELS))


def add(s, la, lb, lr):
    p = mx.exp(la)
    kl = (p * (la - lb)).sum(axis=-1)
    s["kl"] += kl.reshape(-1).tolist()
    ta, tb = mx.argmax(la, -1).reshape(-1), mx.argmax(lb, -1).reshape(-1)
    s["agree"] += (ta == tb).sum().item()
    s["n"] += ta.size
    s["dmax"] = max(s["dmax"], mx.abs(la - lb).max().item())
    top2 = mx.sort(lr.reshape(-1, lr.shape[-1]), axis=-1)[:, -2:]
    gap = (top2[:, 1] - top2[:, 0]).tolist()
    s["flip_gaps"] += [round(gap[i], 4) for i, same in enumerate((ta == tb).tolist()) if not same]


acc = {f"{x}|{y} {k}": dict(kl=[], agree=0, n=0, dmax=0.0, flip_gaps=[])
       for k in ("verify", "decode") for x, y in PAIRS}


def add_all(kind, lps):
    for x, y in PAIRS:
        add(acc[f"{x}|{y} {kind}"], lps[x], lps[y], lps["ref"])


W._VATT_Q8_DEC = False
t0 = time.time()
for b in range(a.blocks):
    assert cache[FA[0]].offset == pos, (cache[FA[0]].offset, pos)
    block = inp[:, pos:pos + a.T]
    r = b % len(MODES)
    lps = {m: verify(block, m) for m in MODES[r:] + MODES[:r]}
    add_all("verify", lps)
    W._VATT_Q8 = mode("lib")
    lm(block, cache=cache)                               # move on: the block as a plain slab
    quantize()
    mx.eval([c.state for c in cache])
    pos += a.T
    if b % 16 == 15:
        print(f"  {b + 1} blocks, {time.time() - t0:.0f} s", flush=True)


def step(x, m):
    W._VATT_Q8_DEC = mode(m)
    snap = [c.offset if type(c).__name__ in ("KVCache", "QuantizedKVCache") else list(c.state) for c in cache]
    out = lm(x, cache=cache)
    lp = out.logits[0, -1].astype(mx.float32)
    lp = lp - mx.logsumexp(lp)
    mx.eval(lp)
    for c, sn in zip(cache, snap):
        if type(c).__name__ in ("KVCache", "QuantizedKVCache"):
            c.offset = sn
        else:
            c.state = sn
    return lp


for i in range(a.decode):
    x = inp[:, pos:pos + 1]
    r = i % len(MODES)
    lps = {m: step(x, m)[None] for m in MODES[r:] + MODES[:r]}
    add_all("decode", lps)
    W._VATT_Q8_DEC = mode("lib")
    lm(x, cache=cache)
    quantize()
    mx.eval([c.state for c in cache])
    pos += 1

res = dict(tokens=a.tokens, blocks=a.blocks, T=a.T, decode=a.decode, modes=MODES,
           model=os.path.basename(os.environ["QWEN_MODEL"].rstrip("/")),
           kvq_prefill_fused=W.KVQ_PREFILL_FUSED, pairs={})
ok = True
for k, s in acc.items():
    kl = sorted(s["kl"])
    n = len(kl)
    if not n:
        continue
    r = dict(n=n, kl_mean=sum(kl) / n, kl_p99=kl[min(n - 1, int(0.99 * n))], kl_max=kl[-1],
             top1=100.0 * s["agree"] / s["n"], max_abs_dlogprob=s["dmax"], ref_gap_at_flips=s["flip_gaps"])
    r["gate"] = r["kl_mean"] <= 1e-4 and r["top1"] >= 99.7
    if k.startswith("lib|"):
        ok &= r["gate"]
    res["pairs"][k] = r
    print(f"{k:17s}: {n} positions  KL mean {r['kl_mean']:.3g}  p99 {r['kl_p99']:.3g}  "
          f"max {r['kl_max']:.3g}  top-1 {r['top1']:.2f}%  max|d logprob| {r['max_abs_dlogprob']:.3g}  "
          f"ref top-2 gap at flips {s['flip_gaps']}" + (f"  {'PASS' if r['gate'] else 'FAIL'}"
                                                        if k.startswith("lib|") else ""), flush=True)
res["gate"] = ok
print(f"gate (KL(lib||kernel) <= 1e-4, top-1 >= 99.7 %, verify and decode): {'PASS' if ok else 'FAIL'}",
      flush=True)
if a.json:
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    json.dump(res, open(a.json, "w"), indent=1)
sys.stdout.flush()
os._exit(0)
