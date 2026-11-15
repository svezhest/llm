"""Teacher-forced numerics gate of QWEN_VERIFY_ATT=2 (the one-pass verify-block
attention, vatt.py) on the real model: the verify forward (speculative_verify=True,
the path a DFlash round takes) over blocks of T real tokens at a long context,
each block once per variant on the same cache state (the round aborted after
each), then fed as a plain slab to move on. Variants:

  lib   the library's one-query calls (QWEN_VERIFY_ATT=0)
  vec   MLX's vector kernel, 32 // G queries per call (QWEN_VERIFY_ATT=1: lib's bits)
  vatt  the one-pass kernel (QWEN_VERIFY_ATT=2)

Per pair: KL(a || b) over every block position (float32 log-softmax of the
logits), mean / p99 / max, top-1 agreement, max |d logit|. The gate
(night rules): KL(vec || vatt) mean <= 1e-4, top-1 >= 99.7 %.

  cd src/worker && QWEN_MODEL=../../weights/qwen38-27b-mlx-4bit QWEN_TEXT_ONLY=1 QWEN_MRQMM4=1 \\
    QWEN_KV_BITS=0 QWEN_VERIFY_ATT=2 QWEN_MAX_BATCH=1 QWEN_WIRED_LIMIT_GB=27 QWEN_CACHE_LIMIT_GB=1 \\
    QWEN_SDPA_FUSED_MIN_KV=8192 QWEN_PREFILL_BUDGET=16777216 .venv/bin/python \\
    ../../scripts/perf/dflash2/vatt_kl.py --tokens 32768 --blocks 64 --json $PERF_OUT/vatt/kl.json
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
ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, default=32768, help="context before the first block")
ap.add_argument("--blocks", type=int, default=64)
ap.add_argument("--T", type=int, default=8)
ap.add_argument("--json", default="")
a = ap.parse_args()

import mlx.core as mx  # noqa: E402
from mlx_vlm.models.cache import make_prompt_cache  # noqa: E402
from mlx_vlm.speculative.cache_state import abort_speculative_round  # noqa: E402

eng = W.Engine(os.environ["QWEN_MODEL"])
eng.load()
lm = eng.model.language_model
W._probe_verify_attention(lm)        # no drafter loaded: the load does not run it
assert W._VATT, "QWEN_VERIFY_ATT=2 self-test did not pass"
NQ = W._VERIFY_ATT_Q
tok = getattr(eng.processor, "tokenizer", eng.processor)
text = ""
for f in ["README.md", "CLAUDE.md", "BACKLOG.md"] + sorted(glob.glob(os.path.join(ROOT, "docs/perf/2026-1*/*.md"))):
    text += open(os.path.join(ROOT, f)).read() + "\n\n"
    if len(text) > 5 * (a.tokens + a.T * a.blocks + 64):
        break
ids = tok.encode(text)[:a.tokens + a.T * a.blocks]
assert len(ids) == a.tokens + a.T * a.blocks, len(ids)
print(f"loaded; {len(ids)} tokens; vector kernel {NQ} queries per call", flush=True)

cache = make_prompt_cache(lm)
inp = mx.array([ids])
t0 = time.time()
pos = 0
while pos < a.tokens:
    n = min(W._slab(pos, a.tokens - pos, set()), a.tokens - pos)
    lm(inp[:, pos:pos + n], cache=cache)
    mx.eval([c.state for c in cache])
    mx.clear_cache()
    pos += n
print(f"prefill {a.tokens} in {time.time() - t0:.0f} s", flush=True)
lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int32)


def verify(block, mode):
    W._VATT = mode == "vatt"
    W._VERIFY_ATT_Q = NQ if mode == "vec" else 0
    out = lm(block, cache=cache, speculative_verify=True)
    lp = out.logits[0].astype(mx.float32)
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    mx.eval(lp)
    abort_speculative_round(out.gdn_states)
    mx.eval([c.state for c in cache])
    return lp


MODES = ("lib", "vec", "vatt")
PAIRS = (("lib", "vec"), ("vec", "vatt"), ("lib", "vatt"))
acc = {p: dict(kl=[], agree=0, n=0, dmax=0.0) for p in PAIRS}
t0 = time.time()
for b in range(a.blocks):
    off = cache[[i for i, c in enumerate(cache) if type(c).__name__ == "KVCache"][0]].offset
    assert off == pos, (off, pos)
    block = inp[:, pos:pos + a.T]
    lps = {m: verify(block, m) for m in (MODES if b % 2 else MODES[::-1])}
    for x, y in PAIRS:
        p = mx.exp(lps[x])
        kl = (p * (lps[x] - lps[y])).sum(axis=-1)
        s = acc[(x, y)]
        s["kl"] += kl.tolist()
        s["agree"] += (mx.argmax(lps[x], -1) == mx.argmax(lps[y], -1)).sum().item()
        s["n"] += a.T
        s["dmax"] = max(s["dmax"], mx.abs(lps[x] - lps[y]).max().item())
    W._VATT, W._VERIFY_ATT_Q = False, NQ
    lm(block, cache=cache)                               # move on: the block as a plain slab
    mx.eval([c.state for c in cache])
    pos += a.T
    if b % 16 == 15:
        print(f"  {b + 1} blocks, {time.time() - t0:.0f} s", flush=True)

res = dict(tokens=a.tokens, blocks=a.blocks, T=a.T, model=os.environ["QWEN_MODEL"], pairs={})
for (x, y), s in acc.items():
    kl = sorted(s["kl"])
    n = len(kl)
    r = dict(n=n, kl_mean=sum(kl) / n, kl_p99=kl[min(n - 1, int(0.99 * n))], kl_max=kl[-1],
             top1=100.0 * s["agree"] / s["n"], max_abs_dlogprob=s["dmax"])
    res["pairs"][f"{x}|{y}"] = r
    print(f"{x:4s} vs {y:4s}: {n} positions  KL mean {r['kl_mean']:.3g}  p99 {r['kl_p99']:.3g}  "
          f"max {r['kl_max']:.3g}  top-1 {r['top1']:.2f}%  max|d logprob| {r['max_abs_dlogprob']:.3g}",
          flush=True)
g = res["pairs"]["vec|vatt"]
res["gate"] = g["kl_mean"] <= 1e-4 and g["top1"] >= 99.7
print(f"gate (KL(vec||vatt) <= 1e-4, top-1 >= 99.7 %): {'PASS' if res['gate'] else 'FAIL'}", flush=True)
if a.json:
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    json.dump(res, open(a.json, "w"), indent=1)
sys.stdout.flush()
os._exit(0)
