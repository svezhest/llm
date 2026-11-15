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

Per pair: KL(lib || vq8) mean / p99 / max over every position (fp32
log-softmax of the logits), top-1 agreement, max |d logprob|. Gate (night
rules): KL mean <= 1e-4, top-1 >= 99.7 %.

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
ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, default=32768, help="context before the first block")
ap.add_argument("--blocks", type=int, default=64)
ap.add_argument("--T", type=int, default=8)
ap.add_argument("--decode", type=int, default=128)
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
for f in ["README.md", "CLAUDE.md", "BACKLOG.md"] + sorted(glob.glob(os.path.join(ROOT, "docs/perf/2026-1*/*.md"))):
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


def verify(block, on):
    W._VATT_Q8 = on
    out = lm(block, cache=cache, speculative_verify=True)
    lp = out.logits[0].astype(mx.float32)
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    mx.eval(lp)
    abort_speculative_round(out.gdn_states)
    mx.eval([c.state for c in cache])
    return lp


def add(s, la, lb, n):
    p = mx.exp(la)
    kl = (p * (la - lb)).sum(axis=-1)
    s["kl"] += kl.reshape(-1).tolist()
    s["agree"] += (mx.argmax(la, -1) == mx.argmax(lb, -1)).sum().item()
    s["n"] += n
    s["dmax"] = max(s["dmax"], mx.abs(la - lb).max().item())


acc = {"verify": dict(kl=[], agree=0, n=0, dmax=0.0), "decode": dict(kl=[], agree=0, n=0, dmax=0.0)}
W._VATT_Q8_DEC = False
t0 = time.time()
for b in range(a.blocks):
    assert cache[FA[0]].offset == pos, (cache[FA[0]].offset, pos)
    block = inp[:, pos:pos + a.T]
    if b % 2:
        la, lb = verify(block, False), verify(block, True)
    else:
        lb = verify(block, True)
        la = verify(block, False)
    add(acc["verify"], la, lb, a.T)
    W._VATT_Q8 = False
    lm(block, cache=cache)                               # move on: the block as a plain slab
    quantize()
    mx.eval([c.state for c in cache])
    pos += a.T
    if b % 16 == 15:
        print(f"  {b + 1} blocks, {time.time() - t0:.0f} s", flush=True)


def step(x, on):
    W._VATT_Q8_DEC = on
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
    if i % 2:
        la, lb = step(x, False), step(x, True)
    else:
        lb = step(x, True)
        la = step(x, False)
    add(acc["decode"], la[None], lb[None], 1)
    W._VATT_Q8_DEC = False
    lm(x, cache=cache)
    quantize()
    mx.eval([c.state for c in cache])
    pos += 1

res = dict(tokens=a.tokens, blocks=a.blocks, T=a.T, decode=a.decode, model=os.environ["QWEN_MODEL"],
           kvq_prefill_fused=W.KVQ_PREFILL_FUSED, pairs={})
ok = True
for k, s in acc.items():
    kl = sorted(s["kl"])
    n = len(kl)
    if not n:
        continue
    r = dict(n=n, kl_mean=sum(kl) / n, kl_p99=kl[min(n - 1, int(0.99 * n))], kl_max=kl[-1],
             top1=100.0 * s["agree"] / s["n"], max_abs_dlogprob=s["dmax"])
    r["gate"] = r["kl_mean"] <= 1e-4 and r["top1"] >= 99.7
    ok &= r["gate"]
    res["pairs"][f"lib|vq8 {k}"] = r
    print(f"{k:6s} lib vs vq8: {n} positions  KL mean {r['kl_mean']:.3g}  p99 {r['kl_p99']:.3g}  "
          f"max {r['kl_max']:.3g}  top-1 {r['top1']:.2f}%  max|d logprob| {r['max_abs_dlogprob']:.3g}  "
          f"{'PASS' if r['gate'] else 'FAIL'}", flush=True)
res["gate"] = ok
print(f"gate (KL <= 1e-4, top-1 >= 99.7 %, verify and decode): {'PASS' if ok else 'FAIL'}", flush=True)
if a.json:
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    json.dump(res, open(a.json, "w"), indent=1)
sys.stdout.flush()
os._exit(0)
