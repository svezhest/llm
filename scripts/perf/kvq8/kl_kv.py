"""Teacher-forced distance of quantized KV from fp16 KV on one model (gate (i)
of docs/perf/2026-11-15/design/kv_q8.md).

One process, the model loaded as the worker loads it (fp16 cast, fast paths,
rows attention, fused GDN, pfqmm). The same --tokens tokens of the repo's
docs go through each variant in turn, as the worker's text path would run
them:
  fp16  QWEN_KV_BITS=0: fp16 KV, the fixed slab (fused SDPA from
        QWEN_SDPA_FUSED_MIN_KV keys);
  q8    QWEN_KV_BITS=8: quantized after every slab / step once the cache
        reaches quantized_kv_start (maybe_quantize_kv_cache, the worker's
        self.quantize), the library's quantized SDPA, slabs shrunk to
        QWEN_PREFILL_BUDGET // offset (_step);
  q8f   q8 + QWEN_KVQ_PREFILL_FUSED=1: long slabs attend the dequantized KV
        through the fused kernel, the fixed slab.
Prefill: the first --tokens tokens in the variant's own slabs (sizing._step);
then --decode tokens one forward each (teacher forcing through decode: the
L = 1 attention a generated token takes). Each variant keeps the final hidden
states (post-norm, return_hidden) of every scored position on the host; the
logits are built afterwards, --chunk positions at a time, through the
model's lm_head for every variant (the vocab-wide logits never exist for
the whole text). One KV cache at a time: the variants never share memory.

Reported per pair (q8 / q8f against fp16, q8f against q8) and region:
"fp16 part" (positions below quantized_kv_start: every variant reads the
same fp16 KV there, so KL must be 0 — a check of the harness), "prefill"
(positions from the start on), "decode": KL(a || b) mean / p99 / max, top-1
agreement, and each variant's mean NLL of the true next token.

  cd src/worker && QWEN_MODEL=../../weights/qwen38-27b-mlx-4bit QWEN_WIRED_LIMIT_GB=27 \\
    QWEN_CACHE_LIMIT_GB=1 PYTHONPATH=. .venv/bin/python ../../scripts/perf/kvq8/kl_kv.py \\
    --tokens 32768 --decode 512 --json $PERF_OUT/kvq8/kl_32k.json
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "prof"))
from doc_notes import NOTES  # noqa: E402
sys.path.insert(0, os.getcwd())
import mlx.core as mx  # noqa: E402
import worker as W  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, default=32768, help="prefilled tokens")
ap.add_argument("--decode", type=int, default=512, help="teacher-forced one-token steps after the prefill")
ap.add_argument("--variants", nargs="+", default=["fp16", "q8", "q8f"],
                help="fp16 / qN / qNf (N = KV bits; f = QWEN_KVQ_PREFILL_FUSED); the first is the reference")
ap.add_argument("--start", type=int, default=W.KV_KWARGS["quantized_kv_start"], help="quantized_kv_start")
ap.add_argument("--chunk", type=int, default=256, help="positions per lm_head chunk")
ap.add_argument("--json", default="")
ap.add_argument("--cpu", action="store_true", help="smoke test of the script on the CPU: the test "
                "suite's tiny random model (8-bit weights), random tokens")
a = ap.parse_args()
if a.cpu:
    mx.set_default_device(mx.cpu)

from mlx_vlm.generate.common import maybe_quantize_kv_cache  # noqa: E402
from mlx_vlm.models.cache import make_prompt_cache  # noqa: E402
from qwen_worker.logprobs import _lm_head  # noqa: E402

t0 = time.time()
if a.cpu:                                       # the test suite's tiny random model, random tokens
    sys.path.insert(0, os.path.join(os.getcwd(), "tests"))
    from conftest import V, build_lm  # noqa: E402
    lm = build_lm("dense", 8, mx.float16)
    ids = np.random.default_rng(0).integers(8, V, a.tokens + a.decode + 1).tolist()
else:
    import pf_common as C  # noqa: E402
    model, proc = C.load(os.environ.get("QWEN_MODEL"))
    lm = model.language_model
    import pfqmm  # noqa: E402
    pfqmm.install(lm)
    tok = getattr(proc, "tokenizer", proc)
    text = ""
    for f in list(NOTES) + sorted(p for p in
            glob.glob(os.path.join(ROOT, "docs", "**", "*.md"), recursive=True)
            if os.path.relpath(p, ROOT) not in NOTES):
        text += open(os.path.join(ROOT, f), encoding="utf-8").read() + "\n\n"
        if len(text) > 6 * (a.tokens + a.decode):
            break
    ids = tok.encode(text)[: a.tokens + a.decode + 1]
    assert len(ids) == a.tokens + a.decode + 1, f"only {len(ids)} tokens of documents"
print(f"loaded {os.environ.get('QWEN_MODEL', 'tiny (cpu)')} in {time.time() - t0:.1f} s, active "
      f"{mx.get_active_memory() / 2**30:.2f} GiB; SDPA_FUSED {W.SDPA_FUSED}, MIN_KV {W.SDPA_FUSED_MIN_KV}, "
      f"PREFILL_BUDGET {W.PREFILL_BUDGET}", flush=True)
P, N = a.tokens, a.tokens + a.decode           # hidden state j predicts ids[j + 1], j < N


def parse(v):
    if v == "fp16":
        return None, False
    assert v[0] == "q", v
    return int(v[1:].rstrip("f")), v.endswith("f")


def run(variant):
    """-> (N, hidden) fp16 host array of final hidden states, prefill slabs used."""
    bits, fused = parse(variant)
    W.KVQ_PREFILL_FUSED = fused
    W._FIXED_SLAB = bool((bits is None or fused) and W.SDPA_FUSED and W.FP16)
    lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int32)      # text: positions = cache offset (a text hit's state)
    lm._position_ids = None
    cache = make_prompt_cache(lm)
    inp = mx.array([ids])
    out, slabs, pos = [], [], 0
    t = time.time()

    def quantize():
        if bits is not None:
            maybe_quantize_kv_cache(cache, quantized_kv_start=a.start, kv_group_size=64, kv_bits=bits)

    while pos < N:
        n = min(W._step(pos), P - pos) if pos < P else 1
        h = lm(inp[:, pos:pos + n], cache=cache, skip_logits=True, return_hidden=True).hidden_states[-1]
        quantize()
        mx.eval(h, [c.state for c in cache])
        out.append(np.array(h[0].astype(mx.float16)))
        if pos < P:
            slabs.append(n)
        pos += n
        if pos == P:
            print(f"  {variant}: prefill {P} tokens in {time.time() - t:.1f} s ({len(slabs)} slabs, "
                  f"min {min(slabs)}), peak {mx.get_peak_memory() / 2**30:.2f} GiB", flush=True)
            t = time.time()
        if W.CLEAR_CACHE:
            mx.clear_cache()
    q = [c for c in cache if hasattr(c, "bits")]
    print(f"  {variant}: decode {a.decode} steps in {time.time() - t:.1f} s; quantized FA layers "
          f"{len(q)}, peak {mx.get_peak_memory() / 2**30:.2f} GiB", flush=True)
    del cache
    mx.clear_cache()
    return np.concatenate(out), slabs


hid, info = {}, {}
for v in a.variants:
    mx.reset_peak_memory()
    hid[v], slabs = run(v)
    info[v] = dict(slabs=len(slabs), min_slab=min(slabs), peak_gib=round(mx.get_peak_memory() / 2**30, 2))

regions = {"fp16 part": (0, min(a.start, P)), "prefill": (min(a.start, P), P), "decode": (P, N)}
pairs = [(a.variants[0], v) for v in a.variants[1:]]
pairs += [(x, y) for x in a.variants[1:] for y in a.variants[1:] if x < y and y == x + "f"]
tgt = np.array(ids[1:N + 1])
acc = {p: {r: dict(kl=[], agree=0, n=0) for r in regions} for p in pairs}
nll = {v: {r: [] for r in regions} for v in a.variants}
t = time.time()
for lo in range(0, N, a.chunk):
    hi = min(N, lo + a.chunk)
    lp = {}
    for v in a.variants:
        lg = _lm_head(lm, mx.array(hid[v][lo:hi])[None])[0].astype(mx.float32)
        lp[v] = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    y = mx.array(tgt[lo:hi])
    picked = {v: np.array(mx.take_along_axis(lp[v], y[:, None], axis=-1)[:, 0]) for v in a.variants}
    for x, z in pairs:
        kl = np.array((mx.exp(lp[x]) * (lp[x] - lp[z])).sum(axis=-1))
        agree = np.array(mx.argmax(lp[x], -1) == mx.argmax(lp[z], -1))
        for r, (r0, r1) in regions.items():
            s0, s1 = max(lo, r0) - lo, min(hi, r1) - lo
            if s1 > s0:
                acc[(x, z)][r]["kl"].append(kl[s0:s1])
                acc[(x, z)][r]["agree"] += int(agree[s0:s1].sum())
                acc[(x, z)][r]["n"] += s1 - s0
    for v in a.variants:
        for r, (r0, r1) in regions.items():
            s0, s1 = max(lo, r0) - lo, min(hi, r1) - lo
            if s1 > s0:
                nll[v][r].append(-picked[v][s0:s1])
print(f"logits of {N} positions x {len(a.variants)} variants in {time.time() - t:.1f} s", flush=True)

report = dict(model=os.environ.get("QWEN_MODEL"), tokens=P, decode=a.decode, start=a.start,
              variants=info, pairs={}, nll={})
for v in a.variants:
    report["nll"][v] = {r: float(np.concatenate(x).mean()) for r, x in nll[v].items() if x}
    print(f"{v:5s} NLL " + "  ".join(f"{r} {m:.4f}" for r, m in report["nll"][v].items())
          + f"   slabs {info[v]['slabs']} (min {info[v]['min_slab']}), peak {info[v]['peak_gib']} GiB")
for (x, z), rs in acc.items():
    key = f"{z} vs {x}"
    report["pairs"][key] = {}
    for r, d in rs.items():
        if not d["n"]:
            continue
        kl = np.concatenate(d["kl"])
        s = dict(positions=d["n"], kl_mean=float(kl.mean()), kl_p99=float(np.quantile(kl, 0.99)),
                 kl_max=float(kl.max()), top1=d["agree"] / d["n"])
        report["pairs"][key][r] = s
        print(f"{key:12s} {r:9s} {s['positions']:6d} pos  KL mean {s['kl_mean']:.3g}  p99 {s['kl_p99']:.3g}  "
              f"max {s['kl_max']:.3g}  top-1 {100 * s['top1']:.2f}%", flush=True)
if a.json:
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    json.dump(report, open(a.json, "w"), indent=1)
