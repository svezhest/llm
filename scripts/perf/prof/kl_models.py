"""Teacher-forced distance between two CHECKPOINTS of one model (e.g. quants)
that do not fit in memory together: pass 1 (--save DIR) runs the reference and
writes its per-position log-probabilities (fp16, full vocabulary) to DIR; pass
2 (--ref DIR) runs another checkpoint on the same tokens and reports, per text
and overall, KL(ref || this) mean / p99 / max and top-1 agreement.

Both passes load the model as the worker does (fp16 cast, fast paths, rows
attention, fused GDN where the bits allow, pfqmm) and prefill in fixed slabs
of --slab tokens (the same cut for every checkpoint), logits for every
position. Texts: --text NAME=PATH (repeatable; first --tokens tokens of each).

  cd src/worker && QWEN_MODEL=../../weights/<ref> QWEN_KV_BITS=0 PYTHONPATH=. .venv/bin/python \\
    ../../scripts/perf/prof/kl_models.py --save $PERF_OUT/kl/ref --text en=../../README.md ...
  QWEN_MODEL=../../weights/<other> ... kl_models.py --ref $PERF_OUT/kl/ref --text ... (same texts)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import mlx.core as mx  # noqa: E402
import pf_common as C  # noqa: E402
import worker as W  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--save", default="")
ap.add_argument("--ref", default="")
ap.add_argument("--text", action="append", required=True, help="NAME=PATH")
ap.add_argument("--tokens", type=int, default=3072)
ap.add_argument("--slab", type=int, default=256)
ap.add_argument("--json", default="")
a = ap.parse_args()
assert bool(a.save) != bool(a.ref), "exactly one of --save / --ref"

t0 = time.time()
model, proc = C.load(os.environ.get("QWEN_MODEL"))
lm = model.language_model
import pfqmm  # noqa: E402
pfqmm.install(lm)
tok = getattr(proc, "tokenizer", proc)
print(f"loaded {os.environ.get('QWEN_MODEL')} in {time.time() - t0:.1f} s, active "
      f"{mx.get_active_memory() / 2**30:.2f} GiB", flush=True)


def logprobs(ids):
    """-> list of (n, V) float32 arrays: log p(next | prefix) at every position."""
    from mlx_vlm.models.cache import make_prompt_cache
    cache = make_prompt_cache(lm)
    inp = mx.array([ids])
    pos, out = 0, []
    while pos < len(ids) - 1:
        n = min(a.slab, len(ids) - 1 - pos)
        lg = lm(inp[:, pos:pos + n], cache=cache).logits[0].astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        mx.eval(lp, [c.state for c in cache])
        out.append(lp)
        pos += n
        mx.clear_cache()
    return out


report = {"model": os.environ.get("QWEN_MODEL"), "slab": a.slab, "texts": {}}
all_kl, all_agree, all_n = [], 0, 0
for spec in a.text:
    name, path = spec.split("=", 1)
    ids = tok.encode(open(path, encoding="utf-8").read())[: a.tokens]
    t1 = time.time()
    lps = logprobs(ids)
    if a.save:
        os.makedirs(a.save, exist_ok=True)
        full = np.concatenate([np.array(x.astype(mx.float16)) for x in lps])
        np.save(os.path.join(a.save, f"{name}.npy"), full)
        json.dump(ids, open(os.path.join(a.save, f"{name}.ids.json"), "w"))
        print(f"{name}: {len(ids)} tokens saved ({full.nbytes / 2**30:.2f} GiB) in {time.time() - t1:.0f} s",
              flush=True)
        continue
    assert json.load(open(os.path.join(a.ref, f"{name}.ids.json"))) == ids, f"{name}: other tokens"
    ref = np.load(os.path.join(a.ref, f"{name}.npy"), mmap_mode="r")
    kls, agree, n, off = [], 0, 0, 0
    for y in lps:
        m = y.shape[0]
        x = mx.array(np.asarray(ref[off:off + m], dtype=np.float32))
        kls.append((mx.exp(x) * (x - y)).sum(axis=-1))
        agree += (mx.argmax(x, -1) == mx.argmax(y, -1)).sum().item()
        n += m
        off += m
    kl = np.array(mx.concatenate(kls))
    all_kl.append(kl)
    all_agree += agree
    all_n += n
    r = dict(positions=n, kl_mean=float(kl.mean()), kl_p99=float(np.quantile(kl, 0.99)),
             kl_max=float(kl.max()), top1=agree / n)
    report["texts"][name] = r
    print(f"{name}: {n} positions  KL mean {r['kl_mean']:.4g}  p99 {r['kl_p99']:.3g}  max {r['kl_max']:.3g}  "
          f"top-1 {100 * r['top1']:.2f}%  ({time.time() - t1:.0f} s)", flush=True)
if a.ref:
    kl = np.concatenate(all_kl)
    report["all"] = dict(positions=all_n, kl_mean=float(kl.mean()), kl_p99=float(np.quantile(kl, 0.99)),
                         kl_max=float(kl.max()), top1=all_agree / all_n)
    r = report["all"]
    print(f"ALL: {all_n} positions  KL mean {r['kl_mean']:.4g}  p99 {r['kl_p99']:.3g}  max {r['kl_max']:.3g}  "
          f"top-1 {100 * r['top1']:.2f}%", flush=True)
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump(report, open(a.json, "w"), indent=1)
