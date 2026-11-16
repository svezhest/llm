"""Where does an MTP round's time go on the 27B? After a real prompt prefill
(the Engine as the worker loads it: QWEN_* env, QWEN_DRAFT_MTP or an MTP
QWEN_DRAFT), each stage synced and timed, medians over --iters:

  plain        one decode step (1 token, return_hidden), as the plain arm
  draft k      the head: flush of 1 pending position + k - 1 chained steps
  verify T     the target over T = k + 1 tokens, speculative_verify, committed
               with 0 drafts kept (the cache grows by one per call)
  round k      draft + verify in one eval, as _spec queues them

  cd src/worker && <27B env> QWEN_DRAFT_MTP=../../weights/qwen38-27b-mtp PYTHONPATH=. \\
    .venv/bin/python ../../scripts/perf/mtp/round_prof.py --ctx 2048
"""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.getcwd())
import worker as W  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ctx", type=int, default=2048)
ap.add_argument("--iters", type=int, default=12)
ap.add_argument("--ks", default="1,2,3,4,5,7")
a = ap.parse_args()

import mlx.core as mx  # noqa: E402
from mlx_vlm.models.cache import make_prompt_cache  # noqa: E402
from mlx_vlm.speculative.cache_state import commit_speculative_round  # noqa: E402

eng = W.Engine(os.environ["QWEN_MODEL"])
eng.load()
mt, lm = eng.mtp, eng.model.language_model
assert mt is not None, "no MTP head loaded"
print(f"VERIFY_MR {W.VERIFY_MR} active {mx.get_active_memory() / 2**30:.2f} GiB", flush=True)

mx.random.seed(0)
ids = mx.random.randint(1000, 100000, (1, a.ctx)).astype(mx.int32)
cache = make_prompt_cache(lm)
mt.start(0)
for s in range(0, a.ctx - 1, 2048):
    e = min(a.ctx - 1, s + 2048)
    h = lm.model(ids[:, s:e], cache=cache)
    mt.push(h)
    mt.emit_ids(ids[0, s + 1:e + 1].tolist())
    mt.flush()
    mx.eval(h, [c.state for c in cache], mt.state())
b = int(ids[0, -1])
T = {}


def tm(name, f):
    t = time.perf_counter()
    y = f()
    mx.eval(y)
    T.setdefault(name, []).append((time.perf_counter() - t) * 1000)
    return y


def plain():
    o = lm(mx.array([[b]]), cache=cache, return_hidden=True)
    return o.logits, o.hidden_states[-1], [c.state for c in cache]


def verify(fed):
    out = lm(fed, cache=cache, speculative_verify=True, return_hidden=True)
    lg, hs = out.logits, out.hidden_states[-1]
    mx.eval(lg, hs)
    commit_speculative_round(lm, cache, out.gdn_states, 0, fed.shape[1])
    W._own_states(cache)
    return lg, hs[:, :1], [c.state for c in cache]


ks = [int(k) for k in a.ks.split(",")]
QMV0 = W.VERIFY_QMV                            # QWEN_VERIFY_QMV=N: mrqmv installed, compared
paths = [("", 0)] + ([(" qmv", QMV0)] if QMV0 else [])
for it in range(a.iters):
    o = tm("plain step", plain)
    mt.round_end(0, o[1])                      # the plain step's position joins the head
    mt.emit(b)
    for k in ks:
        d = tm(f"draft k={k}", lambda: mt.draft(k))
        mt.round_end(0, None)
        for tag, q in paths:
            W.VERIFY_QMV = q
            if q and k + 1 > q:
                continue
            fed = mx.concatenate([mx.array([[b]], dtype=mx.int32), d], axis=1)
            _, hs, _ = tm(f"verify T={k + 1}{tag}", lambda: verify(fed))
            mt.round_end(0, hs)
            mt.emit(b)
            t = time.perf_counter()            # the round as _spec queues it
            d2 = mt.draft(k)
            mx.async_eval(d2)
            fed = mx.concatenate([mx.array([[b]], dtype=mx.int32), d2], axis=1)
            _, hs, st = verify(fed)
            mx.eval(st)
            T.setdefault(f"round k={k}{tag}", []).append((time.perf_counter() - t) * 1000)
            mt.round_end(0, hs)
            mt.emit(b)
    W.VERIFY_QMV = QMV0

skip = 2
print(f"ctx {a.ctx}, medians over {a.iters - skip} iterations (first {skip} dropped), ms:")
for k, v in T.items():
    print(f"  {k:14s} {statistics.median(v[skip:]):7.2f}   (min {min(v[skip:]):.2f})")
