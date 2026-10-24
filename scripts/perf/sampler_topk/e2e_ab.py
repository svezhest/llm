"""End-to-end A/B of QWEN_SAMPLER_TOPK_FAST on the real 9B: 16 concurrent
card-sampled requests (temperature 0.6, top_p 0.95, top_k 20, seeded,
ignore_eos, --gen tokens) through the in-process Engine + _Scheduler (no
gRPC), rounds alternating off / on (ABBA: samplers read the global when a
request is admitted, so a round is all one variant). Per round: the median
per-request decode tok/s (first -> last token), aggregate tok/s over the
round's decode window, the step period (= 1 / median per-request tok/s);
medians per variant. Prompts are short (~40 tokens): prefill is negligible.

  cd src/worker && QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 \\
    QWEN_BATCH_PREFILL_SHARE=1 PYTHONPATH=. .venv/bin/python \\
    ../../scripts/perf/sampler_topk/e2e_ab.py --rounds 6 --json $PERF_OUT/sampler_topk/e2e_ab.json
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time

sys.path.insert(0, os.getcwd())
import worker as W  # noqa: E402
import worker_pb2 as pb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "prof"))
from wtest import TASKS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--B", type=int, default=16)
ap.add_argument("--gen", type=int, default=256)
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--json", default="")
a = ap.parse_args()
assert not os.environ.get("QWEN_DRAFT"), "no drafter here (card sampling would not take it anyway)"

import logging  # noqa: E402
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")  # -> stderr
eng = W.Engine(os.environ["QWEN_MODEL"])
eng.load()
tok = getattr(eng.processor, "tokenizer", eng.processor)
print(f"loaded; max_batch {eng.max_batch}", flush=True)


def request(i, rid):
    p = pb.Params(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.0, seed=1000 + i,
                  max_tokens=a.gen, ignore_eos=True,
                  chat_template_kwargs_json=json.dumps({"enable_thinking": False}))
    req = pb.Request(id=rid, messages=[pb.Message(role="user", content=TASKS[i % len(TASKS)])], params=p)
    formatted, _ = eng._render(req, [])
    ids = tok.encode(formatted)
    sampling = dict(W.SAMPLING_KWARGS)
    for f in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "seed"):
        if p.HasField(f):
            sampling[f] = getattr(p, f)
    ctl = W._Control(p, tok.stopping_criteria.eos_token_ids, tok, eng.vocab)
    return W._Req(id=rid, p=p, ids=ids, max_tokens=eng._limit(p, len(ids)), sampling=sampling, ctl=ctl,
                  formatted=formatted)


def drain(r, out):
    times = []
    while True:
        ev = r.out.get()
        if ev[0] != "tok":
            break
        times.append(time.perf_counter())
    out.append(times)


def one_round(fast, tag):
    W.SAMPLER_TOPK_FAST = fast
    reqs = [request(i, f"{tag}-{i}") for i in range(a.B)]
    outs, ths = [], []
    for r in reqs:
        eng.sched.submit(r)
        th = threading.Thread(target=drain, args=(r, outs))
        th.start()
        ths.append(th)
    for th in ths:
        th.join()
    W.SAMPLER_TOPK_FAST = False
    tps = [(len(t) - 1) / (t[-1] - t[0]) for t in outs if len(t) > 2]
    lo, hi = max(t[0] for t in outs), min(t[-1] for t in outs)     # all rows decoding
    agg = sum(sum(lo <= x <= hi for x in t) for t in outs) / max(hi - lo, 1e-9)
    return dict(tps=statistics.median(tps), agg=agg, step_ms=1e3 / statistics.median(tps))


one_round(False, "warm0")
one_round(True, "warm1")
res = {"off": [], "on": []}
for k in range(a.rounds):
    v = ("off", "on", "on", "off")[k % 4]
    o = one_round(v == "on", f"r{k}-{v}")
    res[v].append(o)
    print(f"round {k} {v:3s}: per-request {o['tps']:5.1f} tok/s, step {o['step_ms']:5.1f} ms, "
          f"aggregate {o['agg']:6.1f} tok/s", flush=True)
summ = {v: {m: statistics.median(o[m] for o in rs) for m in ("tps", "agg", "step_ms")} for v, rs in res.items()}
for v, s in summ.items():
    print(f"{v:3s}: step {s['step_ms']:.1f} ms, per-request {s['tps']:.1f} tok/s, aggregate {s['agg']:.1f} tok/s")
import mlx.core as mx  # noqa: E402
print(f"peak {mx.get_peak_memory() / 1e9:.2f} GB")
print(f"step on/off x{summ['on']['step_ms'] / summ['off']['step_ms']:.3f}", flush=True)
if a.json:
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    json.dump({"rounds": res, "summary": summ}, open(a.json, "w"), indent=1)
    print("wrote", a.json)
sys.stdout.flush()
os._exit(0)
