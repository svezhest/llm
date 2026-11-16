"""In-process A/B of QWEN_VERIFY_MRQMM (DFlash verify blocks / the drafter's
lm_head through mrqmm) on the real 9B + DFlash drafter: the real Engine and
_Scheduler (no gRPC), one process, variants interleaved per (round, task),
medians over rounds (docs/measuring.md).

The process starts with QWEN_VERIFY_MRQMM=1 (mrqmm installed, the fused GDN
in-projections registered for verify); a variant only flips the module
globals between requests (the scheduler is idle then):
  plain        no DFlash (eng.draft = None): the plain greedy loop
  off/<blk>    DFlash, knob 0 (today's path: exact qmv <= 8, quantized_matmul above)
  head/<blk>   DFlash, drafter lm_head on mrqmm only (output must stay == off)
  on/<blk>     DFlash, verify blocks 3..16 + drafter lm_head on mrqmm
  <blk>: ad = adaptive (QWEN_DRAFT_BLOCK=0), adN = adaptive with ceiling N, 8 / 16 = fixed.

Phase speed: the 5 coding tasks (scripts/perf/prof/wtest.py TASKS[:5], greedy,
thinking off, max_tokens 256) x variants x --rounds; per request decode tok/s
(tokens after the first / time from the first to the last token), accept rate
(the worker's "dflash: ... accept rate" line), rounds.
Phase identity: all 16 wtest TASKS, plain vs each DFlash variant of --ident:
identical answers, first divergence (token index) otherwise.

  cd src/worker && QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 \\
    QWEN_DRAFT=../../weights/qwen35-9b-dflash QWEN_VERIFY_MRQMM=1 \\
    PYTHONPATH=. .venv/bin/python ../../scripts/perf/verify_mrqmm/dflash_ab.py \\
    --rounds 3 --json $PERF_OUT/verify_mrqmm/dflash_ab.json
"""
import argparse
import json
import logging
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.getcwd())
import worker as W  # noqa: E402
import worker_pb2 as pb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "prof"))
from wtest import TASKS  # noqa: E402  (no MLX there: gRPC client module, TASKS only)

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--variants", default="plain,off/ad,head/ad,on/ad,off/8,on/8,off/16,on/16")
ap.add_argument("--ident", default="off/ad,on/ad,on/16", help="DFlash variants checked against plain on all 16 tasks")
ap.add_argument("--ident-rounds", type=int, default=1)
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--skip-speed", action="store_true")
ap.add_argument("--skip-ident", action="store_true")
ap.add_argument("--json", default="")
a = ap.parse_args()

assert W.VERIFY_MR, "start with QWEN_VERIFY_MRQMM=1 (installs mrqmm + the fused GDN verify)"
assert os.environ.get("QWEN_DRAFT"), "needs QWEN_DRAFT"


class Grab(logging.Handler):
    """The worker's 'dflash: N tokens, accept rate X% (a/d drafts)' lines."""
    def __init__(self):
        super().__init__()
        self.last = None

    def emit(self, rec):
        m = re.search(r"dflash: (\d+) tokens, accept rate (\d+)% \((\d+)/(\d+) drafts\)", rec.getMessage())
        if m:
            self.last = tuple(int(g) for g in m.groups())


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")  # -> stderr
grab = Grab()
logging.getLogger("qwen.worker").addHandler(grab)

eng = W.Engine(os.environ["QWEN_MODEL"])
eng.load()
DRAFT = eng.draft
assert DRAFT is not None, "drafter not loaded (QWEN_KV_BITS=0?)"
tok = getattr(eng.processor, "tokenizer", eng.processor)
print(f"loaded; max_batch {eng.max_batch}, verify exact max T {W.VERIFY_EXACT_MAX_T}", flush=True)


def set_variant(v):
    """Flip the globals the scheduler reads (idle scheduler: between requests)."""
    if v == "plain":
        eng.draft = None
        return
    knob, blk = v.split("/")
    eng.draft = DRAFT
    W.VERIFY_MR = knob == "on"
    W.VERIFY_MR_HEAD = knob in ("on", "head")
    # ad = adaptive up to the drafter's 16; adN = adaptive with ceiling N; N = fixed N
    W.DRAFT_BLOCK = int(blk[2:] or 0) if blk.startswith("ad") else int(blk)
    DRAFT.prefer_requested_block_size = not blk.startswith("ad")


def request(text, rid):
    req = pb.Request(id=rid, messages=[pb.Message(role="user", content=text)],
                             params=pb.Params(temperature=0.0, presence_penalty=0.0,
                                              max_tokens=a.max_tokens,
                                              chat_template_kwargs_json=json.dumps({"enable_thinking": False})))
    formatted, _ = eng._render(req, [])
    ids = tok.encode(formatted)
    p = req.params
    sampling = dict(W.SAMPLING_KWARGS)
    for f in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "seed"):
        if p.HasField(f):
            sampling[f] = getattr(p, f)
    ctl = W._Control(p, tok.stopping_criteria.eos_token_ids, tok, eng.vocab)
    return W._Req(id=rid, p=p, ids=ids, max_tokens=eng._limit(p, len(ids)), sampling=sampling, ctl=ctl,
                  formatted=formatted)


def run(text, rid):
    grab.last = None
    r = request(text, rid)
    t0 = time.perf_counter()
    eng.sched.submit(r)
    toks, times = [], []
    while True:
        ev = r.out.get()
        if ev[0] != "tok":
            break
        toks.append(ev[1])
        times.append(time.perf_counter())
    n = len(toks)
    tps = (n - 1) / (times[-1] - times[0]) if n > 2 else 0.0
    out = dict(tokens=toks, n=n, ttft=times[0] - t0 if times else None, tps=tps, finish=ev[1:])
    if grab.last:
        out["accept"] = 100.0 * grab.last[2] / max(1, grab.last[3])
        out["drafted"] = grab.last[3]
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


res = {"variants": a.variants, "rounds": a.rounds, "speed": [], "ident": []}
VS = a.variants.split(",")
if not a.skip_speed:
    run(TASKS[0], "warm")                                  # compile / warm everything once
    for v in VS:
        set_variant(v)
        run("Say OK.", f"warm-{v}")
    for rnd in range(a.rounds):
        for ti, text in enumerate(TASKS[:5]):
            order = VS[rnd % len(VS):] + VS[:rnd % len(VS)]     # rotate: no variant always first
            if (rnd + ti) % 2:
                order = order[::-1]
            for v in order:
                set_variant(v)
                o = run(text, f"s{rnd}-code{ti}-{v}")
                o.update(variant=v, round=rnd, task=ti)
                res["speed"].append(o)
                print(f"r{rnd} code{ti} {v:8s} n={o['n']:3d} tps={o['tps']:6.1f} "
                      f"accept={o.get('accept', 0):5.1f}%", flush=True)
    plain = {(o["round"], o["task"]): o["tokens"] for o in res["speed"] if o["variant"] == "plain"}
    print("\n== speed: median decode tok/s over rounds x tasks (median of per-task medians) ==")
    summ = {}
    for v in VS:
        rows = [o for o in res["speed"] if o["variant"] == v]
        per_task = [med([o["tps"] for o in rows if o["task"] == t]) for t in range(5)]
        same = sum(o["tokens"] == plain.get((o["round"], o["task"])) for o in rows)
        summ[v] = dict(tps=med(per_task), per_task=per_task, accept=med([o.get("accept") for o in rows]),
                       same_as_plain=f"{same}/{len(rows)}")
        print(f"{v:8s} tps {summ[v]['tps']:6.1f}  per task " + " ".join(f"{x:5.1f}" for x in per_task)
              + f"  accept {summ[v]['accept'] or 0:5.1f}%  == plain {same}/{len(rows)}", flush=True)
    res["speed_summary"] = summ

if not a.skip_ident:
    print("\n== identity vs plain greedy, 16 tasks ==", flush=True)
    for rnd in range(a.ident_rounds):
        for ti, text in enumerate(TASKS):
            set_variant("plain")
            ref = run(text, f"i{rnd}-t{ti}-plain")["tokens"]
            for v in a.ident.split(","):
                set_variant(v)
                got = run(text, f"i{rnd}-t{ti}-{v}")["tokens"]
                div = next((i for i, (x, y) in enumerate(zip(ref, got)) if x != y),
                           None if len(ref) == len(got) else min(len(ref), len(got)))
                res["ident"].append(dict(round=rnd, task=ti, variant=v, same=div is None, diverge=div,
                                         n=len(ref)))
    for v in a.ident.split(","):
        rows = [o for o in res["ident"] if o["variant"] == v]
        print(f"{v:8s} identical {sum(o['same'] for o in rows)}/{len(rows)}; divergence at "
              f"{[o['diverge'] for o in rows if not o['same']]}", flush=True)

set_variant("off/ad")
import mlx.core as mx  # noqa: E402
res["peak_gb"] = mx.get_peak_memory() / 1e9
print(f"peak {res['peak_gb']:.2f} GB", flush=True)
if a.json:
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    for o in res["speed"]:
        o.pop("tokens", None)
    json.dump(res, open(a.json, "w"), indent=1)
    print("wrote", a.json)
sys.stdout.flush()
os._exit(0)          # the scheduler thread never returns (worker.py: MLX stream teardown)
