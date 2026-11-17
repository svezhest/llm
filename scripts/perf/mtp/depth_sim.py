"""The MTP head's gate (MTPDraft: draft count per round, plain bursts) replayed
on recorded acceptance, no model, no GPU: pure Python over accept_probe.py's
JSON. mtp.py is loaded with stand-ins for mlx and the worker's config, so the
policy that runs is the file's own code (`--mtp` another copy of it, e.g. an
older commit's `git show <rev>:src/worker/qwen_worker/mtp.py`; the per-round
gate before 2026-11-18 is driven through its depth(c1, T, top)).

Per text and trajectory (greedy / sampled) of one head (`--head`): at every
generated position p accept_probe.py recorded the chain of KMAX drafts the
head gives from the real context, so a round anchored at p with k drafts
keeps the longest prefix of chain[p][:k] equal to the trajectory's next
tokens (a sample-match verify on the sampled one: the recorded draw), emits
that + 1 and costs T(k); a burst of n plain steps emits n at n * T(0) + one
catch-up flush. T(k) per mode from `--ms` (ms of a plain step and of a
1..KMAX-draft round: measured, e.g. round_prof.py), each round's time drawn
around it with a log-normal spread `--cv`, now and then an outlier
(`--outlier`: a pause, a cold kernel; a kind's first sample sets its Clock
time) and a drift of the machine's speed
per request (`--drift`; the variants of a text in one pass share it). The rounds' times feed the class's Clock as _spec
does (every round but a request's first), record() its acceptance.

Requests: `--rounds` passes over the texts in their order; per text the
variants in `--variants` back to back (`mtpK`: a fixed K drafts; `mtp0`:
the gate), each in its own process (Clock) or, `--shared`, all in one as
ab_mtp.py runs them (the fixed variants' rounds then feed the gate's
statistics and round times too). Prints per text the median over passes
of tokens per ms x plain, the median over passes of the gate over the best
fixed depth of the pass, the gate's rounds by draft count and burst tokens.

  python3 scripts/perf/mtp/depth_sim.py $PERF_OUT/mtp_head/accept.json --head q8 \\
    --ms greedy=28.1,40.0,48.1,53.3,58.4 --ms sampled=30.4,44.0,55.0,66.3,72.0
"""
import argparse
import importlib.util
import json
import logging
import math
import os
import random
import statistics
import sys
import types


def load_mtp(path):
    """mtp.py as a module, mlx and the package's config / kernels stood in for
    (the gate's policy is host statistics only)."""
    mlx, core = types.ModuleType("mlx"), types.ModuleType("mlx.core")
    mlx.core = core
    pkg = types.ModuleType("qwen_worker")
    pkg.__path__ = []
    cfg = types.ModuleType("qwen_worker.config")
    cfg.MTP_K, cfg.MTP_KMAX, cfg.log = 0, 4, logging.getLogger("depth_sim")
    ker = types.ModuleType("qwen_worker.kernels")
    ker._rows_linear = None
    sys.modules.update({"mlx": mlx, "mlx.core": core, "qwen_worker": pkg,
                        "qwen_worker.config": cfg, "qwen_worker.kernels": ker})
    spec = importlib.util.spec_from_file_location("qwen_worker.mtp", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["qwen_worker.mtp"] = m
    spec.loader.exec_module(m)
    return m, cfg


ap = argparse.ArgumentParser()
ap.add_argument("json")
ap.add_argument("--mtp", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "../../../src/worker/qwen_worker/mtp.py"))
ap.add_argument("--head", default="q8")
ap.add_argument("--modes", default="greedy,sampled")
ap.add_argument("--texts", default="", help="keys in order (default: the JSON's)")
ap.add_argument("--ms", action="append", default=[],
                help="mode=T0,T1,..,TKMAX: a plain step and a k-draft round, ms")
ap.add_argument("--variants", default="mtp1,mtp2,mtp3,mtp4,mtp0")
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--cv", type=float, default=0.15)
ap.add_argument("--drift", type=float, default=0.1)
ap.add_argument("--outlier", default="0.02,2.0", help="p,x: a round x times its time with chance p")
ap.add_argument("--cold", type=float, default=1.0, help="the process's first round of each draft count: x its time")
ap.add_argument("--flush", type=float, default=5.0, help="a burst's catch-up flush, ms")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--shared", action="store_true",
                help="one Clock for all variants, as ab_mtp.py's one process (else one each)")
ap.add_argument("--set", action="append", default=[], help="MTPDraft.NAME=value (a constant to try)")
a = ap.parse_args()

M, CFG = load_mtp(a.mtp)
for kv in a.set:
    name, val = kv.split("=", 1)
    setattr(M.MTPDraft, name, eval(val))
d = json.load(open(a.json))
kmax = int(d["args"]["kmax"])
CFG.MTP_KMAX = kmax
ms = {"greedy": [28.1, 40.0, 48.1, 53.3, 58.4], "sampled": [30.4, 44.0, 55.0, 66.3, 72.0]}
for x in a.ms:
    mode, v = x.split("=")
    ms[mode] = [float(t) for t in v.split(",")]
NEW = hasattr(M.MTPDraft, "plan")
rng = random.Random(a.seed)
P_OUT, X_OUT = (float(x) for x in a.outlier.split(","))
COLD = set()


def took(base):
    """A round's time: log-normal around `base`, now and then an outlier."""
    return base * math.exp(rng.gauss(0.0, a.cv)) * (X_OUT if rng.random() < P_OUT else 1.0)


def gate(mt, top, conf):
    """(k, burst steps) of the next round: the file's own plan(), or the old
    per-round gate's depth() and round()'s bookkeeping."""
    if NEW:
        return mt.plan(top)
    k = mt.depth(conf, [mt._ms(j) for j in range(top + 1)], top)
    if k:
        return k, 0
    mt.low += 1
    mt.ks[0] += 1
    mt.miss = mt.run - 1
    return 0, mt.BURSTS[min(mt.low, len(mt.BURSTS)) - 1]


def request(t, mt, clock, T, speed):
    """One request over text t -> (tokens, ms)."""
    h = t["heads"][a.head]
    gen, chain, conf = t["gen"], h["chain"], h["conf"]
    n = len(chain)
    i = tok = 0
    ms_ = 0.0
    first = True
    while i < n:
        top = min(mt.KMAX, n - i)
        if mt.k:
            k, steps = min(mt.k, top), 0
            mt.ks[k] += 1
        else:
            k, steps = gate(mt, top, conf[i][0])
            if k:
                mt.ks[k] += 1
        if not k:
            s = min(steps, n - i)
            dt = sum(took(T[0] * speed) for _ in range(s)) + a.flush * speed
            ms_, tok, i = ms_ + dt, tok + s, i + s
            mt.burst_tokens += s
            if not first:
                clock.add(M.MTPDraft.kind_of(0), dt / s)
            first = False
            continue
        acc = 0
        while acc < k and i + 1 + acc < len(gen) and chain[i][acc] == gen[i + 1 + acc]:
            acc += 1
        dt = took(T[k] * speed) * (a.cold if (id(clock), k) not in COLD else 1.0)
        COLD.add((id(clock), k))
        ms_, tok, i = ms_ + dt, tok + acc + 1, i + acc + 1
        if not first:
            clock.add(M.MTPDraft.kind_of(k), dt)
        first = False
        mt.record(acc, k)
    return tok, ms_


variants = a.variants.split(",")
for mode in a.modes.split(","):
    T = ms[mode]
    texts = [t for t in d["texts"] if t["mode"] == mode]
    if a.texts:
        by = {t["key"]: t for t in texts}
        texts = [by[k] for k in a.texts.split(",")]
    roots = {v: M.Clock() for v in variants}
    res, ks, burst = {}, {}, {}
    speed = 1.0
    for r in range(a.rounds):
        for t in texts:
            speed = min(1.5, max(0.67, speed * math.exp(rng.gauss(0.0, a.drift))))
            for v in variants:
                clock = roots[variants[0] if a.shared else v].at((0, mode != "greedy", False))
                mt = M.MTPDraft.__new__(M.MTPDraft)
                mt.k, mt.clock = int(v[3:]) or None, clock
                mt.fresh()
                tok, ms_ = request(t, mt, clock, T, speed)
                res.setdefault((t["key"], v), []).append(tok / ms_ * T[0] * speed)
                if v == "mtp0":
                    ks.setdefault(t["key"], [0] * len(mt.ks))
                    ks[t["key"]] = [x + y for x, y in zip(ks[t["key"]], mt.ks)]
                    burst[t["key"]] = burst.get(t["key"], 0) + mt.burst_tokens / tok
    print(f"== {mode}: T {T} ms, cv {a.cv}, outliers {a.outlier}, drift {a.drift}, {a.rounds} passes, "
          f"{'plan()' if NEW else 'the per-round gate'} of {a.mtp}")
    print("| text | " + " | ".join(variants) + " | gate / best fixed | gate's rounds by k | burst tokens |")
    print("|---|" + "---|" * (len(variants) + 3))
    for t in texts:
        m = {v: statistics.median(res[(t["key"], v)]) for v in variants}
        fixed = [v for v in variants if v != "mtp0"]
        ratio = ""
        if "mtp0" in m and fixed:            # per pass (the same machine speed), then the median
            ratio = f"{statistics.median(res[(t['key'], 'mtp0')][r] / max(res[(t['key'], v)][r] for v in fixed) for r in range(a.rounds)):.3f}"
        print(f"| {t['key']} | " + " | ".join(f"{m[v]:.3f}" for v in variants)
              + f" | {ratio} | {ks.get(t['key'], '')} | {100 * burst.get(t['key'], 0) / a.rounds:.1f} % |")
