"""Replay of the per-round arm choice (qwen_worker.mtp.Chooser + MTPDraft's
adaptive draft count + Shadows: the worker's own classes) over a dump of both
drafters' drafts at every position of a greedy answer (CPU, no model).

The dump: ab_mtp.py --dump (QWEN_SPEC_SHADOW=all with plain steps forced: at
every position of the plain answer DFlash's block and QWEN_MTP_KMAX chained MTP
drafts, never verified). A draft's accepted prefix at a position does not depend
on which arm took the rounds before it (the emitted tokens are the target's
argmax whichever arm drafted; DFlash's context and the head's entries are the
target's hidden states), so any policy's rounds follow from the dump; the wall
time is a table of ms per kind (--ms, e.g. the "clock" of an ab_mtp.py JSON)
plus the cost of the shadow drafts. A sampling cell's dump works the same way
(the emitted tokens are the target's draws; a verify would draw again, with
the same distribution). Not modelled: the head's chained entries of accepted
drafts (the dump's are all recomputed), a drafter's catch-up after a switch,
the machine's drift, lookup rounds.

  cd src/worker && PYTHONPATH=. .venv/bin/python ../../scripts/perf/mtp/chooser_sim.py \\
      $PERF_OUT/mtp/dump.json [--ms '{"dflash": 180, "plain": 80, "mtp": [113, 125, 140, 155]}']
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import qwen_worker.config as cfg  # noqa: E402
import qwen_worker.mtp as M  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dump", nargs="+")
ap.add_argument("--ms", default="", help="JSON: dflash, plain (a step inside a burst), mtp [k = 1..], "
                                         "mtp_draft (one shadow draft step), dflash_draft (a shadow block), "
                                         "burst (the host's read between two bursts)")
ap.add_argument("--block", type=int, default=8)
ap.add_argument("--policies", default="plain,dflash,mtp,hybrid,auto,best")
ap.add_argument("--set", default="", help="Chooser constants, e.g. WARM=4,MARGIN=0.1,TAU=1500")
ap.add_argument("--stitch", default="", help="also one request made of these dumps' answers in a row "
                                             "(keys, comma separated): a change of the text's kind")
ap.add_argument("--tokens", type=int, default=0, help="only the first N tokens of each answer")
ap.add_argument("-v", action="store_true")
a = ap.parse_args()

MS = dict(dflash=180.0, plain=80.0, mtp=[113.0, 125.0, 140.0, 155.0], mtp_draft=3.5,
          dflash_draft=18.0, burst=3.0)
docs = [json.load(open(f)) for f in a.dump]
if a.ms:
    MS.update(json.loads(a.ms))
else:
    ck = docs[0].get("clock") or {}
    ck = ck.get("ms") or next((c["ms"] for c in ck.values() if isinstance(c, dict) and c.get("ms")), {})
    MS.update({k: ck[k] for k in ("dflash", "plain") if ck.get(k)})
    for k in range(len(MS["mtp"])):
        if ck.get(str(("mtp", k + 1))):
            MS["mtp"][k] = ck[str(("mtp", k + 1))]


for kv in filter(None, a.set.split(",")):
    k, v = kv.split("=")
    for cls in (M.Chooser, M.MTPDraft):
        if hasattr(cls, k) and not isinstance(getattr(cls, k), (dict, tuple)):
            setattr(cls, k, type(getattr(cls, k))(float(v)))
    if k == "START":
        M.Chooser.START = tuple(v.split("/"))


def prefix(d, t):
    return next((i for i, (x, y) in enumerate(zip(d, t)) if x != y), min(len(d), len(t)))


def load(o):
    """-> (the answer's tokens, DFlash's drafts per anchor, the head's)."""
    toks = [int(t) for t in o["tokens"]]
    D, H = [], []
    for r in o["trace"]:
        sh = dict((x, d) for x, d in r[6])
        D.append(sh.get("dflash") or [])
        H.append(sh.get("mtp") or [])
    if a.tokens:
        toks, D, H = toks[:a.tokens], D[:a.tokens - 1], H[:a.tokens - 1]
    return toks, D, H


class Stats(M.MTPDraft):
    """MTPDraft's estimator alone (next_k / learn / expect), no head."""

    def __init__(self):
        self.fresh()


def clock():
    c = M.Clock({"dflash": MS["dflash"], "plain": MS["plain"]})
    for k, ms in enumerate(MS["mtp"]):
        c.c[("mtp", k + 1)] = ms
    return c


def mtp_ms(k):
    return MS["mtp"][min(k, len(MS["mtp"])) - 1]


def run(toks, D, H, arms, oracle=False):
    """One request's rounds -> (ms, rounds per arm, tokens per arm, shadow
    drafts per arm, the arms in order)."""
    cfg.MTP_K = 0
    ck = clock()
    mt = Stats()
    mt.clock = ck
    ch = M.Chooser(arms, mt, ck) if len(arms) > 1 else None
    sh = M.Shadows()
    sh.emit(toks[0])
    n, i, t = len(toks), 0, 0.0
    took, got, shadows = ({k: 0 for k in arms} for _ in range(3))
    picks = []
    while i < n - 1:
        left = n - 1 - i
        kind, also = arms[0], []
        if ch is not None:
            kind = ch.pick(None, t)
            also = ch.shadows()
        if oracle:                              # the best arm of this round, known in advance
            best = None
            for k in arms:
                if k == "plain":
                    v = 1.0 / MS["plain"]
                elif k == "dflash":
                    v = (1 + prefix(D[i], toks[i + 1:])) / MS["dflash"]
                else:
                    v = max((1 + prefix(H[i][:j], toks[i + 1:])) / mtp_ms(j) for j in range(1, 5))
                if best is None or v > best[0]:
                    best = (v, k)
            kind, also = best[1], []
            ch.cur = kind
        sd = []
        for x in also:
            if x == "mtp":
                d = H[i][:min(mt.next_k(), left)]
                t += MS["mtp_draft"] * len(d)
            else:
                if left + 1 < a.block or not D[i]:
                    continue
                d = D[i]
                t += MS["dflash_draft"]
            sd.append((x, d))
            shadows[x] += 1
        if kind == "plain":
            k = min(1 if oracle or ch is None else ch.steps(), left)
            new = toks[i + 1:i + 1 + k]
            t += MS["plain"] * k + (MS["burst"] if ch is not None else 0.0)   # (alone: the plain loop)
            if ch is not None:
                ch.done(0, 0, True, [x for x, _ in sd], steps=k)
        else:
            if kind == "mtp":
                k = mt.next_k()
                if oracle:
                    k = max(range(1, 5), key=lambda j: (1 + prefix(H[i][:j], toks[i + 1:])) / mtp_ms(j))
                drafts = H[i][:min(k, left)]
            else:
                drafts = D[i][:min(a.block - 1, left)]
            bs = len(drafts) + 1                # (the dump's last positions: shorter drafts)
            acc = min(prefix(drafts, toks[i + 1:]), left)
            new = toks[i + 1:i + 1 + min(acc + 1, left)]
            t += (mtp_ms(bs - 1) if kind == "mtp" else MS["dflash"]) if bs > 1 else MS["plain"]
            if kind == "mtp" and bs > 1:
                mt.record(acc, bs - 1)
            if ch is not None:
                ch.done(acc, bs - 1, kind != "dflash" or bs == a.block, [x for x, _ in sd])
        for x, d in sd:
            sh.add(x, d)
        took[kind] += 1
        got[kind] += len(new)
        picks.append(kind[0] if kind != "plain" else "." )
        for tok in new:
            sh.emit(tok)
        if ch is not None:
            for x, sa, n_ in sh.settle():
                ch.sample(x, sa, n_)
        i += len(new)
    return t, took, got, shadows, "".join(picks)


POL = {"plain": (["plain"], None), "dflash": (["dflash"], None), "mtp": (["mtp"], None),
       "hybrid": (["dflash", "mtp"], None), "auto": (["dflash", "mtp", "plain"], None),
       "auto-d": (["mtp", "plain"], None), "auto-m": (["dflash", "plain"], None),
       "best": (["dflash", "mtp", "plain"], "oracle")}
print("ms:", MS)
items = [(o["key"],) + load(o) for doc in docs for o in doc["dump"]]
if a.stitch:
    got = {k: (t, D, H) for k, t, D, H in items}
    toks, D, H = [], [], []
    for k in a.stitch.split(","):
        t, d, h = got[k]
        toks, D, H = toks + t, D + d + [[]], H + h + [[]]    # (the last token of a part: no drafts)
    items.append(("+".join(a.stitch.split(",")), toks, D[:-1], H[:-1]))
worst = {}
for key, toks, D, H in items:
    n = len(toks)
    aD = [prefix(D[i], toks[i + 1:]) for i in range(len(D))]
    aH = [prefix(H[i], toks[i + 1:]) for i in range(len(H))]
    print(f"\n{key}: {n} tokens; per position: dflash accepts {sum(aD) / len(aD):.2f} of {a.block - 1}, "
          f"mtp {sum(aH) / len(aH):.2f} of 4, first draft right dflash {sum(x > 0 for x in aD) / len(aD):.2f} "
          f"mtp {sum(x > 0 for x in aH) / len(aH):.2f}")
    res = {}
    for name in a.policies.split(","):
        arms, mode = POL[name]
        t, took, got, shadows, picks = run(toks, D, H, arms, oracle=mode == "oracle")
        res[name] = t
        base = res.get("plain") or (n - 1) * MS["plain"]
        print(f"  {name:7s} x{base / t:5.2f} of plain  {1000 * (n - 1) / t:5.1f} tok/s  rounds {took}"
              + (f"  tokens {got}  shadows {shadows}" if len(arms) > 1 else ""))
        if a.v and len(arms) > 1:
            print("          " + picks)
    single = min(res[k] for k in ("plain", "dflash", "mtp") if k in res)
    for name in res:
        if name.startswith(("hybrid", "auto")):
            r = single / res[name]
            worst[name] = min(worst.get(name, 9.0), r)
            print(f"  {name} / best of plain, dflash, mtp: {r:.3f}" + ("" if r >= 0.97 else "  <-- below 0.97"))
if worst:
    print("\nworst cell: " + ", ".join(f"{k} {v:.3f}" for k, v in worst.items()))
