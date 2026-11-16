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
plus the cost of the shadow drafts. Not modelled: the head's chained entries
of accepted drafts (the dump's are all recomputed), a drafter's catch-up after
a switch, the machine's drift.

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
ap.add_argument("dump")
ap.add_argument("--ms", default="", help="JSON: dflash, plain (a synchronous step), mtp [k = 1..], "
                                         "step (a plain decode step with step-ahead), mtp_draft (one "
                                         "shadow draft step), dflash_draft (a shadow block)")
ap.add_argument("--block", type=int, default=8)
ap.add_argument("--policies", default="plain,dflash,mtp,hybrid,hybrid+p,best")
ap.add_argument("--set", default="", help="Chooser constants, e.g. WARM=4,MARGIN=0.1")
ap.add_argument("--stitch", default="", help="also one request made of these dumps' answers in a row "
                                             "(keys, comma separated): a change of the text's kind")
ap.add_argument("-v", action="store_true")
a = ap.parse_args()

MS = dict(dflash=180.0, plain=82.0, step=80.0, mtp=[113.0, 125.0, 140.0, 155.0], mtp_draft=3.5,
          dflash_draft=18.0)
doc = json.load(open(a.dump))
if a.ms:
    MS.update(json.loads(a.ms))
elif doc.get("clock", {}).get("ms"):
    ck = doc["clock"]["ms"]
    MS.update({k: ck[k] for k in ("dflash", "plain") if ck.get(k)})
    for k in range(len(MS["mtp"])):
        if ck.get(str(("mtp", k + 1))):
            MS["mtp"][k] = ck[str(("mtp", k + 1))]


for kv in filter(None, a.set.split(",")):
    k, v = kv.split("=")
    setattr(M.Chooser, k, type(getattr(M.Chooser, k))(float(v)))


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


def run(toks, D, H, arms, force=None, oracle=False):
    """One request's rounds -> (ms, rounds per arm, shadow drafts per arm)."""
    M.CLOCK = clock()
    cfg.MTP_K = 0
    mt = Stats()
    ch = M.Chooser(arms, mt) if len(arms) > 1 else None
    sh = M.Shadows()
    sh.emit(toks[0])
    n, i, t = len(toks), 0, 0.0
    took, shadows, picks = {k: 0 for k in arms}, {k: 0 for k in arms}, []
    while i < n - 1:
        left = n - 1 - i
        kind, also = arms[0], []
        if ch is not None:
            kind = ch.pick(force)
            also = ch.shadows()
        if oracle:                              # the best arm of this round, known in advance
            best = None
            for k in arms:
                if k == "plain":
                    v = 1.0 / MS["plain"]
                elif k == "dflash":
                    v = (1 + prefix(D[i], toks[i + 1:])) / MS["dflash"]
                else:
                    v = max((1 + prefix(H[i][:j], toks[i + 1:])) / MS["mtp"][j - 1] for j in range(1, 5))
                if best is None or v > best[0]:
                    best = (v, k)
            kind, also = best[1], []
            ch.cur = kind
        if kind == "plain":
            bs, drafts = 1, []
        elif kind == "mtp":
            k = mt.next_k()
            if oracle:
                k = max(range(1, 5), key=lambda j: (1 + prefix(H[i][:j], toks[i + 1:])) / MS["mtp"][j - 1])
            bs = min(k + 1, left + 1)
            drafts = H[i][:bs - 1]
        else:
            bs = min(a.block, left + 1)
            drafts = D[i][:bs - 1]
        bs = len(drafts) + 1                    # (the dump's last positions: shorter drafts)
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
        acc = min(prefix(drafts, toks[i + 1:]), left)
        new = toks[i + 1:i + 1 + min(acc + 1, left)]
        t += (MS["plain"] if kind == "plain" else MS["mtp"][min(bs, len(MS["mtp"]) + 1) - 2] if kind == "mtp"
              else MS["dflash"]) if bs > 1 or kind == "plain" else MS["plain"]
        if kind == "mtp" and bs > 1:
            mt.record(acc, bs - 1)
        if ch is not None:
            ch.done(acc, bs - 1, kind != "dflash" or bs == a.block, [x for x, _ in sd])
            for x, d in sd:
                sh.add(x, d)
        took[kind] += 1
        picks.append(kind[0])
        for tok in new:
            sh.emit(tok)
        if ch is not None:
            for x, sa, n_ in sh.settle():
                ch.sample(x, sa, n_)
        i += len(new)
    return t, took, shadows, "".join(picks)


POL = {"plain": (["plain"], None), "dflash": (["dflash"], None), "mtp": (["mtp"], None),
       "hybrid": (["dflash", "mtp"], None), "hybrid+p": (["dflash", "mtp", "plain"], None),
       "best": (["dflash", "mtp", "plain"], "oracle")}
print("ms:", MS)
items = [(o["key"],) + load(o) for o in doc["dump"]]
if a.stitch:
    got = {k: (t, D, H) for k, t, D, H in items}
    toks, D, H = [], [], []
    for k in a.stitch.split(","):
        t, d, h = got[k]
        toks, D, H = toks + t, D + d + [[]], H + h + [[]]    # (the last token of a part: no drafts)
    items.append(("+".join(a.stitch.split(",")), toks, D[:-1], H[:-1]))
for key, toks, D, H in items:
    n = len(toks)
    base = (n - 1) * MS["step"]
    aD = [prefix(D[i], toks[i + 1:]) for i in range(len(D))]
    aH = [prefix(H[i], toks[i + 1:]) for i in range(len(H))]
    print(f"\n{key}: {n} tokens; per position: dflash accepts {sum(aD) / len(aD):.2f} of {a.block - 1}, "
          f"mtp {sum(aH) / len(aH):.2f} of 4, first draft right dflash {sum(x > 0 for x in aD) / len(aD):.2f} "
          f"mtp {sum(x > 0 for x in aH) / len(aH):.2f}")
    res = {}
    for name in a.policies.split(","):
        arms, mode = POL[name]
        t, took, shadows, picks = run(toks, D, H, arms, oracle=mode == "oracle")
        res[name] = t
        print(f"  {name:9s} x{base / t:5.2f} of plain  {1000 * (n - 1) / t:5.1f} tok/s  rounds {took}"
              + (f"  shadows {shadows}" if any(shadows.values()) else ""))
        if a.v and len(arms) > 1:
            print("            " + picks)
    single = min(res[k] for k in ("dflash", "mtp") if k in res)
    for name in res:
        if name.startswith("hybrid"):
            print(f"  {name} / best single drafter: {single / res[name]:.3f}")
