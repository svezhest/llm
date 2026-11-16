"""Offline replay of prompt lookup decoding (QWEN_LOOKUP) on recorded greedy
answers: no model. For each (prompt ids, plain greedy answer) of an
ab_lookup.py --dump-ids file, walk the answer as _spec would: a lookup round
where the index (the worker's own _Lookup) has a match of >= min tokens
leaving >= 2 drafts (accepted = the drafts' common prefix with the answer, +1
target token), else

  plain   a plain step (the no-drafter mode, with _spec's burst detail: a
          match found after a step-ahead token costs one more plain token)
  dflash  a drafter round worth --dflash-tokens tokens at --dflash-ms

and price it with a verify cost per row count (ms; --cost T:ms,... from
ab_lookup.py --oracle-ks, interpolated) -> estimated tok/s per policy.
Greedy answers are fixed, so lookup acceptance is exact (up to the near-tie
divergences of the verify class); the drafter rounds are an average.

  python3 sim.py $PERF_OUT/lookup/ids.json --cost 1:58,3:100,8:120,16:190 --dflash-ms 153 --dflash-tokens 5.8
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src", "worker"))
from qwen_worker.lookup import _Gate, _Lookup  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dump")
ap.add_argument("--cost", default="1:58,3:100,8:120,16:190", help="verify rows T:ms (T=1: a plain step)")
ap.add_argument("--dflash-ms", type=float, default=153.0)
ap.add_argument("--dflash-tokens", type=float, default=5.8)
ap.add_argument("--mins", default="2,3,4,5,6,8")
ap.add_argument("--ks", default="3,5,7,11,15")
ap.add_argument("--k0", type=int, default=0, help="adaptive K: start, doubled (+1) after a full accept")
ap.add_argument("--backoff", type=int, default=0)
ap.add_argument("--dflash-per", default="", help="workload:tokens per drafter round,... (else --dflash-tokens)")
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--nmax", type=int, default=8)
ap.add_argument("--burst", type=int, default=16)
a = ap.parse_args()

pts = sorted((int(t), float(m)) for t, m in (x.split(":") for x in a.cost.split(",")))


def cost(T):
    for (t0, m0), (t1, m1) in zip(pts, pts[1:]):
        if t0 <= T <= t1:
            return m0 + (m1 - m0) * (T - t0) / (t1 - t0)
    return pts[-1][1] if T > pts[-1][0] else pts[0][1]


def walk(ids, out, mn, K, mode, k0=0, dtok=None):
    """-> (ms, lookup rounds, lookup tokens emitted, other units)."""
    lk = _Lookup(ids, a.n, a.nmax)
    e, ms, lr, lt, other, credit = 1, 0.0, 0, 0, 0, 0.0
    lk.add(out[0])
    N = len(out)
    gate = _Gate(k0, K, mn, a.nmax, a.backoff)


    def prop():
        left = N - e
        if left < 2:
            return None
        m, p = lk.propose(gate.size(left))
        return p if gate.ok(m, p) else None

    while e < N:
        p = prop()
        if p is not None:
            acc = 0
            while acc < len(p) and p[acc] == out[e + acc]:
                acc += 1
            k = min(acc + 1, N - e)
            for t in out[e:e + k]:
                lk.add(t)
            e += k
            ms += cost(len(p) + 1)
            lr, lt = lr + 1, lt + k
            gate.after(acc, len(p))
            continue
        if mode == "dflash":
            credit += dtok or a.dflash_tokens   # (fractional tokens per round, carried)
            k = max(1, min(int(credit), N - e))
            credit -= k
            for t in out[e:e + k]:
                lk.add(t)
            e += k
            ms += a.dflash_ms
            other += 1
            continue
        steps = 0                               # a plain burst
        while e < N:
            lk.add(out[e])
            e, ms, steps = e + 1, ms + cost(1), steps + 1
            if e >= N or steps >= a.burst:
                break
            if prop() is not None:
                lk.add(out[e])                  # the queued step's token: the anchor
                e, ms = e + 1, ms + cost(1)
                break
        other += steps
    return ms, lr, lt, other


if a.dump.endswith(".jsonl"):                  # ab_lookup.py --jsonl: the plain runs
    d = [dict(workload=x["workload"], round=x["round"], ids=x["ids"], out=x["tokens"])
         for x in map(json.loads, open(a.dump)) if x["variant"] == "plain"]
else:
    d = json.load(open(a.dump))["dump"]
seen, cases = set(), []
for x in d:                                     # one per workload (rounds repeat the same answer)
    if x["workload"] not in seen:
        seen.add(x["workload"])
        cases.append(x)
mins = [int(x) for x in a.mins.split(",")]
ks = [int(x) for x in a.ks.split(",")]
for mode in ("plain", "dflash"):
    per = {w: float(t) for w, t in (x.split(":") for x in a.dflash_per.split(",") if x)}
    print(f"\n== {mode}: est. speed-up vs {mode} alone (lookup tokens share) ==")
    print("workload   " + " ".join(f"m{mn}/k{K}".rjust(10) for mn in mins for K in ks))
    for x in cases:
        ids, out = x["ids"], x["out"]
        dtok = per.get(x["workload"], a.dflash_tokens)
        base = (cost(1) if mode == "plain" else a.dflash_ms / dtok) * (len(out) - 1)
        row = []
        for mn in mins:
            for K in ks:
                ms, lr, lt, _ = walk(ids, out, mn, K, mode, a.k0, dtok)
                row.append(f"{base / ms:5.2f}({100 * lt / max(1, len(out) - 1):3.0f}%)")
        print(f"{x['workload']:10s} " + " ".join(r.rjust(10) for r in row))
print("\nexact lookup acceptance (m3, k15): per workload accepted drafts / round")
for x in cases:
    ids, out = x["ids"], x["out"]
    ms, lr, lt, _ = walk(ids, out, 3, 15, "plain")
    print(f"{x['workload']:10s} rounds {lr:4d}  tokens/round {lt / max(1, lr):5.2f}  share {100 * lt / max(1, len(out) - 1):3.0f}%"
          f"  answer {len(out)}")
