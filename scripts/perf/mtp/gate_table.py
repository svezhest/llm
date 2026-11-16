"""The gate's tables from ab_mtp.py JSONs (one or several runs of the same cells).

Per cell (prompt, greedy / sampling "@s") and variant: the median decode tok/s and
the median over the pairs (the same round of the same run: the variants of a pair
ran back to back) of tok/s over plain's. The gate: `--auto` (default "auto") >=
`--pass` (0.97) x the best of the other variants, plain included, by those
ratios. Then, per cell, the arms the chooser took: share of the rounds and of
the tokens (spec_stats["arms"]: dflash / mtp / plain / lookup), shadow drafts
and switches per request.

  python3 scripts/perf/mtp/gate_table.py $PERF_OUT/mtp/gate_*.json
"""
import argparse
import collections
import json
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("json", nargs="+")
ap.add_argument("--auto", default="auto")
ap.add_argument("--pass", dest="ok", type=float, default=0.97)
a = ap.parse_args()

rows = []
for i, f in enumerate(a.json):
    for o in json.load(open(f)).get("speed", []):
        rows.append(dict(o, run=i))
cells, variants = [], []
for o in rows:
    if o["key"] not in cells:
        cells.append(o["key"])
    if o["variant"] not in variants:
        variants.append(o["variant"])
plain = {(o["run"], o["round"], o["key"]): o["tps"] for o in rows if o["variant"] == "plain" and o["tps"]}
med = statistics.median

print("| cell | " + " | ".join(f"{v} tok/s | x plain" for v in variants if v != "plain")
      + " | plain tok/s | pairs | " + a.auto + " / best | |")
print("|---|" + "---|" * (2 * (len(variants) - 1) + 4))
miss = []
for c in cells:
    out, ratio, n = [], {"plain": 1.0}, 0
    for v in variants:
        rs = [o for o in rows if o["key"] == c and o["variant"] == v and o["tps"]]
        if v == "plain":
            continue
        pr = [o["tps"] / plain[(o["run"], o["round"], c)] for o in rs if (o["run"], o["round"], c) in plain]
        if pr:
            ratio[v] = med(pr)
        n = max(n, len(pr))
        out.append(f"{med([o['tps'] for o in rs]):.1f} | {ratio[v]:.2f}" if pr else "- | -")
    pl = [o["tps"] for o in rows if o["key"] == c and o["variant"] == "plain" and o["tps"]]
    best = max((v for v in ratio if v != a.auto), key=ratio.get)
    r = ratio.get(a.auto, 0.0) / ratio[best]
    ok = r >= a.ok
    if not ok:
        miss.append(c)
    print(f"| {c} | " + " | ".join(out) + f" | {med(pl):.1f} | {n}{' (< 2)' if n < 2 else ''} | "
          f"{r:.3f} of {best} | {'pass' if ok else 'MISS'} |")
print(f"\n{len(cells) - len(miss)} of {len(cells)} cells pass"
      + (f"; below {a.ok}: {', '.join(miss)}" if miss else ""))

print(f"\n| cell | requests | rounds: dflash / mtp / plain / lookup | tokens: dflash / mtp / plain / lookup | "
      f"shadow drafts per request (dflash / mtp) | switches |")
print("|---|---|---|---|---|---|")
for c in cells:
    rs = [o for o in rows if o["key"] == c and o["variant"] == a.auto and o.get("arms") is not None]
    if not rs:
        continue
    rd, tk = collections.Counter(), collections.Counter()
    for o in rs:
        for k, (r_, t_) in o["arms"].items():
            rd[k] += r_
            tk[k] += t_
    kinds = ("dflash", "mtp", "plain", "lookup")

    def share(cnt):
        tot = sum(cnt.values()) or 1
        return " / ".join(f"{100.0 * cnt[k] / tot:.0f} %" for k in kinds)
    sh = [o.get("shadows") or {} for o in rs]
    print(f"| {c} | {len(rs)} | {share(rd)} | {share(tk)} | "
          f"{med([s.get('dflash', 0) for s in sh]):.0f} / {med([s.get('mtp', 0) for s in sh]):.0f} | "
          f"{med([o.get('switches', 0) for o in rs]):.0f} |")
