"""Paired speed ratios of dflash_ab.py JSONs: per (round, task), each variant's
decode tok/s over the baseline variant's of the same (round, task) — the
drift-robust view of an interleaved run. Median and quartiles.

  python scripts/perf/verify_mrqmm/paired.py [--base off/ad] A.json [B.json ...]
"""
import argparse
import json
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="off/ad")
ap.add_argument("files", nargs="+")
a = ap.parse_args()
for f in a.files:
    r = json.load(open(f))
    sp = r["speed"]
    base = {(o["round"], o["task"]): o["tps"] for o in sp if o["variant"] == a.base}
    print(f"{f} (peak {r.get('peak_gb', 0):.2f} GB)")
    for v in r["variants"].split(","):
        rat = [o["tps"] / base[(o["round"], o["task"])] for o in sp if o["variant"] == v]
        q = st.quantiles(rat, n=4)
        print(f"  {v:8s} x{st.median(rat):.3f} (q1 {q[0]:.2f} q3 {q[2]:.2f}) vs {a.base}, paired n={len(rat)}")
