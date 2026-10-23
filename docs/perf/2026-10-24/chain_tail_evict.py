"""What the states CHAIN_TAIL keeps cost under cache pressure: the 2026-10-16
event sim of 8 AppWorld episodes (docs/perf/2026-10-16/evict_sim.py, the
worker's order: MRU among live entries, wanted last, lookup+store clock,
300 s stale) with the states per chain entry measured by chain_tail_sim.py
(appworld, 300-token first message: 5.26 / 4.35 / 3.93 at CHAIN_TAIL 4096 /
2048 / 0) and 51.5 MB per 9B GDN state. Pure Python, ~1 min."""
import os, statistics, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "2026-10-16"))
import evict_sim as es

es.ST = 51.5e6
for gb in (3.0, 4.0, 5.0, 6.0, 8.0):
    row = []
    for tail, nst in ((4096, 5.26), (2048, 4.35), (0, 3.93)):
        es.NST = nst
        rs = [es.run("mru", gb=gb, seed=s, lookup_ticks=True, stale_s=300.0) for s in range(8)]
        row.append((tail, statistics.mean(r["done"] for r in rs),
                    statistics.mean(r["miss_rate"] for r in rs),
                    statistics.mean(r["lat_mean"] for r in rs)))
    print(f"gb={gb:4.1f} " + "  ".join(f"tail={t:4d} done={d:6.0f} miss={m:5.1%} lat={l:5.1f}s"
                                      for t, d, m, l in row))
