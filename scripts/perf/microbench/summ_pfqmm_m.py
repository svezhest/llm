#!/usr/bin/env python3
"""Aggregate microbench_pfqmm_m.py JSON: per (shape, M) TFLOPS of every
variant, then the slab-weighted GEMM time per 32-layer 9B slab (sum over the
shapes of calls per slab x median ms) and speedups vs mlx / vs pfqmm.

Charging rules: a call where MLX runs split-K is charged at MLX's time for
every variant (no variant may replace it bit-identically); `pfqmm` is
charged at MLX's time where the worker would not take it (`pfqmm_use`).
JSON written before 2026-10-17 06:02 (sweep1/2/3, small) ran the old pfqmm
kernel and has `pfqmm_use` = the old rule; later JSON has `pfqmm_use` = the
current pfqmm._use and `old_rule` beside it.

  python3 scripts/perf/microbench/summ_pfqmm_m.py docs/perf/2026-10-17/raw/pfqmm_m_sweep3.json
"""
import collections
import json
import sys


def main(path):
    d = json.load(open(path))["results"]
    by = collections.defaultdict(dict)
    tags = []
    for r in d:
        by[(r["shape"], r["M"])][r["variant"]] = r
        if r["variant"] not in tags:
            tags.append(r["variant"])
    print("shape    M     path   | " + " ".join(f"{t[:12]:>12s}" for t in tags))
    for (s, M), v in by.items():
        print(f"{s:8s} {M:5d} {v['mlx']['mlx_path']:6s} "
              + " ".join(f"{v[t]['tflops']:12.2f}" if t in v else " " * 12 for t in tags))
    print("\nslab-weighted GEMM time (32 layers, split-K calls stay on mlx), speedup vs mlx and vs pfqmm")
    for M in sorted({k[1] for k in by}):
        tot = {}
        for (s, MM), v in by.items():
            if MM != M:
                continue
            for t, r in v.items():
                if t == "pfqmm" and not r["pfqmm_use"]:
                    tt = v["mlx"]["ms"]
                else:
                    tt = r["ms"] if (r["mlx_path"] == "qmm_t" or t == "mlx") else v["mlx"]["ms"]
                tot[t] = tot.get(t, 0) + r["calls"] * tt
        b, p = tot["mlx"], tot.get("pfqmm")
        line = f"M={M:5d} mlx {b:7.1f} ms"
        if p:
            line += f" pfqmm {p:7.1f} ms (x{b / p:.3f}) | " + " ".join(
                f"{t}:{p / x:.3f}" for t, x in sorted(tot.items(), key=lambda kv: kv[1]) if t != "mlx")
        print(line)


if __name__ == "__main__":
    for f in sys.argv[1:]:
        main(f)
