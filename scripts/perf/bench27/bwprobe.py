#!/usr/bin/env python3
"""Bandwidth probe run in its own process before every benchmark session (the drift
control of the cross-system comparison): the achievable GPU read rate right now.

A 1 GiB fp16 array: a full sum (reads 1 GiB) and a copy (1 GiB read + 1 GiB write),
alternated for --seconds, each op synced; prints/writes the medians in GB/s (1e9).
On this machine (2026-10-16, rattn_microbench.md): sum 339, copy 356 GB/s.

  src/worker/.venv/bin/python scripts/perf/bench27/bwprobe.py --seconds 30 --json probe.json
"""
import argparse
import json
import statistics
import time

import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=30.0)
ap.add_argument("--gib", type=float, default=1.0)
ap.add_argument("--json", default="")
a = ap.parse_args()

n = int(a.gib * 2**30) // 2
x = mx.random.normal((n,), dtype=mx.float16)
mx.eval(x)
nbytes = n * 2
sums, copies = [], []
for _ in range(3):                          # warm-up: kernels, pages
    mx.eval(x.sum())
    mx.eval(x + 0)
t_end = time.perf_counter() + a.seconds
while time.perf_counter() < t_end:
    t = time.perf_counter()
    mx.eval(x.sum())
    sums.append(nbytes / (time.perf_counter() - t) / 1e9)
    t = time.perf_counter()
    y = x + 0
    mx.eval(y)
    copies.append(2 * nbytes / (time.perf_counter() - t) / 1e9)
    del y
res = {"sum_gbs": round(statistics.median(sums), 1), "copy_gbs": round(statistics.median(copies), 1),
       "sum_max": round(max(sums), 1), "sum_min": round(min(sums), 1), "n": len(sums),
       "seconds": a.seconds, "gib": a.gib, "mlx": mx.__version__,
       "device": str(mx.default_device()), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
print(json.dumps(res), flush=True)
if a.json:
    with open(a.json, "w") as f:
        json.dump(res, f, indent=1)
