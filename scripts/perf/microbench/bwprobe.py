#!/usr/bin/env python3
"""Achievable GPU read rate on this machine (the 'bwprobe' of
docs/perf/2026-10-16/rattn_microbench.md): a 1 GiB fp16 buffer reduced by
sum and by max (read only), and copied (read + write). Medians over --reps
evaluations, each synced. Use it as the drift probe before / after a
measurement session (the GPU speed drifts +-20-30 % over minutes).
NEEDS THE GPU LEASE.

  src/worker/.venv/bin/python scripts/perf/microbench/bwprobe.py [--gib 1] [--reps 15] [--json out]
"""
import argparse
import json
import statistics
import time

import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--gib", type=float, default=1.0)
ap.add_argument("--reps", type=int, default=15)
ap.add_argument("--json")
a = ap.parse_args()

n = int(a.gib * 2**30 // 2)
x = mx.random.uniform(shape=(n // 4096, 4096)).astype(mx.float16)
mx.eval(x)
nbytes = x.nbytes
ops = {
    "sum": (lambda: x.sum(), nbytes),
    "max": (lambda: x.max(), nbytes),
    "copy": (lambda: x + 0, 2 * nbytes),
}
res = {}
for name, (fn, moved) in ops.items():
    for _ in range(3):
        mx.eval(fn())
    ts = []
    for _ in range(a.reps):
        mx.synchronize()
        t = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t)
    med = statistics.median(ts)
    res[name] = dict(ms=med * 1e3, gbps=moved / med / 1e9, min_ms=min(ts) * 1e3)
    print(f"bwprobe {name:5s} {med * 1e3:7.2f} ms  {moved / med / 1e9:6.1f} GB/s "
          f"(best {moved / min(ts) / 1e9:.1f})", flush=True)
res["when"] = time.strftime("%Y-%m-%d %H:%M:%S")
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
