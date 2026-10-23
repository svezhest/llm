"""Batch-invariant mode, kernel cost (GPU lease; no model, random 8-bit
weights at the 9B's shapes, ~1.3 GB): microseconds per matmul for rows
1..16, interleaved variants, medians over rounds:

  qmv      mx.quantized_matmul (the default path at 1..2 rows)
  def      mrqmm's default config (3..8 rows 2 x 8, 9..16 rows 4 x 4)
  inv      the batch-invariant config (KS 8; NA 2, 9..16 rows NA
           mrqmm.INVARIANT_NA16)
  inv48    KS 8 with NA 4 at 9..16 rows (bit-identical to inv: NA only
           places outputs; the candidate if it is faster)

and the step's linears (sum x layer counts: q 8, k/v 16, o 8, fused GDN in 24,
GDN out 24, gate/up 64, down 32, lm_head 1) for the default dispatch (qmv at
1..2 rows, def above) vs the mode (inv / inv48 everywhere). Each timed call
is a chain of distinct weight copies (>= 256 MB per shape, the lm_head one
copy) so the weights stream from DRAM as in a decode step.

  cd src/worker && PYTHONPATH=. python ../../scripts/perf/batch_invariant/kern_bench.py \\
      [--rounds 5] [--rows 1,2,3,4,8,9,12,16] --out $PERF_OUT/bi_kern.json
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.getcwd())
import mlx.core as mx  # noqa: E402

import mrqmm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=5)
ap.add_argument("--rows", default="1,2,3,4,8,9,12,16")
ap.add_argument("--reps", type=int, default=32, help="calls per timed chain")
ap.add_argument("--smoke", action="store_true", help="tiny shapes, 1 round: checks the script only")
ap.add_argument("--out", default=os.path.join(os.environ.get("PERF_OUT", "/tmp/perf"), "bi_kern.json"))
a = ap.parse_args()
mx.set_default_device(mx.gpu)

# (name, K, N, per-step count)
SHAPES = [("q_proj", 4096, 8192, 8), ("k/v_proj", 4096, 1024, 16), ("o_proj", 4096, 4096, 8),
          ("gdn_in_fused", 4096, 12352, 24), ("gdn_out", 4096, 4096, 24),
          ("mlp_gate/up", 4096, 12288, 64), ("mlp_down", 12288, 4096, 32),
          ("lm_head", 4096, 248320, 1)]
if a.smoke:
    SHAPES = [(n, 256, 4096 if n == "q_proj" else 512, c) for n, _, _, c in SHAPES]
    a.rounds, a.reps = 1, 2
ROWS = [int(x) for x in a.rows.split(",")]


def weights(N, K, copies):
    out = []
    for c in range(copies):
        mx.random.seed(c)
        w = mx.random.randint(0, 256, (N, K), dtype=mx.uint8).view(mx.uint32)
        s = (mx.random.uniform(shape=(N, K // 64)) * 2e-3 + 1e-4).astype(mx.float16)
        b = (mx.random.normal((N, K // 64)) * 3e-2).astype(mx.float16)
        mx.eval(w, s, b)
        out.append((w, s, b))
    return out


def cfg(K, N, rows, v):
    MB = 1 if rows <= 8 else 2
    if v == "def":
        mrqmm.set_invariant(False)
        c = mrqmm._config(K, N, MB)
    else:
        mrqmm.set_invariant(True)
        c = mrqmm._config(K, N, MB)
        if v == "inv48" and MB == 2 and N >= 4096 and N % 32 == 0:
            c = (4, c[1])
    mrqmm.set_invariant(False)
    return c


def timed(fn, ws, x):
    ys = [fn(x, *ws[i % len(ws)]) for i in range(a.reps)]
    mx.eval(ys)                                     # warm (compile, first touch)
    t = time.perf_counter()
    ys = [fn(x, *ws[i % len(ws)]) for i in range(a.reps)]
    mx.eval(ys)
    return 1e6 * (time.perf_counter() - t) / a.reps


VARIANTS = ["qmv", "def", "inv", "inv48"]
res = {}
for name, K, N, cnt in SHAPES:
    size = N * K * (1 + 4 / 64)
    ws = weights(N, K, 1 if a.smoke else max(1, -(-(256 << 20) // int(size))))
    x16 = (mx.random.normal((16, 1, K)) * 2).astype(mx.float16)
    mx.eval(x16)
    for rows in ROWS:
        x = x16[:rows]
        fns = {"qmv": lambda x, w, s, b: mx.quantized_matmul(x, w, s, b, transpose=True,
                                                             group_size=64, bits=8)}
        for v in ("def", "inv", "inv48"):
            c = cfg(K, N, rows, v)
            fns[v] = (lambda c: lambda x, w, s, b: mrqmm.mrqmm(x, w, s, b, cfg=c))(c)
        samples = {v: [] for v in VARIANTS}
        for r in range(a.rounds):
            for v in (VARIANTS if r % 2 == 0 else VARIANTS[::-1]):
                samples[v].append(timed(fns[v], ws, x))
        res[f"{name}/{rows}"] = {v: statistics.median(s) for v, s in samples.items()}
        print(f"{name:13s} K={K:5d} N={N:6d} rows={rows:2d} " + " ".join(
            f"{v}={res[f'{name}/{rows}'][v]:8.1f}us" for v in VARIANTS), flush=True)
    del ws

step = {}
for rows in ROWS:
    tot = {v: 0.0 for v in VARIANTS}
    for name, K, N, cnt in SHAPES:
        for v in VARIANTS:
            tot[v] += cnt * res[f"{name}/{rows}"][v] / 1e3
    default = tot["qmv"] if rows <= 2 else tot["def"]
    step[rows] = dict(default_ms=default, inv_ms=tot["inv"], inv48_ms=tot["inv48"],
                      inv_x=tot["inv"] / default, inv48_x=tot["inv48"] / default)
    print(f"step linears rows={rows:2d}: default {default:6.2f} ms, inv {tot['inv']:6.2f} "
          f"(x{tot['inv'] / default:.3f}), inv48 {tot['inv48']:6.2f} (x{tot['inv48'] / default:.3f})",
          flush=True)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump({"per_call_us": res, "step_linears": step}, open(a.out, "w"), indent=1)
print("->", a.out)
