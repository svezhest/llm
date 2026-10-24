#!/usr/bin/env python3
"""cascade_mma pass-1 tile / chunk sweep (the prefix half alone), synthetic 9B
layout (Hq 16, Hkv 4, D 256), 8 layers per "step" (distinct K / V per layer,
one eval per step), interleaved rounds, medians. Also the error vs the fp32 mx
reference per tile. NEEDS THE GPU LEASE.

  cd src/worker && PYTHONPATH=. .venv/bin/python ../../scripts/perf/cascade_mma/tile_sweep.py
Env: TS_CFG ("8192x8,8192x16,32768x8,32768x16")  TS_TILES (every cascade_mma.TILES
     key for D 256)  TS_CHUNKS ("0,512,1024,2048"; 0 = pick_chunk)  TS_ROUNDS (5)
     TS_STEPS (10)  PERF_OUT (/tmp/perf): cascade_mma_tiles.json
"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.getcwd())
import mlx.core as mx  # noqa: E402

import cascade_mma  # noqa: E402

HQ, HKV, D, L = 16, 4, 256, 8
CFG = os.environ.get("TS_CFG", "8192x8,8192x16,32768x8,32768x16")
TILES = os.environ.get("TS_TILES", "32x16,32x32q,64x16q,64x32q").split(",")
CHUNKS = [int(c) for c in os.environ.get("TS_CHUNKS", "0,512,1024,2048").split(",")]
ROUNDS = int(os.environ.get("TS_ROUNDS", "5"))
STEPS = int(os.environ.get("TS_STEPS", "10"))
OUT = os.path.join(os.environ.get("PERF_OUT", "/tmp/perf"), "cascade_mma_tiles.json")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res = {}
    for spec in CFG.split(","):
        P, B = (int(x) for x in spec.split("x"))
        mx.random.seed(P + B)
        q = mx.random.normal((B, HQ, 1, D)).astype(mx.float16)
        kv = [(mx.random.normal((1, HKV, P + 256, D)).astype(mx.float16) * 0.5,
               mx.random.normal((1, HKV, P + 256, D)).astype(mx.float16) * 0.5) for _ in range(L)]
        mx.eval(q, kv)
        ro, rl = cascade_mma.reference(q, *kv[0], P, 1 / 16)
        flops = L * 4 * B * HQ * D * P
        variants = [(t, c) for t in TILES for c in CHUNKS]
        errs = {}
        for t, c in variants:
            cascade_mma.TILE = t
            o, lse = cascade_mma.prefix_attention(q, *kv[0], P, 1 / 16, chunk=c or None)
            d = o - ro
            errs[(t, c)] = (mx.sqrt(mx.mean(d * d)) / mx.sqrt(mx.mean(ro * ro))).item()
        times = {v: [] for v in variants}
        for r in range(ROUNDS):
            for t, c in (variants if r % 2 == 0 else variants[::-1]):
                cascade_mma.TILE = t

                def step():
                    return [cascade_mma.prefix_attention(q, k, v, P, 1 / 16, chunk=c or None) for k, v in kv]
                mx.eval(step())
                ts = []
                for _ in range(STEPS):
                    t0 = time.perf_counter()
                    mx.eval(step())
                    ts.append(time.perf_counter() - t0)
                times[(t, c)].append(1e3 * statistics.median(ts))
        cascade_mma.TILE = ""
        for v in variants:
            ms = statistics.median(times[v])
            log(f"P={P} B={B} tile {v[0]:7s} chunk {v[1] or 'auto':>4}: {ms:6.2f} ms/8 layers "
                f"({flops / ms / 1e9:.2f} TFLOPS)  rms {errs[v]:.1e}  {[round(x, 2) for x in times[v]]}")
            res[f"{spec}/{v[0]}/{v[1]}"] = dict(ms=ms, all=times[v], tflops=flops / ms / 1e9, rms=errs[v])
        with open(OUT, "w") as f:
            json.dump(res, f, indent=1)
        del kv, q
        mx.clear_cache()


if __name__ == "__main__":
    main()
