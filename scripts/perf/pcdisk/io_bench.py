#!/usr/bin/env python3
"""Disk tier (pcdisk.py) write / read speed on entries shaped like the 9B's
(no model): 8 full-attention layers x fp16 K/V (1, 4, T, 256) = 32 KiB/token in
two segments (a 4K "system" piece + the rest), and 5 recurrent states of 24
GDN layers (conv fp16 (1, 3, 8192) + state fp32 (1, 32, 128, 128) = ~50 MiB
each), i.e. what an AppWorld entry holds. Per size and round, interleaved:

  flush_ms   offer + flush on this ("scheduler") thread: the stall a write costs
  write_s    the writer thread's time for the files (GB/s = file bytes / it)
  read_ms    the loader thread's time (files -> numpy -> constant mx.array)
  finish_ms  finish() on this thread: the stall a load costs
  peak_gb    MLX peak during the round (the entry + what the tier adds)
and every loaded array is compared bit for bit with the original (regenerated
from its seed: the original is gone, as after an eviction).

Run from src/worker with the worker's venv (MLX on the GPU, ~2x the largest
size in memory):

    cd src/worker && /Users/user/Projects/llm/src/worker/.venv/bin/python \\
        ../../scripts/perf/pcdisk/io_bench.py --sizes-gb 1,2.5 --rounds 3 \\
        --dir $PERF_OUT/pcdisk_io --out $PERF_OUT/pcdisk_io.json
    (--cpu --sizes-gb 0.4 --rounds 1: a smoke run without the GPU)
"""
import argparse
import json
import os
import shutil
import statistics as st
import sys
import time

sys.path.insert(0, os.getcwd())
import mlx.core as mx  # noqa: E402

import pcdisk  # noqa: E402
import worker  # noqa: E402
from mlx_vlm.models.cache import ArraysCache, KVCache  # noqa: E402

FA, HKV, D = 8, 4, 256
GDN, CONV, VH, DK = 24, 8192, 32, 128
SYS = 4096


def arr(seed, shape, dtype):
    return mx.random.normal(shape, key=mx.random.key(seed)).astype(dtype)


def build(tokens, base):
    """A worker._Entry of `tokens` tokens; every array from seed base + i."""
    n = [base]
    sys_ = min(SYS, tokens // 4)

    def nxt(shape, dtype):
        n[0] += 1
        return arr(n[0], shape, dtype)
    segs, chain = [], []
    for a, b in ((0, sys_), (sys_, tokens)):
        kv = [(nxt((1, HKV, b - a, D), mx.float16), nxt((1, HKV, b - a, D), mx.float16))
              for _ in range(FA)]
        s = worker._Seg(a, b, kv)
        mx.eval(s.arrays)
        segs.append(s)
        chain.append((s, a, b))

    def state():
        st_ = [[nxt((1, 3, CONV), mx.float16), nxt((1, VH, DK, DK), mx.float32)]
               for _ in range(GDN)]
        mx.eval([x for layer in st_ for x in layer])
        return st_
    ckpts = {p: state() for p in sorted({sys_, *(max(sys_ + 1, tokens - k) for k in (3, 2, 1))})}
    end = state()
    layout = []
    for i in range(GDN + FA):
        layout.append(("kv", KVCache) if i % 4 == 3 else ("arr", ArraysCache, 2))
    ids = list(range(tokens))
    return worker._Entry(ids, chain, ckpts, end, layout, False), n[0]


def check(e, base):
    """The loaded entry's arrays == the ones generated from the seeds."""
    n = [base]

    def nxt(shape, dtype):
        n[0] += 1
        return arr(n[0], shape, dtype)
    ok = True
    for s, a, b in e.chain:
        for k, v in s.kv:
            for x in (k, v):
                ok &= mx.array_equal(x, nxt(x.shape, x.dtype)).item()
    for p in sorted(e.ckpts) + [None]:
        st_ = e.end if p is None else e.ckpts[p]
        for layer in st_:
            for x in layer:
                ok &= mx.array_equal(x, nxt(x.shape, x.dtype)).item()
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes-gb", default="1,2.5")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--dir", default=os.path.join(os.environ.get("PERF_OUT", "/tmp/perf"), "pcdisk_io"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--cpu", action="store_true", help="smoke run on the CPU device")
    args = ap.parse_args()
    if args.cpu:
        mx.set_default_device(mx.cpu)
    sizes = [float(x) for x in args.sizes_gb.split(",")]
    shutil.rmtree(args.dir, ignore_errors=True)
    disk = pcdisk.DiskTier(args.dir, 64 * pcdisk.GIB, "io_bench", seg_cls=worker._Seg,
                           entry_cls=worker._Entry, min_tokens=0)
    print(f"dir {args.dir}, F_NOCACHE {pcdisk.NOCACHE}, mlx {mx.__version__}, device {mx.default_device()}")
    rows = []
    for r in range(args.rounds):
        for gb in sizes:
            tokens = max(64, int((gb * 1e9 - 5 * 52e6) / (FA * 2 * HKV * D * 2)))
            mx.clear_cache()
            mx.reset_peak_memory()
            e, last = build(tokens, 1000 * (r * len(sizes) + sizes.index(gb)))
            nbytes = e.nbytes
            ws0 = disk.stats()["write_s"]
            t = time.perf_counter()
            disk.offer(e)
            disk.flush()
            flush_ms = (time.perf_counter() - t) * 1e3
            base = 1000 * (r * len(sizes) + sizes.index(gb))
            del e                                       # evicted: only the views hold it
            assert disk.drain(600)
            st0 = disk.stats()
            write_s = st0["write_s"] - ws0
            d = max(disk.entries.values(), key=lambda x: x.used)
            mx.clear_cache()
            job = disk.load(d, [])
            job.done.wait(600)
            assert disk.drain(600) and disk.done() == [job]
            t = time.perf_counter()
            got = disk.finish(job)
            finish_ms = (time.perf_counter() - t) * 1e3
            ok = check(got, base)
            peak = mx.get_peak_memory() / 1e9
            del got
            with disk.lock:
                paths = disk._drop(d)
            disk._unlink(paths)
            assert disk.drain(600)
            row = {"round": r, "size_gb": gb, "tokens": tokens, "entry_gb": nbytes / 1e9,
                   "flush_ms": flush_ms, "write_s": write_s, "write_gbps": nbytes / 1e9 / write_s,
                   "read_ms": job.ms, "read_gbps": job.nbytes / 1e9 / (job.ms / 1e3),
                   "finish_ms": finish_ms, "peak_gb": peak, "bit_exact": ok}
            rows.append(row)
            print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in row.items()), flush=True)
    print("\nmedians over rounds:")
    print("| entry GB | flush ms | write s | write GB/s | read ms | read GB/s | finish ms | peak GB | exact |")
    print("|---|---|---|---|---|---|---|---|---|")
    summary = []
    for gb in sizes:
        xs = [x for x in rows if x["size_gb"] == gb]
        m = {k: st.median(x[k] for x in xs) for k in
             ("entry_gb", "flush_ms", "write_s", "write_gbps", "read_ms", "read_gbps", "finish_ms",
              "peak_gb")}
        m["exact"] = all(x["bit_exact"] for x in xs)
        summary.append(m)
        print(f"| {m['entry_gb']:.2f} | {m['flush_ms']:.1f} | {m['write_s']:.2f} | {m['write_gbps']:.2f} | "
              f"{m['read_ms']:.0f} | {m['read_gbps']:.2f} | {m['finish_ms']:.1f} | {m['peak_gb']:.2f} | "
              f"{m['exact']} |")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"rows": rows, "summary": summary, "nocache": pcdisk.NOCACHE}, f, indent=1)
    shutil.rmtree(args.dir, ignore_errors=True)
    if not all(x["bit_exact"] for x in rows):
        sys.exit("FAIL: a loaded entry differs from the original")


if __name__ == "__main__":
    main()
