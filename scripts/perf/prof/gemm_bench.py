"""Prefill GEMM rates on M1 Max: 8-bit qmm (transpose) vs fp16 matmul, by M and shape.
  .venv/bin/python gemm_bench.py"""
import time, statistics, sys, os
import mlx.core as mx
SHAPES = [("gate/up", 4096, 12288), ("down", 12288, 4096), ("gdn qkv", 4096, 8192),
          ("o/z/out", 4096, 4096), ("k/v", 4096, 1024)]
MS = [int(x) for x in os.environ.get("GB_M", "512,1024,2048,4096").split(",")]
REPS = int(os.environ.get("GB_REPS", "5"))
mx.random.seed(0)


def timeit(f, n=8):
    for _ in range(2):
        mx.eval(f())
    ts = []
    for _ in range(REPS):
        mx.synchronize()
        t = time.perf_counter()
        outs = [f() for _ in range(n)]
        mx.eval(outs)
        ts.append((time.perf_counter() - t) / n)
    return statistics.median(ts)


for name, K, N in SHAPES:
    w = (mx.random.normal((N, K)) * 0.02).astype(mx.float16)
    wq, s, b = mx.quantize(w, group_size=64, bits=8)
    s, b = s.astype(mx.float16), b.astype(mx.float16)
    wt = mx.contiguous(w.T)
    mx.eval(w, wq, s, b, wt)
    for M in MS:
        x = (mx.random.normal((1, M, K))).astype(mx.float16)
        mx.eval(x)
        fl = 2.0 * M * N * K
        tq = timeit(lambda: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=8))
        tf = timeit(lambda: x @ wt)
        tft = timeit(lambda: x @ w.T)
        print(f"{name:8s} M={M:5d} K={K:5d} N={N:5d}: qmm {tq*1e3:7.2f} ms {fl/tq/1e12:5.2f} TF | "
              f"fp16 x@W {tf*1e3:7.2f} ms {fl/tf/1e12:5.2f} TF | x@W.T {fl/tft/1e12:5.2f} TF", flush=True)
