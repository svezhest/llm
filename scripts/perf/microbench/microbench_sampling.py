#!/usr/bin/env python3
"""Per-row logits post-processing cost at batch B (vocab 248320, fp16 logits,
no model): what the batched step adds on top of the forward. NEEDS THE GPU
LEASE (small: < 100 MB).

Measured per step, for B in MS_B:
  greedy-rows    argmax(l - logsumexp(l)) per row, concatenated (worker greedy,
                 one row at a time: byte-identical to the B=1 path per row)
  greedy-batch   the same as one (B, V) op
  sample-rows    the worker's make_sampler (T 0.7, top_p 0.95, top_k 20,
                 explicit per-row key) per row
  penalty-rows   presence + frequency penalty per row (_Penalties.__call__)
  bitmask-batch  llguidance.mlx.apply_token_bitmask over (B, V) with one random
                 int32 mask row per request (grammar rows)
Reported: wall ms per step (graph build + eval) and graph-build ms alone (CPU
time the scheduler thread spends; overlapped with the GPU when stepping ahead).

Usage: cd src/worker && uv run python <scratchpad>/design/microbench_sampling.py
       MS_B="1,2,4,8,16,32"  MS_STEPS=50 ; --smoke : CPU, small vocab
"""
import os
import sys
import time

SMOKE = "--smoke" in sys.argv
import mlx.core as mx  # noqa: E402

if SMOKE:
    mx.set_default_device(mx.cpu)

V = 4096 if SMOKE else 248320
BS = [int(x) for x in os.environ.get("MS_B", "1,2,4" if SMOKE else "1,2,4,8,16,32").split(",")]
STEPS = int(os.environ.get("MS_STEPS", "3" if SMOKE else "50"))


def make_sampler(temp, top_p, min_p, top_k, seed):
    """worker.py make_sampler (sampling branch), unchanged."""
    key = mx.random.key(seed)
    ninf = mx.array(-float("inf"), dtype=mx.float32)

    def sampler(logits):
        nonlocal key
        x = logits.astype(mx.float32) / temp
        if min_p > 0:
            p = mx.softmax(x, axis=-1)
            x = mx.where(p < p.max(axis=-1, keepdims=True) * min_p, ninf, x)
        if 0 < top_k < x.shape[-1]:
            kth = mx.topk(x, top_k, axis=-1).min(axis=-1, keepdims=True)
            x = mx.where(x < kth, ninf, x)
        if top_p < 1.0:
            idx = mx.argsort(x, axis=-1)
            xs = mx.take_along_axis(x, idx, axis=-1)
            drop = mx.cumsum(mx.softmax(xs, axis=-1), axis=-1) <= 1 - top_p
            drop[..., -1] = False
            x = mx.put_along_axis(x, idx, mx.where(drop, ninf, xs), axis=-1)
        key, sub = mx.random.split(key)
        return mx.random.categorical(x, key=sub)
    return sampler


def timed(fn, B):
    build = 0.0
    for i in range(STEPS + 3):
        logits = (mx.random.normal((B, V)) * 3).astype(mx.float16)
        mx.eval(logits)
        if i == 3:
            mx.synchronize()
            t = time.perf_counter()
            build = 0.0
        tb = time.perf_counter()
        out = fn(logits)
        build += time.perf_counter() - tb
        mx.eval(out)
    mx.synchronize()
    return (time.perf_counter() - t) / STEPS * 1e3, build / STEPS * 1e3


def main():
    import numpy as np
    try:
        import llguidance.mlx as llmx
    except Exception:  # noqa: BLE001
        llmx = None
    print(f"vocab {V}")
    for B in BS:
        samplers = [make_sampler(0.7, 0.95, 0.0, 20, 1234 + i) for i in range(B)]
        counts = [mx.zeros((V,), dtype=mx.float32) for _ in range(B)]
        mx.eval(counts)
        masks = np.random.randint(-2**31, 2**31 - 1, size=(B, (V + 31) // 32), dtype=np.int64).astype(np.int32)

        def greedy_rows(l):
            return mx.concatenate([mx.argmax(l[i:i + 1] - mx.logsumexp(l[i:i + 1], axis=-1, keepdims=True),
                                             axis=-1) for i in range(B)])

        def greedy_batch(l):
            return mx.argmax(l - mx.logsumexp(l, axis=-1, keepdims=True), axis=-1)

        def sample_rows(l):
            return mx.concatenate([samplers[i](l[i:i + 1]) for i in range(B)])

        def penalty_rows(l):
            out = []
            for i in range(B):
                x = l[i:i + 1].astype(mx.float32)
                x = x - 0.5 * counts[i] - 1.5 * (counts[i] > 0)
                out.append(x)
            return mx.concatenate(out)

        def bitmask_batch(l):
            return llmx.apply_token_bitmask(l, masks)

        cases = [("greedy-rows", greedy_rows), ("greedy-batch", greedy_batch),
                 ("sample-rows", sample_rows), ("penalty-rows", penalty_rows)]
        if llmx is not None and not SMOKE:
            cases.append(("bitmask-batch", bitmask_batch))
        parts = []
        for name, fn in cases:
            ms, build = timed(fn, B)
            parts.append(f"{name} {ms:6.2f} ms (build {build:5.2f})")
        print(f"B={B:2d}: " + " | ".join(parts))
    if SMOKE:
        print("smoke run: code paths OK")


if __name__ == "__main__":
    main()
