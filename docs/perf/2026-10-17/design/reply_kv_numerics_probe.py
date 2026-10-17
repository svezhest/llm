#!/usr/bin/env python3
"""CPU probe for reply_kv.md: one forward of a T-token prefill slab vs T one-token
(decode-shaped) forwards from the same cache, on the tests' tiny random qwen3_5 models
(fp32 dense; 8-bit weights + fp16 scales + QWEN_GDN_FUSE). Prints, per layer, the max
|difference| of the GDN conv / recurrent state and of the full-attention K / V.

    cd src/worker && uv run --group dev python ../../docs/perf/2026-10-17/design/reply_kv_numerics_probe.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "..", "..", "..", "..", "src", "worker")
sys.path.insert(0, os.path.join(WORKER, "tests"))
sys.path.insert(0, WORKER)
import conftest as C  # noqa: E402  (pins MLX to the CPU)
import mlx.core as mx  # noqa: E402
from mlx_vlm.models.cache import make_prompt_cache  # noqa: E402

import worker  # noqa: E402


def feed(lm, ids, steps):
    cache, pos = make_prompt_cache(lm), 0
    for n in steps:
        lm(mx.array([ids[pos:pos + n]]), cache=cache)
        mx.eval([c.state for c in cache])
        pos += n
    return cache


def d(a, b):
    return mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()


for name in ("dense fp32", "8-bit fused"):
    if name.startswith("dense"):
        lm = C.build_lm("dense", None, mx.float32)
    else:
        lm = C.build_lm("dense", 8, mx.float16)
        os.environ["QWEN_MRQMM"] = "0"
        worker._install_gdn_fuse(lm)
    ids = C.prompts([40], 3)[0]
    for T in (1, 2, 5):
        a, b = feed(lm, ids, [32, T]), feed(lm, ids, [32] + [1] * T)
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            if hasattr(x, "cache"):
                out.append(f"L{i} conv {d(x.cache[0], y.cache[0]):.2g} state {d(x.cache[1], y.cache[1]):.2g}")
            else:
                n = x.offset
                out.append(f"L{i} k {d(x.keys[..., :n, :], y.keys[..., :n, :]):.2g} "
                           f"v {d(x.values[..., :n, :], y.values[..., :n, :]):.2g}")
        print(f"{name} T={T}: " + "; ".join(out))
