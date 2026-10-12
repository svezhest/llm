#!/usr/bin/env python3
"""Real-model chunked prefill: default SDPA dispatch vs force_fused (mlx >= 0.32.2).

Loads the model like the worker (microbench_decode_batch.load_like_worker) and
prefills N random token ids with the worker's slab loop (prefill_row).
Configs (argv, comma list):
  default      worker policy: slab = clamp(2048, BUDGET // offset, >= 256), default SDPA
  fused        slab fixed 2048 at any offset + force_fused=True for causal slabs (q_len > 8)
               whose kv_len >= PF_FUSED_MIN_KV (16384)
  fusedadapt   worker slab policy + the same force_fused rule (kernel effect only)
Prints wall seconds, tok/s and peak GB per config. NEEDS THE GPU LEASE.

  QWEN_MODEL=... <venv>/bin/python microbench_prefill_fused.py <len> <config>[,<config>...]
"""
import os
import sys
import time

import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import microbench_decode_batch as mbd  # noqa: E402

FUSED_MIN_KV = int(os.environ.get("PF_FUSED_MIN_KV", "16384"))
_orig = mx.fast.scaled_dot_product_attention
_state = {"force": False, "hits": 0}


def _patched(q, k, v, *args, **kw):
    if (_state["force"] and q.shape[2] > 8 and kw.get("mask") == "causal"
            and k.shape[2] >= FUSED_MIN_KV):
        kw["force_fused"] = True
        _state["hits"] += 1
    return _orig(q, k, v, *args, **kw)


mx.fast.scaled_dot_product_attention = _patched


def main():
    n = int(sys.argv[1])
    configs = sys.argv[2].split(",")
    model = mbd.load_like_worker(mbd.MODEL)
    from mlx_vlm.generate.common import generation_stream
    import random
    random.seed(0)
    ids = [random.randrange(1000, 150000) for _ in range(n)]
    budget0 = mbd.PREFILL_BUDGET
    with mx.stream(generation_stream):        # warm-up: JIT the kernels once
        c, _ = mbd.prefill_row(model, ids[:3000])
    mx.synchronize()
    del c
    for cfg in configs:
        _state["force"] = cfg in ("fused", "fusedadapt")
        _state["hits"] = 0
        mbd.PREFILL_BUDGET = 2048 * 10**7 if cfg == "fused" else budget0
        mx.clear_cache()
        mx.reset_peak_memory()
        t = time.perf_counter()
        with mx.stream(generation_stream):
            cache, _ = mbd.prefill_row(model, ids)
        mx.synchronize()
        dt = time.perf_counter() - t
        print(f"mlx {mx.__version__} len {n} {cfg:10s}: {dt:7.1f} s  {n / dt:6.1f} tok/s  "
              f"peak {mx.get_peak_memory() / 1e9:5.2f} GB  fused-calls {_state['hits']}", flush=True)
        del cache
        mx.clear_cache()


if __name__ == "__main__":
    main()
