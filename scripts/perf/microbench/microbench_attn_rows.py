#!/usr/bin/env python3
"""Decode attention for B rows, ONE full-attention layer, random K/V (no
weights): what the per-row design costs vs one batched kernel. NEEDS THE GPU
LEASE. Shapes from weights/<model>/config.json (q heads, kv heads, head_dim
256); multiply by the model's full-attention layer count (printed) for a
whole forward.

Variants per (ctx, B):
  rows-fp16   B KVCache objects: update_and_fetch + SDPA per row, concat
              (the proposed design for fp16 KV, e.g. the 9B)
  rows-q8/q4  same with QuantizedKVCache rows (27B q8, 35B q4)
  batch-fp16  one BatchKVCache (merged rows, equal lengths -> no mask, one
              mx.fast.scaled_dot_product_attention call)
  ragged-fp16 rows of different lengths in one left-padded buffer through
              mlx-vlm's ragged Metal kernel (_qwen3_5_ragged_decode_attention)
  masked-fp16 rows of different lengths in a shared right-padded buffer, one
              SDPA call with a boolean (B,1,1,Lmax) mask (a "slot" layout)

Usage:
  cd src/worker && uv run python <scratchpad>/design/microbench_attn_rows.py [model_dir]
  MA_CTX="1024,8192,32768"  MA_B="1,2,4,8,16"  MA_STEPS=30  MA_MEM_GB=3 (per config cap)
  --smoke : CPU, tiny shapes (code-path check only; ragged kernel skipped)
"""
import json
import os
import sys
import time

SMOKE = "--smoke" in sys.argv
import mlx.core as mx  # noqa: E402

if SMOKE:
    mx.set_default_device(mx.cpu)

from mlx_lm.models.base import scaled_dot_product_attention  # noqa: E402
from mlx_vlm.models.cache import BatchKVCache, KVCache  # noqa: E402

REPO = "/Users/user/Projects/llm"
HERE = os.path.dirname(os.path.abspath(__file__))
CTXS = [int(x) for x in os.environ.get("MA_CTX", "64,128" if SMOKE else "1024,8192,32768").split(",")]
BS = [int(x) for x in os.environ.get("MA_B", "1,2,4" if SMOKE else "1,2,4,8,16").split(",")]
STEPS = int(os.environ.get("MA_STEPS", "3" if SMOKE else "30"))
MEM_GB = float(os.environ.get("MA_MEM_GB", "3"))
DT = mx.float16


def kv_row(ctx, nkv, hd):
    c = KVCache()
    k = (mx.random.normal((1, nkv, ctx, hd)) * 0.5).astype(DT)
    v = (mx.random.normal((1, nkv, ctx, hd)) * 0.5).astype(DT)
    c.update_and_fetch(k, v)
    mx.eval(c.keys, c.values)
    return c


def run_rows(rows, nq, nkv, hd, scale):
    B = len(rows)
    t = time.perf_counter()
    build = 0.0
    for _ in range(STEPS):
        tb = time.perf_counter()
        q = (mx.random.normal((B, nq, 1, hd))).astype(DT)
        k = (mx.random.normal((B, nkv, 1, hd))).astype(DT)
        v = (mx.random.normal((B, nkv, 1, hd))).astype(DT)
        outs = []
        for i, c in enumerate(rows):
            kk, vv = c.update_and_fetch(k[i:i + 1], v[i:i + 1])
            outs.append(scaled_dot_product_attention(q[i:i + 1], kk, vv, cache=c, scale=scale, mask=None))
        out = mx.concatenate(outs, axis=0)
        build += time.perf_counter() - tb
        mx.eval(out, [c.state for c in rows])
    mx.synchronize()
    return (time.perf_counter() - t) / STEPS * 1e6, build / STEPS * 1e6


def run_batch(cache, nq, nkv, hd, scale, B):
    t = time.perf_counter()
    for _ in range(STEPS):
        q = (mx.random.normal((B, nq, 1, hd))).astype(DT)
        k = (mx.random.normal((B, nkv, 1, hd))).astype(DT)
        v = (mx.random.normal((B, nkv, 1, hd))).astype(DT)
        kk, vv = cache.update_and_fetch(k, v)
        out = mx.fast.scaled_dot_product_attention(q, kk, vv, scale=scale, mask=None)
        mx.eval(out, kk, vv)
    mx.synchronize()
    return (time.perf_counter() - t) / STEPS * 1e6


def run_ragged(keys, values, pads, nq, hd, scale):
    from mlx_vlm.models.qwen3_5.language import _qwen3_5_ragged_decode_attention
    B = keys.shape[0]
    t = time.perf_counter()
    for _ in range(STEPS):
        q = (mx.random.normal((B, nq, 1, hd))).astype(DT)
        out = _qwen3_5_ragged_decode_attention(q, keys, values, pads, scale)
        if out is None:
            return None
        mx.eval(out)
    mx.synchronize()
    return (time.perf_counter() - t) / STEPS * 1e6


def run_masked(keys, values, lengths, nq, hd, scale):
    B, _, L, _ = keys.shape
    mask = (mx.arange(L)[None, :] < mx.array(lengths)[:, None])[:, None, None, :]
    t = time.perf_counter()
    for _ in range(STEPS):
        q = (mx.random.normal((B, nq, 1, hd))).astype(DT)
        out = mx.fast.scaled_dot_product_attention(q, keys, values, scale=scale, mask=mask)
        mx.eval(out)
    mx.synchronize()
    return (time.perf_counter() - t) / STEPS * 1e6


def main():
    path = next((a for a in sys.argv[1:] if not a.startswith("--")),
                os.path.join(REPO, "weights", "ornith15-9b-mlx-8bit"))
    cfg = json.load(open(os.path.join(path, "config.json")))
    t = cfg.get("text_config", cfg)
    nq, nkv, hd = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    if SMOKE:
        nq, nkv, hd = 4, 2, 64
    n_fa = t["num_hidden_layers"] // t.get("full_attention_interval", 4)
    scale = hd ** -0.5
    name = os.path.basename(os.path.normpath(path))
    print(f"{name}: q_heads {nq}, kv_heads {nkv}, head_dim {hd}; {n_fa} full-attention layers "
          f"(multiply the per-layer numbers by {n_fa})")
    res = []
    for ctx in CTXS:
        for B in BS:
            per_row = ctx * nkv * hd * 2 * 2
            if B * per_row * 2.2 / 1e9 > MEM_GB:
                print(f"ctx {ctx} B {B}: skipped (> MA_MEM_GB)")
                continue
            line = {"model": name, "ctx": ctx, "B": B}
            rows = [kv_row(ctx, nkv, hd) for _ in range(B)]
            line["rows_fp16_us"], line["rows_build_us"] = run_rows(rows, nq, nkv, hd, scale)
            for bits in (8, 4):
                qrows = [c.to_quantized(group_size=64, bits=bits) for c in rows]
                mx.eval([c.state for c in qrows])
                line[f"rows_q{bits}_us"], _ = run_rows(qrows, nq, nkv, hd, scale)
                del qrows
            if B > 1:
                bc = BatchKVCache.merge(rows)
                mx.eval(bc.keys, bc.values)
                line["batch_fp16_us"] = run_batch(bc, nq, nkv, hd, scale, B)
                del bc
                lengths = [ctx - r * max(1, ctx // (4 * B)) for r in range(B)]
                Lm = max(lengths)
                keys = (mx.random.normal((B, nkv, Lm, hd)) * 0.5).astype(DT)
                values = (mx.random.normal((B, nkv, Lm, hd)) * 0.5).astype(DT)
                mx.eval(keys, values)
                if not SMOKE:
                    line["ragged_fp16_us"] = run_ragged(keys, values, [Lm - n for n in lengths], nq, hd, scale)
                line["masked_fp16_us"] = run_masked(keys, values, lengths, nq, hd, scale)
                del keys, values
            del rows
            mx.clear_cache()
            res.append(line)
            print("  ".join(f"{k}={v:.0f}" if isinstance(v, float) else f"{k}={v}" for k, v in line.items()))
    if SMOKE:
        print("smoke run: code paths OK, nothing written")
        return
    out = os.path.join(HERE, "..", "results", f"microbench_attn_rows_{name}_{int(time.time())}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    main()
