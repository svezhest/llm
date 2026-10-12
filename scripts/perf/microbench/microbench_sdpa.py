"""GPU microbenchmark: mx.fast.scaled_dot_product_attention at head_dim 256.

Mirrors the full-attention layers of the Qwen3.5-family checkpoints (default:
weights/ornith15-9b-mlx-8bit -> 16 Q heads, 4 KV heads, head_dim 256) during a
chunked prefill: a slab of q_len queries attends to kv_len keys that already sit
in the KV cache, with MLX's lower-right-aligned "causal" mask (exactly what
mlx-vlm's Qwen3_5Attention passes on a cached slab).

Layouts follow the model: queries come from a [B, L, H, D] projection and are
transposed to [B, H, L, D] (non-contiguous); keys/values are a [..., :kv_len, :]
slice of a KVCache-style buffer whose capacity is rounded up to 256 tokens.

Modes (each row of the table):
  default  - mx.fast.scaled_dot_product_attention(..., mask="causal"): whatever
             the installed MLX dispatches (fused or its unfused fallback)
  fused    - same call with force_fused=True (MLX >= 0.32.2 only; "n/a" before)
  unfused  - explicit matmul -> causal mask -> softmax(fp32) -> matmul, i.e. the
             fallback graph written out (calibration, and the reference output)
`maxdiff` is max |out - unfused| over the whole output.

Also (--decode) times q_len 1..16 (decode, DFlash verify blocks) and batched
single-token decode B=1..8, which is where the M1/M2 threadgroup bug of
mlx#4596/#4643 lived (bf16 only: run with --dtype bfloat16 to exercise it).

DO NOT run without the GPU lease. Example:
  <venv>/bin/python -I microbench_sdpa.py --label mlx-0.31.2 --json out_old.json
"""
import argparse
import json
import math
import os
import statistics
import time

import mlx.core as mx

REPO = "/Users/user/Projects/llm"


def text_config(path):
    with open(os.path.join(path, "config.json")) as f:
        c = json.load(f)
    return c.get("text_config", c)


def kv_buffer(n_kv, kv_len, d, dtype, b=1):
    cap = (kv_len + 255) // 256 * 256
    k = mx.random.normal((b, n_kv, cap, d)).astype(dtype)
    v = mx.random.normal((b, n_kv, cap, d)).astype(dtype)
    return k[..., :kv_len, :], v[..., :kv_len, :]


def queries(n_q, q_len, d, dtype, b=1):
    return mx.random.normal((b, q_len, n_q, d)).astype(dtype).transpose(0, 2, 1, 3)


def unfused(q, k, v, scale):
    """The MLX fallback graph, written out (GQA via reshape, causal lower-right)."""
    B, Hq, L, D = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    g = Hq // Hkv
    qg = (q * scale).reshape(B, Hkv, g, L, D)
    s = qg @ k[:, :, None].swapaxes(-1, -2)                       # B,Hkv,g,L,S
    if L > 1:
        qi = mx.arange(S - L, S)[:, None]
        ki = mx.arange(S)[None]
        s = mx.where(qi >= ki, s, mx.array(-float("inf"), s.dtype))
    p = mx.softmax(s.astype(mx.float32), axis=-1).astype(q.dtype)
    return (p @ v[:, :, None]).reshape(B, Hq, L, D)


def has_force_fused():
    try:
        q = mx.zeros((1, 1, 1, 64), mx.float16)
        mx.eval(mx.fast.scaled_dot_product_attention(q, q, q, scale=1.0, force_fused=True))
        return True
    except TypeError:
        return False
    except Exception:  # noqa: BLE001 - exists, but raised for another reason
        return True


def bench(fn, warmup, iters):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts), min(ts)


def run_case(q, k, v, scale, modes, warmup, iters, max_score_gb, ff, allow_big=False):
    B, Hq, L, D = q.shape
    S = k.shape[2]
    # causal over a cached slab: each query sees S-L+i+1 keys
    flops = 4 * B * Hq * D * (L * (S - L) + L * (L + 1) / 2)
    score_gb = B * Hq * L * S * (2 + 4 + 2) / 1e9                 # fp16 s, fp32 softmax, p
    mask = "causal" if L > 1 else None
    ref = None
    rows = []
    for mode in modes:
        if mode == "fused" and not ff:
            rows.append((mode, None, None, None, None, "n/a (no force_fused)"))
            continue
        if mode in ("unfused", "default") and score_gb > max_score_gb and not (
                mode == "default" and allow_big):
            # "default" falls back to the same unfused graph whenever MLX has no
            # fused kernel for the shape (head_dim 256, q_len > 8, q_len != kv_len
            # on M1): 2048 x 120K would need ~30 GB of transient scores.
            rows.append((mode, None, None, None, None, f"skipped (~{score_gb:.1f} GB scores)"))
            continue
        if mode == "default":
            fn = lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        elif mode == "fused":
            fn = lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask,
                                                              force_fused=True)
        else:
            fn = lambda: unfused(q, k, v, scale)
        mx.clear_cache()
        mx.reset_peak_memory()
        base = mx.get_active_memory()
        try:
            med, mn = bench(fn, warmup, iters)
        except Exception as e:  # noqa: BLE001 - e.g. force_fused rejecting a shape
            rows.append((mode, None, None, None, None, f"error: {type(e).__name__}: {e}"[:120]))
            continue
        peak = (mx.get_peak_memory() - base) / 1e9
        out = fn()
        mx.eval(out)
        if mode == "unfused":
            ref = out
        rows.append((mode, med, mn, flops / med / 1e12, peak, out))
    final = []
    for mode, med, mn, tf, peak, out in rows:
        if isinstance(out, mx.array):
            diff = (f"{mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max().item():.2e}"
                    if ref is not None and mode != "unfused" else "")
            final.append((mode, med, mn, tf, peak, diff))
        else:
            final.append((mode, med, mn, tf, peak, out))
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--config", default=f"{REPO}/weights/ornith15-9b-mlx-8bit")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--q-lens", default="512,2048")
    ap.add_argument("--kv-lens", default="8192,32768,122880")
    ap.add_argument("--modes", default="default,fused,unfused")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--max-score-gb", type=float, default=12.0,
                    help="skip the explicit unfused graph above this transient size")
    ap.add_argument("--allow-big", action="store_true",
                    help="run the default dispatch even above --max-score-gb (OOM risk on 32 GB)")
    ap.add_argument("--decode", action="store_true",
                    help="also q_len 1/2/4/8/16 and batched decode B=1/2/4/8")
    ap.add_argument("--json", help="also dump raw rows here")
    a = ap.parse_args()

    tc = text_config(a.config)
    n_q, n_kv = tc["num_attention_heads"], tc["num_key_value_heads"]
    d = tc.get("head_dim") or tc["hidden_size"] // n_q
    dtype = getattr(mx, a.dtype)
    scale = d ** -0.5
    ff = has_force_fused()
    modes = a.modes.split(",")
    dev = mx.metal.device_info() if hasattr(mx.metal, "device_info") else {}
    print(f"# SDPA microbench — {a.label}\n")
    print(f"mlx {mx.__version__}, {dev.get('architecture', '?')}, dtype {a.dtype}, "
          f"Hq={n_q} Hkv={n_kv} D={d} (from {os.path.basename(a.config)}), "
          f"force_fused={'yes' if ff else 'no'}, median of {a.iters}\n")
    print("| label | B | q_len | kv_len | mode | median ms | min ms | TFLOPS | peak GB | maxdiff vs unfused |")
    print("|---|---:|---:|---:|---|---:|---:|---:|---:|---|")
    raw = []

    def emit(b, L, S, rows):
        for mode, med, mn, tf, peak, note in rows:
            if med is None:
                print(f"| {a.label} | {b} | {L} | {S} | {mode} | | | | | {note} |")
            else:
                print(f"| {a.label} | {b} | {L} | {S} | {mode} | {med*1e3:.2f} | {mn*1e3:.2f} | "
                      f"{tf:.2f} | {peak:.2f} | {note} |")
            raw.append(dict(label=a.label, mlx=mx.__version__, dtype=a.dtype, B=b, q_len=L,
                            kv_len=S, mode=mode, median_s=med, min_s=mn, tflops=tf,
                            peak_gb=peak, note=note))

    mx.random.seed(0)
    for S in [int(x) for x in a.kv_lens.split(",")]:
        k, v = kv_buffer(n_kv, S, d, dtype)
        mx.eval(k, v)
        for L in [int(x) for x in a.q_lens.split(",")]:
            q = queries(n_q, L, d, dtype)
            mx.eval(q)
            emit(1, L, S, run_case(q, k, v, scale, modes, a.warmup, a.iters, a.max_score_gb, ff,
                                   a.allow_big))
        if a.decode:
            for L in (1, 2, 4, 8, 16):
                q = queries(n_q, L, d, dtype)
                mx.eval(q)
                emit(1, L, S, run_case(q, k, v, scale, modes, a.warmup, max(a.iters, 20),
                                       a.max_score_gb, ff))
        del k, v
        mx.clear_cache()
    if a.decode:
        for S in [int(x) for x in a.kv_lens.split(",")]:
            for b in (2, 4, 8):
                k, v = kv_buffer(n_kv, S, d, dtype, b=b)
                q = queries(n_q, 1, d, dtype, b=b)
                mx.eval(k, v, q)
                emit(b, 1, S, run_case(q, k, v, scale, ["default", "unfused"], a.warmup,
                                       max(a.iters, 20), a.max_score_gb, ff))
                del k, v
                mx.clear_cache()
    if a.json:
        with open(a.json, "w") as f:
            json.dump(raw, f, indent=1)


if __name__ == "__main__":
    main()
