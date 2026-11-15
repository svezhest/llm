"""Attention alone at the 27B's shapes (24 query heads, 4 KV heads, head_dim
256), fp16 KV vs 8-bit KV, per call, on random data (no model). What phase 2
(a quantized vector attention kernel) would replace, and what the q8 gap of
ab_kv.py is made of.

Cases, per key length K (--keys):
  decode   1 query: fp16 = mx.fast.scaled_dot_product_attention (the vector
           kernel); q8 = the library's quantized SDPA (two quantized_matmuls,
           fp16 scores).
  verify   a DFlash block of T (--block) queries at the end of K keys:
           fp16-rows = T one-query calls over growing prefixes (the library's
           verify, QWEN_VERIFY_ATT=0); fp16-vec = ceil(T / nq) calls of nq
           queries with a causal mask (QWEN_VERIFY_ATT: the vector kernel,
           nq = 32 // GQA factor = 5 on the 27B); q8-rows = T one-query
           quantized SDPA calls (what q8 DFlash runs: the kernel is fp16-only).
  prefill  a slab of S queries at the end of K keys, per query token:
           fp16 = fused kernel (force_fused) at the fixed slab (--slab);
           q8 = the library's quantized SDPA at the slab the worker gives it
           (QWEN_PREFILL_BUDGET // K, the scores it materializes);
           q8f = mx.dequantize + fused kernel at the fixed slab
           (QWEN_KVQ_PREFILL_FUSED).
Times: median of --reps calls (each synced), after a warm-up. Error: max /
mean |out - ref| on --check queries, ref = fp32 attention over the KV the
variant reads (dequantized for q8) and over the original fp16 KV (the KV
quantization's own error, `vs fp16 KV`). A row per (case, K, variant);
--json writes them.

  cd src/worker && PYTHONPATH=. .venv/bin/python ../../scripts/perf/kvq8/attn_bench.py \\
    --keys 8192 32768 81920 --json $PERF_OUT/kvq8/attn_bench.json
"""
import argparse
import json
import math
import os
import statistics
import time

import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--keys", type=int, nargs="+", default=[8192, 32768, 81920])
ap.add_argument("--heads", type=int, default=24)
ap.add_argument("--kv-heads", type=int, default=4)
ap.add_argument("--dim", type=int, default=256)
ap.add_argument("--bits", type=int, default=8)
ap.add_argument("--block", type=int, default=8, help="DFlash verify block")
ap.add_argument("--nq", type=int, default=0, help="queries per vector-kernel call (0: 32 // GQA factor)")
ap.add_argument("--slab", type=int, default=2048)
ap.add_argument("--budget", type=int, default=int(os.environ.get("QWEN_PREFILL_BUDGET", str(2048 * 32768))))
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--prefill-reps", type=int, default=5)
ap.add_argument("--check", type=int, default=16, help="queries compared against the fp32 reference")
ap.add_argument("--cases", nargs="+", default=["decode", "verify", "prefill"])
ap.add_argument("--json", default="")
ap.add_argument("--cpu", action="store_true", help="smoke test of the script on the CPU (no timing value)")
a = ap.parse_args()
if a.cpu:
    mx.set_default_device(mx.cpu)
FF = {} if a.cpu else {"force_fused": True}     # the CPU has no fused kernel

from mlx_vlm.models.base import quantized_scaled_dot_product_attention as qsdpa  # noqa: E402

H, HKV, D, G = a.heads, a.kv_heads, a.dim, 64
SCALE = 1.0 / math.sqrt(D)
NQ = a.nq or max(1, 32 // (H // HKV))
rows = []


def timed(fn, reps):
    mx.eval(fn())                               # warm-up (kernel compile, buffers)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def quant(x):
    return mx.quantize(x, group_size=G, bits=a.bits)


def deq(t):
    return mx.dequantize(*t, group_size=G, bits=a.bits)


def ref_attn(q, k, v, causal_end):
    """fp32 attention of queries q (1, H, n, D) whose positions are
    causal_end - n .. causal_end - 1 over keys k / v (1, HKV, K, D)."""
    n, K = q.shape[2], k.shape[2]
    rep = H // HKV
    qf = q.astype(mx.float32).reshape(1, HKV, rep, n, D) * SCALE
    kf, vf = k.astype(mx.float32)[:, :, None], v.astype(mx.float32)[:, :, None]
    s = qf @ kf.swapaxes(-1, -2)                                  # (1, HKV, rep, n, K)
    qi = mx.arange(causal_end - n, causal_end)[:, None]
    s = mx.where(qi >= mx.arange(K)[None], s, -mx.inf)
    return (mx.softmax(s, axis=-1) @ vf).reshape(1, H, n, D)


def err(out, ref):
    d = mx.abs(out.astype(mx.float32) - ref)
    return float(d.max().item()), float(d.mean().item())


def row(case, K, variant, sec, per, e_read, e_fp16, **kw):
    r = dict(case=case, keys=K, variant=variant, ms=round(1e3 * sec, 3), per=per,
             err_max=e_read[0], err_mean=e_read[1], err_vs_fp16_max=e_fp16[0], err_vs_fp16_mean=e_fp16[1], **kw)
    rows.append(r)
    print(f"{case:8s} K={K:6d} {variant:9s} {1e3 * sec:9.3f} ms {per:>14s}  err {e_read[0]:.2e}/{e_read[1]:.2e}"
          f"  vs fp16 KV {e_fp16[0]:.2e}/{e_fp16[1]:.2e}"
          + "".join(f"  {k}={v}" for k, v in kw.items()), flush=True)


for K in a.keys:
    mx.random.seed(K)
    k16 = mx.random.normal((1, HKV, K, D)).astype(mx.float16)
    v16 = mx.random.normal((1, HKV, K, D)).astype(mx.float16)
    kq, vq = quant(k16), quant(v16)
    mx.eval(k16, v16, kq, vq)
    kd, vd = deq(kq), deq(vq)
    mx.eval(kd, vd)

    if "decode" in a.cases:
        q = mx.random.normal((1, H, 1, D)).astype(mx.float16)
        mx.eval(q)
        ref16, refq = ref_attn(q, k16, v16, K), ref_attn(q, kd, vd, K)
        f16 = lambda: mx.fast.scaled_dot_product_attention(q, k16, v16, scale=SCALE)  # noqa: E731
        fq = lambda: qsdpa(q * 1, kq, vq, scale=SCALE, mask=None, group_size=G, bits=a.bits)  # noqa: E731
        row("decode", K, "fp16", timed(f16, a.reps), "per call", err(f16(), ref16), err(f16(), ref16))
        row("decode", K, f"q{a.bits}", timed(fq, a.reps), "per call", err(fq(), refq), err(fq(), ref16))

    if "verify" in a.cases:
        T = a.block
        q = mx.random.normal((1, H, T, D)).astype(mx.float16)
        mx.eval(q)
        ref16, refq = ref_attn(q, k16, v16, K), ref_attn(q, kd, vd, K)

        def rows16():
            return mx.concatenate([mx.fast.scaled_dot_product_attention(
                q[:, :, i:i + 1], k16[:, :, :K - T + i + 1], v16[:, :, :K - T + i + 1], scale=SCALE)
                for i in range(T)], axis=2)

        def vec16():
            out = []
            for i in range(0, T, NQ):
                j = min(T, i + NQ)
                end = K - T + j
                out.append(mx.fast.scaled_dot_product_attention(
                    q[:, :, i:j], k16[:, :, :end], v16[:, :, :end], scale=SCALE, mask="causal"))
            return mx.concatenate(out, axis=2)

        def rowsq():
            out = []
            for i in range(T):
                end = K - T + i + 1
                sl = lambda t: tuple(x[:, :, :end] for x in t)  # noqa: E731
                out.append(qsdpa(q[:, :, i:i + 1] * 1, sl(kq), sl(vq), scale=SCALE, mask=None,
                                 group_size=G, bits=a.bits))
            return mx.concatenate(out, axis=2)

        row("verify", K, "fp16-rows", timed(rows16, a.reps), f"block {T}", err(rows16(), ref16),
            err(rows16(), ref16))
        row("verify", K, "fp16-vec", timed(vec16, a.reps), f"block {T}", err(vec16(), ref16),
            err(vec16(), ref16), nq=NQ)
        row("verify", K, f"q{a.bits}-rows", timed(rowsq, a.reps), f"block {T}", err(rowsq(), refq),
            err(rowsq(), ref16))

    if "prefill" in a.cases:
        S = min(a.slab, K)
        Sq = max(1, min(S, a.budget // K))            # the worker's slab over quantized KV (knob off)
        q = mx.random.normal((1, H, S, D)).astype(mx.float16)
        mx.eval(q)
        C = min(a.check, Sq)                          # the last C queries: inside every variant's slab
        qc = q[:, :, S - C:]
        ref16, refq = ref_attn(qc, k16, v16, K), ref_attn(qc, kd, vd, K)
        f16 = lambda: mx.fast.scaled_dot_product_attention(  # noqa: E731
            q, k16, v16, scale=SCALE, mask="causal", **FF)
        fqf = lambda: mx.fast.scaled_dot_product_attention(  # noqa: E731
            q, deq(kq), deq(vq), scale=SCALE, mask="causal", **FF)
        qs = q[:, :, S - Sq:]
        fq = lambda: qsdpa(qs * 1, kq, vq, scale=SCALE, mask="causal", group_size=G, bits=a.bits)  # noqa: E731
        t = timed(f16, a.prefill_reps)
        o = f16()[:, :, S - C:]
        row("prefill", K, "fp16", t / S, f"per token ({S})", err(o, ref16), err(o, ref16), slab=S)
        t = timed(fq, a.prefill_reps)
        o = fq()[:, :, Sq - C:]
        row("prefill", K, f"q{a.bits}", t / Sq, f"per token ({Sq})", err(o, refq), err(o, ref16), slab=Sq)
        mx.reset_peak_memory()
        base = mx.get_active_memory()
        t = timed(fqf, a.prefill_reps)
        peak = mx.get_peak_memory() - base
        o = fqf()[:, :, S - C:]
        row("prefill", K, f"q{a.bits}f", t / S, f"per token ({S})", err(o, refq), err(o, ref16), slab=S,
            transient_mib=round(peak / 2**20))
    del k16, v16, kq, vq, kd, vd
    mx.clear_cache()

if a.json:
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    json.dump(dict(args=vars(a), rows=rows), open(a.json, "w"), indent=1)
