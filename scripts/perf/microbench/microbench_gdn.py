"""GPU microbenchmark: Gated DeltaNet recurrence kernels (prefill slabs + decode).

CLAUDE.md: GDN is ~22% of a 9B prefill. Three implementations exist once the
libraries are bumped:
  vlm      - mlx_vlm.models.qwen3_5.gated_delta.gated_delta_kernel: what the
             worker runs today (0.6.2 imports it from mlx-lm 0.31.3; 0.7.x
             vendors the same sequential kernel: one SIMD-group per value row)
  lm       - mlx_lm.models.gated_delta.gated_delta_kernel: in mlx-lm 0.32.0 the
             "packed" kernel (mlx-lm#1559, 8 value rows per SIMD-group,
             MLX_GDN_PACKED=0 disables); in 0.31.3 identical to `vlm`
  fast     - mx.fast.gated_delta_update (mlx 0.32.3, mlx#4020): chunk-parallel
             simdgroup kernel (chunk 8 on non-NAX GPUs) for T >= 16, sequential
             below; "n/a" on older MLX. NOTE: casts g to the q dtype (fp16).
Shapes from the checkpoint (default 9B: Hk=16, Hv=32, Dk=Dv=128), inputs like
the model's: q/k rms-normalised (q scaled by Dk^-1), g = exp(-exp(A_log) *
softplus(a + dt_bias)) in fp32, beta = sigmoid(b), fp32 state.
`dy`/`dstate` = max |difference| against `vlm` on the same inputs.

DO NOT run without the GPU lease. Example:
  <venv>/bin/python -I microbench_gdn.py --label new --json gdn_new.json
"""
import argparse
import json
import os
import statistics
import time

import mlx.core as mx
import mlx.nn as nn

REPO = "/Users/user/Projects/llm"


def cfg(path):
    with open(os.path.join(path, "config.json")) as f:
        c = json.load(f)
    return c.get("text_config", c)


def inputs(B, T, hk, hv, dk, dv, dtype):
    q = mx.random.normal((B, T, hk, dk)).astype(dtype)
    k = mx.random.normal((B, T, hk, dk)).astype(dtype)
    inv = dk ** -0.5
    q = (inv ** 2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv * mx.fast.rms_norm(k, None, 1e-6)
    v = mx.random.normal((B, T, hv, dv)).astype(dtype)
    a = mx.random.normal((B, T, hv)).astype(dtype)
    b = mx.random.normal((B, T, hv)).astype(dtype)
    A_log = mx.log(mx.random.uniform(1, 16, (hv,)))
    dt_bias = mx.ones((hv,), dtype)
    g = mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))
    beta = mx.sigmoid(b)
    state = (mx.random.normal((B, hv, dv, dk)) * 0.1).astype(mx.float32)
    mx.eval(q, k, v, g, beta, state)
    return q, k, v, g, beta, state


def impls():
    out = {}
    from mlx_vlm.models.qwen3_5 import gated_delta as vg
    out["vlm"] = lambda q, k, v, g, be, s: vg.gated_delta_kernel(q, k, v, g, be, s)
    try:
        from mlx_lm.models import gated_delta as lg
        out["lm"] = lambda q, k, v, g, be, s: lg.gated_delta_kernel(q, k, v, g, be, s)
    except ImportError:
        pass
    if hasattr(mx.fast, "gated_delta_update"):
        out["fast"] = lambda q, k, v, g, be, s: mx.fast.gated_delta_update(
            q, k, v, g, be, initial_state=s)
    return out


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
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--config", default=f"{REPO}/weights/ornith15-9b-mlx-8bit")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--cases", default="1x512,1x2048,1x1,2x1,4x1,8x1",
                    help="BxT list; T=1 is a decode step")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--json")
    a = ap.parse_args()
    tc = cfg(a.config)
    hk, hv = tc["linear_num_key_heads"], tc["linear_num_value_heads"]
    dk, dv = tc["linear_key_head_dim"], tc["linear_value_head_dim"]
    n_gdn = tc["num_hidden_layers"] - tc["num_hidden_layers"] // tc.get("full_attention_interval", 4)
    dtype = getattr(mx, a.dtype)
    fns = impls()
    import importlib.metadata as md
    vers = {p: md.version(p) for p in ("mlx", "mlx-lm", "mlx-vlm") if _has(md, p)}
    print(f"# GDN microbench — {a.label}\n")
    print(f"{vers}, Hk={hk} Hv={hv} Dk={dk} Dv={dv}, {n_gdn} GDN layers, dtype {a.dtype}, "
          f"MLX_GDN_PACKED={os.environ.get('MLX_GDN_PACKED', '1')}\n")
    print("| label | B | T | impl | median ms/layer | ms for all GDN layers | tok/s (T*B/time) | dy | dstate |")
    print("|---|---:|---:|---|---:|---:|---:|---|---|")
    raw = []
    mx.random.seed(0)
    for case in a.cases.split(","):
        B, T = (int(x) for x in case.split("x"))
        x = inputs(B, T, hk, hv, dk, dv, dtype)
        ref = fns["vlm"](*x)
        mx.eval(ref)
        for name, fn in fns.items():
            try:
                out = fn(*x)
                mx.eval(out)
                t = bench(lambda: fn(*x), a.warmup, a.iters)
            except Exception as e:  # noqa: BLE001
                print(f"| {a.label} | {B} | {T} | {name} | | | | error: {type(e).__name__}: {str(e)[:80]} | |")
                continue
            dy = mx.abs(out[0].astype(mx.float32) - ref[0].astype(mx.float32)).max().item()
            ds = mx.abs(out[1] - ref[1]).max().item()
            print(f"| {a.label} | {B} | {T} | {name} | {t*1e3:.3f} | {t*1e3*n_gdn:.1f} | "
                  f"{B*T/(t*n_gdn):.0f} | {dy:.2e} | {ds:.2e} |")
            raw.append(dict(label=a.label, versions=vers, B=B, T=T, impl=name, sec=t,
                            dy=dy, dstate=ds))
        mx.clear_cache()
    if a.json:
        with open(a.json, "w") as f:
            json.dump(raw, f, indent=1)


def _has(md, p):
    try:
        md.version(p)
        return True
    except md.PackageNotFoundError:
        return False


if __name__ == "__main__":
    main()
