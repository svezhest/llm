"""GPU microbenchmark: affine quantized_matmul cost vs number of rows M.

Answers "what does one more row cost" for batched decode / speculative verify:
for every linear of one decoder layer of the checkpoint (default
weights/ornith15-9b-mlx-8bit: 8-bit, group 64, hidden 4096) it times
mx.quantized_matmul(x[M, K], W[N, K]^T) for M = 1, 2, 3, ..., 32 and prints the
cost relative to M=1, plus a per-token estimate for a whole decode step
(all layers + lm_head) and the implied aggregate tok/s ceiling of batching M
sequences (linears only: attention / GDN / sampling not included).

Weights are random, quantized with mx.quantize (same bits/group as the
checkpoint) and cast to --dtype like QWEN_FP16=1 does (scales/biases fp16).
Each timed call is a chain of `--chain` matmuls over distinct weight copies
(total > 256 MB, so nothing is served from the SLC), linked by a tiny
dependency so the kernels run back to back as in a forward pass; the time is
per matmul.

--vlm-verify additionally times mlx-vlm's exact speculative-verify kernel
(mlx_vlm.models.quantized_verifier.optimized_affine_linear, mlx-vlm >= 0.7;
bits 4/5/8) on the same shapes: that is what 0.7.x uses for multi-row
verification instead of quantized_matmul.

DO NOT run without the GPU lease. Example:
  <venv>/bin/python -I microbench_qmm.py --label mlx-0.31.2 --json qmm_old.json
"""
import argparse
import json
import os
import statistics
import time

import mlx.core as mx

REPO = "/Users/user/Projects/llm"


def load_cfg(path):
    with open(os.path.join(path, "config.json")) as f:
        c = json.load(f)
    tc = c.get("text_config", c)
    q = c.get("quantization") or c.get("quantization_config") or tc.get("quantization") or {}
    return tc, int(q.get("bits", 8)), int(q.get("group_size", 64))


def layer_shapes(tc):
    """(name, K=in, N=out, count per model) for the dense Qwen3.5 layout."""
    h = tc["hidden_size"]
    nq, nkv = tc["num_attention_heads"], tc["num_key_value_heads"]
    d = tc.get("head_dim") or h // nq
    L = tc["num_hidden_layers"]
    fa = L // tc.get("full_attention_interval", 4)
    gdn = L - fa
    hk, hv = tc["linear_num_key_heads"], tc["linear_num_value_heads"]
    dk, dv = tc["linear_key_head_dim"], tc["linear_value_head_dim"]
    shapes = [
        ("q_proj(+gate)", h, 2 * nq * d, fa),
        ("k_proj", h, nkv * d, fa),
        ("v_proj", h, nkv * d, fa),
        ("o_proj", nq * d, h, fa),
        ("gdn in_proj_qkv", h, 2 * hk * dk + hv * dv, gdn),
        ("gdn in_proj_z", h, hv * dv, gdn),
        ("gdn out_proj", hv * dv, h, gdn),
    ]
    if "intermediate_size" in tc and not tc.get("num_experts"):
        i = tc["intermediate_size"]
        shapes += [("mlp gate_proj", h, i, L), ("mlp up_proj", h, i, L),
                   ("mlp down_proj", i, h, L)]
    shapes.append(("lm_head", h, tc["vocab_size"], 1))
    return shapes


def make_weights(K, N, bits, gs, dtype, copies):
    out = []
    for _ in range(copies):
        w = mx.random.normal((N, K), dtype=dtype) * K ** -0.5
        wq, s, b = mx.quantize(w, group_size=gs, bits=bits)
        out.append((wq, s.astype(dtype), b.astype(dtype)))
        mx.eval(out[-1])
    return out


def bench_chain(fn_list, x, warmup, iters):
    def chain():
        y = x
        outs = []
        for fn in fn_list:
            o = fn(y)
            # tiny dependency: the next matmul waits for this one
            y = x + (o[..., :1].sum() * 0).astype(x.dtype)
            outs.append(o[..., :1])
        return outs
    for _ in range(warmup):
        mx.eval(chain())
    mx.synchronize()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        mx.eval(chain())
        mx.synchronize()
        ts.append((time.perf_counter() - t) / len(fn_list))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--config", default=f"{REPO}/weights/ornith15-9b-mlx-8bit")
    ap.add_argument("--bits", type=int, help="override the checkpoint's bits")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--rows", default="1,2,3,4,5,6,7,8,12,16,32")
    ap.add_argument("--chain", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--only", help="comma list of shape names to keep (substring match)")
    ap.add_argument("--vlm-verify", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()

    tc, bits, gs = load_cfg(a.config)
    bits = a.bits or bits
    dtype = getattr(mx, a.dtype)
    rows = [int(r) for r in a.rows.split(",")]
    shapes = layer_shapes(tc)
    if a.only:
        keep = a.only.split(",")
        shapes = [s for s in shapes if any(k in s[0] for k in keep)]
    verify = None
    if a.vlm_verify:
        try:
            from mlx_vlm.models.quantized_verifier import optimized_affine_linear
            import mlx.nn as nn
            verify = (optimized_affine_linear, nn)
        except ImportError as e:
            print(f"(--vlm-verify unavailable: {e})")
    dev = mx.metal.device_info() if hasattr(mx.metal, "device_info") else {}
    print(f"# quantized_matmul microbench — {a.label}\n")
    print(f"mlx {mx.__version__}, {dev.get('architecture', '?')}, {bits}-bit g{gs} affine, "
          f"x/scales {a.dtype}, config {os.path.basename(a.config)}, per-matmul median of "
          f"{a.iters} x chain {a.chain}\n")
    print("| label | shape | K x N | kernel | " + " | ".join(f"M={m}" for m in rows) + " |")
    print("|---|---|---|---|" + "---:|" * len(rows))
    raw, per_shape = [], {}
    mx.random.seed(0)
    for name, K, N, count in shapes:
        wbytes = N * K * bits / 8 + 2 * N * (K // gs) * 2
        copies = max(2, min(a.chain, int(256e6 // wbytes) + 1))
        ws = make_weights(K, N, bits, gs, dtype, copies)
        fns = [(lambda y, w=w: mx.quantized_matmul(y, w[0], w[1], w[2], transpose=True,
                                                  group_size=gs, bits=bits))
               for w in (ws * (a.chain // copies + 1))[: a.chain]]
        kernels = [("quantized_matmul", fns)]
        if verify is not None:
            f, nn = verify
            mods = []
            for wq, s, b in ws:
                # build tiny, then swap in the real arrays (no 1 GB random init)
                m = nn.QuantizedLinear(gs, 8, bias=False, group_size=gs, bits=bits)
                m.weight, m.scales, m.biases = wq, s, b
                mods.append(m)
            vfns = [(lambda y, m=m: (lambda r: None if r is None else r[0])(f(m, y[None])))
                    for m in (mods * (a.chain // copies + 1))[: a.chain]]
            kernels.append(("vlm exact verify", vfns))
        for kname, fl in kernels:
            ts = {}
            for m in rows:
                x = mx.random.normal((m, K)).astype(dtype)
                mx.eval(x)
                try:
                    if kname != "quantized_matmul" and fl[0](x) is None:
                        raise ValueError("kernel declined this shape")
                    ts[m] = bench_chain(fl, x, a.warmup, a.iters)
                except Exception as e:  # noqa: BLE001
                    ts[m] = None
                    print(f"<!-- {name} {kname} M={m}: {type(e).__name__}: {e} -->")
            base = ts.get(1) or next((t for t in ts.values() if t), None)
            cells = []
            for m in rows:
                t = ts[m]
                cells.append("" if t is None else f"{t*1e6:.0f} µs (×{t/base:.2f})")
            print(f"| {a.label} | {name} | {K}x{N} | {kname} | " + " | ".join(cells) + " |")
            for m in rows:
                raw.append(dict(label=a.label, mlx=mx.__version__, shape=name, K=K, N=N,
                                count=count, kernel=kname, M=m, sec=ts[m],
                                gbps=None if ts[m] is None else wbytes / ts[m] / 1e9))
            per_shape[(name, kname)] = (count, ts)
        del ws
        mx.clear_cache()

    # whole decode step (linears only)
    print("\n| label | kernel | M | linears ms/step | × vs M=1 | aggregate tok/s ceiling (linears) |")
    print("|---|---|---:|---:|---:|---:|")
    for kname in sorted({k for _, k in per_shape}):
        step = {}
        for m in rows:
            tot = 0.0
            for (name, kn), (count, ts) in per_shape.items():
                if kn != kname:
                    continue
                if ts.get(m) is None:
                    tot = None
                    break
                tot += count * ts[m]
            step[m] = tot
        for m in rows:
            if step[m] is None or step.get(1) is None:
                continue
            print(f"| {a.label} | {kname} | {m} | {step[m]*1e3:.1f} | {step[m]/step[1]:.2f} | "
                  f"{m / step[m]:.1f} |")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(raw, f, indent=1)


if __name__ == "__main__":
    main()
