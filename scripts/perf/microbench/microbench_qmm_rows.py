#!/usr/bin/env python3
"""Cost of the decode matmuls vs the number of rows M (= batch size B at
decode), per model, with RANDOM quantized weights of the real shapes (shapes
and bits from weights/<model>/config.json; no weights are loaded). NEEDS THE
GPU LEASE (runs Metal kernels), ~1.5 GB peak.

For every linear of one decoder layer (+ lm_head) it times x(M,1,K) @ W^T
through nn.QuantizedLinear (mx.quantized_matmul: qmv below the M1 Max
get_qmv_batch_limit — 6 rows for D,O > 4096, 10 for <= 4096, 14 for <= 2048 —
qmm_splitk from there) for M in MQ_M, over enough distinct weight copies to
defeat the system-level cache, and reports ms and the cost ratio vs M=1 summed over
a whole forward (layer cost x layer count + lm_head). For 4/5-bit weights it
also times mlx-vlm's multi-row "verify" qmv kernel
(qwen3_5/language.py:_target_verify_quantized_linear, input reshaped to
(1, M, K)) — a candidate faster kernel for small batches on the 27B.
For the MoE model it times SwitchGLU (gather_qmm; sorted path when
M * top_k >= 64, mlx_lm/models/switch_layers.py:181) with random routing.

Usage:
  cd src/worker && uv run python <scratchpad>/design/microbench_qmm_rows.py [model_dir ...]
  (default: the three models under weights/)
  MQ_M="1,2,3,4,6,8,12,16,24,32"  MQ_ITERS=50
  --smoke : CPU, tiny shapes (code-path check only)
"""
import json
import os
import sys
import time

SMOKE = "--smoke" in sys.argv
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

if SMOKE:
    mx.set_default_device(mx.cpu)

REPO = "/Users/user/Projects/llm"
HERE = os.path.dirname(os.path.abspath(__file__))
MS = [int(x) for x in os.environ.get("MQ_M", "1,2,4" if SMOKE else "1,2,3,4,6,8,12,16,24,32").split(",")]
ITERS = int(os.environ.get("MQ_ITERS", "3" if SMOKE else "50"))
DTYPE = mx.float16


def qlinear(k, n, bits, gs=64):
    lin = nn.Linear(k, n, bias=False)
    lin.weight = (mx.random.normal((n, k)) * 0.02).astype(DTYPE)
    q = nn.QuantizedLinear.from_linear(lin, group_size=gs, bits=bits)
    q.scales = q.scales.astype(DTYPE)
    q.biases = q.biases.astype(DTYPE)
    mx.eval(q.parameters())
    return q


def timeit(fn, *args):
    for _ in range(3):
        mx.eval(fn(*args))
    mx.synchronize()
    t = time.perf_counter()
    for _ in range(ITERS):
        out = fn(*args)
        mx.eval(out)
    mx.synchronize()
    return (time.perf_counter() - t) / ITERS * 1e3


SUFFIX = {  # linear name -> module-path suffix in config["quantization"] (UD quants)
    "attn.q_proj(+gate)": "self_attn.q_proj", "attn.k_proj": "self_attn.k_proj",
    "attn.v_proj": "self_attn.v_proj", "attn.o_proj": "self_attn.o_proj",
    "gdn.in_proj fused(qkv,z,b,a)": "linear_attn.in_proj_qkv", "gdn.out_proj": "linear_attn.out_proj",
    "lm_head": "lm_head", "moe.shared gate+up": "shared_expert.gate_proj",
    "moe.shared down": "shared_expert.down_proj", "moe.router": "mlp.gate",
    "moe.switch_glu": "switch_mlp.gate_proj",
}


def bits_for(cfg, lname):
    """Per-module bits of mixed (Unsloth Dynamic) quants: the most common
    value among the modules with that suffix, else the global bits."""
    q = cfg.get("quantization") or {}
    suf = SUFFIX.get(lname)
    found = [v.get("bits") for k, v in q.items() if isinstance(v, dict) and suf and k.endswith(suf)]
    if found:
        return max(set(found), key=found.count)
    return q.get("bits", 8)


def shapes(cfg):
    t = cfg.get("text_config", cfg)
    H, hd = t["hidden_size"], t["head_dim"]
    kd = t["linear_num_key_heads"] * t["linear_key_head_dim"]
    vd = t["linear_num_value_heads"] * t["linear_value_head_dim"]
    nv = t["linear_num_value_heads"]
    L = t["num_hidden_layers"]
    n_fa = L // t.get("full_attention_interval", 4)
    lin = {  # name: (K, N, count over the whole model)
        "attn.q_proj(+gate)": (H, t["num_attention_heads"] * hd * 2, n_fa),
        "attn.k_proj": (H, t["num_key_value_heads"] * hd, n_fa),
        "attn.v_proj": (H, t["num_key_value_heads"] * hd, n_fa),
        "attn.o_proj": (t["num_attention_heads"] * hd, H, n_fa),
        "gdn.in_proj fused(qkv,z,b,a)": (H, 2 * kd + vd + vd + 2 * nv, L - n_fa),
        "gdn.out_proj": (vd, H, L - n_fa),
        "lm_head": (H, t["vocab_size"], 1),
    }
    if t.get("num_experts"):
        si = t["shared_expert_intermediate_size"]
        lin["moe.shared gate+up"] = (H, 2 * si, L)
        lin["moe.shared down"] = (si, H, L)
        lin["moe.router"] = (H, t["num_experts"], L)
    else:
        I = t["intermediate_size"]
        lin["mlp.gate+up"] = (H, 2 * I, L)
        lin["mlp.down"] = (I, H, L)
    return t, lin


def bench_model(path):
    cfg = json.load(open(os.path.join(path, "config.json")))
    t, lin = shapes(cfg)
    gbits = (cfg.get("quantization") or {}).get("bits", 8)
    if SMOKE:
        lin = {k: (128, 256, c) for k, (K, N, c) in list(lin.items())[:3]}
        t = dict(t, num_experts=t.get("num_experts") and 8, num_experts_per_tok=2,
                 moe_intermediate_size=64, hidden_size=128)
    name = os.path.basename(os.path.normpath(path))
    print(f"\n== {name}: {gbits}-bit weights (per-module bits from config.quantization where present)")
    rows = []
    total = {m: 0.0 for m in MS}
    total_v = {}
    for lname, (K, N, count) in lin.items():
        bits = bits_for(cfg, lname)
        # Distinct weight copies (> 256 MB in total) so a small matrix is not
        # served from the 48 MB system-level cache: real decode streams every
        # layer's weights once per step. Time per copy.
        wbytes = K * N * (bits / 8 + 4 / 64)
        ncopy = 1 if SMOKE else max(1, min(32, int(256e6 // wbytes) + 1))
        qs = [qlinear(K, N, bits) for _ in range(ncopy)]
        q = qs[0]

        def chain(a):
            return [l(a) for l in qs]
        base = None
        for m in MS:
            x = (mx.random.normal((m, 1, K)) * 0.1).astype(DTYPE)
            ms = timeit(chain, x) / ncopy
            base = base or ms
            total[m] += ms * count
            r = {"model": name, "linear": lname, "bits": bits, "K": K, "N": N, "M": m, "ms": ms,
                 "ratio_vs_M1": ms / base, "kernel": "quantized_matmul"}
            if bits in (4, 5, 8) and not SMOKE and m <= int(os.environ.get("MQ_VERIFY_MAX_M", "8")):
                try:  # mlx-vlm 0.6.2
                    from mlx_vlm.models.qwen3_5.language import _target_verify_quantized_linear as _vq
                except ImportError:  # mlx-vlm 0.7.x
                    from mlx_vlm.models.quantized_verifier import optimized_affine_linear as _vq
                xv = x.reshape(1, m, K)
                if _vq(q, xv) is not None:
                    r["verify_qmv_ms"] = timeit(lambda a: [_vq(l, a) for l in qs], xv) / ncopy
                    total_v[m] = total_v.get(m, 0.0) + r["verify_qmv_ms"] * count
            rows.append(r)
            extra = f"  verify-qmv {r['verify_qmv_ms']:.3f} ms" if "verify_qmv_ms" in r else ""
            print(f"  {lname:30s} {bits}b K={K:6d} N={N:6d} M={m:3d}: {ms:7.3f} ms  x{ms / base:4.2f}{extra}")
        del q, qs
        mx.clear_cache()
    if t.get("num_experts"):
        from mlx_lm.models.switch_layers import SwitchGLU
        E, k, H, I = t["num_experts"], t["num_experts_per_tok"], t["hidden_size"], t["moe_intermediate_size"]
        sw = SwitchGLU(H, I, E)
        nn.quantize(sw, group_size=64, bits=bits_for(cfg, "moe.switch_glu"))
        sw.apply(lambda a: a.astype(DTYPE) if a.dtype == mx.float32 else a)
        mx.eval(sw.parameters())
        base = None
        L = t["num_hidden_layers"]
        for m in MS:
            x = (mx.random.normal((m, 1, H)) * 0.1).astype(DTYPE)
            idx = mx.stack([mx.random.permutation(E)[:k] for _ in range(m)]).reshape(m, 1, k)
            ms = timeit(sw, x, idx)
            base = base or ms
            total[m] += ms * L
            rows.append({"model": name, "linear": "moe.switch_glu (random routing)", "M": m, "ms": ms,
                         "ratio_vs_M1": ms / base, "kernel": "gather_qmm"})
            print(f"  {'moe.switch_glu random routing':30s} E={E} top{k} M={m:3d}: {ms:7.3f} ms  x{ms / base:4.2f}"
                  f"{'  (sorted path)' if m * k >= 64 else ''}")
    print(f"  -- whole forward (sum of matmuls, counts applied):")
    for m in MS:
        print(f"     M={m:3d}: {total[m]:8.2f} ms  x{total[m] / total[MS[0]]:4.2f}  "
              f"-> matmul-bound aggregate x{m * total[MS[0]] / total[m]:4.2f} vs B=1")
    nlin = len(lin)
    full = [m for m in MS if m in total_v and sum(1 for r in rows if r["M"] == m and "verify_qmv_ms" in r) == nlin]
    if full:
        print("  -- whole forward with the verify-qmv kernel (all linears supported):")
        for m in full:
            print(f"     M={m:3d}: {total_v[m]:8.2f} ms  x{total_v[m] / total[MS[0]]:4.2f} vs qmm M=1  "
                  f"(qmm {total[m]:.2f} ms -> verify/qmm {total_v[m] / total[m]:.2f})")
    return rows


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith("--")] or [
        os.path.join(REPO, "weights", d) for d in
        ("ornith15-9b-mlx-8bit", "qwen38-27b-mlx-5bit", "qwen36-35b-a3b-mlx-ud3")]
    rows = []
    for p in paths:
        rows += bench_model(p)
    if SMOKE:
        print("smoke run: code paths OK, nothing written")
        return
    out = os.path.join(HERE, "..", "results", f"microbench_qmm_rows_{int(time.time())}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=1)
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    main()
