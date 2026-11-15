"""Mixed-precision 27B (oQ4e) vs the uniform 4-bit 27B: where the linears of
a decode step and of a DFlash verify block go, on the real weights, in ONE
process (no model: the tensors of a few layers per module kind and width +
both lm_heads, straight from the safetensors; the GPU drifts +-30 % between
processes, so the two checkpoints are only compared here, interleaved).

Phases (--phase, comma separated; default all):

  qmv     the solo decode row (B = 1, mx.quantized_matmul = MLX's qmv) per
          (checkpoint, module kind, width pattern): us per call and GB/s; the
          GDN in-projection per layer pattern unfused (4 calls, oQ4e today),
          fused by width groups (the worker since 2026-11-15) and, on the
          uniform model, its one fused call; "forward" = the sum over one
          forward of each checkpoint by counts (+ lm_head). Answers: is a
          5- / 6-bit qmv slower per byte than 4-bit at these shapes (H3/H4),
          what the four GDN launches cost (H1), the bytes (H2).
  verify  a DFlash verify block of 8 tokens (and the drafter's 7-row head):
          per module kind on the path the worker takes today (mrqmm for 4/5
          bits with N >= 512, mlx-vlm's exact kernels for the rest, the 6-bit
          lm_head through its per-token singleton path, the drafter's head
          through quantized_matmul) vs with QWEN_MRQMM6 + width groups, and the
          uniform model; summed per forward.
  lmhead  the 6-bit lm_head alone: qmv1, the singleton path at 8 tokens,
          quantized_matmul at 7 / 8 rows, mrqmm at 7 / 8 rows (+ --sweep
          configs), and the rejected alternative: re-quantized to 8 bits g64
          (qmv1, mrqmm 8 rows, its weight error vs the 6-bit grid, its bytes).
  acc     mrqmm vs an fp32 reference on oQ4e's real modules at each width
          (rows 3 / 8 / 16): max |d| / max |ref| and rms rel, quantized_matmul's
          at the same rows, rows bit-equal at every row count 3..16.

  cd src/worker && PYTHONPATH=. .venv/bin/python ../../scripts/perf/dflash2/oq4e_bench.py \\
      [--a ../../weights/qwen38-27b-mlx-oq4e] [--b ../../weights/qwen38-27b-mlx-4bit] \\
      [--phase qmv,verify,lmhead,acc] [--rounds 5] [--mb 160] [--sweep 2x8,4x8,4x4] \\
      --out $PERF_OUT/oq4e/oq4e_bench.json

Memory: ~--mb MB per (checkpoint, kind, width pattern) + both lm_heads
(~1.1 GB each) + the re-quantized head (1.4 GB): ~8-9 GB at --mb 160.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.getcwd())
import mrqmm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "..", "..", "..", "weights")
ap = argparse.ArgumentParser()
ap.add_argument("--a", default=os.path.join(W, "qwen38-27b-mlx-oq4e"), help="the mixed-precision checkpoint")
ap.add_argument("--b", default=os.path.join(W, "qwen38-27b-mlx-4bit"), help="the uniform one ('' = none)")
ap.add_argument("--phase", default="qmv,verify,lmhead,acc")
ap.add_argument("--rounds", type=int, default=5)
ap.add_argument("--reps", type=int, default=16, help="units per timed chain (>= the copies)")
ap.add_argument("--mb", type=int, default=160, help="distinct weights per (kind, width pattern), MB")
ap.add_argument("--T", type=int, default=8, help="verify block tokens")
ap.add_argument("--sweep", default="2x8,4x8,4x4", help="mrqmm NAxKS configs for the lm_head (lmhead phase)")
ap.add_argument("--out", default=os.path.join(os.environ.get("PERF_OUT", "/tmp/perf"), "oq4e", "oq4e_bench.json"))
a = ap.parse_args()
PH = set(a.phase.split(","))
mx.set_default_device(mx.gpu)
GS, NL = 64, 64
KERNEL = (4, 5, 6, 8)                  # widths mrqmm has a weight side for
GDN_IN = ("qkv", "z", "b", "a")


class Ckpt:
    """Lazy tensors of one checkpoint + its per-module widths (config.json
    "quantization": base + overrides keyed by full module path)."""

    def __init__(self, d, tag):
        self.d, self.tag = d, tag
        cfg = json.load(open(os.path.join(d, "config.json")))
        self.q = cfg["quantization"]
        assert self.q["group_size"] == GS
        self.idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.files = {}

    def bits(self, p):
        v = self.q.get(p)
        return v["bits"] if isinstance(v, dict) else self.q["bits"]

    def has(self, p):
        return p + ".weight" in self.idx

    def tensor(self, name):
        f = self.idx[name]
        if f not in self.files:
            self.files[f] = mx.load(os.path.join(self.d, f))
        return self.files[f][name]

    def linear(self, prefixes):
        """(w, s, b) fp16 scales (the worker's cast), concatenated over prefixes."""
        parts = [(self.tensor(p + ".weight"), self.tensor(p + ".scales").astype(mx.float16),
                  self.tensor(p + ".biases").astype(mx.float16)) for p in prefixes]
        out = tuple(mx.contiguous(mx.concatenate([x[j] for x in parts]) if len(parts) > 1 else parts[0][j])
                    for j in range(3))
        mx.eval(*out)
        return out


def groups_of(names, bits, fuse):
    """The GDN in-projection ops of one layer: 'none' = 4 calls, 'groups' =
    the worker's _gdn_groups (same width >= 2 concatenated, singletons alone)."""
    if fuse == "none":
        return [((n,), b) for n, b in zip(names, bits)]
    by = {}
    for n, b in zip(names, bits):
        by.setdefault(b, []).append(n)
    return [(tuple(ns), b) for b, ns in by.items()]


def layer_units(ck, fuse):
    """{(kind, pattern): [ops of each layer having it]}; an op = (prefixes, bits)."""
    out = {}
    L = "language_model.model.layers.%d."
    for i in range(NL):
        P = L % i
        if ck.has(P + "linear_attn.in_proj_qkv"):
            names = [P + "linear_attn.in_proj_" + n for n in GDN_IN]
            bits = tuple(ck.bits(n) for n in names)
            out.setdefault(("gdn_in", bits), []).append(groups_of(names, bits, fuse))
            p = P + "linear_attn.out_proj"
            out.setdefault(("gdn_out", (ck.bits(p),)), []).append([((p,), ck.bits(p))])
        else:
            for kind, ms in (("attn_q", ["q_proj"]), ("attn_kv", ["k_proj", "v_proj"]), ("attn_o", ["o_proj"])):
                ps = [P + "self_attn." + m for m in ms]
                out.setdefault((kind, tuple(ck.bits(p) for p in ps)), []).append([((p,), ck.bits(p)) for p in ps])
        ps = [P + "mlp.gate_proj", P + "mlp.up_proj"]
        out.setdefault(("mlp_gate_up", tuple(ck.bits(p) for p in ps)), []).append([((p,), ck.bits(p)) for p in ps])
        p = P + "mlp.down_proj"
        out.setdefault(("mlp_down", (ck.bits(p),)), []).append([((p,), ck.bits(p))])
    p = "language_model.lm_head"
    out[("lm_head", (ck.bits(p),))] = [[((p,), ck.bits(p))]]
    return out


def load_units(ck, layers):
    """Weights of up to --mb MB of the layers' units: [[(w, s, b, bits), ...], ...]."""
    got, size = [], 0
    for ops in layers:
        u = []
        for prefixes, bits in ops:
            w, s, b = ck.linear(prefixes)
            u.append((w, s, b, bits))
            size += w.nbytes + s.nbytes + b.nbytes
        got.append(u)
        if size >= a.mb << 20:
            break
    return got, size


def qmm(x, w, s, b, bits):
    return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=GS, bits=bits)


def module(w, s, b, bits):
    lin = nn.QuantizedLinear(w.shape[1] * 32 // bits, w.shape[0], bias=False, group_size=GS, bits=bits)
    lin.weight, lin.scales, lin.biases = w, s, b
    return lin


def timed(units, run, reps):
    """us per unit: a chain over distinct copies (streams from DRAM as a step)."""
    reps = max(reps, len(units))
    ys = [run(units[i % len(units)]) for i in range(reps)]
    mx.eval(ys)
    t = time.perf_counter()
    ys = [run(units[i % len(units)]) for i in range(reps)]
    mx.eval(ys)
    return 1e6 * (time.perf_counter() - t) / reps


def K_of(w, bits):
    return w.shape[1] * 32 // bits


def xs(K, rows):
    key = (K, rows)
    if key not in _X:
        _X[key] = (mx.random.normal((rows, 1, K), key=mx.random.key(K + rows)) * 2).astype(mx.float16)
        mx.eval(_X[key])
    return _X[key]


_X = {}

# --- paths: how one op (w, s, b, bits) runs ----------------------------------

def p_qmv1(op):
    w, s, b, bits = op
    return qmm(xs(K_of(w, bits), 1), w, s, b, bits)


def p_mr(rows):
    def f(op):
        w, s, b, bits = op
        return mrqmm.mrqmm(xs(K_of(w, bits), rows), w, s, b)
    return f


def p_qmmT(rows):
    def f(op):
        w, s, b, bits = op
        return qmm(xs(K_of(w, bits), rows).reshape(1, rows, -1), w, s, b, bits)
    return f


def p_exact(rows):
    """mlx-vlm's verify path for one module: its exact affine kernel (4 / 5 /
    8 bits), else the per-token singleton gather_qmm (6 bits)."""
    from mlx_vlm.models.quantized_verifier import optimized_affine_linear, singleton_quantized_linear

    def f(op):
        w, s, b, bits = op
        lin = _MODS.setdefault(id(w), module(w, s, b, bits))
        xt = xs(K_of(w, bits), rows).reshape(1, rows, -1)
        y = optimized_affine_linear(lin, xt)
        return y if y is not None else singleton_quantized_linear(lin, xt)
    return f


_MODS = {}


def worker_verify(today, rows):
    """The worker's verify path for one op of a unit: mrqmm where it covers it
    (today: 4 / 5 bits; with QWEN_MRQMM6 also 6), N >= 512, else exact."""
    mr, ex = p_mr(rows), p_exact(rows)
    on = (4, 5) if today else (4, 5, 6)

    def f(op):
        w, s, b, bits = op
        return mr(op) if bits in on and w.shape[0] >= mrqmm.MIN_N else ex(op)
    return f


def run_unit(path):
    return lambda u: [path(op) for op in u]


# -----------------------------------------------------------------------------

ckA = Ckpt(a.a, "oq4e")
ckB = Ckpt(a.b, "u4") if a.b else None
res = {"args": vars(a), "mlx": mx.__version__}


def interleave(jobs, rounds):
    """jobs: {name: (units, run)} -> {name: median us}, all jobs per round, the
    order reversed every other round."""
    names = list(jobs)
    samples = {n: [] for n in names}
    for r in range(rounds):
        for n in (names if r % 2 == 0 else names[::-1]):
            units, run = jobs[n]
            samples[n].append(timed(units, run, a.reps if not n.split("|")[1].startswith("lm_head") else 4))
    return {n: statistics.median(v) for n, v in samples.items()}, samples


def unit_bytes(u):
    return sum(w.nbytes + s.nbytes + b.nbytes for w, s, b, _ in u)


if "qmv" in PH or "verify" in PH:
    cases = [("oq4e/none", ckA, "none"), ("oq4e/groups", ckA, "groups")]
    if ckB:
        cases += [("u4/none", ckB, "none"), ("u4/groups", ckB, "groups")]
    loaded = {}                                   # (tag, key, fuse) -> units
    counts = {}
    for case, ck, fuse in cases:
        for key, layers in layer_units(ck, fuse).items():
            if fuse == "groups" and key[0] != "gdn_in":
                continue                          # same ops as "none": timed once
            units, size = load_units(ck, layers)
            loaded[(case, key)] = units
            counts[(case, key)] = len(layers)
            print(f"loaded {case:12s} {key[0]:12s} {key[1]} x{len(layers)}: {len(units)} copies, "
                  f"{size / 2**20:.0f} MB", flush=True)

    def forward_keys(case):
        """(key in `loaded`, count) of one forward of the case."""
        out = []
        for (c, key), n in counts.items():
            if c == case or (case.endswith("/groups") and c == case.split("/")[0] + "/none"
                             and key[0] != "gdn_in"):
                out.append(((c, key), n))
        assert out, case
        return out

if "qmv" in PH:
    jobs = {f"{c}|{k[0]}|{k[1]}": (u, run_unit(p_qmv1)) for (c, k), u in loaded.items()}
    med, samples = interleave(jobs, a.rounds)
    rows = []
    for (c, k), u in loaded.items():
        t = med[f"{c}|{k[0]}|{k[1]}"]
        gbs = unit_bytes(u[0]) / (t * 1e-6) / 1e9
        rows.append(dict(case=c, kind=k[0], bits=list(k[1]), count=counts[(c, k)], us=t, gbps=gbs,
                         calls=len(u[0]), N=[op[0].shape[0] for op in u[0]]))
        print(f"qmv1 {c:12s} {k[0]:12s} {str(k[1]):14s} x{counts[(c, k)]:2d}  {t:8.1f} us/unit  "
              f"{gbs:6.0f} GB/s  calls {len(u[0])} N={[op[0].shape[0] for op in u[0]]}", flush=True)
    fwd = {}
    for case, *_ in cases:
        fwd[case] = sum(n * med[f"{c}|{k[0]}|{k[1]}"] for (c, k), n in forward_keys(case)) / 1e3
        gdn = sum(n * med[f"{c}|{k[0]}|{k[1]}"] for (c, k), n in forward_keys(case) if k[0] == "gdn_in") / 1e3
        print(f"forward linears, 1 row, {case:12s}: {fwd[case]:7.2f} ms (GDN in-projections {gdn:6.2f} ms)")
    if ckB:
        print(f"oq4e / u4 (fused): today {fwd['oq4e/none'] / fwd['u4/groups']:.3f}, "
              f"with width groups {fwd['oq4e/groups'] / fwd['u4/groups']:.3f}; "
              f"u4 unfused / fused {fwd['u4/none'] / fwd['u4/groups']:.3f}")
    res["qmv"] = dict(per_unit=rows, forward_ms=fwd, samples={k: v for k, v in samples.items()})

if "verify" in PH:
    T = a.T
    paths = {"oq4e/none": worker_verify(True, T), "oq4e/groups": worker_verify(False, T),
             "u4/none": worker_verify(False, T), "u4/groups": worker_verify(False, T)}
    jobs = {}
    for case, *_ in cases:
        for (c, k), n in forward_keys(case):
            name = f"{case}@{c}|{k[0]}|{k[1]}"
            jobs[name] = (loaded[(c, k)], run_unit(paths[case]))
    # the drafter's head: the target's lm_head over T - 1 rows (today on oQ4e:
    # quantized_matmul's qmm; with QWEN_MRQMM6 / on the 4-bit model: mrqmm)
    for case, *_ in cases:
        hk = next((c, k) for (c, k), _ in forward_keys(case) if k[0] == "lm_head")
        p = p_qmmT(T - 1) if case == "oq4e/none" else p_mr(T - 1)
        jobs[f"{case}@draft|lm_head|{hk[1][1]}"] = (loaded[hk], run_unit(p))
    med, samples = interleave(jobs, a.rounds)
    tot = {}
    for case, *_ in cases:
        parts = {}
        for (c, k), n in forward_keys(case):
            parts[k[0]] = parts.get(k[0], 0) + n * med[f"{case}@{c}|{k[0]}|{k[1]}"] / 1e3
        parts["drafter_head"] = next(v for kk, v in med.items() if kk.startswith(f"{case}@draft|")) / 1e3
        tot[case] = dict(parts, total=sum(parts.values()))
        print(f"verify block T={T}, {case:12s}: total {tot[case]['total']:7.2f} ms  " +
              "  ".join(f"{k} {v:6.2f}" for k, v in parts.items()), flush=True)
    res["verify"] = dict(T=T, ms=tot, per_job_us=med)

if "lmhead" in PH:
    T = a.T
    p = "language_model.lm_head"
    bits = ckA.bits(p)
    w, s, b = ckA.linear([p])
    K, N = K_of(w, bits), w.shape[0]
    op = (w, s, b, bits)
    # re-quantized to 8 bits g64 (the rejected alternative), in row chunks
    w8, s8, b8, err2, ref2 = [], [], [], 0.0, 0.0
    for lo in range(0, N, 16384):
        deq = mx.dequantize(w[lo:lo + 16384], s[lo:lo + 16384].astype(mx.float32),
                            b[lo:lo + 16384].astype(mx.float32), group_size=GS, bits=bits)
        q8 = mx.quantize(deq.astype(mx.float16), group_size=GS, bits=8)
        back = mx.dequantize(q8[0], q8[1].astype(mx.float32), q8[2].astype(mx.float32), group_size=GS, bits=8)
        err2 += mx.sum((back - deq) ** 2).item()
        ref2 += mx.sum(deq ** 2).item()
        w8.append(q8[0]), s8.append(q8[1].astype(mx.float16)), b8.append(q8[2].astype(mx.float16))
        mx.eval(w8[-1], s8[-1], b8[-1])
    op8 = (mx.concatenate(w8), mx.concatenate(s8), mx.concatenate(b8), 8)
    mx.eval(*op8[:3])
    del w8, s8, b8
    jobs = {"6bit|qmv1": p_qmv1, f"6bit|singleton{T}": p_exact(T), f"6bit|qmm{T - 1}": p_qmmT(T - 1),
            f"6bit|qmm{T}": p_qmmT(T), f"6bit|mrqmm{T - 1}": p_mr(T - 1), f"6bit|mrqmm{T}": p_mr(T),
            "8bit|qmv1": p_qmv1, f"8bit|mrqmm{T}": p_mr(T)}
    for c in (c for c in a.sweep.split(",") if c):
        cfg = tuple(int(v) for v in c.split("x"))
        jobs[f"6bit|mrqmm{T}:{c}"] = (lambda cfg: lambda o: mrqmm.mrqmm(xs(K, T), o[0], o[1], o[2], cfg=cfg))(cfg)
    names = list(jobs)
    samples = {n: [] for n in names}
    for r in range(a.rounds):
        for n in (names if r % 2 == 0 else names[::-1]):
            samples[n].append(timed([[op8 if n.startswith("8bit") else op]], run_unit(jobs[n]), 4))
    med = {n: statistics.median(v) for n, v in samples.items()}
    for n in names:
        print(f"lm_head {n:16s} {med[n] / 1e3:8.2f} ms", flush=True)
    by = {"6bit": sum(t.nbytes for t in op[:3]), "8bit": sum(t.nbytes for t in op8[:3])}
    print(f"lm_head bytes 6-bit {by['6bit'] / 2**30:.3f} GiB, 8-bit {by['8bit'] / 2**30:.3f} GiB; "
          f"re-quantization weight rms rel error {math.sqrt(err2 / ref2):.2e}")
    res["lmhead"] = dict(K=K, N=N, ms={n: v / 1e3 for n, v in med.items()}, bytes=by,
                         requant_rms_rel=math.sqrt(err2 / ref2))
    del op8

if "acc" in PH:
    P = "language_model.model.layers.%d."
    pick = {}                                      # one module per (name, width)
    for i in range(NL):
        for m in ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
                  "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                  "mlp.gate_proj", "mlp.down_proj"):
            p = (P % i) + m
            if ckA.has(p):
                pick.setdefault((m, ckA.bits(p)), [p])
    for i in range(NL):                            # the GDN width groups (z+b+a, qkv+z, b+a)
        names = [(P % i) + "linear_attn.in_proj_" + n for n in GDN_IN]
        if ckA.has(names[0]):
            bits = [ckA.bits(n) for n in names]
            for g, gb in groups_of(names, bits, "groups"):
                if len(g) > 1:
                    pick.setdefault(("group " + "+".join(n.rsplit("_", 1)[1] for n in g), gb), list(g))
    pick[("lm_head", ckA.bits("language_model.lm_head"))] = ["language_model.lm_head"]
    out = []
    for (name, bits), prefixes in sorted(pick.items()):
        w, s, b = ckA.linear(prefixes)
        if bits not in KERNEL:
            continue
        K = K_of(w, bits)
        X = (mx.random.normal((16, 1, K), key=mx.random.key(5)) * 1.0).astype(mx.float16)
        base = mrqmm.mrqmm(X, w, s, b)
        same = all(mx.array_equal(mrqmm.mrqmm(X[:M], w, s, b), base[:M]).item() for M in range(3, 16))
        row = dict(module=name, bits=bits, K=K, N=w.shape[0], rows_equal_across_M=same)
        for M in (3, 8, 16):
            x = X[:M]
            y = mrqmm.mrqmm(x, w, s, b).reshape(M, -1)
            q = qmm(x, w, s, b, bits).reshape(M, -1)
            e2 = qe2 = r2 = 0.0
            emax = qmax = rmax = 0.0
            for lo in range(0, w.shape[0], 32768):
                hi = lo + 32768
                deq = mx.dequantize(w[lo:hi], s[lo:hi].astype(mx.float32), b[lo:hi].astype(mx.float32),
                                    group_size=GS, bits=bits)
                ref = x.reshape(M, K).astype(mx.float32) @ deq.T
                d = y[:, lo:hi].astype(mx.float32) - ref
                dq = q[:, lo:hi].astype(mx.float32) - ref
                e2 += mx.sum(d * d).item()
                qe2 += mx.sum(dq * dq).item()
                r2 += mx.sum(ref * ref).item()
                emax = max(emax, mx.abs(d).max().item())
                qmax = max(qmax, mx.abs(dq).max().item())
                rmax = max(rmax, mx.abs(ref).max().item())
            row[f"rows{M}"] = dict(rms_rel=math.sqrt(e2 / r2), qmm_rms_rel=math.sqrt(qe2 / r2),
                                   max_rel=emax / rmax, qmm_max_rel=qmax / rmax)
        out.append(row)
        print(f"acc {name:22s} {bits}-bit K={K:5d} N={w.shape[0]:6d} rows_equal={same} " + " ".join(
            f"M{M}: {row[f'rows{M}']['rms_rel']:.2e} (qmm {row[f'rows{M}']['qmm_rms_rel']:.2e})"
            for M in (3, 8, 16)), flush=True)
        del w, s, b
        mx.clear_cache()
    res["acc"] = out

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1, default=str)
print("->", a.out)
