"""The q8 one-pass attention kernels of vatt.py against each other in ONE process (27B layout:
24 q / 4 kv heads, head_dim 256, 8-bit KV in groups of 64): a decode step (one query a head) and
a verify block of T queries, per N keys. Variants (--decode / --verify, comma lists):

  tile         vatt.attention_q8(kernel="tile"): Q8Loader into the fp16 kernel's threadgroup tiles
               (the kernel before 2026-11-17)
  reg:MxWxK    kernel="reg": fragments in registers, M 8-row query blocks a simdgroup, W simdgroups,
               K 8-key fragments a key block (vattq_p1); "reg" alone = vatt's default tile
  regf:MxWxK   kernel="regf": reg with scale and bias taken out of the sums
  <any>@t      the same with VATT_Q8_TIMING = t (cost split, WRONG numbers, errors meaningless):
               1 = no scale / bias, 2 = 1 without the offset subtraction, 3 = no KV reads
  fp16         vatt.attention over the fp16 KV the q8 one was quantized from
  lib          the library's quantized SDPA (one call per query)

The defaults are the whole gate run: the timing split of tile and reg (@0..3), regf, fp16 and lib
at 8K / 32K / 80K / 128K, decode and verify T = 8. Rounds are interleaved over the variants (the
GPU drifts), the order rotating; medians. Each timing evaluates L layers' distinct KV. Errors: rms
relative vs the fp32 reference over mx.dequantize's fp16 KV ("ref": what tile / reg attend) and
over the affine values scale * code + bias in fp32 ("exact": what regf attends); fp16 vs its own
fp32 reference. GB/s = the layer's KV bytes over the time (q8: 1 byte a value + an fp16 scale and
bias per 64). A variant that fails to compile or run is reported and skipped.

  cd src/worker && .venv/bin/python ../../scripts/perf/dflash2/vatt_q8_bench.py --json $PERF_OUT/vatt_q8_reg.json
  # a tile sweep: ... vatt_q8_bench.py 81920 --decode tile,reg:1x2x1,reg:1x4x1,reg:1x2x2,reg:1x1x1 \
  #     --verify tile,reg:2x4x1,reg:3x4x1,reg:2x2x1,reg:2x4x2,reg:1x2x1 --json $PERF_OUT/sweep.json
"""
import json, os, statistics, sys, time
import mlx.core as mx
sys.path.insert(0, os.getcwd())
import vatt  # noqa: E402


def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


T = int(opt("--T", "8"))
ROUNDS = int(opt("--rounds", "9"))
DEC = [x for x in opt("--decode", "lib,tile,tile@1,tile@2,tile@3,reg,reg@1,reg@2,reg@3,regf,fp16").split(",") if x]
VER = [x for x in opt("--verify", "lib,tile,tile@1,tile@2,tile@3,reg,reg@1,reg@2,reg@3,regf,fp16").split(",") if x]
JS = opt("--json", "")
skip = {opt(o, None) for o in ("--T", "--rounds", "--decode", "--verify", "--json")}
NS = [int(x) for x in sys.argv[1:] if not x.startswith("--") and x not in skip] or [8192, 32768, 81920, 131072]
H, HK, D, L = 24, 4, 256, int(os.environ.get("VB_LAYERS", 4))
scale = D ** -0.5
from mlx_vlm.models.base import quantized_scaled_dot_product_attention as qsdpa  # noqa: E402


def rel(a, ref):
    d = a.astype(mx.float32) - ref
    return float(mx.sqrt(mx.mean(d * d)) / mx.sqrt(mx.mean(ref * ref)))


def make(name, N, t):
    """-> f(q, layer) for a variant; q is (1, H, t, D)."""
    if name == "fp16":
        return lambda q, j: vatt.attention(q, Ks[j], Vs[j], N, scale)
    if name == "lib":
        return lambda q, j: mx.concatenate(
            [qsdpa(q[:, :, i:i + 1], tuple(x[..., :N - t + i + 1, :] for x in KQ[j]),
                   tuple(x[..., :N - t + i + 1, :] for x in VQ[j]), scale, None, 64, 8) for i in range(t)], axis=2)
    base, _, timing = name.partition("@")
    kernel, _, tile = base.partition(":")
    if kernel not in vatt.Q8_KERNELS:
        raise SystemExit(f"variant {name}")
    tile = tuple(int(x) for x in tile.split("x")) if tile else None

    def f(q, j):
        if tile:
            vatt.Q8R_DECODE = vatt.Q8R_VERIFY = tile
        return vatt.attention_q8(q, KQ[j], VQ[j], N, scale, kernel=kernel, timing=int(timing or 0))
    return f


R0 = (vatt.Q8R_DECODE, vatt.Q8R_VERIFY)
out = dict(T=T, H=H, HK=HK, layers=L, rounds=ROUNDS, reg_tiles=dict(decode=R0[0], verify=R0[1]), res={})
for N in NS:
    mx.random.seed(0)
    cap = N + 200
    Ks = [mx.random.normal((1, HK, cap, D)).astype(mx.float16) for _ in range(L)]
    Vs = [mx.random.normal((1, HK, cap, D)).astype(mx.float16) for _ in range(L)]
    KQ = [tuple(mx.quantize(k, 64, 8)) for k in Ks]
    VQ = [tuple(mx.quantize(v, 64, 8)) for v in Vs]
    Q = (mx.random.normal((1, H, T, D)) * 4).astype(mx.float16)
    mx.eval(Ks, Vs, KQ, VQ, Q)
    q8b = 2 * HK * N * D * (1 + 4 / 64)
    for kind, names, q in (("decode", DEC, Q[:, :, -1:]), ("verify", VER, Q)):
        t = q.shape[2]
        fns = {n: make(n, N, t) for n in names}
        ref = vatt.reference_q8(q, KQ[0], VQ[0], N, scale)
        refx = vatt.reference_q8(q, KQ[0], VQ[0], N, scale, exact=True)
        ref16 = vatt.reference(q, Ks[0], Vs[0], N, scale)
        mx.eval(ref, refx, ref16)
        err = {}
        for n, f in list(fns.items()):
            vatt.Q8R_DECODE, vatt.Q8R_VERIFY = R0
            try:
                o = f(q, 0)
                mx.eval(o)
                err[n] = (rel(o, ref16),) * 2 if n == "fp16" else (rel(o, ref), rel(o, refx))
            except Exception as e:                      # a kernel that does not compile / run: skipped
                print(f"N={N} {kind} {n} FAILED {str(e)[:2000]}", flush=True)
                out["res"][f"{N}/{kind}/{n}"] = dict(failed=str(e)[:2000])
                fns.pop(n)
        del ref, refx, ref16
        ts = {n: [] for n in fns}
        for r in range(ROUNDS + 1):
            order = list(fns)[r % len(fns):] + list(fns)[:r % len(fns)]
            for n in order:
                vatt.Q8R_DECODE, vatt.Q8R_VERIFY = R0
                t0 = time.perf_counter()
                mx.eval([fns[n](q, j) for j in range(L)])
                ts[n].append((time.perf_counter() - t0) / L * 1e3)
        for n in fns:
            med = statistics.median(ts[n][1:])
            b = 2 * HK * N * D * 2 if n == "fp16" else q8b
            wrong = "@" in n and not n.endswith("@0")
            out["res"][f"{N}/{kind}/{n}"] = dict(ms=med, min_ms=min(ts[n][1:]), gbps=b / med / 1e6,
                                                 rel_ref=None if wrong else err[n][0],
                                                 rel_exact=None if wrong else err[n][1], all_ms=ts[n])
            print(f"N={N:6d} {kind:6s} T={t:2d} {n:13s} {med:7.3f} ms  {b / med / 1e6:6.1f} GB/s  min {min(ts[n][1:]):7.3f}  "
                  + ("(timing variant: numbers wrong by design)" if wrong else
                     f"err vs ref {err[n][0]:.2e}  vs exact {err[n][1]:.2e}"), flush=True)
    del Ks, Vs, KQ, VQ
    mx.clear_cache()
vatt.Q8R_DECODE, vatt.Q8R_VERIFY = R0
if JS:
    os.makedirs(os.path.dirname(JS) or ".", exist_ok=True)
    json.dump(out, open(JS, "w"), indent=1)

# the cost split and the before / after, from the medians
r = out["res"]


def ms(N, kind, n):
    return r.get(f"{N}/{kind}/{n}", {}).get("ms")


for kind in ("decode", "verify"):
    for k in ("tile", "reg"):
        for N in NS:
            t0, t1, t2, t3 = (ms(N, kind, f"{k}@{i}" if i else k) for i in range(4))
            if None not in (t0, t1, t3):
                print(f"split {kind:6s} {k:4s} N={N:6d}: full {t0:.3f} ms; scale / bias loads + fma {t0 - t1:+.3f} "
                      f"({100 * (t0 - t1) / t0:+.1f} % of full: variant 1 is x{t0 / t1:.3f} faster)"
                      + (f"; offset subtraction {t1 - t2:+.3f}" if t2 else "")
                      + f"; packed reads + conversion {t1 - t3:+.3f}; MMAs + stores + merge (no KV reads) {t3:.3f}")
    for N in NS:
        b, a, af, f16 = ms(N, kind, "tile"), ms(N, kind, "reg"), ms(N, kind, "regf"), ms(N, kind, "fp16")
        if b:
            print(f"before / after {kind:6s} N={N:6d}: tile {b:.3f} ms"
                  + (f", reg {a:.3f} (x{b / a:.3f})" if a else "") + (f", regf {af:.3f} (x{b / af:.3f})" if af else "")
                  + (f", fp16 {f16:.3f}" if f16 else ""))
