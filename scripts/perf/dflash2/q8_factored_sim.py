"""CPU numerics of attention over 8-bit KV with scale and bias taken out of the sums
(scores: q . k = sum_g s[k,g] (q_g . w[k,g]) + b[k,g] sum(q_g); values likewise), one kv head,
G query rows, N keys, random normal K / V quantized by mx.quantize (8 bits, groups of 64):

  * how far the affine values s * w + b (what the factored form attends) are from their fp16
    roundings, in the attention output: mx.dequantize's values on this device (the CPU: product and
    sum rounded separately) and the affine value rounded once (one fp16 fma: what the tile / reg
    kernels put in their fragments);
  * the cancellation error of the factored sums in fp32 for a code offset (+1024 = the 0x6400
    trick's value as it is, 0 = the code, -128 = centred), against fp64 over the affine values.

  cd src/worker && .venv/bin/python ../../scripts/perf/dflash2/q8_factored_sim.py [N] [|q| multiplier]
"""
import sys
import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
QM = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
D, G = 256, 6
mx.random.seed(0)
k = mx.random.normal((1, 1, N, D)).astype(mx.float16)
v = mx.random.normal((1, 1, N, D)).astype(mx.float16)
qq = np.array((mx.random.normal((G, D)) * QM).astype(mx.float16).astype(mx.float32)).astype(np.float64)
kq, vq = mx.quantize(k, 64, 8), mx.quantize(v, 64, 8)


def f64(a):
    return np.array(a.astype(mx.float32))[0, 0].astype(np.float64)


def parts(x):
    codes = np.array(x[0])[0, 0].view(np.uint8).reshape(N, D).astype(np.float64)   # lowest byte first
    return codes, f64(x[1]), f64(x[2])


kd, vd = f64(mx.dequantize(*kq, 64, 8)), f64(mx.dequantize(*vq, 64, 8))
kw, ks, kb = parts(kq)
vw, vs, vb = parts(vq)
kx = kw * np.repeat(ks, 64, 1) + np.repeat(kb, 64, 1)
vx = vw * np.repeat(vs, 64, 1) + np.repeat(vb, 64, 1)
sc = D ** -0.5


def att(K, V):
    s = (qq * sc) @ K.T
    p = np.exp(s - s.max(-1, keepdims=True))
    return (p / p.sum(-1, keepdims=True)) @ V


def rr(a, ref):
    return float(np.sqrt(np.mean((a - ref) ** 2)) / np.sqrt(np.mean(ref ** 2)))


def h(a):
    return a.astype(np.float16).astype(np.float64)


o16, ox = att(kd, vd), att(kx, vx)
print(f"N={N} |q| x{QM}: K values, mx.dequantize (fp16) vs affine: rms {np.sqrt(np.mean((kd - kx) ** 2)):.2e}; "
      f"8-bit quantization error rms {np.sqrt(np.mean((kd - f64(k)) ** 2)):.2e}")
print(f"  attention over the affine values vs over mx.dequantize's (fp64 both): rel rms {rr(ox, o16):.2e} "
      f"(K alone {rr(att(kx, vd), o16):.2e}, V alone {rr(att(kd, vx), o16):.2e})")
k1, v1 = h(kx), h(vx)                                              # one rounding: an fp16 fma
o1 = att(k1, v1)
print(f"  ... vs over the affine values rounded to fp16 once (fma): K values rms {np.sqrt(np.mean((k1 - kx) ** 2)):.2e}; "
      f"attention rel rms {rr(ox, o1):.2e} (K alone {rr(att(kx, v1), o1):.2e}, V alone {rr(att(k1, vx), o1):.2e}); "
      f"with the output rounded to fp16 {rr(h(ox), o1):.2e}")
print(f"  fp16 rounding of the output alone: {rr(h(o16), o16):.2e}")
f = np.float32
for off in (1024.0, 0.0, -128.0):
    S = np.zeros((G, N), f)
    for g in range(4):
        acc, qs = np.zeros((G, N), f), np.zeros(G, f)
        for d in range(64 * g, 64 * g + 64):                       # fp32 running sums, one dim a step
            acc += qq[:, d:d + 1].astype(f) * (kw[:, d].astype(f) + f(off))[None, :]
            qs += qq[:, d].astype(f)
        S += acc * ks[:, g].astype(f)[None, :] + qs[:, None] * (kb[:, g] - off * ks[:, g]).astype(f)[None, :]
    S *= f(sc)
    Sx = (qq * sc) @ kx.T
    p = np.exp(S - S.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    O = np.zeros((G, D), f)
    for g in range(4):
        pg = p * vs[:, g].astype(f)[None, :]
        acc = np.zeros((G, 64), f)
        for i in range(0, N, 16):                                  # 16 keys a step
            acc += pg[:, i:i + 16] @ (vw[i:i + 16, 64 * g:64 * g + 64].astype(f) + f(off))
        O[:, 64 * g:64 * g + 64] = acc + (p @ (vb[:, g] - off * vs[:, g]).astype(f))[:, None]
    O = O.astype(np.float64)
    print(f"  factored, fp32, code offset {off:+5.0f}: scores max |d| {np.abs(S - Sx).max():.1e}; output vs fp64 over "
          f"the affine values {rr(O, ox):.2e}, rounded to fp16 {rr(h(O), ox):.2e}; rounded, vs fp64 over "
          f"mx.dequantize's values {rr(h(O), o16):.2e}")
