import sys, time, statistics
import mlx.core as mx
N = int(sys.argv[1]); T = int(sys.argv[2]) if len(sys.argv) > 2 else 9
H, HK, D, L = 24, 4, 256, 4          # L layers' worth of distinct KV per timing (no cache reuse)
mx.random.seed(0)
cap = N + 200
Ks = [mx.random.normal((1, HK, cap, D)).astype(mx.float16) for _ in range(L)]
Vs = [mx.random.normal((1, HK, cap, D)).astype(mx.float16) for _ in range(L)]
Q = (mx.random.normal((1, H, T, D)) * 4).astype(mx.float16)
mx.eval(Ks, Vs, Q)
scale = D ** -0.5
sd = mx.fast.scaled_dot_product_attention

def rows(q, k, v):
    return mx.concatenate([sd(q[:, :, i:i + 1], k[:, :, :N - T + i + 1], v[:, :, :N - T + i + 1], scale=scale, mask=None)
                           for i in range(T)], axis=2)
def one_mask(q, k, v):
    k, v = k[:, :, :N], v[:, :, :N]
    m = mx.arange(N)[None, None, None, :] < (N - T + 1 + mx.arange(T))[None, None, :, None]
    return sd(q, k, v, scale=scale, mask=m)
def one_causal(q, k, v):
    return sd(q, k[:, :, :N], v[:, :, :N], scale=scale, mask="causal")
def one_fused(q, k, v):
    return sd(q, k[:, :, :N], v[:, :, :N], scale=scale, mask="causal", force_fused=True)
def kq(q, k, v):
    k, v = k[:, :, :N], v[:, :, :N]
    R = H // HK
    qq = (q * scale).reshape(1, HK, R, T, D)
    s = mx.matmul(k[:, :, None], qq.swapaxes(-1, -2))            # (1,HK,R,N,T)
    m = (mx.arange(N)[:, None] < (N - T + 1 + mx.arange(T))[None, :])
    s = mx.where(m, s, mx.array(-float("inf"), dtype=s.dtype))
    w = mx.softmax(s, axis=-2, precise=True)
    o = mx.matmul(w.swapaxes(-1, -2), v[:, :, None])              # (1,HK,R,T,D)
    return o.reshape(1, H, T, D)
def single(q, k, v):                                             # one plain decode step's attention
    return sd(q[:, :, :1], k[:, :, :N], v[:, :, :N], scale=scale, mask=None)

fns = dict(single=single, rows=rows, one_mask=one_mask, one_causal=one_causal, one_fused=one_fused, kq=kq)
ref = rows(Q, Ks[0], Vs[0]).astype(mx.float32); mx.eval(ref)
ok = {}
for n, f in list(fns.items()):
    try:
        o = f(Q, Ks[0], Vs[0]).astype(mx.float32); mx.eval(o)
        ok[n] = None if n == "single" else float(mx.abs(o - ref).max())
    except Exception as e:
        print(n, "FAILED", str(e)[:120]); fns.pop(n)
t = {n: [] for n in fns}
for r in range(7):
    for n, f in fns.items():
        mx.reset_peak_memory(); a0 = mx.get_active_memory()
        t0 = time.perf_counter()
        mx.eval([f(Q, Ks[j], Vs[j]) for j in range(L)])
        t[n].append((time.perf_counter() - t0) / L * 1e3)
        if r == 6: ok[n] = (ok[n], (mx.get_peak_memory() - a0) / 2**20)
print(f"N={N} T={T} KV per layer {2*HK*N*D*2/2**20:.0f} MiB; ref |max| {float(mx.abs(ref).max()):.2f}")
for n in fns:
    print(f"{n:11s} {statistics.median(t[n][1:]):7.2f} ms/layer  max|d| vs rows {ok[n][0]}  transient {ok[n][1]:.0f} MiB")
