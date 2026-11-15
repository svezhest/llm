"""One full-attention layer of a DFlash verify block (27B layout: 24 q / 4 kv heads, head_dim 256,
fp16 KV) at N keys, T block queries: the library's one-query calls (rows), MLX's vector kernel in
chunks of 32 // G queries (vec = QWEN_VERIFY_ATT=1), one-pass vatt (QWEN_VERIFY_ATT=2), a plain
decode step's single query (single; vatt1 / rattn1 = the same through vatt / rattn), + a few
one-call variants. Medians over 6 timed rounds of L layers' distinct KV, max |d| vs rows, and
the effective read rate (2 x Hkv x N x D x 2 B per layer / time).

  cd src/worker && .venv/bin/python ../../scripts/perf/dflash2/verify_att_bench.py N [T] [--only a,b] [--json f]
"""
import json, os, sys, time, statistics
import mlx.core as mx
sys.path.insert(0, os.getcwd())
import vatt, rattn  # noqa: E402
args = [x for x in sys.argv[1:] if not x.startswith("--")]
ONLY = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
JS = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else ""
if ONLY:
    args = [x for x in args if x not in (",".join(ONLY),)]
if JS:
    args = [x for x in args if x != JS]
N = int(args[0]); T = int(args[1]) if len(args) > 1 else 8
H, HK, D = int(os.environ.get("VB_HQ", 24)), int(os.environ.get("VB_HKV", 4)), 256
L = int(os.environ.get("VB_LAYERS", 4))   # L layers' worth of distinct KV per timing (no cache reuse)
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
def vec(q, k, v):                                                # QWEN_VERIFY_ATT=1: 32 // G queries per call
    st, out = max(1, 32 // (H // HK)), []
    for a in range(0, T, st):
        b = min(T, a + st)
        out.append(sd(q[:, :, a:b], k[:, :, :N - T + b], v[:, :, :N - T + b], scale=scale, mask="causal",
                      force_fused=True))
    return out[0] if len(out) == 1 else mx.concatenate(out, axis=2)
def vatt2(q, k, v):                                              # QWEN_VERIFY_ATT=2
    return vatt.attention(q, k, v, N, scale)
def vatt1(q, k, v):                                              # a decode step's query through vatt
    return vatt.attention(q[:, :, -1:], k, v, N, scale)
def rattn1(q, k, v):
    return rattn.attention(q[:, :, -1:], [k], [v], [N], scale)

fns = dict(single=single, vatt1=vatt1, rattn1=rattn1, rows=rows, vec=vec, vatt=vatt2, one_mask=one_mask,
           one_causal=one_causal, one_fused=one_fused, kq=kq)
if ONLY:
    fns = {n: f for n, f in fns.items() if n in ONLY or n == "rows"}
ref = vatt.reference(Q, Ks[0], Vs[0], N, scale); mx.eval(ref)
ok = {}
for n, f in list(fns.items()):
    try:
        o = f(Q, Ks[0], Vs[0]).astype(mx.float32); mx.eval(o)
        if n in ("single", "vatt1", "rattn1"):
            r1 = ref[:, :, -1:] if n != "single" else vatt.reference(Q[:, :, :1], Ks[0], Vs[0], N, scale)
        else:
            r1 = ref
        d = o - r1
        ok[n] = (float(mx.abs(d).max()), float(mx.sqrt(mx.mean(d * d)) / mx.sqrt(mx.mean(r1 * r1))))
    except Exception as e:
        print(n, "FAILED", str(e)[:120]); fns.pop(n)
t = {n: [] for n in fns}
for r in range(int(os.environ.get("VB_ROUNDS", 7))):
    for n, f in fns.items():
        mx.reset_peak_memory(); a0 = mx.get_active_memory()
        t0 = time.perf_counter()
        mx.eval([f(Q, Ks[j], Vs[j]) for j in range(L)])
        t[n].append((time.perf_counter() - t0) / L * 1e3)
        ok[n] = ok[n][:2] + ((mx.get_peak_memory() - a0) / 2**20,)
kvb = 2 * HK * N * D * 2
print(f"N={N} T={T} H={H}/{HK} KV per layer {kvb/2**20:.0f} MiB; ref |max| {float(mx.abs(ref).max()):.2f}")
res = {}
for n in fns:
    med = statistics.median(t[n][1:])
    res[n] = dict(ms=med, gbps=kvb / med / 1e6, maxdiff_fp32=ok[n][0], relrms_fp32=ok[n][1],
                  transient_mib=ok[n][2], all_ms=t[n])
    print(f"{n:11s} {med:7.3f} ms/layer  {kvb / med / 1e6:6.1f} GB/s  vs fp32 ref: max|d| {ok[n][0]:.2e} "
          f"rel rms {ok[n][1]:.2e}  transient {ok[n][2]:.0f} MiB", flush=True)
if JS:
    os.makedirs(os.path.dirname(JS) or ".", exist_ok=True)
    json.dump(dict(N=N, T=T, H=H, HK=HK, tile=os.environ.get("VATT_TILE", ""), res=res), open(JS, "w"), indent=1)
