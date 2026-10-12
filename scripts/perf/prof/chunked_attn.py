"""Chunked unfused causal attention (head_dim 256, GQA) for long prefill slabs:
keys in chunks of C, per chunk scores = q k^T (steel GEMM), precise softmax,
p v; chunks merged with their log-sum-exp in fp32. vs force_fused.
  cd src/worker && .venv/bin/python chunked_attn.py"""
import os, time, statistics
import mlx.core as mx


def chunked_sdpa(q, k, v, scale, chunk=16384):
    B, H, L, D = q.shape
    Hk, S = k.shape[1], k.shape[2]
    g = H // Hk
    qr = (q * scale).reshape(B, Hk, g * L, D)                  # row r = head h*g + j, query i  -> (j*L + i)
    off = S - L
    outs, lses = [], []
    for s0 in range(0, S, chunk):
        s1 = min(S, s0 + chunk)
        kc, vc = k[:, :, s0:s1], v[:, :, s0:s1]
        sc = qr @ kc.swapaxes(-1, -2)                            # (B, Hk, g*L, n)
        if s1 > off + 1:                                         # chunk reaches past the first query's position
            qpos = off + mx.arange(L)
            kpos = s0 + mx.arange(s1 - s0)
            m = (kpos[None, :] > qpos[:, None])                  # (L, n)
            m = mx.tile(m, (g, 1))                               # (g*L, n)
            sc = mx.where(m, mx.array(-float("inf"), sc.dtype), sc)
        sc32 = sc.astype(mx.float32)
        lse = mx.logsumexp(sc32, axis=-1, keepdims=True)         # (B,Hk,gL,1)
        p = mx.exp(sc32 - lse).astype(q.dtype)
        outs.append(p @ vc)                                      # (B,Hk,gL,D) fp16
        lses.append(lse)
    if len(outs) == 1:
        o = outs[0]
    else:
        L_ = mx.concatenate(lses, axis=-1)                       # (B,Hk,gL,nc)
        tot = mx.logsumexp(L_, axis=-1, keepdims=True)
        w = mx.exp(L_ - tot)
        o = sum(w[..., i:i + 1] * outs[i].astype(mx.float32) for i in range(len(outs))).astype(q.dtype)
    return o.reshape(B, H, L, D)


def timeit(f, reps=3):
    mx.eval(f())
    ts = []
    for _ in range(reps):
        mx.synchronize(); t = time.perf_counter(); mx.eval(f()); mx.synchronize(); ts.append(time.perf_counter() - t)
    return statistics.median(ts)


if __name__ == "__main__":
    mx.random.seed(0)
    CH = [int(x) for x in os.environ.get("CA_CHUNKS", "8192,16384").split(",")]
    for S in [int(x) for x in os.environ.get("CA_S", "32768,65536,122880").split(",")]:
        for L in [int(x) for x in os.environ.get("CA_L", "2048").split(",")]:
            q = mx.random.normal((1, 16, L, 256)).astype(mx.float16)
            kb = mx.random.normal((1, 4, S + 256, 256)).astype(mx.float16)
            vb = mx.random.normal((1, 4, S + 256, 256)).astype(mx.float16)
            k, v = kb[:, :, :S], vb[:, :, :S]
            mx.eval(q, kb, vb)
            sc = 256 ** -0.5
            ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask="causal", force_fused=True)
            fl = 4.0 * 16 * L * (S - L / 2) * 256
            tf = timeit(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask="causal", force_fused=True))
            line = f"L {L} S {S}: fused {tf*1e3:7.1f} ms {fl/tf/1e12:5.2f} TF"
            for c in CH:
                mx.reset_peak_memory(); base = mx.get_active_memory()
                o = chunked_sdpa(q, k, v, sc, c)
                d = mx.abs(o.astype(mx.float32) - ref.astype(mx.float32)).max().item()
                pk = (mx.get_peak_memory() - base) / 1e9
                t = timeit(lambda: chunked_sdpa(q, k, v, sc, c))
                line += f" | chunk {c}: {t*1e3:7.1f} ms {fl/t/1e12:5.2f} TF x{tf/t:.2f} d={d:.1e} +{pk:.1f}GB"
            print(line, flush=True)
            del q, kb, vb, k, v, ref
            mx.clear_cache()
