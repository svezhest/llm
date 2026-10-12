"""fused vs default SDPA (head_dim 256, 16q/4kv heads, causal, fp16, KV as cache views), interleaved."""
import time, statistics, os
import mlx.core as mx
QS = [int(x) for x in os.environ.get("ST_Q", "2048,1024,512").split(",")]
KVS = [int(x) for x in os.environ.get("ST_KV", "4096,8192,12288,16384,20480,24576,28672,32768,40960").split(",")]
R = int(os.environ.get("ST_R", "5"))
mx.random.seed(0)
for kv in KVS:
    kb = mx.random.normal((1, 4, kv + 2048 + 256, 256)).astype(mx.float16)
    vb = mx.random.normal((1, 4, kv + 2048 + 256, 256)).astype(mx.float16)
    mx.eval(kb, vb)
    line = f"kv {kv:6d}:"
    for ql in QS:
        L = kv  # total keys incl. the slab
        q = mx.random.normal((1, 16, ql, 256)).astype(mx.float16)
        k, v = kb[:, :, :L], vb[:, :, :L]
        mx.eval(q)
        f = {"def": lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625, mask="causal"),
             "fus": lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625, mask="causal", force_fused=True)}
        ts = {n: [] for n in f}
        for n in f: mx.eval(f[n]())
        for r in range(R):
            for n in (("def", "fus") if r % 2 == 0 else ("fus", "def")):
                mx.synchronize(); t = time.perf_counter(); mx.eval(f[n]()); mx.synchronize(); ts[n].append(time.perf_counter() - t)
        d, u = statistics.median(ts["def"]), statistics.median(ts["fus"])
        line += f"  q{ql}: def {d*1e3:6.1f} fus {u*1e3:6.1f} x{d/u:.2f}"
        mx.clear_cache()
    print(line, flush=True)
