"""mx.quantized_matmul by weight bits and row count on a 27B MLP shape (M1 Max)."""
import statistics, time
import mlx.core as mx
N, K = 17408, 5120
mx.random.seed(0)
W = (mx.random.normal((N, K)) * 0.02).astype(mx.float16)
Q = {}
for bits in (4, 5, 6, 8):
    w, s, b = mx.quantize(W, group_size=64, bits=bits)
    s, b = s.astype(mx.float16), b.astype(mx.float16)
    mx.eval(w, s, b)
    Q[bits] = (w, s, b)
ROWS = (1, 2, 3, 4, 6, 8, 16)
X = {m: mx.random.normal((m, 1, K)).astype(mx.float16) for m in ROWS}
mx.eval(list(X.values()))
T = {}
REP = 8          # matmuls per timed eval (one MLP projection x REP)
def run(bits, m):
    w, s, b = Q[bits]
    ys = [mx.quantized_matmul(X[m] + i, w, scales=s, biases=b, transpose=True, group_size=64, bits=bits) for i in range(REP)]
    t = time.perf_counter(); mx.eval(ys); return (time.perf_counter() - t) * 1000 / REP
for rnd in range(7):
    for m in ROWS:
        for bits in Q:
            T.setdefault((bits, m), []).append(run(bits, m))
print(f"quantized_matmul, N={N} K={K}, fp16, ms per matmul (median of 6 rounds, interleaved); x = (rows, 1, K)")
print("bits  bytes(MB) " + " ".join(f"{m:>7d}r" for m in ROWS) + "   | ratio to 1 row: " + " ".join(f"{m:>5d}r" for m in ROWS[1:]) + "  | GB/s at 1 row")
for bits in Q:
    med = {m: statistics.median(T[(bits, m)][1:]) for m in ROWS}
    mb = N * K * bits / 8 / 1e6
    print(f"{bits:4d}  {mb:8.0f}  " + " ".join(f"{med[m]:8.3f}" for m in ROWS) + "   |                 "
          + " ".join(f"{med[m] / med[1]:6.2f}" for m in ROWS[1:]) + f"  | {mb / 1e3 / (med[1] / 1e3):6.0f}")
