"""Dependent chains (as in a decode step): x -> up (4096->12288) -> down (12288->4096) -> ...
and o (4096->4096) chains; qmv vs mrqmm configs at 1 / 2 / 4 rows."""
import sys, time, statistics
sys.path.insert(0, "/Users/user/Projects/llm/.claude/worktrees/agent-aad53b7f0c80606bc/src/worker")
import mlx.core as mx
import mrqmm
mx.set_default_device(mx.gpu)


def mk(N, K, seed):
    mx.random.seed(seed)
    w = mx.random.randint(0, 256, (N, K), dtype=mx.uint8).view(mx.uint32)
    s = (mx.random.uniform(shape=(N, K // 64)) * 1e-3).astype(mx.float16)
    b = (mx.random.normal((N, K // 64)) * 1e-3).astype(mx.float16)
    mx.eval(w, s, b)
    return w, s, b


chains = {
    "mlp": [mk(12288, 4096, i) if i % 2 == 0 else mk(4096, 12288, i) for i in range(16)],
    "sq": [mk(4096, 4096, 100 + i) for i in range(16)],
}
mrqmm.set_invariant(True)


def run(ws, x, f):
    for w in ws:
        x = f(x, *w)
        x = x * 0.01
    return x


def qmv(x, w, s, b):
    return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=8)


for name, ws in chains.items():
    for rows in (1, 2, 4):
        x = mx.random.normal((rows, 1, 4096)).astype(mx.float16)
        V = {"qmv": qmv}
        for na in (1, 2, 4):
            V[f"na{na}"] = (lambda na: lambda x, w, s, b: mrqmm.mrqmm(x, w, s, b, cfg=(na, 8)))(na)
        res = {v: [] for v in V}
        for r in range(9):
            for v, f in (V.items() if r % 2 == 0 else list(V.items())[::-1]):
                mx.eval(run(ws, x, f))
                t = time.perf_counter()
                for _ in range(4):
                    mx.eval(run(ws, x, f))
                res[v].append((time.perf_counter() - t) / 4 / len(ws) * 1e6)
        print(name, rows, " ".join(f"{v}={statistics.median(x):.1f}us" for v, x in res.items()), flush=True)
