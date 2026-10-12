"""GPU: worker.sample_rows == per-row samplers, token for token, over many steps
(seeded keys advance identically), mixed settings and dtypes in one batch."""
import os, sys
sys.path.insert(0, os.getcwd())
import mlx.core as mx
import worker as W
V = 248320
CFGS = [dict(temperature=0.0), dict(temperature=0.6, top_p=0.95, top_k=20), dict(temperature=0.7, top_p=0.9),
        dict(temperature=0.8, min_p=0.05), dict(temperature=1.0), dict(temperature=0.6, top_k=40)]
bad = total = 0
for trial in range(4):
    rows = []
    for i in range(16):
        c = dict(W.SAMPLING_KWARGS, **CFGS[(i + trial) % len(CFGS)])
        rows.append((c, 100 * trial + i, mx.float32 if i % 3 == 0 else mx.float16))
    A = [W.make_sampler(c, seed=s) for c, s, _ in rows]
    Bs = [W.make_sampler(c, seed=s) for c, s, _ in rows]
    for step in range(25):
        mx.random.seed(1000 * trial + step)
        lg = (mx.random.normal((16, V)) * 3).astype(mx.float16)
        xs = [lg[i:i + 1].astype(dt) for i, (_, _, dt) in enumerate(rows)]
        a = mx.concatenate([A[i](xs[i]) for i in range(16)]).tolist()
        b = mx.concatenate(W.sample_rows(Bs, xs)).tolist()
        bad += sum(x != y for x, y in zip(a, b))
        total += 16
print(f"sample_rows vs per-row: {total - bad}/{total} identical")
# bitwise: filtered stack rows == filtered single rows; greedy stack == rows
neq = 0
for c in CFGS:
    s = W.make_sampler(dict(W.SAMPLING_KWARGS, **c), seed=1)
    for seed in range(3):
        mx.random.seed(seed)
        lg = (mx.random.normal((16, V)) * 3).astype(mx.float16)
        if s.greedy:
            st, rw = s(lg), mx.concatenate([s(lg[i:i + 1]) for i in range(16)])
        else:
            st, rw = s.filtered(lg), mx.concatenate([s.filtered(lg[i:i + 1]) for i in range(16)])
        neq += not mx.array_equal(st, rw, equal_nan=True).item()
print("filtered/greedy stacks bitwise equal to rows:", "yes" if neq == 0 else f"NO ({neq})")
