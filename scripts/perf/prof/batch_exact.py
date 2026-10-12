"""Are MLX's row ops bit-identical on a (G, V) batch vs row by row (GPU)?"""
import mlx.core as mx
V = 248320
ok = True
for G in (2, 3, 8, 16):
    for seed in range(3):
        mx.random.seed(seed)
        lg = (mx.random.normal((G, V)) * 3).astype(mx.float16)       # fp16: many exact ties
        x = lg.astype(mx.float32) / 0.6
        kth = mx.topk(x, 20, axis=-1).min(axis=-1, keepdims=True)
        xk = mx.where(x < kth, -float("inf"), x)
        tests = {
            "topk": (lambda a: mx.topk(a, 20, axis=-1), x),
            "topk_min": (lambda a: mx.topk(a, 20, axis=-1).min(axis=-1, keepdims=True), x),
            "argsort": (lambda a: mx.argsort(a, axis=-1), x),
            "argsort_masked": (lambda a: mx.argsort(a, axis=-1), xk),
            "softmax": (lambda a: mx.softmax(a, axis=-1), x),
            "softmax_masked_sorted": (lambda a: mx.softmax(mx.sort(a, axis=-1), axis=-1), xk),
            "cumsum": (lambda a: mx.cumsum(mx.softmax(mx.sort(a, axis=-1), axis=-1), axis=-1), xk),
            "cumsum_full": (lambda a: mx.cumsum(mx.softmax(mx.sort(a, axis=-1), axis=-1), axis=-1), x),
            "max": (lambda a: a.max(axis=-1, keepdims=True), x),
            "lse": (lambda a: mx.logsumexp(a, axis=-1, keepdims=True), lg),
        }
        for name, (f, a) in tests.items():
            b = f(a)
            r = mx.concatenate([f(a[i:i + 1]) for i in range(G)])
            same = mx.array_equal(b, r).item()
            if not same:
                ok = False
                print(f"G={G} seed={seed} {name}: DIFFERENT (max |d| {mx.abs(b.astype(mx.float32) - r.astype(mx.float32)).max().item()})")
print("all identical" if ok else "differences found")
