"""Where do the ~90 ms of a 4-bit fp32 DFlash2 draft_block go? Each stage synced."""
import os, statistics, sys, time
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_vlm import load
from mlx_vlm.speculative.drafters import load_drafter
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
model, proc = load(f"{ROOT}/weights/qwen38-27b-mlx-5bit")
lm = model.language_model
lm.apply(lambda a: a.astype(mx.float16) if a.dtype == mx.bfloat16 else a)
mx.eval(lm.parameters())
BITS = int(os.environ.get("BITS", "4"))
ACT = getattr(mx, os.environ.get("ACT", "float32"))
d, _ = load_drafter(f"{ROOT}/weights/qwen38-27b-dflash2")
nn.quantize(d, group_size=64, bits=BITS, class_predicate=lambda _p, m: isinstance(m, nn.Linear) and m.weight.shape[1] % 64 == 0)
d.update(tree_unflatten([(k, v.astype(mx.float16 if "codebook" in k else ACT) if v.dtype == mx.bfloat16 else v) for k, v in tree_flatten(d.parameters())]))
mx.eval(d.parameters())
cache = d.reset(model)
head = d.lm_head
T = {}
def tm(name, f):
    t = time.perf_counter(); y = f(); mx.eval(y); T.setdefault(name, []).append((time.perf_counter() - t) * 1000); return y
mx.random.seed(0)
BLOCK, CTX = 8, int(os.environ.get("CTX", "5"))
ctx0 = (mx.random.normal((1, 64, 25600)) * 30).astype(ACT)
h_ctx = d.hidden_norm(d.fc(ctx0)); mx.eval(h_ctx)
for layer, c in zip(d.layers, cache):          # fill the caches with 64 context rows
    mx.eval(layer(mx.zeros((1, BLOCK, 5120), dtype=ACT), h_ctx, d.rope, c, None))
for it in range(12):
    ctx = (mx.random.normal((1, CTX, 25600)) * 30).astype(ACT); mx.eval(ctx)
    ids = mx.array([[1000 + it] + [d.config.mask_token_id] * (BLOCK - 1)])
    t0 = time.perf_counter()
    x = tm("embed", lambda: (d.embed_tokens(ids) * d.embed_scale).astype(ACT))
    f = tm("fc (ctx rows)", lambda: d.fc(ctx))
    hc = tm("hidden_norm", lambda: d.hidden_norm(f))
    h = x
    for layer, c in zip(d.layers, cache):
        a_in = tm("layernorms", lambda: layer.input_layernorm(h))
        pk = tm("conv.prepare (kernel_projection + conv)", lambda: layer.attention_conv.prepare(a_in))
        att = tm("self_attn (q,k,v,o proj + sdpa)", lambda: layer.self_attn(pk[0], hc, d.rope, c, None))
        h = tm("conv.finish + residual", lambda: h + layer.attention_conv.finish(att, pk[1]))
        m_n = tm("layernorms", lambda: layer.post_attention_layernorm(h))
        pm = tm("conv.prepare (kernel_projection + conv)", lambda: layer.mlp_conv.prepare(m_n))
        mm = tm("mlp (gate, up, down)", lambda: layer.mlp(pm[0]))
        h = tm("conv.finish + residual", lambda: h + layer.mlp_conv.finish(mm, pm[1]))
    hn = tm("final norm", lambda: d.norm(h)[:, 1:])
    lg = tm("lm_head (7 rows, target 5-bit)", lambda: head(hn.astype(mx.float16)))
    cand = tm("selector argpartition top-16", lambda: mx.argpartition(lg, -16, axis=-1)[..., -16:])
    sel = d.candidate_selector
    un = tm("selector unary+hidden_projection", lambda: (mx.take_along_axis(lg, cand, axis=-1), sel.hidden_projection(hn)))
    def path():
        pred = ids[:, 0]; out = []
        for p in range(BLOCK - 1):
            e = mx.sum(sel.predecessor_codebook(pred)[:, None] * un[1][:, p, None] * sel.successor_codebook(cand[:, p]), axis=-1)
            s = mx.argmax(un[0][:, p] + e, axis=-1).reshape(-1)
            pred = mx.take_along_axis(cand[:, p], s[:, None], axis=-1)[:, 0]; out.append(pred)
        return mx.stack(out, axis=1)
    tm("selector path (7 steps)", path)
    T.setdefault("TOTAL (staged, synced)", []).append((time.perf_counter() - t0) * 1000)
    c2 = [type(c)(max_size=c.max_size, keep=0) if hasattr(c, "max_size") else c for c in cache]
for it in range(8):                               # the real call, one eval
    ctx = (mx.random.normal((1, CTX, 25600)) * 30).astype(ACT); mx.eval(ctx)
    d.lm_head = lambda x: head(x.astype(mx.float16))
    tm("draft_block (one eval)", lambda: d.draft_block(1000 + it, ctx, cache, BLOCK, lambda z: mx.argmax(z, axis=-1), mx.int32))
print(f"bits {BITS} act {ACT} block {BLOCK} ctx rows {CTX}")
for k, v in T.items():
    n = len(v) // (12 if "one eval" not in k else 8)
    print(f"  {k:44s} {statistics.median(v[n * 2:]) * n:7.2f} ms  ({n} calls x {statistics.median(v[n*2:]):.2f})")
