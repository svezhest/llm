"""Teacher-forced KL between two prefill variants on real text: the worker's
slab loop with logits for every position (float32 log-softmax), variant A vs B.
  cd src/worker && QWEN_MODEL=... QWEN_KV_BITS=0 .venv/bin/python kl_check.py --a pf --b gdnfast --tokens 8192"""
import argparse, os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
import pf_common as C
import worker as W

ap = argparse.ArgumentParser()
ap.add_argument("--a", default="pf")
ap.add_argument("--b", default="gdnfast")
ap.add_argument("--tokens", type=int, default=8192)
ap.add_argument("--slab", type=int, default=0, help="0 = worker _step")
a = ap.parse_args()

model, proc = C.load()
lm = model.language_model
import pfqmm
pfqmm.install(lm)
import variants_pf as V
VAR = V.make(W, pfqmm, lm)
tok = getattr(proc, "tokenizer", proc)
R = "/Users/user/Projects/llm"
text = ""
for f in ["README.md", "CLAUDE.md", "BACKLOG.md", "src/worker/worker.py", "src/gateway/src/api.rs", "src/gateway/src/request.rs"]:
    text += open(os.path.join(R, f)).read() + "\n\n"
ids = tok.encode(text)[: a.tokens]
print("tokens", len(ids), flush=True)


def run(v):
    VAR[v]()
    from mlx_vlm.models.cache import make_prompt_cache
    cache = make_prompt_cache(lm)
    cuts = set(W._bounds(ids, -1))
    inp = mx.array([ids])
    pos, outs = 0, []
    while pos < len(ids) - 1:
        n = a.slab or W._slab(pos, len(ids) - 1 - pos, cuts)
        n = min(n, len(ids) - 1 - pos)
        out = lm(inp[:, pos:pos + n], cache=cache)
        lp = out.logits[0].astype(mx.float32)
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        # keep compact: top-1 ids + full logprobs of this slab to compute KL on the fly
        outs.append(lp)
        mx.eval(outs[-1], [c.state for c in cache])
        pos += n
        mx.clear_cache()
    return outs


A = run(a.a)
kls, agree, n = [], 0, 0
B = run(a.b)
for x, y in zip(A, B):
    p = mx.exp(x)
    kl = (p * (x - y)).sum(axis=-1)
    kls.append(kl)
    agree += (mx.argmax(x, -1) == mx.argmax(y, -1)).sum().item()
    n += x.shape[0]
kl = mx.concatenate(kls)
print(f"{a.a} vs {a.b}: {n} positions  KL mean {kl.mean().item():.3g}  p99 {mx.sort(kl)[int(0.99 * n)].item():.3g}  "
      f"max {kl.max().item():.3g}  top-1 agreement {100 * agree / n:.2f}%", flush=True)
