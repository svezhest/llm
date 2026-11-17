"""Teacher-forced KL on the last T tokens of a long real-text prompt, after the
first L-T tokens were prefilled with variant A or B (ab_prefill.py variants:
old / new / pf / nores / slab). The scored tail runs the same code for both
(new policy, one T-token forward): only the cache built by the variant differs.
  cd src/worker && QWEN_MODEL=... QWEN_KV_BITS=0 .venv/bin/python kl_tail.py --len 65536 --a old --b new"""
import argparse, functools, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
import pf_common as C
from doc_notes import NOTES
import worker as W

ap = argparse.ArgumentParser()
ap.add_argument("--len", type=int, default=65536)
ap.add_argument("--tail", type=int, default=2048)
ap.add_argument("--a", default="old")
ap.add_argument("--b", default="new")
a = ap.parse_args()

model, proc = C.load()
lm = model.language_model
import pfqmm
pfqmm.install(lm)
NEW_STEP = W._step


def OLD_STEP(pos):
    return max(W.PREFILL_MIN_STEP, min(W.PREFILL_STEP, W.PREFILL_BUDGET // max(pos, 1)))


def setv(v):
    W._step = OLD_STEP if v in ("old", "pf") else NEW_STEP
    pfqmm.ACTIVE = v in ("new", "pf", "nores")
    return v == "new"


tok = getattr(proc, "tokenizer", proc)
R = "/Users/user/Projects/llm"
text = ""
for f in [*NOTES, "src/worker/worker.py", "src/worker/mrqmm.py",
          "src/gateway/src/api.rs", "src/gateway/src/request.rs", "src/gateway/src/main.rs",
          "src/gateway/src/pool.rs", "src/worker/tests/test_scheduler.py"]:
    text += f"\n\n==== {f}\n" + open(os.path.join(R, f)).read()
ids = tok.encode(text)
print("text tokens", len(ids), flush=True)
while len(ids) < a.len:
    ids = ids + ids
ids = ids[: a.len]
head, tail = ids[: a.len - a.tail], ids[a.len - a.tail:]


def run(v):
    reserve = setv(v)
    from mlx_vlm.models.cache import make_prompt_cache
    cache = make_prompt_cache(lm)
    total = len(ids) if reserve else 0
    cuts = set(W._bounds(head + [0], -1))
    inp = mx.array([head])
    pos = 0
    while pos < len(head):
        n = min(W._slab(pos, len(head) - pos, cuts), len(head) - pos)
        total = W._reserve_kv(cache, total, pos + n, lm, mx.float16, None)
        lm(inp[:, pos:pos + n], cache=cache, skip_logits=True)
        mx.eval([c.state for c in cache])
        pos += n
        mx.clear_cache()
    setv("new")
    out = lm(mx.array([tail]), cache=cache).logits[0].astype(mx.float32)
    lp = out - mx.logsumexp(out, axis=-1, keepdims=True)
    mx.eval(lp)
    del cache
    mx.clear_cache()
    return lp


A = run(a.a)
B = run(a.b)
kl = (mx.exp(A) * (A - B)).sum(-1)
agree = (mx.argmax(A, -1) == mx.argmax(B, -1)).mean().item()
same = mx.array_equal(A, B).item()
print(f"len {a.len} tail {a.tail}: {a.a} vs {a.b}: {'bit-identical logits' if same else ''} KL mean {kl.mean().item():.3g} "
      f"max {kl.max().item():.3g} top-1 {100 * agree:.2f}%", flush=True)
