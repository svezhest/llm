"""B=1 decode step, mode on vs off, synchronous, no scheduler (why x1.45?)."""
import os, sys, time, statistics
W_ = "/Users/user/Projects/llm/.claude/worktrees/agent-aad53b7f0c80606bc/src/worker"
sys.path.insert(0, W_); os.chdir(W_)
import mlx.core as mx
import worker as W
import mrqmm
import mlx.nn as nn
W._Scheduler.start = lambda self: None
eng = W.Engine(os.environ["QWEN_MODEL"]); eng.load()
lm = eng.model.language_model
mx.set_default_device(mx.gpu)
cache = lm.make_cache()
ids = mx.array([[int(i) % 1000 + 100 for i in range(600)]])
lm(ids, cache=cache)
mx.eval([c.state for c in cache])
y = mx.array([[123]])
pos = 600
variant = sys.argv[1:] or ["on", "off"]


def step(sync=True):
    global pos, y
    out = lm(y, cache=cache, position_ids=mx.array([[pos]], dtype=mx.int32))
    pos += 1
    t = out.logits[:, -1, :].argmax(-1)[None]
    if sync:
        mx.eval(t)
    y = t
    return t



def run_async(n):
    global pos, y
    prev = None
    for _ in range(n):
        out = lm(y, cache=cache, position_ids=mx.array([[pos]], dtype=mx.int32))
        pos += 1
        t = out.logits[:, -1, :].argmax(-1)
        mx.async_eval(t)
        if prev is not None:
            prev.item()
        prev = t
        y = t[None]
    prev.item()


res = {"on_sync": [], "off_sync": [], "on_async": [], "off_async": []}
for r in range(6):
    for v in (["on", "off"] if r % 2 == 0 else ["off", "on"]):
        mrqmm.set_invariant(v == "on")
        for _ in range(3):
            step()
        t = time.perf_counter()
        for _ in range(20):
            step()
        res[v + "_sync"].append((time.perf_counter() - t) / 20 * 1e3)
        run_async(3)
        t = time.perf_counter()
        run_async(20)
        res[v + "_async"].append((time.perf_counter() - t) / 20 * 1e3)
print({k: round(statistics.median(x), 2) for k, x in res.items() if x}, flush=True)
import os as _os; _os._exit(0)
