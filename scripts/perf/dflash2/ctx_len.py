"""How much target context does the DFlash2 drafter need? Accepted drafts per
round as a function of the context it starts with: the last N prompt positions'
hidden states (N = 1 is a prompt-cache hit at the entry's end, a few tens a hit
that leaves a short message to prefill, 2047 = the drafter's window = cold).

No worker: the target runs each prompt once (prefill in slabs + a plain greedy
decode, hidden states of the drafter's layers kept), then the DFlash loop is
replayed per N with the drafter alone (the target's tokens are known: a round
accepts the drafts' longest prefix equal to them, the next round's context is
the hidden states of the tokens fed). Drafter as the worker loads it: 4-bit
weights, fp32 activations.

  cd src/worker && PYTHONPATH=. .venv/bin/python ../../scripts/perf/dflash2/ctx_len.py \\
    --json $PERF_OUT/dflash2/ctx_len.json        (PYTHONPATH=. for wtest)
"""
import argparse
import json
import os
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_vlm import load
from mlx_vlm.models.cache import make_prompt_cache
from mlx_vlm.speculative.drafters import load_drafter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "prof"))
from wtest import TASKS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(ROOT, "weights/qwen38-27b-mlx-5bit"))
ap.add_argument("--draft", default=os.path.join(ROOT, "weights/qwen38-27b-dflash2"))
ap.add_argument("--tasks", type=int, default=4)
ap.add_argument("--gen", type=int, default=192)
ap.add_argument("--block", type=int, default=8)
ap.add_argument("--doc-chars", type=int, default=12000)
ap.add_argument("--ns", default="1,4,16,32,64,128,256,512,1024,2047")
ap.add_argument("--no-thinking", action="store_true")
ap.add_argument("--turn2", action="store_true",
                help="replay the second turn (turn 1 + its answer + a follow-up asking to edit it): "
                     "what a hit at the reply's end leaves the drafter")
ap.add_argument("--wired-gb", type=float, default=27.0)
ap.add_argument("--json", default="")
a = ap.parse_args()

try:
    mx.set_wired_limit(int(a.wired_gb * 2**30))
except Exception as e:  # noqa: BLE001
    print("wired limit not set:", e)
model, proc = load(a.model)
lm = model.language_model
lm.apply(lambda x: x.astype(mx.float16) if x.dtype == mx.bfloat16 else x)
mx.eval(lm.parameters())
tok = getattr(proc, "tokenizer", proc)
argmax = lambda x: mx.argmax(x, axis=-1)  # noqa: E731

d, _ = load_drafter(a.draft)
nn.quantize(d, group_size=64, bits=4,
            class_predicate=lambda _p, m: isinstance(m, nn.Linear) and m.weight.shape[1] % 64 == 0)
d.update(tree_unflatten([(k, v.astype(mx.float16 if "codebook" in k else mx.float32)
                          if v.dtype == mx.bfloat16 else v) for k, v in tree_flatten(d.parameters())]))
mx.eval(d.parameters())
_embed = d._embed_input_tokens
d._embed_input_tokens = lambda ids: _embed(ids).astype(mx.float32)
layers = list(d.config.target_layer_ids)
DOC = ("You are a coding assistant. Project notes follow; use them only if relevant.\n\n"
       + open(os.path.join(ROOT, "README.md")).read()[:a.doc_chars])
print(f"loaded; drafter window {d.config.sliding_window}, layers {layers}", flush=True)


FOLLOW = "Now add type hints and a short docstring, and handle invalid input explicitly."


def target(messages):
    """-> (prompt + greedy continuation, prompt length, hidden states of every
    position but the last token's)."""
    s = tok.apply_chat_template([{"role": "system", "content": DOC}] + messages,
                                tokenize=False, add_generation_prompt=True,
                                enable_thinking=not a.no_thinking)
    ids = tok.encode(s)
    cache = make_prompt_cache(lm)
    lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int32)
    lm._position_ids = None
    hs = []
    for lo in range(0, len(ids) - 1, 1024):
        sink = []
        lm.model(mx.array([ids[lo:min(lo + 1024, len(ids) - 1)]]), cache=cache,
                 capture_layer_ids=layers, hidden_sink=sink)
        h = mx.concatenate(sink, axis=-1)
        mx.eval(h, [c.state for c in cache])
        hs.append(h)
        mx.clear_cache()
    y = mx.array([ids[-1:]])
    gen = []
    for _ in range(a.gen):
        out = lm(y, cache=cache, capture_layer_ids=layers)
        h = mx.concatenate(out.hidden_states, axis=-1)
        y = mx.argmax(out.logits[:, -1, :], axis=-1)[None]
        mx.eval(h, y)
        hs.append(h)
        gen.append(y.item())
    H = mx.concatenate(hs, axis=1)
    mx.eval(H)
    return ids + gen, len(ids), H


def replay(seq, n0, H, N, bs):
    """The DFlash loop with N positions of prompt context -> accepted per round."""
    cache = d.reset(model)
    head = d.lm_head
    d.lm_head = lambda x: head(x.astype(mx.float16))
    p, ctx, acc = n0, H[:, max(0, n0 - N):n0], []
    while p + bs <= len(seq):
        got = d.draft_block(int(seq[p]), ctx.astype(mx.float32), cache, bs, argmax, mx.int32)
        got = [int(t) for t in got[0].tolist()]
        ref = seq[p + 1:p + bs]
        k = 0
        while k < len(got) and got[k] == ref[k]:
            k += 1
        acc.append(k)
        ctx = H[:, p:p + k + 1]
        p += k + 1
    return acc


NS = [int(x) for x in a.ns.split(",")]
mean = lambda xs: round(sum(xs) / len(xs), 2) if xs else None  # noqa: E731
res = []
for ti in range(a.tasks):
    msgs = [{"role": "user", "content": TASKS[ti]}]
    seq, n0, H = target(msgs)
    if a.turn2:
        del H
        mx.clear_cache()
        n1 = n0
        answer = tok.decode(seq[n0:]).strip()
        seq, n0, H = target(msgs + [{"role": "assistant", "content": answer},
                                    {"role": "user", "content": FOLLOW}])
        print(f"t{ti}: turn 2 adds {n0 - n1} tokens (the answer + the follow-up)", flush=True)
    print(f"t{ti}: prompt {n0}, generated {len(seq) - n0}", flush=True)
    for N in NS:
        acc = replay(seq, n0, H, N, a.block)
        o = dict(task=ti, prompt=n0, N=N, rounds=len(acc), first4=mean(acc[:4]), first8=mean(acc[:8]),
                 first16=mean(acc[:16]), all=mean(acc), acc=acc)
        res.append(o)
        print(f"t{ti} N={N:5d} rounds {o['rounds']:3d} accepted/round first4 {o['first4']} "
              f"first8 {o['first8']} first16 {o['first16']} all {o['all']}", flush=True)
    del H
    mx.clear_cache()
print("\n== rounds needed for the same tokens (sum over tasks), accepted/round ==")
for N in NS:
    rows = [o for o in res if o["N"] == N]
    acc = [x for o in rows for x in o["acc"]]
    print(f"N={N:5d} rounds {sum(o['rounds'] for o in rows):4d} all {mean(acc)} "
          f"first8 {mean([x for o in rows for x in o['acc'][:8]])}", flush=True)
print(f"peak {mx.get_peak_memory() / 1e9:.2f} GB")
if a.json:
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    json.dump(dict(args=vars(a), runs=res), open(a.json, "w"), indent=1)
