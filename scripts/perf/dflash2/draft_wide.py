"""DFlash2 drafter with quantized weights and fp32 activations: acceptance and
draft_block time vs the unquantized bf16 drafter. Target in fp16 (worker's load)."""
import os
import statistics
import sys
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_vlm import load
from mlx_vlm.models.cache import make_prompt_cache
from mlx_vlm.speculative.drafters import load_drafter

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, f"{ROOT}/scripts/perf/prof")
from wtest import TASKS  # noqa: E402

model, proc = load(f"{ROOT}/weights/qwen38-27b-mlx-5bit")
lm = model.language_model
lm.apply(lambda a: a.astype(mx.float16) if a.dtype == mx.bfloat16 else a)
mx.eval(lm.parameters())
tok = getattr(proc, "tokenizer", proc)
argmax = lambda x: mx.argmax(x, axis=-1)  # noqa: E731
base = mx.get_active_memory()


def widen(d, bits):
    """Quantized weights (bits > 0) + fp32 everything else; codebooks fp16."""
    if bits:
        nn.quantize(d, group_size=64, bits=bits,
                    class_predicate=lambda _p, m: isinstance(m, nn.Linear) and m.weight.shape[1] % 64 == 0)

        def cast(path, a):
            if a.dtype != mx.bfloat16:
                return a
            return a.astype(mx.float16 if "codebook" in path else mx.float32)
        d.update(tree_unflatten([(k, cast(k, v)) for k, v in tree_flatten(d.parameters())]))
    mx.eval(d.parameters())
    embed = d._embed_input_tokens
    d._embed_input_tokens = lambda ids: embed(ids).astype(mx.float32)
    return d


def bind_head(d):
    head = d.lm_head
    d.lm_head = lambda x: head(x.astype(mx.float16))


def fresh(mode, bits=0):
    d, _ = load_drafter(f"{ROOT}/weights/qwen38-27b-dflash2")
    if mode == "wide":
        widen(d, bits)
    mx.eval(d.parameters())
    return d


d0 = fresh("bf16")
layers = list(d0.config.target_layer_ids)
seqs = []
for ti in range(int(os.environ.get("NTASKS", "5"))):
    s = tok.apply_chat_template([{"role": "user", "content": TASKS[ti]}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    ids = tok.encode(s)
    cache = make_prompt_cache(lm)
    lm._rope_deltas = lm._position_ids = None
    out = lm(mx.array([ids]), cache=cache)
    y = mx.argmax(out.logits[:, -1, :], axis=-1)
    gen = [y.item()]
    for _ in range(95):
        out = lm(y[None], cache=cache)
        y = mx.argmax(out.logits[:, -1, :], axis=-1)
        gen.append(y.item())
    seq = ids + gen
    cache = make_prompt_cache(lm)
    sink = []
    lm._rope_deltas = lm._position_ids = None
    lm.model(mx.array([seq]), cache=cache, capture_layer_ids=layers, hidden_sink=sink)
    H = mx.concatenate(sink, axis=-1)
    mx.eval(H)
    seqs.append((seq, len(ids), H))
print("sequences ready", [len(s[0]) for s in seqs], flush=True)


def probe(d, wide, block=8, hd=None):
    lens, ms = [], []
    for seq, n0, H in seqs:
        Hc = H.astype(hd) if hd is not None else H
        for p in range(n0, len(seq) - block, 3):
            cache = d.reset(model)
            if wide:
                bind_head(d)
            ctx = Hc[:, :p]
            mx.eval(ctx)
            t = time.perf_counter()
            got = d.draft_block(int(seq[p]), ctx, cache, block, argmax, mx.int32)
            mx.eval(got)
            ms.append((time.perf_counter() - t) * 1000)
            got = [int(t) for t in got[0].tolist()]
            ref = seq[p + 1:p + block]
            k = 0
            while k < len(got) and got[k] == ref[k]:
                k += 1
            lens.append(k)
    hist = [lens.count(i) for i in range(block)]
    return (f"mean accepted {sum(lens) / len(lens):.2f} of {block - 1} ({len(lens)} probes, hist {hist}); "
            f"draft_block median {statistics.median(ms[3:]):.1f} ms (whole context each time)")


def incr(d, wide, block=8, hd=None):
    """draft_block as in the loop: the cache persists, each round feeds only
    the new context rows (here 5 per round) -> the per-round drafter cost."""
    seq, n0, H = seqs[0]
    Hc = H.astype(hd) if hd is not None else H
    cache = d.reset(model)
    if wide:
        bind_head(d)
    ms = []
    p = n0
    got = d.draft_block(int(seq[p]), Hc[:, :p], cache, block, argmax, mx.int32)
    mx.eval(got)
    while p + 5 + block < len(seq):
        ctx = Hc[:, p:p + 5]
        p += 5
        mx.eval(ctx)
        t = time.perf_counter()
        got = d.draft_block(int(seq[p]), ctx, cache, block, argmax, mx.int32)
        mx.eval(got)
        ms.append((time.perf_counter() - t) * 1000)
    return f"incremental draft_block (5 new context rows) median {statistics.median(ms):.1f} ms"


for b in (8, 5):
    print(f"bf16 as loaded, block {b}:", probe(d0, False, b), flush=True)
print("bf16 as loaded:", incr(d0, False), f"; drafter {(mx.get_active_memory() - base) / 1e9:.2f} GB", flush=True)
del d0
mx.clear_cache()
for bits in [int(x) for x in os.environ.get("BITS", "4,8,6,5").split(",")]:
    d = fresh("wide", bits)
    gb = (mx.get_active_memory() - base) / 1e9
    for b in (8, 5):
        print(f"{bits}-bit wide, block {b}:", probe(d, True, b, mx.float32), flush=True)
    print(f"{bits}-bit wide:", incr(d, True, 8, mx.float32), f"; drafter {gb:.2f} GB", flush=True)
    del d
    mx.clear_cache()
print("peak GB", mx.get_peak_memory() / 1e9)
