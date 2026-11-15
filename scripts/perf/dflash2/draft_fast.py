"""DFlash2 drafter variants on the real target, in one process: acceptance
(teacher forcing, as draft_wide.py) and the per-round drafter block
(incremental, as the loop runs it), variants interleaved; optional stage
profile (each stage synced, as draft_prof.py).

Variants (comma separated): d<bits>[+mr][+cc]
  d<bits>  the drafter quantized to <bits> (the engine's own when it is
           QWEN_DRAFT_BITS, else loaded next to it with the worker's code)
  +mr      QWEN_DRAFT_MRQMM (linears on mrqmm_wide)
  +cc      QWEN_DRAFT_COMPILE (compiled grouped convolution)

  cd src/worker && QWEN_MODEL=../../weights/qwen38-27b-mlx-4bit QWEN_TEXT_ONLY=1 QWEN_MRQMM4=1 \\
    QWEN_KV_BITS=0 QWEN_MAX_BATCH=1 QWEN_DRAFT=../../weights/qwen38-27b-dflash2 PYTHONPATH=. \\
    .venv/bin/python ../../scripts/perf/dflash2/draft_fast.py --variants d4,d4+mr,d4+mr+cc \\
    --json $PERF_OUT/drafter/draft_fast.json
"""
import argparse
import json
import logging
import os
import statistics
import sys
import time

sys.path.insert(0, os.getcwd())
import worker as W  # noqa: E402
import worker_pb2 as pb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "prof"))
sys.path.insert(0, HERE)
from prompts2k import prompts  # noqa: E402
from wtest import TASKS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variants", default="d4,d4+mr,d4+mr+cc")
ap.add_argument("--tasks", type=int, default=5, help="short code TASKS (+ the three 2K prompts)")
ap.add_argument("--no-2k", action="store_true")
ap.add_argument("--gen", type=int, default=96, help="greedy tokens per sequence (probed)")
ap.add_argument("--stride", type=int, default=3, help="probe every N positions")
ap.add_argument("--block", type=int, default=8)
ap.add_argument("--incr-rounds", type=int, default=40, help="incremental blocks per variant")
ap.add_argument("--stages", default="", help="variants to stage-profile (synced)")
ap.add_argument("--json", default="")
a = ap.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
import mlx.core as mx  # noqa: E402
from mlx_vlm.models.cache import make_prompt_cache  # noqa: E402

from qwen_worker import config as cfg  # noqa: E402
from qwen_worker.engine import _load_drafter, _setup_drafter_fast  # noqa: E402
from qwen_worker.kernels import _verify_head  # noqa: E402

eng = W.Engine(os.environ["QWEN_MODEL"])
eng.load()
assert eng.draft is not None and eng.draft_wide
lm = eng.model.language_model
tok = getattr(eng.processor, "tokenizer", eng.processor)
base_active = mx.get_active_memory()
DRAFTS = {W.DRAFT_BITS: eng.draft}
print("drafter fast paths (engine's):", _setup_drafter_fast(eng.draft, True, keep_fp32=True), flush=True)


def parse(v):
    head, *flags = v.split("+")
    return int(head[1:]), "mr" in flags, "cc" in flags


for v in a.variants.split(","):
    bits = parse(v)[0]
    if bits not in DRAFTS:
        m0 = mx.get_active_memory()
        d, wide = _load_drafter(eng.model, os.environ["QWEN_DRAFT"], bits)
        d.prefer_requested_block_size = True
        print(f"d{bits}:", _setup_drafter_fast(d, wide, keep_fp32=True),
              f"{(mx.get_active_memory() - m0) / 1e9:.2f} GB", flush=True)
        DRAFTS[bits] = d


def use(v):
    bits, mr, cc = parse(v)
    cfg.DRAFT_MRQMM, cfg.DRAFT_COMPILE = mr, cc
    return DRAFTS[bits]


def bind(d):
    cache = d.reset(eng.model)
    head = _verify_head(d.lm_head)
    d.lm_head = lambda x: head(x.astype(mx.float16))
    return cache


argmax = lambda x: mx.argmax(x, axis=-1)  # noqa: E731
layers = list(eng.draft.config.target_layer_ids)
texts = [(f"code{i}", t) for i, t in enumerate(TASKS[:a.tasks])]
if not a.no_2k:
    texts += prompts(tok)
seqs = []
for name, text in texts:
    s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    ids = tok.encode(s)
    cache = make_prompt_cache(lm)
    lm._rope_deltas = lm._position_ids = None
    out = lm(mx.array([ids]), cache=cache)
    y = mx.argmax(out.logits[:, -1, :], axis=-1)
    gen = [y.item()]
    for _ in range(a.gen - 1):
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
    del cache
    mx.clear_cache()
    seqs.append((name, seq, len(ids), H))
print("sequences ready", [(s[0], len(s[1])) for s in seqs], flush=True)

VS = a.variants.split(",")
res = {"variants": VS, "gen": a.gen, "stride": a.stride, "block": a.block,
       "model": os.environ["QWEN_MODEL"], "accept": {}, "incr": {}, "stages": {}}

# --- acceptance: per probe position, every variant (interleaved)
B = a.block
lens = {v: {} for v in VS}
for name, seq, n0, H in seqs:
    Hc = H.astype(mx.float32)
    for p in range(n0, len(seq) - B, a.stride):
        ctx = Hc[:, max(0, p - 2047):p]
        for v in VS:
            d = use(v)
            cache = bind(d)
            got = d.draft_block(int(seq[p]), ctx, cache, B, argmax, mx.int32)
            got = [int(t) for t in got[0].tolist()]
            ref = seq[p + 1:p + B]
            k = 0
            while k < len(got) and got[k] == ref[k]:
                k += 1
            lens[v].setdefault(name, []).append(k)
for v in VS:
    per = {n: sum(x) / len(x) for n, x in lens[v].items()}
    allx = [k for x in lens[v].values() for k in x]
    res["accept"][v] = dict(mean=sum(allx) / len(allx), probes=len(allx), per_prompt=per)
    print(f"accept {v:12s} mean {sum(allx) / len(allx):.3f} of {B - 1} ({len(allx)} probes) "
          + " ".join(f"{n}={x:.2f}" for n, x in per.items()), flush=True)

# --- incremental drafter block (5 new context rows a round), interleaved
name, seq, n0, H = max(seqs, key=lambda s: s[2])        # the longest prompt
Hc = H.astype(mx.float32)
state = {}
for v in VS:
    d = use(v)
    c = bind(d)
    mx.eval(d.draft_block(int(seq[n0]), Hc[:, max(0, n0 - 2047):n0], c, B, argmax, mx.int32))
    state[v] = c
ms = {v: [] for v in VS}
p = n0
for r in range(a.incr_rounds):
    if p + 5 + B >= len(seq):
        p = n0                                   # wrap: the caches only grow (window 2047)
    ctx = Hc[:, p:p + 5]
    p += 5
    mx.eval(ctx)
    order = VS if r % 2 == 0 else VS[::-1]
    for v in order:
        d = use(v)                               # lm_head: bound by bind() above
        t = time.perf_counter()
        got = d.draft_block(int(seq[p]), ctx, state[v], B, argmax, mx.int32)
        mx.eval(got)
        ms[v].append((time.perf_counter() - t) * 1000)
for v in VS:
    x = ms[v][3:]
    res["incr"][v] = dict(median_ms=statistics.median(x), ms=ms[v])
    print(f"block {v:12s} median {statistics.median(x):6.2f} ms (min {min(x):.2f}, {len(x)} blocks)", flush=True)

# --- staged profile (synced after every stage)
for v in [s for s in a.stages.split(",") if s]:
    d = use(v)
    cache = bind(d)
    mx.eval(d.draft_block(int(seq[n0]), Hc[:, max(0, n0 - 2047):n0], cache, B, argmax, mx.int32))
    T = {}

    def tm(k, f):
        t = time.perf_counter()
        y = f()
        mx.eval(y)
        T.setdefault(k, []).append((time.perf_counter() - t) * 1000)
        return y
    p = n0
    for it in range(12):
        ctx = Hc[:, p:p + 5]
        p += 5
        mx.eval(ctx)
        ids = mx.array([[int(seq[p])] + [d.config.mask_token_id] * (B - 1)])
        t0 = time.perf_counter()
        x = tm("embed", lambda: d._embed_input_tokens(ids))
        hc = tm("fc + hidden_norm (ctx rows)", lambda: d.hidden_norm(d.fc(ctx)))
        h = x
        for layer, c in zip(d.layers, cache):
            a_in = tm("layernorms", lambda: layer.input_layernorm(h))
            pk = tm("conv.prepare (kernel_projection + conv)", lambda: layer.attention_conv.prepare(a_in))
            att = tm("self_attn (q,k,v,o + ctx k,v + sdpa)", lambda: layer.self_attn(pk[0], hc, d.rope, c, None))
            h = tm("conv.finish + residual", lambda: h + layer.attention_conv.finish(att, pk[1]))
            m_n = tm("layernorms", lambda: layer.post_attention_layernorm(h))
            pm = tm("conv.prepare (kernel_projection + conv)", lambda: layer.mlp_conv.prepare(m_n))
            mm = tm("mlp (gate, up, down)", lambda: layer.mlp(pm[0]))
            h = tm("conv.finish + residual", lambda: h + layer.mlp_conv.finish(mm, pm[1]))
        hn = tm("final norm", lambda: d.norm(h)[:, 1:])
        lg = tm("lm_head (target, 7 rows)", lambda: d._logits(hn))
        tm("selector", lambda: d.candidate_selector.select(hn, lg, ids[:, 0], argmax))
        T.setdefault("TOTAL (staged, synced)", []).append((time.perf_counter() - t0) * 1000)
    out = {}
    print(f"stages {v}:")
    for k, x in T.items():
        n = len(x) // 12
        val = statistics.median(x[n * 2:]) * n
        out[k] = val
        print(f"  {k:44s} {val:7.2f} ms ({n} calls)")
    res["stages"][v] = out

res["peak_gb"] = mx.get_peak_memory() / 1e9
print(f"peak {res['peak_gb']:.2f} GB", flush=True)
if a.json:
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    json.dump(res, open(a.json, "w"), indent=1)
    print("wrote", a.json)
sys.stdout.flush()
os._exit(0)
