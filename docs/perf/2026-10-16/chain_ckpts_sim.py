"""An AppWorld-shaped episode through the real scheduler + prompt cache (tiny
random model, CPU, production knobs: slab 2048 / min 256 / stride 2048 / cap
12): checkpoints per entry with the chain policy off and on, and the entry's
size scaled to the 9B (32 KiB of KV per token, 49 MB per GDN state).
SIM_USER0=8000: the first user message an 8K document instead of 300 tokens
(the chain keeps that message's strides: review 2026-10-16 #6).
SIM_FRESH=1 (2026-10-17): also a FRESH entry (a MISS: nothing absorbed) for the
first prompt of a 60-turn episode past 50K tokens, with
QWEN_PROMPT_CACHE_CHAIN_FRESH_TOKENS at 0 (the cap policy, before) and 16384
(the chain policy)."""
import os, sys, random
_W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src", "worker")
sys.path[:0] = [_W, os.path.join(_W, "tests")]
import conftest  # noqa (CPU device)
from conftest import IM_START, build_lm, prompts, request, run, scheduler
import mlx.core as mx
import worker

worker.PROMPT_CACHE = True
lm = build_lm("dense", None, mx.float32)
A, NL, TH, NN, THC, IM_END = 11, 12, 13, 14, 15, 16
GEN = [IM_START, A, NL, TH, NL]


def episode(turns, seed, system=4700, user0=int(os.environ.get("SIM_USER0", "300"))):
    rnd = random.Random(seed)
    sys_ = [IM_START] + prompts([system], seed)[0]
    hist = sys_ + [IM_START] + prompts([user0], seed + 1)[0]
    out = []
    for i in range(turns):
        p = hist + GEN
        out.append(p)
        ans = prompts([50], seed + 100 + i)[0]
        tool = prompts([min(6000, int(rnd.lognormvariate(5.93, 1.0)) + 30)], seed + 200 + i)[0]
        hist = p[:-1] + [NN, THC, NN] + ans + [IM_END, NL] + [IM_START] + tool
    return out


def go(marks, turns, seed):
    worker.PROMPT_CACHE_CHAIN_MARKS = marks
    sched = scheduler(lm, max_batch=1)
    pc = sched.eng._pcache
    rows = []
    for i, ids in enumerate(episode(turns, seed)):
        r = request(ids, 1, f"t{i}")
        run(sched, [r])
        (e,) = pc.entries
        segs = {id(s): s for s, _, _ in e.chain}
        kv_tokens = sum(s.b - s.a for s in segs.values())
        rows.append((len(e.ids), len(e.ckpts), kv_tokens, r.reused))
    return rows


for turns, seed in ((30, 1), (60, 2)):
    off, on = go(-1, turns, seed), go(3, turns, seed)
    for (L, c0, kv, ru0), (_, c1, _, ru1) in list(zip(off, on))[4::5] + [(off[-1], on[-1])]:
        mb = lambda c: (kv * 32 * 1024 + (c + 1) * 49e6) / 1e6
        print(f"turns={turns} L={L:6d} ckpts off={c0:2d} on={c1:2d}  9B MB off={mb(c0):6.0f} on={mb(c1):6.0f}  "
              f"reused off={ru0} on={ru1}")


if os.environ.get("SIM_FRESH"):
    ps = episode(200, 3)
    ids = next(p for p in ps if len(p) >= 50000)
    for fresh in (0, 16384):
        worker.PROMPT_CACHE_CHAIN_FRESH_TOKENS = fresh
        sched = scheduler(lm, max_batch=1)
        run(sched, [request(ids, 1, "fresh")])
        (e,) = sched.eng._pcache.entries
        mb = (len(e.ids) * 32 * 1024 + (len(e.ckpts) + 1) * 49e6) / 1e6
        print(f"fresh L={len(e.ids)} CHAIN_FRESH_TOKENS={fresh:5d} ckpts={len(e.ckpts):2d}  "
              f"9B MB={mb:6.0f} (states {(len(e.ckpts) + 1) * 49:4.0f})")
