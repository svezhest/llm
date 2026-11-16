"""Are the DFlash-vs-plain divergences near-ties? Plain greedy decode (library
only, single-row qmv, target in fp16) of the tasks ab27.py's identity phase
reported as diverging: the top-1 / top-2 logit gap at the diverging token
against the gaps of all steps.

  cd src/worker && PYTHONPATH=. .venv/bin/python ../../scripts/perf/dflash2/diverge_margin.py \\
    ../../docs/perf/2026-11-10/raw/ab27_kern5_thinking.json
DM_MODEL=<weights dir> picks another checkpoint (default the 5-bit 27B).
"""
import json
import os
import statistics
import sys

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.models.cache import make_prompt_cache

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "prof"))
from wtest import TASKS  # noqa: E402

res = json.load(open(sys.argv[1]))
thinking = bool(res.get("thinking"))
bad = [(o["task"], o["variant"], o["diverge"]) for o in res["ident"] if not o["same"]]
prompts = {o["task"]: o["prompt"] for o in res["speed"] if "prompt" in o}
model, proc = load(os.environ.get("DM_MODEL") or os.environ.get("MODEL")
                   or os.path.join(ROOT, "weights/qwen38-27b-mlx-5bit"))
lm = model.language_model
lm.apply(lambda a: a.astype(mx.float16) if a.dtype == mx.bfloat16 else a)
mx.eval(lm.parameters())
tok = getattr(proc, "tokenizer", proc)
out = []
allgaps = []
for ti, variant, at in bad:
    s = tok.apply_chat_template([{"role": "user", "content": TASKS[ti]}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=thinking)
    ids = tok.encode(s)
    cache = make_prompt_cache(lm)
    lm._rope_deltas = lm._position_ids = None
    y = mx.array([ids])
    gaps, toks = [], []
    for _ in range(at + 1):
        lg = lm(y, cache=cache).logits[0, -1].astype(mx.float32)
        top = mx.sort(mx.topk(lg, 2))
        t = mx.argmax(lg)
        mx.eval(top, t)
        gaps.append(float(top[1].item() - top[0].item()))
        toks.append(t.item())
        y = mx.array([[toks[-1]]])
    allgaps += gaps[:-1]
    o = dict(task=ti, variant=variant, diverge=at, prompt_tokens=len(ids),
             prompt_matches_worker=prompts.get(ti) == len(ids) if ti in prompts else None,
             gap_at_diverge=round(gaps[at], 4), median_gap_before=round(statistics.median(gaps[:-1]), 3),
             steps_with_smaller_gap=sum(g <= gaps[at] for g in gaps[:-1]), steps=at,
             token=tok.decode([toks[at]]))
    out.append(o)
    print(o, flush=True)
qs = statistics.quantiles(allgaps, n=100)
print(f"all {len(allgaps)} earlier steps: gap p1 {qs[0]:.3f} p5 {qs[4]:.3f} p50 {qs[49]:.3f}")
json.dump(dict(source=sys.argv[1], rows=out, gap_p1=qs[0], gap_p5=qs[4], gap_p50=qs[49]),
          open(sys.argv[1].replace(".json", "_margins.json"), "w"), indent=1)
