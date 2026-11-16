"""Feasibility probe: stock mlx-vlm 0.7.6 generation (its own loop, no worker) of the
27B target with its native MTP head as the drafter (weights/qwen38-27b-mtp, made by
split_mtp.py), vs the same loop without a drafter. Acceptance and decode tok/s per
prompt class (code / en / ru of the bench27 prompt set), variants interleaved per round.

  cd src/worker && BENCH27=<dir with prompts.py + prompts/manifest.json> \
    .venv/bin/python ../../scripts/perf/mtp/stock_mtp.py --model ../../weights/qwen38-27b-mlx-4bit \
    --draft ../../weights/qwen38-27b-mtp --variants plain,mtp2,mtp3,mtp4 --json $PERF_OUT/mtp/stock.json
mtpN = block of N tokens (1 anchor + N-1 MTP drafts, chained).
"""
import argparse
import json
import os
import statistics
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--draft", required=True)
ap.add_argument("--variants", default="plain,mtp2,mtp3,mtp4")
ap.add_argument("--prompts", default="code-2k-0,en-2k-0,ru-2k-0")
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--max-tokens", type=int, default=128)
ap.add_argument("--fp16", type=int, default=1, help="cast the LM bf16 -> fp16 as the worker does")
ap.add_argument("--json", default="")
a = ap.parse_args()

sys.path.insert(0, os.environ["BENCH27"])
import prompts as P  # noqa: E402
import mlx.core as mx  # noqa: E402
from mlx_vlm import load  # noqa: E402
from mlx_vlm.generate import stream_generate  # noqa: E402
from mlx_vlm.speculative.drafters import load_drafter  # noqa: E402

model, proc = load(a.model, lazy=False)
if a.fp16:
    lm = model.language_model
    lm.apply(lambda x: x.astype(mx.float16) if x.dtype == mx.bfloat16 else x)
    mx.eval(lm.parameters())
draft, kind = load_drafter(a.draft)
if a.fp16:
    draft.apply(lambda x: x.astype(mx.float16) if x.dtype == mx.bfloat16 else x)
mx.eval(draft.parameters())
tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
man = P.load_manifest(os.path.join(os.environ["BENCH27"], "prompts", "manifest.json"))
prs = {p["id"]: p for p in man["prompts"]}
texts = {}
for pid in a.prompts.split(","):
    texts[pid] = tok.apply_chat_template(P.messages(man, prs[pid]), add_generation_prompt=True,
                                         tokenize=False, enable_thinking=False)
print(f"loaded; active {mx.get_active_memory() / 1e9:.2f} GB; drafter {kind} {type(draft).__name__}", flush=True)


def run(pid, v):
    kw = dict(max_tokens=a.max_tokens, temperature=0.0, prefill_step_size=2048)
    if v != "plain":
        kw.update(draft_model=draft, draft_kind="mtp", draft_block_size=int(v[3:]))
        draft.accept_lens, draft.draft_lens = [], []
    toks, times = [], []
    t0 = time.perf_counter()
    for r in stream_generate(model, proc, texts[pid], **kw):
        if r.token is not None:
            toks.append(int(r.token))
            times.append(time.perf_counter())
    n = len(toks)
    dt = times[-1] - times[0] if n > 2 else 0
    o = dict(prompt=pid, variant=v, n=n, tps=(n - 1) / dt if dt else 0, ttft=times[0] - t0, tokens=toks)
    if v != "plain":
        acc, dl = list(draft.accept_lens), list(draft.draft_lens)
        o.update(rounds=len(acc), accepted=sum(acc), drafted=sum(dl),
                 accept=100.0 * sum(acc) / max(1, sum(dl)), per_round=sum(acc) / max(1, len(acc)))
    mx.clear_cache()
    return o


VS = a.variants.split(",")
res = []
run(list(texts)[0], "plain")
for v in VS[1:]:
    run(list(texts)[0], v)
for rnd in range(a.rounds):
    for i, pid in enumerate(texts):
        order = VS[rnd % len(VS):] + VS[:rnd % len(VS)]
        if (rnd + i) % 2:
            order = order[::-1]
        for v in order:
            o = run(pid, v)
            o["round"] = rnd
            res.append(o)
            print(f"r{rnd} {pid:10s} {v:6s} n={o['n']} tps={o['tps']:6.2f} accept={o.get('accept', 0):5.1f}% "
                  f"per_round={o.get('per_round', 0):4.2f}", flush=True)
print("\n== median tok/s, acceptance; identity vs plain ==")
summ = {}
for pid in texts:
    ref = next(o["tokens"] for o in res if o["prompt"] == pid and o["variant"] == "plain")
    for v in VS:
        rows = [o for o in res if o["prompt"] == pid and o["variant"] == v]
        s = dict(tps=statistics.median(o["tps"] for o in rows),
                 accept=statistics.median(o.get("accept", 0) for o in rows),
                 per_round=statistics.median(o.get("per_round", 0) for o in rows),
                 same=sum(o["tokens"] == ref for o in rows))
        summ[f"{pid}/{v}"] = s
        print(f"{pid:10s} {v:6s} tps {s['tps']:6.2f} accept {s['accept']:5.1f}% drafts/round "
              f"{s['per_round']:4.2f} same {s['same']}/{len(rows)}")
if a.json:
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    with open(a.json, "w") as f:
        json.dump(dict(args=vars(a), runs=res, summary=summ), f, indent=1)
