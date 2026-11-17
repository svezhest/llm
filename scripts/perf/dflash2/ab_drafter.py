"""In-process A/B of DFlash2 drafter variants on the 27B (a sibling of ab27.py):
the real Engine and _Scheduler, one process, variants interleaved per (round,
prompt) with rotation, medians over rounds (docs/measuring.md).

Variants (comma separated):
  plain             no DFlash (eng.draft = None)
  d<bits>[+mr][+cc] DFlash with the drafter quantized to <bits> (the engine's
                    own when it is QWEN_DRAFT_BITS, else loaded next to it with
                    the worker's code: _load_drafter); +mr QWEN_DRAFT_MRQMM,
                    +cc QWEN_DRAFT_COMPILE. Verify knobs as the engine loaded them.

Prompts: the three ~2K-token prompts of prompts2k.py (code / en / ru) and the
first --tasks short code TASKS. Identity: --ident variants vs plain on the 16 TASKS.

  cd src/worker && QWEN_MODEL=../../weights/qwen38-27b-mlx-4bit QWEN_TEXT_ONLY=1 QWEN_MRQMM4=1 \\
    QWEN_KV_BITS=0 QWEN_MAX_BATCH=1 QWEN_WIRED_LIMIT_GB=27 QWEN_DRAFT=../../weights/qwen38-27b-dflash2 \\
    PYTHONPATH=. .venv/bin/python ../../scripts/perf/dflash2/ab_drafter.py --rounds 3 \\
    --variants plain,d4,d4+mr --json $PERF_OUT/drafter/ab.json
"""
import argparse
import collections
import json
import logging
import os
import statistics
import sys
import time
import zlib

sys.path.insert(0, os.getcwd())
import worker as W  # noqa: E402
import worker_pb2 as pb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "prof"))
sys.path.insert(0, HERE)
from prompts2k import SHORT, prompts  # noqa: E402
from wtest import TASKS  # noqa: E402  (no MLX there: gRPC client module, TASKS only)

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--variants", default="plain,d4,d4+mr")
ap.add_argument("--ident", default="d4,d4+mr", help="DFlash variants checked against plain on all 16 tasks")
ap.add_argument("--tasks", type=int, default=0, help="speed phase: + the first N wtest TASKS")
ap.add_argument("--no-2k", action="store_true")
ap.add_argument("--short", action="store_true", help="+ two short prose prompts (en / ru, prompts2k.SHORT)")
ap.add_argument("--sampling", default="", help="T,top_p,top_k (e.g. 0.6,0.95,20): sampled requests, "
                "seeded per (round, prompt); DFlash variants with QWEN_DRAFT_SAMPLING=1. Variant flag "
                "+tk = QWEN_SAMPLER_TOPK_FAST (also plain+tk)")
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--thinking", action="store_true", help="enable_thinking=True (default off, as the 9B runs)")
ap.add_argument("--thinking-budget", type=int, default=-1,
                help="with --thinking: a thinking budget (ThinkingBudgetCriteria, as the handler builds it)")
ap.add_argument("--ident-tasks", type=int, default=16)
ap.add_argument("--skip-speed", action="store_true")
ap.add_argument("--skip-ident", action="store_true")
ap.add_argument("--json", default="")
a = ap.parse_args()
assert os.environ.get("QWEN_DRAFT"), "needs QWEN_DRAFT"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")  # -> stderr

import mlx.core as mx  # noqa: E402

t_load = time.time()
eng = W.Engine(os.environ["QWEN_MODEL"])
eng.load()
DRAFT = eng.draft
assert DRAFT is not None, "drafter not loaded (QWEN_KV_BITS=0?)"
tok = getattr(eng.processor, "tokenizer", eng.processor)
cfg = DRAFT.config
print(f"loaded in {time.time() - t_load:.0f} s; active {mx.get_active_memory() / 1e9:.2f} GB; "
      f"max_batch {eng.max_batch}; drafter {type(DRAFT).__name__} block {cfg.block_size} "
      f"runtime {getattr(cfg, 'runtime_block_size', None)} window {cfg.sliding_window} "
      f"layers {list(cfg.target_layer_ids)}; draft bits {W.DRAFT_BITS}; "
      f"verify exact max T {W.VERIFY_EXACT_MAX_T}; VERIFY_MR {W.VERIFY_MR}", flush=True)
from qwen_worker import config as cfg  # noqa: E402
from qwen_worker.engine import _load_drafter, _setup_drafter_fast  # noqa: E402

DRAFTS = {W.DRAFT_BITS: DRAFT}
if a.sampling:
    W.DRAFT_SAMPLING = True                     # read at admission (sched_admit)
print("drafter fast paths (engine's):", _setup_drafter_fast(DRAFT, True, keep_fp32=True), flush=True)


def parse(v):
    head, *flags = v.split("+")
    return (None if head == "plain" else int(head[1:])), "mr" in flags, "cc" in flags


for v in set(a.variants.split(",")) | set(a.ident.split(",")):
    if v and parse(v)[0] is not None and parse(v)[0] not in DRAFTS:
        bits = parse(v)[0]
        m0 = mx.get_active_memory()
        d, wide = _load_drafter(eng.model, os.environ["QWEN_DRAFT"], bits)
        d.prefer_requested_block_size = DRAFT.prefer_requested_block_size
        print(f"d{bits}:", _setup_drafter_fast(d, wide, keep_fp32=True),
              f"+{(mx.get_active_memory() - m0) / 1e9:.2f} GB", flush=True)
        DRAFTS[bits] = d


def set_variant(v):
    """Flip what the scheduler reads (idle scheduler: between requests)."""
    W.SAMPLER_TOPK_FAST = "tk" in v.split("+")[1:]       # read at each request's sampler
    bits, mr, cc = parse(v)
    if bits is None:
        eng.draft = None
        return
    eng.draft, eng.draft_wide = DRAFTS[bits], True
    cfg.DRAFT_MRQMM, cfg.DRAFT_COMPILE = mr, cc


def request(text, rid):
    samp = dict(temperature=0.0)
    if a.sampling:
        t, tp, tk = a.sampling.split(",")
        samp = dict(temperature=float(t), top_p=float(tp), top_k=int(tk),
                    seed=zlib.crc32(rid.rsplit("-", 1)[0].encode()) % (1 << 31))   # the same seed for every variant
    req = pb.Request(id=rid, messages=[pb.Message(role="user", content=text)],
                     params=pb.Params(presence_penalty=0.0, max_tokens=a.max_tokens,
                                      chat_template_kwargs_json=json.dumps({"enable_thinking": a.thinking}),
                                      **samp))
    formatted, _ = eng._render(req, [])
    ids = tok.encode(formatted)
    p = req.params
    sampling = dict(W.SAMPLING_KWARGS)
    for f in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "seed"):
        if p.HasField(f):
            sampling[f] = getattr(p, f)
    ctl = W._Control(p, tok.stopping_criteria.eos_token_ids, tok, eng.vocab)
    think = None
    if a.thinking and a.thinking_budget >= 0:
        from mlx_vlm.utils import ThinkingBudgetCriteria
        think = ThinkingBudgetCriteria(tokenizer=tok, thinking_budget=a.thinking_budget,
                                       thinking_end_token=W.THINK_CLOSE, thinking_start_token=W.THINK_OPEN,
                                       enable_thinking=True, prompt_preopens_thinking=True)
    return W._Req(id=rid, p=p, ids=ids, max_tokens=eng._limit(p, len(ids)), sampling=sampling, ctl=ctl,
                  formatted=formatted, thinking=think)


def run(text, rid, spec):
    D = eng.draft or DRAFT
    D.accept_lens, D.draft_lens = [], []
    r = request(text, rid)
    t0 = time.perf_counter()
    eng.sched.submit(r)
    toks, times = [], []
    while True:
        ev = r.out.get()
        if ev[0] != "tok":
            break
        toks.append(ev[1])
        times.append(time.perf_counter())
    n = len(toks)
    dt = times[-1] - times[0] if n > 2 else 0.0
    out = dict(tokens=toks, n=n, prompt=len(r.ids), ttft=times[0] - t0 if times else None,
               tps=(n - 1) / dt if dt else 0.0, finish=[str(x) for x in ev[1:2]])
    if spec:
        acc, dl = [int(x) for x in D.accept_lens], [int(x) for x in D.draft_lens]
        if acc:
            out.update(rounds=len(acc), drafted=sum(dl), accepted=sum(acc),
                       accept=100.0 * sum(acc) / max(1, sum(dl)), per_round=sum(acc) / len(acc),
                       round_ms=1000.0 * dt / len(acc),
                       hist=dict(sorted(collections.Counter(acc).items())),
                       blocks=dict(sorted(collections.Counter(d + 1 for d in dl).items())))
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


res = {"variants": a.variants, "rounds": a.rounds, "thinking": a.thinking, "max_tokens": a.max_tokens,
       "thinking_budget": a.thinking_budget,
       "model": os.environ["QWEN_MODEL"], "draft": os.environ["QWEN_DRAFT"], "draft_bits": W.DRAFT_BITS,
       "verify_mr": W.VERIFY_MR, "draft_block": cfg.DRAFT_BLOCK,
       "speed": [], "ident": []}
VS = a.variants.split(",")
PROMPTS = (([] if a.no_2k else prompts(tok)) + (list(SHORT) if a.short else [])
           + [(f"code{i}", t) for i, t in enumerate(TASKS[:a.tasks])])
NT = len(PROMPTS)
res["prompts"] = [n for n, _ in PROMPTS]
if not a.skip_speed:
    set_variant("plain")
    run(TASKS[0], "warm", False)                           # compile / warm everything once
    for v in VS:
        set_variant(v)
        o = run("Say OK.", f"warm-{v}", v != "plain")
        print(f"warm {v}: n={o['n']} ok", flush=True)
    for rnd in range(a.rounds):
        for ti, (pname, text) in enumerate(PROMPTS):
            order = VS[rnd % len(VS):] + VS[:rnd % len(VS)]     # rotate: no variant always first
            if (rnd + ti) % 2:
                order = order[::-1]
            for v in order:
                set_variant(v)
                o = run(text, f"s{rnd}-{pname}-{v}", v != "plain")
                o.update(variant=v, round=rnd, task=ti, prompt_name=pname)
                res["speed"].append(o)
                print(f"r{rnd} {pname:6s} {v:10s} n={o['n']:3d} tps={o['tps']:6.2f} "
                      f"accept={o.get('accept', 0):5.1f}% per_round={o.get('per_round', 0):4.2f} "
                      f"round_ms={o.get('round_ms', 0):6.1f} hist={o.get('hist', '')}", flush=True)
    plain = {(o["round"], o["task"]): o["tokens"] for o in res["speed"] if o["variant"] == "plain"}
    print("\n== speed: median decode tok/s (median over tasks of per-task medians over rounds) ==")
    summ = {}
    for v in VS:
        rows = [o for o in res["speed"] if o["variant"] == v]
        per_task = [med([o["tps"] for o in rows if o["task"] == t]) for t in range(NT)]
        same = sum(o["tokens"] == plain.get((o["round"], o["task"])) for o in rows)
        summ[v] = dict(tps=med(per_task), per_task=per_task, accept=med([o.get("accept") for o in rows]),
                       per_round=med([o.get("per_round") for o in rows]),
                       round_ms=med([o.get("round_ms") for o in rows]),
                       same_as_plain=f"{same}/{len(rows)}" if plain else None)
        s = summ[v]
        print(f"{v:10s} tps {s['tps']:6.2f}  per prompt " + " ".join(f"{x:5.1f}" for x in per_task)
              + f"  accept {s['accept'] or 0:5.1f}%  drafts/round {s['per_round'] or 0:4.2f}"
              + f"  round {s['round_ms'] or 0:6.1f} ms  == plain {s['same_as_plain']}", flush=True)
    res["speed_summary"] = summ
    ref_v = next((v for v in VS if v != "plain"), None)
    tp = {(o["variant"], o["round"], o["task"]): o["tps"] for o in res["speed"]}
    print(f"\n== paired ratio vs {ref_v} (median over rounds x prompts; per prompt) ==")
    for v in VS:
        rat = [tp[(v, r, t)] / tp[(ref_v, r, t)] for r in range(a.rounds) for t in range(NT)
               if tp.get((ref_v, r, t))]
        per = [med([tp[(v, r, t)] / tp[(ref_v, r, t)] for r in range(a.rounds) if tp.get((ref_v, r, t))])
               for t in range(NT)]
        summ[v].update(ratio=med(rat), ratio_per_prompt=per)
        print(f"{v:10s} x{med(rat):.3f}  per prompt " + " ".join(f"x{x:.3f}" for x in per), flush=True)

if not a.skip_ident:
    print(f"\n== identity vs plain greedy, {a.ident_tasks} tasks ==", flush=True)
    for ti, text in enumerate(TASKS[:a.ident_tasks]):
        set_variant("plain")
        ref = run(text, f"i-t{ti}-plain", False)["tokens"]
        for v in a.ident.split(","):
            set_variant(v)
            got = run(text, f"i-t{ti}-{v}", True)["tokens"]
            div = next((i for i, (x, y) in enumerate(zip(ref, got)) if x != y),
                       None if len(ref) == len(got) else min(len(ref), len(got)))
            res["ident"].append(dict(task=ti, variant=v, same=div is None, diverge=div, n=len(ref)))
            print(f"t{ti} {v}: {'same' if div is None else f'diverges at {div}'} (n={len(ref)})", flush=True)
    for v in a.ident.split(","):
        rows = [o for o in res["ident"] if o["variant"] == v]
        print(f"{v:8s} identical {sum(o['same'] for o in rows)}/{len(rows)}; divergence at "
              f"{[o['diverge'] for o in rows if not o['same']]}", flush=True)

res["peak_gb"] = mx.get_peak_memory() / 1e9
res["active_gb"] = mx.get_active_memory() / 1e9
print(f"peak {res['peak_gb']:.2f} GB, active {res['active_gb']:.2f} GB", flush=True)
if a.json:
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    for o in res["speed"]:
        o.pop("tokens", None)
    json.dump(res, open(a.json, "w"), indent=1)
    print("wrote", a.json)
sys.stdout.flush()
os._exit(0)          # the scheduler thread never returns (worker.py: MLX stream teardown)
