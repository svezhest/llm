"""In-process A/B of DFlash2 speculative decoding on the 27B (5-bit) target: the
real Engine and _Scheduler (no gRPC), one process, variants interleaved per
(round, task), medians over rounds (CLAUDE.md "Measuring performance"). A
sibling of scripts/perf/verify_mrqmm/dflash_ab.py with per-round statistics.

Variants (comma separated):
  plain          no DFlash (eng.draft = None): the plain greedy loop
  plainnf        plain with the fused GDN in-projection off (QWEN_GDN_FUSE=0 at
                 run time); --ident plainnf: the fusion's identity at B = 1
                 (bit-exact claim: a width-grouped or uniform fusion == the
                 library's four calls, 2026-11-15)
  e<cap>/<blk>   DFlash; verify blocks of <= cap tokens through mlx-vlm's exact
                 verify kernels (QWEN_VERIFY_EXACT_MAX_T), longer ones through
                 mx.quantized_matmul (e0 = always quantized_matmul)
                 <blk>: ad = adaptive up to the drafter's runtime block (DFlash2: 5),
                 adN = adaptive with ceiling N, N = fixed N (1 anchor + N-1 drafts)
  m<cap>/<blk>   the same with QWEN_VERIFY_MRQMM on (verify blocks through mrqmm
                 where it covers the module; needs a kernel for the model's bits)
  ...:va         the verify attention through the one-pass kernel (QWEN_VERIFY_ATT=2,
                 vatt.py; the worker loaded with QWEN_VERIFY_ATT=2), else the vector
                 kernel's path (QWEN_VERIFY_ATT=1)

Per request: decode tok/s (tokens after the first / time from the first to the
last token), rounds, accepted drafts per round, ms per round.

  cd src/worker && QWEN_MODEL=../../weights/qwen38-27b-mlx-5bit QWEN_KV_BITS=0 \\
    QWEN_DRAFT=../../weights/qwen38-27b-dflash2 PYTHONPATH=. .venv/bin/python \\
    ../../scripts/perf/dflash2/ab27.py --rounds 3 --json $PERF_OUT/dflash2/ab27.json
"""
import argparse
import collections
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
from wtest import TASKS  # noqa: E402  (no MLX there: gRPC client module, TASKS only)

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--variants", default="plain,e8/ad,e8/3,e8/5,e8/8,e0/8")
ap.add_argument("--ident", default="e8/ad,e8/8", help="DFlash variants checked against plain on all 16 tasks")
ap.add_argument("--tasks", type=int, default=5, help="speed phase: the first N wtest TASKS")
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
MR0 = (W.VERIFY_MR, W.VERIFY_MR_HEAD)
VA0 = W._VATT


def set_variant(v):
    """Flip the globals the scheduler reads (idle scheduler: between requests)."""
    W.GDN_FUSE = v != "plainnf"
    if v in ("plain", "plainnf"):
        eng.draft = None
        return
    v, _, att = v.partition(":")
    assert att in ("", "va"), att
    assert VA0 or not att, "':va' needs QWEN_VERIFY_ATT=2 (self-test passed)"
    W._VATT = att == "va"
    knob, blk = v.split("/")
    eng.draft = DRAFT
    W.VERIFY_MR, W.VERIFY_MR_HEAD = MR0 if knob[0] == "m" else (False, False)
    W.VERIFY_EXACT_MAX_T = int(knob[1:])
    # ad = adaptive up to the runtime block; adN = adaptive with ceiling N; N = fixed N
    W.DRAFT_BLOCK = int(blk[2:] or 0) if blk.startswith("ad") else int(blk)
    DRAFT.prefer_requested_block_size = not blk.startswith("ad")


def request(text, rid):
    req = pb.Request(id=rid, messages=[pb.Message(role="user", content=text)],
                     params=pb.Params(temperature=0.0, presence_penalty=0.0, max_tokens=a.max_tokens,
                                      chat_template_kwargs_json=json.dumps({"enable_thinking": a.thinking})))
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
    DRAFT.accept_lens, DRAFT.draft_lens = [], []
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
        acc, dl = [int(x) for x in DRAFT.accept_lens], [int(x) for x in DRAFT.draft_lens]
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
       "speed": [], "ident": []}
VS = a.variants.split(",")
NT = a.tasks
if not a.skip_speed:
    set_variant("plain")
    run(TASKS[0], "warm", False)                           # compile / warm everything once
    for v in VS:
        set_variant(v)
        o = run("Say OK.", f"warm-{v}", not v.startswith("plain"))
        print(f"warm {v}: n={o['n']} ok", flush=True)
    for rnd in range(a.rounds):
        for ti, text in enumerate(TASKS[:NT]):
            order = VS[rnd % len(VS):] + VS[:rnd % len(VS)]     # rotate: no variant always first
            if (rnd + ti) % 2:
                order = order[::-1]
            for v in order:
                set_variant(v)
                o = run(text, f"s{rnd}-code{ti}-{v}", not v.startswith("plain"))
                o.update(variant=v, round=rnd, task=ti)
                res["speed"].append(o)
                print(f"r{rnd} code{ti} {v:8s} n={o['n']:3d} tps={o['tps']:6.2f} "
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
        print(f"{v:8s} tps {s['tps']:6.2f}  per task " + " ".join(f"{x:5.1f}" for x in per_task)
              + f"  accept {s['accept'] or 0:5.1f}%  drafts/round {s['per_round'] or 0:4.2f}"
              + f"  round {s['round_ms'] or 0:6.1f} ms  == plain {s['same_as_plain']}", flush=True)
    res["speed_summary"] = summ

if not a.skip_ident:
    print(f"\n== identity vs plain greedy, {a.ident_tasks} tasks ==", flush=True)
    for ti, text in enumerate(TASKS[:a.ident_tasks]):
        set_variant("plain")
        ref = run(text, f"i-t{ti}-plain", False)["tokens"]
        for v in a.ident.split(","):
            set_variant(v)
            got = run(text, f"i-t{ti}-{v}", not v.startswith("plain"))["tokens"]
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
