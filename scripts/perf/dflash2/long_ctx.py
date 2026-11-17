"""A long conversation on the 27B with the drafter and the prompt cache: does
it fit, and do the turns after the first hit? (QWEN_PROMPT_CACHE_MOVE: one KV
copy; with the copies an entry past ~40K tokens was "not kept".)

Real Engine + _Scheduler. A system document of --tokens tokens (the repo's
docs, cut by tokens), then --turns turns: turn 1 is cold, each next one is
the previous prompt + its answer + a follow-up. --cold-ref: the last turn once
more with prompt_cache "off" (a cold prefill of the whole prompt: its answer
must equal the hit's when the hit resumed at a checkpoint — thinking on; a hit
through a reply is the batched-decode class, compared by tokens only).

Per turn: prompt tokens, reused, TTFT, tok/s, DFlash rounds / accepted / ms per
round, the peak (mx.get_peak_memory, reset at admission; split at the first
token into prefill and decode) and what the cache holds. --ab / --alt: plain
decode and the library's verify attention against the loaded configuration.

  cd src/worker && QWEN_MODEL=../../weights/qwen38-27b-mlx-5bit QWEN_KV_BITS=0 QWEN_MRQMM5=1 \\
    QWEN_DRAFT=../../weights/qwen38-27b-dflash2 QWEN_PROMPT_CACHE=1 QWEN_PROMPT_CACHE_MOVE=1 \\
    QWEN_PROMPT_CACHE_GB=6 QWEN_PROMPT_CACHE_CKPTS=4 QWEN_MAX_BATCH=1 QWEN_WIRED_LIMIT_GB=27 \\
    QWEN_CACHE_LIMIT_GB=1 QWEN_SDPA_FUSED_MIN_KV=8192 QWEN_PREFILL_BUDGET=16777216 QWEN_PRESENCE_PENALTY=0 \\
    QWEN_MAX_CONTEXT=65536 PYTHONPATH=. .venv/bin/python ../../scripts/perf/dflash2/long_ctx.py \\
    --tokens 60000 --json $PERF_OUT/dflash2/long_ctx.json
"""
import argparse
import glob
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import worker as W  # noqa: E402
import worker_pb2 as pb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "prof"))
from doc_notes import NOTES  # noqa: E402
from wtest import TASKS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, default=60000, help="the system document's length")
ap.add_argument("--turns", type=int, default=3)
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--no-thinking", action="store_true")
ap.add_argument("--cold-ref", action="store_true")
ap.add_argument("--ab", type=int, default=0,
                help="after the turns: N rounds of the last prompt (prompt_cache read: resumes from the "
                     "entry, stores nothing) as plain decode / DFlash with the library's per-token verify "
                     "attention / DFlash with the vector attention kernel (QWEN_VERIFY_ATT)")
ap.add_argument("--ab-variants", default="plain,rows,vector",
                help="--ab's variants, in rotation: plain / rows / vector (QWEN_VERIFY_ATT=1) / vatt (=2, "
                     "needs the worker loaded with QWEN_VERIFY_ATT=2); a '+d' suffix: decode steps through "
                     "vatt too (QWEN_DECODE_ATT, loaded with it), else MLX's decode SDPA")
ap.add_argument("--ab-mode", default="read",
                help="--ab's prompt_cache mode: read (a copy of the entry, nothing stored) or '' (the "
                     "default: with QWEN_PROMPT_CACHE_MOVE the entry moves into the row and comes back "
                     "-- one KV copy, for contexts where two do not fit)")
ap.add_argument("--alt", choices=["rows", "plain"], default="",
                help="even turns differently, odd ones as loaded (speeds at a context too long for --ab's "
                     "second KV copy): rows = the library's one-query verify attention (QWEN_VERIFY_ATT "
                     "off), plain = no drafter")
ap.add_argument("--gc", action="store_true", help="gc.collect() after every request, with the bytes it freed")
ap.add_argument("--json", default="")
a = ap.parse_args()
assert os.environ.get("QWEN_PROMPT_CACHE") == "1"


class Grab(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, rec):
        m = rec.getMessage()
        if "prompt cache" in m or "dflash:" in m:
            self.lines.append(m[:260])


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
grab = Grab()
logging.getLogger("qwen.worker").addHandler(grab)

import mlx.core as mx  # noqa: E402

eng = W.Engine(os.environ["QWEN_MODEL"])
eng.load()
DRAFT = eng.draft
tok = getattr(eng.processor, "tokenizer", eng.processor)
think_close = tok.convert_tokens_to_ids("</think>")
text = ""
for f in list(NOTES) + sorted(
        glob.glob(os.path.join(ROOT, "docs/perf/2026-10-1*/*.md"))):
    text += open(os.path.join(ROOT, f)).read() + "\n\n"
    if len(text) > 6 * a.tokens:
        break
ids = tok.encode(text)
assert len(ids) >= a.tokens, f"only {len(ids)} tokens of documents"
DOC = ("You are a coding assistant. Project notes follow; use them only if relevant.\n\n"
       + tok.decode(ids[:a.tokens]))
FOLLOW = ["Now add type hints and a short docstring, and handle invalid input explicitly.",
          "Add three unit tests for it with pytest.",
          "Make it a small command-line tool with argparse."]
pc = eng._pcache
print(f"loaded; drafter {'on' if DRAFT is not None else 'OFF'}, move: {eng.sched.move_state()}; "
      f"active {mx.get_active_memory() / 2**30:.2f} GiB", flush=True)


def run(messages, rid, mode=""):
    grab.lines = []
    if DRAFT is not None:
        DRAFT.accept_lens, DRAFT.draft_lens = [], []
    req = pb.Request(id=rid, messages=[pb.Message(role=r, content=c) for r, c in messages],
                     params=pb.Params(temperature=0.0, presence_penalty=0.0, max_tokens=a.max_tokens,
                                      prompt_cache=mode, chat_template_kwargs_json=json.dumps(
                                          {"enable_thinking": not a.no_thinking})))
    formatted, _ = eng._render(req, [])
    pids = tok.encode(formatted)
    p = req.params
    sampling = dict(W.SAMPLING_KWARGS)
    for f in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "seed"):
        if p.HasField(f):
            sampling[f] = getattr(p, f)
    ctl = W._Control(p, tok.stopping_criteria.eos_token_ids, tok, eng.vocab)
    r = W._Req(id=rid, p=p, ids=pids, max_tokens=eng._limit(p, len(pids)), sampling=sampling, ctl=ctl,
               formatted=formatted)
    t0 = time.perf_counter()
    eng.sched.submit(r)
    toks, times, end = [], [], None
    pre_peak = act0 = 0
    while True:
        ev = r.out.get()
        if ev[0] != "tok":
            end = ev
            break
        if not toks:                             # decode's own peak: counters only, no MLX op
            pre_peak, act0 = mx.get_peak_memory(), mx.get_active_memory()
            mx.reset_peak_memory()
        toks.append(ev[1])
        times.append(time.perf_counter())
    dec_peak = mx.get_peak_memory()
    time.sleep(0.2)                              # the row's store runs right after its last event
    while eng.sched.task is not None or eng.sched.rows:
        time.sleep(0.05)
    if a.gc:                                     # a leak that a collection frees = a reference cycle
        import gc
        before = mx.get_active_memory()
        n = gc.collect()
        print(f"      gc: {n} objects, active {before / 2**30:.2f} -> {mx.get_active_memory() / 2**30:.2f} GiB",
              flush=True)
    acc = [int(x) for x in DRAFT.accept_lens] if DRAFT is not None and eng.draft is not None else []
    dt = times[-1] - times[0] if len(toks) > 2 else 0.0
    return dict(id=rid, prompt=len(pids), reused=r.reused, n=len(toks), tokens=toks, end=list(end),
                ttft=round(times[0] - t0, 2) if times else None,
                tps=round((len(toks) - 1) / dt, 2) if dt else 0.0, rounds=len(acc),
                accepted=round(sum(acc) / len(acc), 2) if acc else None,
                peak_gib=round(max(pre_peak, dec_peak) / 2**30, 2),
                prefill_peak_gib=round(pre_peak / 2**30, 2), decode_peak_gib=round(dec_peak / 2**30, 2),
                active_first_gib=round(act0 / 2**30, 2),
                round_ms=round(1e3 * dt / len(acc), 1) if acc and dt else None,
                active_gib=round(mx.get_active_memory() / 2**30, 2),
                cache_gib=round(pc.bytes / 2**30, 2), entries=[len(e.ids) for e in pc.entries],
                log=list(grab.lines))


def answer(toks):
    cut = toks.index(think_close) + 1 if think_close in toks else 0 if a.no_thinking else len(toks)
    return tok.decode(toks[cut:]).strip() or "(no answer yet)"


def show(o, same=None):
    print(f"{o['id']:7s} prompt {o['prompt']:6d} reused {o['reused']:6d} ttft {o['ttft'] or -1:7.2f}s "
          f"tps {o['tps']:5.2f} n {o['n']:4d} rounds {o['rounds']:3d} accepted/round {o['accepted']} "
          f"round {o['round_ms']} ms peak {o['peak_gib']:.2f} GiB (prefill {o['prefill_peak_gib']:.2f}, decode "
          f"{o['decode_peak_gib']:.2f}, active at the first token {o['active_first_gib']:.2f}) "
          f"active {o['active_gib']:.2f} GiB cache {o['cache_gib']:.2f} GiB "
          f"entries {o['entries']} end {o['end']}" + ("" if same is None else f" same={same}"), flush=True)
    for line in o["log"]:
        print("      ", line, flush=True)


res = []
msgs = [("system", DOC), ("user", TASKS[0])]
NQ = W._VERIFY_ATT_Q
for t in range(a.turns):
    W._VERIFY_ATT_Q = 0 if a.alt == "rows" and t % 2 else NQ
    eng.draft = None if a.alt == "plain" and t % 2 else DRAFT
    o = run(msgs, f"turn{t + 1}")
    o["verify_att_q"], o["drafter"] = W._VERIFY_ATT_Q, eng.draft is not None
    W._VERIFY_ATT_Q, eng.draft = NQ, DRAFT
    show(o)
    res.append(o)
    if o["end"][0] != "done":
        break
    last = list(msgs)
    msgs = msgs + [("assistant", answer(o["tokens"])), ("user", FOLLOW[t % len(FOLLOW)])]
if a.cold_ref and res[-1]["end"][0] == "done":
    o = run(last, "cold", "off")
    show(o, same=o["tokens"] == res[-1]["tokens"])
    o["same"] = o["tokens"] == res[-1]["tokens"]
    res.append(o)
if a.ab and res[-1]["end"][0] == "done":
    nq, va0, vd0 = W._VERIFY_ATT_Q, W._VATT, W._VATT_DEC
    names = a.ab_variants.split(",")
    ref = None
    for k in range(a.ab):
        order = names[k % len(names):] + names[:k % len(names)]      # rotate: no variant always first
        for name in order:
            base, _, dec = name.partition("+")      # "+d": decode steps through vatt (QWEN_DECODE_ATT)
            assert not dec or vd0, "'+d' needs QWEN_DECODE_ATT (self-test passed)"
            eng.draft = None if base == "plain" else DRAFT
            W._VERIFY_ATT_Q = nq if base in ("vector", "vatt") else 0
            W._VATT = va0 and base == "vatt"
            W._VATT_DEC = dec == "d"
            o = run(last, f"{name}{k + 1}", a.ab_mode)
            o["variant"], o["round"] = name, k
            if base == "plain" and not dec and ref is None:
                ref = o["tokens"]
            o["same"] = o["tokens"] == res[-1]["tokens"]
            o["same_as_plain"] = ref is not None and o["tokens"] == ref
            show(o, same=o["same"])
            res.append(o)
    eng.draft, W._VERIFY_ATT_Q, W._VATT, W._VATT_DEC = DRAFT, nq, va0, vd0
    import statistics
    for name in names:
        rows = [o for o in res if o.get("variant") == name]
        print(f"ab {name:7s} tps median {statistics.median(o['tps'] for o in rows):6.2f} "
              f"(all {[o['tps'] for o in rows]}) round_ms {[o['round_ms'] for o in rows]} "
              f"same as the first plain {sum(o['same_as_plain'] for o in rows)}/{len(rows)}", flush=True)
print(f"process peak {max(o['peak_gib'] for o in res):.2f} GiB", flush=True)
if a.json:
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    for o in res:
        o.pop("tokens", None)
    json.dump(dict(args=vars(a), env={k: v for k, v in os.environ.items() if k.startswith("QWEN_")},
                   runs=res), open(a.json, "w"), indent=1)
sys.stdout.flush()
os._exit(0)
