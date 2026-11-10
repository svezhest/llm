"""DFlash2 after a prompt-cache hit: how much acceptance does the drafter lose
when its context (the target's hidden states) covers only the re-computed tail
of the prompt? Real Engine + _Scheduler, QWEN_PROMPT_CACHE=1, thinking on.

Per task, with a ~3K-token system document:
  cold1   turn 1, prompt_cache "off" (no lookup: full prefill, full drafter context; stored)
  repeat  turn 1 again (hit at its deepest checkpoint: the drafter sees the tail only)
  hit2    turn 2 = turn 1 + its answer + a follow-up (--followup-chars of pasted text),
          normal: resumes at turn 1's entry, the drafter sees the new tail
  cold2   turn 2 again, prompt_cache "off"
The tokens of a pair are the same greedy answer (checked), so the accepted
drafts per round differ only through the drafter's context.

  cd src/worker && QWEN_MODEL=../../weights/qwen38-27b-mlx-5bit QWEN_KV_BITS=0 QWEN_MRQMM5=1 \\
    QWEN_PROMPT_CACHE=1 QWEN_DRAFT=../../weights/qwen38-27b-dflash2 PYTHONPATH=. .venv/bin/python \\
    ../../scripts/perf/dflash2/cache_hit.py --json $PERF_OUT/dflash2/cache_hit.json
"""
import argparse
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
from wtest import TASKS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", type=int, default=3)
ap.add_argument("--max-tokens", type=int, default=192)
ap.add_argument("--doc-chars", type=int, default=12000)
ap.add_argument("--followup-chars", type=int, default=1500, help="pasted text in the turn-2 message")
ap.add_argument("--no-thinking", action="store_true")
ap.add_argument("--json", default="")
a = ap.parse_args()
assert os.environ.get("QWEN_DRAFT") and os.environ.get("QWEN_PROMPT_CACHE") == "1"


class Grab(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, rec):
        m = rec.getMessage()
        if "prompt cache" in m:
            self.lines.append(m[:200])


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
grab = Grab()
logging.getLogger("qwen.worker").addHandler(grab)

eng = W.Engine(os.environ["QWEN_MODEL"])
eng.load()
DRAFT = eng.draft
assert DRAFT is not None
tok = getattr(eng.processor, "tokenizer", eng.processor)
think_close = tok.convert_tokens_to_ids("</think>")
DOC = ("You are a coding assistant. Project notes follow; use them only if relevant.\n\n"
       + open(os.path.join(ROOT, "README.md")).read()[:a.doc_chars])
PASTE = open(os.path.join(ROOT, "BACKLOG.md")).read()[:a.followup_chars]
print(f"loaded; block {W.DRAFT_BLOCK or 'adaptive'}, VERIFY_MR {W.VERIFY_MR}", flush=True)


def run(messages, rid, mode=""):
    grab.lines = []
    DRAFT.accept_lens, DRAFT.draft_lens = [], []
    req = pb.Request(id=rid, messages=[pb.Message(role=r, content=c) for r, c in messages],
                     params=pb.Params(temperature=0.0, presence_penalty=0.0, max_tokens=a.max_tokens,
                                      prompt_cache=mode, chat_template_kwargs_json=json.dumps(
                                          {"enable_thinking": not a.no_thinking})))
    formatted, _ = eng._render(req, [])
    ids = tok.encode(formatted)
    p = req.params
    sampling = dict(W.SAMPLING_KWARGS)
    for f in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "seed"):
        if p.HasField(f):
            sampling[f] = getattr(p, f)
    ctl = W._Control(p, tok.stopping_criteria.eos_token_ids, tok, eng.vocab)
    r = W._Req(id=rid, p=p, ids=ids, max_tokens=eng._limit(p, len(ids)), sampling=sampling, ctl=ctl,
               formatted=formatted)
    t0 = time.perf_counter()
    eng.sched.submit(r)
    toks, times = [], []
    while True:
        ev = r.out.get()
        if ev[0] != "tok":
            break
        toks.append(ev[1])
        times.append(time.perf_counter())
    acc = [int(x) for x in DRAFT.accept_lens]
    dt = times[-1] - times[0] if len(toks) > 2 else 0.0
    mean = lambda xs: round(sum(xs) / len(xs), 2) if xs else None  # noqa: E731
    return dict(id=rid, prompt=len(ids), n=len(toks), tokens=toks, ttft=round(times[0] - t0, 2),
                tps=round((len(toks) - 1) / dt, 2) if dt else 0.0, rounds=len(acc),
                acc_first4=mean(acc[:4]), acc_first8=mean(acc[:8]), acc_first16=mean(acc[:16]),
                acc_all=mean(acc), first16=acc[:16], cache=list(grab.lines))


def answer(toks):
    """The reply's content (after </think>) as the client would send it back."""
    cut = toks.index(think_close) + 1 if think_close in toks else len(toks)
    return tok.decode(toks[cut:]).strip() or "(no answer yet)"


res = []
run([("user", "Say OK.")], "warm", "off")
for ti in range(a.tasks):
    t1 = [("system", DOC), ("user", TASKS[ti])]
    cold1 = run(t1, f"t{ti}-cold1", "off")
    repeat = run(t1, f"t{ti}-repeat")
    t2 = t1 + [("assistant", answer(cold1["tokens"])),
               ("user", "Now add type hints and a short docstring, and handle invalid input explicitly. "
                        "For reference, the tool printed:\n" + PASTE)]
    hit2 = run(t2, f"t{ti}-hit2")
    cold2 = run(t2, f"t{ti}-cold2", "off")
    for name, o, ref in (("cold1", cold1, None), ("repeat", repeat, cold1), ("hit2", hit2, cold2),
                         ("cold2", cold2, None)):
        o.update(task=ti, case=name, same=None if ref is None else o["tokens"] == ref["tokens"])
        res.append(o)
        print(f"t{ti} {name:6s} prompt {o['prompt']:5d} ttft {o['ttft']:6.2f}s tps {o['tps']:5.2f} "
              f"rounds {o['rounds']:3d} accepted/round first4 {o['acc_first4']} first8 {o['acc_first8']} "
              f"first16 {o['acc_first16']} all {o['acc_all']} same={o['same']} first16={o['first16']}",
              flush=True)
        for line in o["cache"]:
            print("      ", line, flush=True)

import mlx.core as mx  # noqa: E402

print(f"peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)
if a.json:
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    for o in res:
        o.pop("tokens", None)
    json.dump(dict(args=vars(a), runs=res, peak_gb=mx.get_peak_memory() / 1e9), open(a.json, "w"), indent=1)
sys.stdout.flush()
os._exit(0)
