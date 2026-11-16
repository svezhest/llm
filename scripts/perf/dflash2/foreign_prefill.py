"""A long conversation's cache entry and a foreign request beside it (single stream,
QWEN_PROMPT_CACHE_MOVE): does the foreign prefill fit next to the entry, and does
the entry survive it?

Real Engine + _Scheduler, greedy, the worker's env as given. Steps:
  1. --writer N > 0: a conversation whose system document is N tokens, one turn
     (cold prefill; its entry = prompt + reply stays in the cache);
  2. --foreign M tokens of an unrelated document (another prompt, common prefix
     ~3), prompt_cache --foreign-mode ("read" = nothing stored), --repeat times;
  3. --writer > 0: the conversation's next turn -- must hit its own entry;
  4. --ab N: N rounds of that next turn again (prompt_cache "": with MOVE the entry
     moves into the row and comes back), plain decode and DFlash, --max-ab-tokens
     tokens (--ab-user: another last message), greedy: tok/s, DFlash rounds / accepted / ms per round, peaks; with
     --ab-lib and quantized KV also with the library's quantized SDPA (QWEN_VATT_Q8
     off for that run) -- interleaved in this process, tokens compared with the
     first plain run. --probe: a short coding prompt (plain + DFlash) first, the
     machine's speed at this moment (for runs compared across processes).
Per request: prompt, reused, TTFT, the prefill peak (mx.get_peak_memory, reset at
admission) and the active memory before it, the cache's entries. The prefill
transient = prefill peak - active before admission; what the GPU must hold
beside it is the buffer pool (<= QWEN_CACHE_LIMIT_GB).

  cd src/worker && <the production env> PYTHONPATH=. .venv/bin/python \\
    ../../scripts/perf/dflash2/foreign_prefill.py --writer 78000 --foreign 14300 --json $PERF_OUT/fp.json
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prof"))
from doc_notes import NOTES  # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("--writer", type=int, default=0)
ap.add_argument("--foreign", type=int, default=14300)
ap.add_argument("--foreign-mode", default="read")
ap.add_argument("--repeat", type=int, default=1)
ap.add_argument("--max-tokens", type=int, default=64)
ap.add_argument("--ab", type=int, default=0)
ap.add_argument("--ab-lib", action="store_true")
ap.add_argument("--max-ab-tokens", type=int, default=256)
ap.add_argument("--ab-ignore-eos", action="store_true",
                help="A/B runs generate --max-ab-tokens (EOS skipped; note: token options turn DFlash off)")
ap.add_argument("--ab-user", default="", help="the A/B turn's last user message instead of the next turn's "
                "(e.g. one asking for a long answer, so that runs reach --max-ab-tokens with DFlash on)")
ap.add_argument("--probe", action="store_true")
ap.add_argument("--json", default="")
a = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
import mlx.core as mx  # noqa: E402

eng = W.Engine(os.environ["QWEN_MODEL"])
eng.load()
tok = getattr(eng.processor, "tokenizer", eng.processor)
pc = eng._pcache
print(f"loaded: active {mx.get_active_memory() / 2**30:.2f} GiB, budget {eng.sched.budget / 2**30:.2f} GiB, "
      f"prefill reserve {getattr(eng.sched, 'prefill_reserve', 0) / 2**30:.2f} GiB", flush=True)


def doc(files, n, head):
    text = ""
    for f in files:
        text += open(f, encoding="utf-8").read() + "\n\n"
        if len(text) > 6 * n:
            break
    ids = tok.encode(text)
    assert len(ids) >= n, f"only {len(ids)} tokens"
    return head + tok.decode(ids[:n])


WDOC = doc([os.path.join(ROOT, f) for f in list(NOTES)]
           + sorted(glob.glob(os.path.join(ROOT, "docs/perf/2026-10-1*/*.md"))), max(a.writer, 1),
           "You are a writing assistant. The draft so far:\n\n")
FDOC = doc(sorted(glob.glob(os.path.join(ROOT, "src/worker/qwen_worker/*.py"))), a.foreign,
           "Rate the following text for clarity on a scale of 1-10, one line:\n\n")


DRAFT = eng.draft


def run(messages, rid, mode="", max_tokens=None, quiet=False, ignore_eos=False):
    if DRAFT is not None:
        DRAFT.accept_lens, DRAFT.draft_lens = [], []
    req = pb.Request(id=rid, messages=[pb.Message(role=r, content=c) for r, c in messages],
                     params=pb.Params(temperature=0.0, presence_penalty=0.0,
                                      max_tokens=max_tokens or a.max_tokens,
                                      prompt_cache=mode, ignore_eos=ignore_eos,
                                      chat_template_kwargs_json=json.dumps({"enable_thinking": False})))
    formatted, _ = eng._render(req, [])
    pids = tok.encode(formatted)
    p = req.params
    sampling = dict(W.SAMPLING_KWARGS)
    for f in ("temperature", "presence_penalty"):
        sampling[f] = getattr(p, f)
    ctl = W._Control(p, tok.stopping_criteria.eos_token_ids, tok, eng.vocab)
    r = W._Req(id=rid, p=p, ids=pids, max_tokens=eng._limit(p, len(pids)), sampling=sampling, ctl=ctl,
               formatted=formatted)
    act0 = mx.get_active_memory()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    eng.sched.submit(r)
    toks, first, pre_peak, end, last = [], None, 0, None, None
    while True:
        ev = r.out.get()
        if ev[0] != "tok":
            end = ev
            break
        if first is None:
            first = time.perf_counter()
            pre_peak = mx.get_peak_memory()
        last = time.perf_counter()
        toks.append(ev[1])
    time.sleep(0.2)
    while eng.sched.task is not None or eng.sched.rows:
        time.sleep(0.05)
    o = dict(id=rid, mode=mode, prompt=len(pids), reused=r.reused, n=len(toks), end=[str(x) for x in end[:2]],
             ttft=round(first - t0, 2) if first else None, active_before_gib=round(act0 / 2**30, 2),
             prefill_peak_gib=round(pre_peak / 2**30, 2),
             transient_gib=round((pre_peak - act0) / 2**30, 2) if pre_peak else None,
             peak_gib=round(mx.get_peak_memory() / 2**30, 2), active_after_gib=round(mx.get_active_memory() / 2**30, 2),
             cache_gib=round(eng._pcache.bytes / 2**30, 2), entries=[len(e.ids) for e in eng._pcache.entries],
             text=tok.decode(toks)[:120])
    acc = [int(x) for x in DRAFT.accept_lens] if DRAFT is not None and eng.draft is not None else []
    dt = last - first if first and len(toks) > 2 else 0.0
    o.update(tps=round((len(toks) - 1) / dt, 2) if dt else None, rounds=len(acc),
             accepted=round(sum(acc) / len(acc), 2) if acc else None,
             round_ms=round(1e3 * dt / len(acc), 1) if acc and dt else None)
    o["tokens"] = toks
    print(json.dumps({k: v for k, v in o.items() if k != "tokens"}, ensure_ascii=False), flush=True)
    return o


res = []
if a.probe:      # the machine's speed now: a short prompt, plain and DFlash
    for d in ("plain", "dflash"):
        eng.draft = DRAFT if d == "dflash" else None
        o = run([("user", "Write a Python function that merges two sorted lists, with a docstring.")],
                f"probe-{d}", "read", 128)
        o["variant"] = f"probe-{d}"
        res.append(o)
    eng.draft = DRAFT
wmsgs = [("system", WDOC), ("user", "Continue the draft with one more paragraph.")]
if a.writer:
    o = run(wmsgs, "writer1")
    res.append(o)
    wmsgs = wmsgs + [("assistant", o["text"]), ("user", "Good. One more, shorter.")]
for i in range(a.repeat):
    res.append(run([("user", FDOC)], f"foreign{i + 1}", a.foreign_mode))
if a.writer and res[-1]["end"][0] == "done":
    res.append(run(wmsgs, "writer2"))
if a.ab and a.writer and res[-1]["end"][0] == "done":
    import statistics
    q8 = (W._VATT_Q8, W._VATT_Q8_DEC)
    names = ["plain", "dflash"] + (["plain-lib", "dflash-lib"] if a.ab_lib and any(q8) else [])
    ref = None
    amsgs = wmsgs[:-1] + [("user", a.ab_user)] if a.ab_user else wmsgs
    for k in range(a.ab):
        order = names[k % len(names):] + names[:k % len(names)]
        for name in order:
            eng.draft = DRAFT if name.startswith("dflash") else None
            lib = name.endswith("-lib")
            W._VATT_Q8, W._VATT_Q8_DEC = (False, False) if lib else q8
            o = run(amsgs, f"{name}{k + 1}", "", a.max_ab_tokens, ignore_eos=a.ab_ignore_eos)
            o["variant"] = name
            if name == "plain" and ref is None:
                ref = o["tokens"]
            o["same_as_plain"] = ref is not None and o["tokens"] == ref
            res.append(o)
    eng.draft, (W._VATT_Q8, W._VATT_Q8_DEC) = DRAFT, q8
    for name in names:
        rows = [o for o in res if o.get("variant") == name]
        rm = [o["round_ms"] for o in rows if o["round_ms"]]
        print(f"ab {name:10s} tps median {statistics.median(o['tps'] for o in rows):6.2f} "
              f"(all {[o['tps'] for o in rows]}) round_ms {rm} accepted {[o['accepted'] for o in rows]} "
              f"same as the first plain {sum(o['same_as_plain'] for o in rows)}/{len(rows)}", flush=True)
print(f"process peak {max(o['peak_gib'] for o in res):.2f} GiB", flush=True)
for o in res:
    o.pop("tokens", None)
env = {k: v for k, v in os.environ.items() if k.startswith("QWEN_")}
if a.json:
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    json.dump(dict(env=env, args=vars(a), results=res), open(a.json, "w"), indent=1)
