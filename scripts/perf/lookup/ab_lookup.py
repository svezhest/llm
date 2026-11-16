"""In-process A/B of prompt lookup decoding (QWEN_LOOKUP) on the 27B + DFlash2:
the real Engine and _Scheduler (no gRPC), one process, variants interleaved per
(round, workload) in a rotating order, medians over rounds (CLAUDE.md
"Measuring performance"). A sibling of scripts/perf/dflash2/ab27.py.

Variants (comma separated; options after ':' as k=v joined by '+'):
  plain        no drafter, no lookup: the plain greedy loop
  dflash       the drafter alone (QWEN_LOOKUP=0)
  hyb          QWEN_LOOKUP=1 with the drafter: lookup round when matched, else its block
  splice       QWEN_LOOKUP=splice with the drafter
  lplain       QWEN_LOOKUP=1 without the drafter: lookup rounds + plain steps
  options: min=<QWEN_LOOKUP_MIN> k=<QWEN_LOOKUP_K> n=<QWEN_LOOKUP_N> nmax=<..._NMAX>
           burst=<QWEN_LOOKUP_BURST> k0=<QWEN_LOOKUP_K0> backoff=<0|1>, e.g. hyb:min=5+k=7

Workloads (prompts in prompts/, fixed): rewrite_a / rewrite_b ("rewrite this
file with one change", ~1.6K / ~2.6K-token sources), reemit (turn 2 re-emits
turn 1's code with an addition; turn 1 generated here by plain greedy), code
(a short coding task), code2k (unit tests for a ~3K-token module), en2k / ru2k
(explain a ~2K-token English / Russian text). Thinking off.

Per request: decode tok/s = (n - 1) / (t_last - t_first), the request's
_spec stats (rounds per source, drafts accepted), tokens (identity vs plain of
the same round and workload).

  cd src/worker && QWEN_MODEL=../../weights/qwen38-27b-mlx-4bit QWEN_DRAFT=../../weights/qwen38-27b-dflash2 \\
    QWEN_TEXT_ONLY=1 QWEN_MRQMM4=1 QWEN_KV_BITS=0 QWEN_MAX_BATCH=1 QWEN_WIRED_LIMIT_GB=27 \\
    QWEN_PROMPT_CACHE=1 PYTHONPATH=. .venv/bin/python ../../scripts/perf/lookup/ab_lookup.py \\
    --rounds 3 --json $PERF_OUT/lookup/ab.json
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
PROMPTS = os.path.join(HERE, "prompts")
sys.path.insert(0, os.path.join(HERE, "..", "prof"))
from wtest import TASKS  # noqa: E402  (no MLX there)

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--variants", default="plain,dflash,hyb,splice,lplain")
ap.add_argument("--workloads", default="rewrite_a,rewrite_b,reemit,code,code2k,en2k,ru2k")
ap.add_argument("--max-tokens", type=int, default=384)
ap.add_argument("--json", default="")
ap.add_argument("--dump-ids", default="", help="write prompt ids + plain outputs here (for sim.py)")
ap.add_argument("--jsonl", default="", help="append every request (ids, tokens, stats) here as it ends")
ap.add_argument("--oracle-ks", default="", help="after the A/B: rounds of K perfect drafts (the plain answer) "
                "on --oracle-workload, no drafter -> ms per verify round of K + 1 rows, e.g. 2,3,5,7,11,15")
ap.add_argument("--oracle-workload", default="code")
ap.add_argument("--oracle-rounds", type=int, default=3)
a = ap.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

import mlx.core as mx  # noqa: E402

t_load = time.time()
eng = W.Engine(os.environ["QWEN_MODEL"])
eng.load()
DRAFT = eng.draft
tok = getattr(eng.processor, "tokenizer", eng.processor)
print(f"loaded in {time.time() - t_load:.0f} s; active {mx.get_active_memory() / 1e9:.2f} GB; "
      f"max_batch {eng.max_batch}; drafter {type(DRAFT).__name__ if DRAFT else None}; "
      f"DRAFT_BLOCK {W.DRAFT_BLOCK}; VERIFY_MR {W.VERIFY_MR}", flush=True)
BASE = dict(LOOKUP_MIN=W.LOOKUP_MIN, LOOKUP_K=W.LOOKUP_K, LOOKUP_N=W.LOOKUP_N, LOOKUP_NMAX=W.LOOKUP_NMAX,
            LOOKUP_BURST=W.LOOKUP_BURST, LOOKUP_K0=W.LOOKUP_K0, LOOKUP_BACKOFF=W.LOOKUP_BACKOFF)
OPT = dict(min="LOOKUP_MIN", k="LOOKUP_K", n="LOOKUP_N", nmax="LOOKUP_NMAX", burst="LOOKUP_BURST",
           k0="LOOKUP_K0", backoff="LOOKUP_BACKOFF")


def set_variant(v):
    """Flip the globals the scheduler reads (idle scheduler: between requests)."""
    name, _, opts = v.partition(":")
    for k, x in BASE.items():
        setattr(W, k, x)
    for kv in filter(None, opts.split("+")):
        k, x = kv.split("=")
        setattr(W, OPT[k], int(x))
    eng.draft = DRAFT if name in ("dflash", "hyb", "splice") else None
    W.LOOKUP = {"plain": "0", "dflash": "0", "hyb": "1", "splice": "splice", "lplain": "1"}[name]
    if name in ("dflash", "hyb", "splice"):
        assert DRAFT is not None, "needs QWEN_DRAFT"


def read(name):
    return open(os.path.join(PROMPTS, name), encoding="utf-8").read()


def fence(src):
    return f"```python\n{src}```"


def messages(w, turn1=None):
    if w == "rewrite_a":
        return [("user", "Rewrite this file with one change: rename the function `_step` to `_slab_step` "
                         "everywhere (its definition and every call; the docstrings too). Output the complete "
                         "updated file in one ```python block and nothing else.\n\n" + fence(read("rewrite_a.py.txt")))]
    if w == "rewrite_b":
        return [("user", "Rewrite this file with one change: add a module-level constant `MAX_STOP_LEN = 64` "
                         "(with a one-line comment) right after the imports, and keep everything else exactly "
                         "as it is. Output the complete updated file in one ```python block and nothing else."
                         "\n\n" + fence(read("rewrite_b.py.txt")))]
    if w == "reemit":
        return [("user", REEMIT_Q1), ("assistant", turn1),
                ("user", "Add a `__len__` method and a `clear()` method to the class, and print the complete "
                         "updated code again.")]
    if w == "code":
        return [("user", TASKS[0])]
    if w == "code2k":
        return [("user", "Write pytest unit tests for the public behaviour of this module (the sampler and "
                         "the penalties).\n\n" + fence(read("tests_src.py.txt")))]
    if w == "en2k":
        return [("user", read("en_context.md") + "\n\n---\nExplain in your own words, in three or four "
                                                 "paragraphs, what this project is and how its parts fit "
                                                 "together.")]
    if w == "ru2k":
        return [("user", "Ниже — журнал изменений проекта.\n\n" + read("ru_context.txt")
                 + "\n\n---\nРасскажи своими словами, в трёх-четырёх абзацах, как развивался проект и "
                   "что в нём главное.")]
    raise ValueError(w)


REEMIT_Q1 = ("Write a Python class `LRUCache` with `get(key)` and `put(key, value)` in O(1) using "
             "collections.OrderedDict, with a capacity and short docstrings. Only the code.")


def request(msgs, rid, max_tokens):
    req = pb.Request(id=rid, messages=[pb.Message(role=r, content=c) for r, c in msgs],
                     params=pb.Params(temperature=0.0, presence_penalty=0.0, max_tokens=max_tokens,
                                      chat_template_kwargs_json=json.dumps({"enable_thinking": False})))
    formatted, _ = eng._render(req, [])
    ids = tok.encode(formatted)
    p = req.params
    sampling = dict(W.SAMPLING_KWARGS)
    for f in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "seed"):
        if p.HasField(f):
            sampling[f] = getattr(p, f)
    ctl = W._Control(p, tok.stopping_criteria.eos_token_ids, tok, eng.vocab)
    return W._Req(id=rid, p=p, ids=ids, max_tokens=eng._limit(p, len(ids)), sampling=sampling, ctl=ctl,
                  formatted=formatted)


def run(msgs, rid, max_tokens):
    if DRAFT is not None:
        DRAFT.accept_lens, DRAFT.draft_lens = [], []
    r = request(msgs, rid, max_tokens)
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
    return dict(tokens=toks, ids=r.ids, n=n, prompt=len(r.ids), reused=r.reused,
                ttft=times[0] - t0 if times else None, tps=(n - 1) / dt if dt else 0.0,
                finish=[str(x) for x in ev[1:2]], stats=r.spec_stats)


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


VS = a.variants.split(",")
WS = a.workloads.split(",")
res = dict(variants=VS, workloads=WS, rounds=a.rounds, max_tokens=a.max_tokens, model=os.environ["QWEN_MODEL"],
           draft=os.environ.get("QWEN_DRAFT"), base=BASE, runs=[])

set_variant("plain")
turn1 = None
if "reemit" in WS:
    t1 = run([("user", REEMIT_Q1)], "turn1", 512)
    turn1 = tok.decode(t1["tokens"])
    res["turn1"] = turn1
    print(f"turn 1: {t1['n']} tokens ({t1['finish']})", flush=True)
MSGS = {w: messages(w, turn1) for w in WS}
for v in VS:                                                   # warm every path once
    set_variant(v)
    o = run([("user", "Say OK.")], f"warm-{v}", 8)
    print(f"warm {v}: n={o['n']}", flush=True)
for rnd in range(a.rounds):
    for wi, w in enumerate(WS):
        order = VS[(rnd + wi) % len(VS):] + VS[:(rnd + wi) % len(VS)]   # rotate: no variant always first
        if (rnd + wi) % 2:
            order = order[::-1]
        for v in order:
            set_variant(v)
            o = run(MSGS[w], f"r{rnd}-{w}-{v}", a.max_tokens)
            o.update(variant=v, round=rnd, workload=w)
            res["runs"].append(o)
            if a.jsonl:
                with open(a.jsonl, "a") as f:
                    f.write(json.dumps({k: o[k] for k in ("variant", "round", "workload", "n", "prompt", "tps",
                                                          "ttft", "stats", "tokens")}
                                       | ({"ids": o["ids"]} if v == "plain" else {})) + "\n")
            s = o["stats"] or {}
            print(f"r{rnd} {w:9s} {v:16s} n={o['n']:4d} prompt={o['prompt']:5d} reused={o['reused']:5d} "
                  f"tps={o['tps']:6.2f} ttft={o['ttft'] or 0:6.2f} "
                  f"dflash={s.get('dflash_rounds', 0)}r {s.get('dflash_accepted', 0)}/{s.get('dflash_drafted', 0)} "
                  f"lookup={s.get('lookup_rounds', 0)}r {s.get('lookup_accepted', 0)}/{s.get('lookup_drafted', 0)} "
                  f"plain={s.get('plain_steps', 0)}", flush=True)

plain = {(o["round"], o["workload"]): o["tokens"] for o in res["runs"] if o["variant"] == "plain"}
print("\n== median decode tok/s per workload (over rounds); xN = vs dflash / vs plain; identity vs plain ==")
summ = {}
for w in WS:
    base = {v: med([o["tps"] for o in res["runs"] if o["workload"] == w and o["variant"] == v]) for v in VS}
    for v in VS:
        rows = [o for o in res["runs"] if o["workload"] == w and o["variant"] == v]
        same = sum(o["tokens"] == plain.get((o["round"], w)) for o in rows) if plain else None
        div = [next((i for i, (x, y) in enumerate(zip(o["tokens"], plain[(o["round"], w)])) if x != y), None)
               for o in rows if plain and o["tokens"] != plain.get((o["round"], w))]
        s = dict(tps=base[v], tps_rounds=[o["tps"] for o in rows], n=med([o["n"] for o in rows]),
                 same=f"{same}/{len(rows)}" if plain else None, diverge=div,
                 vs_dflash=base[v] / base["dflash"] if base.get("dflash") else None,
                 vs_plain=base[v] / base["plain"] if base.get("plain") else None,
                 stats=rows[-1]["stats"] if rows else None)
        summ[f"{w}/{v}"] = s
        print(f"{w:9s} {v:16s} tps {s['tps']:6.2f} [{' '.join(f'{x:.1f}' for x in s['tps_rounds'])}] "
              f"x{s['vs_dflash'] or 0:.3f} dflash x{s['vs_plain'] or 0:.3f} plain  n={s['n']}  "
              f"== plain {s['same']} {div if div else ''}", flush=True)
res["summary"] = summ

if a.oracle_ks:
    class Oracle:
        """_Lookup stand-in: the plain answer's next k tokens, always (a match of 99)."""
        answer = None

        def __init__(self, ids, *args):
            self.n0, self.toks = len(ids), list(ids)

        def add(self, t):
            self.toks.append(int(t))

        def propose(self, k):
            at = len(self.toks) - self.n0
            return 99, list(self.answer[at:at + k])

    real = W._Lookup
    w = a.oracle_workload
    ref = run(MSGS[w], "oracle-ref", a.max_tokens)
    set_variant("plain")
    Oracle.answer = ref["tokens"]
    print(f"\n== oracle rounds on {w}: K perfect drafts per verify round (no drafter) ==", flush=True)
    res["oracle"] = []
    ks = [int(x) for x in a.oracle_ks.split(",")]
    for rnd in range(a.oracle_rounds):
        for K in (ks if rnd % 2 == 0 else ks[::-1]):
            if K == 0:
                set_variant("plain")
            else:
                set_variant(f"lplain:k={K}+min=1")
                W._Lookup = Oracle
            try:
                o = run(MSGS[w], f"oracle-r{rnd}-k{K}", a.max_tokens)
            finally:
                W._Lookup = real
            s = o["stats"] or {}
            nr = s.get("lookup_rounds", 0) + s.get("plain_steps", 0) if K else o["n"] - 1
            dt = (o["n"] - 1) / o["tps"] if o["tps"] else 0
            ms = 1000 * dt / max(1, nr)
            res["oracle"].append(dict(round=rnd, k=K, tps=o["tps"], n=o["n"], rounds=nr, ms_per_round=ms,
                                      same=o["tokens"] == ref["tokens"], stats=s))
            print(f"r{rnd} K={K:2d} tps={o['tps']:6.2f} rounds={nr} ms/round={ms:6.1f} "
                  f"same={o['tokens'] == ref['tokens']}", flush=True)
    for K in ks:
        rows = [x for x in res["oracle"] if x["k"] == K]
        print(f"K={K:2d} (verify rows {K + 1:2d}): median {med([x['ms_per_round'] for x in rows]):6.1f} ms/round, "
              f"{med([x['tps'] for x in rows]):6.2f} tok/s", flush=True)
res["peak_gb"] = mx.get_peak_memory() / 1e9
print(f"peak {res['peak_gb']:.2f} GB", flush=True)
if a.dump_ids:
    dump = [dict(workload=o["workload"], round=o["round"], ids=o["ids"], out=o["tokens"])
            for o in res["runs"] if o["variant"] == "plain"]
    json.dump(dict(dump=dump), open(a.dump_ids, "w"))
if a.json:
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    for o in res["runs"]:
        o.pop("ids", None)
        o["tokens_n"] = len(o.pop("tokens"))
    json.dump(res, open(a.json, "w"), indent=1)
    print("wrote", a.json)
sys.stdout.flush()
os._exit(0)          # the scheduler thread never returns (worker.py: MLX stream teardown)
