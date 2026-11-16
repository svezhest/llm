"""In-process A/B of the speculative drafters on the 27B: plain vs DFlash vs the
target's MTP head vs both (hybrid), the real Engine and _Scheduler (no gRPC), one
process, variants interleaved per (round, prompt), medians over rounds
(CLAUDE.md "Measuring performance"). Sibling of scripts/perf/dflash2/ab27.py.

Load with both drafters: QWEN_DRAFT=auto (they are found next to the model dir), or
QWEN_DRAFT=<DFlash dir> QWEN_DRAFT_MTP=<MTP dir>.
Variants (comma separated):
  auto         what QWEN_DRAFT=auto runs: both drafters + plain steps as arms (mtp.Chooser),
               lookup rounds (QWEN_LOOKUP=draft), mrqmv verify, sampling requests speculative
  pplain       the chooser's plain arm alone (plain steps inside _spec, step-ahead bursts)
  plain        eng.draft = eng.mtp = None
  dflash       DFlash alone (eng.mtp = None)
  mtpK         the MTP head alone, K drafts per round (QWEN_MTP_K); mtpKr = QWEN_MTP_REFEED=1
  hybK         both, per-round choice (QWEN_DRAFT_HYBRID=1), MTP rounds of K drafts
  <v>+p        any of the above with plain steps as an arm (QWEN_SPEC_PLAIN=1)
  <v>+k<N>      the adaptive draft count's ceiling N (QWEN_MTP_KMAX)
  <v>+q4        +q on the 4-bit modules only (the 5 / 6 / 8-bit ones on their old path)
  <v>+x        verify blocks through mlx-vlm's exact multi-row qmv (QWEN_VERIFY_MRQMM off)
  <v>+q        verify blocks of 2..N tokens through mrqmv (load with QWEN_VERIFY_QMV=N; without
               +q the variant runs with it off)
  K = 0: the adaptive draft count (QWEN_MTP_K=0)
Prompt sets: --set bench27 (the cross-framework prompts: code / en / ru, ~2K tokens of
context; BENCH27 = the dir with prompts.py + prompts/manifest.json of
scripts/perf/bench27), --set wtest (scripts/perf/prof/wtest.py TASKS, short prompts),
--set short (SHORT below: a ~300-word story asked in one line, ru / en / code; keys = --prompts).
--modes greedy,sampling: the cells are modes x prompts; sampling = temperature 0.6 / top_p 0.95 /
top_k 20 / seed per request (QWEN_DRAFT_SAMPLING for the speculative variants). Per request:
decode tok/s = (n - 1) / (t_last - t_first), rounds, accepted drafts per round per drafter,
the rounds and tokens per arm (spec_stats["arms"]). The gate's table: gate_table.py.

  cd src/worker && QWEN_MODEL=../../weights/qwen38-27b-mlx-4bit QWEN_TEXT_ONLY=1 QWEN_MRQMM4=1 \\
    QWEN_KV_BITS=0 QWEN_MAX_BATCH=1 QWEN_WIRED_LIMIT_GB=27 QWEN_PRESENCE_PENALTY=0 \\
    QWEN_DRAFT=../../weights/qwen38-27b-dflash2 QWEN_DRAFT_MTP=../../weights/qwen38-27b-mtp \\
    BENCH27=../../scripts/perf/bench27 PYTHONPATH=. .venv/bin/python ../../scripts/perf/mtp/ab_mtp.py \\
    --variants plain,dflash,mtp3,hyb3 --rounds 3 --json $PERF_OUT/mtp/ab.json
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
from wtest import TASKS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--variants", default="plain,dflash,mtp3,hyb3")
ap.add_argument("--set", default="bench27", choices=["bench27", "wtest", "short"])
ap.add_argument("--prompts", default="code-2k-0,en-2k-0,ru-2k-0", help="bench27 prompt ids")
ap.add_argument("--tasks", type=int, default=5, help="wtest: the first N TASKS")
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--sampling", action="store_true", help="= --modes sampling")
ap.add_argument("--modes", default="", help="greedy,sampling: the cells are modes x prompts (a sampling "
                                            "cell's key ends in @s); default greedy")
ap.add_argument("--speed-cells", default="", help="the speed rounds' cells (default: every cell)")
ap.add_argument("--ident", default="", help="variants checked against plain greedy (wtest TASKS)")
ap.add_argument("--ident-tasks", type=int, default=16)
ap.add_argument("--skip-speed", action="store_true")
ap.add_argument("--json", default="")
ap.add_argument("--trace", action="store_true", help="QWEN_SPEC_TRACE: per-round rows in the JSON")
ap.add_argument("--dump", action="store_true",
                help="per prompt one request of plain steps with both drafters' shadow drafts at every "
                     "position (QWEN_SPEC_SHADOW=all): the input of chooser_sim.py")
ap.add_argument("--warm-tokens", type=int, default=0, help="max_tokens of the warm-up requests (0 = --max-tokens)")
a = ap.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
import mlx.core as mx  # noqa: E402

t_load = time.time()
eng = W.Engine(os.environ["QWEN_MODEL"])
eng.load()
D, M = eng.draft, eng.mtp
tok = getattr(eng.processor, "tokenizer", eng.processor)
print(f"loaded in {time.time() - t_load:.0f} s; active {mx.get_active_memory() / 1e9:.2f} GB; "
      f"dflash {type(D).__name__ if D else None} mtp {type(M).__name__ if M else None}; "
      f"DRAFT_BLOCK {W.DRAFT_BLOCK} VERIFY_MR {W.VERIFY_MR}", flush=True)
REFEED0, MR0 = W.MTP_REFEED, (W.VERIFY_MR, W.VERIFY_MR_HEAD)
KMAX0 = W.MTP_KMAX
# +q4: mrqmv on the 4-bit modules only (the marks of the others hidden for that variant)
import mrqmv as _MQ  # noqa: E402
QMV_MARKS = [(m, m.__dict__.get("_qmv"), m.__dict__.get("_qmv_fused"))
             for _, m in eng.model.language_model.named_modules()
             if isinstance(m, dict) and (m.__dict__.get("_qmv") or m.__dict__.get("_qmv_fused"))]


def qmv_bits(only4):
    for m, q, f in QMV_MARKS:
        if q:
            m.__dict__["_qmv"] = not only4 or _MQ.bits_of(m["weight"], m["scales"]) == 4
        if f:
            ids, groups = f
            m.__dict__["_qmv_fused"] = (ids, tuple(g for g in groups if not only4
                                                   or _MQ.bits_of(g[1], g[2]) == 4))


QMV0 = W.VERIFY_QMV                     # after load: 0 unless mrqmv installed


def set_variant(v, sampling=False):
    """Flip what the scheduler reads (idle scheduler: between requests)."""
    W.MTP_REFEED, W.DRAFT_HYBRID, W.SPEC_SHADOW, W.LOOKUP = REFEED0, "1", "", "0"
    W.DRAFT_SAMPLING = sampling
    flags = set(v.split("+")[1:])
    v = v.split("+")[0]
    if v in ("auto", "pplain"):
        flags |= {"p", "q"} if QMV0 else {"p"}
    W.SPEC_PLAIN = "p" in flags
    W.SPEC_TRACE = a.trace or v == "dump"
    W.MTP_KMAX = next((int(f[1:]) for f in flags if f.startswith("k")), KMAX0)
    W.VERIFY_MR, W.VERIFY_MR_HEAD = (False, False) if "x" in flags else MR0
    W.VERIFY_QMV = QMV0 if ("q" in flags or "q4" in flags) else 0
    qmv_bits("q4" in flags)
    if ("q" in flags or "q4" in flags) and not QMV0:
        raise SystemExit(f"{v}+q: mrqmv not installed (QWEN_VERIFY_QMV unset or its self-test failed)")
    if v == "dump":
        eng.draft, eng.mtp = D, M
        W.MTP_K, W.SPEC_PLAIN, W.DRAFT_HYBRID, W.SPEC_SHADOW = 0, True, "plain", "all"
    elif v == "auto":
        eng.draft, eng.mtp = D, M
        W.MTP_K, W.LOOKUP, W.DRAFT_SAMPLING = 0, "draft", True
    elif v == "pplain":
        eng.draft, eng.mtp = D, M
        W.MTP_K, W.DRAFT_HYBRID, W.DRAFT_SAMPLING = 0, "plain", True
    elif v == "plain":
        eng.draft = eng.mtp = None
    elif v == "dflash":
        eng.draft, eng.mtp = D, None
    elif v.startswith("mtp"):
        eng.draft, eng.mtp = None, M
        W.MTP_K = int(v[3:].rstrip("r"))
        W.MTP_REFEED = v.endswith("r")
    elif v.startswith("hyb"):
        eng.draft, eng.mtp = D, M
        W.MTP_K = int(v[3:].rstrip("r"))
        W.MTP_REFEED = v.endswith("r")
    else:
        raise SystemExit(f"unknown variant {v}")


SHORT = {
    "ru": "Напиши короткий рассказ (около 300 слов) о старом часовщике, который однажды утром "
          "находит в своей мастерской часы, идущие назад.",
    "en": "Write a short story (about 300 words) about an old clockmaker who one morning finds a "
          "clock in his workshop that runs backwards.",
    "code": "Write a Python function that merges overlapping intervals in a list of (start, end) "
            "pairs, with a docstring and three doctests.",
}


def messages_of(key):
    if key.startswith("short-"):                # prompts2k.SHORT: short-en / short-ru
        sys.path.insert(0, os.path.join(HERE, "..", "dflash2"))
        from prompts2k import SHORT as S2
        return [pb.Message(role="user", content=dict(S2)[key[6:] + "_prose"])]
    if a.set == "short":
        return [pb.Message(role="user", content=SHORT[key])]
    if a.set == "wtest":
        return [pb.Message(role="user", content=TASKS[int(key)])]
    import prompts as P
    man = P.load_manifest(os.path.join(os.environ["BENCH27"], "prompts", "manifest.json"))
    pr = {p["id"]: p for p in man["prompts"]}[key]
    return [pb.Message(role=m["role"], content=m["content"]) for m in P.messages(man, pr)]


if a.set == "bench27":
    sys.path.insert(0, os.environ["BENCH27"])
KEYS = a.prompts.split(",") if a.set in ("bench27", "short") else [str(i) for i in range(a.tasks)]
MODES = (a.modes or ("sampling" if a.sampling else "greedy")).split(",")
KEYS = [k + ("@s" if m == "sampling" else "") for m in MODES for k in KEYS]     # the cells
SPEED = a.speed_cells.split(",") if a.speed_cells else KEYS


def sv(v, cell):
    set_variant(v, cell.endswith("@s"))
MSGS = {}


def request(cell, rid, seed=0, max_tokens=0):
    key, _, mode = cell.partition("@")
    if key not in MSGS:
        MSGS[key] = messages_of(key)
    kw = dict(temperature=0.0)
    if mode == "s":
        kw = dict(temperature=0.6, top_p=0.95, top_k=20, seed=seed)
    req = pb.Request(id=rid, messages=MSGS[key],
                     params=pb.Params(presence_penalty=0.0, max_tokens=max_tokens or a.max_tokens,
                                      chat_template_kwargs_json=json.dumps({"enable_thinking": False}),
                                      **kw))
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


def stats(dr):
    acc, dl = [int(x) for x in dr.accept_lens], [int(x) for x in dr.draft_lens]
    if not acc:
        return {}
    return dict(rounds=len(acc), drafted=sum(dl), accepted=sum(acc), accept=100.0 * sum(acc) / max(1, sum(dl)),
                per_round=sum(acc) / len(acc), hist=dict(sorted(collections.Counter(acc).items())))


def run(key, rid, v, seed=0, max_tokens=0):
    for dr in (D, M):
        if dr is not None:
            dr.accept_lens, dr.draft_lens = [], []
    r = request(key, rid, seed, max_tokens)
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
    if W.SPEC_TRACE:
        out["trace"] = (r.spec_stats or {}).get("trace")
        out["times"] = [round(t, 4) for t in times]
    if v == "dump":
        out["ids"] = [int(t) for t in r.ids]
    if v != "plain":
        sd = stats(D) if D is not None and eng.draft is not None else {}
        sm = stats(M) if M is not None and eng.mtp is not None else {}
        rounds = sd.get("rounds", 0) + sm.get("rounds", 0)
        st = r.spec_stats or {}
        out["plain_steps"] = st.get("plain_steps", 0)
        out.update({k: st[k] for k in ("arms", "shadows", "switches") if k in st})
        out.update(dflash=sd, mtp=sm, rounds=rounds, round_ms=1000.0 * dt / rounds if rounds else None,
                   tok_per_round=(n - 1) / rounds if rounds else None)
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _name(p):                           # (no local paths in the JSON)
    return os.path.basename(os.path.normpath(p)) if p and os.sep in p else p


res = {"args": dict(vars(a), json=_name(a.json)), "model": _name(os.environ["QWEN_MODEL"]),
       "draft": _name(os.environ.get("QWEN_DRAFT")), "draft_mtp": _name(os.environ.get("QWEN_DRAFT_MTP")),
       "kv_bits": os.environ.get("QWEN_KV_BITS"), "loaded": dict(dflash=D is not None, mtp=M is not None),
       "speed": [], "ident": [], "dump": []}
VS = a.variants.split(",")


def dump():
    """Write what has been measured so far (after every request: a stopped run keeps its rows)."""
    if not a.json:
        return
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    out = dict(res, speed=[{k: x for k, x in o.items() if k != "tokens"} for o in res["speed"]],
               clock={str(t): {"g": c.g, "ms": {str(k): c.ms(k) for k in c.c}}
                      for t, c in W.CLOCK.sub.items()},
               peak_gb=mx.get_peak_memory() / 1e9, active_gb=mx.get_active_memory() / 1e9)
    json.dump(out, open(a.json + ".tmp", "w"), indent=1)
    os.replace(a.json + ".tmp", a.json)


if not a.skip_speed or a.dump:
    sv("plain", KEYS[0])
    run(KEYS[0], "warm", "plain", max_tokens=a.warm_tokens)
if a.dump:
    for key in KEYS:
        sv("dump", key)
        o = run(key, f"dump-{key}", "dump")
        o.update(variant="dump", key=key)
        res["dump"].append(o)
        print(f"dump {key}: n={o['n']} rounds={len(o['trace'] or [])} tps={o['tps']:.2f}", flush=True)
        dump()
if not a.skip_speed:
    # every variant once per mode (kernels compiled; the chooser's round times
    # per class of requests: its probe rounds are not the gate's)
    for key in [next(k for k in SPEED if k.endswith("@s") == m) for m in sorted({k.endswith("@s") for k in SPEED})]:
        for v in VS:
            sv(v, key)
            o = run(key, f"warm-{v}", v, max_tokens=a.warm_tokens)
            print(f"warm {key} {v}: n={o['n']} tps={o['tps']:.2f} arms={o.get('arms', '')}", flush=True)
if not a.skip_speed:
    for rnd in range(a.rounds):
        for ki, key in enumerate(SPEED):
            order = VS[rnd % len(VS):] + VS[:rnd % len(VS)]
            if (rnd + ki) % 2:
                order = order[::-1]
            for v in order:
                sv(v, key)
                o = run(key, f"s{rnd}-{key}-{v}", v, seed=1000 * rnd + ki)
                o.update(variant=v, round=rnd, key=key)
                res["speed"].append(o)
                sd, sm = o.get("dflash") or {}, o.get("mtp") or {}
                print(f"r{rnd} {key:10s} {v:7s} n={o['n']:3d} tps={o['tps']:6.2f} "
                      f"tok/round={o.get('tok_per_round') or 0:4.2f} round_ms={o.get('round_ms') or 0:6.1f} "
                      f"dflash={sd.get('rounds', 0)}r/{sd.get('accept', 0):4.1f}% "
                      f"mtp={sm.get('rounds', 0)}r/{sm.get('accept', 0):4.1f}% {sm.get('hist', '')} "
                      f"plain={o.get('plain_steps', 0)} arms={o.get('arms', '')} "
                      f"shadows={o.get('shadows', '')} switches={o.get('switches', '')}",
                      flush=True)
                dump()
    plain = {(o["round"], o["key"]): o["tokens"] for o in res["speed"] if o["variant"] == "plain"}
    print("\n== median decode tok/s per prompt (median over rounds) ==")
    summ = {}
    for v in VS:
        rows = [o for o in res["speed"] if o["variant"] == v]
        per = {k: med([o["tps"] for o in rows if o["key"] == k]) for k in SPEED}
        same = sum(o["tokens"] == plain.get((o["round"], o["key"])) for o in rows)
        summ[v] = dict(per_prompt=per, tps=med(per.values()),
                       tok_per_round=med([o.get("tok_per_round") for o in rows]),
                       round_ms=med([o.get("round_ms") for o in rows]),
                       mtp_accept=med([(o.get("mtp") or {}).get("accept") for o in rows]),
                       dflash_accept=med([(o.get("dflash") or {}).get("accept") for o in rows]),
                       mtp_rounds_share=med([(o.get("mtp") or {}).get("rounds", 0) / o["rounds"]
                                             for o in rows if o.get("rounds")]),
                       same_as_plain=f"{same}/{len(rows)}")
        s = summ[v]
        print(f"{v:7s} " + " ".join(f"{k} {x:6.2f}" for k, x in per.items())
              + f" | tok/round {s['tok_per_round'] or 0:4.2f} round {s['round_ms'] or 0:6.1f} ms"
              + f" mtp acc {s['mtp_accept'] or 0:4.1f}% dfl acc {s['dflash_accept'] or 0:4.1f}%"
              + f" mtp share {s['mtp_rounds_share'] or 0:.2f} == plain {s['same_as_plain']}", flush=True)
    res["speed_summary"] = summ

if a.ident:
    a.set = "wtest"
    MSGS.clear()
    KEYS = [str(i) for i in range(a.ident_tasks)]
    print(f"\n== identity vs plain greedy, {a.ident_tasks} wtest tasks ==", flush=True)
    for ti in range(a.ident_tasks):
        set_variant("plain")
        ref = run(str(ti), f"i-t{ti}-plain", "plain")["tokens"]
        for v in a.ident.split(","):
            set_variant(v)
            got = run(str(ti), f"i-t{ti}-{v}", v)["tokens"]
            div = next((i for i, (x, y) in enumerate(zip(ref, got)) if x != y),
                       None if len(ref) == len(got) else min(len(ref), len(got)))
            res["ident"].append(dict(task=ti, variant=v, same=div is None, diverge=div, n=len(ref)))
            print(f"t{ti} {v}: {'same' if div is None else f'diverges at {div}'} (n={len(ref)})", flush=True)
            dump()
    for v in a.ident.split(","):
        rows = [o for o in res["ident"] if o["variant"] == v]
        print(f"{v:7s} identical {sum(o['same'] for o in rows)}/{len(rows)}; divergence at "
              f"{[o['diverge'] for o in rows if not o['same']]}", flush=True)

res["peak_gb"] = mx.get_peak_memory() / 1e9
res["active_gb"] = mx.get_active_memory() / 1e9
print(f"peak {res['peak_gb']:.2f} GB, active {res['active_gb']:.2f} GB", flush=True)
if a.json:
    dump()
    print("wrote", a.json)
sys.stdout.flush()
os._exit(0)          # the scheduler thread never returns (worker.py: MLX stream teardown)
