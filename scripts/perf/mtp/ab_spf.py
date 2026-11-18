"""In-process A/B of a speculative row's prefill (sched_spec.PREFILL_NO_SEED,
sched_spec.PREFILL_DEFER): the time to the first token of a cold prompt and of a
prompt-cache hit, plain against the drafters (the MTP head alone, DFlash alone,
both as QWEN_DRAFT=auto runs them), each drafter variant with the switches on
(the default) and off. The real Engine and _Scheduler (no gRPC), one process,
variants interleaved per (round, cell), medians over rounds (docs/measuring.md).
Sibling of ab_mtp.py (its variant setup); load with QWEN_DRAFT=auto
QWEN_DRAFT_DFLASH=<dir> QWEN_DRAFT_MTP=<dir>, BENCH27 = scripts/perf/bench27.

Variants: plain | head (the gated head, mtp0+q) | dflash | auto, with flags
  +old    both switches off (the prefill as before)
  +eager  PREFILL_DEFER off (PREFILL_NO_SEED on): the last slab's drafter work before the first token
  +seed   PREFILL_NO_SEED off (PREFILL_DEFER on)
  +det    a fixed schedule for the identity pass: the head 2 drafts a round (MTP_K=2; DFlash's rounds
          have no confidence gate here); "both" = both drafters attached, DFlash's rounds
          (DRAFT_HYBRID=dflash), "both-mtp" = the head's rounds
Per cell (a bench27 prompt id) and round, each variant:
  cold  the round's prompt (bench27 prompts.messages with session tag 1 + round: "Session <id>/<tag>."
        starts the system text; every tag gives the same token count, checked) and prompt_cache "off":
        no lookup, so a cold prefill whatever the cache holds; the entries are stored as a cold turn's
        are (the store is in plain's time and the drafters', and it absorbs the variant before's);
  hit   the next turn (the prompt + assistant "OK." + user "Continue."), prompt_cache "read": it
        resumes from the entries the cold request just stored (at the prompt's last message start,
        ~25 tokens prefilled; the drafters' stored context, QWEN_DRAFT_TAIL_END positions, caught up
        -- after the first token with PREFILL_DEFER) and stores nothing, so the next variant's hit
        resumes from its own cold run's entries too (the hit's own store is not in its time: the
        suffix's KV copy, about equal for all; the drafter-context copies are still taken).
  Every variant of a round runs the same prompts: tokens comparable with its +old (informative: the
  gates read round times).
Timings: ttft = submit -> the first token; total = submit -> the last of --max-tokens tokens (the
deferred work is moved behind the first token, not removed: total shows it).
--ident: before the speed rounds, the deterministic variants (+det) with the switches off and on on
one tag per cell, the prompt cache emptied before each (the hit in prompt_cache "read"):
the cold and hit tokens and the stored entries' arrays (sha256 of the drafter's context, the K / V,
the states) must be identical between the two states.
"""
import argparse
import hashlib
import json
import logging
import os
import statistics
import sys
import time

sys.path.insert(0, os.getcwd())
import worker as W  # noqa: E402
import worker_pb2 as pb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--cells", default="code-2k-0=plain,head+old,head,dflash+old,dflash,auto+old,auto",
                help="';'-separated cell=variant,variant,...")
ap.add_argument("--ident", default="", help="';'-separated cell=variant,... (+det variants), checked off vs on")
ap.add_argument("--max-tokens", type=int, default=64)
ap.add_argument("--json", default="")
a = ap.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

t_load = time.time()
eng = W.Engine(os.environ["QWEN_MODEL"])
eng.load()
D, M = eng.draft, eng.mtp
if D is None or M is None:
    raise SystemExit("load both drafters (QWEN_DRAFT=auto + QWEN_DRAFT_DFLASH / QWEN_DRAFT_MTP)")
tok = getattr(eng.processor, "tokenizer", eng.processor)
QMV0 = W.VERIFY_QMV
print(f"loaded in {time.time() - t_load:.0f} s; active {mx.get_active_memory() / 1e9:.2f} GB; "
      f"VERIFY_QMV {QMV0}; PROMPT_CACHE {W.PROMPT_CACHE} REPLY {W.PROMPT_CACHE_REPLY}", flush=True)
if not W.PROMPT_CACHE:
    raise SystemExit("the hit needs the prompt cache (QWEN_PROMPT_CACHE)")


def set_variant(v):
    """Flip what the scheduler reads (between requests: the scheduler is idle)."""
    base, *fl = v.split("+")
    fl = set(fl)
    W.PREFILL_NO_SEED = not ({"old", "seed"} & fl)
    W.PREFILL_DEFER = not ({"old", "eager"} & fl)
    W.DRAFT_HYBRID, W.LOOKUP, W.SPEC_PLAIN, W.DRAFT_SAMPLING = "1", "0", False, True
    W.MTP_K = 2 if "det" in fl else 0
    W.VERIFY_QMV = QMV0 if base in ("head", "auto", "both", "both-mtp") else 0
    if base == "plain":
        eng.draft = eng.mtp = None
    elif base == "head":
        eng.draft, eng.mtp = None, M
    elif base == "dflash":
        eng.draft, eng.mtp = D, None
    elif base == "auto":
        eng.draft, eng.mtp = D, M
        W.LOOKUP, W.SPEC_PLAIN = "draft", True
    elif base in ("both", "both-mtp"):
        eng.draft, eng.mtp = D, M
        W.DRAFT_HYBRID = "dflash" if base == "both" else "mtp"
    else:
        raise SystemExit(f"unknown variant {v}")


sys.path.insert(0, os.environ["BENCH27"])
import prompts as P  # noqa: E402
MAN = P.load_manifest(os.path.join(os.environ["BENCH27"], "prompts", "manifest.json"))
PR = {p["id"]: p for p in MAN["prompts"]}


def msgs(cell, tag, turn):
    m = [pb.Message(role=x["role"], content=x["content"])
         for x in P.messages(MAN, PR[cell], f"{tag:02d}", check=tag == 0)]
    if turn:
        m += [pb.Message(role="assistant", content="OK."), pb.Message(role="user", content="Continue.")]
    return m


def request(cell, tag, turn, rid, cache_mode):
    req = pb.Request(id=rid, messages=msgs(cell, tag, turn),
                     params=pb.Params(presence_penalty=0.0, max_tokens=a.max_tokens, temperature=0.0,
                                      prompt_cache=cache_mode,
                                      chat_template_kwargs_json=json.dumps({"enable_thinking": False})))
    formatted, _ = eng._render(req, [])
    ids = tok.encode(formatted)
    p = req.params
    sampling = dict(W.SAMPLING_KWARGS)
    for f in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "seed"):
        if p.HasField(f):
            sampling[f] = getattr(p, f)
    ctl = W._Control(p, tok.stopping_criteria.eos_token_ids, tok, eng.vocab)
    return W._Req(id=rid, p=p, ids=ids, max_tokens=eng._limit(p, len(ids)), sampling=sampling,
                  ctl=ctl, formatted=formatted)


def idle():
    """Until the scheduler has nothing left (the finished row's stores done)."""
    s = eng.sched
    for _ in range(400):
        if not (s.rows or s.task or s.lane or s.waiting):
            break
        time.sleep(0.005)
    time.sleep(0.05)


def run(cell, tag, turn, rid, cache_mode):
    r = request(cell, tag, turn, rid, cache_mode)
    t0 = time.perf_counter()
    eng.sched.submit(r)
    toks, times = [], []
    while True:
        ev = r.out.get()
        if ev[0] != "tok":
            break
        toks.append(ev[1])
        times.append(time.perf_counter())
    idle()
    return dict(tokens=toks, n=len(toks), prompt=len(r.ids), reused=r.reused, ids=r.ids,
                ttft=times[0] - t0 if times else None, total=times[-1] - t0 if times else None)


def digest(x, h, cut=None):
    """Into sha256 `h`: an evaluated array's bytes through numpy (no MLX op on this thread: graphs
    are the scheduler thread's), `cut` = (lo, hi) on axis -2 (a KV piece)."""
    if x is None:
        return
    if isinstance(x, (list, tuple)):
        for y in x:
            digest(y, h, cut)
        return
    h.update(f"{x.dtype}{x.shape}{cut}".encode())
    try:
        y = np.asarray(x)
        if cut is not None:
            y = y[..., cut[0]:cut[1], :]
        h.update(np.ascontiguousarray(y).tobytes())
    except Exception:  # noqa: BLE001 -- a dtype numpy lacks: its shape only
        h.update(b"?")


def entries(ids, k=48):
    """sha256 per prompt-cache entry of this conversation (its first k ids are `ids`'): ids,
    checkpoints, every array (the KV pieces, the end state with the drafter's context, the
    checkpoint states); the scheduler is idle."""
    out = []
    for e in sorted(list(eng._pcache.entries), key=lambda e: (len(e.ids), e.ids)):
        if e.ids[:k] != ids[:k]:
            continue
        h = hashlib.sha256(repr((e.ids, sorted(e.ckpts), e.reply)).encode())
        for s, lo, hi in e.chain:
            digest(s.arrays, h, (lo - s.a, hi - s.a))
        digest(e.end, h)
        for p in sorted(e.ckpts):
            digest(e.ckpts[p], h)
        out.append(h.hexdigest()[:16])
    return out


def cells(spec):
    out = []
    for part in filter(None, spec.split(";")):
        c, vs = part.split("=")
        out.append((c, vs.split(",")))
    return out


CELLS, IDENT = cells(a.cells), cells(a.ident)
res = {"args": vars(a), "model": os.path.basename(os.path.normpath(os.environ["QWEN_MODEL"])),
       "speed": [], "ident": []}


def dump():
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        out = dict(res, speed=[{k: x for k, x in o.items() if k not in ("tokens", "hit_tokens")}
                               for o in res["speed"]],
                   peak_gb=mx.get_peak_memory() / 1e9)
        json.dump(out, open(a.json + ".tmp", "w"), indent=1)
        os.replace(a.json + ".tmp", a.json)


# every tag of a cell gives the same prompt length (the slabs are the same)
for c, vs in CELLS + IDENT:
    ns = {len(request(c, t, turn, "len", "off").ids) for t in (0, *range(1, a.rounds + 1), 95, 96, 97, 98, 99)
          for turn in (0, 1)}
    print(f"{c}: prompt lengths over tags (cold, hit) {sorted(ns)}", flush=True)
    if len(ns) != 2:
        print(f"WARNING {c}: the tags do not all give the same length", flush=True)

# warm-up: every base variant once on the first cell (kernels, the Clock's round times)
wc = CELLS[0][0]
for i, v in enumerate(["plain", "head", "dflash", "auto"]):
    set_variant(v)
    o = run(wc, 98 - i, 0, f"warm-{v}", "off")
    h = run(wc, 98 - i, 1, f"warmh-{v}", "read")
    print(f"warm {wc} {v}: cold ttft {o['ttft']:.3f} s, hit ttft {h['ttft']:.3f} s (reused {h['reused']})",
          flush=True)

def clear():
    """Empty the prompt cache (the scheduler is idle: nothing else touches it now)."""
    while eng._pcache.evict(spill=False):
        pass


# identity: the same tag, switches off then on (the stored entries compared after the cold request),
# the cache emptied before each: else the second store absorbs the first's entries and takes the
# chain policy (another checkpoint set: 2026-11-18, 8K, 4 vs 3 checkpoints; diff_spf.py)
for c, vs in IDENT:
    for v in vs:
        got = {}
        for state in ("old", "on"):
            clear()
            set_variant(v + ("+old" if state == "old" else ""))
            o = run(c, 99, 0, f"i-{c}-{v}-{state}", "off")
            ent = entries(o["ids"])
            hh = run(c, 99, 1, f"ih-{c}-{v}-{state}", "read")
            got[state] = (o["tokens"], ent, hh["tokens"], hh["reused"])
        same = [got["old"][k] == got["on"][k] for k in range(4)]
        row = dict(cell=c, variant=v, cold_tokens=same[0], entries=same[1], hit_tokens=same[2], resume=same[3],
                   n=len(got["on"][0]), hit_reused=got["on"][3])
        res["ident"].append(row)
        print(f"ident {c} {v}: cold tokens {'same' if same[0] else 'DIFFER'}, entries "
              f"{'same' if same[1] else 'DIFFER'} ({len(got['on'][1])}), hit tokens "
              f"{'same' if same[2] else 'DIFFER'} (reused {got['on'][3]})", flush=True)
        dump()

# speed: per round, per cell, the variants interleaved (rotated; reversed every other cell), all on
# the round's tag: the cold request ("off") stores the prompt's entries, absorbing the variant
# before's; the hit ("read") resumes from them and stores nothing
for rnd in range(a.rounds):
    for ki, (c, vs) in enumerate(CELLS):
        order = vs[rnd % len(vs):] + vs[:rnd % len(vs)]
        if (rnd + ki) % 2:
            order = order[::-1]
        t = 1 + rnd
        for v in order:
            set_variant(v)
            o = run(c, t, 0, f"s{rnd}-{c}-{v}", "off")
            h = run(c, t, 1, f"sh{rnd}-{c}-{v}", "read")
            row = dict(cell=c, variant=v, round=rnd, tag=t, prompt=o["prompt"], n=o["n"], tokens=o["tokens"],
                       ttft=o["ttft"], total=o["total"], hit_ttft=h["ttft"], hit_total=h["total"],
                       hit_reused=h["reused"], hit_prompt=h["prompt"], hit_tokens=h["tokens"])
            res["speed"].append(row)
            print(f"r{rnd} {c:10s} {v:12s} cold ttft {o['ttft']:7.3f} total {o['total']:7.3f} | hit ttft "
                  f"{h['ttft']:6.3f} total {h['total']:6.3f} (reused {h['reused']}/{h['prompt']})", flush=True)
            dump()


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def ratio(c, v, ref, key):
    """Median over rounds of v's / ref's time in the same round (pairs of one round)."""
    rows = {(o["round"], o["variant"]): o for o in res["speed"] if o["cell"] == c}
    return med([rows[(r, v)][key] / rows[(r, ref)][key] for r in range(a.rounds)
                if (r, v) in rows and (r, ref) in rows and rows[(r, ref)][key]])


print("\n== medians over rounds (s); x plain = the median of the per-round ratios; "
      "on/old = the variant against its +old ==")
summ = {}
for c, vs in CELLS:
    print(f"-- {c} ({next(o['prompt'] for o in res['speed'] if o['cell'] == c)} tokens)")
    for v in vs:
        rows = [o for o in res["speed"] if o["cell"] == c and o["variant"] == v]
        parts = v.split("+")
        old = "+".join([parts[0], *[x for x in parts[1:] if x not in ("old", "eager", "seed")], "old"])
        s = dict(ttft=med([o["ttft"] for o in rows]), total=med([o["total"] for o in rows]),
                 hit_ttft=med([o["hit_ttft"] for o in rows]), hit_total=med([o["hit_total"] for o in rows]),
                 x_plain=ratio(c, v, "plain", "ttft") if "plain" in vs else None,
                 hit_x_plain=ratio(c, v, "plain", "hit_ttft") if "plain" in vs else None,
                 vs_old=ratio(c, v, old, "ttft") if old in vs and v != old else None,
                 hit_vs_old=ratio(c, v, old, "hit_ttft") if old in vs and v != old else None,
                 total_vs_old=ratio(c, v, old, "total") if old in vs and v != old else None)
        # the same tokens as its +old in the same round (informative: the gates read round times)
        if old in vs and v != old:
            pair = {(o["round"]): o for o in res["speed"] if o["cell"] == c and o["variant"] == old}
            s["same_as_old"] = f"{sum(o['tokens'] == pair[o['round']]['tokens'] for o in rows if o['round'] in pair)}/{len(rows)}"
        summ[f"{c}/{v}"] = s

        def f(x, fmt="{:6.3f}"):
            return fmt.format(x) if x is not None else "     -"
        print(f"{v:12s} cold ttft {f(s['ttft'])} (x plain {f(s['x_plain'], '{:5.3f}')}, on/old "
              f"{f(s['vs_old'], '{:5.3f}')}) total {f(s['total'])} (on/old {f(s['total_vs_old'], '{:5.3f}')}) | "
              f"hit ttft {f(s['hit_ttft'])} (x plain {f(s['hit_x_plain'], '{:5.3f}')}, on/old "
              f"{f(s['hit_vs_old'], '{:5.3f}')})" + (f" | tokens == +old {s['same_as_old']}" if "same_as_old" in s else ""),
              flush=True)
res["summary"] = summ
if res["ident"]:
    bad = [o for o in res["ident"] if not (o["cold_tokens"] and o["entries"] and o["hit_tokens"] and o["resume"])]
    print(f"\n== identity (+det, switches off vs on): {len(res['ident']) - len(bad)}/{len(res['ident'])} "
          f"identical" + (f"; DIFFER: {[(o['cell'], o['variant']) for o in bad]}" if bad else ""), flush=True)
res["peak_gb"] = mx.get_peak_memory() / 1e9
print(f"peak {res['peak_gb']:.2f} GB", flush=True)
dump()
if a.json:
    print("wrote", a.json)
sys.stdout.flush()
os._exit(0)          # the scheduler thread never returns (worker.py: MLX stream teardown)
