"""Which part of a stored prompt-cache entry differs between the spec-prefill switch states
(sched_spec.PREFILL_NO_SEED / PREFILL_DEFER), component by component. One process (the real
Engine and _Scheduler, both drafters loaded as for ab_spf.py).

For each variant (ab_spf.py's names: both+det, head+det, dflash+det, ...) and each state
(old = both off, eager = NO_SEED alone, seed = DEFER alone, on = both): the prompt cache is
emptied (the scheduler idle), the cold request (prompt_cache "off", --max-tokens) runs, and every
array of every entry it left (the prompt entry absorbed by its reply entry) is read through numpy
(no MLX op on this thread) and compared with the variant's "old" run: per component its name
(kv[piece][i] / end:gdn<j>[k] / end:ctx[0]=the drafter's hidden-state tail / ckpt<p>:...), shape, dtype, equal, max |diff|, the differing elements; plus the entry's ids
length, reply, checkpoint positions and piece layout, and the tokens.
--noclear: first the ab_spf.py --ident order for the first variant (old, then on, the cache NOT
emptied between: the second store absorbs the first's entries), to show what that order alone
changes in the entry (its checkpoint set: the absorbing store takes the chain policy).
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

ap = argparse.ArgumentParser()
ap.add_argument("--cell", default="code-8k-0")
ap.add_argument("--variants", default="both+det,head+det,dflash+det")
ap.add_argument("--states", default="old,eager,seed,on")
ap.add_argument("--max-tokens", type=int, default=32)
ap.add_argument("--noclear", action="store_true")
ap.add_argument("--json", default="")
a = ap.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

eng = W.Engine(os.environ["QWEN_MODEL"])
eng.load()
D, M = eng.draft, eng.mtp
tok = getattr(eng.processor, "tokenizer", eng.processor)
QMV0 = W.VERIFY_QMV
if D is None or M is None or not W.PROMPT_CACHE:
    raise SystemExit("load both drafters and the prompt cache (QWEN_PROMPT_CACHE=1)")
FLAGS = {"old": "+old", "eager": "+eager", "seed": "+seed", "on": ""}


def set_variant(v):            # as ab_spf.py
    base, *fl = v.split("+")
    fl = set(fl)
    W.PREFILL_NO_SEED = not ({"old", "seed"} & fl)
    W.PREFILL_DEFER = not ({"old", "eager"} & fl)
    W.DRAFT_HYBRID, W.LOOKUP, W.SPEC_PLAIN, W.DRAFT_SAMPLING = "1", "0", False, True
    W.MTP_K = 2 if "det" in fl else 0
    W.VERIFY_QMV = QMV0 if base in ("head", "auto", "both", "both-mtp") else 0
    eng.draft, eng.mtp = {"plain": (None, None), "head": (None, M), "dflash": (D, None),
                          "auto": (D, M), "both": (D, M), "both-mtp": (D, M)}[base]
    if base == "auto":
        W.LOOKUP, W.SPEC_PLAIN = "draft", True
    if base in ("both", "both-mtp"):
        W.DRAFT_HYBRID = "dflash" if base == "both" else "mtp"


sys.path.insert(0, os.environ["BENCH27"])
import prompts as P  # noqa: E402
MAN = P.load_manifest(os.path.join(os.environ["BENCH27"], "prompts", "manifest.json"))
PR = {p["id"]: p for p in MAN["prompts"]}


def request(rid):
    msgs = [pb.Message(role=x["role"], content=x["content"]) for x in P.messages(MAN, PR[a.cell], "99", check=False)]
    req = pb.Request(id=rid, messages=msgs, params=pb.Params(
        presence_penalty=0.0, max_tokens=a.max_tokens, temperature=0.0, prompt_cache="off",
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
    s = eng.sched
    for _ in range(1000):
        if not (s.rows or s.task or s.lane or s.waiting):
            break
        time.sleep(0.005)
    time.sleep(0.1)


def run(rid):
    r = request(rid)
    eng.sched.submit(r)
    toks = []
    while True:
        ev = r.out.get()
        if ev[0] != "tok":
            break
        toks.append(ev[1])
    idle()
    return toks


def clear():
    """Empty the prompt cache (the scheduler is idle: nothing else touches it now)."""
    pc = eng._pcache
    while pc.evict(spill=False):
        pass


def host(x):
    """An evaluated MLX array as numpy, bf16 as its float32 values; None: unreadable."""
    try:
        return np.array(x, copy=True)
    except Exception:  # noqa: BLE001
        pass
    try:
        u = np.frombuffer(memoryview(x).cast("B"), dtype=np.uint16).reshape(x.shape)
        return (u.astype(np.uint32) << 16).view(np.float32)
    except Exception:  # noqa: BLE001
        return None


def snapshot():
    """Every entry -> (meta, {component name: (shape, dtype, numpy)})."""
    out = []
    for e in sorted(list(eng._pcache.entries), key=lambda e: (len(e.ids), e.ids)):
        n_rec = sum(1 for lay in e.layout if lay[0] == "arr")
        comps = {}

        def state(prefix, st):
            for j, layer in enumerate(st or []):
                name = f"{prefix}:gdn{j}" if j < n_rec else f"{prefix}:ctx"
                for k, x in enumerate(layer):
                    if x is not None:
                        comps[f"{name}[{k}]"] = x
        segs = []
        for pi, (s, lo, hi) in enumerate(e.chain):
            segs.append((s.a, s.b, lo, hi))
            for i, x in enumerate(s.arrays):
                comps[f"kv[{pi}][{i}]"] = (x, lo - s.a, hi - s.a)
        state("end", e.end)
        for p in sorted(e.ckpts):
            state(f"ckpt{p}", e.ckpts[p])
        arrs = {}
        for name, x in comps.items():
            cut = None
            if isinstance(x, tuple):
                x, lo, hi = x
                cut = (lo, hi)
            y = host(x)
            if y is not None and cut is not None:
                y = y[..., cut[0]:cut[1], :]
            arrs[name] = (tuple(x.shape), str(x.dtype), y)
        meta = dict(ids=len(e.ids), reply=e.reply, ckpts=sorted(e.ckpts), pieces=segs)
        out.append((meta, arrs))
    return out


def compare(ref, got):
    """Per entry: meta equal?, per component: (shape, dtype, equal, max |diff|, differing)."""
    rows = []
    for ei, ((m0, a0), (m1, a1)) in enumerate(zip(ref, got)):
        rows.append(dict(entry=ei, meta_ref=m0, meta=m1, meta_equal=m0 == m1))
        for name in sorted(set(a0) | set(a1)):
            if name not in a0 or name not in a1:
                rows.append(dict(entry=ei, name=name, only="ref" if name in a0 else "this"))
                continue
            (s0, d0, x0), (s1, d1, x1) = a0[name], a1[name]
            r = dict(entry=ei, name=name, shape=s1, dtype=d1, same_shape=s0 == s1 and d0 == d1)
            if x0 is None or x1 is None:
                r["equal"] = None
            elif x0.shape != x1.shape:
                r["equal"] = False
            else:
                diff = x0 != x1
                if np.issubdtype(x0.dtype, np.floating):
                    diff &= ~(np.isnan(x0) & np.isnan(x1))
                r["equal"] = not diff.any()
                if not r["equal"]:
                    r["differ"] = int(diff.sum())
                    r["of"] = int(diff.size)
                    if np.issubdtype(x0.dtype, np.floating):
                        r["max_abs"] = float(np.nanmax(np.abs(x0.astype(np.float64) - x1.astype(np.float64))))
                    idx = np.argwhere(diff)
                    r["first_at"] = idx[0].tolist()
            rows.append(r)
    if len(ref) != len(got):
        rows.append(dict(entries_ref=len(ref), entries=len(got)))
    return rows


def report(title, rows, toks_same):
    bad = [r for r in rows if r.get("equal") is False or r.get("only") or r.get("meta_equal") is False
           or r.get("same_shape") is False or "entries_ref" in r]
    unread = [r["name"] for r in rows if "name" in r and r.get("equal") is None and not r.get("only")]
    print(f"\n== {title}: tokens {'same' if toks_same else 'DIFFER'}; "
          f"{sum(1 for r in rows if r.get('equal'))} components equal, {len(bad)} not"
          + (f", unreadable {unread}" if unread else ""), flush=True)
    for r in rows:
        if "meta_equal" in r:
            print(f"  entry {r['entry']}: meta {'same' if r['meta_equal'] else 'DIFFER'}: {r['meta']}"
                  + ("" if r["meta_equal"] else f"  (ref {r['meta_ref']})"), flush=True)
    for r in bad:
        if "name" in r:
            print(f"  {r['entry']} {r['name']}: " + (f"only in {r['only']}" if r.get("only") else
                  f"{r['shape']} {r['dtype']} differ {r.get('differ')}/{r.get('of')} max|d| {r.get('max_abs')} "
                  f"first at {r.get('first_at')}"), flush=True)


res = {"cell": a.cell, "runs": []}


def dump():
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


vs = a.variants.split(",")
if a.noclear:
    v = vs[0]
    clear()
    set_variant(v + FLAGS["old"])
    t0 = run(f"nc-{v}-old")
    s0 = snapshot()
    set_variant(v + FLAGS["on"])           # NOT cleared: absorbs the old run's entries (ab_spf.py --ident)
    t1 = run(f"nc-{v}-on")
    rows = compare(s0, snapshot())
    report(f"{v}: old then on, cache NOT emptied (ab_spf.py's --ident order)", rows, t0 == t1)
    res["runs"].append(dict(variant=v, state="on-noclear", tokens_same=t0 == t1, rows=rows))
    dump()
    s0 = None
for v in vs:
    ref = None
    for st in a.states.split(","):
        clear()
        set_variant(v + FLAGS[st])
        t = time.perf_counter()
        toks = run(f"d-{v}-{st}")
        dt = time.perf_counter() - t
        snap = snapshot()
        print(f"{v} {st}: {len(toks)} tokens in {dt:.1f} s; entries {[m for m, _ in snap]}", flush=True)
        if ref is None:
            ref = (toks, snap)
            continue
        rows = compare(ref[1], snap)
        report(f"{v}: {st} vs old (cache emptied before each)", rows, toks == ref[0])
        res["runs"].append(dict(variant=v, state=st, tokens_same=toks == ref[0], rows=rows))
        dump()
        snap = None
    ref = None
    clear()
print("done", flush=True)
dump()
sys.stdout.flush()
os._exit(0)
