"""Where does a DFlash round go? Stage timings of the worker's own `_spec`
rounds (real Engine + _Scheduler, no gRPC, no repo edits: the stages are
wrapped from outside), plus plain decode at the same context for reference.

Rounds cycle through modes (--modes, default normal,stage,normal,deep):

  normal  nothing synced; host timestamps only:
            draft_build   draft_block() building the drafter's graph
            draft_submit  its mx.async_eval (walk + Metal encoding; the GPU starts
                          on the first committed command buffer)
            verify_build  the verify forward's graph (lm(..., speculative_verify))
            verify_submit mx.async_eval(targets, hidden)
            walk          _speculative_walk: .tolist() of drafts + targets = the
                          host waiting for the round's GPU work
            commit, own   commit_speculative_round / _own_states (host part)
            tail          the rest until the next round's draft_block(): emit,
                          seen, the yield back to the tick loop, _regrow,
                          admission checks, _dflash_next_block_size
          period = one round start to the next. GPU idle between rounds ("host
          bubble") is from the end of walk (the GPU has nothing left) to the next
          round's draft_submit: bubble_lo = up to its start, bubble_hi = up to its
          end (encoding overlaps the GPU, so the truth is in between).
  stage   mx.synchronize() before and an eval after each stage: draft (GPU +
          build), verify (whole forward incl. pick), walk, commit, own_states;
          the stages' sum vs the normal period shows the sync overhead.
  deep    as stage, and inside the verify forward each linear call
          (_linear / _linears), each block attention (kernels._block_attention)
          and each GDN with-states update (gated_delta_update_with_states) is
          isolated: its array inputs are evaluated first (that time goes to
          v_other), then it runs and its outputs are evaluated (-> v_lin / v_attn
          / v_gdn). v_other = norms, conv, gates, rope, KV append, splits, the
          sync overhead (one sync ~ --sync-cost, measured at start).

Requests: the first one prefills the prompt (a --tokens system document of the
repo's docs + a coding task, thinking off) and stores it; the next ones read
the prompt cache ("read": resume at the stored end, nothing stored), so every
request decodes at the same context. --plain N: N more requests with the
drafter off (eng.draft = None), alternated with the DFlash ones: the plain
step's tokens/s at the same context (decode tok/s = (n - 1) / (t_last - t_first)).

  cd src/worker && QWEN_MODEL=../../weights/qwen38-27b-mlx-4bit \\
    QWEN_DRAFT=../../weights/qwen38-27b-dflash2 QWEN_TEXT_ONLY=1 QWEN_MRQMM4=1 QWEN_KV_BITS=0 \\
    QWEN_MAX_BATCH=1 QWEN_WIRED_LIMIT_GB=27 QWEN_PROMPT_CACHE=1 QWEN_PRESENCE_PENALTY=0 \\
    PYTHONPATH=. .venv/bin/python ../../scripts/perf/dflash2/spec_prof.py --tokens 2000 \\
    --reqs 3 --plain 2 --max-tokens 512 --json $PERF_OUT/spec_prof_2k.json
NEEDS THE GPU LEASE.
"""
import argparse
import collections
import glob
import json
import os
import statistics
import sys
import threading
import time

sys.path.insert(0, os.getcwd())
import mlx.core as mx  # noqa: E402
import worker as W  # noqa: E402
import worker_pb2 as pb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "prof"))
from wtest import TASKS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, default=2000, help="system document tokens (0 = none)")
ap.add_argument("--reqs", type=int, default=3, help="DFlash requests (the first prefills)")
ap.add_argument("--plain", type=int, default=2, help="plain-decode requests, alternated")
ap.add_argument("--max-tokens", type=int, default=512)
ap.add_argument("--task", type=int, default=0)
ap.add_argument("--modes", default="normal,stage,normal,deep")
ap.add_argument("--skip", type=int, default=4, help="rounds of each request not counted (warm-up)")
ap.add_argument("--json", default="")
a = ap.parse_args()
assert os.environ.get("QWEN_PROMPT_CACHE") == "1", "QWEN_PROMPT_CACHE=1 (requests after the first read it)"

from mlx_vlm.models.qwen3_5 import gated_delta as gd, speculative_verifier as sv  # noqa: E402
from mlx_vlm.speculative import cache_state, common  # noqa: E402
import qwen_worker.kernels as K  # noqa: E402
import qwen_worker.sched_spec as SS  # noqa: E402

eng = W.Engine(os.environ["QWEN_MODEL"])
eng.load()
DRAFT = eng.draft
assert DRAFT is not None, "QWEN_DRAFT"
tok = getattr(eng.processor, "tokenizer", eng.processor)
LM = type(eng.model.language_model)
MODES = a.modes.split(",")
SCHED = "scheduler"

# ------------------------------------------------------------------ the sync cost
_sync = mx.synchronize
z = mx.zeros((8,))
mx.eval(z)
ts = []
for _ in range(50):
    y = z + 1
    t = time.perf_counter()
    mx.eval(y)
    ts.append(time.perf_counter() - t)
SYNC_MS = statistics.median(ts) * 1e3
print(f"one tiny eval + sync: {SYNC_MS:.3f} ms", flush=True)

# ------------------------------------------------------------------ instruments
S = {"rec": None, "n": 0, "on": False, "deep": False, "mark": 0.0, "depth": 0, "in_verify": False}
ROUNDS = []                     # finished round records
now = time.perf_counter


def on():
    return S["on"] and threading.current_thread().name == SCHED


def _arrs(x, out):
    if isinstance(x, mx.array):
        out.append(x)
    elif isinstance(x, (list, tuple)):
        for y in x:
            _arrs(y, out)
    elif isinstance(x, dict):
        for y in x.values():
            _arrs(y, out)
    elif hasattr(x, "logits"):                  # a LanguageModelOutput
        _arrs([x.logits, x.hidden_states], out)
    return out


def add(key, dt):
    r = S["rec"]
    if r is not None:
        r[key] = r.get(key, 0.0) + dt * 1e3


_draft_block = type(DRAFT).draft_block


def draft_block(self, *args, **kw):
    if not on():
        return _draft_block(self, *args, **kw)
    t = now()
    prev = S["rec"]
    if prev is not None:
        prev["next_start"] = t
        ROUNDS.append(prev)
    mode = MODES[S["n"] % len(MODES)]
    S["n"] += 1
    rec = {"mode": mode, "start": t, "req": S.get("req")}
    S["rec"] = rec
    if mode != "normal":
        _sync()
        rec["pre_sync"] = (now() - t) * 1e3
        t = now()
    out = _draft_block(self, *args, **kw)
    t1 = now()
    rec["draft_build"] = (t1 - t) * 1e3
    if mode != "normal":
        mx.eval(out)
        rec["draft"] = (now() - t) * 1e3
    rec["phase"] = "draft_submit"
    return out


_async_eval = mx.async_eval


def async_eval(*arrs):
    r = S["rec"]
    if r is None or not on() or "phase" not in r:
        return _async_eval(*arrs)
    ph = r.pop("phase")
    t = now()
    _async_eval(*arrs)
    t1 = now()
    r[ph] = r.get(ph, 0.0) + (t1 - t) * 1e3
    r[ph + "_t0"], r[ph + "_t1"] = t, t1
    return None


_lm_call = LM.__call__


def lm_call(self, *args, **kw):
    r = S["rec"]
    if r is None or not on() or not kw.get("speculative_verify"):
        return _lm_call(self, *args, **kw)
    mode = r["mode"]
    t = now()
    if mode != "normal":
        _sync()
        r["pre_sync"] = r.get("pre_sync", 0.0) + (now() - t) * 1e3
        t = now()
    S["mark"], S["deep"], S["in_verify"] = t, mode == "deep", True
    try:
        out = _lm_call(self, *args, **kw)
    finally:
        S["in_verify"] = False
    t1 = now()
    r["verify_build"] = (t1 - t) * 1e3
    if mode != "normal":
        mx.eval(out.logits, out.hidden_states)
        t2 = now()
        r["verify"] = (t2 - t) * 1e3
        if mode == "deep":
            r["v_other"] = r.get("v_other", 0.0) + (t2 - S["mark"]) * 1e3
    r["phase"] = "verify_submit"
    return out


def deep(cls, fn, self_arg):
    def w(*args, **kw):
        r = S["rec"]
        if r is None or not S["in_verify"] or not S["deep"] or S["depth"] or not on():
            return fn(*args, **kw)
        ins = _arrs(list(args[1:] if self_arg else args), [])
        mx.eval(ins)
        t = now()
        r["v_other"] = r.get("v_other", 0.0) + (t - S["mark"]) * 1e3
        S["depth"] += 1
        try:
            out = fn(*args, **kw)
        finally:
            S["depth"] -= 1
        mx.eval(_arrs(out, []))
        t1 = now()
        r[cls] = r.get(cls, 0.0) + (t1 - t) * 1e3
        r["n_" + cls] = r.get("n_" + cls, 0) + 1
        S["mark"] = t1
        return out
    return w


V = sv.Qwen3_5BatchInvariantForward
V._linear = deep("v_lin", V._linear, True)
V._linears = deep("v_lin", V._linears, True)
K._block_attention = deep("v_attn", K._block_attention, False)
gd.gated_delta_update_with_states = deep("v_gdn", gd.gated_delta_update_with_states, False)


def timed(key, fn, sync_after=None):
    def w(*args, **kw):
        r = S["rec"]
        if r is None or not on():
            return fn(*args, **kw)
        t = now()
        out = fn(*args, **kw)
        if r["mode"] != "normal" and sync_after is not None:
            mx.eval(sync_after(args, out))
        t1 = now()
        r[key] = r.get(key, 0.0) + (t1 - t) * 1e3
        r[key + "_t1"] = t1
        return out
    return w


common._speculative_walk = timed("walk", common._speculative_walk)
cache_state.commit_speculative_round = timed(
    "commit", cache_state.commit_speculative_round,
    lambda args, out: [x for c in args[1] if hasattr(c, "cache") for x in c.cache if x is not None]
    + [c.state for c in args[1] if not hasattr(c, "cache")])


def own_states(caches):
    r = S["rec"]
    if r is None or not on():
        return _own_states(caches)
    t = now()
    _own_states(caches)                 # its own async_eval is not a stage: phase unset
    if r["mode"] != "normal":
        mx.eval([x for c in caches if hasattr(c, "cache") for x in c.cache if x is not None])
    t1 = now()
    r["own"] = r.get("own", 0.0) + (t1 - t) * 1e3
    r["own_t1"] = t1
    return None


_own_states = SS._own_states
SS._own_states = own_states
mx.async_eval = async_eval
LM.__call__ = lm_call
type(DRAFT).draft_block = draft_block

# ------------------------------------------------------------------ requests
text = ""
for f in ["README.md", "CLAUDE.md", "BACKLOG.md"] + sorted(glob.glob(os.path.join(ROOT, "docs/perf/2026-10-1*/*.md"))):
    text += open(os.path.join(ROOT, f)).read() + "\n\n"
    if len(text) > 6 * max(a.tokens, 1):
        break
ids = tok.encode(text)
DOC = ("You are a coding assistant. Project notes follow; use them only if relevant.\n\n"
       + tok.decode(ids[:a.tokens])) if a.tokens else "You are a coding assistant."
MSGS = [("system", DOC), ("user", TASKS[a.task % len(TASKS)])]
pc = eng._pcache


def run(rid, mode):
    req = pb.Request(id=rid, messages=[pb.Message(role=r, content=c) for r, c in MSGS],
                     params=pb.Params(temperature=0.0, presence_penalty=0.0, max_tokens=a.max_tokens,
                                      prompt_cache=mode, chat_template_kwargs_json=json.dumps(
                                          {"enable_thinking": False})))
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
    DRAFT.accept_lens, DRAFT.draft_lens = [], []
    S["req"], S["rec"], S["n"] = rid, None, 0
    t0 = now()
    eng.sched.submit(r)
    toks, times = [], []
    while True:
        ev = r.out.get()
        if ev[0] != "tok":
            break
        toks.append(ev[1])
        times.append(now())
    while eng.sched.task is not None or eng.sched.rows:
        time.sleep(0.02)
    S["rec"] = None                              # the last round has no successor: dropped
    dt = times[-1] - times[0] if len(toks) > 2 else 0.0
    acc = [int(x) for x in DRAFT.accept_lens]
    return dict(id=rid, prompt=len(pids), reused=r.reused, n=len(toks), ttft=times[0] - t0 if times else None,
                tps=(len(toks) - 1) / dt if dt else 0.0, rounds=len(acc),
                accepted=sum(acc) / len(acc) if acc else None, toks=toks)


print(f"loaded; drafter {type(DRAFT).__name__}, verify att q {W._VERIFY_ATT_Q}, block {W.cfg.DRAFT_BLOCK}; "
      f"active {mx.get_active_memory() / 2**30:.2f} GiB", flush=True)
S["on"] = True
reqs = []
plan = ["spec"] + [x for i in range(max(a.reqs - 1, a.plain)) for x in
                   (["plain"] if i < a.plain else []) + (["spec"] if i < a.reqs - 1 else [])]
for i, kind in enumerate(plan):
    eng.draft = DRAFT if kind == "spec" else None
    n0 = len(ROUNDS)
    o = run(f"{kind}{i}", "" if i == 0 else "read")
    eng.draft = DRAFT
    o["kind"] = kind
    for rr in ROUNDS[n0:n0 + a.skip]:
        rr["skip"] = True
    print(f"{o['id']:7s} prompt {o['prompt']} reused {o['reused']} ttft {o['ttft']:.2f}s n {o['n']} "
          f"tok/s {o['tps']:.2f} rounds {o['rounds']} accepted/round {o['accepted']}", flush=True)
    reqs.append(o)
same = [o["toks"] == reqs[0]["toks"] for o in reqs]
print(f"tokens equal to the first request: {same}", flush=True)

# ------------------------------------------------------------------ report
KEYS = ["period", "draft_build", "draft_submit", "verify_build", "verify_submit", "walk", "commit", "own",
        "tail", "bubble_lo", "gpu_busy", "pre_sync", "draft", "verify", "v_lin", "v_attn",
        "v_gdn", "v_other", "n_v_lin", "n_v_attn", "n_v_gdn"]
out = {"sync_ms": SYNC_MS, "requests": [{k: v for k, v in o.items() if k != "toks"} for o in reqs],
       "same_tokens": same, "modes": {}}
by = collections.defaultdict(list)
for r in ROUNDS:
    if r.get("skip") or "next_start" not in r or "walk_t1" not in r:
        continue
    r["period"] = (r["next_start"] - r["start"]) * 1e3
    last = max(r.get("own_t1", 0.0), r.get("commit_t1", 0.0), r["walk_t1"])
    r["tail"] = (r["next_start"] - last) * 1e3
    if "draft_submit_t0" in r:
        r["gpu_busy"] = (r["walk_t1"] - r["draft_submit_t0"]) * 1e3
    by[r["mode"]].append(r)
# between rounds (GPU idle: the round's work ended with walk): walk end of a
# normal round -> the next round's draft_block() entry, + that round's graph
# build when it is a normal one (its draft_submit then starts the GPU)
seq = [r for r in ROUNDS if "walk_t1" in r]
for r0, r1 in zip(seq, seq[1:]):
    if r0["mode"] == "normal" and r0.get("req") == r1.get("req") and not r0.get("skip"):
        r0["bubble_lo"] = (r1["start"] - r0["walk_t1"]) * 1e3 + statistics.median(
            x["draft_build"] for x in seq if x["mode"] == "normal")
for mode, rs in by.items():
    med = {k: statistics.median([r[k] for r in rs if k in r]) for k in KEYS if any(k in r for r in rs)}
    med["rounds"] = len(rs)
    out["modes"][mode] = med
    print(f"\n[{mode}] {len(rs)} rounds (medians, ms):", flush=True)
    print("  " + "  ".join(f"{k}={v:.2f}" for k, v in med.items() if k != "rounds"), flush=True)
if "normal" in out["modes"] and "stage" in out["modes"]:
    st = out["modes"]["stage"]
    parts = {k: st.get(k, 0.0) for k in ("draft", "verify", "walk", "commit", "own", "tail")}
    print(f"\nstage sum {sum(parts.values()):.1f} ms vs normal period {out['modes']['normal']['period']:.1f} "
          f"ms: " + " ".join(f"{k} {v:.1f}" for k, v in parts.items()), flush=True)
if "normal" in out["modes"] and "stage" in out["modes"]:
    st, nm = out["modes"]["stage"], out["modes"]["normal"]
    gpu = st["draft"] - st["draft_build"] + st["verify"] - st["verify_build"]
    out["stage_gpu_ms"] = gpu
    print(f"GPU work of a round (stage: draft + verify, synced, minus their graph builds) {gpu:.1f} ms; "
          f"the normal period exposes {nm['period'] - gpu:.1f} ms more (host, encode throttling, the "
          f"commit/own/tail and between-round gap {nm.get('bubble_lo', float('nan')):.1f} ms)", flush=True)
plain = [o for o in reqs if o["kind"] == "plain"]
if plain:
    out["plain_tps"] = statistics.median(o["tps"] for o in plain)
    out["plain_step_ms"] = 1e3 / out["plain_tps"]
    print(f"plain decode: {out['plain_tps']:.2f} tok/s = {out['plain_step_ms']:.1f} ms/token", flush=True)
spec = [o for o in reqs if o["kind"] == "spec"]
out["spec_tps"] = statistics.median(o["tps"] for o in spec)
out["rounds_raw"] = [{k: v for k, v in r.items() if not k.endswith(("_t0", "_t1")) and k not in ("start", "next_start")}
                     for r in ROUNDS]
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
os._exit(0)
