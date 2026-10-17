"""What an extra slab boundary costs end to end (S2c, docs/perf/2026-10-17/slabs.md).

The worker's Engine + _Scheduler in-process (Engine.load as worker.py; prompt cache on), with
`--bg` background rows decoding (distinct AppWorld prompts of ~--bg-tokens, ignore_eos, greedy)
for the whole run. A base conversation X (a recorded AppWorld call's messages, ~--base tokens,
ending in a tool output) is prefilled once and stays in the prompt cache. Each probe =
X + an assistant reply (~--reply tokens) + a tool output (~--tool tokens), max_tokens 1, all
distinct (no probe resumes from another's suffix); it resumes at X's last message start r0
(forced to be a slab end), so it prefills the same ~reply + tool + generation-prompt tail.
The variant sets the slab ends inside that suffix (a wrapper of worker._bounds, applied to the
probe's prompt only; the background rows and X keep the worker's cuts):

  1      no cut                                one slab
  2tail  the final <|im_start|> (the worker's cut: the ~5-token generation-prompt tail)
  2msg   the tool message's <|im_start|> (reply | tool + tail)
  3      both                                  reply | tool | tail

A "/s" suffix runs the variant with a sync before each of the probe's slabs (the decode
step queued ahead then lands in `wait`, not in the slab). Per probe the scheduler thread
logs: slabs, sum of _advance (prefill time as sched_timer counts it), the waits, decode
steps between the first slab and the join (count, time), TTFT (first slab start -> join),
first-token time (-> the token is out). Variants rotate per round (drift).

  cd src/worker && QWEN_MODEL=/Users/user/Projects/llm/weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 \
    QWEN_PROMPT_CACHE=1 QWEN_PROMPT_CACHE_GB=6 QWEN_BATCH_RESERVE_TOKENS=1024 QWEN_PRESENCE_PENALTY=0 \
    QWEN_PROMPT_CACHE_REPLY=1 PYTHONPATH=. .venv/bin/python ../../scripts/perf/prof/slab_boundary.py \
    --rounds 8 --json $PERF_OUT/slabs_boundary.json
"""
import argparse, glob, json, os, random, statistics, sys, threading, time
sys.path.insert(0, os.getcwd())
if os.environ.get("SCHED_WORKER_DIR"):          # a frozen copy of src/worker (worker.py, pfqmm.py, ...)
    sys.path.insert(0, os.environ["SCHED_WORKER_DIR"])
import mlx.core as mx
import worker as W
import worker_pb2 as pb

ap = argparse.ArgumentParser()
ap.add_argument("--variants", default="1,2tail,2msg,3,1/s,2tail/s,2msg/s,3/s")
ap.add_argument("--rounds", type=int, default=8)
ap.add_argument("--bg", type=int, default=7)
ap.add_argument("--bg-tokens", type=int, default=10000)
ap.add_argument("--base", default="10240", help="X sizes (comma list: probes alternate)")
ap.add_argument("--reply", type=int, default=60)
ap.add_argument("--tool", type=int, default=740)
ap.add_argument("--tasks-dir", default=os.path.expanduser(
    "~/Projects/appworld/experiments/outputs/ornith9b_react_dev/tasks"))
ap.add_argument("--phases", action="store_true",
                help="inside the probe's slabs: sync + time the cache restore / store / _join and every mx.eval "
                     "(the slab body's eval = its GPU time); moves lazy work into the timed phase")
ap.add_argument("--json")
a = ap.parse_args()

eng = W.Engine(os.environ["QWEN_MODEL"])
eng.load()
tok = getattr(eng.processor, "tokenizer", eng.processor)
S = W._Scheduler
IM = eng._im_start

# ---------------------------------------------------------------- cut variants
CUR = {"probe": None, "v": None}           # probe request id -> variant (its _bounds)
_bounds = W._bounds
BIDS = {}                                  # id(ids) of probe/base prompts -> what to do


def bounds(ids, mark):
    out = _bounds(ids, mark)
    how = BIDS.get(len(ids))
    if how is None:
        return out
    starts = [i for i, t in enumerate(ids) if t == mark]
    if how[0] == "base":                   # X: its last message start must be a cut (r0)
        r0 = starts[-1]
        return sorted(set(out) | {r0})
    v = how[1]
    r0, m1, m2 = starts[-3], starts[-2], starts[-1]
    keep = [c for c in out if c <= r0] + [r0]
    keep += {"1": [], "2tail": [m2], "2msg": [m1], "3": [m1, m2]}[v]
    return sorted(set(keep))


W._bounds = bounds

# ---------------------------------------------------------------- instruments (scheduler thread)
EV = []                                    # (kind, t0, t1, info)
_adv, _it = S._advance, S._iterate
SYNC = {"on": False}


def advance(self):
    row = self.task_row
    rid = row.req.id
    w = 0.0
    if SYNC["on"] and rid == CUR["probe"]:
        t = time.perf_counter(); mx.synchronize(); w = time.perf_counter() - t
    t0 = time.perf_counter()
    try:
        return _adv(self)
    finally:
        EV.append(("adv", t0, time.perf_counter(), dict(id=rid, wait=w, rows=len(self.rows),
                                                         done=self.task_row is not row)))


def iterate(self):
    t0 = time.perf_counter()
    try:
        return _it(self)
    finally:
        EV.append(("it", t0, time.perf_counter(), dict(rows=len(self.rows))))


S._advance, S._iterate = advance, iterate
PH = {"on": False, "t": {}}               # phase -> list of ms inside the current probe advance


def _phase(owner, name, key):
    orig = getattr(owner, name)

    def f(*args, **kw):
        if not PH["on"]:
            return orig(*args, **kw)
        mx.synchronize()
        t = time.perf_counter()
        r = orig(*args, **kw)
        if key == "restore":
            mx.eval([x for c in r for x in (list(c.cache) if hasattr(c, "cache") else [c.keys, c.values])
                     if x is not None])
        mx.synchronize()
        PH["t"].setdefault(key, []).append(1e3 * (time.perf_counter() - t))
        return r
    setattr(owner, name, f)


if a.phases:
    _phase(W._PrefixCache, "restore", "restore")
    _phase(W._PrefixCache, "store", "store")
    _phase(S, "_join", "join")
    _eval = mx.eval

    def ev(*args, **kw):
        if not PH["on"]:
            return _eval(*args, **kw)
        t = time.perf_counter()
        r = _eval(*args, **kw)
        mx.synchronize()
        PH["t"].setdefault("eval", []).append(1e3 * (time.perf_counter() - t))
        return r
    mx.eval = ev
    W.mx.eval = ev
    _adv2 = S._advance

    def advance_ph(self):
        on = self.task_row is not None and self.task_row.req.id == CUR["probe"]
        PH["on"], PH["t"] = on, {}
        try:
            return _adv2(self)
        finally:
            if on:
                EV[-1][3]["phases"] = PH["t"]
            PH["on"] = False
    S._advance = advance_ph


# ---------------------------------------------------------------- prompts
def load_calls():
    out = []
    for f in sorted(glob.glob(os.path.join(a.tasks_dir, "*", "logs", "lm_calls.jsonl"))):
        for line in open(f):
            d = json.loads(line)
            m = d["input"]["messages"]
            pt = (d["output"].get("usage") or {}).get("prompt_tokens") or 0
            if m and m[-1]["role"] == "user":
                out.append((f.split(os.sep)[-3], pt, [{"role": x["role"], "content": x.get("content") or ""} for x in m]))
    return out


CALLS = load_calls()


def closest(n, exclude=()):
    best = min((c for c in CALLS if c[0] not in exclude), key=lambda c: abs(c[1] - n))
    return best


def params(max_tokens, ignore_eos=False):
    return pb.Params(max_tokens=max_tokens, ignore_eos=ignore_eos, temperature=0.0, presence_penalty=0.0,
                     seed=7, chat_template_kwargs_json=json.dumps({"enable_thinking": False}))


def request(rid, msgs, max_tokens, ignore_eos=False):
    return pb.Request(id=rid, messages=[pb.Message(role=m["role"], content=m["content"]) for m in msgs],
                      params=params(max_tokens, ignore_eos))


def render_ids(msgs):
    r = request("x", msgs, 1)
    formatted, _ = eng._render(r, [])
    return tok.encode(formatted)


WORDS = ("name id status amount date user playlist song album artist email phone count total "
         "created updated access token page limit offset title description value price").split()


def text_of(seed, n_tok, kind):
    """~n_tok tokens of AppWorld-like text, distinct per seed (first token too)."""
    rnd = random.Random(seed)
    if kind == "reply":
        head = f"Probe {seed}: let me look at the next page of results."
        lines = [head]
        while len(tok.encode("\n".join(lines))) < n_tok - 20:
            lines.append(f"apis.app_{seed}.{rnd.choice(WORDS)}_{rnd.choice(WORDS)}(page_index={rnd.randrange(99)})")
        body = "\n".join(lines)
        return f"{body}\n\n```python\nprint(apis.app_{seed}.show(page_index={seed}))\n```"
    lines = [f"Output {seed}:", "```", "["]
    while len(tok.encode("\n".join(lines))) < n_tok:
        lines.append(f' {{"{rnd.choice(WORDS)}_{seed}": "{rnd.choice(WORDS)} {rnd.randrange(10**6)}", '
                     f'"{rnd.choice(WORDS)}": {rnd.randrange(10**4)}}},')
    return "\n".join(lines) + "\n]\n```\n\n"


def run_req(req, out):
    t_tok = None
    for ev in eng.generate(req):
        k = ev.WhichOneof("kind")
        if k == "token" and t_tok is None:
            t_tok = time.perf_counter()
        elif k == "done":
            out.update(finish=ev.done.finish, usage=ev.done.usage)
        elif k == "error":
            out.update(error=ev.error.msg)
    out["t_tok"] = t_tok


# background rows
bg_out, bg_th, used = [], [], set()
for i in range(a.bg):
    task, pt, msgs = closest(a.bg_tokens, used)
    used.add(task)
    o = {}
    th = threading.Thread(target=run_req, args=(request(f"bg-{i}", msgs, 60000, ignore_eos=True), o), daemon=True)
    th.start()
    bg_out.append(o); bg_th.append(th)
    print(f"bg {i}: task {task} ~{pt} tokens", flush=True)

# bases
BASES = {}
for k, n in enumerate(int(x) for x in a.base.split(",")):
    task, pt, msgs = closest(n, used)
    used.add(task)
    ids = render_ids(msgs)
    BIDS[len(ids)] = ("base",)
    o = {}
    run_req(request(f"base-{n}", msgs, 1), o)
    BASES[n] = msgs
    print(f"base {n}: task {task}, {len(ids)} tokens, {o.get('finish')} {o.get('error', '')}", flush=True)

while True:                                # every background row decoding
    if len(eng.sched.rows) >= a.bg and eng.sched.task is None and not eng.sched.waiting:
        break
    time.sleep(1)
print("background rows decoding", flush=True)
time.sleep(5)

variants = a.variants.split(",")
res = []
seed = 0
for r in range(a.rounds):
    k = r % len(variants)
    order = variants[k:] + variants[:k]
    for n in BASES:
        for v in order:
            seed += 1
            name, _, flag = v.partition("/")
            msgs = BASES[n] + [{"role": "assistant", "content": text_of(seed, a.reply, "reply")},
                               {"role": "user", "content": text_of(seed, a.tool, "tool")}]
            ids = render_ids(msgs)
            BIDS[len(ids)] = ("probe", name)
            rid = f"probe-{seed}"
            CUR["probe"] = rid
            SYNC["on"] = flag == "s"
            o = {}
            e0 = len(EV)
            t_sub = time.perf_counter()
            run_req(request(rid, msgs, 1), o)
            SYNC["on"] = False
            del BIDS[len(ids)]
            ev = EV[e0:]
            adv = [e for e in ev if e[0] == "adv" and e[3]["id"] == rid]
            if not adv:
                print("no slabs?", rid, o); continue
            t_first, t_join = adv[0][1], adv[-1][2]
            its = [e for e in ev if e[0] == "it" and t_first <= e[1] < t_join]
            u = o.get("usage")
            rec = dict(round=r, base=n, variant=v, id=rid, prompt=len(ids), slabs=len(adv),
                       slab_ms=[1e3 * (e[2] - e[1]) for e in adv], wait_ms=[1e3 * e[3]["wait"] for e in adv],
                       adv_ms=sum(1e3 * (e[2] - e[1]) for e in adv), steps=len(its),
                       steps_ms=sum(1e3 * (e[2] - e[1]) for e in its),
                       ttft_ms=1e3 * (t_join - t_first), tok_ms=1e3 * ((o.get("t_tok") or t_join) - t_first),
                       queue_ms=1e3 * (t_first - t_sub), cached=u.cached_tokens if u else None,
                       rows=adv[0][3]["rows"], err=o.get("error"), phases=[e[3].get("phases") for e in adv])
            res.append(rec)
            print(f"r{r} base {n} {v:8s} prompt {len(ids)} cached {rec['cached']} slabs {rec['slabs']} "
                  f"[{', '.join(f'{x:.0f}' for x in rec['slab_ms'])}] adv {rec['adv_ms']:.0f} wait "
                  f"{sum(rec['wait_ms']):.0f} steps {rec['steps']} ({rec['steps_ms']:.0f} ms) ttft {rec['ttft_ms']:.0f} "
                  f"rows {rec['rows']}" + (" phases " + json.dumps([{k: [round(x) for x in v] for k, v in (p or {}).items()}
                                                                   for p in rec["phases"]]) if a.phases else ""), flush=True)
            time.sleep(1.0)

for i in range(a.bg):
    eng.cancel(f"bg-{i}")
for th in bg_th:
    th.join(timeout=60)
# steady decode step (rows = bg, no prefill around): iterate events with an iterate before and after
st = [e for i, e in enumerate(EV[1:-1], 1) if e[0] == "it" and EV[i - 1][0] == "it" and EV[i + 1][0] == "it"
      and e[3]["rows"] == a.bg]
step_ms = statistics.median(1e3 * (e[2] - e[1]) for e in st) if st else float("nan")
after = [EV[i + 1] for i, e in enumerate(EV[:-1]) if e[0] == "adv" and EV[i + 1][0] == "it"]
after_ms = statistics.median(1e3 * (e[2] - e[1]) for e in after) if after else float("nan")
print(f"\nsteady decode step ({a.bg} rows): {step_ms:.1f} ms (n={len(st)}); first step after a slab: "
      f"{after_ms:.1f} ms (n={len(after)})")
print("\n| base | variant | n | slabs | slab ms (median each) | sum _advance ms | wait ms | decode steps in TTFT | TTFT ms |")
print("|---|---|---|---|---|---|---|---|---|")
for n in BASES:
    for v in variants:
        xs = [x for x in res if x["base"] == n and x["variant"] == v and not x["err"]]
        if not xs:
            continue
        md = lambda f: statistics.median(f(x) for x in xs)  # noqa: E731
        ns = xs[0]["slabs"]
        per = [statistics.median(x["slab_ms"][j] for x in xs if len(x["slab_ms"]) > j) for j in range(ns)]
        print(f"| {n} | {v} | {len(xs)} | {ns} | {' + '.join(f'{p:.0f}' for p in per)} | {md(lambda x: x['adv_ms']):.0f} | "
              f"{md(lambda x: sum(x['wait_ms'])):.0f} | {md(lambda x: x['steps']):.0f} | {md(lambda x: x['ttft_ms']):.0f} |")
if a.json:
    json.dump(dict(rows=res, step_ms=step_ms, after_ms=after_ms, args=vars(a)), open(a.json, "w"), indent=1)
sys.stdout.flush()
os._exit(0)
