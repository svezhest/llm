"""In-process profile of the worker's decode step (the real Engine + _Scheduler,
no gRPC). Builds the Engine exactly like worker.py main() (same env), drives it
with B concurrent requests (real handler `Engine.generate`, or a bare consumer
that only drains the token queue), and instruments the scheduler thread by
wrapping methods (no repo edits):

  tick / _iterate / _step / LanguageModel.__call__ (forward graph build, inside
  _step) / _sample (per-row post-processing) / mx.async_eval (graph walk +
  Metal encoding) / mx.array.tolist+item (host wait for the GPU) / _compact /
  _regrow / mx.clear_cache (count) / quantize.

Steady state = decode ticks with exactly B rows, no prefill pending, previous
tick also a B-row decode tick. Prints medians per section (ms per step).

  cd src/worker && QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 \
    QWEN_BATCH_PREFILL_SHARE=1 PYTHONPATH=. .venv/bin/python $SP/prof/step_prof.py \
    --B 1,4,8,16 --ctx 1k --gen 300 [--consumer handler|bare] [--sync] [--dot] [--json out]
"""
import argparse, collections, json, os, statistics, sys, threading, time
sys.path.insert(0, os.getcwd())
import mlx.core as mx
import worker as W
import worker_pb2 as pb

ap = argparse.ArgumentParser()
ap.add_argument("--B", default="1,4,8,16")
ap.add_argument("--ctx", default="1k")          # 1k (~600-token prompts) or 8k (~8K)
ap.add_argument("--gen", type=int, default=300)
ap.add_argument("--consumer", default="handler")
ap.add_argument("--sync", action="store_true", help="eval each step at once (GPU time)")
ap.add_argument("--dot", action="store_true", help="count graph nodes of one steady step")
ap.add_argument("--rounds", type=int, default=1)
ap.add_argument("--json")
ap.add_argument("--ignore-eos", action="store_true")
ap.add_argument("--switch", type=float, default=0.0, help="sys.setswitchinterval (s)")
ap.add_argument("--serve", default="", help="addr: run the gRPC server in-process, drive with --drive")
ap.add_argument("--drive", default="", help="shell command run against the in-process server")
ap.add_argument("--ab", default="", help="comma list of variants alternated per round: new,old")
ap.add_argument("--ab-steps", type=int, default=0, help="alternate --ab variants every N steps inside one run")
ap.add_argument("--samplings", default="greedy", help="comma list alternated per round: greedy,card,pen,cardpen")
ap.add_argument("--cprofile", default="", help="B to run under cProfile (scheduler thread)")
a = ap.parse_args()

TASKS = [
    "Write a Python function that parses an ISO-8601 duration string into seconds",
    "Implement an LRU cache in Rust with O(1) get and put",
    "Write a Go worker pool with bounded goroutines and graceful shutdown",
    "Write a SQL query for the longest streak of consecutive order days per customer",
    "Implement binary search over a rotated sorted array in TypeScript",
    "Write a bash script that rotates and compresses old log files",
    "Write a C function that reverses a singly linked list, iteratively and recursively",
    "Implement a thread-safe token-bucket rate limiter in Python",
    "Write a Kotlin coroutine that fetches three URLs concurrently with a timeout",
    "Write a Python asyncio downloader with bounded concurrency and retries",
    "Implement Dijkstra's algorithm in C++ with a binary heap",
    "Explain open addressing versus separate chaining in hash maps",
    "Describe TCP congestion control and how BBR differs from CUBIC",
    "Explain optimistic versus pessimistic locking with SQL examples",
    "Explain the JVM G1 garbage collector in depth",
    "Describe the on-disk layout of a B-tree index",
]


def filler(n_chars, seed):
    import hashlib
    words = "alpha beta gamma delta buffer socket thread mutex queue cache index shard".split()
    out, i = [], 0
    while sum(len(x) + 1 for x in out) < n_chars:
        h = hashlib.sha256(f"{seed}/{i}".encode()).digest()
        out.append(f"[{seed}:{i:04d}] " + " ".join(words[b % len(words)] for b in h[:8]))
        i += 1
    return "\n".join(out)


NCH = {"1k": 1500, "8k": 28000}[a.ctx]


SAMPLINGS = {
    "greedy": dict(temperature=0.0, presence_penalty=0.0),
    "pen": dict(temperature=0.0, presence_penalty=1.5),
    "card": dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.0),
    "cardpen": dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5),
    "topp": dict(temperature=0.7, top_p=0.9, presence_penalty=0.0),
}
SAMP = ["greedy"]


def request(i, rid):
    sysmsg = "You are a meticulous senior engineer. Reference notes follow.\n" + filler(NCH, f"s{i}")
    user = TASKS[i % len(TASKS)] + ". Answer in great detail, at least 2000 words, with many code examples."
    p = pb.Params(max_tokens=a.gen, ignore_eos=a.ignore_eos, seed=1234 + i, **SAMPLINGS[SAMP[0]],
                  chat_template_kwargs_json=json.dumps({"enable_thinking": False}))
    return pb.Request(id=rid, messages=[pb.Message(role="system", content=sysmsg),
                                        pb.Message(role="user", content=user)], params=p)


# ---------------------------------------------------------------- instruments
SCHED = "scheduler"
rec = []                     # one dict per tick
cur = None
flags = {"step": False, "iter": False}
acc = collections.Counter()


def on_sched():
    return threading.current_thread().name == SCHED


def timed(name, fn, method=True):
    def w(*args, **kw):
        if not on_sched() or cur is None:
            return fn(*args, **kw)
        t, c = time.perf_counter(), time.thread_time()
        try:
            return fn(*args, **kw)
        finally:
            cur[name] = cur.get(name, 0.0) + time.perf_counter() - t
            cur[name + "_cpu"] = cur.get(name + "_cpu", 0.0) + time.thread_time() - c
            cur["n_" + name] = cur.get("n_" + name, 0) + 1
    return w


def old_iterate(self):
    """worker.py _Scheduler._iterate before the step-profile changes (verbatim)."""
    for r in self.rows:
        if r.req.cancel.is_set():
            self._finish(r, "cancelled")
        elif r.n == r.req.max_tokens:
            self._finish(r, "length")
    self._compact()
    if not self.rows:
        return
    rows = self.rows
    ahead = not any(r.sync for r in rows)
    nxt = self._step(rows) if ahead else None
    live = [r for r in rows if not r.first and not r.done]
    if len(live) == 1:
        toks = [live[0].y.item()]
    else:
        toks = mx.concatenate([r.y for r in live]).tolist() if live else []
    for r, tok in zip(live, toks):
        req = r.req
        try:
            if req.thinking is not None:
                req.thinking(tok)
            why = req.ctl.stop(tok) if req.ctl.active else (
                "stop" if self.eng.stopping(tok) else None)
            if why == "stop":
                self._finish(r, "stop")
                continue
            if not ahead:
                if req.guide is not None:
                    req.guide.consume(tok)
                req.ctl.history.append(tok)
            req.out.put(("tok", tok, why != "skip"))
            r.n += 1
        except Exception as e:  # noqa: BLE001
            self._fail(r, e)
    if not ahead:
        self._compact()
        rows = self.rows
        nxt = self._step(rows) if rows else []
    for r, y in zip(rows, nxt):
        if not r.done:
            r.y = r.req.thinking.apply_forced_token(y) if r.req.thinking is not None else y
            r.first = False
    self.steps += 1
    self.failures = 0
    if self.steps % 256 == 0:
        self.clear = True
    self._compact()


IMPL = {"new": None, "old": old_iterate}
VARS_AB = a.ab.split(",") if a.ab else ["new"]
CURV = ["new"]


S = W._Scheduler
_tick, _iter, _step, _sample = S.tick, S._iterate, S._step, S._sample
IMPL["new"] = _iter


def _sleepy(ms):
    def f(self):
        r = _iter(self)
        time.sleep(ms / 1e3)            # extra host time between steps (slack probe)
        return r
    return f


_SR = W.sample_rows


def _per_row(samplers, logits):
    return [s(x) for s, x in zip(samplers, logits)]


def _with_sampling(fn):
    def f(self):
        W.sample_rows = fn
        try:
            return _iter(self)
        finally:
            W.sample_rows = _SR
    return f


IMPL["rowsamp"] = _with_sampling(_per_row)


def _oldall(self):
    """the scheduler step as before the step profile: old token read + per-row draws"""
    W.sample_rows = _per_row
    try:
        return old_iterate(self)
    finally:
        W.sample_rows = _SR


IMPL["oldall"] = _oldall
for _ms in (1, 2, 4, 6, 8, 12):
    IMPL[f"s{_ms}"] = _sleepy(_ms)
last_end = [None]


def tick(self):
    global cur
    t0 = time.perf_counter()
    cur = {"t0": t0, "B0": len(self.rows), "task": self.task is not None,
           "gap": (t0 - last_end[0]) if last_end[0] else 0.0}
    try:
        return _tick(self)
    finally:
        t1 = time.perf_counter()
        cur["tick"] = t1 - t0
        last_end[0] = t1
        rec.append(cur)
        cur = None


def iterate(self):
    flags["iter"] = True
    if a.ab_steps:
        CURV[0] = VARS_AB[(self.steps // a.ab_steps) % len(VARS_AB)]
        if cur is not None:
            cur["v"] = CURV[0]
            cur["sw"] = self.steps % a.ab_steps < 2       # first steps after a switch
    t = time.perf_counter()
    try:
        return IMPL[CURV[0]](self)
    finally:
        flags["iter"] = False
        if cur is not None:
            cur["iterate"] = time.perf_counter() - t
            cur["B"] = len(self.rows)


DOT = {"want": False, "done": None}


def step(self, rows):
    flags["step"] = True
    t = time.perf_counter()
    try:
        return _step(self, rows)
    finally:
        flags["step"] = False
        if cur is not None:
            cur["step"] = cur.get("step", 0.0) + time.perf_counter() - t
            cur["Bstep"] = len(rows)


S.tick, S._iterate, S._step = tick, iterate, step
S._sample = timed("sample", _sample)
S._compact = timed("compact", S._compact)
S._regrow = timed("regrow", S._regrow)

LM = None
_async_eval, _eval = mx.async_eval, mx.eval


def async_eval(*arrs):
    if on_sched() and cur is not None and flags["step"]:
        if DOT["want"]:
            DOT["want"] = False
            path = os.environ.get("PERF_OUT", "/tmp/perf") + "/prof/_step_graph.dot"
            mx.export_to_dot(path, *arrs)
            DOT["done"] = open(path).read()
        t, c = time.perf_counter(), time.thread_time()
        if a.sync:
            _async_eval(*arrs)
            cur["enc_cpu"] = cur.get("enc_cpu", 0.0) + time.thread_time() - c
            cur["enc"] = cur.get("enc", 0.0) + time.perf_counter() - t
            _eval(*arrs)
        else:
            _async_eval(*arrs)
        cur["async_eval"] = cur.get("async_eval", 0.0) + time.perf_counter() - t
        cur["async_cpu"] = cur.get("async_cpu", 0.0) + time.thread_time() - c
        return None
    return _async_eval(*arrs)


mx.async_eval = async_eval
_tolist, _item = mx.array.tolist, mx.array.item


def tolist(self):
    if on_sched() and cur is not None and flags["iter"] and not flags["step"]:
        t = time.perf_counter()
        r = _tolist(self)
        cur["wait"] = cur.get("wait", 0.0) + time.perf_counter() - t
        return r
    return _tolist(self)


def item(self):
    if on_sched() and cur is not None and flags["iter"] and not flags["step"]:
        t = time.perf_counter()
        r = _item(self)
        cur["wait"] = cur.get("wait", 0.0) + time.perf_counter() - t
        return r
    return _item(self)


mx.array.tolist, mx.array.item = tolist, item
mx.clear_cache = timed("clear_cache", mx.clear_cache)

# ---------------------------------------------------------------- engine
if a.switch:
    sys.setswitchinterval(a.switch)
model_path = os.environ.get("QWEN_MODEL")
eng = W.Engine(model_path)
eng.load()
LM = type(eng.model.language_model)
LM.__call__ = timed("fwd", LM.__call__)
eng.sched.quantize = timed("quantize", eng.sched.quantize)
tok = getattr(eng.processor, "tokenizer", eng.processor)
print(f"loaded; max_batch {eng.max_batch} sync={a.sync} consumer={a.consumer}", flush=True)


def run_handler(req, out):
    n = 0
    for ev in eng.generate(req):
        if ev.WhichOneof("kind") == "token":
            n += 1
    out.append(n)


def run_bare(req, out):
    formatted, _ = eng._render(req, [])
    ids = tok.encode(formatted)
    p = req.params
    sampling = dict(W.SAMPLING_KWARGS)
    for f in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "seed"):
        if p.HasField(f):
            sampling[f] = getattr(p, f)
    ctl = W._Control(p, tok.stopping_criteria.eos_token_ids, tok, eng.vocab)
    r = W._Req(id=req.id, p=p, ids=ids, max_tokens=eng._limit(p, len(ids)), sampling=sampling, ctl=ctl,
               formatted=formatted)
    eng.sched.submit(r)
    n = 0
    while True:
        ev = r.out.get()
        if ev[0] != "tok":
            break
        n += 1
    out.append(n)


KEYS = ["period", "tick", "gap", "iterate", "step", "fwd", "fwd_cpu", "sample", "async_eval", "async_cpu", "enc", "enc_cpu", "wait",
        "consume", "compact", "regrow", "quantize", "host"]


def summarize(B, recs):
    st = []
    prev = None
    for i, r in enumerate(recs):
        ok = (r.get("Bstep") == B and r["B0"] == B and not r["task"] and "iterate" in r)
        if ok and prev is not None and i + 1 < len(recs):
            r = dict(r)
            r["period"] = recs[i + 1]["t0"] - r["t0"]
            r["consume"] = r["iterate"] - r.get("step", 0) - r.get("wait", 0) - r.get("compact", 0)
            r["host"] = r["period"] - r.get("wait", 0)
            st.append(r)
        prev = r if ok else None
    if not st:
        return None
    st = st[5:] or st
    med = {k: 1e3 * statistics.median([r.get(k, 0.0) for r in st]) for k in KEYS}
    med["n"] = len(st)
    med["n_sample"] = statistics.median([r.get("n_sample", 0) for r in st])
    med["clear_cache_per_100"] = 100 * sum(r.get("n_clear_cache", 0) for r in st) / len(st)
    return med


def ab_report(recs, B, tag):
    out = {}
    for vv in VARS_AB:
        sel, waits, fwd = [], [], []
        for i in range(len(recs) - 1):
            r = recs[i]
            if (r.get("v") == vv and not r.get("sw") and r.get("Bstep") == B and r["B0"] == B
                    and not r["task"] and recs[i + 1].get("v") == vv and "iterate" in recs[i + 1]):
                sel.append(1e3 * (recs[i + 1]["t0"] - r["t0"]))
                waits.append(1e3 * r.get("wait", 0))
                fwd.append(1e3 * r.get("fwd_cpu", 0))
        if len(sel) >= 8:
            q = statistics.quantiles(sel, n=4)
            out[vv] = statistics.median(sel)
            print(f"{tag} B={B:2d} {vv}: period median {statistics.median(sel):.2f} ms "
                  f"(q1 {q[0]:.1f} q3 {q[2]:.1f}, n={len(sel)}) wait {statistics.median(waits):.2f} "
                  f"fwd_cpu {statistics.median(fwd):.2f}", flush=True)
    return out


results = {}
pr = None
if a.serve:
    import subprocess
    stop_event = threading.Event()
    server, _, _ = W.serve(eng, a.serve, stop_event)
    time.sleep(1.0)
    print("drive:", a.drive, flush=True)
    t0 = time.time()
    subprocess.run(a.drive, shell=True)
    print(f"drive done in {time.time() - t0:.1f}s; ticks {len(rec)}", flush=True)
    recs = list(rec)
    if a.json:
        json.dump(recs, open(a.json + ".recs", "w"))
    nd = sum(1 for r in recs if "iterate" in r)
    print(f"decode ticks {nd}, prefill ticks {sum(1 for r in recs if 'iterate' not in r)}", flush=True)
    for B in sorted({r.get("Bstep") for r in recs if r.get("Bstep")}):
        if a.ab_steps:
            results[f"serve_{B}"] = ab_report(recs, B, "serve")
            continue
        m = summarize(B, recs)
        results[f"serve_{B}"] = m
        if m and m["n"] >= 10:
            print(f"serve B={B:2d} n={m['n']} | " + " ".join(f"{k}={m[k]:.2f}" for k in KEYS) +
                  f" | clear/100={m['clear_cache_per_100']:.1f}", flush=True)
    if a.json:
        json.dump(results, open(a.json, "w"), indent=1)
    server.stop(0)
    os._exit(0)
SLIST = a.samplings.split(",")
for rnd in range(a.rounds):
 for sm in (SLIST if rnd % 2 == 0 else SLIST[::-1]):
  SAMP[0] = sm
  for v in ((VARS_AB if rnd % 2 == 0 else VARS_AB[::-1]) if not a.ab_steps else ["mix"]):
    CURV[0] = v if v != "mix" else VARS_AB[0]
    for B in [int(x) for x in a.B.split(",")]:
        rec.clear()
        if a.dot:
            DOT["want"] = False
        outs, ths = [], []
        fn = run_handler if a.consumer == "handler" else run_bare
        t0 = time.time()
        if str(B) == a.cprofile:
            import cProfile
            pr = cProfile.Profile()
        for i in range(B):
            th = threading.Thread(target=fn, args=(request(i, f"prof-{rnd}-{B}-{i}"), outs))
            th.start()
            ths.append(th)
        if a.dot or pr is not None:
            # once all rows run: grab one step's graph / profile the scheduler thread
            while not (len(eng.sched.rows) == B and eng.sched.task is None):
                time.sleep(0.05)
            time.sleep(1.0)
            if a.dot:
                DOT["want"] = True
            if pr is not None:
                sys.setprofile(None)
                threading.setprofile(None)
                # profile the scheduler thread by injecting a profiler via a tick wrapper
                orig = S.tick

                def ptick(self, orig=orig):
                    pr.enable()
                    try:
                        return orig(self)
                    finally:
                        pr.disable()
                S.tick = ptick
                time.sleep(5.0)
                S.tick = orig
        for th in ths:
            th.join()
        wall = time.time() - t0
        if a.ab_steps:
            recs = list(rec)
            for vv in VARS_AB:
                # a tick's period belongs to its variant; drop the steps right after a switch
                sel = []
                for i in range(len(recs) - 1):
                    r = recs[i]
                    if (r.get("v") == vv and not r.get("sw") and r.get("Bstep") == B and r["B0"] == B
                            and not r["task"] and recs[i + 1].get("v") == vv):
                        sel.append(1e3 * (recs[i + 1]["t0"] - r["t0"]))
                if sel:
                    q = statistics.quantiles(sel, n=4)
                    results.setdefault(f"{B}_{vv}", []).append(statistics.median(sel))
                    print(f"round {rnd} B={B:2d} {vv}: period median {statistics.median(sel):.2f} ms "
                          f"(q1 {q[0]:.1f} q3 {q[2]:.1f}, n={len(sel)}) wait "
                          f"{1e3 * statistics.median([r.get('wait', 0) for r in recs if r.get('v') == vv]):.2f}", flush=True)
            continue
        m = summarize(B, list(rec))
        results[f"{rnd}_{sm}_{v}_{B}"] = m
        rr = [r for r in rec if r.get("Bstep") == B and not r["task"]]
        per = [1e3 * (rr[i + 1]["t0"] - rr[i]["t0"]) for i in range(len(rr) - 1)]
        if per:
            ch = max(1, len(per) // 10)
            print("  period by decile:", [round(statistics.median(per[i:i + ch]), 1) for i in range(0, len(per), ch)],
                  " wait:", [round(1e3 * statistics.median([r.get("wait", 0) for r in rr[i:i + ch]]), 1) for i in range(0, len(rr), ch)], flush=True)
        if m:
            print(f"round {rnd} {sm} {v} B={B:2d} ctx={a.ctx} n={m['n']} tokens={sum(outs)} wall={wall:.1f}s | " +
                  " ".join(f"{k}={m[k]:.2f}" for k in KEYS) +
                  f" | samples/step={m['n_sample']} clear/100={m['clear_cache_per_100']:.1f}", flush=True)
        else:
            print(f"round {rnd} B={B}: no steady steps", flush=True)
        if a.dot and DOT["done"]:
            txt = DOT["done"]
            import re
            ops = collections.Counter(re.findall(r'label ="?([A-Za-z_]+)', txt))
            nodes = sum(ops.values())
            print(f"  graph B={B}: {nodes} labelled nodes; top: {ops.most_common(25)}", flush=True)
            open(os.environ.get("PERF_OUT", "/tmp/perf") + f"/prof/_step_graph_B{B}.dot", "w").write(txt)
            DOT["done"] = None
        if pr is not None:
            import pstats, io
            s = io.StringIO()
            pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(30)
            print(s.getvalue()[:6000])
            s = io.StringIO()
            pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(40)
            print(s.getvalue()[:8000])
            pr = None
if a.json:
    json.dump(results, open(a.json, "w"), indent=1)
os._exit(0)
