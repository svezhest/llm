"""Batch-invariant mode on the real model (GPU lease): a row decoded in a batch
gives the same bits as alone. In-process (the real Engine + _Scheduler, no
gRPC), the Engine built from the env exactly as worker.py main() does; run
with QWEN_BATCH_INVARIANT=1 (run_gpu.zsh sets the env).

Phases, all on one loaded model, nothing else on the GPU:
  ref      every prompt alone (B = 1), greedy, ignore_eos, G tokens, logprobs
           + top-5 per step; then once more alone with golden = those tokens
           (solo is deterministic, and golden changes nothing when it is the
           model's own answer)
  forced   for B in --B (2, 4, 8, 16): B prompts of mixed lengths, rotated so
           each prompt sits beside other rows / at another batch index each
           time, max_tokens staggered (rows end at different steps: the batch
           shrinks to the B = 1 library call), half of them submitted after
           the first half decodes (rows join a running batch, their prefill
           interleaved with its decode); every row forced with its ref tokens
           (_Req.golden, decode teacher forcing): each step's (log p, top-5)
           must be == ref's
  free     all prompts at once, no golden: the greedy tokens must be == ref's
  control  the mode off (mrqmm.set_invariant(False), between phases, idle),
           forced at B = 16: mismatches expected (the check has teeth). At
           3..8 rows the default config IS the invariant one (KS 8), so the
           off mode differs at 9..16 rows (KS 4) and at B <= 2 (qmv)

Pass: forced and free all equal; control has mismatches. Output: a JSON with
per-phase counts + the first mismatches, a text summary on stdout.

  cd src/worker && QWEN_MODEL=... QWEN_KV_BITS=0 QWEN_BATCH_INVARIANT=1 \\
    QWEN_PROMPT_CACHE=0 QWEN_PRESENCE_PENALTY=0 PYTHONPATH=. python \\
    ../../scripts/perf/batch_invariant/check_real.py --out $PERF_OUT/bi_check.json
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.getcwd())
import mlx.core as mx  # noqa: E402

import mrqmm  # noqa: E402
import worker as W  # noqa: E402
import worker_pb2 as pb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--B", default="2,4,8,16")
ap.add_argument("--n", type=int, default=16, help="prompts")
ap.add_argument("--gen", type=int, default=96, help="tokens per ref row")
ap.add_argument("--lens", default="300,800,1500,3000,6000", help="prompt tokens, cycled")
ap.add_argument("--top", type=int, default=5)
ap.add_argument("--out", default=os.path.join(os.environ.get("PERF_OUT", "/tmp/perf"), "bi_check.json"))
a = ap.parse_args()

TASKS = [
    "Write a Python function that parses an ISO-8601 duration string into seconds",
    "Implement an LRU cache in Rust with O(1) get and put",
    "Write a Go worker pool with bounded goroutines and graceful shutdown",
    "Write a SQL query for the longest streak of consecutive order days per customer",
    "Implement binary search over a rotated sorted array in TypeScript",
    "Write a bash script that rotates and compresses old log files",
    "Explain open addressing versus separate chaining in hash maps",
    "Describe TCP congestion control and how BBR differs from CUBIC",
]


def filler(n_chars, seed):
    words = "alpha beta gamma delta buffer socket thread mutex queue cache index shard".split()
    out, i = [], 0
    while sum(len(x) + 1 for x in out) < n_chars:
        h = hashlib.sha256(f"{seed}/{i}".encode()).digest()
        out.append(f"[{seed}:{i:04d}] " + " ".join(words[b % len(words)] for b in h[:8]))
        i += 1
    return "\n".join(out)


assert W.BATCH_INVARIANT, "run with QWEN_BATCH_INVARIANT=1"
eng = W.Engine(os.environ["QWEN_MODEL"])
eng.load()
tok = getattr(eng.processor, "tokenizer", eng.processor)
assert mrqmm.INVARIANT and not eng.sched.cascade and eng.draft is None
print(f"loaded; max_batch {eng.max_batch}", flush=True)
LENS = [int(x) for x in a.lens.split(",")]


def prompt_ids(i):
    """Chat-rendered ids of prompt i (~LENS[i % len] tokens: filler lines)."""
    n = LENS[i % len(LENS)]
    lines = filler(int(n * 4.5) + 2000, f"bi{i}").split("\n")
    lo, hi = 1, len(lines)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tok.encode("\n".join(lines[:mid]))) <= n:
            lo = mid
        else:
            hi = mid - 1
    req = pb.Request(id=f"p{i}", messages=[
        pb.Message(role="system", content="Reference notes follow.\n" + "\n".join(lines[:lo])),
        pb.Message(role="user", content=TASKS[i % len(TASKS)] + ". Answer in detail.")],
        params=pb.Params(chat_template_kwargs_json=json.dumps({"enable_thinking": False})))
    formatted, _ = eng._render(req, [])
    return tok.encode(formatted)


PROMPTS = [prompt_ids(i) for i in range(a.n)]
print("prompt tokens:", [len(p) for p in PROMPTS], flush=True)


def make(i, rid, gen, golden=None):
    p = pb.Params(max_tokens=gen, temperature=0.0, presence_penalty=0.0, ignore_eos=True,
                  logprobs=True, top_logprobs=a.top)
    sampling = dict(W.SAMPLING_KWARGS)
    for f in ("temperature", "presence_penalty"):
        sampling[f] = getattr(p, f)
    ctl = W._Control(p, tok.stopping_criteria.eos_token_ids, tok, eng.vocab)
    r = W._Req(id=rid, p=p, ids=list(PROMPTS[i]), max_tokens=gen, sampling=sampling, ctl=ctl)
    r.golden = golden
    return r


def collect(r, out):
    toks, lps = [], []
    while True:
        ev = r.out.get()
        if ev[0] == "tok":
            toks.append(ev[1])
            lps.append(ev[3])
        else:
            out[r.id] = dict(toks=toks, lps=lps, end=list(ev), batch_mean=r.batch_mean)
            return


def run(reqs, stagger=0):
    """Submit reqs (the last `stagger` once the first ones decode), wait."""
    out, ths = {}, []
    first = reqs[:len(reqs) - stagger]
    for r in first:
        eng.sched.submit(r)
        ths.append(threading.Thread(target=collect, args=(r, out)))
        ths[-1].start()
    if stagger:
        t0 = time.time()
        while eng.sched.steps < 8 + START[0] and time.time() - t0 < 600:
            time.sleep(0.02)
        for r in reqs[len(first):]:
            eng.sched.submit(r)
            ths.append(threading.Thread(target=collect, args=(r, out)))
            ths[-1].start()
    for t in ths:
        t.join()
    return out


START = [0]


def cmp(ref, got, n):
    """-> (steps compared, mismatching steps, first mismatch, max |d log p|)"""
    bad, first, dmax = 0, None, 0.0
    for k in range(n):
        x, y = ref["lps"][k], got["lps"][k]
        d = abs(x[0] - y[0])
        dmax = max(dmax, d)
        if x != y or ref["toks"][k] != got["toks"][k]:
            bad += 1
            if first is None:
                first = dict(step=k, ref=[ref["toks"][k], x], got=[got["toks"][k], y])
    return n, bad, first, dmax


report = {"lens": [len(p) for p in PROMPTS], "gen": a.gen, "phases": {}}
t = time.time()
REF = {}
for i in range(a.n):
    REF[i] = run([make(i, f"ref{i}", a.gen)])[f"ref{i}"]
print(f"ref: {a.n} solo runs in {time.time() - t:.0f}s", flush=True)
again = {i: run([make(i, f"refg{i}", a.gen, REF[i]["toks"])])[f"refg{i}"] for i in range(min(4, a.n))}
solo_ok = all(again[i]["toks"] == REF[i]["toks"] and again[i]["lps"] == REF[i]["lps"] for i in again)
report["phases"]["solo_repeat_golden"] = {"equal": solo_ok}
print(f"solo repeat with golden: {'equal' if solo_ok else 'DIFFERENT'}", flush=True)


def forced_phase(B, tag, rot):
    idx = [(rot + j) % a.n for j in range(B)]
    gens = [a.gen - 7 * (j % 8) for j in range(B)]
    reqs = [make(i, f"{tag}{B}-{j}", g, REF[i]["toks"]) for j, (i, g) in enumerate(zip(idx, gens))]
    START[0] = eng.sched.steps
    t = time.time()
    out = run(reqs, stagger=B // 2)
    rows, steps, bad, dmax, firsts = 0, 0, 0, 0.0, []
    for j, (i, g) in enumerate(zip(idx, gens)):
        got = out[f"{tag}{B}-{j}"]
        n, b, f, d = cmp(REF[i], got, g)
        rows, steps, bad, dmax = rows + 1, steps + n, bad + (b > 0), max(dmax, d)
        if f is not None and len(firsts) < 4:
            firsts.append(dict(prompt=i, slot=j, batch_mean=got["batch_mean"], **f))
    res = dict(rows=rows, steps=steps, rows_bad=bad, max_dlp=dmax, first=firsts,
               mean_batch=sum(out[k]["batch_mean"] for k in out) / len(out), s=time.time() - t)
    print(f"{tag} B={B}: {rows} rows, {steps} steps, rows with a mismatch {bad}, "
          f"max |d log p| {dmax:.3g}, mean batch {res['mean_batch']:.1f}, {res['s']:.0f}s", flush=True)
    return res


for k, B in enumerate(int(x) for x in a.B.split(",")):
    report["phases"][f"forced_B{B}"] = forced_phase(B, "f", 3 * k)

# free run: all prompts at once, plain greedy
t = time.time()
START[0] = eng.sched.steps
out = run([make(i, f"free{i}", a.gen) for i in range(a.n)], stagger=a.n // 2)
fb = [i for i in range(a.n) if out[f"free{i}"]["toks"] != REF[i]["toks"]]
lb = [i for i in range(a.n) if out[f"free{i}"]["lps"] != REF[i]["lps"]]
report["phases"]["free"] = dict(rows=a.n, rows_tokens_differ=fb, rows_lps_differ=lb, s=time.time() - t)
print(f"free B={a.n}: tokens differ in {len(fb)} rows, logprobs in {len(lb)}", flush=True)

# control: mode off (idle now: the scheduler waits for work)
mrqmm.set_invariant(False)
report["phases"]["control_off"] = forced_phase(min(16, a.n), "c", 1)
mrqmm.set_invariant(True)

ph = report["phases"]
ok = (ph["solo_repeat_golden"]["equal"]
      and all(v["rows_bad"] == 0 for k, v in ph.items() if k.startswith("forced"))
      and not ph["free"]["rows_tokens_differ"] and not ph["free"]["rows_lps_differ"]
      and ph["control_off"]["rows_bad"] > 0)
report["pass"] = ok
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(report, open(a.out, "w"), indent=1)
print("PASS" if ok else "FAIL", "->", a.out, flush=True)
os._exit(0 if ok else 1)
