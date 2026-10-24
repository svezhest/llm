"""Batch-invariant mode, decode step cost on the real model (GPU lease):
in-process A/B, the mode toggled every --ab-steps decode steps inside one
run (mrqmm.set_invariant on the scheduler thread at the start of a tick), B
rows of ~1K-token prompts decoding (ignore_eos), no prefill pending.

The step-ahead makes the period between two decode ticks the GPU's step time;
a tick's step is built with the variant set at its start, so the first 2
steps after a switch are dropped (the next tick waits for the previous
variant's step), and so are the first 2 windows of each run (the default
configs' kernels compile on first use: only the mode's were warmed at load).
Per B and round: median period on / off; overall: the median of the per-round
ratios. With PROMPT_CACHE off there are no cascade groups either way, so the
A/B is the matmul dispatch alone (on: mrqmm for 1..16 rows with KS 8; off:
qmv at 1..2 rows, mrqmm 2 x 8 / 4 x 4 at 3..8 / 9..16).

  cd src/worker && QWEN_MODEL=... QWEN_KV_BITS=0 QWEN_BATCH_INVARIANT=1 QWEN_PROMPT_CACHE=0 \\
    QWEN_BATCH_PREFILL_SHARE=1 PYTHONPATH=. python ../../scripts/perf/batch_invariant/step_ab.py \\
    --B 1,2,4,8,16 --rounds 3 --out $PERF_OUT/bi_step_ab.json
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import threading
import time

sys.path.insert(0, os.getcwd())
import mrqmm  # noqa: E402
import worker as W  # noqa: E402
import worker_pb2 as pb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--B", default="1,2,4,8,16")
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--ab-steps", type=int, default=16)
ap.add_argument("--windows", type=int, default=12, help="measured windows per variant and run")
ap.add_argument("--out", default=os.path.join(os.environ.get("PERF_OUT", "/tmp/perf"), "bi_step_ab.json"))
a = ap.parse_args()
GEN = (2 * a.windows + 4) * a.ab_steps + 64

assert W.BATCH_INVARIANT, "run with QWEN_BATCH_INVARIANT=1 (installs mrqmm on every module)"
eng = W.Engine(os.environ["QWEN_MODEL"])
eng.load()
tok = getattr(eng.processor, "tokenizer", eng.processor)
print(f"loaded; max_batch {eng.max_batch}, cascade {eng.sched.cascade}", flush=True)


def filler(n_chars, seed):
    words = "alpha beta gamma delta buffer socket thread mutex queue cache index shard".split()
    out, i = [], 0
    while sum(len(x) + 1 for x in out) < n_chars:
        h = hashlib.sha256(f"{seed}/{i}".encode()).digest()
        out.append(f"[{seed}:{i:04d}] " + " ".join(words[b % len(words)] for b in h[:8]))
        i += 1
    return "\n".join(out)


def make(i, rid):
    req = pb.Request(id=rid, messages=[
        pb.Message(role="system", content="Reference notes follow.\n" + filler(1500, f"ab{i}")),
        pb.Message(role="user", content=f"Task {i}: summarize the notes in great detail.")],
        params=pb.Params(chat_template_kwargs_json=json.dumps({"enable_thinking": False})))
    formatted, _ = eng._render(req, [])
    ids = tok.encode(formatted)
    p = pb.Params(max_tokens=GEN, temperature=0.0, presence_penalty=0.0, ignore_eos=True)
    sampling = dict(W.SAMPLING_KWARGS, temperature=0.0, presence_penalty=0.0)
    ctl = W._Control(p, tok.stopping_criteria.eos_token_ids, tok, eng.vocab)
    return W._Req(id=rid, p=p, ids=ids, max_tokens=GEN, sampling=sampling, ctl=ctl)


S = W._Scheduler
_iter = S._iterate
MEAS = {"on": False, "t_on": 0}
FIXED = [None]
REC = []


def iterate(self):
    if MEAS["on"]:
        k = self.steps - MEAS["t_on"]
        want = (k // a.ab_steps) % 2 == 0          # even windows: the mode on
        if FIXED[0] is not None:
            want = FIXED[0]
        if want != mrqmm.INVARIANT:
            mrqmm.set_invariant(want)
        REC.append((time.perf_counter(), len(self.rows), self.task is not None, want,
                    k % a.ab_steps < 2 or k < 2 * a.ab_steps))
    return _iter(self)


S._iterate = iterate


def drain(r):
    while r.out.get()[0] == "tok":
        pass


report = {"gen": GEN, "ab_steps": a.ab_steps, "runs": []}
for rnd in range(a.rounds):
    Bs = [int(x) for x in a.B.split(",")]
    for B in (Bs if rnd % 2 == 0 else Bs[::-1]):
      for fx in (True, False, None):
        FIXED[0] = fx
        mrqmm.set_invariant(True)
        reqs = [make(i, f"ab{rnd}-{B}-{i}") for i in range(B)]
        ths = [threading.Thread(target=drain, args=(r,)) for r in reqs]
        for r, t in zip(reqs, ths):
            eng.sched.submit(r)
            t.start()
        while not (len(eng.sched.rows) == B and eng.sched.task is None):
            time.sleep(0.01)
        REC.clear()
        MEAS["t_on"], MEAS["on"] = eng.sched.steps, True
        for t in ths:
            t.join()
        MEAS["on"] = False
        mrqmm.set_invariant(True)
        per = {True: [], False: []}
        for (t0, b0, task0, v0, skip0), (t1, *_r) in zip(REC, REC[1:]):
            if b0 == B and not task0 and not skip0:
                per[v0].append(1e3 * (t1 - t0))
        if fx is not None:
            print(f"fixed {fx} B={B}: {statistics.median(per[fx]):.2f} ms n={len(per[fx])}", flush=True)
            continue
        run = dict(round=rnd, B=B, on_ms=statistics.median(per[True]), off_ms=statistics.median(per[False]),
                   n_on=len(per[True]), n_off=len(per[False]))
        run["x"] = run["on_ms"] / run["off_ms"]
        report["runs"].append(run)
        print(f"round {rnd} B={B:2d}: on {run['on_ms']:.2f} ms, off {run['off_ms']:.2f} ms, "
              f"x{run['x']:.3f} (n {run['n_on']}/{run['n_off']})", flush=True)

summary = {}
for B in sorted({r["B"] for r in report["runs"]}):
    xs = [r["x"] for r in report["runs"] if r["B"] == B]
    summary[B] = dict(x_median=statistics.median(xs), x_rounds=xs,
                      on_ms=statistics.median(r["on_ms"] for r in report["runs"] if r["B"] == B),
                      off_ms=statistics.median(r["off_ms"] for r in report["runs"] if r["B"] == B))
    print(f"B={B:2d}: on/off x{summary[B]['x_median']:.3f} (rounds {', '.join(f'{x:.3f}' for x in xs)}), "
          f"on {summary[B]['on_ms']:.2f} ms off {summary[B]['off_ms']:.2f} ms", flush=True)
report["summary"] = summary
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(report, open(a.out, "w"), indent=1)
print("->", a.out, flush=True)
os._exit(0)
