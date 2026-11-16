"""Plain greedy decode speed of the loaded model + a fixed GPU bandwidth probe
in the same process, for comparing two checkpoints that cannot share a
process (two 27B builds do not fit at once). The GPU drifts +-20-30 % over
minutes (CLAUDE.md "Measuring performance"): run this for model A, B, A, B,
... (gate_oq4e.sh step 5) and compare tok/s per probe GB/s, not raw tok/s.

Real Engine + _Scheduler (no gRPC), whatever the env loads (QWEN_DRAFT unset:
plain decode). Per request: a system document of --tokens tokens (the repo's
docs) + a coding task, greedy, thinking off, --max-tokens tokens, prompt
cache off; decode tok/s = (n - 1) / (last - first token). Before each request
the scheduler thread (the only MLX thread after load: the probe runs at the
start of the tick that admits the request) streams a fixed set of random
4-bit weights (--probe-mb MB of 17408 x 5120 matrices, MLX qmv, 1 row) and
reports GB/s.

  cd src/worker && QWEN_MODEL=../../weights/qwen38-27b-mlx-oq4e QWEN_KV_BITS=0 QWEN_TEXT_ONLY=1 \\
    QWEN_WIRED_LIMIT_GB=27 QWEN_PROMPT_CACHE=0 QWEN_MAX_BATCH=1 PYTHONPATH=. .venv/bin/python \\
    ../../scripts/perf/dflash2/decode_probe.py --tokens 2000 --requests 4 --json $PERF_OUT/oq4e/probe_a1.json
"""
import argparse
import json
import logging
import os
import statistics
import sys
import threading
import time

sys.path.insert(0, os.getcwd())
import worker as W  # noqa: E402
import worker_pb2 as pb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "prof"))
from wtest import TASKS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, default=2000, help="system document tokens")
ap.add_argument("--requests", type=int, default=4)
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--probe-mb", type=int, default=1024)
ap.add_argument("--probe-reps", type=int, default=3, help="passes over the probe weights per probe")
ap.add_argument("--json", default="")
a = ap.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
import mlx.core as mx  # noqa: E402

t_load = time.time()
eng = W.Engine(os.environ["QWEN_MODEL"])
eng.load()
assert eng.draft is None, "plain decode only: unset QWEN_DRAFT"
tok = getattr(eng.processor, "tokenizer", eng.processor)

# --- the probe, run on the scheduler thread ---------------------------------
PROBE = {"want": False, "done": threading.Event(), "gbps": None, "w": None}
N, K = 17408, 5120


def _probe():
    if PROBE["w"] is None:
        n = max(1, (a.probe_mb << 20) // (N * K // 2 + N * K // 64 * 4))
        ws = []
        for i in range(n):
            w = mx.random.randint(0, 256, (N, K // 2), dtype=mx.uint8, key=mx.random.key(i)).view(mx.uint32)
            s = (mx.random.uniform(shape=(N, K // 64), key=mx.random.key(1000 + i)) * 1e-3).astype(mx.float16)
            ws.append((w, s, mx.zeros_like(s)))
        mx.eval(ws)
        PROBE["w"] = ws
        PROBE["x"] = mx.random.normal((1, 1, K)).astype(mx.float16)
    ws, x = PROBE["w"], PROBE["x"]

    def once():
        return [mx.quantized_matmul(x, *w, transpose=True, group_size=64, bits=4) for w in ws]
    mx.eval(once())                                          # warm
    t = time.perf_counter()
    for _ in range(a.probe_reps):
        mx.eval(once())
    dt = time.perf_counter() - t
    nbytes = sum(w.nbytes + s.nbytes + b.nbytes for w, s, b in ws) * a.probe_reps
    return nbytes / dt / 1e9


_tick = W._Scheduler.tick


def tick(self):
    if PROBE["want"]:
        PROBE["want"] = False
        try:
            PROBE["gbps"] = _probe()
        finally:
            PROBE["done"].set()
    return _tick(self)


W._Scheduler.tick = tick

# --- requests ---------------------------------------------------------------
text = ""
for f in ["README.md", "CLAUDE.md", "BACKLOG.md"]:
    text += open(os.path.join(ROOT, f)).read() + "\n\n"
ids = tok.encode(text)
assert len(ids) >= a.tokens, len(ids)
DOC = ("You are a coding assistant. Project notes follow; use them only if relevant.\n\n"
       + tok.decode(ids[:a.tokens]))


def run(task, rid):
    req = pb.Request(id=rid, messages=[pb.Message(role="system", content=DOC), pb.Message(role="user", content=task)],
                     params=pb.Params(temperature=0.0, presence_penalty=0.0, max_tokens=a.max_tokens,
                                      prompt_cache="off",
                                      chat_template_kwargs_json=json.dumps({"enable_thinking": False})))
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
    PROBE["done"].clear()
    PROBE["want"] = True
    eng.sched.submit(r)
    times = []
    while True:
        ev = r.out.get()
        if ev[0] != "tok":
            break
        times.append(time.perf_counter())
    PROBE["done"].wait(60)
    n = len(times)
    tps = (n - 1) / (times[-1] - times[0]) if n > 2 else 0.0
    return dict(n=n, prompt=len(pids), tps=tps, probe_gbps=PROBE["gbps"])


print(f"loaded in {time.time() - t_load:.0f} s; active {mx.get_active_memory() / 2**30:.2f} GiB", flush=True)
run("Say OK.", "warm")
out = []
for i in range(a.requests):
    o = run(TASKS[i % len(TASKS)], f"r{i}")
    o["tps_per_100gbps"] = 100.0 * o["tps"] / o["probe_gbps"] if o["probe_gbps"] else None
    out.append(o)
    print(f"r{i} n={o['n']} prompt={o['prompt']} tps={o['tps']:.2f} probe={o['probe_gbps']:.0f} GB/s "
          f"tps/100GBps={o['tps_per_100gbps']:.3f}", flush=True)
summ = dict(model=os.path.basename(os.path.normpath(os.environ["QWEN_MODEL"])),
            tps=statistics.median(o["tps"] for o in out),
            probe_gbps=statistics.median(o["probe_gbps"] for o in out),
            tps_per_100gbps=statistics.median(o["tps_per_100gbps"] for o in out),
            active_gib=mx.get_active_memory() / 2**30, requests=out, args=vars(a))
print(f"SUMMARY {summ['model']}: tps {summ['tps']:.2f}  probe {summ['probe_gbps']:.0f} GB/s  "
      f"tps/100GBps {summ['tps_per_100gbps']:.3f}", flush=True)
if a.json:
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    json.dump(summ, open(a.json, "w"), indent=1)
sys.stdout.flush()
os._exit(0)          # the scheduler thread never returns
