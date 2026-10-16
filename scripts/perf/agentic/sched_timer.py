#!/usr/bin/env python3
"""Run the worker with its scheduler thread's time split logged (no repo edits).

Wraps _Scheduler._advance (a prefill slab, synced by its mx.eval: ~GPU time of the
slab + the decode step queued before it), _Scheduler._iterate (one decode step; with
step-ahead it waits for the previous step: ~GPU time of a step) and _Scheduler._admit,
and logs every 10 s through the worker's logger:

  sched_time: wall=10.0 advance=6.12 iterate=3.70 admit=0.01 idle=0.17 slabs=5 steps=21 by_rows=5:12/2.10,6:9/1.60

(by_rows = rows: steps / seconds in _iterate). Launch exactly like worker.py, from
src/worker (cascade, worker_pb2 ... are imported from there):

  cd src/worker && QWEN_MODEL=... PYTHONPATH=. uv run python ../../scripts/perf/agentic/sched_timer.py
  SCHED_WORKER_DIR=/path/to/a/frozen/copy  # import that directory's worker.py instead
"""
import collections
import os
import sys
import time

d = os.environ.get("SCHED_WORKER_DIR")
sys.path.insert(0, d if d else os.getcwd())
import worker as W  # noqa: E402

S = W._Scheduler
acc = {"advance": 0.0, "iterate": 0.0, "admit": 0.0, "slabs": 0, "steps": 0}
by_rows = collections.defaultdict(lambda: [0, 0.0])
state = {"t": time.time()}


def emit():
    now = time.time()
    wall = now - state["t"]
    if wall < 10:
        return
    busy = acc["advance"] + acc["iterate"] + acc["admit"]
    W.log.info("sched_time: wall=%.1f advance=%.2f iterate=%.2f admit=%.2f idle=%.2f slabs=%d steps=%d by_rows=%s",
               wall, acc["advance"], acc["iterate"], acc["admit"], max(wall - busy, 0), acc["slabs"],
               acc["steps"], ",".join(f"{k}:{v[0]}/{v[1]:.2f}" for k, v in sorted(by_rows.items())))
    for k in acc:
        acc[k] = 0 if isinstance(acc[k], int) else 0.0
    by_rows.clear()
    state["t"] = now


def wrap(name, key, count=None):
    orig = getattr(S, name)

    def f(self, *a, **kw):
        rows = len(self.rows)
        t = time.perf_counter()
        try:
            return orig(self, *a, **kw)
        finally:
            dt = time.perf_counter() - t
            acc[key] += dt
            if count:
                acc[count] += 1
            if name == "_iterate" and rows:
                by_rows[rows][0] += 1
                by_rows[rows][1] += dt
            emit()
    setattr(S, name, f)


wrap("_advance", "advance", "slabs")
wrap("_iterate", "iterate", "steps")
wrap("_admit", "admit")
W.log.info("sched_timer: scheduler time split every 10 s (worker module %s)", W.__file__)
W.main()
