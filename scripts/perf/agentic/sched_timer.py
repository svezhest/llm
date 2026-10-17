#!/usr/bin/env python3
"""Run the worker with its scheduler thread's time split logged (no repo edits).

Wraps _Scheduler._advance (a prefill slab, synced by its mx.eval: ~GPU time of the
slab + the decode step queued before it), _Scheduler._iterate (one decode step; with
step-ahead it waits for the previous step: ~GPU time of a step) and _Scheduler._admit,
and logs every 10 s through the worker's logger:

  sched_time: wall=10.0 advance=6.12 iterate=3.70 admit=0.01 idle=0.17 slabs=5 steps=21 by_rows=5:12/2.10,6:9/1.60

(by_rows = rows: steps / seconds in _iterate). Per prefill tick (one _advance = one slab;
the first one also does the cache lookup + restore, the last one the store + _join) a line

  sched_slab: id=... pos=10240 n=432 cut=end last=1 first=0 rows=6 t_ms=812.3 lookup_ms=0.4 restore_ms=0.0
              store_ms=35.1 ckpt=1 join_ms=0.3 prompt=11015 reused=10208

(pos = cache offset before the slab, n = tokens prefilled in it, t_ms = the whole _advance:
with rows decoding it also waits for the decode step queued ahead of it; SCHED_SLAB_SYNC=1
syncs first and logs that wait as wait_ms, a small change of the schedule; SCHED_SLAB_LOG=0
turns the lines off). Launch exactly like worker.py, from
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
SLAB_LOG = os.environ.get("SCHED_SLAB_LOG", "1") == "1"
SLAB_SYNC = os.environ.get("SCHED_SLAB_SYNC", "0") == "1"
cur = collections.defaultdict(float)       # sub-phase times (s) / counts inside the current _advance
phase = {"in": None}


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


def _offset(row):
    for c in (row.cache or []) if row is not None else []:
        if not hasattr(c, "cache") and hasattr(c, "offset"):
            return int(c.offset)
    return None


def timed(owner, name, key, count=False):
    orig = getattr(owner, name)

    def f(*a, **kw):
        t = time.perf_counter()
        try:
            return orig(*a, **kw)
        finally:
            if phase["in"] is not None:
                cur[key] += time.perf_counter() - t
                if count:
                    cur[key + "_n"] += 1
    setattr(owner, name, f)


if SLAB_LOG:
    import mlx.core as mx
    PC = W._PrefixCache
    timed(PC, "lookup", "lookup")
    timed(PC, "restore", "restore")
    timed(PC, "store", "store", count=True)   # store calls _recurrent_states once itself
    timed(S, "_join", "join")
    timed(W, "_recurrent_states", "ckpt", count=True)   # the closure calls the module global
    orig_adv = S._advance

    def advance(self):
        row = self.task_row
        before = _offset(row)
        cur.clear()
        phase["in"] = "advance"
        rows = len(self.rows)
        wait = 0.0
        if SLAB_SYNC:
            t = time.perf_counter()
            mx.synchronize()
            wait = time.perf_counter() - t
        t = time.perf_counter()
        try:
            return orig_adv(self)
        finally:
            dt = time.perf_counter() - t
            phase["in"] = None
            try:
                req = row.req
                done = self.task_row is not row
                after = len(req.ids) - 1 if done and row.y is not None else _offset(row)
                reused = getattr(req, "reused", 0) or 0
                pos = before if before is not None else reused
                n = (after - pos) if after is not None else -1
                cut = ("end" if done else "?" if after is None or after >= len(req.ids)
                       else "msg" if req.ids[after] == self.eng._im_start else "step")
                plan = ""
                if before is None and not req.images:
                    # the first slab: the prompt's slab ends past the resume point (m = a message
                    # start, s = a step), and the message starts the chain policy keeps (the end of
                    # the system block = starts[1], the last 3)
                    im = self.eng._im_start
                    allb = self._bounds(req)
                    b = [c for c in allb if c > pos]
                    prev = max((c for c in allb if c <= pos), default=0)   # the cut chain's last end
                    starts = [i for i, t in enumerate(req.ids) if t == im]
                    plan = (" plan=" + ",".join(f"{c}{'m' if req.ids[c] == im else 's'}" for c in b)
                            + " marks=" + ",".join(str(x) for x in starts[1:2] + starts[-3:])
                            + f" nstarts={len(starts)} prev={prev}")
                W.log.info("sched_slab: id=%s pos=%d n=%d cut=%s first=%d last=%d rows=%d t_ms=%.1f%s lookup_ms=%.1f "
                           "restore_ms=%.1f store_ms=%.1f ckpt=%d join_ms=%.1f prompt=%d reused=%d%s",
                           req.id, pos, n, cut, int(before is None), int(done), rows, 1000 * dt,
                           f" wait_ms={1000 * wait:.1f}" if SLAB_SYNC else "", 1000 * cur["lookup"],
                           1000 * cur["restore"], 1000 * cur["store"], int(cur["ckpt_n"] - cur["store_n"]),
                           1000 * cur["join"], len(req.ids), reused, plan)
            except Exception as e:  # noqa: BLE001 — logging only
                W.log.info("sched_slab: log failed: %r", e)
    S._advance = advance

wrap("_advance", "advance", "slabs")
wrap("_iterate", "iterate", "steps")
wrap("_admit", "admit")
W.log.info("sched_timer: scheduler time split every 10 s (worker module %s)", W.__file__)
W.main()
