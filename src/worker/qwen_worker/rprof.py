"""QWEN_SPEC_PROF (tools): where a speculative round's wall time goes.

One RoundProf per request (`req.prof`), marked by _spec (each round, each burst
of plain steps) and by the plain loop (_iterate at B = 1, the reference). The
marks read the host clocks only (perf_counter, the thread's CPU time) and
append Python floats: no MLX call, so a profiled round queues, evaluates and
waits for exactly what an unprofiled one does (tests/test_rprof.py counts the
syncs and evals with the profile on and off). Off (None): one `is not None`
test per mark.

A record runs from the end of the generator's yield to the next round's
start, its spans in phases:

  pre      host before the record's first GPU work is queued (the Chooser,
           the head's gate, the chained-draft graph, a DFlash block's graph;
           for one draft the verify graph too: the seed is already there):
           the GPU has nothing of the request
  build    host from the first queued work to the main sync (the verify and
           sampler graphs, the encode in async_eval): overlaps the drafts
  wait     blocked in the main sync (the walk's tolist; a step's item())
  other    blocked in any other read (the seed's confidence, shadow drafts,
           step1, the budget's steps)
  post     host after the main sync to the yield (commit, own states,
           requant, statistics, the head's settle, the emits)
  between  the yield to the next record's start (the scheduler's tick)
  gpu      the first queued work to the end of the main sync
  exposed  wall - gpu: none of the record's work queued or waited for. An
           upper bound of the GPU's idle time: work queued with the verify
           behind its targets (the head's verify forward) runs into `post`;
           QWEN_SPEC_PROF=2 measures that `tail` in every PROBE-th round
           (a sync right after the walk; those rounds are left out of the
           other spans' statistics)

`cpu_*`: the thread's CPU time in the host phases (less than the wall: the
thread was descheduled or waited for the GIL, e.g. behind a stream handler).
Laps: the host and wait spans by name (summed per record); "gc": Python's
cyclic collector on this thread inside the record (it overlaps the lap it
interrupted: not part of the sum).
"""

import gc
import threading
import time


class RoundProf:
    PROBE = 8                           # QWEN_SPEC_PROF=2: every PROBE-th main sync measures the tail
    SPANS = ("pre", "build", "wait", "other", "post", "between")
    FIELDS = ("wall", "pre", "build", "wait", "other", "post", "between", "gpu", "exposed",
              "cpu_pre", "cpu_build", "cpu_post", "cpu_between", "tokens")

    __slots__ = ("level", "recs", "r", "t", "c", "prev", "t_end", "c_end", "nsync")
    _on = None                          # (the profile whose record is open, its thread) for the gc callback
    _gc_t0 = None

    def __init__(self, level=1):
        self.level = int(level)
        self.recs = []                  # finished records (their `between` known)
        self.r = None                   # the record under way
        self.prev = None                # ended, waiting for the next start
        self.t = self.c = self.t_end = self.c_end = 0.0
        self.nsync = 0

    # ---- marks ----

    @staticmethod
    def _gc(phase, info):
        on = RoundProf._on
        if on is None or on[1] != threading.get_ident():
            return
        if phase == "start":
            RoundProf._gc_t0 = time.perf_counter()
        elif RoundProf._gc_t0 is not None and on[0].r is not None:
            laps = on[0].r["laps"]
            laps["gc"] = laps.get("gc", 0.0) + time.perf_counter() - RoundProf._gc_t0
            RoundProf._gc_t0 = None

    def begin(self):
        """A record starts (after the yield / at the step's entry)."""
        if RoundProf._gc not in gc.callbacks:
            gc.callbacks.append(RoundProf._gc)
        RoundProf._on = (self, threading.get_ident())
        t, c = time.perf_counter(), time.thread_time()
        p = self.prev
        if p is not None:
            p["between"] = t - self.t_end
            p["cpu_between"] = c - self.c_end
            p["wall"] = t - p["t0"]
            self.recs.append(p)
            self.prev = None
        self.r = dict(t0=t, first=None, sync=None, probe=False, laps={}, pre=0.0, build=0.0,
                      wait=0.0, other=0.0, post=0.0, cpu_pre=0.0, cpu_build=0.0, cpu_post=0.0)
        self.t, self.c = t, c

    def _phase(self, r):
        return "pre" if r["first"] is None else "build" if r["sync"] is None else "post"

    def lap(self, name):
        """Host time since the last mark, as `name` (its phase: pre / build / post)."""
        r = self.r
        if r is None:
            return
        t, c = time.perf_counter(), time.thread_time()
        ph = self._phase(r)
        d = t - self.t
        r[ph] += d
        r["cpu_" + ph] += c - self.c
        r["laps"][name] = r["laps"].get(name, 0.0) + d
        self.t, self.c = t, c

    def queued(self, name):
        """lap(name), then the record's first GPU work is queued (call it
        right before that async_eval: its encode is `build`)."""
        self.lap(name)
        r = self.r
        if r is not None and r["first"] is None:
            r["first"] = self.t

    def wait(self, name, main=False):
        """Blocked since the last mark (a read of a lazy array): `main` is
        the record's sync (the walk; each step of a burst)."""
        r = self.r
        if r is None:
            return
        t, c = time.perf_counter(), time.thread_time()
        d = t - self.t
        r["wait" if main else "other"] += d
        r["laps"][name] = r["laps"].get(name, 0.0) + d
        if main:
            if r["first"] is None:
                r["first"] = self.t
            r["sync"] = t
        self.t, self.c = t, c

    def probe(self):
        """QWEN_SPEC_PROF=2: does this main sync measure the tail (the caller
        then synchronizes and marks wait("tail"))?"""
        r = self.r
        if self.level < 2 or r is None:
            return False
        self.nsync += 1
        r["probe"] = self.nsync % self.PROBE == 0
        return r["probe"]

    def end(self, key, tokens, steps=1):
        """The record ends (before the yield): `key` its kind ("mtp1",
        "dflash", "mtp0" a burst, "loop" a plain-loop step...), `tokens`
        emitted, `steps` plain steps in it (a burst: per-step statistics)."""
        r = self.r
        if r is None:
            return
        self.lap("rest")
        r.update(key=key, tokens=tokens, steps=max(1, steps))
        self.prev, self.r = r, None
        self.t_end, self.c_end = self.t, self.c

    def drop(self):
        """The record under way is not one (nothing queued: the loop ended)."""
        self.r = None

    # ---- results ----

    @staticmethod
    def _row(r):
        wall = r["wall"]
        gpu = (r["sync"] - r["first"]) if r["sync"] is not None and r["first"] is not None else 0.0
        out = {k: r[k] for k in ("pre", "build", "wait", "other", "post", "between",
                                 "cpu_pre", "cpu_build", "cpu_post", "cpu_between")}
        out.update(wall=wall, gpu=gpu, exposed=wall - gpu)
        s = r["steps"]
        out = {k: 1000.0 * v / s for k, v in out.items()}       # ms (per step)
        out["tokens"] = r["tokens"] / s
        return out

    def raw(self, skip=1):
        """[key, steps, probe, {field: ms}, {lap: ms}] per record (the first
        `skip`: the prompt window, the first seed - left out)."""
        out = []
        for r in list(self.recs)[skip:]:
            s = r["steps"]
            out.append([r["key"], s, r["probe"], {k: round(v, 3) for k, v in self._row(r).items()},
                        {k: round(1000.0 * v / s, 3) for k, v in r["laps"].items()}])
        return out

    @staticmethod
    def stats(xs):
        xs = sorted(xs)
        if not xs:
            return None
        n = len(xs)
        q = lambda p: xs[min(n - 1, int(p * n))]  # noqa: E731
        return dict(n=n, mean=round(sum(xs) / n, 3), p50=round(q(0.5), 3), p90=round(q(0.9), 3))

    @classmethod
    def aggregate(cls, raw):
        """Records of raw() (any number of requests) -> {key: {field: stats,
        "laps": {lap: mean ms}, "tail": stats of the probes, "rounds": n}}."""
        by = {}
        for key, steps, probe, f, laps in raw:
            by.setdefault(key, []).append((probe, f, laps))
        out = {}
        for key, rs in sorted(by.items()):
            main = [(f, laps) for p, f, laps in rs if not p]
            tails = [laps.get("tail", 0.0) for p, f, laps in rs if p]
            d = {k: cls.stats([f[k] for f, _ in main]) for k in cls.FIELDS} if main else {}
            names = sorted({n for _, laps in main for n in laps})
            d["laps"] = {n: round(sum(laps.get(n, 0.0) for _, laps in main) / len(main), 3)
                         for n in names} if main else {}
            d["tail"] = cls.stats(tails)
            d["rounds"] = len(main)
            out[key] = d
        return out

    def summary(self, skip=1):
        raw = self.raw(skip)
        return dict(keys=self.aggregate(raw), raw=raw)
