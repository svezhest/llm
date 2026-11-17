"""QWEN_SPEC_PROF (qwen_worker/rprof.py) on the tiny random model (CPU):
profiling a request changes nothing the GPU sees - the same answer and the
same host reads, evals and queued evals, in the same order, as without it
(level 1; level 2 adds its tail syncs and nothing else) - and its records add
up (the spans of a record are its wall time). The head's gated rounds with
plain bursts, DFlash beside the head with shadow drafts, the plain loop.
Also sched_spec.ROUND_OVERLAP (what the profile measured: the round's reads
wait for what they read only): the same answer and reads either way."""
import random

import mlx.core as mx
import pytest
from mlx_vlm.speculative.common import _speculative_walk

import worker
from conftest import prompts, request, run, scheduler
from test_mtp import Oracle, head
from test_scheduler import FakeDraft, no_metal_verify

# every host wait and every eval (nosync's list) + the queued evals
WAITS = [(mx, "eval"), (mx, "async_eval"), (mx, "synchronize")] + [
    (mx.array, n) for n in ("item", "tolist", "__int__", "__float__", "__bool__", "__index__")]


def counted(monkeypatch, log):
    for owner, name in WAITS:
        fn = getattr(owner, name)

        def w(*a, _fn=fn, _name=name, **k):
            log.append(_name)
            return _fn(*a, **k)
        monkeypatch.setattr(owner, name, w)


@pytest.fixture(autouse=True)
def fixed(monkeypatch):
    """The verify's generic paths; a fresh round-time table that takes no
    samples (no decision depends on the wall clock)."""
    no_metal_verify(monkeypatch)
    monkeypatch.setattr(worker, "CLOCK", worker.Clock())
    monkeypatch.setattr(worker.Clock, "add", lambda self, kind, ms: None)


class Script(worker.Chooser):
    """The Chooser with its timing taken out (picks and shadow drafts by
    round count: the same decisions in every run)."""

    def pick(self, force=None, now=None):
        self.rounds += 1
        self.r, self.probe = None, False
        self.cur = ("mtp", "mtp", "dflash")[self.rounds % 3]
        return self.cur

    def shadows(self):
        return ["dflash"] if self.cur == "mtp" and self.rounds % 2 else []


def one(dense, monkeypatch, how, level, overlap=True):
    """-> (tokens, the counted calls, req.prof) of one request."""
    ids = prompts([25], 41)[0]
    ref = run(scheduler(dense, max_batch=1), [request(ids, 40, "a")])["a"]
    full = ids + ref[0]
    log = []
    with monkeypatch.context() as m:
        m.setattr(worker, "SPEC_PROF", level)
        m.setattr(worker, "ROUND_OVERLAP", overlap)
        sched = scheduler(dense, max_batch=1)
        if how != "plain":
            m.setattr(worker, "MTP_K", 0)
            m.setattr(worker.MTPDraft, "BURSTS", (1, 2, 4, 8))
            sched.eng.mtp = Oracle(head(dense), dense, full, script=(3, 0, 4, 1, 0, 0, 2))
        if how == "hybrid":
            m.setattr(worker, "Chooser", Script)
            emitted = []

            class Rec(FakeDraft):
                def draft_block(self, b, hidden, cache, bs, sampler, dtype):
                    return mx.array([(ref[0][len(emitted):] + [0] * bs)[:bs - 1]], dtype=dtype)
            sched.eng.draft = Rec(ref[0])
            stop = sched.eng.stopping
            sched.eng.stopping = lambda t: emitted.append(t) or stop(t)
        x = request(ids, 40, "a")
        counted(m, log)
        out = run(sched, [x])["a"]
    assert out == ref
    return out, log, x.prof


@pytest.mark.parametrize("how", ["mtp", "hybrid", "plain"])
def test_profile_adds_no_sync(dense, monkeypatch, how):
    _, off, none = one(dense, monkeypatch, how, 0)
    _, on, prof = one(dense, monkeypatch, how, 1)
    assert none is None and prof is not None
    assert on == off and "item" in on
    keys = {r[0] for r in prof.raw()}
    want = {"mtp": {"mtp0", "mtp1", "mtp4"}, "hybrid": {"mtp0", "dflash"}, "plain": {"loop"}}[how]
    assert want <= keys, keys
    for r in prof.recs:                          # the spans are the wall time
        s = sum(r[k] for k in ("pre", "build", "wait", "other", "post", "between"))
        assert s == pytest.approx(r["wall"], rel=1e-9, abs=1e-9)
        assert r["first"] is None or r["sync"] is None or r["first"] <= r["sync"]
    agg = prof.summary()["keys"]
    assert all(v["rounds"] and v["wall"]["n"] == v["rounds"] for v in agg.values())
    if how == "hybrid":                          # shadow drafts read in the head's bursts
        assert any("shadow" in laps for *_, laps in prof.raw())
    if how != "plain":                           # the head's own verify: the walk is its sync
        assert all(laps.get("walk", 0.0) > 0.0 for k, *_, laps in prof.raw() if k[:3] == "mtp" and k != "mtp0")


def test_probe_level_adds_only_its_syncs(dense, monkeypatch):
    _, off, _ = one(dense, monkeypatch, "mtp", 0)
    _, on, prof = one(dense, monkeypatch, "mtp", 2)
    probes = [r for r in prof.recs + [prof.prev, prof.r] if r is not None and r["probe"]]
    assert probes and all("tail" in r["laps"] for r in probes)
    assert on.count("synchronize") == off.count("synchronize") + len(probes)
    assert [x for x in on if x != "synchronize"] == [x for x in off if x != "synchronize"]
    agg = prof.aggregate(prof.raw(skip=0))           # (probes: their own statistic only)
    assert sum(v["tail"]["n"] for v in agg.values() if v["tail"]) == sum(r["probe"] for r in prof.recs)


def test_cprof_level_adds_no_sync(dense, monkeypatch):
    """QWEN_SPEC_PROF=3: Python's profiler over every CPROF-th verify build
    (those records probes: out of the spans' statistics); no MLX call."""
    monkeypatch.setattr(worker.RoundProf, "CPROF", 2)
    _, off, _ = one(dense, monkeypatch, "mtp", 0)
    _, on, prof = one(dense, monkeypatch, "mtp", 3)
    assert [x for x in on if x != "synchronize"] == [x for x in off if x != "synchronize"]
    cp = prof.summary()["cprof"]
    assert cp["builds"] >= 1 and cp["top"]
    assert all(len(row) == 4 and row[2] >= 0.0 for row in cp["top"])


def test_queued_walk_is_the_library_walk(monkeypatch):
    """The walk of a round with ROUND_OVERLAP: mlx-vlm's on every block,
    reading the arrays themselves (tolist of drafts and targets: no new op)."""
    rnd = random.Random(5)
    for _ in range(200):
        n = rnd.randint(1, 6)
        d = mx.array([[rnd.randint(0, 2) for _ in range(n)]], dtype=mx.int32)
        t = mx.array([[rnd.randint(0, 2) for _ in range(n + 1)]], dtype=mx.int32)
        budget = rnd.randint(1, n + 2)
        assert worker._queued_walk(d, t, budget) == _speculative_walk(d, t, budget)
    seen, tolist = [], mx.array.tolist
    monkeypatch.setattr(mx.array, "tolist", lambda a: seen.append(a) or tolist(a))
    assert worker._queued_walk(d, t, 9) == _speculative_walk(d, t, 9)
    assert seen[0] is d and seen[1] is t
    assert worker._queued_row(mx.array([4, 5])) == [4, 5]


@pytest.mark.parametrize("how", ["mtp", "hybrid"])
def test_round_overlap_only_splits_the_eval(dense, monkeypatch, how):
    """ROUND_OVERLAP on / off: the same answer (one() checks it against the
    plain loop), the same host reads and evals in the same order; on, the
    head's verify is a queued eval of its own in each of its rounds."""
    nv, verify = [0], worker.MTPDraft.verify

    def counted_verify(self, h, targets):
        nv[0] += 1
        return verify(self, h, targets)
    monkeypatch.setattr(worker.MTPDraft, "verify", counted_verify)
    _, seq, _ = one(dense, monkeypatch, how, 0, overlap=False)
    own, nv[0] = nv[0], 0
    _, ovl, _ = one(dense, monkeypatch, how, 0, overlap=True)
    assert own and nv[0] == own
    assert ovl.count("async_eval") == seq.count("async_eval") + own
    assert [x for x in ovl if x != "async_eval"] == [x for x in seq if x != "async_eval"]
