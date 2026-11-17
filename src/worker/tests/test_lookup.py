"""Prompt lookup decoding (QWEN_LOOKUP, lookup.py + _Scheduler._spec) on the
tiny random model (CPU).

The index (_Lookup) and the drafter's held context (_DraftCtx) as units; then
the scheduler with a stand-in lookup that proposes the known greedy answer
(a wrong draft now and then: accepted prefix + rollback) on a schedule, so
every kind of round runs: lookup rounds between plain steps without a
drafter ("1", eng.draft None), lookup rounds between the drafter's blocks
("1") and the splice ("splice"). The answers must be the plain loop's; the
drafter's stand-in checks that its context always ends right before its
anchor (every position since its last block was handed over)."""
import logging
import types

import mlx.core as mx
import pytest

import worker
from conftest import engine, prompts, request, run, scheduler
from test_scheduler import dense_or_fused, no_metal_verify  # noqa: F401 (a fixture)


# --- units ------------------------------------------------------------------------
def test_lookup_latest_longest_match():
    lk = worker._Lookup([1, 2, 3, 4, 5, 9, 2, 3, 4, 6, 7, 8, 2, 3, 4], n_min=3, n_max=8)
    # suffix 2 3 4: occurrences followed by 5 and 6; equal length 3 -> the latest
    assert lk.propose(3) == (3, [6, 7, 8])
    assert lk.propose(5) == (3, [6, 7, 8, 2, 3])
    lk.add(6)                                    # 3 4 6 -> only the second occurrence
    assert lk.propose(2) == (4, [7, 8])
    # the longer match wins over the latest
    lk = worker._Lookup([0, 1, 2, 3, 4, 50, 9, 2, 3, 4, 60, 8, 1, 2, 3, 4], n_min=3, n_max=8)
    assert lk.propose(4) == (4, [50, 9, 2, 3])
    # no match / too short / k = 0
    assert worker._Lookup([1, 2, 3, 4, 5]).propose(4) == (0, [])
    assert worker._Lookup([1, 2]).propose(4) == (0, [])
    assert worker._Lookup([1, 2, 3, 1, 2, 3]).propose(0) == (0, [])


def test_lookup_never_matches_its_own_suffix():
    """An n-gram enters the index only when its next token is known: a
    repeated token run proposes the run, never the current end itself."""
    lk = worker._Lookup([5, 5, 5], n_min=3)
    assert lk.propose(4) == (0, [])
    lk.add(5)
    assert lk.propose(4) == (3, [5, 5, 5, 5])    # the run goes on (period 1)
    lk.extend([5, 5])
    assert lk.propose(3) == (5, [5, 5, 5])
    lk = worker._Lookup([9, 1, 2, 3, 1, 2, 3], n_min=3)
    assert lk.propose(5) == (3, [1, 2, 3, 1, 2])  # a cycle: period 3


def test_lookup_incremental_equals_bulk():
    ids = [int(x) for x in mx.random.randint(0, 6, (400,), key=mx.random.key(3)).tolist()]
    a, b = worker._Lookup(ids[:100], 2, 6), worker._Lookup(ids, 2, 6)
    for t in ids[100:]:
        a.add(t)
    assert a.idx == b.idx and a.toks == b.toks
    for k in (1, 5, 15):
        assert a.propose(k) == b.propose(k)


def test_draft_ctx_trims_like_the_drafter():
    """Over the window: the last `win` positions, every rotating cache's
    offset moved on by the rest (the drafter's own `cache.offset += skip`);
    with a non-rotating cache everything is kept."""
    from mlx_vlm.models.cache import KVCache, RotatingKVCache
    chunks = [mx.full((1, n, 2), i, dtype=mx.float16) for i, n in enumerate([5, 3, 4, 2])]
    rot = [RotatingKVCache(max_size=6, keep=0) for _ in range(2)]
    d = worker._DraftCtx(6, rot)
    for c in chunks:
        d.add(c)
    assert bool(d) and len(d.chunks) <= 3          # the oldest dropped while the rest covers 6
    h = d.take(rot)
    assert h.shape[1] == 6 and h[0, :, 0].tolist() == [2, 2, 2, 2, 3, 3]
    assert [c.offset for c in rot] == [8, 8] and not d
    d = worker._DraftCtx(6, [RotatingKVCache(max_size=6, keep=0), KVCache()])
    for c in chunks:
        d.add(c)
    assert d.take(None).shape[1] == 14
    d = worker._DraftCtx(6, None)                 # (a stand-in drafter: no caches)
    d.add(chunks[0])
    assert d.take(None) is chunks[0]
    d = worker._DraftCtx(6, None, on=False)       # no drafter takes it: nothing held
    for c in chunks:
        d.add(c)
    assert not d and not d.chunks


# --- the scheduler ------------------------------------------------------------------
class FakeLookup:
    """_Lookup stand-in: proposes the known answer's next k tokens in every
    `every`-th stretch of 4 positions (else no match), the last one wrong at every other
    proposal. Checks that it was given the prompt + the emitted tokens."""
    answer, every, made = None, 2, None

    def __init__(self, ids, n_min=3, n_max=8, cand=8):
        self.prompt, self.toks, self.q, self.p = list(ids), list(ids), 0, 0
        FakeLookup.made = self

    def add(self, t):
        self.toks.append(int(t))

    def propose(self, k):
        at = len(self.toks) - len(self.prompt)
        assert self.toks[len(self.prompt):] == self.answer[:at], "lookup lost an emitted token"
        self.q += 1
        if (at // 4) % self.every:              # matches in stretches of 4 positions
            return 0, []
        d = list(self.answer[at:at + k])
        self.p += 1
        if self.p % 2 and d:
            d[-1] = (d[-1] + 1) % 400
        return 8, d


class CtxDraft:
    """DFlash stand-in: proposes the answer from the position its context
    says it is at (the prompt's positions + every position the target cache
    took since: sum of the contexts it was given), and checks that its anchor
    is the answer's token there — so a context handed over short or twice
    fails the request."""

    def __init__(self, answer, n_prompt, block=4, window=4096):
        self.answer, self.n_prompt, self.seen = answer, n_prompt, 0
        self.config = types.SimpleNamespace(target_layer_ids=[0, 3], sliding_window=window,
                                            block_size=block, runtime_block_size=None)
        self.prefer_requested_block_size = True
        self.accept_lens, self.draft_lens = [], []
        self.calls = 0

    def reset(self, model):
        self.seen = 0
        return None

    def draft_block(self, b, hidden, cache, bs, sampler, dtype):
        self.seen += hidden.shape[1]
        self.calls += 1
        at = self.seen - self.n_prompt + 1          # tokens emitted (b the last)
        assert b == self.answer[at - 1], f"drafter context off: {at=} {b=}"
        d = (self.answer[at:at + bs - 1] + [0] * bs)[:bs - 1]
        if self.calls % 3 == 0:
            d[-1] = (d[-1] + 1) % 400
        return mx.array([d], dtype=dtype)


@pytest.fixture
def look(monkeypatch):
    no_metal_verify(monkeypatch)
    monkeypatch.setattr(worker, "LOOKUP", "1")
    monkeypatch.setattr(worker, "_Lookup", FakeLookup)
    FakeLookup.every, FakeLookup.made, FakeLookup.answer = 2, None, None
    monkeypatch.setattr(worker, "LOOKUP_K", 3)  # short rounds: every kind of round, often
    return monkeypatch


def answer_of(dense, p, n, eng=None, rid="a"):
    """The plain loop's answer (QWEN_LOOKUP off), also the stand-in's."""
    was = worker.LOOKUP
    worker.LOOKUP = "0"
    try:
        out = run(scheduler(dense, max_batch=1, eng=eng), [request(p, n, rid)])[rid]
    finally:
        worker.LOOKUP = was
    FakeLookup.answer = out[0]
    return out


def lstats(caplog):
    """(lookup rounds, plain steps) from the request's end log line."""
    import re
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("dflash:")]
    m = re.search(r"lookup (\d+) rounds.*?(?:, (\d+) plain steps)?$", lines[-1]) if lines else None
    return (int(m.group(1)), int(m.group(2) or 0)) if m else (0, 0)


@pytest.mark.parametrize("every,k", [(1, 3), (2, 3), (3, 3), (2, 15), (2, 2)])
def test_lookup_without_drafter_is_plain_greedy(dense_or_fused, look, caplog, every, k):
    """No drafter: lookup rounds where the stand-in matches, plain step-ahead
    bursts elsewhere; the plain loop's tokens. dense_fused: 8-bit weights,
    the fused GDN in-projection."""
    dense = dense_or_fused
    caplog.set_level(logging.INFO)
    look.setattr(worker, "LOOKUP_K", k)
    FakeLookup.every = every
    p = prompts([25], 31)[0]
    plain = answer_of(dense, p, 40)
    sched = scheduler(dense, max_batch=1)
    assert sched.eng.draft is None
    got = run(sched, [request(p, 40, "a")])["a"]
    assert got == plain
    rounds, steps = lstats(caplog)
    assert rounds >= 2 and (steps > 0 or every == 1 or k == 15)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 17])
def test_lookup_without_drafter_short_answers(dense, look, n):
    p = prompts([20], 32)[0]
    plain = answer_of(dense, p, n)
    assert run(scheduler(dense, max_batch=1), [request(p, n, "a")])["a"] == plain


def test_lookup_stop_inside_an_accepted_block(dense, look):
    """The stop token among accepted drafts ends the answer there, as plain."""
    p = prompts([25], 33)[0]
    free = answer_of(dense, p, 40)[0]
    stop = free[13]
    eng = engine(dense, eos=(3, stop))
    plain = answer_of(dense, p, 40, eng=engine(dense, eos=(3, stop)))
    assert plain[2] == ("done", "stop") and len(plain[0]) <= 13
    FakeLookup.answer, FakeLookup.every = free, 1
    assert run(scheduler(dense, max_batch=1, eng=eng), [request(p, 40, "a")])["a"] == plain


def test_lookup_without_drafter_demoted(dense, look, caplog):
    """A second request demotes the lookup row to a batch row; both answers
    are the plain loop's."""
    caplog.set_level(logging.INFO)
    ps = prompts([25, 12], 34)
    other = answer_of(dense, ps[1], 20, rid="b")
    plain = answer_of(dense, ps[0], 30)
    sched = scheduler(dense, max_batch=4)
    a, b = request(ps[0], 30, "a"), request(ps[1], 20, "b")
    assert run(sched, [a, b], arrive={4: [b]}) == {"a": plain, "b": other}
    assert any("demoted" in r.getMessage() for r in caplog.records)




@pytest.mark.parametrize("mode", ["1", "splice"])
@pytest.mark.parametrize("every", [1, 2, 3])
def test_lookup_with_drafter(dense_or_fused, look, caplog, mode, every):
    """Lookup rounds and the drafter's blocks interleaved ("1") / the
    drafter's block every round, replaced by the agreeing continuation
    ("splice"): the plain answer, and the drafter always sees its anchor's
    whole context (CtxDraft's check)."""
    dense = dense_or_fused
    caplog.set_level(logging.INFO)
    look.setattr(worker, "LOOKUP", mode)
    FakeLookup.every = every
    p = prompts([25], 35)[0]
    plain = answer_of(dense, p, 40)
    sched = scheduler(dense, max_batch=1)
    d = sched.eng.draft = CtxDraft(plain[0], len(p))
    assert run(sched, [request(p, 40, "a")])["a"] == plain
    rounds, steps = lstats(caplog)
    assert rounds >= 2 and steps == 0
    if mode == "1":
        assert len(d.accept_lens) >= (1 if every > 1 else 0)
    else:
        assert d.calls >= rounds


def test_lookup_off_is_untouched(dense, look, caplog):
    """QWEN_LOOKUP=0: no index, the drafter alone."""
    caplog.set_level(logging.INFO)
    look.setattr(worker, "LOOKUP", "0")
    p = prompts([25], 36)[0]
    plain = answer_of(dense, p, 30)
    sched = scheduler(dense, max_batch=1)
    sched.eng.draft = CtxDraft(plain[0], len(p))
    assert run(sched, [request(p, 30, "a")])["a"] == plain
    assert FakeLookup.made is None and lstats(caplog) == (0, 0)
    # no drafter, lookup off: the plain loop (not _spec)
    assert run(scheduler(dense, max_batch=1), [request(p, 30, "a")])["a"] == plain
    assert FakeLookup.made is None


def test_real_index_on_a_repetitive_answer(dense, monkeypatch, caplog):
    """The real _Lookup: the tiny model's greedy answers fall into cycles,
    which the index proposes; the answer stays the plain loop's."""
    caplog.set_level(logging.INFO)
    no_metal_verify(monkeypatch)
    p = prompts([25], 30)[0]
    plain = run(scheduler(dense, max_batch=1), [request(p, 60, "a")])["a"]
    monkeypatch.setattr(worker, "LOOKUP", "1")
    monkeypatch.setattr(worker, "LOOKUP_MIN", 3)
    assert run(scheduler(dense, max_batch=1), [request(p, 60, "a")])["a"] == plain
    rounds, steps = lstats(caplog)
    assert rounds >= 1 and steps > 0


def test_lookup_without_drafter_quantized_kv_is_plain(dense, look, caplog):
    """Quantized KV: no lookup rounds (a round's rollback needs fp16 KV)."""
    caplog.set_level(logging.INFO)
    p = prompts([25], 37)[0]
    sched = scheduler(dense, max_batch=1, kv_bits=8, kv_start=0)
    run(sched, [request(p, 20, "a")])
    assert FakeLookup.made is None


@pytest.mark.parametrize("k0", [2, 3])
def test_lookup_adaptive_k(dense, look, caplog, k0):
    """QWEN_LOOKUP_K0: rounds start at k0 drafts, grow after a full accept,
    restart at k0 after a miss; answers unchanged."""
    caplog.set_level(logging.INFO)
    look.setattr(worker, "LOOKUP_K", 15)
    look.setattr(worker, "LOOKUP_K0", k0)
    FakeLookup.every = 1
    sizes = []
    real = FakeLookup.propose

    def spy(self, k):
        sizes.append(k)
        return real(self, k)
    look.setattr(FakeLookup, "propose", spy)
    p = prompts([25], 38)[0]
    plain = answer_of(dense, p, 60)
    assert run(scheduler(dense, max_batch=1), [request(p, 60, "a")])["a"] == plain
    assert sizes[0] == k0 and max(sizes) > k0


def test_gate():
    g = worker._Gate(3, 15, 3, 32, True)
    assert g.size(100) == 3 and g.size(2) == 2
    assert g.ok(3, [1, 2]) and not g.ok(2, [1, 2]) and not g.ok(9, [1])
    g.after(3, 3)                                # full: 2k + 1
    assert (g.k, g.need) == (7, 3)
    g.after(7, 7)
    g.after(15, 15)
    assert g.k == 15
    g.after(1, 15)                               # a miss: k0, need x2
    assert (g.k, g.need) == (3, 6)
    g.after(2, 3)                                # kept >= half: need stays
    assert g.need == 6
    for _ in range(5):
        g.after(0, 3)
    assert g.need == 32                          # capped at n_max
    g.after(3, 3)
    assert g.need == 3
    g = worker._Gate(0, 15, 3, 8, False)         # fixed K, no backoff
    g.after(0, 15)
    assert (g.k, g.need) == (15, 3)


def test_lookup_draft_mode(dense, look, caplog):
    """QWEN_LOOKUP=draft: lookup rounds for the drafter's rows only; without a
    drafter the plain loop (no index, not _spec)."""
    caplog.set_level(logging.INFO)
    look.setattr(worker, "LOOKUP", "draft")
    p = prompts([25], 39)[0]
    plain = answer_of(dense, p, 30)
    FakeLookup.made = None
    assert run(scheduler(dense, max_batch=1), [request(p, 30, "a")])["a"] == plain
    assert FakeLookup.made is None
    sched = scheduler(dense, max_batch=1)
    sched.eng.draft = CtxDraft(plain[0], len(p))
    assert run(sched, [request(p, 30, "a")])["a"] == plain
    assert FakeLookup.made is not None and lstats(caplog)[0] >= 1
