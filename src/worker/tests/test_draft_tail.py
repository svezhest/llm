"""DFlash rows and the prompt cache (2026-11-11, docs/perf/2026-11-10/
dflash2_27b.md) on the tiny random model (CPU).

QWEN_DRAFT_TAIL: the states a DFlash row stores (checkpoints, the end, a
cancelled prefill's prefix, a reply's end) carry the drafter's context there,
the target's hidden states of the last N positions; a DFlash row resuming
from one starts its drafter with them + what it prefills itself (the same
arrays a cold run gives it for those positions). Answers and the cached bits
are unchanged; the context rides through absorb / chain and the disk tier.

Reply KV (QWEN_PROMPT_CACHE_REPLY) for DFlash rows: at stop / length the
entry is extended by the tokens the target cache took in the rounds, as a
batch row's; also for a row demoted to the batch mid-answer."""
import logging

import mlx.core as mx
import pytest

import worker
import test_reply_kv as rk
from conftest import IM_START, Cold, check_cold, engine, prompts, request, run, scheduler
from test_partial_store import cancel_after, prompt, slab_ends
from test_prompt_cache import GEN, cold, conversation, small  # noqa: F401
from test_scheduler import FakeDraft, no_metal_verify


class Rec(FakeDraft):
    """FakeDraft that keeps the context of every draft_block call; a window
    that holds the whole prompt."""

    def __init__(self, answer, window=4096):
        super().__init__(answer)
        self.config.sliding_window = window
        self.ctx = []

    def draft_block(self, b, hidden, cache, bs, sampler, dtype):
        self.ctx.append(hidden)
        return super().draft_block(b, hidden, cache, bs, sampler, dtype)


def spec(sched, ids, answer, n, rid):
    """One DFlash request -> (its output, the drafter's first context, the request)."""
    d = sched.eng.draft = Rec(answer[0])
    x = request(ids, n, rid)
    out = run(sched, [x])[rid]
    assert d.accept_lens, "the request did not take the speculative path"
    return out, d.ctx[0], x


def tail_of(e, p):
    return worker._state_tail(e.end if p == len(e.ids) else e.ckpts[p], e.layout)


def eq(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and mx.array_equal(a, b).item()


@pytest.fixture
def tail(small, monkeypatch, request):
    no_metal_verify(monkeypatch)
    n = getattr(request, "param", 16)
    monkeypatch.setattr(worker, "DRAFT_TAIL", n)
    monkeypatch.setattr(worker, "DRAFT_TAIL_END", n)        # one length: the checks below
    return n


@pytest.mark.parametrize("tail", [5, 8, 16, 1000], indirect=True)
def test_states_carry_the_last_positions_hidden_states(dense, tail):
    """Every state of a DFlash row's entry holds the hidden states of the
    `tail` positions before it (fewer near the start), in a buffer of its
    own; the cached bits are the plain prefill's."""
    ids = prompt(400, user=70)
    plain = cold(dense, ids, 12, "x")
    sched = scheduler(dense, max_batch=1)
    out, full, _ = spec(sched, ids, plain, 12, "x")
    assert out == plain and full.shape[1] == len(ids)      # every prompt position
    (e,) = sched.eng._pcache.entries
    assert len(e.ckpts) >= 3
    for p in [*e.ckpts, len(e.ids)]:
        t = tail_of(e, p)
        assert eq(t, full[:, max(0, p - tail):p]), p
    assert check_cold(sched.eng._pcache, Cold(dense))
    # counted in the entry's bytes
    assert e.nbytes == sum(a.nbytes for a in {id(a): a for a in e.arrays}.values())
    assert sum(tail_of(e, p).nbytes for p in [*e.ckpts, len(e.ids)]) <= e.state_bytes()


def test_off_stores_no_context(dense, tail, monkeypatch):
    monkeypatch.setattr(worker, "DRAFT_TAIL", 0)
    monkeypatch.setattr(worker, "DRAFT_TAIL_END", 0)
    ids = prompt(401, user=70)
    plain = cold(dense, ids, 12, "x")
    sched = scheduler(dense, max_batch=1)
    assert spec(sched, ids, plain, 12, "x")[0] == plain
    (e,) = sched.eng._pcache.entries
    assert all(tail_of(e, p) is None for p in [*e.ckpts, len(e.ids)])
    out, ctx, x = spec(sched, ids, plain, 12, "y")          # the exact repeat: one position
    assert out == plain and x.reused == len(ids) - 1 and ctx.shape[1] == 1


@pytest.mark.parametrize("tail", [5, 16], indirect=True)
def test_exact_repeat_starts_the_drafter_with_the_stored_context(dense, tail):
    """A hit at the entry's end prefills nothing: the drafter gets the end
    state's context + the last prompt token's hidden state, the arrays the
    cold run had for those positions; the entry stored again keeps it."""
    ids = prompt(402, user=70)
    plain = cold(dense, ids, 12, "x")
    sched = scheduler(dense, max_batch=1)
    _, full, _ = spec(sched, ids, plain, 12, "x")
    out, ctx, x = spec(sched, ids, plain, 12, "y")
    assert out == plain and x.reused == len(ids) - 1
    assert eq(ctx, full[:, -(tail + 1):])
    (e,) = sched.eng._pcache.entries
    assert eq(tail_of(e, len(e.ids)), full[:, -(tail + 1):-1])
    assert check_cold(sched.eng._pcache, Cold(dense))


def test_a_plain_rows_state_has_no_context(dense, tail):
    """An entry stored by a plain row (no drafter then): a DFlash row
    resuming from it starts with what it prefills, as before; the entry it
    stores has the context from there on."""
    ids = prompt(403, user=70)
    plain = cold(dense, ids, 12, "x")
    sched = scheduler(dense, max_batch=1)
    assert run(sched, [request(ids, 12, "p")])["p"] == plain
    out, ctx, x = spec(sched, ids, plain, 12, "y")
    assert out == plain and x.reused == len(ids) - 1 and ctx.shape[1] == 1
    (e,) = sched.eng._pcache.entries
    assert tail_of(e, len(e.ids)) is None                   # nothing was prefilled: none to keep
    assert all(tail_of(e, p) is None for p in e.ckpts)


def test_next_turn_resumes_with_the_checkpoints_context(dense, tail):
    """Turn 2 resumes at a checkpoint of turn 1's entry (a message start):
    the drafter starts with that state's context + the positions prefilled,
    = the cold run's last positions; the chain entry it stores (turn 1
    absorbed) carries the context at every state, and turn 3 resumes from
    it the same way."""
    a = conversation(3, 410, system=30, user=lambda i: 9, answer=lambda i: 6)
    sched = scheduler(dense, max_batch=1)
    for k, ids in enumerate(a):
        plain = cold(dense, ids, 8, "x")
        ref = scheduler(dense, max_batch=1)                 # the cold run's context
        _, full, _ = spec(ref, ids, plain, 8, "x")
        out, ctx, x = spec(sched, ids, plain, 8, f"t{k}")
        assert out == plain
        if k:
            assert 0 < x.reused < len(ids) - 1
            assert eq(ctx, full[:, x.reused - tail:])
        (e,) = sched.eng._pcache.entries
        for p in [*e.ckpts, len(e.ids)]:
            assert eq(tail_of(e, p), full[:, max(0, p - tail):p]), (k, p)
    assert e.chained
    assert check_cold(sched.eng._pcache, Cold(dense))


def test_cancelled_dflash_prefill_stores_its_prefix(dense, tail):
    """A DFlash run cancelled mid-prefill stores the prefix it holds (as a
    plain prefill: prompt[:pos] at the last completed slab end), with the
    drafter's context at pos; the retry resumes there, answers as cold and
    starts its drafter with the cold run's positions."""
    ids = prompt(404, user=70)
    plain = cold(dense, ids, 12, "x")
    ref = scheduler(dense, max_batch=1)
    _, full, _ = spec(ref, ids, plain, 12, "x")
    sched = scheduler(dense, max_batch=1)
    sched.eng.draft = Rec(plain[0])
    x = request(ids, 12, "x")
    assert run(sched, [x], on_tick=cancel_after(x, 3))["x"] == ([], [], ("done", "cancelled"))
    pos = slab_ends(ids)[2]
    (e,) = sched.eng._pcache.entries
    assert e.partial and e.ids == ids[:pos]
    assert eq(tail_of(e, pos), full[:, pos - tail:pos])
    assert check_cold(sched.eng._pcache, Cold(dense))
    out, ctx, y = spec(sched, ids, plain, 12, "y")
    assert out == plain and y.reused == pos
    assert eq(ctx, full[:, pos - tail:])
    (e,) = sched.eng._pcache.entries
    assert not e.partial and e.ids == ids[:-1]
    assert check_cold(sched.eng._pcache, Cold(dense))


def test_context_survives_the_disk_tier(dense, tail, tmp_path, monkeypatch):
    """An entry evicted to disk and loaded back for the next turn: its
    states still carry the drafter's context (a state's blob holds every
    item of it)."""
    from test_pcdisk import BIG, TINY, tier
    a = conversation(2, 420, system=40, user=lambda i: 9, answer=lambda i: 10)
    b = [IM_START] + prompts([50], 427)[0] + GEN
    plain = [cold(dense, ids, 6, "x") for ids in a]
    ref = scheduler(dense, max_batch=1)
    _, full, _ = spec(ref, a[1], plain[1], 6, "x")
    sched = scheduler(dense, max_batch=1)
    pc = sched.eng._pcache
    disk = pc.disk = tier(tmp_path)
    monkeypatch.setattr(worker, "PROMPT_CACHE_GB", TINY)
    spec(sched, a[0], plain[0], 6, "a0")
    assert disk.drain()
    sched.eng.draft = None
    run(sched, [request(b, 4, "b")])                        # evicts a[0]'s entry
    assert disk.drain() and [e.ids for e in pc.entries] == [b[:-1]]
    monkeypatch.setattr(worker, "PROMPT_CACHE_GB", BIG)
    out, ctx, x = spec(sched, a[1], plain[1], 6, "a1")
    assert out == plain[1] and x.reused >= worker.PREFILL_STEP
    assert disk.stats()["loads"] == 1
    assert eq(ctx, full[:, x.reused - tail:])


# --- reply KV for DFlash rows ---------------------------------------------------
@pytest.fixture
def reply(tail, monkeypatch):
    monkeypatch.setattr(worker, "PROMPT_CACHE_REPLY", True)
    return tail


def near(a, b, tol=1e-5):
    """Decode-shaped vs prefill-slab hidden states: equal within rounding."""
    assert a.shape == b.shape and a.dtype == b.dtype, (a.shape, b.shape)
    d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()
    return d <= tol * max(mx.abs(b.astype(mx.float32)).max().item(), 1e-3)


def cold_ctx(monkeypatch, lm, ids, answer, n):
    """The drafter's whole context in a cold DFlash run of `ids` (reply KV off)."""
    with monkeypatch.context() as m:
        m.setattr(worker, "PROMPT_CACHE_REPLY", False)
        return spec(scheduler(lm, max_batch=1), ids, answer, n, "ref")[1]


def test_dflash_reply_extends_the_entry_and_the_next_turn_resumes_after_it(dense, reply, monkeypatch,
                                                                           caplog):
    """A DFlash row ending with "length": its entry becomes prompt + the
    tokens its cache took (every generated token but, usually, the last: the
    target's own token of the last round was never fed), its end state
    carries the last positions' hidden states; the next turn resumes right
    after the reply, answers as a cold worker and starts its drafter with
    that context + what it prefilled."""
    caplog.set_level(logging.INFO, logger="qwen.worker")
    sched = scheduler(dense, max_batch=1)
    pc = sched.eng._pcache
    p, prev = rk.first(430), None
    for t in range(3):
        plain = rk.off_run(monkeypatch, dense, p, 9 + t, "x")[0]
        full = cold_ctx(monkeypatch, dense, p, plain, 9 + t)
        out, ctx, x = spec(sched, p, plain, 9 + t, f"t{t}")
        assert out == plain and out[2] == ("done", "length")
        (e,) = pc.entries
        k = len(e.ids) - len(p)
        assert e.ids == (p + out[0])[:len(e.ids)] and len(out[0]) - 1 <= k <= len(out[0])
        assert e.reply == len(p) - 1 and e.end is not None and len(p) - 1 not in e.ckpts
        assert rk.close(pc.restore(e, len(e.ids)), rk.cold_cache(monkeypatch, dense, e.ids), dense)
        if prev is not None:
            assert x.reused == prev                          # right after the previous reply
            assert ctx.shape == full[:, prev - reply:].shape and near(ctx, full[:, prev - reply:])
        # the end state's context: the positions before the reply's end, as
        # the next turn's cold prefill computes them
        q = rk.nxt(p, out[0], 440 + t)
        nfull = cold_ctx(monkeypatch, dense, q, rk.off_run(monkeypatch, dense, q, 4, "x")[0], 4)
        L = len(e.ids)
        assert near(tail_of(e, L), nfull[:, L - reply:L])
        assert any(f"id=t{t} prompt cache reply kept" in m for m in caplog.messages)
        p, prev = q, L


def stop_case(lm, monkeypatch, k):
    """A prompt whose greedy answer has a token at index k that occurs
    nowhere before it (and, for the junk case, after it): the stop token."""
    for seed in range(450, 480):
        p = rk.first(seed)
        probe = rk.off_run(monkeypatch, lm, p, 12, "probe")[0][0]
        if probe.count(probe[k]) == 1 and 3 not in probe:
            return p, probe
    raise AssertionError("no prompt found")


@pytest.mark.parametrize("k,kept", [(2, None), (3, 4), (4, 4), (5, None), (6, 7), (7, 7)])
def test_dflash_reply_at_a_stop_token(dense, reply, monkeypatch, k, kept):
    """FakeDraft: rounds of 3 drafts, the last one wrong every other round ->
    generated index 4 and 7 are the target's own tokens, 1-3 and 5-6 accepted
    drafts. The stop token as the target's own token: never fed, the entry
    is prompt + the answer. As the last accepted draft: fed, in the entry
    (the template puts it back: the next turn resumes after it). As an
    accepted draft with accepted drafts after it: the cache ran past the
    stop, nothing is stored and the prompt entry stays."""
    p, probe = stop_case(dense, monkeypatch, k)
    stop = probe[k]
    eng = engine(dense, eos=(stop,))
    sched = scheduler(dense, max_batch=1, eng=eng)
    pc = eng._pcache
    d = eng.draft = Rec(probe)
    x = request(p, 12, "a")
    out = run(sched, [x])["a"]
    assert out[0] == probe[:k] and out[2] == ("done", "stop") and d.accept_lens
    (e,) = pc.entries
    if kept is None:
        assert e.ids == p[:-1] and e.reply is None and e.end is not None
    else:
        assert e.ids == p + probe[:kept] and e.reply == len(p) - 1
        assert tail_of(e, len(e.ids)) is not None
    q = rk.nxt(p, probe[:k] + [stop], 481, eos=False)
    ref = run(scheduler(dense, max_batch=1, eng=engine(dense, eos=(stop,))), [request(q, 6, "b")])["b"]
    eng.draft = None
    y = request(q, 6, "b")
    assert run(sched, [y])["b"] == ref
    if kept == k + 1:
        assert y.reused == len(p) + kept                    # after the stop token


def test_demoted_dflash_row_keeps_its_reply(dense, reply, monkeypatch):
    """A DFlash row demoted to a batch row mid-answer: row.fed holds its
    rounds' tokens too, so its reply is stored at its end like any batch
    row's (prompt + completion), within rounding of a cold prefill."""
    pa, pb_ = rk.first(490), rk.first(495)
    plain = rk.off_run(monkeypatch, dense, pa, 20, "a")[0]
    other = rk.off_run(monkeypatch, dense, pb_, 6, "b")[0]
    sched = scheduler(dense, max_batch=4)
    pc = sched.eng._pcache
    d = sched.eng.draft = Rec(plain[0])
    a, b = request(pa, 20, "a"), request(pb_, 6, "b")
    got = run(sched, [a, b], arrive={10: [b]})           # after a's prefill (~7 slabs) + rounds
    assert got == {"a": plain, "b": other}
    assert 0 < len(d.accept_lens) < 5                       # rounds, then a batch row
    (e,) = [e for e in pc.entries if e.ids[:len(pa)] == pa]
    assert e.ids == pa + plain[0] and e.reply == len(pa) - 1
    assert rk.close(pc.restore(e, len(e.ids)), rk.cold_cache(monkeypatch, dense, e.ids), dense)


def test_dflash_reasoning_reply_is_not_stored(dense, reply, caplog):
    """Thinking on (the prompt pre-opens <think>): as for a batch row, the
    reply is not stored and the prompt entry keeps its end state (and its
    drafter context: a regenerate's full hit)."""
    caplog.set_level(logging.INFO, logger="qwen.worker")
    ids = prompt(497, user=40)                              # ends with <think>\n
    sched = scheduler(dense, max_batch=1)
    sched.eng._think = [13, 15]                             # test_prompt_cache: TH, THC
    plain = run(scheduler(dense, max_batch=1), [request(ids, 10, "x")])["x"]
    out, _, _ = spec(sched, ids, plain, 10, "a")
    assert out == plain
    (e,) = sched.eng._pcache.entries
    assert e.ids == ids[:-1] and e.reply is None and tail_of(e, len(e.ids)) is not None
    assert any("id=a prompt cache reply not kept: reasoning" in m for m in caplog.messages)


def test_dflash_reply_store_failure_restores_the_prompt_entry(dense, reply, monkeypatch, caplog):
    """The reply entry's arrays failing to evaluate: it goes, the prompt
    entry it absorbed comes back, the request ended normally."""
    p = rk.first(499)
    plain = rk.off_run(monkeypatch, dense, p, 9, "x")[0]
    sched = scheduler(dense, max_batch=1)
    pc = sched.eng._pcache
    store = sched._store_reply

    def broken(row, cache, tail=None):
        new, covered = store(row, cache, tail)
        assert new is not None and covered

        class Bad(worker._Entry):                           # its evaluation fails
            arrays = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
        new.__class__ = Bad
        return new, covered
    sched._store_reply = broken
    out, _, _ = spec(sched, p, plain, 9, "a")
    assert out == plain
    (e,) = pc.entries
    assert e.ids == p[:-1] and e.reply is None and pc.evictions == 1
    assert pc.bytes == pc.bytes_of(pc.entries)


# --- QWEN_DRAFT_TAIL_END: the end state's own length ------------------------------
@pytest.mark.parametrize("end,window", [(40, 4096), (1000, 4096), (1000, 30), (0, 4096)])
def test_end_state_keeps_its_own_length(dense, tail, monkeypatch, end, window):
    """Checkpoints keep QWEN_DRAFT_TAIL positions, the entry's end
    QWEN_DRAFT_TAIL_END, at most the drafter's window - 1 (all it looks
    at); the exact repeat starts the drafter with them."""
    monkeypatch.setattr(worker, "DRAFT_TAIL_END", end)
    ids = prompt(500, user=70)
    plain = cold(dense, ids, 12, "x")
    ref = scheduler(dense, max_batch=1)
    _, full, _ = spec(ref, ids, plain, 12, "x")
    sched = scheduler(dense, max_batch=1)
    d = sched.eng.draft = Rec(plain[0], window)
    assert run(sched, [request(ids, 12, "x")])["x"] == plain
    (e,) = sched.eng._pcache.entries
    L, n = len(e.ids), min(end, window - 1)
    for p in e.ckpts:
        assert eq(tail_of(e, p), full[:, max(0, p - tail):p]), p
    if not end:
        assert tail_of(e, L) is None
        return
    assert eq(tail_of(e, L), full[:, max(0, L - n):L])
    d = sched.eng.draft = Rec(plain[0], window)
    y = request(ids, 12, "y")
    assert run(sched, [y])["y"] == plain and y.reused == L
    assert eq(d.ctx[0], full[:, max(0, L - n):][:, -(window - 1):])


def test_long_reply_keeps_the_whole_end_context(dense, reply, monkeypatch):
    """A reply of more rounds than the chunk list holds (it is compacted
    every 32 chunks): the end state still carries the last
    QWEN_DRAFT_TAIL_END positions, prompt positions included."""
    monkeypatch.setattr(worker, "DRAFT_TAIL_END", 150)

    def sched_():                                           # nothing stops the tiny model
        return scheduler(dense, max_batch=1, eng=engine(dense, eos=()))
    p = rk.first(510)
    with monkeypatch.context() as m:
        m.setattr(worker, "PROMPT_CACHE_REPLY", False)
        plain = run(sched_(), [request(p, 130, "x")])["x"]
        q = rk.nxt(p, plain[0], 511)
        nfull = spec(sched_(), q, run(sched_(), [request(q, 4, "x")])["x"], 4, "ref")[1]
    assert plain[2] == ("done", "length")
    sched = sched_()
    out, _, _ = spec(sched, p, plain, 130, "a")
    assert out == plain and len(sched.eng.draft.accept_lens) > 34
    (e,) = sched.eng._pcache.entries
    L = len(e.ids)
    assert e.reply == len(p) - 1 and L >= len(p) + 129
    assert near(tail_of(e, L), nfull[:, L - 150:L])


def test_prefill_keeps_the_drafters_whole_window(dense, tail):
    """The prefill's rolling context covers the drafter's window whatever
    the slab cuts (a short last slab after longer ones used to leave less)."""
    for seed in range(520, 526):
        ids = prompt(seed, user=40 + seed % 5)
        plain = cold(dense, ids, 6, "x")
        sched = scheduler(dense, max_batch=1)
        d = sched.eng.draft = Rec(plain[0], 20)
        assert run(sched, [request(ids, 6, "x")])["x"] == plain
        assert d.ctx[0].shape[1] == 19


def test_checkpoint_inside_the_end_window_takes_the_end_context(dense, tail, monkeypatch):
    """Turn 2 resumes at a checkpoint (a message start) of turn 1's entry
    that the entry's end context covers: its part before the resume point
    is more than the checkpoint's own QWEN_DRAFT_TAIL positions, the drafter
    starts with the cold run's whole context. A checkpoint before the end
    window keeps its own."""
    a = conversation(2, 530, system=30, user=lambda i: 9, answer=lambda i: 6)
    plain = [cold(dense, ids, 8, "x") for ids in a]
    ref = scheduler(dense, max_batch=1)
    _, full, _ = spec(ref, a[1], plain[1], 8, "x")
    for end in (1000, 12):
        monkeypatch.setattr(worker, "DRAFT_TAIL_END", end)
        sched = scheduler(dense, max_batch=1)
        spec(sched, a[0], plain[0], 8, "t0")
        (e,) = sched.eng._pcache.entries
        out, ctx, x = spec(sched, a[1], plain[1], 8, "t1")
        L, at = len(e.ids), x.reused
        assert out == plain[1] and 0 < at < L
        if end == 1000:
            assert eq(ctx, full)                            # every position of the prompt
        else:                                               # the end context does not reach it
            assert end - (L - at) < tail
            assert eq(ctx, full[:, at - tail:])
        (e2,) = sched.eng._pcache.entries
        # an array of its own went into the new entry: no view of the old one's buffer
        assert all(tail_of(e2, p) is not tail_of(e, L) for p in [*e2.ckpts, len(e2.ids)])
