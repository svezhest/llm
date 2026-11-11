"""One KV copy per conversation (QWEN_PROMPT_CACHE_MOVE, single stream): a hit
moves the entry's KV into the row (the entry leaves the cache), nothing is
copied at the prefill's end, and the row's own buffers become the entry when
it ends (prompt, or prompt + reply).

The mode must change memory only: most tests play one script of requests on
two fresh single-stream schedulers, with the copies (the reference) and with
the move, and compare the answers, the reused tokens and what the cache
holds afterwards (ids, checkpoints, flags, every resumable position's caches
bit for bit). Where the mode differs by design (a branch abandoned while its
entry was in a row) the difference is asserted."""
import gc
import logging
import weakref

import mlx.core as mx
import pytest

import worker
import test_reply_kv as rk
from conftest import IM_START, engine, positions, prompts, request, run, same, scheduler
from test_partial_store import cancel_after
from test_scheduler import FakeDraft, no_metal_verify

A, NL, TH, NN, THC, IM_END = rk.A, rk.NL, rk.TH, rk.NN, rk.THC, rk.IM_END
THINK = [IM_START, A, NL, TH, NL]                 # the generation prompt, thinking on


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(worker, "PROMPT_CACHE", True)
    monkeypatch.setattr(worker, "PROMPT_CACHE_REPLY", True)
    monkeypatch.setattr(worker, "PREFILL_STEP", 8)
    monkeypatch.setattr(worker, "PREFILL_MIN_STEP", 4)
    monkeypatch.setattr(worker, "PROMPT_CACHE_STRIDE", 16)
    monkeypatch.setattr(worker, "PROMPT_CACHE_CKPTS", 12)
    monkeypatch.setattr(worker, "PROMPT_CACHE_KEEP_TOKENS", 8)
    monkeypatch.setattr(worker, "PROMPT_CACHE_KEEP_WAIT_S", float("inf"))
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_MARKS", 3)
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_TAIL", 4096)
    monkeypatch.setattr(worker, "DRAFT_TAIL", 16)
    monkeypatch.setattr(worker, "DRAFT_TAIL_END", 16)
    no_metal_verify(monkeypatch)


@pytest.fixture(params=["dense", "dense_fused"])
def lm(request):
    return request.getfixturevalue(request.param)


class Rec(FakeDraft):
    def __init__(self, answer, window=4096):
        super().__init__(answer)
        self.config.sliding_window = window
        self.ctx = []

    def draft_block(self, b, hidden, cache, bs, sampler, dtype):
        self.ctx.append(hidden)
        return super().draft_block(b, hidden, cache, bs, sampler, dtype)


class Play:
    """One fresh single-stream scheduler; do() runs a request to its end."""

    def __init__(self, lm, move, think=None, eos=None, **kw):
        self.move = move
        eng = engine(lm) if eos is None else engine(lm, eos=eos)
        self.sched = scheduler(lm, max_batch=1, eng=eng, **kw)
        self.sched.eng._think = think
        self.pc = self.sched.eng._pcache
        self.trace, self.ctx, self.reqs = [], [], {}
        self.held = False                   # a test that keeps the row itself (a logged traceback)
        self.rows, lookup = [], self.sched._prompt_cache

        def spy(row):                       # every text row passes here once
            self.rows.append(weakref.ref(row))
            return lookup(row)
        self.sched._prompt_cache = spy

    def do(self, ids, n, rid, on_tick=None, draft=None, thinking=None, **params):
        d = None
        if draft is not None:
            d = self.sched.eng.draft = Rec(draft)
        x = self.reqs[rid] = request(ids, n, rid, thinking=thinking, **params)
        gc.collect()
        gc.disable()
        try:
            out = run(self.sched, [x], on_tick=on_tick(x) if on_tick else None)[rid]
            # a finished row dies with its last reference: no cycle keeps its
            # caches (GBs on a real model) waiting for the cyclic collector
            alive = [r for r in self.rows if r() is not None]
            assert not alive or self.held, "the row is kept alive by a reference cycle"
        finally:
            gc.enable()
        self.rows.clear()
        self.sched.eng.draft = None
        if d is not None:
            assert d.accept_lens, "the request did not take the speculative path"
            self.ctx.append(d.ctx[0])
        self.trace.append((rid, out, x.reused))
        assert self.sched.reserved == 0 and not self.sched.rows and self.sched.task is None
        return out[0]


def play(monkeypatch, lm, script, move, **kw):
    with monkeypatch.context() as m:
        m.setattr(worker, "PROMPT_CACHE_MOVE", move)
        p = Play(lm, move, **kw)
        assert p.sched._move_on() == move
        script(p)
        return p


def tail_of(e, p):
    return worker._state_tail(e.end if p == len(e.ids) else e.ckpts[p], e.layout)


def same_entry(pa, x, pb, y):
    assert (x.ids, sorted(x.ckpts), x.end is None, x.reply, x.partial, x.chained, x.low,
            x.diverge) == (y.ids, sorted(y.ckpts), y.end is None, y.reply, y.partial, y.chained,
                           y.low, y.diverge)
    for p in positions(x):
        assert same(pa.restore(x, p), pb.restore(y, p)), p
        tx, ty = tail_of(x, p), tail_of(y, p)
        assert (tx is None) == (ty is None), p
        assert tx is None or (tx.shape == ty.shape and mx.array_equal(tx, ty).item()), p
    # (the copies keep the dead tail of a segment resumed below its end; the row's buffers have none)
    assert x.nbytes >= y.nbytes == sum(a.nbytes for a in {id(a): a for a in y.arrays}.values())


def same_cache(pa, pb, subset=False):
    """The move cache `pb` holds what the copy cache `pa` holds (`subset`:
    its entries are among pa's)."""
    by = {tuple(e.ids): e for e in pa.entries}
    if not subset:
        assert sorted(by) == sorted(tuple(e.ids) for e in pb.entries)
        assert pa.bytes >= pb.bytes
    for y in pb.entries:
        same_entry(pa, by[tuple(y.ids)], pb, y)
    assert (pa.hits, pa.misses, pa.seq) == (pb.hits, pb.misses, pb.seq)
    assert pb.bytes == pb.bytes_of(pb.entries)


def parity(monkeypatch, lm, script, subset=False, **kw):
    a = play(monkeypatch, lm, script, False, **kw)
    b = play(monkeypatch, lm, script, True, **kw)
    assert a.trace == b.trace
    assert len(a.ctx) == len(b.ctx)
    for u, v in zip(a.ctx, b.ctx):                  # the drafter's first context of each request
        assert u.shape == v.shape and mx.array_equal(u, v).item()
    same_cache(a.pc, b.pc, subset)
    return a, b


def chat(turns=4, seed=10, n=lambda t: 5 + t, user=12):
    def script(p):
        ids = rk.first(seed)
        for t in range(turns):
            toks = p.do(ids, n(t), f"t{t}")
            ids = rk.nxt(ids, toks, seed + 10 + t, user=user)
    return script


def think_first(seed, system=24, user=10):
    return [IM_START] + prompts([system], seed)[0] + [IM_START] + prompts([user], seed + 1)[0] + THINK


def think_next(prompt, answer, seed, user=12):
    """The template's re-render with the reasoning dropped: <think>\\n\\n</think>\\n\\n + answer."""
    return (prompt[:-1] + [NN, THC, NN] + answer + [IM_END, NL, IM_START] + prompts([user], seed)[0]
            + THINK)


# --- parity with the copies --------------------------------------------------------
def test_chat_with_reply_kv(lm, small, monkeypatch, caplog):
    """Four turns, each resuming after the previous reply: same answers, same
    reused tokens, one entry (prompt + reply) with the same checkpoints and
    the same bits; while a turn runs the cache holds nothing of it."""
    caplog.set_level(logging.INFO, logger="qwen.worker")
    a, b = parity(monkeypatch, lm, chat())
    assert all(r > 0 for _, _, r in b.trace[1:])
    (e,) = b.pc.entries
    assert e.reply is not None and len(e.chain) == 2          # prompt + reply: views of one buffer
    moved = [m for m in caplog.messages if "(moved: the entry's KV is the row's now)" in m]
    assert len(moved) == 3
    assert sum("(moved: the row's own buffers)" in m for m in caplog.messages) == 4


def test_the_entry_is_out_of_the_cache_while_its_row_runs(dense, small, monkeypatch):
    """The hit empties the entry: its segments hold nothing (their buffers
    were released layer by layer), the cache's bytes drop to zero, and the
    row's store brings one entry back."""
    monkeypatch.setattr(worker, "PROMPT_CACHE_MOVE", True)
    p = Play(dense, True)
    ids = rk.first(11)
    toks = p.do(ids, 6, "a")
    (e,) = p.pc.entries
    segs = [s for s, _, _ in e.chain]
    before = p.pc.bytes
    assert before > 0 and all(s.nbytes for s in segs)
    seen = []

    def watch(x):
        def on_tick(t, s):
            if s.rows or s.task:
                seen.append((len(p.pc.entries), p.pc.bytes))
        return on_tick
    p.do(rk.nxt(ids, toks, 12), 6, "b", on_tick=watch)
    assert seen and set(seen[1:]) == {(0, 0)}                 # (the first tick: before the lookup)
    assert all(s.nbytes == 0 and not s.arrays and all(x is None for x in s.kv) for s in segs)
    assert e.chain == [] and not any(e is x for x in p.pc.entries)
    (new,) = p.pc.entries
    assert new.ids[:len(e.ids)] == e.ids and p.pc.bytes > before


def test_chat_without_reply_kv(lm, small, monkeypatch):
    """QWEN_PROMPT_CACHE_REPLY=0: entries are prompts; each turn resumes at
    a checkpoint of the previous prompt, which it absorbs (a chain)."""
    monkeypatch.setattr(worker, "PROMPT_CACHE_REPLY", False)
    a, b = parity(monkeypatch, lm, chat(5))
    (e,) = b.pc.entries
    assert e.reply is None and e.chained and len(e.chain) == 1


def test_chain_policy_matches(dense, small, monkeypatch):
    """The checkpoints an entry keeps (chain / fresh policies, thinning) do
    not depend on the mode: a long conversation, fresh entries past the
    threshold, strides, a small cap."""
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_FRESH_TOKENS", 40)
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_TAIL", 24)
    monkeypatch.setattr(worker, "PROMPT_CACHE_CKPTS", 5)
    a, b = parity(monkeypatch, dense, chat(6, seed=40, user=30))
    (e,) = b.pc.entries
    assert e.chained and 2 <= len(e.ckpts) <= 5


def test_thinking_replies_keep_the_prompt_entry(lm, small, monkeypatch, caplog):
    """A reply with a reasoning section is not stored: the row gives back
    the prompt entry (its end state the one saved at the prefill's end); the
    next turn resumes at one of its checkpoints and absorbs it."""
    caplog.set_level(logging.INFO, logger="qwen.worker")

    def script(p):
        ids = think_first(50)
        for t in range(4):
            toks = p.do(ids, 6 + t, f"t{t}")
            ids = think_next(ids, toks[3:], 60 + t)
    a, b = parity(monkeypatch, lm, script, think=[TH, THC])
    (e,) = b.pc.entries
    assert e.reply is None and e.end is not None and len(e.chain) == 1
    assert all(0 < r < len(b.reqs[f"t{t}"].ids) - 1 for t, (_, _, r) in enumerate(b.trace) if t)


def test_a_long_unstored_reply_is_cut_off_the_buffers(dense, small, monkeypatch):
    """A prompt entry given back by a row that decoded a KVCache step or
    more past it holds buffers of its own length (taken out layer by layer:
    nothing pinned beyond what it counts); the row's caches are emptied."""
    step = 256
    n = step + 40

    def script(p):
        ids = think_first(70)
        toks = p.do(ids, n, "a")
        assert len(toks) == n
        p.do(think_next(ids, toks[:5], 71), 4, "b")
    a, b = parity(monkeypatch, dense, script, think=[TH, THC], eos=())
    # the entry of "a" as it was stored: replay it alone
    with monkeypatch.context() as m:
        m.setattr(worker, "PROMPT_CACHE_MOVE", True)
        p = Play(dense, True, think=[TH, THC], eos=())
        rows = []
        orig = p.sched._settle

        def settle(row, *args, **kw):
            out = orig(row, *args, **kw)
            rows.append([row.cache[j].keys is None for j in p.sched.fa_layers])
            return out
        p.sched._settle = settle
        ids = think_first(70)
        p.do(ids, n, "a")
        (e,) = p.pc.entries
        ((seg, lo, hi),) = e.chain
        assert (lo, hi) == (0, len(ids) - 1)
        assert all(x.shape[-2] == hi for kv in seg.kv for x in kv)
        (emptied,) = rows
        assert emptied and all(emptied)


def test_siblings_share_what_they_shared(dense, small, monkeypatch):
    """Two conversations over one system prompt: the second's entry holds
    the first's segment for the shared part. Moving one conversation's entry
    releases only what no other entry holds; the other stays whole."""
    def script(p):
        sys = [IM_START] + prompts([40], 80)[0]
        a = sys + [IM_START] + prompts([10], 81)[0] + rk.GEN
        b = sys + [IM_START] + prompts([10], 82)[0] + rk.GEN
        ta = p.do(a, 5, "a0")
        tb = p.do(b, 5, "b0")
        a = rk.nxt(a, ta, 83)
        ta = p.do(a, 5, "a1")
        b = rk.nxt(b, tb, 84)
        tb = p.do(b, 5, "b1")
        p.do(rk.nxt(a, ta, 85), 5, "a2")
        p.do(rk.nxt(b, tb, 86), 5, "b2")
    a = play(monkeypatch, dense, script, False)
    b = play(monkeypatch, dense, script, True)
    assert a.trace == b.trace
    assert all(r > 40 for rid, _, r in b.trace if rid[1] != "0")
    by = {tuple(e.ids): e for e in a.pc.entries}
    assert sorted(by) == sorted(tuple(e.ids) for e in b.pc.entries) and len(by) == 2
    for y in b.pc.entries:
        same_entry(a.pc, by[tuple(y.ids)], b.pc, y)


def test_read_only_requests_do_not_move(dense, small, monkeypatch):
    """prompt_cache "read" stores nothing, so it must leave the entry where
    it is (a copy, as before); "off" looks nothing up and stores by the
    row's buffers."""
    def script(p):
        ids = rk.first(90)
        toks = p.do(ids, 5, "a")
        nxt = rk.nxt(ids, toks, 91)
        seen = []

        def watch(x):
            def on_tick(t, s):
                seen.append(len(p.pc.entries))
            return on_tick
        p.do(nxt, 5, "r", on_tick=watch, prompt_cache="read")
        assert set(seen) == {1}
        p.do(nxt, 5, "o", prompt_cache="off")
        p.do(nxt, 5, "n")
    a, b = parity(monkeypatch, dense, script)
    assert [r for _, _, r in b.trace][1] > 0 and [r for _, _, r in b.trace][2] == 0


def test_prefill_only_requests(dense, small, monkeypatch):
    """max_tokens 0 (prompt_logprobs) ends inside the prefill: the row's
    buffers become the prompt entry there."""
    def script(p):
        ids = rk.first(95)
        p.do(ids, 0, "tf", prompt_logprobs=True)
        p.do(ids, 6, "gen")
        p.do(ids + prompts([9], 96)[0], 0, "tf2", prompt_logprobs=True)
    a, b = parity(monkeypatch, dense, script)
    assert b.trace[1][2] > 0


# --- a row that ends early -----------------------------------------------------------
@pytest.mark.parametrize("kind", ["miss", "hit"])
def test_cancelled_prefill_keeps_its_prefix(lm, small, monkeypatch, kind, caplog):
    """A prefill cancelled after some slabs: the prefix it holds is stored
    as before (now the row's own buffers, with the checkpoints inherited
    from the entry it took), and the retry resumes there."""
    caplog.set_level(logging.INFO, logger="qwen.worker")

    def script(p):
        ids = rk.first(100, user=60 if kind == "miss" else 10)
        if kind == "hit":
            toks = p.do(ids, 5, "t0")
            ids = rk.nxt(ids, toks, 101, user=60)
        assert p.do(ids, 6, "x", on_tick=lambda x: cancel_after(x, 3)) == []
        p.do(ids, 6, "y")
    a, b = parity(monkeypatch, lm, script)
    x = next(t for t in b.trace if t[0] == "x")
    assert x[1][2] == ("done", "cancelled")
    assert b.trace[-1][2] > (0 if kind == "miss" else b.trace[0][2] + 8)
    assert any("(moved: the row's own buffers)" in m and "(cancelled at" in m for m in caplog.messages)


def cancel_at_lookup(p, rid):
    """Cancel request `rid` in its generator's first step (before any slab)."""
    lookup = p.sched._prompt_cache

    def wrapped(row):
        out = lookup(row)
        if row.req.id == rid:
            row.req.cancel.set()
        return out
    p.sched._prompt_cache = wrapped


def test_cancel_before_the_first_slab_gives_the_entry_back(lm, small, monkeypatch, caplog):
    """A hit at the entry's end, cancelled before it prefilled anything: the
    entry is back as it was (same object, its KV now the row's buffers), and
    the retry hits it in full."""
    caplog.set_level(logging.INFO, logger="qwen.worker")

    def script(p):
        ids = rk.first(110)
        toks = p.do(ids, 5, "t0")
        (e,) = p.pc.entries
        ids = rk.nxt(ids, toks, 111)
        cancel_at_lookup(p, "x")
        assert p.do(ids, 6, "x") == []
        (back,) = p.pc.entries
        assert back.ids == e.ids and (not p.move or back is e)
        p.do(ids, 6, "y")
    a, b = parity(monkeypatch, lm, script)
    assert b.trace[1][1][2] == ("done", "cancelled") and b.trace[1][2] == b.trace[2][2] > 0
    assert sum("gives its entry back" in m for m in caplog.messages) == 1


def test_cancel_before_the_first_slab_at_a_checkpoint(dense, small, monkeypatch):
    """A hit below the entry's end (the thinking flow), cancelled before its
    first slab: the row holds the entry's KV up to the resume point only, so
    that prefix comes back (the rest of the entry went with the move — the
    copies would have kept it); the retry resumes at the same point and
    answers the same."""
    def script(p):
        ids = think_first(120)
        toks = p.do(ids, 6, "t0")
        ids = think_next(ids, toks[3:], 121)
        cancel_at_lookup(p, "x")
        assert p.do(ids, 6, "x") == []
        p.do(ids, 6, "y")
    a = play(monkeypatch, dense, script, False, think=[TH, THC])
    b = play(monkeypatch, dense, script, True, think=[TH, THC])
    assert a.trace == b.trace
    at = b.trace[1][2]
    assert 0 < at == b.trace[2][2]
    (e,) = b.pc.entries
    same_entry(a.pc, next(x for x in a.pc.entries if x.ids == e.ids), b.pc, e)


def test_cancel_while_decoding_keeps_the_prompt(lm, small, monkeypatch):
    """A row cancelled mid-answer (a stop string, a disconnect): no reply
    entry, the prompt entry from the row's buffers."""
    def script(p):
        ids = rk.first(130)
        toks = p.do(ids, 5, "t0")
        ids = rk.nxt(ids, toks, 131)

        def stop(x):
            def on_tick(t, s):
                if s.rows and s.rows[0].n >= 3:
                    x.cancel.set()
            return on_tick
        got = p.do(ids, 40, "x", on_tick=stop)
        assert 3 <= len(got) < 40
        p.do(ids, 6, "y")
    a, b = parity(monkeypatch, lm, script, eos=())
    assert b.trace[1][1][2] == ("done", "cancelled")
    assert b.trace[2][2] == len(b.reqs["y"].ids) - 1          # the prompt entry's end: a full hit


def test_failed_step_drops_the_rows_kv(dense, small, monkeypatch, caplog):
    """A decode step that fails ends the row with an error. Its KV arrays
    hang on the failed graph, so nothing is evaluated or stored: the entry
    that moved into the row is lost (the copies keep it), the next turn is a
    miss with the same answer."""
    caplog.set_level(logging.INFO, logger="qwen.worker")

    def script(p):
        ids = rk.first(140)
        toks = p.do(ids, 5, "t0")
        ids = rk.nxt(ids, toks, 141)
        step, calls = p.sched._step, [0]

        def bad(rows):
            calls[0] += 1
            if calls[0] == 3:
                raise RuntimeError("boom")
            return step(rows)
        p.sched._step = bad
        tick = p.sched.tick

        def safe():                         # as the scheduler thread's loop (_run)
            try:
                tick()
            except Exception as e:  # noqa: BLE001
                p.sched._fail_rows(e)
        p.sched.tick = safe
        p.do(ids, 12, "x")
        p.sched._step, p.sched.tick = step, tick
        if p.move:
            assert p.pc.entries == [] and p.pc.bytes == 0
        p.do(ids, 6, "y")
    a = play(monkeypatch, dense, script, False)
    b = play(monkeypatch, dense, script, True)
    assert [t[:2] for t in a.trace] == [t[:2] for t in b.trace]
    assert b.trace[1][1][2][0] == "error"
    assert a.trace[2][2] == len(a.reqs["y"].ids) - 1 and b.trace[2][2] == 0
    assert sum("the row failed, its KV is not stored; the entry moved into it is lost" in m
               for m in caplog.messages) == 1
    assert [e.ids for e in a.pc.entries] == [e.ids for e in b.pc.entries]


def test_failed_store_costs_the_entry_only(dense, small, monkeypatch, caplog):
    """The store at the row's end raising: the request ended as it did, the
    cache holds nothing of the conversation (the entry was in the row), and
    the next turn is a miss that answers the same."""
    caplog.set_level(logging.INFO, logger="qwen.worker")
    monkeypatch.setattr(worker, "PROMPT_CACHE_MOVE", True)
    p = Play(dense, True)
    ids = rk.first(150)
    toks = p.do(ids, 5, "t0")
    ids = rk.nxt(ids, toks, 151)
    store = p.pc.store

    def boom(*a, **kw):
        raise RuntimeError("no store")
    p.pc.store = boom
    p.held = True                           # caplog keeps the logged traceback's frames
    got = p.do(ids, 6, "x")
    p.pc.store, p.held = store, False
    assert len(got) == 6 and p.trace[-1][1][2] == ("done", "length")
    assert p.pc.entries == [] and p.pc.bytes == 0
    assert any("prompt cache store failed" in m for m in caplog.messages)
    ref = run(scheduler(dense, max_batch=1), [request(ids, 6, "x")])["x"]
    assert p.trace[-1][1] == ref
    assert p.do(ids, 6, "y") == ref[0] and p.trace[-1][2] == 0


# --- DFlash rows ---------------------------------------------------------------------
def test_dflash_rows(dense, small, monkeypatch):
    """DFlash rows: same answers, the same drafter context at every hit
    (the states carry it: QWEN_DRAFT_TAIL), the same entries; exact repeat,
    next turns after the reply, a cancelled prefill."""
    def script(p):
        ids = rk.first(160, user=40)
        for t in range(3):
            plain = run(scheduler(dense, max_batch=1), [request(ids, 9 + t, "ref")])["ref"][0]
            toks = p.do(ids, 9 + t, f"t{t}", draft=plain)
            assert toks == plain
            if t == 1:
                assert p.do(ids, 9 + t, "rep", draft=plain) == plain
            ids = rk.nxt(ids, toks, 170 + t, user=30)
        plain = run(scheduler(dense, max_batch=1), [request(ids, 6, "ref")])["ref"][0]
        p.do(ids, 6, "y", draft=plain)
    a, b = parity(monkeypatch, dense, script)
    assert len(b.ctx) == 5 and all(r > 0 for _, _, r in b.trace[1:])


def test_dflash_cancelled_prefill(dense, small, monkeypatch):
    """A DFlash row's prefill cancelled after two slabs: the prefix with the
    drafter's context at its end, from the row's buffers."""
    def script(p):
        ids = rk.first(180, user=10)
        plain = run(scheduler(dense, max_batch=1), [request(ids, 6, "ref")])["ref"][0]
        toks = p.do(ids, 6, "t0", draft=plain)
        ids = rk.nxt(ids, toks, 181, user=60)
        d = p.sched.eng.draft = Rec([0] * 16)
        x = request(ids, 6, "x")
        out = run(p.sched, [x], on_tick=cancel_after(x, 2))["x"]
        p.sched.eng.draft = None
        assert out == ([], [], ("done", "cancelled")) and not d.accept_lens
        p.trace.append(("x", out, x.reused))
        plain = run(scheduler(dense, max_batch=1), [request(ids, 6, "ref")])["ref"][0]
        p.do(ids, 6, "y", draft=plain)
    a, b = parity(monkeypatch, dense, script)
    assert b.trace[-1][2] > b.trace[1][2]


def test_dflash_thinking_reply(dense, small, monkeypatch):
    """A DFlash row whose reply is not stored (reasoning): the prompt entry
    with its end context, from the row's buffers."""
    def script(p):
        ids = think_first(190, user=30)
        for t in range(3):
            plain = run(scheduler(dense, max_batch=1), [request(ids, 8, "ref")])["ref"][0]
            assert p.do(ids, 8, f"t{t}", draft=plain) == plain
            ids = think_next(ids, plain[3:], 191 + t, user=20)
    a, b = parity(monkeypatch, dense, script, think=[TH, THC])
    (e,) = b.pc.entries
    assert e.reply is None and tail_of(e, len(e.ids)) is not None


# --- when it applies -----------------------------------------------------------------
def test_off_with_a_batch_or_quantized_kv(dense, small, monkeypatch):
    """The switch does nothing with max_batch > 1 or a quantized KV: the
    entry stays in the cache while its row runs."""
    monkeypatch.setattr(worker, "PROMPT_CACHE_MOVE", True)
    for kw in (dict(max_batch=2), dict(max_batch=1, kv_bits=8, kv_start=0)):
        sched = scheduler(dense, **kw)
        assert not sched._move_on() and sched.move_state().startswith("off: needs ")
        ids = rk.first(200)
        toks = run(sched, [request(ids, 5, "a")])["a"][0]
        seen = []
        run(sched, [request(rk.nxt(ids, toks, 201), 5, "b")],
            on_tick=lambda t, s: seen.append(len(s.eng._pcache.entries)))
        assert set(seen) == {1}
    assert scheduler(dense, max_batch=1).move_state().startswith("on")
    monkeypatch.setattr(worker, "PROMPT_CACHE_MOVE", False)
    assert scheduler(dense, max_batch=1).move_state() == "off"


def test_the_cache_gives_way_to_a_lone_row(dense, small, monkeypatch, caplog):
    """An empty batch admits without a memory check; with the move on, the
    other conversations' entries are evicted for the row's reservation at
    its lookup (its own entry is in the row already)."""
    caplog.set_level(logging.INFO, logger="qwen.worker")
    monkeypatch.setattr(worker, "PROMPT_CACHE_MOVE", True)
    p = Play(dense, True)
    a = rk.first(210)
    ta = p.do(a, 5, "a0")
    b = rk.first(220)
    p.do(b, 5, "b0")
    assert len(p.pc.entries) == 2
    nxt = request(rk.nxt(a, ta, 211), 5, "a1")
    p.sched.budget = p.sched._reserve(nxt) + 1          # the row fits, nothing beside it
    out = run(p.sched, [nxt])["a1"]
    assert out[2] == ("done", "length") and nxt.reused > 0
    (e,) = p.pc.entries
    assert e.ids[:len(a)] == a
    assert any("entries evicted for the row of id=a1" in m for m in caplog.messages)


def test_a_siblings_entry_moves_when_a_copy_does_not_fit(dense, small, monkeypatch, caplog):
    """A hit on another branch's entry (a shared system prompt) copies its
    part and leaves it whole — unless the copy does not fit beside it: then
    it moves into the row as well (what it held past the resume point is
    given up instead of the memory), and the conversation it belonged to
    resumes at the shared part of the new entry."""
    caplog.set_level(logging.INFO, logger="qwen.worker")
    monkeypatch.setattr(worker, "PROMPT_CACHE_MOVE", True)
    sys = [IM_START] + prompts([40], 230)[0]
    a = sys + [IM_START] + prompts([10], 231)[0] + rk.GEN
    b = sys + [IM_START] + prompts([10], 232)[0] + rk.GEN
    for tight in (False, True):
        p = Play(dense, True)
        ta = p.do(a, 5, "a0")
        (ea,) = p.pc.entries
        x = request(b, 5, "b0")
        if tight:
            p.sched.budget = p.sched._reserve(x) + ea.nbytes // 2
        seen = []
        out = run(p.sched, [x], on_tick=lambda t, s: seen.append(len(p.pc.entries)))["b0"]
        assert out[2] == ("done", "length") and 0 < x.reused <= 41
        assert out == run(scheduler(dense, max_batch=1), [request(b, 5, "b0")])["b0"]
        moved = any("entry of another branch moves into the row" in m for m in caplog.messages)
        assert moved == tight
        if tight:
            assert set(seen[1:]) <= {0, 1} and 0 in seen[1:]
            (eb,) = p.pc.entries
            assert eb.ids[:len(b)] == b and len(eb.chain) <= 2
        else:
            assert set(seen) == {1, 2} or set(seen) == {1}
            assert len(p.pc.entries) == 2 and any(e is ea for e in p.pc.entries)
        p.sched.budget = 1 << 40
        nxt = request(rk.nxt(a, ta, 233), 5, "a1")
        out = run(p.sched, [nxt])["a1"]
        assert out == run(scheduler(dense, max_batch=1), [request(nxt.ids, 5, "a1")])["a1"]
        assert nxt.reused == (x.reused if tight else len(a) + len(ta))
        caplog.clear()


def test_more_checkpoints_than_the_cap(dense, small, monkeypatch):
    """A prompt with more checkpoint positions than QWEN_PROMPT_CACHE_CKPTS:
    with the move the states are thinned while the prefill runs (they stay
    with the row until it ends: never more than the cap + 1 of its own
    alive); same answers, the later turns hit as deep."""
    monkeypatch.setattr(worker, "PROMPT_CACHE_CKPTS", 4)
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_FRESH_TOKENS", 0)
    held = []
    thin = worker._thin

    def spy(ckpts, cap, protect):
        held.append(len(ckpts))
        return thin(ckpts, cap, protect)
    monkeypatch.setattr(worker, "_thin", spy)

    def script(p):
        ids = rk.first(240, system=150, user=40)
        for t in range(3):
            toks = p.do(ids, 5, f"t{t}")
            ids = rk.nxt(ids, toks, 241 + t, user=40)
    a = play(monkeypatch, dense, script, False)
    copies = max(held)
    held.clear()
    b = play(monkeypatch, dense, script, True)
    assert [t[:2] for t in a.trace] == [t[:2] for t in b.trace]
    # (at the store: the cap of its own + the cap inherited; during the prefill cap + 1)
    assert copies > 8 >= max(held)
    (x,), (y,) = a.pc.entries, b.pc.entries
    assert x.ids == y.ids and len(y.ckpts) <= 4 and (x.reply, x.chained) == (y.reply, y.chained)
    assert all(r > 150 for _, _, r in b.trace[1:])


def test_off_with_the_disk_tier(dense, small, monkeypatch):
    monkeypatch.setattr(worker, "PROMPT_CACHE_MOVE", True)
    sched = scheduler(dense, max_batch=1)
    sched.eng._pcache.disk = object()
    assert not sched._move_on() and "QWEN_PROMPT_CACHE_DISK" in sched.move_state()
