"""The prompt cache's segment store and the phase-1 review fixes, on the tiny
random model (CPU): every resumable position of every entry restores the
cold prefill's bits, entries share immutable KV segments (siblings, chat
turns), restore builds the buffer the first update used to build, nothing
pins a row's or a segment's buffer, unique-byte accounting and refcounted
eviction, and the admission / growth / store policy (review findings 1-5, 7)."""
import gc
import logging
import types
import weakref

import mlx.core as mx
import numpy as np
import pytest
from numpy.lib.array_utils import byte_bounds

import worker
from conftest import (Cold, IM_START, check_cold, positions, prompts, request, run, same,
                      scheduler, vision_ids, vision_inputs)
from test_prompt_cache import GEN, cold, conversation, fill, small  # noqa: F401


# --- helpers ------------------------------------------------------------------
def spans(arrays):
    return [byte_bounds(np.array(a, copy=False)) for a in arrays]


def overlap(xs, ys):
    return any(a < d and c < b for a, b in spans(xs) for c, d in spans(ys))


def ptr(a):
    return np.array(a, copy=False).__array_interface__["data"][0]


def kv_arrays(caches):
    return [x for c in caches if not hasattr(c, "cache")
            for k in (c.keys, c.values) for x in (k if isinstance(k, (tuple, list)) else [k])]


def closure(f):
    """A function's closure cells by name, its nested functions' too."""
    cells = dict(zip(f.__code__.co_freevars, f.__closure__ or ()))
    for c in list(cells.values()):
        if isinstance(c.cell_contents, types.FunctionType):
            cells = {**closure(c.cell_contents), **cells}
    return cells


def segs(pc):
    return list({id(s): s for e in pc.entries for s, _, _ in e.chain}.values())


def two_chats(n_a=8, n_b=6, seed=40):
    """Two conversations over one system prompt (21 tokens)."""
    a = conversation(n_a, seed)
    b = conversation(n_b, seed + 10)
    sys = a[0][:21]
    return a, [sys + p[21:] for p in b]


# --- A-T1: hit == cold, bitwise, for every resumable position -----------------
@pytest.mark.parametrize("kw,ckpts", [({}, 12), ({"kv_bits": 8, "kv_start": 40}, 4)],
                         ids=["fp16", "q8"])
def test_entries_equal_cold_bitwise(dense, small, monkeypatch, kw, ckpts):
    """Two interleaved chats on one system prompt, a regenerate and an older
    prompt again; after every request every entry restores, at every
    position best() can return, exactly the caches of a cold prefill of
    its ids up to there (chains of shared segments included)."""
    monkeypatch.setattr(worker, "PROMPT_CACHE_CKPTS", ckpts)
    a, b = two_chats()
    order = [x for pair in zip(a, b) for x in pair] + a[len(b):] + [a[-1], a[2], b[1]]
    sched = scheduler(dense, **kw)
    pc = sched.eng._pcache
    ref = Cold(dense, **kw)
    checked = hits = 0
    for i, ids in enumerate(order):
        r = request(ids, 3, f"q{i}")
        run(sched, [r])
        hits += r.reused > 0
        checked += check_cold(pc, ref)
    assert hits >= len(order) // 2 and checked > 3 * len(order)
    assert any(len(e.chain) > 1 for e in pc.entries)          # chains were checked
    if kw:
        assert any(e.quantized for e in pc.entries) and any(not e.quantized for e in pc.entries)


# --- A-T2: siblings share the prefix segment ----------------------------------
def test_store_shares_prefix_segments(dense, small):
    """16 siblings over a 40-token system prompt: one segment holds the first
    request's whole prompt, every sibling adds only [40, L); the cache's
    bytes are every buffer once."""
    system = [IM_START] + prompts([39], 120)[0]
    users = prompts([8 + i % 5 for i in range(16)], 121)
    ps = [system + [IM_START] + u + GEN for u in users]
    sched = scheduler(dense)
    pc = sched.eng._pcache
    for i, p in enumerate(ps):
        r = request(p, 2, f"s{i}")
        run(sched, [r])
        assert r.reused == (40 if i else 0)
    first = pc.entries[0]
    (s0, lo0, hi0), = first.chain
    assert (s0.a, s0.b, lo0, hi0) == (0, len(ps[0]) - 1, 0, len(ps[0]) - 1)
    for e, p in zip(pc.entries[1:], ps[1:]):
        (s, lo, hi), (t, lo1, hi1) = e.chain
        assert s is s0 and (lo, hi) == (0, 40)
        assert (t.a, t.b, lo1, hi1) == (40, len(p) - 1, 40, len(p) - 1)
        assert all(k.shape[-2] == t.b - t.a for kv in t.kv for k in kv)
        assert e.nbytes >= s0.nbytes + t.nbytes                 # its shared segment counts
    per_token = s0.nbytes // (s0.b - s0.a)
    kv = sum(s.nbytes for s in segs(pc))
    assert kv == per_token * (len(ps[0]) - 1 + sum(len(p) - 1 - 40 for p in ps[1:]))
    states = {id(a): a for e in pc.entries for a in e.arrays if not any(
        a is x for s in segs(pc) for x in s.arrays)}
    assert pc.bytes == kv + sum(a.nbytes for a in states.values())
    assert pc.bytes == pc.bytes_of(pc.entries) and len(pc.entries) == 16


# --- A-T3: restore layout ------------------------------------------------------
def test_restore_layout(dense, small):
    """restore(e, at, first=n) pre-sizes the buffer as the first update did
    from the old [:at] view: same shape, same bits after the first slab,
    same logits; the first slab writes it in place. A 3-piece chain
    restores bitwise equal to a cold (1-piece) prefill."""
    convo = conversation(3, 130)
    sched = scheduler(dense)
    pc = sched.eng._pcache
    for i, ids in enumerate(convo):
        run(sched, [request(ids, 2, f"t{i}")])
    (e,) = pc.entries
    assert len(e.chain) == 3
    ref = Cold(dense)
    for p in positions(e):
        assert same(pc.restore(e, p), ref(e.ids[:p]))
    lm = dense
    p = max(x for x in e.ckpts if x < len(e.ids) - 8)
    for n in (5, 300):
        x = mx.array([prompts([n], 131)[0]])
        new = pc.restore(e, p, first=n)
        fa = [c for c in new if not hasattr(c, "cache")]
        assert all(c.offset == p and c.keys.shape[2] == p + -(-n // 256) * 256 for c in fa)
        old = pc.restore(e, p)                          # the old restore: [:p] views
        for c in old:
            if not hasattr(c, "cache"):
                c.keys, c.values = c.keys[..., :p, :], c.values[..., :p, :]
        before = [ptr(c.keys) for c in fa]
        outs = []
        for caches in (new, old):
            lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int32)
            lm._position_ids = None
            out = lm(x, cache=caches)
            mx.eval(out.logits, [c.state for c in caches])
            outs.append(out.logits)
        assert mx.array_equal(outs[0], outs[1]).item()
        for c, d in zip(new, old):
            if not hasattr(c, "cache"):
                assert c.keys.shape == d.keys.shape and mx.array_equal(c.keys, d.keys).item()
                assert mx.array_equal(c.values, d.values).item()
        assert [ptr(c.keys) for c in fa] == before            # written in place


# --- A-T4 / A-T5: refcounted segments, nothing pins a buffer -------------------
def test_eviction_frees_unshared_segments(dense, small):
    """A segment lives while some entry's chain holds it: evicting the parent
    keeps the piece its child shares, evicting the child frees both."""
    system = [IM_START] + prompts([39], 140)[0]
    pa, pb_ = (system + [IM_START] + u + GEN for u in prompts([9, 11], 141))
    sched = scheduler(dense)
    pc = sched.eng._pcache
    run(sched, [request(pa, 2, "a")])
    run(sched, [request(pb_, 2, "b")])
    parent, child = pc.entries
    assert child.chain[0][0] is parent.chain[0][0]
    shared, own = weakref.ref(parent.chain[0][0]), weakref.ref(child.chain[1][0])
    assert pc.evict(keep=(child,)) and pc.entries == [child]
    del parent
    gc.collect()
    assert shared() is not None and own() is not None
    assert pc.bytes == pc.bytes_of(pc.entries) == child.nbytes
    assert pc.evict() and pc.entries == [] and pc.bytes == 0
    del child
    gc.collect()
    assert shared() is None and own() is None


def test_restore_never_views_a_segment(dense, small):
    """A restored row holds buffers of its own: with every entry evicted,
    every segment is collected and its memory freed while the row lives."""
    ids = [IM_START] + prompts([60], 150)[0] + GEN
    sched = scheduler(dense)
    pc = sched.eng._pcache
    run(sched, [request(ids, 2, "a")])
    (e,) = pc.entries
    p = positions(e)[-2]
    row = pc.restore(e, p, first=3)
    seg_arrays = [a for s, _, _ in e.chain for a in s.arrays]
    assert not overlap(kv_arrays(row), seg_arrays)
    refs = [weakref.ref(s) for s, _, _ in e.chain]
    seg_bytes = sum(s.nbytes for s, _, _ in e.chain)
    del seg_arrays, e
    gc.collect()
    active = mx.get_active_memory()
    while pc.evict():
        pass
    gc.collect()
    assert all(r() is None for r in refs)
    assert active - mx.get_active_memory() >= seg_bytes
    assert all(c.keys is not None for c in row if not hasattr(c, "cache"))


def test_store_copies_out_of_the_row(dense, small):
    """The entry stored after a prefill holds no view of the row's KV
    buffers, so the row's first decode write is in place (no copy-on-write
    of the whole buffer); a hit's row likewise (its store copies [at, L))."""
    system = [IM_START] + prompts([39], 160)[0]
    ps = [system + [IM_START] + u + GEN for u in prompts([9, 13], 161)]
    sched = scheduler(dense)
    pc = sched.eng._pcache
    join = sched._join
    for i, p in enumerate(ps):
        rec = {}

        def spy(row):
            join(row)
            rec["overlap"] = overlap(kv_arrays(row.cache), [a for e in pc.entries for a in e.arrays])
            rec["row"], rec["before"] = row, [ptr(a) for a in kv_arrays(row.cache)]
        sched._join = spy

        def after(t, s):
            if "row" in rec and "after" not in rec and rec["row"].n >= 1:
                rec["after"] = [ptr(a) for a in kv_arrays(rec["row"].cache)]
        r = request(p, 4, f"r{i}")
        run(sched, [r], on_tick=after)
        assert r.reused == (40 if i else 0)
        assert rec["overlap"] is False
        assert rec["after"] == rec["before"]                  # written in place


# --- A-T6: a long chat is one growing chain -------------------------------------
def test_chat_is_one_chain(dense, small, monkeypatch):
    """30 turns: one entry whose chain grows by at most one piece a turn; its
    bytes are the KV of its tokens + the dead tails of resumed turns + its
    states; the last turn's entry restores as cold at its deepest positions."""
    monkeypatch.setattr(worker, "PROMPT_CACHE_CKPTS", 6)
    convo = conversation(30, 170, system=6)
    sched = scheduler(dense)
    pc = sched.eng._pcache
    prev = 0
    for i, ids in enumerate(convo):
        run(sched, [request(ids, 2, f"t{i}")])
        (e,) = pc.entries
        assert len(e.chain) <= prev + 1
        prev = len(e.chain)
        live = {id(s): s for s, _, _ in e.chain}
        assert sum(s.b - s.a for s in live.values()) >= len(e.ids)
        dead = sum(s.b - s.a for s in live.values()) - len(e.ids)
        assert dead <= i * (worker.PREFILL_STEP + len(GEN)), (i, dead)
        per_token = e.chain[0][0].nbytes // (e.chain[0][0].b - e.chain[0][0].a)
        states = sum(a.nbytes for a in e.arrays) - sum(s.nbytes for s in live.values())
        assert pc.bytes == per_token * (len(e.ids) + dead) + states
    assert prev > 20
    ref = Cold(dense)
    for p in positions(e)[-3:]:                     # the deep ones: many pieces each
        assert same(pc.restore(e, p), ref(e.ids[:p])), p


# --- A-T7: quantized child of an fp16 parent ------------------------------------
def test_quantized_child_of_fp16_parent(dense, small):
    """A row resumed from an fp16 entry below quantized_kv_start is quantized
    by its own prefill ([0, at) re-encoded): its entry copies all of [0, L)
    out of the row instead of reusing the parent's fp16 pieces; both
    entries restore as cold."""
    kw = dict(kv_bits=8, kv_start=40)
    convo = conversation(2, 180, system=16, user=lambda i: 6, answer=lambda i: 12)
    assert len(convo[0]) < 40 < len(convo[1])
    sched = scheduler(dense, **kw)
    pc = sched.eng._pcache
    run(sched, [request(convo[0], 2, "a")])
    r = request(convo[1], 2, "b")
    run(sched, [r])
    floor, child = pc.entries
    assert r.reused > 0 and not floor.quantized and child.quantized
    (s, lo, hi), = child.chain
    assert (s.a, lo, hi) == (0, 0, len(child.ids)) and s is not floor.chain[0][0]
    check_cold(pc, Cold(dense, **kw))


# --- A-T8: admission keeps the entry it would resume from (finding 1) ------------
def test_admission_keeps_the_resumable_entry(dense, small):
    """Quantized KV (start 40): an entry sharing the longest prefix with the
    waiting request is quantized past it (best() = 0), an fp16 one shares
    less but has a checkpoint inside: admission keeps the latter, evicts the
    former, and the request hits it."""
    kw = dict(kv_bits=8, kv_start=40)
    head, a, b, c = prompts([24, 6, 37, 6], 190)
    f_ids = head + a                                    # the fp16 entry: 29 tokens
    q_ids = head + b                                    # quantized: 60 tokens
    r_ids = head + b[:6] + c                            # shares 30 with q, 24 with f
    run_ids = prompts([10], 191)[0]
    sched = scheduler(dense, **kw)
    pc = sched.eng._pcache
    run(sched, [request(f_ids, 2, "f")])
    run(sched, [request(q_ids, 2, "q")])
    f_e, q_e = pc.entries
    assert not f_e.quantized and q_e.quantized
    ends = set(worker._bounds(r_ids, IM_START)) | {len(r_ids) - 1}
    assert q_e.best(r_ids, ends)[1] > f_e.best(r_ids, ends)[1]          # longer common prefix
    assert q_e.best(r_ids, ends)[0] == 0 and f_e.best(r_ids, ends)[0] == 16
    a_req, r = request(run_ids, 20, "run"), request(r_ids, 4, "r")
    rows = []

    def squeeze(t, s):
        if t == 3:          # r fits beside the running row and the fp16 entry, not both
            s.budget = s.reserved + s._reserve(r) + f_e.nbytes + 1
        rows.append(len(s.rows))
    got = run(sched, [a_req, r], arrive={3: [r]}, on_tick=squeeze)
    assert max(rows) == 2                                    # admitted beside the row
    assert pc.evictions >= 1 and r.reused == 16
    assert got["r"] == cold(dense, r_ids, 4, "r", **kw)


# --- A-T9: no eviction that would not admit (finding 2) -----------------------
def test_admission_evicts_nothing_in_vain(dense, small):
    """Evicting every other entry would not let the request in (budget, or
    the live-memory check): _fits says no and evicts nothing."""
    sched = scheduler(dense)
    pc = sched.eng._pcache
    ps = fill(sched)
    req = request(prompts([12], 200)[0], 4, "x")
    sched.reserved = 1 << 20
    need = sched._reserve(req)
    sched.budget = sched.reserved + need + pc.bytes - 1          # only the cache in the way
    sched.cap = mx.get_active_memory() - pc.bytes + need - (1 << 20)   # live: no room anyway
    assert not sched._fits(req, 1) and pc.evictions == 0 and len(pc.entries) == 3
    # an entry it would resume from and the rows already over the budget
    hit = request(ps[0][:30] + prompts([6], 201)[0], 4, "h")
    need = sched._reserve(hit)
    sched.cap = 1 << 50
    sched.budget = sched.reserved + need + pc.entries[0].nbytes - 1
    assert not sched._fits(hit, 1) and pc.evictions == 0
    sched.budget += 1                                            # the kept one alone fits
    assert sched._fits(hit, 1) and pc.evictions == 2 and len(pc.entries) == 1
    sched.reserved = 0


# --- A-T10: the store is bounded by the batch's headroom (finding 3) --------------
def test_store_bounded_by_the_budget(dense, small, caplog):
    """The entry stored beside running rows stays within budget + prefill
    reserve - reserved: LRU entries go first, and the new one is not kept
    when it alone is over (then nothing else is evicted: the cache stays as
    it was, the drop counts as an eviction); the request's tokens are
    unchanged."""
    ids = [IM_START] + prompts([50], 210)[0] + GEN
    probe = scheduler(dense)
    run(probe, [request(ids, 4, "p")])
    size = probe.eng._pcache.entries[0].nbytes
    ref = cold(dense, ids, 4, "x")
    caplog.set_level(logging.INFO, logger="qwen.worker")
    for room, kept in ((size + 1, True), (size - 1, False)):
        sched = scheduler(dense)
        pc = sched.eng._pcache
        fill(sched)
        before = list(pc.entries)
        assert pc.bytes > 0
        x = request(ids, 4, "x")

        def squeeze(t, s):
            if s.task_row is not None and s.task_row.req is x:
                s.budget = s.reserved + room
            else:                                   # stored: the row's growth has room again
                s.budget = 1 << 40
        assert run(sched, [x], on_tick=squeeze)["x"] == ref
        if kept:
            assert [e.ids for e in pc.entries] == [ids[:-1]] and pc.bytes <= room
            assert pc.evictions == 3
        else:
            assert pc.entries == before and pc.bytes == pc.bytes_of(before)
            assert pc.evictions == 1
    assert any("not kept: over the batch's headroom" in m for m in caplog.messages)


def test_store_that_cannot_keep_leaves_the_cache_intact(dense, small, monkeypatch, caplog):
    """Partial-A review finding 1: the cache holds a0 (turn 1 of chat A) and o
    (another chat) exactly at the headroom, and turn 2 of A's entry alone is
    over it. It is not kept and nothing changes: a0 is not absorbed, o not
    evicted (the store used to absorb a0, evict o, then drop the new entry:
    an empty cache), and the new segment is never copied (finding 4: sized
    from the row's shapes first). The next turn 2 with room hits a0, and the
    size estimate was the copy's exact size."""
    t1, t2 = conversation(2, 250, system=12, user=lambda i: 6 if i == 0 else 40,
                          answer=lambda i: 30)
    o_ids = [IM_START] + prompts([5], 251)[0] + GEN
    probe = scheduler(dense)                        # turn 2's entry beside a0: its size
    run(probe, [request(t1, 2, "p1")])
    run(probe, [request(t2, 2, "p2")])
    (a1,) = probe.eng._pcache.entries
    size = a1.nbytes
    sched = scheduler(dense)
    pc = sched.eng._pcache
    run(sched, [request(t1, 2, "a0")])
    run(sched, [request(o_ids, 2, "o")])
    before, room = list(pc.entries), pc.bytes
    assert len(before) == 2 and size > room                    # before the store, a0 + o fit
    ref = cold(dense, t2, 4, "x")
    copies = []
    orig = worker._Seg.copy_from

    def spy(cls, cache, a, b):
        est = worker._Seg.estimate(cache, a, b)
        seg = orig(cache, a, b)
        copies.append((a, b, est, seg.nbytes))
        return seg
    monkeypatch.setattr(worker._Seg, "copy_from", classmethod(spy))
    caplog.set_level(logging.INFO, logger="qwen.worker")
    x = request(t2, 4, "x")

    def squeeze(t, s):
        s.budget = s.reserved + room if s.task_row is not None and s.task_row.req is x else 1 << 40
    assert run(sched, [x], on_tick=squeeze)["x"] == ref
    assert x.reused > 0
    assert pc.entries == before and pc.bytes == room and pc.evictions == 1
    assert copies == []                                         # nothing copied for it
    assert any("not kept: over the batch's headroom" in m for m in caplog.messages)
    y = request(t2, 4, "x")
    assert run(sched, [y])["x"] == ref and y.reused > 0
    assert len(pc.entries) == 2 and pc.entries[0] is before[1]   # a0 absorbed, o kept
    new = pc.entries[1]
    assert new.ids == t2[:-1] and new.nbytes == size
    ((a, b, est, got),) = copies
    assert (a, b) == (new.chain[-1][0].a, new.chain[-1][0].b) and est == got


def test_admission_wait_for_a_kept_entry_is_bounded(dense, small, monkeypatch, caplog):
    """Partial-A review finding 2 (_fits, fake clock): when only the entry the
    head would resume from keeps it out, it waits only for a hit of at least
    QWEN_PROMPT_CACHE_KEEP_TOKENS and only QWEN_PROMPT_CACHE_KEEP_WAIT_S from
    the first refusal; then (or at once for a shallower hit) the entry is
    evicted too and the head admitted. The clock does not start when the
    entry is not what blocks it."""
    now = [1000.0]
    monkeypatch.setattr(worker, "time", types.SimpleNamespace(time=lambda: now[0]))
    monkeypatch.setattr(worker, "PROMPT_CACHE_KEEP_WAIT_S", 10.0)
    caplog.set_level(logging.INFO, logger="qwen.worker")

    def setup():
        sched = scheduler(dense)
        pc = sched.eng._pcache
        ws = fill(sched)
        hit = request(ws[0][:30] + prompts([6], 261)[0], 4, "h")
        target, depth = pc.peek(hit.ids, sched._ends(hit))
        assert target is pc.entries[0] and depth >= worker.PREFILL_STEP
        sched.reserved = 1 << 20
        # the target alone is in the way
        sched.budget = sched.reserved + sched._reserve(hit) + target.nbytes - 1
        return sched, pc, hit, target, depth

    # deep enough: refused for 10 s from the first refusal, then admitted
    sched, pc, hit, target, depth = setup()
    monkeypatch.setattr(worker, "PROMPT_CACHE_KEEP_TOKENS", depth)
    assert not sched._fits(hit, 1) and hit.keep_since == now[0] and pc.evictions == 0
    now[0] += 10.0
    assert not sched._fits(hit, 1) and pc.evictions == 0 and target in pc.entries
    now[0] += 0.5
    assert sched._fits(hit, 1) and pc.entries == [] and pc.evictions == 3
    assert any("gives up its" in m and "waited 10.5 s" in m for m in caplog.messages)

    # under the threshold: not waited for
    sched, pc, hit, target, depth = setup()
    monkeypatch.setattr(worker, "PROMPT_CACHE_KEEP_TOKENS", depth + 1)
    assert sched._fits(hit, 1) and pc.entries == [] and pc.evictions == 3

    # the entry is not what blocks: no clock, nothing evicted
    sched, pc, hit, target, depth = setup()
    monkeypatch.setattr(worker, "PROMPT_CACHE_KEEP_TOKENS", depth)
    sched.cap = mx.get_active_memory() - pc.bytes + sched._reserve(hit) - (1 << 20)
    assert not sched._fits(hit, 1) and hit.keep_since is None and pc.evictions == 0
    sched.cap = 1 << 50                             # the others in the way, not the target
    sched.budget += 1
    assert sched._fits(hit, 1) and pc.entries == [target] and hit.keep_since is None


@pytest.mark.parametrize("deep", [True, False], ids=["waits-then-gives-up", "shallow"])
def test_admission_gives_up_the_kept_entry_end_to_end(dense, small, monkeypatch, deep):
    """Through the scheduler on a fake clock (1 s a tick): a request whose
    kept entry is in the way waits beside a long-running row at most
    QWEN_PROMPT_CACHE_KEEP_WAIT_S (5 s here), then joins the batch cold; a
    hit shallower than QWEN_PROMPT_CACHE_KEEP_TOKENS joins at once.
    Outputs as cold."""
    now = [1000.0]
    monkeypatch.setattr(worker, "time", types.SimpleNamespace(time=lambda: now[0]))
    monkeypatch.setattr(worker, "PROMPT_CACHE_KEEP_WAIT_S", 5.0)
    sched = scheduler(dense)
    pc = sched.eng._pcache
    ws = fill(sched)
    a_ids = prompts([10], 106)[0]
    b_ids = ws[0][:30] + prompts([8], 107)[0]
    target = pc.entries[0]
    depth = pc.peek(b_ids, set(worker._bounds(b_ids, IM_START)) | {len(b_ids) - 1})[1]
    monkeypatch.setattr(worker, "PROMPT_CACHE_KEEP_TOKENS", depth if deep else depth + 1)
    a, b = request(a_ids, 30, "a"), request(b_ids, 6, "b")
    seen = {}

    def squeeze(t, s):
        now[0] += 1.0
        if t == 3:
            s.budget = s.reserved + s._reserve(b) + target.nbytes - 1
        if s.task_row is not None and s.task_row.req is b and "b" not in seen:
            seen["b"] = t                       # admitted in the tick before
            assert len(s.rows) == 1             # beside the running row
    got = run(sched, [a, b], arrive={3: [b]}, on_tick=squeeze)
    assert got == {"a": cold(dense, a_ids, 30, "a"), "b": cold(dense, b_ids, 6, "b")}
    assert b.reused == 0 and target not in pc.entries
    if deep:                                    # refused at t = 3..8 (<= 5 s), admitted at 9
        assert seen["b"] - 1 == 3 + 6 and b.keep_since is not None
    else:
        assert seen["b"] - 1 == 3 and b.keep_since is not None


# --- A-T11: a growing row does not evict the running prefill's parent (finding 4) ---
def test_regrow_keeps_the_running_prefills_parent(dense, small):
    """A row growing past the headroom between the slabs of a prefill that
    resumed from entry E evicts the other entries, not E (its pieces stay
    allocated until that prefill stores)."""
    sched = scheduler(dense)
    pc = sched.eng._pcache
    ws = fill(sched)
    target = pc.entries[0]
    g = request(prompts([8], 220)[0], 30, "g")
    p_ids = ws[0][:32] + prompts([40], 221)[0]                  # 5 slabs after the hit
    p = request(p_ids, 2, "p")
    log = []

    def squeeze(t, s):
        task = s.task_row
        if task is not None and task.req is p and task.parent is not None:
            (row,) = s.rows
            row.tokens = row.pos                                # it outgrew its reservation
            grown = s._bytes(min(row.pos + 1024, len(row.req.ids) + row.req.max_tokens + 1))
            s.budget = s.reserved - row.reserved + grown + target.nbytes // 2
            s._regrow(row)
            log.append(target in pc.entries)
            s.budget = 1 << 40
    got = run(sched, [g, p], arrive={2: [p]}, on_tick=squeeze)
    assert log and all(log) and pc.evictions >= 2
    assert p.reused == 32 and len(got["g"][0]) == 30
    assert got["p"] == cold(dense, p_ids, 2, "p")


# --- Partial-A review finding 3: nothing of a stored prefill outlives the store ----
@pytest.mark.parametrize("spec", [False, True], ids=["prefill", "dflash"])
def test_stored_prefill_lets_go_of_its_parent(dense, small, monkeypatch, spec):
    """After a resumed prefill stored its entry, the parent's segments are
    held by cache entries only, not by the snapshot closure that the
    caller's frame keeps (a DFlash run keeps it for its whole decode): with
    every entry evicted while the row still runs, they are freed, and the
    closure holds neither the parent's pieces nor checkpoint states."""
    from test_scheduler import FakeDraft, no_metal_verify
    no_metal_verify(monkeypatch)
    t1, t2 = conversation(2, 270)
    plain = cold(dense, t2, 30, "b")
    sched = scheduler(dense)
    pc = sched.eng._pcache
    run(sched, [request(t1, 2, "a")])
    refs = [weakref.ref(s) for s in segs(pc)]
    assert refs
    if spec:
        sched.eng.draft = FakeDraft(plain[0])
    snaps = []
    lookup = sched._prompt_cache

    def keep(row):                          # the closure stays referenced, as in _spec
        out = lookup(row)
        snaps.append(out[-1])
        return out
    sched._prompt_cache = keep
    b = request(t2, 30, "b")
    checked = {}

    def after(t, s):
        if checked or not any(e.ids == t2[:-1] for e in pc.entries):
            return
        assert (s.task is not None) if spec else bool(s.rows)      # the row still runs
        assert b.reused > 0
        while pc.evict():
            pass
        gc.collect()
        checked["freed"] = [w() is None for w in refs]
        cells = closure(snaps[-1])
        checked["closure"] = (cells["inherit"].cell_contents, dict(cells["ckpts"].cell_contents))
        checked["rounds"] = len(sched.eng.draft.accept_lens) if spec else 0
    assert run(sched, [b], on_tick=after)["b"] == plain
    assert checked["freed"] and all(checked["freed"])
    assert checked["closure"] == (None, {})
    if spec:                                # DFlash decoded on after the check
        assert len(sched.eng.draft.accept_lens) > checked["rounds"]


# --- A-T12: quantized entries hold no checkpoint below the start (finding 5) -------
def test_quantized_entries_have_no_subfloor_checkpoints(dense, small):
    """kv_start 40: a quantized entry keeps (and protects) no checkpoint
    below 40, its own or inherited from an fp16 parent; the fp16 floor
    entry keeps its own."""
    kw = dict(kv_bits=8, kv_start=40)
    convo = conversation(3, 230, system=16, user=lambda i: 6, answer=lambda i: 14)
    sched = scheduler(dense, **kw)
    pc = sched.eng._pcache
    long = [IM_START] + prompts([10], 231)[0] + [IM_START] + prompts([60], 232)[0] + GEN
    run(sched, [request(long, 2, "l")])
    (e,) = pc.entries
    assert e.quantized and e.ckpts and min(e.ckpts) >= 40
    for i, ids in enumerate(convo):
        run(sched, [request(ids, 2, f"t{i}")])
    for e in pc.entries:
        assert e.quantized == (len(e.ids) >= 40)
        if e.quantized:
            assert all(q >= 40 for q in e.ckpts), sorted(e.ckpts)
    assert any(not e.quantized and e.ckpts for e in pc.entries)


# --- finding 7: queue_ms runs until the prefill starts -------------------------------
def test_queue_ms_counts_until_the_prefill_starts(dense, monkeypatch):
    """Admitted while decode has its share of the GPU (after another
    prefill): t0 / queue_ms are taken when its first slab runs."""
    now = [1000.0]
    monkeypatch.setattr(worker, "time", types.SimpleNamespace(time=lambda: now[0]))

    class Timed:
        """The model, a prefill slab taking 10 s of the fake clock."""

        def __init__(self, lm):
            object.__setattr__(self, "lm", lm)

        def __call__(self, *a, **k):
            if k.get("skip_logits"):
                now[0] += 10.0
            return self.lm(*a, **k)

        def __getattr__(self, name):
            return getattr(self.lm, name)

        def __setattr__(self, name, v):
            setattr(self.lm, name, v)
    sched = scheduler(dense, max_batch=8, share=0.5)
    sched.lm = Timed(dense)
    admitted = {}
    admit = sched._admit

    def spy():
        n = len(sched.waiting)
        head = sched.waiting[0]
        admit()
        if len(sched.waiting) < n:
            admitted[head.id] = now[0]
    sched._admit = spy
    a, b = (request(p, 20, k) for p, k in zip(prompts([12, 9], 240), "ab"))
    a.t_arrival = b.t_arrival = now[0]

    def clock(t, s):
        now[0] += 0.1
    run(sched, [a, b], on_tick=clock)
    assert b.t0 - admitted["b"] > 0.3                         # decode ran in between
    assert b.queue_ms == int((b.t0 - b.t_arrival) * 1000)
    assert a.t0 == admitted["a"]                               # an empty batch starts at once


# --- agentic eviction order (QWEN_PROMPT_CACHE, victim) ---------------------------
def stub(n, seq, used):
    e = types.SimpleNamespace(ids=[0] * n, seq=seq, used=used, arrays=[], nbytes=0)
    return e


def test_victim_order(monkeypatch):
    """Wanted entries last; stale ones (not used within the last round:
    more clock ticks than 2 x entries since; a turn = a lookup + a store)
    first, LRU among them; when every entry is live, the most recently used
    (its next turn is the furthest away)."""
    monkeypatch.setattr(worker, "time", types.SimpleNamespace(time=lambda: 110.0))  # none idle
    pc = worker._PrefixCache()
    a, b, c, d = stub(1, 20, 100.0), stub(1, 22, 101.0), stub(1, 24, 102.0), stub(1, 26, 103.0)
    pc.entries = [a, b, c, d]
    pc.seq = 28                                     # a round = 4 turns = 8 ticks: all live
    assert pc.victim(pc.entries) is d               # MRU
    assert pc.victim(pc.entries, wanted=[d]) is c
    assert pc.victim(pc.entries, wanted=[a, b, c, d]) is d
    pc.seq = 29                                     # a: 9 ticks old > 8: stale
    assert pc.victim(pc.entries) is a
    pc.seq = 31                                     # a, b stale: LRU among them
    assert pc.victim(pc.entries) is a and pc.victim([b, c, d]) is b
    assert pc.victim(pc.entries, wanted=[a]) is b   # a stale entry wanted: kept
    assert pc.victim([c, d], wanted=[c, d]) is d


def test_admission_spares_entries_the_queue_would_resume_from(dense, small):
    """Admission evicting for the head of the queue spares the entry the
    head would resume from (as before) and, now, the ones of the requests
    behind it, though the most recently used one is wanted by the second
    in line; it evicts the next most recent instead. The memoized peek
    follows every change of the cache."""
    sched = scheduler(dense)
    pc = sched.eng._pcache
    ws = fill(sched, n=4)
    head = request(ws[0][:30] + prompts([6], 300)[0], 4, "h")
    second = request(ws[3][:30] + prompts([6], 301)[0], 4, "s")
    third = request(prompts([12], 302)[0], 4, "t")                 # resumes from nothing
    for r in (head, second, third):
        sched.submit(r)
    e0, e1, e2, e3 = pc.entries
    assert sched._wanted() == [e0, e3]
    v, memo = pc.version, second.peeked
    assert sched._peek(second)[0] is e3 and second.peeked is memo  # memoized: no change
    assert memo[0] == v and memo[2] >= worker.PREFILL_STEP
    assert pc.victim([e1, e2, e3]) is e3                           # e3: the MRU, would go first
    sched.reserved = 1 << 20
    sched.budget = sched.reserved + sched._reserve(head) + pc.bytes - 1
    assert sched._fits(head, 1)
    assert pc.entries == [e0, e1, e3] and pc.evictions == 1        # e2, not e3 nor e0
    assert sched._peek(second)[0] is e3 and second.peeked[0] == pc.version
    assert pc.evict(keep=[e0, e1]) and pc.entries == [e0, e1]    # its entry goes: peek again
    assert sched._peek(second) == pc.peek(second.ids, sched._ends(second)) and \
        sched._peek(second)[0] is not e3
    sched.waiting.clear()
    sched.reserved = 0


def test_round_robin_episodes_keep_hitting(dense, small, monkeypatch):
    """8 agent episodes taking turns round-robin under a cache cap that holds
    7 of their entries: LRU evicts exactly the episode whose turn is next,
    and its re-prefill evicts the next one in line (every turn a miss after
    the first round); the round-aware order evicts the most recent one:
    about one miss a round. Outputs equal either way (and to cold runs)."""
    now = [1000.0]

    def clock():
        now[0] += 0.01
        return now[0]
    monkeypatch.setattr(worker, "time", types.SimpleNamespace(time=clock))
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_TAIL", 0)
    monkeypatch.setattr(worker, "PROMPT_CACHE_CKPTS", 4)    # fresh entries as big as chained ones
    monkeypatch.setattr(worker, "PROMPT_CACHE_GB", 3.0)
    eps = [conversation(5, 400 + 10 * k, system=6, user=lambda i: 6, answer=lambda i: 9)
           for k in range(8)]
    order = [(k, t) for t in range(5) for k in range(8)]

    def go():
        sched = scheduler(dense)
        pc = sched.eng._pcache
        outs, misses = {}, []
        for k, t in order:
            r = request(eps[k][t], 3, f"e{k}t{t}")
            if pc.entries:   # the cap: 7 entries of today's largest size, not 8
                worker.PROMPT_CACHE_GB = 7.5 * max(e.nbytes for e in pc.entries) / worker.GIB
            outs.update(run(sched, [r]))
            misses.append(r.reused == 0)
        assert len(pc.entries) == 7 and pc.evictions >= 4
        return outs, sum(misses[8:])

    with monkeypatch.context() as m:
        m.setattr(worker._PrefixCache, "victim",
                  lambda self, left, wanted=(): min(left, key=lambda e: e.used))
        lru, lru_misses = go()
    new, new_misses = go()
    assert new == lru
    for k, t in order[-8:]:
        rid = f"e{k}t{t}"
        assert new[rid] == cold(dense, eps[k][t], 3, rid)
    # measured: LRU 28 of 32 (every turn from the second round on: 24), the
    # round-aware order 5 (2 in the round where the entries outgrow the cap,
    # then one a round)
    assert lru_misses >= 24
    assert new_misses <= 5


# --- review of the chain / eviction patch (docs/perf/2026-10-16/reviews/chain_evict.md) ---
def queue_entries(sched, ws, n, seed=500):
    """n waiting requests, the i-th resuming from the i-th entry a slab deep."""
    reqs = [request(ws[i][:30] + prompts([6], seed + i)[0], 2, f"w{i}") for i in range(n)]
    for r in reqs:
        sched.submit(r)
    return reqs


def test_wanted_copies_the_queue_under_the_lock(dense, small):
    """Finding 1: handlers append to `waiting` while _wanted peeks (each peek
    may take milliseconds): it iterates a copy taken under the lock, so a
    submit in between cannot raise "deque mutated during iteration" (which
    failed every running row). Finding 4: only the first max_batch waiting
    requests count. Finding 7: the admitted request whose prefill has not
    started yet counts too, first."""
    import threading
    sched = scheduler(dense, max_batch=2)
    pc = sched.eng._pcache
    ws = fill(sched, n=3)
    e0, e1, e2 = pc.entries
    queue_entries(sched, ws, 3)
    real = sched._peek

    def peek(req):
        sched.submit(request(prompts([12], 600)[0], 2, "late"))     # a handler meanwhile
        return real(req)
    sched._peek = peek
    assert sched._wanted() == [e0, e1]                 # the third is past max_batch
    del sched._peek
    with sched.cv:                                     # the copy is taken under the lock
        t = threading.Thread(target=sched._wanted)
        t.start()
        t.join(0.2)
        assert t.is_alive()
    t.join()
    row = types.SimpleNamespace(req=request(ws[2][:30] + prompts([6], 610)[0], 2, "adm"),
                                started=False)
    sched.task_row = row
    assert sched._wanted() == [e2, e0, e1]
    row.started = True                                 # its lookup ran: row.parent is kept
    assert sched._wanted() == [e0, e1]
    sched.task_row = None
    sched.waiting.clear()


def test_wanted_needs_a_slab_deep_hit(dense, small):
    """An entry a waiting request would resume from less than a slab deep is
    not spared (as _fits keeps the head's only from a slab on)."""
    sched = scheduler(dense)
    pc = sched.eng._pcache
    a = [IM_START] + prompts([5], 620)[0] + [IM_START] + prompts([30], 621)[0] + GEN
    run(sched, [request(a, 2, "a")])
    (e,) = pc.entries
    shallow = request(a[:7] + prompts([20], 622)[0] + GEN, 2, "s")
    deep = request(a[:30] + prompts([6], 623)[0] + GEN, 2, "d")
    assert sched._peek(shallow)[0] is e and 0 < sched._peek(shallow)[1] < worker.PREFILL_STEP
    assert sched._peek(deep)[0] is e and sched._peek(deep)[1] >= worker.PREFILL_STEP
    sched.submit(shallow)
    assert sched._wanted() == []
    sched.submit(deep)
    assert sched._wanted() == [e]
    sched.waiting.clear()


def test_peek_memo_follows_an_added_entry(dense, small):
    """A waiting request's memoized peek (no hit) is redone after a store
    adds an entry it would resume from (pc.version bumps once per kept
    store: at publish, not again for its segment)."""
    sched = scheduler(dense)
    pc = sched.eng._pcache
    fill(sched, n=2)
    x = prompts([30], 630)[0]
    waiting = request(x + prompts([6], 631)[0], 2, "w")
    assert sched._peek(waiting) == (None, 0)
    v = pc.version
    run(sched, [request(x + prompts([6], 632)[0], 2, "other")])
    assert pc.version == v + 1
    new = pc.entries[-1]
    e, depth = sched._peek(waiting)
    assert e is new and depth >= worker.PREFILL_STEP
    sched.submit(waiting)
    assert sched._wanted() == [new]
    sched.waiting.clear()


def half(pc, ids):
    """The KV the entry for `ids` (prompt[:-1]) covers, in tokens."""
    (e,) = [e for e in pc.entries if e.ids == ids[:-1]]
    return max(hi for _, _, hi in e.chain)


@pytest.mark.parametrize("parent", [False, True], ids=["fresh", "resumed"])
def test_store_survives_a_failing_wanted(dense, small, monkeypatch, parent):
    """Finding 2: the queue peek raising during a store that evicts costs
    only the order: it is asked before anything changes, the entry is fully
    built (KV over all its tokens; it was published without its segment
    before) and every hit answers as cold — a fresh entry (before: IndexError
    on its next hit) and one resumed from a parent (before: silently wrong
    output)."""
    sched = scheduler(dense)
    pc = sched.eng._pcache
    fill(sched, n=3)
    t1, t2 = conversation(2, 91, system=30, user=lambda i: 30)
    if parent:
        run(sched, [request(t1, 2, "t1")])
        ids = t2
    else:
        ids = [IM_START] + prompts([40], 777)[0] + GEN

    def boom():
        raise RuntimeError("deque mutated during iteration")
    sched._wanted = boom
    monkeypatch.setattr(worker, "PROMPT_CACHE_GB", (pc.bytes * 0.9) / worker.GIB)
    ev = pc.evictions
    r = request(ids, 3, "a")
    assert run(sched, [r])["a"] == cold(dense, ids, 3, "a")
    assert pc.evictions > ev and (r.reused > 0) == parent
    assert half(pc, ids) == len(ids) - 1
    del sched._wanted
    monkeypatch.setattr(worker, "PROMPT_CACHE_GB", 3.0)
    again = request(ids, 6, "regen")
    assert run(sched, [again])["regen"] == cold(dense, ids, 6, "regen")
    assert again.reused == len(ids) - 1
    assert check_cold(pc, Cold(dense))


def test_store_failing_while_evicting_rolls_back(dense, small, monkeypatch):
    """Finding 2: anything raising between the entry's publish and its
    segment (here the eviction order) leaves no half-built entry: the new
    one goes, the turn it absorbed comes back, the cache restores cold bits
    and a regenerate answers as cold."""
    sched = scheduler(dense)
    pc = sched.eng._pcache
    fill(sched, n=2)
    t1, t2 = conversation(2, 92, system=30, user=lambda i: 30)
    run(sched, [request(t1, 2, "t1")])
    before = list(pc.entries)
    v = pc.version

    def boom(left, wanted=()):
        raise RuntimeError("victim failed")
    pc.victim = boom
    monkeypatch.setattr(worker, "PROMPT_CACHE_GB", (pc.bytes * 0.9) / worker.GIB)
    r = request(t2, 3, "t2")
    assert run(sched, [r])["t2"] == cold(dense, t2, 3, "t2") and r.reused > 0
    assert set(map(id, pc.entries)) == set(map(id, before))
    assert all(e.ids != t2[:-1] for e in pc.entries)
    assert pc.bytes == pc.bytes_of(pc.entries) and pc.version > v
    del pc.victim
    monkeypatch.setattr(worker, "PROMPT_CACHE_GB", 3.0)
    again = request(t2, 6, "regen")
    assert run(sched, [again])["regen"] == cold(dense, t2, 6, "regen")
    assert check_cold(pc, Cold(dense))


def test_give_up_evicts_the_heads_entry_first(dense, small, monkeypatch):
    """Finding 3: when the head gives up the hit that alone keeps it out,
    that entry goes first; another waiting request's entry (the most recent
    one) stays when evicting the head's alone admits it."""
    monkeypatch.setattr(worker, "PROMPT_CACHE_KEEP_TOKENS", 1 << 30)   # give up at once
    sched = scheduler(dense)
    pc = sched.eng._pcache
    h_ids = [IM_START] + prompts([120], 801)[0] + GEN
    s_ids = [IM_START] + prompts([20], 802)[0] + GEN
    run(sched, [request(h_ids, 1, "H")])
    run(sched, [request(s_ids, 1, "S")])
    H, S = pc.entries
    head = request(h_ids[:100] + prompts([6], 803)[0], 2, "head")
    second = request(s_ids[:18] + prompts([6], 804)[0], 2, "second")
    sched.submit(head)
    sched.submit(second)
    assert sched._wanted() == [H, S]
    sched.reserved = 1 << 20
    need = sched._reserve(head)
    sched.budget = sched.reserved + need + (S.nbytes + H.nbytes) // 2   # H alone over
    assert S.nbytes < H.nbytes // 2
    assert sched._fits(head, 1)
    assert pc.entries == [S] and pc.evictions == 1
    sched.waiting.clear()
    sched.reserved = 0


def test_lookups_advance_the_round_clock(dense, small):
    """Finding 5: the staleness clock ticks at every lookup too (a turn = a
    lookup + a store: rule > 2 x entries). An entry hit again and again
    while another is never used: the unused one turns stale and goes first
    (with a store-only clock both stayed live and the hit one, the most
    recent, went)."""
    sched = scheduler(dense)
    pc = sched.eng._pcache
    ws = fill(sched, n=2)
    A, B = pc.entries
    assert (pc.seq, A.seq, B.seq) == (4, 2, 4)
    q = request(ws[0][:30] + prompts([6], 640)[0], 2, "q")
    for _ in range(5):
        hit, _, _ = pc.lookup(q.ids, sched._ends(q))
        assert hit is A
    assert pc.seq == 9 and A.seq == 9
    assert pc.victim(pc.entries) is B
    pc.lookup(q.ids, sched._ends(q))
    assert pc.seq - B.seq > 2 * len(pc.entries)


# --- round 2 of the review ---------------------------------------------------------
def fake_clock(monkeypatch, step=0.01, start=1000.0):
    now = [start]

    def clock():
        now[0] += step
        return now[0]
    monkeypatch.setattr(worker, "time", types.SimpleNamespace(time=clock))
    return now


def test_round_robin_with_vision_traffic_keeps_hitting(dense, small, monkeypatch):
    """R2-1: the round-robin episodes of test_round_robin_episodes_keep_hitting
    with a vision request after every text turn. Vision traffic touches no
    entry and does not tick the round clock (ticking it made every live
    episode stale: 28 of 32 misses, LRU again); the misses stay at the
    text-only level."""
    fake_clock(monkeypatch)
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_TAIL", 0)
    monkeypatch.setattr(worker, "PROMPT_CACHE_CKPTS", 4)
    monkeypatch.setattr(worker, "PROMPT_CACHE_GB", 3.0)
    eps = [conversation(5, 400 + 10 * k, system=6, user=lambda i: 6, answer=lambda i: 9)
           for k in range(8)]
    order = [(k, t) for t in range(5) for k in range(8)]
    sched = scheduler(dense)
    pc = sched.eng._pcache
    misses = []
    for k, t in order:
        r = request(eps[k][t], 3, f"e{k}t{t}")
        if pc.entries:   # the cap: 7 entries of today's largest size, not 8
            worker.PROMPT_CACHE_GB = 7.5 * max(e.nbytes for e in pc.entries) / worker.GIB
        run(sched, [r])
        misses.append(r.reused == 0)
        ids = vision_ids(700 + len(misses))
        v = request(ids, 2, f"v{len(misses)}")
        v.images, v.vision = ["img.png"], vision_inputs(ids)
        assert run(sched, [v])[v.id][2][0] == "done"
    assert len(pc.entries) == 7 and pc.evictions >= 4
    assert sum(misses[8:]) <= 5            # text only: 5 (LRU / ticking vision: 28)


def test_idle_entries_go_first(dense, small, monkeypatch):
    """R2-1: an entry untouched for more than QWEN_PROMPT_CACHE_STALE_S is
    stale even within the round of ticks: it goes before the live MRU one
    (0 = off: the MRU goes). A waiting request's entry is never idle."""
    now = fake_clock(monkeypatch)
    monkeypatch.setattr(worker, "PROMPT_CACHE_STALE_S", 300.0)
    for stale_s, gone in ((300.0, 0), (0.0, 2)):
        monkeypatch.setattr(worker, "PROMPT_CACHE_STALE_S", stale_s)
        sched = scheduler(dense)
        pc = sched.eng._pcache
        ws = prompts([40] * 4, 650)
        run(sched, [request(ws[0], 2, "a")])
        now[0] += 400.0                                        # a: idle 400 s
        for i in (1, 2):
            run(sched, [request(ws[i], 2, "bc"[i - 1])])
        entries = list(pc.entries)
        a = entries[0]
        assert pc.seq - a.seq <= 2 * len(entries)              # live by the tick rule
        assert pc.stale(a, worker.time.time()) == (stale_s > 0)
        assert not pc.stale(a, worker.time.time(), wanted=True)
        assert pc.victim(entries, wanted=entries) is entries[2]  # all wanted: MRU
        monkeypatch.setattr(worker, "PROMPT_CACHE_GB", (pc.bytes * 1.2) / worker.GIB)
        run(sched, [request(ws[3], 2, "d")])                    # the store evicts one
        assert pc.evictions == 1
        assert entries[gone] not in pc.entries and len(pc.entries) == 3
        monkeypatch.setattr(worker, "PROMPT_CACHE_GB", 3.0)


def test_evict_survives_a_failing_wanted(dense, small, caplog):
    """R2-3: _fits / _regrow pass the queue peek to evict as a callable; its
    failure is logged and costs only the order (it used to leave tick() and
    fail every running row)."""
    sched = scheduler(dense)
    pc = sched.eng._pcache
    fill(sched, n=3)

    def boom():
        raise RuntimeError("peek failed")
    with caplog.at_level(logging.ERROR, logger="qwen.worker"):
        assert pc.evict(wanted=boom)
    assert len(pc.entries) == 2 and "entries are unknown" in caplog.text
