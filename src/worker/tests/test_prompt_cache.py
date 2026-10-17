"""The prompt cache (QWEN_PROMPT_CACHE) on the tiny random model: what an
entry keeps (absorbed turns, thinned checkpoints, the quantized-KV floor),
that every hit still answers exactly as a cold run, the elastic accounting
against the batch budget, and decode between short prefills."""
import logging
import re
import types

import pytest

import worker
from conftest import IM_START, Cold, check_cold, prompts, request, run, scheduler

# stand-ins for the chat template's tokens (anything but IM_START / EOS)
A, NL, TH, NN, THC, IM_END = 11, 12, 13, 14, 15, 16
GEN = [IM_START, A, NL, TH, NL]          # <|im_start|>assistant\n<think>\n


@pytest.fixture
def small(monkeypatch):
    """Cache on, slabs of 8 (message starts >= 4 apart cut too), a stride of 16;
    admission waits for any hit a slab deep, without a time bound (the
    bounded wait has its own tests, on a fake clock). The reply KV off (on by
    default): these tests count prefill stores, entries = prompt[:-1];
    test_reply_kv.py turns it on."""
    monkeypatch.setattr(worker, "PROMPT_CACHE", True)
    monkeypatch.setattr(worker, "PROMPT_CACHE_REPLY", False)
    monkeypatch.setattr(worker, "PREFILL_STEP", 8)
    monkeypatch.setattr(worker, "PREFILL_MIN_STEP", 4)
    monkeypatch.setattr(worker, "PROMPT_CACHE_STRIDE", 16)
    monkeypatch.setattr(worker, "PROMPT_CACHE_CKPTS", 12)
    monkeypatch.setattr(worker, "PROMPT_CACHE_KEEP_TOKENS", 8)
    monkeypatch.setattr(worker, "PROMPT_CACHE_KEEP_WAIT_S", float("inf"))
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_MARKS", 3)
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_TAIL", 4096)


def cold(lm, ids, n, rid, **kw):
    """The answer of a fresh worker (empty cache)."""
    return run(scheduler(lm, max_batch=1, **kw), [request(ids, n, rid)])[rid]


def ends_of(ids):
    return set(worker._bounds(ids, IM_START)) | {len(ids) - 1}


def below(ids, c):
    """The checkpoint _prompt_cache maps message start `c` to."""
    b = [x for x in worker._bounds(ids, IM_START) if x <= c and x < len(ids) - 1]
    return max(b, default=None)


def conversation(turns, seed, system=20, user=lambda i: 6 + 3 * i % 9, answer=lambda i: 9 + i % 4):
    """Prompts of one chat, turn by turn, as the template renders them with
    the reasoning dropped: turn N+1 starts with turn N's prompt minus its last
    token ("\\n" after <think>), then "\\n\\n</think>\\n\\n" + answer."""
    sys = [IM_START] + prompts([system], seed)[0]
    users = prompts([user(i) for i in range(turns)], seed + 1)
    answers = prompts([answer(i) for i in range(turns)], seed + 2)
    out, hist = [], sys
    for i in range(turns):
        p = hist + [IM_START] + users[i] + GEN
        out.append(p)
        hist = p[:-1] + [NN, THC, NN] + answers[i] + [IM_END, NL]
    return out


# the store line as scripts/bench_concurrency.py parses it
RX_STORE = re.compile(r"id=(\S+) prompt cache store: (\d+) tokens, (\d+) checkpoints, ([\d.]+) MB "
                      r"\((\d+) entries, ([\d.]+) MB total\) reused=(\d+) gap=(\d+) absorbed=(\d+)")
RX_STATES = re.compile(r"pieces=\d+ states=([\d.]+) MB")


def test_turns_absorb_the_previous_entry(dense, small, caplog):
    """Each turn's entry absorbs the previous turn's (one entry per
    conversation, not one per turn), every turn hits and answers as cold."""
    convo = conversation(6, 50)
    ref = [cold(dense, ids, 6, f"t{i}") for i, ids in enumerate(convo)]
    caplog.set_level(logging.INFO, logger="qwen.worker")
    sched = scheduler(dense)
    pc = sched.eng._pcache
    bits = Cold(dense)
    for i, ids in enumerate(convo):
        r = request(ids, 6, f"t{i}")
        assert run(sched, [r])[r.id] == ref[i]
        assert [e.ids for e in pc.entries] == [ids[:-1]]
        assert (r.reused > 0) == (i > 0)
        assert check_cold(pc, bits)               # the entry, bit for bit (review finding 6)
    assert (pc.hits, pc.misses, pc.evictions) == (5, 1, 0)
    stores = [RX_STORE.search(m) for m in caplog.messages
              if "prompt cache store" in m and "id=cold" not in m]
    assert len(stores) == 6 and all(stores)
    assert [int(m.group(9)) for m in stores] == [0, 1, 1, 1, 1, 1]          # absorbed
    assert all(int(m.group(8)) >= 0 for m in stores)                       # gap = split - reused
    assert pc.bytes == pc.entries[0].nbytes
    # states=: the entry's checkpoints + end state, every array once
    (e,) = pc.entries
    one = sum(a.nbytes for layer in e.end for a in layer if a is not None)
    assert e.state_bytes() == (len(e.ckpts) + 1) * one
    states = [RX_STATES.search(m) for m in caplog.messages
              if "prompt cache store" in m and "id=cold" not in m]
    assert all(states) and float(states[-1].group(1)) == round(e.state_bytes() / 1e6, 1)


def test_repeat_of_an_absorbed_prompt(dense, small):
    """The previous turn's prompt again (after the next turn absorbed its
    entry): resumes from the new entry at the old entry's deepest checkpoint.
    The old END state only moves over where it is a slab end of the new
    prompt (else the new entry's KV there was cut differently from a cold
    run); then the repeat resumes there, as from the old entry."""
    p1, p2 = conversation(2, 51)
    sched = scheduler(dense)
    pc = sched.eng._pcache
    run(sched, [request(p1, 4, "a")])
    old = pc.entries[0]
    L = len(old.ids)
    assert old.best(p1, ends_of(p1))[0] == L                    # the old entry: its end
    run(sched, [request(p2, 4, "b")])
    (new,) = pc.entries
    assert L not in worker._bounds(p2, IM_START) and L not in new.ckpts
    assert check_cold(pc, Cold(dense))           # token ids alone miss a wrong cut
    assert set(old.ckpts) <= set(new.ckpts)                    # its checkpoints moved over
    depth = max(p for p in old.ckpts if p in ends_of(p1))
    assert new.best(p1, ends_of(p1))[0] == depth
    r = request(p1, 6, "c")
    assert run(sched, [r])["c"] == cold(dense, p1, 6, "c") and r.reused == depth

    # the next prompt opens a message right at the old end, a slab end there
    for pad in range(8):
        q1 = [IM_START] + prompts([12 + pad], 52)[0] + [IM_START] + prompts([7], 53)[0] + GEN
        q2 = q1[:-1] + [IM_START] + prompts([9], 54)[0] + GEN
        if len(q1) - 1 in worker._bounds(q2, IM_START):
            break
    else:
        pytest.fail("no layout puts a slab end at the old entry's end")
    sched = scheduler(dense)
    pc = sched.eng._pcache
    run(sched, [request(q1, 4, "a")])
    run(sched, [request(q2, 4, "b")])
    (new,) = pc.entries
    assert len(q1) - 1 in new.ckpts
    assert check_cold(pc, Cold(dense))
    assert new.best(q1, ends_of(q1))[0] == len(q1) - 1
    r = request(q1, 6, "c")
    assert run(sched, [r])["c"] == cold(dense, q1, 6, "c") and r.reused == len(q1) - 1


def test_thin_rule():
    """Unprotected first, the left one of the closest pair, never the deepest."""
    ck = dict.fromkeys([6, 14, 22, 30, 38, 46, 47])
    assert sorted(worker._thin(dict(ck), 4)) == [14, 30, 38, 47]   # 46, then 6 (span from 0)
    assert sorted(worker._thin(dict(ck), 4, protect={6, 46})) == [6, 22, 46, 47]
    assert sorted(worker._thin(dict(ck), 1, protect={6, 14})) == [47]   # cap below the protected
    assert worker._thin({}, 0) == {} and worker._thin({5: 1}, 0) == {5: 1}


def test_thinning_keeps_the_system_boundary(dense, small, monkeypatch):
    """A short system prompt + a long message: thinning keeps the end of the
    system block (the old rule dropped it first), so a new conversation over
    the same system prompt resumes there."""
    system = [IM_START] + prompts([5], 60)[0]
    ids = system + [IM_START] + prompts([120], 61)[0] + GEN
    sys_ck = below(ids, len(system))
    assert sys_ck == len(system)                                 # a message start >= 4 in
    full = scheduler(dense)
    run(full, [request(ids, 2, "full")])
    every = full.eng._pcache.entries[0].ckpts                    # 12 kept: no thinning yet
    monkeypatch.setattr(worker, "PROMPT_CACHE_CKPTS", 4)
    assert sys_ck in every and sys_ck not in worker._thin(dict(every), 4)   # the old rule

    sched = scheduler(dense)
    pc = sched.eng._pcache
    run(sched, [request(ids, 2, "a")])
    e = pc.entries[0]
    assert len(e.ckpts) == 4 and sys_ck in e.ckpts
    starts = [i for i, t in enumerate(ids) if t == IM_START]
    assert {below(ids, c) for c in starts[-3:]} - {None} <= set(e.ckpts)
    other = system + [IM_START] + prompts([30], 62)[0] + GEN
    r = request(other, 6, "b")
    assert run(sched, [r])["b"] == cold(dense, other, 6, "b") and r.reused == sys_ck


def test_thinning_keeps_the_last_boundaries_over_many_turns(dense, small, monkeypatch):
    """30 turns, 6 checkpoints: one entry, which keeps the end of the system
    block and the last three message starts; the last turn answers as cold."""
    monkeypatch.setattr(worker, "PROMPT_CACHE_CKPTS", 6)
    convo = conversation(30, 70, system=6)
    sched = scheduler(dense)
    pc = sched.eng._pcache
    for i, ids in enumerate(convo):
        r = request(ids, 2, f"t{i}")
        got = run(sched, [r])
        (e,) = pc.entries
        starts = [j for j, t in enumerate(ids) if t == IM_START]
        protect = {below(ids, c) for c in starts[1:2] + starts[-3:]} - {None}
        assert len(e.ckpts) <= 6 and protect <= set(e.ckpts), i
    assert got == {r.id: cold(dense, ids, 2, r.id)} and r.reused > 0


def chain_parts(ids, marks=3):
    """The positions the chain policy keeps for prompt `ids`: (the end of the
    system block + the last `marks` message starts, the strides within the
    last QWEN_PROMPT_CACHE_CHAIN_TAIL tokens, the strides inside the first
    user message), mapped to slab ends as _prompt_cache maps them."""
    end = len(ids) - 1
    starts = [i for i, t in enumerate(ids) if t == IM_START]
    fixed = {below(ids, c) for c in starts[1:2] + starts[-marks:]} - {None}
    strides = {below(ids, c) for c in range(worker.PROMPT_CACHE_STRIDE, end,
                                             worker.PROMPT_CACHE_STRIDE)} - {None}
    doc = {p for p in strides if len(starts) > 2 and starts[1] <= p < starts[2]}
    return fixed, {p for p in strides if p >= end - worker.PROMPT_CACHE_CHAIN_TAIL}, doc


def test_chain_keeps_the_last_boundaries_and_the_tail(dense, small, monkeypatch):
    """A 30-turn chain (each turn absorbs the previous one): the entry keeps
    exactly the end of the system block, the last three message starts and
    the strides in the last QWEN_PROMPT_CACHE_CHAIN_TAIL tokens (+ its end
    state), not the cap-12 set of the fresh policy; every resumable position
    restores the cold prefill's bits and every turn answers as cold."""
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_TAIL", 40)
    convo = conversation(30, 71, system=6)
    sched = scheduler(dense)
    pc = sched.eng._pcache
    bits = Cold(dense)
    deep = 0
    for i, ids in enumerate(convo):
        r = request(ids, 2, f"t{i}")
        got = run(sched, [r])
        (e,) = pc.entries
        fixed, tail, doc = chain_parts(ids)
        if i:
            assert fixed <= set(e.ckpts) <= fixed | tail | doc, i  # the protected survive
            assert len(e.ckpts) <= len(fixed) + len(tail) + len(doc) <= 4 + 3 + len(doc)
            assert r.reused >= max(chain_parts(convo[i - 1])[0])   # resumed at a kept boundary
        deep = max(deep, len(e.ids) - min(e.ckpts))
        if i % 10 == 9:
            assert check_cold(pc, bits)
    assert got == {r.id: cold(dense, ids, 2, r.id)}
    assert deep > 100          # the history is far longer than what the chain keeps

    # the same chain under the fresh policy keeps more (the cap)
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_MARKS", -1)
    off = scheduler(dense)
    for i, ids in enumerate(convo):
        run(off, [request(ids, 2, f"t{i}")])
    (o,) = off.eng._pcache.entries
    assert len(o.ckpts) == worker.PROMPT_CACHE_CKPTS > len(e.ckpts)
    assert o.nbytes - e.nbytes == (len(o.ckpts) - len(e.ckpts)) * (e.state_bytes()
                                                                   // (len(e.ckpts) + 1))


def test_chain_repeat_of_an_older_turn(dense, small, monkeypatch):
    """After a chain absorbed its turns: the current turn again (regenerate)
    resumes at the entry's end; the previous turn again resumes at its last
    message start (a protected boundary); the turn before that (its boundary
    is gone, no tail strides) at the end of the system block. All as cold."""
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_TAIL", 0)
    convo = conversation(8, 72, system=6)
    sched = scheduler(dense)
    pc = sched.eng._pcache
    for i, ids in enumerate(convo):
        run(sched, [request(ids, 2, f"t{i}")])
    (e,) = pc.entries
    fixed, tail, doc = chain_parts(convo[-1])
    assert set(e.ckpts) == fixed | doc and tail == set()
    for k, ids in (("regen", convo[-1]), ("prev", convo[-2]), ("older", convo[-3])):
        r = request(ids, 6, k)
        assert run(sched, [r])[k] == cold(dense, ids, 6, k), k
        starts = [j for j, t in enumerate(ids) if t == IM_START]
        assert r.reused == {"regen": len(ids) - 1, "prev": below(ids, starts[-1]),
                            "older": max(doc | {below(ids, starts[1])})}[k], k
    assert check_cold(pc, Cold(dense))


@pytest.mark.parametrize("cap", [5, 7])
def test_chain_thinning_drops_strides_never_the_marks(dense, small, monkeypatch, cap):
    """A chain with more candidates than QWEN_PROMPT_CACHE_CKPTS (a long
    tail): thinning (_thin with protect = the chain's marks: the end of the
    system block + the last 4 message starts, one more than the fresh
    policy protects) drops tail strides, never a mark; every turn still
    answers as cold, every kept position restores cold bits."""
    monkeypatch.setattr(worker, "PROMPT_CACHE_CKPTS", cap)
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_MARKS", 4)
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_TAIL", 400)
    convo = conversation(10, 75, system=6, user=lambda i: 20, answer=lambda i: 20)
    sched = scheduler(dense)
    pc = sched.eng._pcache
    thinned = 0
    for i, ids in enumerate(convo):
        r = request(ids, 2, f"t{i}")
        assert run(sched, [r])[r.id] == cold(dense, ids, 2, r.id), i
        (e,) = pc.entries
        if i:
            fixed, tail, doc = chain_parts(ids, marks=4)
            assert len(fixed) == 5 or i == 1
            assert fixed <= set(e.ckpts) <= fixed | tail | doc, i
            assert len(e.ckpts) <= cap
            thinned += len((tail | doc) - fixed) > cap - len(fixed)
    assert thinned >= 5                        # thinning ran on chain entries
    assert check_cold(pc, Cold(dense))


SYS, USER = 17, 18          # stand-ins for the role tokens after <|im_start|>


@pytest.mark.parametrize("system", [False, True], ids=["no-system", "system"])
def test_document_strides_follow_the_user_role(dense, small, monkeypatch, system):
    """R2-2: the strides the chain keeps for a shared document are the first
    *user* message's, found by its role token: without a system message
    starts[1] is the first reply (whose strides would be kept instead). A
    sibling conversation diverging 100 tokens into the document resumes
    near the divergence, as cold."""
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_TAIL", 16)
    sched = scheduler(dense)
    sched.eng._im_user = USER
    pc = sched.eng._pcache
    doc, reply = prompts([120, 60], 76)
    head = ([IM_START, SYS] + prompts([6], 77)[0]) if system else []
    hist = head + [IM_START, USER] + doc
    convo = []
    for i in range(3):
        p = hist + GEN
        convo.append(p)
        hist = p[:-1] + [NN, THC, NN] + (reply if i == 0 else prompts([9], 78 + i)[0]) + \
            [IM_END, NL, IM_START, USER] + prompts([6], 90 + i)[0]
    for i, ids in enumerate(convo):
        r = request(ids, 3, f"t{i}")
        assert run(sched, [r])[r.id] == cold(dense, ids, 3, r.id)
    (e,) = pc.entries
    ids = convo[-1]
    starts = [j for j, t in enumerate(ids) if t == IM_START]
    u = starts[1 if system else 0]
    nxt = starts[starts.index(u) + 1]
    strides = {below(ids, c) for c in range(16, len(ids) - 1, 16)} - {None}
    in_doc = {p for p in strides if u <= p < nxt}
    in_reply = {p for p in strides if nxt <= p < starts[starts.index(u) + 2]}
    assert len(in_doc) >= 4 and in_reply
    assert in_doc <= set(e.ckpts) and not (in_reply - set.union(*chain_parts(ids)[:2])) & set(e.ckpts)
    sib = ids[: u + 100] + [IM_START, USER] + prompts([6], 79)[0] + GEN
    r = request(sib, 4, "sib")
    assert run(sched, [r])["sib"] == cold(dense, sib, 4, "sib")
    assert r.reused == max(p for p in in_doc if p <= u + 100)
    assert check_cold(pc, Cold(dense))


@pytest.mark.parametrize("marks", [3, -1], ids=["chain", "off"])
def test_fresh_entries_keep_the_cap_policy(dense, small, monkeypatch, marks):
    """An entry that absorbs nothing (a new conversation) keeps today's
    policy: strides deep in a long first message too. The chain keeps the
    strides of the first user message (a shared document: another
    conversation over it, diverging deep inside, still resumes near the
    divergence), but drops those of a long later message once it is deep in
    the history (QWEN_PROMPT_CACHE_CHAIN_MARKS=-1: the cap decides)."""
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_MARKS", marks)
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_TAIL", 16)
    convo = conversation(4, 73, system=6, user=lambda i: 120 if i < 2 else 6)
    t1 = convo[0]
    sched = scheduler(dense)
    pc = sched.eng._pcache
    run(sched, [request(t1, 2, "a")])
    (e1,) = pc.entries
    fixed, tail, doc = chain_parts(t1)
    assert len(doc) >= 4 and doc <= set(e1.ckpts)            # strides deep in the message
    for i, ids in enumerate(convo[1:], 1):
        r = request(ids, 4, f"t{i}")
        assert run(sched, [r])[r.id] == cold(dense, ids, 4, r.id) and r.reused > 0
    (e,) = pc.entries
    ids = convo[-1]
    starts = [j for j, t in enumerate(ids) if t == IM_START]
    second = {below(ids, c) for c in range(16, starts[4], 16)} - {None}
    second = {p for p in second if starts[3] <= p < starts[4]}       # the 2nd user message
    assert len(second) >= 4
    if marks >= 0:
        assert doc == chain_parts(ids)[2] and doc <= set(e.ckpts)
        assert set(e.ckpts) <= set.union(*chain_parts(ids)) and not second & set(e.ckpts)
    else:
        assert len(e.ckpts) == worker.PROMPT_CACHE_CKPTS
    # another conversation over the document, diverging 100 tokens into it
    s1 = starts[1]
    sib = t1[: s1 + 100] + [IM_START] + prompts([6], 74)[0] + GEN
    r = request(sib, 4, "sib")
    assert run(sched, [r])["sib"] == cold(dense, sib, 4, "sib")
    if marks >= 0:
        assert r.reused == max(p for p in doc if p <= s1 + 100) > below(sib, s1) + 64
    assert check_cold(pc, Cold(dense))


def test_quantized_floor_entry_survives(dense, small):
    """Quantized KV: a newer turn past quantized_kv_start does not absorb the
    unquantized entry before it (the only one resumable below the start), so
    another conversation over the same system prompt still hits it."""
    kw = dict(kv_bits=8, kv_start=40)
    convo = conversation(3, 80, system=16, user=lambda i: 6, answer=lambda i: 10)
    assert len(convo[0]) < 40 < len(convo[1])
    sched = scheduler(dense, **kw)
    pc = sched.eng._pcache
    for i, ids in enumerate(convo):
        r = request(ids, 4, f"t{i}")
        assert run(sched, [r])[r.id] == cold(dense, ids, 4, r.id, **kw)
    floor, last = pc.entries
    assert not floor.quantized and floor.ids == convo[0][:-1]
    assert last.quantized and last.ids == convo[2][:-1]         # turn 2 absorbed turn 1
    system = convo[0][:17]
    other = system + [IM_START] + prompts([30], 81)[0] + GEN
    r = request(other, 6, "o")
    assert run(sched, [r])["o"] == cold(dense, other, 6, "o", **kw)
    assert r.reused == below(other, 17) and 0 < r.reused < 40
    # without the floor entry the quantized one could not serve it
    assert last.best(other, ends_of(other))[0] == 0
    assert check_cold(pc, Cold(dense, **kw))


def test_store_failure_does_not_fail_the_request(dense, small, monkeypatch):
    sched = scheduler(dense)

    def boom(*a, **k):
        raise MemoryError("no room for the entry")
    monkeypatch.setattr(sched.eng._pcache, "store", boom)
    ids = [IM_START] + prompts([30], 90)[0] + GEN
    got = run(sched, [request(ids, 6, "a")])
    assert got["a"] == cold(dense, ids, 6, "a") and got["a"][2] == ("done", "length")
    assert sched.eng._pcache.entries == []


def test_decode_runs_between_short_prefills(dense, monkeypatch):
    """Rows running and several one-slab prefills waiting (cache hits, short
    prompts): each prefill is charged to decode's time share like any slab,
    so decode steps run between them; tokens as solo."""
    now = [1000.0]

    def clock():
        now[0] += 0.01
        return now[0]
    monkeypatch.setattr(worker, "time", types.SimpleNamespace(time=clock))

    class Timed:
        """The model, a prefill slab taking 1 s of the fake clock."""

        def __init__(self, lm):
            object.__setattr__(self, "lm", lm)

        def __call__(self, *a, **k):
            if k.get("skip_logits"):
                now[0] += 1.0
            return self.lm(*a, **k)

        def __getattr__(self, name):
            return getattr(self.lm, name)

        def __setattr__(self, name, v):
            setattr(self.lm, name, v)
    ps = prompts([12, 6, 7, 5, 8], 95)

    def make():
        return [request(ps[0], 60, "long")] + [request(p, 4, f"s{i}") for i, p in enumerate(ps[1:])]
    ref = {}
    for r in make():
        ref.update(run(scheduler(dense, max_batch=1), [r]))
    sched = scheduler(dense, max_batch=8, share=0.5)
    sched.lm = Timed(dense)
    joins = []
    join = sched._join
    sched._join = lambda row: (joins.append((row.req.id, sched.steps)), join(row))
    reqs = make()
    got = run(sched, reqs, arrive={3: reqs[1:]})
    assert got == ref
    steps = [s for rid, s in joins if rid != "long"]
    assert len(steps) == 4 and all(a < b for a, b in zip(steps, steps[1:])), joins


def fill(sched, n=3, seed=100, size=40):
    """n unrelated entries in the scheduler's cache."""
    ps = prompts([size] * n, seed)
    for i, p in enumerate(ps):
        run(sched, [request(p, 2, f"w{i}")])
    return ps


def test_admission_evicts_the_cache_on_demand(dense, small):
    """Budget = rows + the cache: a request that fits beside the rows but
    not beside the whole cache gets in by evicting entries (outputs
    unchanged); with room, nothing is evicted. All four entries were stored
    within the last round (none stale): the most recent one goes (the
    running row's, _PrefixCache.victim)."""
    a_ids, b_ids = prompts([10, 9], 105)
    ref = {"a": cold(dense, a_ids, 30, "a"), "b": cold(dense, b_ids, 6, "b")}
    for pressure in (False, True):
        sched = scheduler(dense)
        pc = sched.eng._pcache
        fill(sched)
        first = pc.entries[0]
        a, b = request(a_ids, 30, "a"), request(b_ids, 6, "b")
        rows, before = [], []

        def squeeze(t, s):
            if t == 3:
                before[:] = pc.entries
                if pressure:            # b fits beside a, not beside every entry
                    s.budget = s.reserved + s._reserve(b) + pc.bytes - 1
            rows.append(len(s.rows))
        got = run(sched, [a, b], arrive={3: [b]}, on_tick=squeeze)
        assert got == ref
        assert max(rows) == 2                                    # b joined while a ran
        assert (pc.evictions > 0) == pressure
        assert len(before) == 4 and first in pc.entries
        assert (before[-1] in pc.entries) != pressure            # a's entry, the most recent


def test_admission_keeps_the_entry_it_would_resume_from(dense, small):
    """When the entry the waiting request would hit is in the way even alone,
    it is not evicted, and neither are the others (that would not admit it):
    the request waits for the running row and then hits."""
    sched = scheduler(dense)
    pc = sched.eng._pcache
    ws = fill(sched)
    a_ids = prompts([10], 106)[0]
    b_ids = ws[0][:30] + prompts([8], 107)[0]
    target = pc.entries[0]
    a, b = request(a_ids, 30, "a"), request(b_ids, 6, "b")
    rows = []

    def squeeze(t, s):
        if t == 3:
            s.budget = s.reserved + s._reserve(b) + target.nbytes - 1
        rows.append(len(s.rows))
    got = run(sched, [a, b], arrive={3: [b]}, on_tick=squeeze)
    assert got == {"a": cold(dense, a_ids, 30, "a"), "b": cold(dense, b_ids, 6, "b")}
    assert max(rows) == 1                                        # b waited for a
    assert pc.evictions == 0 and b.reused > 0                    # nothing went, it hit
    assert target in pc.entries


def test_growing_rows_evict_the_cache_before_cutting(dense, small, monkeypatch):
    """Rows growing past budget + prefill reserve: the cache gives way
    first; no row is cut when the rows alone fit."""
    monkeypatch.setattr(worker, "RESERVE_TOKENS", 2)
    ps = prompts([8, 8], 54)
    probe = scheduler(dense)
    short, grown = probe._bytes(8 + 2), probe._bytes(8 + 40 + 1)
    sched = scheduler(dense, budget=1 << 40)
    pc = sched.eng._pcache
    fill(sched)
    a, b = request(ps[0], 40, "a"), request(ps[1], 40, "b")

    def squeeze(t, s):
        if len(s.rows) == 2 and s.budget > 2 * short:    # both in: headroom for the rows only
            s.budget, s.prefill_reserve = 2 * short, 2 * (grown - short) + 4096
            assert pc.bytes > s.prefill_reserve
    got = run(sched, [a, b], on_tick=squeeze)
    assert all(len(got[k][0]) == 40 for k in "ab")            # nobody cut
    assert pc.evictions > 0 and sched.reserved == 0
