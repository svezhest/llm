"""Two prompt-cache policies of 2026-10-17 (docs/perf/2026-10-17/design/
partial_store.md), on the tiny random model (CPU):

A. a text prefill cancelled (or failing outside the forward) after at least
   one slab stores the prefix it holds, prompt[:pos] at the last completed
   slab end, through the normal store (budget, absorb, chain policy): a
   retry resumes there and answers bit for bit as a cold run;
B. a fresh entry of at least QWEN_PROMPT_CACHE_CHAIN_FRESH_TOKENS takes the
   chain checkpoint policy (shorter ones keep the cap policy; 0 = off)."""
import inspect
import logging

import pytest

import worker
from conftest import Cold, IM_START, check_cold, prompts, request, run, scheduler
from test_prompt_cache import (GEN, NL, NN, THC, IM_END, RX_STORE, below, chain_parts, cold,
                               conversation, fill, small)  # noqa: F401


def prompt(seed, user=60, system=12):
    """<|im_start|> system, <|im_start|> user, generation prompt."""
    return ([IM_START] + prompts([system], seed)[0] + [IM_START] + prompts([user], seed + 1)[0]
            + GEN)


def slab_ends(ids, after=0):
    return [b for b in worker._bounds(ids, IM_START) if b > after]


def cancel_after(x, k):
    """on_tick: cancel request x once its prefill completed k slabs (one
    _advance per tick while it is the task: no decode rows here)."""
    seen = [0]

    def on_tick(t, s):
        if s.task_row is not None and s.task_row.req is x and s.task_row.started:
            seen[0] += 1
            if seen[0] == k:
                x.cancel.set()
    return on_tick


def stores(caplog, rid):
    return [m for m in caplog.messages if f"id={rid} prompt cache store" in m]


# --- A: store on cancel --------------------------------------------------------
@pytest.mark.parametrize("kind", ["miss", "hit", "chain"])
def test_cancel_stores_the_prefix_and_the_retry_resumes_there(dense, small, caplog, kind):
    """Cancelled after k slabs: an entry for prompt[:pos], pos the k-th slab
    end past the resume point, with the row's state at pos as its end state,
    restoring the cold prefill's bits at every resumable position; logged
    as a store (the parsers' regex) + "(cancelled at pos/len)". A retry of
    the prompt resumes at pos, answers as cold and absorbs it; its
    checkpoints are the uncancelled run's (+ pos at most: the partial
    entry's end; below the cap here — past it pos takes part in the
    thinning, so the set is a subset of uncancelled + pos thinned to the
    cap): absorbing a cancelled prefix alone makes no chain.
    hit: resumed from a sibling's system block; chain: the conversation's
    previous turn cached (the partial absorbs it, the retry keeps the chain
    policy)."""
    if kind == "chain":
        prev, ids = conversation(2, 300, system=12, user=lambda i: 6 if i == 0 else 60)
    else:
        ids = prompt(300)
        prev = ids[:13] + [IM_START] + prompts([9], 302)[0] + GEN      # the same system block
    caplog.set_level(logging.INFO, logger="qwen.worker")

    def warm(sched):
        if kind != "miss":
            run(sched, [request(prev, 2, "w")])

    ref_sched = scheduler(dense)                    # the uncancelled run's entry
    warm(ref_sched)
    run(ref_sched, [request(ids, 2, "ref")])
    ref_entry = ref_sched.eng._pcache.entries[-1]
    assert ref_entry.ids == ids[:-1]

    sched = scheduler(dense)
    pc = sched.eng._pcache
    warm(sched)
    x = request(ids, 6, "x")
    got = run(sched, [x], on_tick=cancel_after(x, 3))["x"]
    assert got == ([], [], ("done", "cancelled"))
    pos = slab_ends(ids, x.reused)[2]
    assert (x.reused > 0) == (kind != "miss")
    part = [e for e in pc.entries if e.partial]
    assert len(part) == 1 and part[0].ids == ids[:pos] and part[0].end is not None
    assert part[0].chained == (kind == "chain")
    assert [e.ids for e in pc.entries if not e.partial] == ([] if kind != "hit" else [prev[:-1]])
    (line,) = stores(caplog, "x")
    m = RX_STORE.search(line)
    assert m and int(m.group(2)) == pos and line.endswith(f" (cancelled at {pos}/{len(ids)})")
    assert int(m.group(9)) == (kind == "chain")                          # absorbed
    bits = Cold(dense)
    assert check_cold(pc, bits)

    y = request(ids, 6, "y")
    assert run(sched, [y])["y"] == cold(dense, ids, 6, "y")
    assert y.reused == pos
    (full,) = [e for e in pc.entries if e.ids == ids[:-1]]
    assert not full.partial and full.chained == (kind == "chain")
    assert not any(e.partial for e in pc.entries)                        # absorbed
    assert set(ref_entry.ckpts) <= set(full.ckpts) <= set(ref_entry.ckpts) | {pos}
    line = stores(caplog, "y")[0]                        # (then the cold reference's)
    assert RX_STORE.search(line).group(9) == "1" and "cancelled at" not in line
    assert check_cold(pc, bits)


@pytest.mark.parametrize("kind", ["miss", "hit"])
def test_cancel_before_the_first_slab_stores_nothing(dense, small, caplog, kind):
    """Cancelled before its first slab completes (MISS: nothing prefilled;
    HIT: nothing past the resume point): no store, the cache as it was."""
    ids = prompt(310)
    sched = scheduler(dense)
    pc = sched.eng._pcache
    if kind == "hit":
        run(sched, [request(ids[:13] + [IM_START] + prompts([9], 312)[0] + GEN, 2, "w")])
    before, nbytes = list(pc.entries), pc.bytes
    x = request(ids, 6, "x")
    lookup = sched._prompt_cache

    def cancel_now(row):                    # the generator's first step: before any slab
        out = lookup(row)
        row.req.cancel.set()
        return out
    sched._prompt_cache = cancel_now
    caplog.set_level(logging.INFO, logger="qwen.worker")
    assert run(sched, [x])["x"] == ([], [], ("done", "cancelled"))
    assert (x.reused > 0) == (kind == "hit")
    assert pc.entries == before and pc.bytes == nbytes and not stores(caplog, "x")
    assert sched.reserved == 0 and sched.task is None


def test_partial_store_is_bounded_by_the_budget(dense, small, caplog):
    """The partial store goes through the same headroom check as a full one
    (budget + prefill reserve - reserved): when the prefix's entry alone is
    over it, it is not kept and the cache stays as it was (counted as an
    eviction); with room it is kept, other entries evicted for it."""
    ids = prompt(320, user=80)
    probe = scheduler(dense)
    x = request(ids, 4, "p")
    run(probe, [x], on_tick=cancel_after(x, 4))
    (e,) = probe.eng._pcache.entries
    size, pos = e.nbytes, len(e.ids)
    caplog.set_level(logging.INFO, logger="qwen.worker")
    for room, kept in ((size + 1, True), (size - 1, False)):
        sched = scheduler(dense)
        pc = sched.eng._pcache
        fill(sched)
        before = list(pc.entries)
        x = request(ids, 4, "x")
        cancel = cancel_after(x, 4)

        def squeeze(t, s):
            cancel(t, s)
            s.budget = s.reserved + room if s.task_row is not None and s.task_row.req is x \
                else 1 << 40
        assert run(sched, [x], on_tick=squeeze)["x"][2] == ("done", "cancelled")
        if kept:
            assert [e.ids for e in pc.entries] == [ids[:pos]] and pc.bytes <= room
            assert pc.evictions == 3 and len(stores(caplog, "x")) == 1
        else:
            assert pc.entries == before and pc.evictions == 1
            assert any(f"id=x prompt cache: the cancelled prefill's prefix ({pos}/{len(ids)} "
                       "tokens) was not kept" in m for m in caplog.messages)
        caplog.clear()
    assert sched.reserved == 0


def test_partial_store_beside_decoding_rows(dense, small, monkeypatch):
    """A row decodes while another request prefills (decode and prefill
    ticks alternate) and is cancelled: the prefix is stored from _advance
    (where the full store runs: never between a decode step being queued
    and its tokens being read), the decoding row answers as alone, and a
    retry resumes at the stored prefix, as cold."""
    a_ids = prompts([10], 330)[0]
    ids = prompt(331, user=80)
    ref_a = cold(dense, a_ids, 40, "a")
    sched = scheduler(dense)
    pc = sched.eng._pcache
    a, x = request(a_ids, 40, "a"), request(ids, 4, "x")
    target = slab_ends(ids)[3]
    fa = dense.model.fa_idx
    calls, steps = [], []
    orig = pc.store

    def spy(*args, **kw):
        names = [f.function for f in inspect.stack()]
        calls.append(("_advance" in names, "_iterate" in names, kw.get("partial")))
        return orig(*args, **kw)
    monkeypatch.setattr(pc, "store", spy)

    def on_tick(t, s):
        row = s.task_row
        if row is not None and row.req is x:
            s.resume = 0.0 if t % 2 else float("inf")     # odd ticks prefill, even ones decode
            steps.append(s.steps)
            if row.cache is not None and row.cache[fa].offset == target:
                x.cancel.set()
    got = run(sched, [a, x], arrive={3: [x]}, on_tick=on_tick)
    assert got["a"] == ref_a and got["x"] == ([], [], ("done", "cancelled"))
    assert steps[-1] - steps[0] >= 3                      # a decoded during x's prefill
    assert calls[-1] == (True, False, True)                 # x's prefix, from _advance
    assert all(adv and not it for adv, it, _ in calls)
    (e,) = [e for e in pc.entries if e.partial]
    assert e.ids == ids[:target]
    assert check_cold(pc, Cold(dense))
    y = request(ids, 6, "y")
    assert run(sched, [y])["y"] == cold(dense, ids, 6, "y") and y.reused == target


def test_dflash_prefill_cancel_stores_the_prefix_too(dense, small, monkeypatch):
    """A DFlash run (_spec) cancelled mid-prefill stores the prefix it holds
    as the same plain prefill does (since 2026-11-11; the drafter's context
    at the stored end: tests/test_draft_tail.py), and its retry resumes there."""
    from test_scheduler import FakeDraft, no_metal_verify
    no_metal_verify(monkeypatch)
    ids = prompt(340)
    plain = cold(dense, ids, 20, "x")
    pos = slab_ends(ids)[2]
    sched = scheduler(dense, max_batch=1)
    sched.eng.draft = FakeDraft(plain[0])
    specs = []
    orig = sched._spec

    def spec(row):
        specs.append(row)
        return orig(row)
    sched._spec = spec
    x = request(ids, 20, "x")
    assert run(sched, [x], on_tick=cancel_after(x, 3))["x"] == ([], [], ("done", "cancelled"))
    (e,) = sched.eng._pcache.entries
    assert specs and e.partial and e.ids == ids[:pos]
    assert check_cold(sched.eng._pcache, Cold(dense))
    sched.eng.draft = FakeDraft(plain[0])
    y = request(ids, 20, "y")
    assert run(sched, [y])["y"] == plain and y.reused == pos and len(specs) == 2
    plain_sched = scheduler(dense, max_batch=1)
    x = request(ids, 20, "x")
    run(plain_sched, [x], on_tick=cancel_after(x, 3))
    assert [e.ids for e in plain_sched.eng._pcache.entries] == [ids[:pos]]


def test_failure_outside_the_forward_stores_the_prefix(dense, small, monkeypatch, caplog):
    """A prefill failing after its slab was evaluated (the caches hold
    exactly prompt[:pos]) stores the prefix and fails the request; one
    failing inside the forward (the caches' offsets moved) stores nothing."""
    ids = prompt(350)
    caplog.set_level(logging.INFO, logger="qwen.worker")
    sched = scheduler(dense)
    n = [0]
    orig = sched._slab_done

    def boom(t):
        n[0] += 1
        if n[0] == 3:
            raise RuntimeError("after the slab")
        orig(t)
    monkeypatch.setattr(sched, "_slab_done", boom)
    got = run(sched, [request(ids, 4, "x")])["x"]
    assert got[2][0] == "error"
    pos = slab_ends(ids)[2]
    assert [e.ids for e in sched.eng._pcache.entries] == [ids[:pos]]
    assert stores(caplog, "x")[0].endswith(f" (failed at {pos}/{len(ids)})")
    assert check_cold(sched.eng._pcache, Cold(dense))

    sched = scheduler(dense)
    q = sched.quantize
    n[0] = 0

    def broken(cache):
        n[0] += 1
        if n[0] == 3:
            raise RuntimeError("inside the forward")
        q(cache)
    monkeypatch.setattr(sched, "quantize", broken)
    got = run(sched, [request(ids, 4, "z")])["z"]
    assert got[2][0] == "error" and sched.eng._pcache.entries == []
    # never tried (the cache's own offset check would refuse it too)
    assert not stores(caplog, "z") and not any("id=z" in m and "not stored" in m
                                               for m in caplog.messages)


@pytest.mark.parametrize("k", [2, 8], ids=["below-start", "quantized"])
def test_quantized_kv_partial(dense, small, k):
    """Quantized KV (start 40): a prefix cancelled below the start is an
    unquantized entry (resumable anywhere: a cold run's fp16 bits), past it
    a quantized one resumable from the start on; either way the retry
    resumes at the prefix and answers as cold."""
    kw = dict(kv_bits=8, kv_start=40)
    ids = prompt(360, user=80)
    sched = scheduler(dense, **kw)
    pc = sched.eng._pcache
    x = request(ids, 4, "x")
    run(sched, [x], on_tick=cancel_after(x, k))
    pos = slab_ends(ids)[k - 1]
    (e,) = pc.entries
    assert e.ids == ids[:pos] and e.quantized == (pos > 40) and (pos > 40) == (k == 8)
    assert e.low == (40 if e.quantized else 0) and all(p >= e.low for p in e.ckpts)
    bits = Cold(dense, **kw)
    assert check_cold(pc, bits)
    y = request(ids, 6, "y")
    assert run(sched, [y])["y"] == cold(dense, ids, 6, "y", **kw) and y.reused == pos
    assert check_cold(pc, bits)


# --- B: the chain policy for long fresh entries ----------------------------------
def agentic(seed, turns=4, system=12, user0=30, reply=40, tool=40):
    """One agent episode's prompt (a fresh entry: nothing cached): system,
    first user message, `turns` x (assistant answer, tool output), the
    generation prompt."""
    hist = [IM_START] + prompts([system], seed)[0] + [IM_START] + prompts([user0], seed + 1)[0]
    for i in range(turns):
        hist += ([IM_START] + [NN, THC, NN] + prompts([reply], seed + 10 + i)[0] + [IM_END, NL]
                 + [IM_START] + prompts([tool], seed + 20 + i)[0])
    return hist + GEN


def sys_strides(ids):
    """The strides inside the system block (before the first user message;
    no role token in these tests: starts[1]) a long fresh entry keeps."""
    starts = [j for j, t in enumerate(ids) if t == IM_START]
    end = len(ids) - 1
    strides = {below(ids, c) for c in range(worker.PROMPT_CACHE_STRIDE, end,
                                             worker.PROMPT_CACHE_STRIDE)} - {None}
    return {p for p in strides if len(starts) > 2 and p < starts[1]}


def entry(lm, ids, fresh, **kw):
    sched = scheduler(lm, **kw)
    worker.PROMPT_CACHE_CHAIN_FRESH_TOKENS = fresh
    run(sched, [request(ids, 2, "x")])
    (e,) = sched.eng._pcache.entries
    return sched, e


def test_long_fresh_entry_takes_the_chain_policy(dense, small, monkeypatch):
    """At or past the threshold a fresh entry keeps exactly the chain set
    (end of the system block, the last three message starts, the tail and
    first-user-message strides); one token shorter it keeps the cap policy,
    as with the knob at 0 (today's behaviour: strides deep in the history
    too). Every kept position restores the cold bits, the next turn
    resumes at a kept boundary and answers as cold."""
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_TAIL", 16)
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_FRESH_TOKENS", 0)
    ids = agentic(400)
    end = len(ids) - 1
    fixed, tail, doc = chain_parts(ids)
    assert doc and tail
    sched, on = entry(dense, ids, end)                  # >= the threshold
    assert set(on.ckpts) == fixed | tail | doc and not sys_strides(ids)
    _, short = entry(dense, ids, end + 1)               # one token short of it
    _, off = entry(dense, ids, 0)
    assert set(short.ckpts) == set(off.ckpts) and len(off.ckpts) == worker.PROMPT_CACHE_CKPTS
    deep = set(off.ckpts) - (fixed | tail | doc)
    assert len(deep) >= 4                               # strides deep in the history
    assert on.nbytes < off.nbytes and not on.chained and not on.partial
    worker.PROMPT_CACHE_CHAIN_FRESH_TOKENS = end
    assert check_cold(sched.eng._pcache, Cold(dense))
    nxt = ids[:-1] + [NN, THC, NN] + prompts([9], 410)[0] + [IM_END, NL, IM_START] + \
        prompts([7], 411)[0] + GEN
    r = request(nxt, 4, "n")
    assert run(sched, [r])["n"] == cold(dense, nxt, 4, "n")
    starts = [j for j, t in enumerate(ids) if t == IM_START]
    assert r.reused >= below(ids, starts[-1])           # at the entry's last mark (or its end)


def test_long_fresh_entry_keeps_the_divergence_point(dense, small, monkeypatch):
    """A long fresh entry resumed from a sibling (a HIT that absorbs
    nothing) keeps the checkpoint at the divergence point from it, deep in
    the history where the chain set has none: a third sibling diverging
    there resumes at it, as cold."""
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_TAIL", 16)
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_FRESH_TOKENS", 64)
    ids = agentic(420)
    starts = [j for j, t in enumerate(ids) if t == IM_START]
    split = starts[3] + 20                              # inside the first tool output
    sched = scheduler(dense)
    pc = sched.eng._pcache
    sib = ids[:split] + [IM_START] + prompts([6], 421)[0] + GEN
    run(sched, [request(sib, 2, "s")])
    x = request(ids, 2, "x")
    run(sched, [x])
    assert 0 < x.reused <= split
    e = pc.entries[-1]
    assert e.ids == ids[:-1] and not e.chained
    at = below(ids, split)
    fixed, tail, doc = chain_parts(ids)
    assert at not in fixed | tail | doc and set(e.ckpts) == fixed | tail | doc | {at}
    pc.entries.remove(pc.entries[0])                    # only the long entry left
    pc._recount(added=False)
    third = ids[:split] + [IM_START] + prompts([8], 422)[0] + GEN
    r = request(third, 4, "t")
    assert run(sched, [r])["t"] == cold(dense, third, 4, "t") and r.reused == at
    assert check_cold(pc, Cold(dense))


@pytest.fixture
def prod(monkeypatch):
    """The production knobs (slab 2048, min 256, stride 2048, cap 12, chain
    3 marks / 4096 tail, fresh threshold 16384); the reply KV off, as in
    test_prompt_cache's `small` (these tests count prefill stores)."""
    monkeypatch.setattr(worker, "PROMPT_CACHE", True)
    monkeypatch.setattr(worker, "PROMPT_CACHE_REPLY", False)
    for k, v in dict(PREFILL_STEP=2048, PREFILL_MIN_STEP=256, PROMPT_CACHE_STRIDE=2048,
                     PROMPT_CACHE_CKPTS=12, PROMPT_CACHE_CHAIN_MARKS=3,
                     PROMPT_CACHE_CHAIN_TAIL=4096, PROMPT_CACHE_CHAIN_FRESH_TOKENS=16384,
                     PROMPT_CACHE_KEEP_TOKENS=8192).items():
        monkeypatch.setattr(worker, k, v)


def test_20k_fresh_entry_at_production_knobs(dense, prod):
    """A 20K-token agentic prompt after a MISS (production knobs): the chain
    set (vs 12 checkpoints under the cap policy), every kept position
    restores the cold prefill's bits; the 4K prompt of the same episode
    keeps the cap policy."""
    ids = agentic(430, turns=14, system=4700, user0=300, reply=150, tool=920)
    assert 20000 <= len(ids) < 21000
    sched, e = entry(dense, ids, 16384, max_batch=1)
    fixed, tail, doc = chain_parts(ids)
    sysb = sys_strides(ids)
    assert len(sysb) == 2                               # 2048, 4096 in the 4.7K system block
    assert set(e.ckpts) == fixed | tail | doc | sysb and len(e.ckpts) <= 8
    _, off = entry(dense, ids, 0, max_batch=1)
    assert len(off.ckpts) == 12
    worker.PROMPT_CACHE_CHAIN_FRESH_TOKENS = 16384
    assert check_cold(sched.eng._pcache, Cold(dense))
    # (at these knobs a 4K entry's cap set and chain set coincide: the 4096
    # tail covers every stride; the small-knob test shows the difference)
    four = agentic(431, turns=2, system=1000, user0=300, reply=150, tool=1300)
    _, short = entry(dense, four, 16384, max_batch=1)
    _, short_off = entry(dense, four, 0, max_batch=1)
    assert 4000 < len(four) < 4500 and set(short.ckpts) == set(short_off.ckpts)


# --- review 2026-10-17 (docs/perf/2026-10-17/reviews/partial_store.md) ------------
@pytest.mark.parametrize("sibling", ["kept", "evicted"])
def test_fresh_divergence_point_survives_a_cancel(dense, small, monkeypatch, sibling):
    """Finding 1: a long fresh prompt resumed from a sibling, cancelled past
    the divergence point and retried, keeps the uncancelled run's
    checkpoints exactly: the retry's divergence point ignores its own
    partial entry (whose end would be the split) and the partial carries
    its own (`diverge`) for when the sibling is gone by then."""
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_TAIL", 16)
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_FRESH_TOKENS", 64)
    ids = agentic(420)
    starts = [j for j, t in enumerate(ids) if t == IM_START]
    split = starts[3] + 20
    sib = ids[:split] + [IM_START] + prompts([6], 421)[0] + GEN
    at = below(ids, split)
    ref = scheduler(dense)
    run(ref, [request(sib, 2, "s")])
    run(ref, [request(ids, 2, "x")])
    ref_e = ref.eng._pcache.entries[-1]
    assert at in ref_e.ckpts
    sched = scheduler(dense)
    pc = sched.eng._pcache
    run(sched, [request(sib, 2, "s")])
    x = request(ids, 2, "x")
    run(sched, [x], on_tick=cancel_after(x, 4))
    (p,) = [e for e in pc.entries if e.partial]
    assert len(p.ids) > at and at in p.ckpts and p.diverge == {at}
    if sibling == "evicted":
        pc.entries = [p]
        pc._recount(added=False)
    y = request(ids, 2, "y")
    assert run(sched, [y])["y"] == cold(dense, ids, 2, "y") and y.reused == len(p.ids)
    (full,) = [e for e in pc.entries if e.ids == ids[:-1]]
    assert sorted(full.ckpts) == sorted(ref_e.ckpts)
    assert check_cold(pc, Cold(dense))


@pytest.mark.parametrize("fresh", [0, 64])
def test_long_system_block_sibling(dense, small, monkeypatch, fresh):
    """Finding 2: a long fresh entry keeps the strides inside its (long)
    system block, so a new session whose system prompt differs near its
    end resumes at the last one (it resumed at 0 before the fix), as cold
    and at least as deep as under the cap policy."""
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_TAIL", 16)
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_FRESH_TOKENS", fresh)
    ids = agentic(440, system=150)
    sched = scheduler(dense)
    run(sched, [request(ids, 2, "a")])
    (e,) = sched.eng._pcache.entries
    if fresh:
        fixed, tail, doc = chain_parts(ids)
        assert set(e.ckpts) <= fixed | tail | doc | sys_strides(ids) and sys_strides(ids) & set(e.ckpts)
    sib = ids[:140] + prompts([20], 441)[0] + [IM_START] + prompts([30], 442)[0] + GEN
    r = request(sib, 2, "b")
    assert run(sched, [r])["b"] == cold(dense, sib, 2, "b")
    assert r.reused == max(p for p in e.ckpts if p < 140) >= 96     # 96: the cap policy's


@pytest.mark.parametrize("path", ["cancel", "failure"])
def test_prefix_store_failure_never_changes_the_outcome(dense, small, monkeypatch, caplog, path):
    """Finding 4: an exception from the early-stop store is logged; a cancel
    still ends as cancelled, a failure keeps its own error."""
    ids = prompt(370)
    sched = scheduler(dense)
    lookup = sched._prompt_cache

    def broken(row):
        out = lookup(row)
        real = out[-1]

        def snapshot(cache, upto=None, why=None):
            if upto is not None:
                raise RuntimeError("snapshot broke")
            return real(cache)
        return (*out[:-1], snapshot)
    sched._prompt_cache = broken
    caplog.set_level(logging.INFO, logger="qwen.worker")
    x = request(ids, 4, "x")
    if path == "failure":
        n = [0]
        orig = sched._slab_done

        def boom(t):
            n[0] += 1
            if n[0] == 3:
                raise RuntimeError("after the slab")
            orig(t)
        monkeypatch.setattr(sched, "_slab_done", boom)
        got = run(sched, [x])["x"]
        assert got[2] == ("error", "RuntimeError", "after the slab")
    else:
        got = run(sched, [x], on_tick=cancel_after(x, 3))["x"]
        assert got == ([], [], ("done", "cancelled"))
    why = "failed" if path == "failure" else "cancelled"
    assert any(f"id=x prompt cache: storing the {why} prefill's prefix failed" in m
               for m in caplog.messages)
    assert sched.eng._pcache.entries == [] and sched.reserved == 0 and sched.task is None


@pytest.mark.parametrize("off", ["not-a-slab-end", "offset"])
def test_prefix_store_guard(dense, small, caplog, off):
    """Finding 5: the store refuses a prefix length that is not a slab end
    of the prompt, or one the cache does not hold (offsets), with a warning
    and nothing stored."""
    ids = prompt(380)
    sched = scheduler(dense)
    lookup = sched._prompt_cache

    def shifted(row):
        out = lookup(row)
        real = out[-1]

        def snapshot(cache, upto=None, why=None):
            if upto is None:
                return real(cache)
            bad = upto + 1 if off == "not-a-slab-end" else slab_ends(ids, upto)[0]
            return real(cache, bad, why)
        return (*out[:-1], snapshot)
    sched._prompt_cache = shifted
    caplog.set_level(logging.INFO, logger="qwen.worker")
    x = request(ids, 4, "x")
    assert run(sched, [x], on_tick=cancel_after(x, 3))["x"] == ([], [], ("done", "cancelled"))
    assert sched.eng._pcache.entries == [] and not stores(caplog, "x")
    assert any("id=x prompt cache: the cancelled prefill's prefix is not stored" in m
               for m in caplog.messages)


@pytest.mark.parametrize("sibling", ["kept", "evicted"])
def test_divergence_point_past_the_cancel(dense, small, monkeypatch, sibling):
    """Round 2, R2-1/R2-3: cancelled BEFORE the divergence point d (the
    partial entry holds no state there) and retried, the sibling still
    cached or evicted meanwhile: the retry takes a checkpoint at d anyway
    (the partial's carried `diverge` goes into `want`) and keeps exactly the
    uncancelled run's checkpoints. A stride of 64 and a sibling diverging
    without a message start, so nothing else puts a checkpoint at d."""
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_TAIL", 16)
    monkeypatch.setattr(worker, "PROMPT_CACHE_STRIDE", 64)
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_FRESH_TOKENS", 64)
    ids = agentic(420)
    starts = [j for j, t in enumerate(ids) if t == IM_START]
    split = starts[3] + 56
    sib = ids[:split] + prompts([30], 421)[0] + GEN            # no mark at the divergence
    at = below(ids, split)
    ref = scheduler(dense)
    run(ref, [request(sib, 2, "s")])
    run(ref, [request(ids, 2, "x")])
    ref_e = ref.eng._pcache.entries[-1]
    assert at in ref_e.ckpts
    sched = scheduler(dense)
    pc = sched.eng._pcache
    run(sched, [request(sib, 2, "s")])
    x = request(ids, 2, "x")
    run(sched, [x], on_tick=cancel_after(x, 1))
    (p,) = [e for e in pc.entries if e.partial]
    assert len(p.ids) < at and p.diverge == {at} and at not in p.ckpts
    if sibling == "evicted":
        pc.entries = [p]
        pc._recount(added=False)
    y = request(ids, 2, "y")
    assert run(sched, [y])["y"] == cold(dense, ids, 2, "y")
    (full,) = [e for e in pc.entries if e.ids == ids[:-1]]
    assert sorted(full.ckpts) == sorted(ref_e.ckpts)
    assert check_cold(pc, Cold(dense))


def test_system_block_strides_thin_before_the_marks(dense, small, monkeypatch):
    """Round 2, R2-2: the system block's strides are unprotected in the
    fresh policy: with more of them than the cap leaves room for, thinning
    drops strides, never the end of the system block or the last message
    starts (fresh[1] = the marks only). The last starts are a few tokens
    apart and the strides 32: were the strides protected too, the closest
    pairs (the marks) would go first."""
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_TAIL", 0)
    monkeypatch.setattr(worker, "PROMPT_CACHE_STRIDE", 32)
    monkeypatch.setattr(worker, "PROMPT_CACHE_CKPTS", 5)
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_FRESH_TOKENS", 64)
    ids = agentic(450, turns=2, system=150, user0=2, reply=4, tool=4)
    fixed, tail, doc = chain_parts(ids)
    sysb = sys_strides(ids)
    assert len(fixed) == 4 and len(sysb) >= 3 and not tail and not doc
    sched, e = entry(dense, ids, 64)
    assert fixed <= set(e.ckpts) <= fixed | sysb and len(e.ckpts) == 5
    assert check_cold(sched.eng._pcache, Cold(dense))
