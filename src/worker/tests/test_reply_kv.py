"""Reply KV in the prompt cache (QWEN_PROMPT_CACHE_REPLY): when a request ends
its entry is extended by the tokens the model consumed while decoding, so
the next turn (prompt + reply + a new message) resumes after the reply.

Numerics: the reply's KV / end state come from decode-shaped calls (1 token
per forward). On the CPU these are NOT bit-identical to a cold prefill slab
of the same tokens: the GDN recurrence over a T-token slab rounds
differently from T one-token steps from layer 0 on (fp32: ~1e-9 in the
state, ~1e-6 in the KV; 8-bit / fp16 scales: ~4e-6 in the state, ~1 fp16 ulp
in the KV; the 8-bit quantized_matmul itself is row-exact on the CPU, the
fp32 matmuls are not). So hits through a reply are compared to cold runs by
closeness of the caches and by the greedy tokens, not bit for bit."""
import logging

import mlx.core as mx
import numpy as np
import pytest

import worker
from conftest import IM_START, FakeTok, prompts, request, run, scheduler, view

A, NL, TH, NN, THC, IM_END = 11, 12, 13, 14, 15, 16
GEN = [IM_START, A, NL, TH, NN, THC, NN]     # <|im_start|>assistant\n<think>\n\n</think>\n\n (thinking off)


@pytest.fixture
def small(monkeypatch):
    """As test_prompt_cache.small, with the reply KV on."""
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


@pytest.fixture(params=["dense", "dense_fused"])
def lm(request):
    return request.getfixturevalue(request.param)


def first(seed, system=24, user=10):
    return [IM_START] + prompts([system], seed)[0] + [IM_START] + prompts([user], seed + 1)[0] + GEN


def nxt(prompt, reply, seed, user=12, eos=True):
    """The next turn as the template renders it: the reply, <|im_end|>\\n, a
    new user message, the generation prompt. `eos` False: the reply already
    ends with the stop token the model emitted."""
    return prompt + reply + ([IM_END] if eos else []) + [NL, IM_START] + prompts([user], seed)[0] + GEN


def below(ids, c):
    b = [x for x in worker._bounds(ids, IM_START) if x <= c and x < len(ids) - 1]
    return max(b, default=0)


def off_run(monkeypatch, lm, ids, n, rid, **kw):
    """A fresh worker with the reply KV off (the reference: a cold prefill)."""
    with monkeypatch.context() as m:
        m.setattr(worker, "PROMPT_CACHE_REPLY", False)
        sched = scheduler(lm, max_batch=1, **kw)
        r = request(ids, n, rid)
        return run(sched, [r])[rid], sched, r


def cold_cache(monkeypatch, lm, ids, **kw):
    """The cold prefill's caches over `ids` (the entry a fresh worker stores
    for ids + [x], restored at its end)."""
    _, sched, _ = off_run(monkeypatch, lm, list(ids) + [1], 1, "cold", **kw)
    (e,) = sched.eng._pcache.entries
    assert e.ids == list(ids)
    return sched.eng._pcache.restore(e, len(ids))


def close(a, b, lm, tol=None):
    """Same shapes, and the values within decode-vs-prefill rounding
    (relative to each array's largest value)."""
    if tol is None:
        tol = 1e-5 if lm.model.embed_tokens.weight.dtype == mx.float32 else 1e-2
    x, y = view(a), view(b)
    assert len(x) == len(y)
    for u, v in zip(x, y):
        assert u.shape == v.shape and u.dtype == v.dtype
        if u.dtype in (mx.uint32,):
            continue                                  # packed quantized words: the scales say it
        d = mx.abs(u.astype(mx.float32) - v.astype(mx.float32)).max().item()
        assert d <= tol * max(mx.abs(v.astype(mx.float32)).max().item(), 1e-3), d
    return True


def test_reply_extends_the_entry_and_the_next_turn_resumes_after_it(lm, small, monkeypatch, caplog):
    """A "length" row fed every token it sent: the entry becomes prompt +
    reply (one entry: the prompt's absorbed, its end state dropped), the
    next turns resume right after each reply (cached_tokens counts it) and
    answer as a cold worker; the reply entry's caches are the cold ones
    within rounding."""
    caplog.set_level(logging.INFO, logger="qwen.worker")
    sched = scheduler(lm)
    pc = sched.eng._pcache
    p = first(10)
    turns = []
    for t in range(3):
        r = request(p, 5 + t, f"t{t}")
        got = run(sched, [r])[r.id]
        toks = got[0]
        assert got == off_run(monkeypatch, lm, p, 5 + t, r.id)[0]       # == cold
        assert got[2] == ("done", "length") and len(toks) == 5 + t
        (e,) = pc.entries
        assert e.ids == p + toks and e.reply == len(p) - 1 and e.end is not None
        assert len(p) - 1 not in e.ckpts                                # the prompt's end state went
        assert all(c < len(p) - 1 for c in e.ckpts)
        assert r.reused == (0 if t == 0 else len(turns[-1]))           # after the previous reply
        assert close(pc.restore(e, len(e.ids)), cold_cache(monkeypatch, lm, e.ids), lm)
        turns.append(e.ids)
        p = nxt(p, toks, 20 + t)
    assert (pc.hits, pc.misses) == (2, 1)
    assert pc.seq == 6                                                 # 2 ticks a turn, as without
    kept = [m for m in caplog.messages if "prompt cache reply kept" in m]
    assert len(kept) == 3 and all("absorbed=1" in m for m in kept)
    assert pc.bytes == pc.entries[0].nbytes


@pytest.mark.parametrize("sync", [False, True], ids=["step-ahead", "sync"])
def test_stop_token_is_in_the_entry_only_when_fed(dense, small, monkeypatch, sync):
    """A row stopping on a token: the step-ahead already fed it (it is in
    the entry and the next turn resumes after it), a sync row never did."""
    p = first(30)
    probe = off_run(monkeypatch, dense, p, 10, "probe")[0][0]
    k = next(i for i in range(3, 10) if probe[i] not in probe[:i])
    stop = probe[k]

    class AllowAll:                                   # a grammar that masks nothing: sync mode
        def consume(self, tok):
            pass

        def sample(self, logits, sampler):
            return sampler(logits)
    sched = scheduler(dense)
    pc = sched.eng._pcache
    r = request(p, 10, "a", guide=AllowAll() if sync else None, stop_token_ids=[stop])
    got = run(sched, [r])["a"]
    assert got[0] == probe[:k] and got[2] == ("done", "stop")
    (e,) = pc.entries
    fed = probe[:k] if sync else probe[:k + 1]
    assert e.ids == p + fed and e.reply == len(p) - 1
    q = nxt(p, probe[:k] + [stop], 31, eos=False)   # the template puts the stop token back
    r2 = request(q, 6, "b")
    assert run(sched, [r2])["b"] == off_run(monkeypatch, dense, q, 6, "b")[0]
    assert r2.reused == len(p) + len(fed)


@pytest.mark.parametrize("think", [True, False], ids=["think-ids-known", "unknown"])
def test_reasoning_replies_are_not_stored(dense, small, caplog, think):
    """Thinking on (the prompt pre-opens <think>; the budget forces
    \\n</think> into the fed tokens): clients rarely send the reasoning back,
    so the reply is not stored and the prompt entry (its end state: a
    regenerate's full hit) stays. The same run with the think ids unknown to
    the worker: stored, the forced tokens included (what was fed)."""
    from mlx_vlm.utils import ThinkingBudgetCriteria
    caplog.set_level(logging.INFO, logger="qwen.worker")
    tok = FakeTok({"</think>": [THC], "<think>": [TH], "\n": [NL]})
    crit = ThinkingBudgetCriteria(tokenizer=tok, thinking_budget=3, thinking_end_token="</think>",
                                  thinking_start_token="<think>", enable_thinking=True,
                                  prompt_preopens_thinking=True)
    p = [IM_START] + prompts([20], 40)[0] + [IM_START, A, NL, TH]
    sched = scheduler(dense)
    sched.eng._think = [TH, THC] if think else None
    toks = run(sched, [request(p, 9, "a", thinking=crit)])["a"][0]
    assert toks[4:6] == [NL, THC], toks
    (e,) = sched.eng._pcache.entries
    if think:
        assert e.ids == p[:-1] and e.reply is None and e.end is not None
        assert any("id=a prompt cache reply not kept: reasoning" in m for m in caplog.messages)
    else:
        assert e.ids == p + toks


def test_thinking_off_replies_are_stored(dense, small):
    """The thinking-off header (<think>\n\n</think>\n\n) closes the reasoning
    in the prompt: with the think ids known, a reply without think tags is
    stored."""
    sched = scheduler(dense)
    sched.eng._think = [TH, THC]
    p = first(45)
    toks = run(sched, [request(p, 6, "a")])["a"][0]
    assert TH not in toks and THC not in toks
    (e,) = sched.eng._pcache.entries
    assert e.ids == p + toks and e.reply == len(p) - 1


def test_thinking_on_reply_without_think_ids_is_not_stored(dense, small, caplog):
    """Thinking on (the prompt pre-opens <think>), no budget, no think id in
    what the model generated: the whole reply is reasoning as _GenState
    splits it (no </think> yet), so it is skipped all the same."""
    caplog.set_level(logging.INFO, logger="qwen.worker")
    p = [IM_START] + prompts([20], 42)[0] + [IM_START, A, NL, TH]
    sched = scheduler(dense)
    sched.eng._think = [TH, THC]
    toks = run(sched, [request(p, 6, "a")])["a"][0]
    assert TH not in toks and THC not in toks, toks
    (e,) = sched.eng._pcache.entries
    assert e.ids == p[:-1] and e.reply is None and e.end is not None
    msg = [m for m in caplog.messages if "id=a prompt cache reply not kept: reasoning" in m]
    assert len(msg) == 1 and "6 of 6 generated tokens" in msg[0], msg


def test_think_id_in_the_content_with_thinking_off_is_stored(dense, small, monkeypatch):
    """Thinking off (the generation prompt closes </think>), and the model
    emits the <think> id inside its content (an answer that mentions the
    literal `<think>`): content, not a reasoning section (_GenState: plain
    text), so the reply is stored and the next turn resumes after it. The
    worker's <think> id is set to a token the greedy reply holds (the tiny
    model never emits a chosen id on its own)."""
    p = first(47)
    probe = off_run(monkeypatch, dense, p, 8, "probe")[0][0]
    x = next(t for t in probe[1:] if t not in (THC, NN, IM_END, TH))
    sched = scheduler(dense)
    sched.eng._think = [x, THC]
    assert not worker._GenState.for_prompt(p, sched.eng._think).in_thinking
    toks = run(sched, [request(p, 8, "a")])["a"][0]
    assert toks == probe and x in toks
    (e,) = sched.eng._pcache.entries
    assert e.ids == p + toks and e.reply == len(p) - 1
    q = nxt(p, toks, 48)
    r2 = request(q, 6, "b")
    assert run(sched, [r2])["b"] == off_run(monkeypatch, dense, q, 6, "b")[0]
    assert r2.reused == len(p) + len(toks)                         # after the reply


def test_reply_retokenized_differently_falls_back(dense, small, monkeypatch):
    """The next prompt holds other ids than the reply (re-tokenized): the hit
    stops at the last slab end before the reply, as with the knob off, and
    the stale reply entry is absorbed by the new one (its checkpoints kept)."""
    p = first(50)
    sched = scheduler(dense)
    pc = sched.eng._pcache
    toks = run(sched, [request(p, 6, "a")])["a"][0]
    other = toks[:2] + [(toks[2] + 1) % 400 + 20] + toks[3:]
    q = nxt(p, other, 51)
    r = request(q, 6, "b")
    assert run(sched, [r])["b"] == off_run(monkeypatch, dense, q, 6, "b")[0]
    with monkeypatch.context() as m:                  # the reuse without the knob
        m.setattr(worker, "PROMPT_CACHE_REPLY", False)
        s0 = scheduler(dense)
        run(s0, [request(p, 6, "a")])
        r0 = request(q, 6, "b")
        run(s0, [r0])
    assert r.reused == r0.reused and 0 < r.reused < len(p) - 1
    (e,) = pc.entries                                 # the stale reply entry went
    assert e.ids[: len(q)] == q and e.reply == len(q) - 1


def test_regenerate_resumes_at_the_last_message_start(dense, small, monkeypatch):
    """The same prompt again: the prompt's end state went with the extension,
    so it resumes at the last message start (not at len - 1), answers as
    cold, and its reply replaces the older one."""
    p = first(60)
    sched = scheduler(dense)
    pc = sched.eng._pcache
    a = run(sched, [request(p, 6, "a")])["a"]
    (old,) = pc.entries
    ends = set(worker._bounds(p, IM_START)) | {len(p) - 1}
    last = max(i for i, t in enumerate(p) if t == IM_START)
    r = request(p, 4, "b")
    b = run(sched, [r])["b"]
    assert b == off_run(monkeypatch, dense, p, 4, "b")[0] and b[0] == a[0][:4]
    assert r.reused == max(c for c in old.ckpts if c in ends)
    assert below(p, last) <= r.reused < len(p) - 1
    (e,) = pc.entries
    assert e.ids == p + b[0] and e.reply == len(p) - 1


def test_reply_not_kept_over_the_headroom(dense, small, caplog):
    """The extension alone over the batch's headroom: not kept, the prompt
    entry stays as it was (nothing absorbed, evicted or copied)."""
    caplog.set_level(logging.INFO, logger="qwen.worker")
    sched = scheduler(dense)
    pc = sched.eng._pcache
    seen = {}

    def squeeze(t, s):
        if pc.entries and "e" not in seen:            # after the prefill's store
            seen["e"] = pc.entries[0]
            s.budget = pc.entries[0].nbytes           # the prompt entry fits, + the reply not
    p = first(70)
    got = run(sched, [request(p, 6, "a")], on_tick=squeeze)["a"]
    assert got[2] == ("done", "length")
    assert pc.entries == [seen["e"]] and seen["e"].ids == p[:-1] and seen["e"].reply is None
    assert pc.evictions == 1 and pc.bytes == seen["e"].nbytes
    assert any(m.startswith("prompt cache reply not kept: over the batch's headroom")
               for m in caplog.messages)


def test_cancelled_request_stores_nothing_extra(dense, small):
    sched = scheduler(dense)
    pc = sched.eng._pcache
    p = first(80)
    r = request(p, 30, "a")

    def cancel(t, s):
        if s.rows and s.rows[0].n >= 3:
            r.cancel.set()
    got = run(sched, [r], on_tick=cancel)["a"]
    assert got[2] == ("done", "cancelled") and 3 <= len(got[0]) < 30
    (e,) = pc.entries
    assert e.ids == p[:-1] and e.reply is None


def test_entry_gone_stores_nothing(dense, small, caplog):
    """The prompt entry evicted while the row decoded: no extension."""
    caplog.set_level(logging.INFO, logger="qwen.worker")
    sched = scheduler(dense)
    pc = sched.eng._pcache

    def drop(t, s):
        if pc.entries:
            pc.evict()
    run(sched, [request(first(90), 6, "a")], on_tick=drop)
    assert pc.entries == []
    assert any("reply not kept: its prompt entry is gone" in m for m in caplog.messages)


class Owned:
    """Spies on _reply_cache: keeps alive (no buffer address reused) every
    stacked recurrent-state array and the finishing row's own state arrays
    (at B = 1 right after a shrink: _gdn_split's views of the old stack) ->
    check(e): e's end state shares no buffer with any of them."""

    def __init__(self, sched):
        self.sched, self.arrs, self.orig = sched, [], sched._reply_cache
        sched._reply_cache = self

    def __call__(self, row, i):
        s = self.sched
        if s.gdn is not None:
            self.arrs += [a for c in s.gdn.values() for a in c.cache]
        self.arrs += [a for c in row.cache if hasattr(c, "cache") for a in c.cache if a is not None]
        return self.orig(row, i)

    def check(self, e):
        ends = [np.asarray(a) for layer in e.end for a in layer if a is not None]
        seen = [np.asarray(a) for a in self.arrs]
        assert ends and seen
        assert all(a.shape[0] == 1 for a in ends)
        return not any(np.may_share_memory(x, y) for x in ends for y in seen)


def test_batched_replies_match_solo(dense_fused, small, monkeypatch):
    """Two conversations decoding together (stacked GDN state: the finished
    row's state is gathered out of the stack): each entry is the one a lone
    run stores, bit for bit (CPU 8-bit rows are row-exact), its end state a
    buffer of one row; both next turns resume after their replies."""
    lm = dense_fused
    ps = [first(100), first(110, system=30, user=7)]
    solo = []
    for i, p in enumerate(ps):
        s = scheduler(lm, max_batch=1)
        toks = run(s, [request(p, 5 + 3 * i, f"c{i}")])[f"c{i}"][0]
        (e,) = s.eng._pcache.entries
        solo.append((toks, s.eng._pcache.restore(e, len(e.ids))))
    sched = scheduler(lm, max_batch=4)
    pc = sched.eng._pcache
    own = Owned(sched)
    rs = [request(p, 5 + 3 * i, f"c{i}") for i, p in enumerate(ps)]
    got = run(sched, rs)
    assert max(r.batch_mean for r in rs) > 1 and len(pc.entries) == 2
    for i, p in enumerate(ps):
        toks, caches = solo[i]
        assert got[f"c{i}"][0] == toks
        e = next(x for x in pc.entries if x.ids[: len(p)] == p)
        assert e.ids == p + toks
        assert own.check(e)                           # a gather, never a view of the stack
        x, y = view(pc.restore(e, len(e.ids))), view(caches)
        assert all(mx.array_equal(u, v).item() for u, v in zip(x, y))
    qs = [nxt(p, got[f"c{i}"][0], 120 + i) for i, p in enumerate(ps)]
    rs = [request(q, 5, f"n{i}") for i, q in enumerate(qs)]
    out = run(sched, rs)
    for i, q in enumerate(qs):
        assert rs[i].reused == len(ps[i]) + len(got[f"c{i}"][0])
        assert out[f"n{i}"] == off_run(monkeypatch, lm, q, 5, f"n{i}")[0]


def test_lone_row_after_a_shrink_owns_its_state(dense_fused, small):
    """B = 2 -> 1: X stops at the end of step j, Y read its last token in the
    same step and ends with "length" at the top of the next one, before any
    B = 1 step replaced its state, which is still _gdn_split's view of the
    2-row stack. Its entry's end state must be a buffer of its own."""
    lm = dense_fused
    px, py = first(400), first(410, system=30, user=7)

    def go(stop, my, on_tick=None, spy=False):
        sched = scheduler(lm, max_batch=4)
        own = Owned(sched) if spy else None
        x = request(px, 20, "x", **({"stop_token_ids": [stop]} if stop is not None else {}))
        y = request(py, my, "y")
        return sched, own, run(sched, [x, y], on_tick=on_tick)

    snaps = []
    _, _, base = go(None, 20, lambda t, s: snaps.append({r.req.id: r.n for r in s.rows}))
    tx = base["x"][0]
    pick = None
    for a, b in zip(snaps, snaps[1:]):                # a step where both rows read a token
        if {"x", "y"} <= a.keys() and {"x", "y"} <= b.keys() and b["x"] == a["x"] + 1 \
                and b["y"] == a["y"] + 1 and a["x"] >= 2 and tx[a["x"]] not in tx[:a["x"]]:
            pick = (tx[a["x"]], b["y"])
            break
    assert pick is not None
    sched, own, got = go(*pick, spy=True)
    assert got["x"][2] == ("done", "stop") and got["y"][2] == ("done", "length")
    e = next(e for e in sched.eng._pcache.entries if e.ids[: len(py)] == py)
    assert e.reply == len(py) - 1 and e.ids == py + got["y"][0]
    assert own.check(e)


def test_one_async_eval_for_rows_ending_together(dense_fused, small, monkeypatch):
    """Review finding 3: rows ending in the same step queue their copies and
    end states in ONE async_eval (one command buffer between two steps), and
    the host never evaluates them (no mx.eval in the pass)."""
    lm = dense_fused
    px, py = first(600), first(610, system=30, user=7)
    snaps = []
    sched = scheduler(lm, max_batch=4)
    run(sched, [request(px, 20, "x"), request(py, 20, "y")],
        on_tick=lambda t, s: snaps.append({r.req.id: r.n for r in s.rows}))
    b = next(b for a, b in zip(snaps, snaps[1:]) if {"x", "y"} <= a.keys() and {"x", "y"} <= b.keys()
             and b["x"] == a["x"] + 1 and b["y"] == a["y"] + 1)
    sched = scheduler(lm, max_batch=4)
    calls, inside = [], [False]
    real_async, real_eval = mx.async_eval, mx.eval

    def async_eval(*a):
        if inside[0]:
            calls.append("async")
        return real_async(*a)

    def eval_(*a):
        if inside[0]:
            calls.append("eval")
        return real_eval(*a)
    monkeypatch.setattr(worker.mx, "async_eval", async_eval)
    monkeypatch.setattr(worker.mx, "eval", eval_)
    drop = sched._drop_done

    def spy():
        ending = sum(r.done for r in sched.rows)
        inside[0] = ending == 2
        try:
            drop()
        finally:
            inside[0] = False
    sched._drop_done = spy
    got = run(sched, [request(px, b["x"], "x"), request(py, b["y"], "y")])
    assert got["x"][2] == got["y"][2] == ("done", "length")
    assert calls == ["async"]
    assert sum(e.reply is not None for e in sched.eng._pcache.entries) == 2


def test_reply_store_spares_the_running_prefills_parent(dense, small, monkeypatch, caplog):
    """Review finding 1: a reply store runs between the slabs of another
    request's prefill (C, resumed from B's reply entry); with the cache over
    its cap it must not evict C's parent (its pieces stay allocated in C's
    _Inherit: the count would drop, the memory not), as _regrow spares it."""
    caplog.set_level(logging.INFO, logger="qwen.worker")
    sched = scheduler(dense, max_batch=4)
    pc = sched.eng._pcache
    pB = first(200)
    toksB = run(sched, [request(pB, 4, "b")])["b"][0]
    qB = nxt(pB, toksB, 201, user=60)                 # B's next turn: a multi-slab prefill
    d = request(first(300, system=30, user=9), 30, "d")
    c = request(qB, 3, "c")
    seen = []

    def hook(t, s):
        tr = s.task_row
        if tr is not None and tr.req is c and tr.started and tr.parent is not None:
            if any(not r.done for r in s.rows):
                s.resume = float("inf")              # decode only: D ends mid-prefill
                monkeypatch.setattr(worker, "PROMPT_CACHE_GB", 1e-9)   # every store must evict
            else:
                s.resume = 0.0
            seen.append(any(x is tr.parent for x in pc.entries))
    out = run(sched, [d, c], arrive={3: [c]}, on_tick=hook)
    assert out["c"][2] == ("done", "length") and c.reused == len(pB) + len(toksB)
    assert len(seen) >= 2 and all(seen), "the running prefill's parent was evicted"
    assert any("id=d prompt cache reply kept" in m for m in caplog.messages)


def aligned(lm, ps, n=20):
    """max_tokens per prompt so that every row reads its last token in the
    same decode step (they all end with "length" in one _drop_done pass)."""
    snaps = []
    sched = scheduler(lm, max_batch=4)
    run(sched, [request(p, n, f"r{i}") for i, p in enumerate(ps)],
        on_tick=lambda t, s: snaps.append({r.req.id: r.n for r in s.rows}))
    ids = {f"r{i}" for i in range(len(ps))}
    return next(b for a, b in zip(snaps, snaps[1:]) if ids <= a.keys() and ids <= b.keys()
                and all(b[k] == a[k] + 1 for k in ids))


def test_reply_keeps_its_prompts_recency(dense, small, monkeypatch, caplog):
    """Review R2-1: B finished (its reply entry stored, its next turn
    imminent), A still decoding (prefilled after B): under pressure A's entry
    goes first, as without the knob (the most recently prefilled is the
    furthest from its next turn); a reply store does not refresh recency."""
    caplog.set_level(logging.INFO, logger="qwen.worker")
    sched = scheduler(dense, max_batch=4)
    pc = sched.eng._pcache
    b = request(first(700), 6, "b")
    a = request(first(710, system=30, user=9), 40, "a")
    c = request([IM_START] + prompts([6], 720)[0] + GEN, 3, "c")
    state = {}

    def hook(t, s):
        rows = {r.req.id: r for r in s.rows}
        if "c" not in state and "b" not in rows and "a" in rows \
                and any(e.reply is not None for e in pc.entries):
            eb = next(e for e in pc.entries if e.reply is not None)
            ea = next(e for e in pc.entries if e.reply is None)
            assert ea.ids == a.ids[:-1] and eb.ids[: len(b.ids)] == b.ids
            assert pc.victim(pc.entries) is ea          # the order, before any eviction
            state.update(c=t, ea=ea, eb=eb)
            # room for nothing more: C's store must evict one of them
            monkeypatch.setattr(worker, "PROMPT_CACHE_GB", (pc.bytes + 1) / worker.GIB)
            s.submit(c)
    run(sched, [b, a], arrive={2: [a]}, on_tick=hook)       # the hook submits C
    assert "c" in state and c.out.get_nowait()
    assert any(e is state["eb"] for e in pc.entries), "the finished conversation's reply was evicted"
    assert not any(e is state["ea"] for e in pc.entries)
    assert any("id=a prompt cache reply not kept: its prompt entry is gone" in m
               for m in caplog.messages)


def test_three_rows_ending_together_keep_their_replies(dense_fused, small, monkeypatch):
    """Review R2-1 (same pass): three rows ending in one _drop_done pass
    keep all three reply entries when they fit together (a later reply store
    must not take an earlier one of the pass for the most recent entry)."""
    lm = dense_fused
    ps = [first(800), first(810, system=30, user=7), first(820, system=20, user=12)]
    b = aligned(lm, ps)

    def go():
        sched = scheduler(lm, max_batch=4)
        got = run(sched, [request(p, b[f"r{i}"], f"r{i}") for i, p in enumerate(ps)])
        assert all(got[f"r{i}"][2] == ("done", "length") for i in range(3))
        return sched.eng._pcache
    pc = go()
    assert sum(e.reply is not None for e in pc.entries) == 3
    monkeypatch.setattr(worker, "PROMPT_CACHE_GB", pc.bytes / worker.GIB)   # exactly the three
    pc = go()
    assert sum(e.reply is not None for e in pc.entries) == 3 and pc.evictions == 0


def test_failed_async_eval_restores_the_prompt_entries(dense, small, monkeypatch, caplog):
    """Review R2-2: the one async_eval of a pass fails: the reply entries
    go, the prompt entries they absorbed come back, counted as evictions."""
    caplog.set_level(logging.INFO, logger="qwen.worker")
    sched = scheduler(dense)
    pc = sched.eng._pcache
    inside = [False]
    real = mx.async_eval

    def async_eval(*a):
        if inside[0]:
            raise RuntimeError("[metal] out of memory (simulated)")
        return real(*a)
    monkeypatch.setattr(worker.mx, "async_eval", async_eval)
    drop = sched._drop_done

    def spy():
        inside[0] = any(r.done for r in sched.rows)
        try:
            drop()
        finally:
            inside[0] = False
    sched._drop_done = spy
    p = first(900)
    got = run(sched, [request(p, 6, "a")])
    assert got["a"][2] == ("done", "length")
    (e,) = pc.entries
    assert e.ids == p[:-1] and e.reply is None and pc.bytes == e.nbytes
    assert pc.evictions == 1
    assert any("prompt cache reply not kept: queueing the copies failed" in m for m in caplog.messages)


def test_reply_beside_the_spared_parent_over_the_headroom(dense, small, monkeypatch, caplog):
    """Review R2-3: the reply fits the headroom alone, but not beside the
    running prefill's parent (which cannot be evicted): not kept (logged),
    instead of published over the headroom."""
    caplog.set_level(logging.INFO, logger="qwen.worker")
    sched = scheduler(dense, max_batch=4)
    pc = sched.eng._pcache
    pB = first(200)
    toksB = run(sched, [request(pB, 4, "b")])["b"][0]
    d = request(first(300, system=30, user=9), 30, "d")
    c = request(nxt(pB, toksB, 201, user=60), 3, "c")
    seen = {}

    def hook(t, s):
        tr = s.task_row
        rows = [r for r in s.rows if not r.done]
        if tr is not None and tr.req is c and tr.started and tr.parent is not None and rows:
            s.resume = float("inf")                  # decode only: D ends mid-prefill
            (dr,) = rows
            ed = next(e for e in pc.entries if e.ids == d.ids[:-1])
            # headroom after D ends = its prompt entry + C's parent: the
            # extension alone fits (+ its reply), not beside the parent
            s.budget = s.reserved - dr.reserved + pc.bytes_of([ed, tr.parent])
            seen["parent"] = tr.parent
        elif tr is not None and tr.req is c:
            s.resume = 0.0
    run(sched, [d, c], arrive={3: [c]}, on_tick=hook)
    assert "parent" in seen
    assert any(m.startswith("prompt cache reply not kept: over the batch's headroom")
               and "beside the running prefill's entry" in m for m in caplog.messages)
    assert not any("id=d prompt cache reply kept" in m for m in caplog.messages)


def test_prompt_part_absorb_keeps_only_reachable_checkpoints(dense, small, monkeypatch):
    """Review finding 4: a reply entry covered by its prompt part (a
    regenerate) whose checkpoints reach past the new entry's length: only
    those below it move over (the chain policy would hide it: off here)."""
    monkeypatch.setattr(worker, "PROMPT_CACHE_CHAIN_MARKS", -1)
    sched = scheduler(dense)
    pc = sched.eng._pcache
    p = first(500)
    run(sched, [request(p, 8, "a")])
    (r,) = pc.entries
    deep = len(r.ids) - 2                            # inside the reply, past the regenerate's end
    r.ckpts[deep] = r.end                            # (as an absorbed longer prefix entry leaves)
    r.index()
    pc._recount()
    run(sched, [request(p, 3, "b")])
    (e,) = pc.entries
    assert len(e.ids) == len(p) + 3 < deep
    assert all(c < len(e.ids) for c in e.ckpts) and deep not in e.ckpts


def test_quantized_kv(dense, small, monkeypatch):
    """Quantized KV past the start: the extension is quantized like the row,
    the next turn resumes after the reply. A row that crosses the start while
    decoding: its fp16 prompt entry stays (the floor entry), the extension is
    the whole row, quantized."""
    kw = dict(kv_bits=8, kv_start=40)
    sched = scheduler(dense, **kw)
    pc = sched.eng._pcache
    p = first(130, system=40)
    assert len(p) > 40
    toks = run(sched, [request(p, 6, "a")])["a"][0]
    (e,) = pc.entries
    assert e.quantized and e.ids == p + toks and e.low == 40
    q = nxt(p, toks, 131)
    r = request(q, 6, "b")
    assert run(sched, [r])["b"] == off_run(monkeypatch, dense, q, 6, "b", **kw)[0]
    assert r.reused == len(p) + len(toks)

    sched = scheduler(dense, **kw)
    pc = sched.eng._pcache
    p = first(140, system=20, user=8)
    assert len(p) < 40 < len(p) + 12
    toks = run(sched, [request(p, 12, "a")])["a"][0]
    floor, ext = pc.entries
    assert not floor.quantized and floor.ids == p[:-1] and floor.reply is None
    assert ext.quantized and ext.ids == p + toks and ext.reply == len(p) - 1
    # 8-bit KV: a value a rounding away from a level boundary moves a whole
    # level (the scales / biases of its group follow the group's min / max)
    assert close(pc.restore(ext, len(ext.ids)), cold_cache(monkeypatch, dense, ext.ids, **kw), dense,
                 tol=1e-3)


def test_reply_cache_gates(caplog):
    """What _reply_cache refuses (it builds nothing then); a row still in a
    cascade group is the one refusal it logs (the others are no store
    candidates at all, or logged by _store_reply)."""
    caplog.set_level(logging.INFO, logger="qwen.worker")
    s = worker._Scheduler.__new__(worker._Scheduler)
    s.fa_layers, s.gdn = [], None

    class Row:
        why, spec, group, cache = "stop", False, None, []
        req = type("Q", (), {"images": [], "id": "x"})()
    for attr, val in [("why", "cancelled"), ("why", None), ("spec", True), ("group", object())]:
        r = Row()
        setattr(r, attr, val)
        assert s._reply_cache(r, 0) is None
    r = Row()
    r.req = type("Q", (), {"images": ["img"], "id": "x"})()
    assert s._reply_cache(r, 0) is None
    assert s._reply_cache(Row(), 0) == []
    skipped = [m for m in caplog.messages if "prompt cache reply skipped" in m]
    assert skipped == [m for m in skipped if m.startswith("id=x prompt cache reply skipped: "
                                                          "in a cascade group (P=?)")]
    assert len(skipped) == 1
    r = Row()
    r.why, r.group = "cancelled", type("G", (), {"P": 4096})()   # cancelled: no candidate, no line
    assert s._reply_cache(r, 0) is None
    r = Row()
    r.group = type("G", (), {"P": 4096})()
    assert s._reply_cache(r, 0) is None
    assert sum("reply skipped: in a cascade group (P=4096)" in m for m in caplog.messages) == 1


def test_knob_off_keeps_the_prompt_entry(dense, small, monkeypatch):
    monkeypatch.setattr(worker, "PROMPT_CACHE_REPLY", False)
    sched = scheduler(dense)
    p = first(150)
    run(sched, [request(p, 6, "a")])
    (e,) = sched.eng._pcache.entries
    assert e.ids == p[:-1] and e.reply is None
