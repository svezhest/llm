"""Preemption instead of the "length" cut (QWEN_PREEMPT): a row that would
outgrow the batch's headroom goes to the prompt cache and back to the head of
the queue, and resumes where it stopped. The client must see one stream: the
same tokens (greedy and seeded sampling: the key goes on), penalties,
_Control's counters, a grammar row, the thinking budget's forced tokens and
logprobs as without the preemption; Usage keeps the first prefill's numbers.

The CPU fp32 model: a batched step matches a lone one to ~1e-6 and the
resumed row's KV is the row's own (copied bits), so token streams are equal."""
import logging

import pytest

import worker
from conftest import FakeTok, prompts, request, scheduler

NL, CLOSE, OPEN = 11, 12, 13


@pytest.fixture
def pre(monkeypatch):
    monkeypatch.setattr(worker, "PROMPT_CACHE", True)
    monkeypatch.setattr(worker, "PROMPT_CACHE_REPLY", True)
    monkeypatch.setattr(worker, "PREFILL_STEP", 8)
    monkeypatch.setattr(worker, "PREFILL_MIN_STEP", 4)
    monkeypatch.setattr(worker, "RESERVE_TOKENS", 4)     # rows outgrow their reservation at once
    monkeypatch.setattr(worker, "PREEMPT", 2)
    monkeypatch.setattr(worker, "DEBUG_NO_SYNC", True)      # no wait inside the steps


def drive(sched, reqs, arrive=None, on_tick=None, max_ticks=5000):
    """conftest.run, keeping every event (logprobs included)."""
    arrive = dict(arrive or {})
    later = {id(r) for rs in arrive.values() for r in rs}
    for r in reqs:
        if id(r) not in later:
            sched.submit(r)
    out = {r.id: [] for r in reqs}
    for t in range(max_ticks):
        for r in arrive.pop(t, []):
            sched.submit(r)
        if on_tick is not None:
            on_tick(t, sched)
        if not (sched.rows or sched.task or sched.lane or sched.waiting) and not arrive:
            return out
        sched.tick()
        for r in reqs:
            while not r.out.empty():
                out[r.id].append(r.out.get())
    raise AssertionError("scheduler did not finish")


def force(sched, rid, times=1):
    """Make the growth of row `rid` fail the headroom (as a too-small budget
    would) the first `times` times it outgrows its reservation (after about
    RESERVE_TOKENS tokens): the pre-step check (ahead) -> preempt, the
    step's -> the cut."""
    orig, n = sched._regrow, [0]

    def regrow(row, ahead=0):
        if (row.req.id == rid and n[0] < times
                and row.pos + ahead >= row.tokens and sum(not r.done for r in sched.rows) > 1):
            n[0] += 1
            if ahead:
                return False
            sched._finish(row, "length")
            return False
        return orig(row, ahead)
    sched._regrow = regrow
    return n


def tokens(evs):
    return [e[1] for e in evs if e[0] == "tok"]


def pair(make, at=6, times=1, max_batch=4, **kw):
    """(reference: the requests together, no preemption), (preempted run)."""
    with pytest.MonkeyPatch.context() as m:
        m.setattr(worker, "RESERVE_TOKENS", at)
        pens = {}

        def on_tick(t, s):                  # the first request's penalty state, as it ends
            for r in s.rows:
                if r.req.id == "a" or r.req.id == "think":
                    pens[s] = r.pen
        with pytest.MonkeyPatch.context() as m0:
            m0.setattr(worker, "PREEMPT", 0)
            s0 = scheduler(dense_lm[0], max_batch=max_batch, **kw)
            ref = drive(s0, make(), on_tick=on_tick)
        sched = scheduler(dense_lm[0], max_batch=max_batch, **kw)
        reqs = make()
        hits = force(sched, reqs[0].id, times)
        got = drive(sched, reqs, on_tick=on_tick)
    if s0 in pens and sched in pens:        # the counts went on (not restarted)
        assert worker.mx.array_equal(pens[s0].counts, pens[sched].counts).item()
    return ref, got, reqs, sched, hits


dense_lm = [None]


@pytest.fixture(autouse=True)
def _lm(dense):
    dense_lm[0] = dense


def two(first_kw=None, n=24):
    ps = prompts([20, 14, 9], 31)
    kw = first_kw or {}

    def make():
        return [request(ps[0], n, "a", **kw), request(ps[1], 40, "b"),
                request(ps[2], 30, "c", temperature=0.8, seed=3)]
    return make


@pytest.mark.parametrize("kw", [
    {},
    {"temperature": 0.9, "top_p": 0.9, "top_k": 40, "seed": 7},
    {"presence_penalty": 4.0, "frequency_penalty": 3.0, "repetition_penalty": 1.3},
    {"logprobs": True, "top_logprobs": 3},
    {"ignore_eos": True, "min_tokens": 10},
], ids=["greedy", "seeded", "penalties", "logprobs", "ignore_eos"])
def test_preempted_stream_is_the_uninterrupted_one(pre, kw, caplog):
    caplog.set_level(logging.INFO, logger="qwen.worker")
    ref, got, reqs, sched, hits = pair(two(kw))
    a = reqs[0]
    assert hits[0] == 1 and a.preempts == 1
    assert any("preempted at" in m and "KV kept" in m for m in caplog.messages)
    assert any("prompt cache HIT" in m and "id=a " in m for m in caplog.messages)
    for k in ref:
        assert tokens(got[k]) == tokens(ref[k]), k
        assert got[k][-1] == ref[k][-1], k
    if kw.get("logprobs"):
        for x, y in zip(got["a"], ref["a"]):
            if x[0] == "tok":
                assert abs(x[3][0] - y[3][0]) < 1e-4 and [t for t, _ in x[3][1]] == [t for t, _ in y[3][1]]
    assert len(tokens(got["a"])) == 24 and got["a"][-1] == ("done", "length")
    # Usage: the prompt as sent, the first prefill's cached tokens
    assert a.prompt_tokens == 20 and a.prompt_ids == prompts([20, 14, 9], 31)[0]
    assert a.reused == 0 and a.queue_ms >= 0
    assert sched.reserved == 0 and not sched.waiting


def test_preempted_grammar_and_thinking_budget_rows(pre):
    from mlx_vlm.utils import ThinkingBudgetCriteria
    from test_scheduler import AllowAll
    tok = FakeTok({"</think>": [CLOSE], "<think>": [OPEN], "\n": [NL]})
    ps = prompts([12, 14, 9], 41)

    def make():
        crit = ThinkingBudgetCriteria(tokenizer=tok, thinking_budget=5, thinking_end_token="</think>",
                                      thinking_start_token="<think>", enable_thinking=True,
                                      prompt_preopens_thinking=True)
        return [request(ps[0] + [OPEN], 20, "think", thinking=crit, guide=AllowAll()),
                request(ps[1], 30, "b"), request(ps[2], 30, "c", bad_words=["ab"])]
    for at in (3, 5, 6):          # before, at, after the forced \n</think>
        ref, got, reqs, sched, hits = pair(make, at=at)
        assert hits[0] == 1 and reqs[0].preempts == 1, at
        assert got == ref, at
        assert reqs[0].guide.seen == tokens(got["think"]), at
        assert tokens(got["think"])[6:8] == [NL, CLOSE]


def test_preempt_twice_then_cut(pre, monkeypatch):
    """QWEN_PREEMPT = 1: the second time the row is cut with "length" (the
    old behaviour, no livelock)."""
    monkeypatch.setattr(worker, "PREEMPT", 1)
    ref, got, reqs, sched, hits = pair(two(), at=6, times=2)
    a = tokens(got["a"])
    assert reqs[0].preempts == 1 and hits[0] == 2
    assert got["a"][-1] == ("done", "length") and 6 <= len(a) < 24
    assert a == tokens(ref["a"])[:len(a)]
    assert tokens(got["b"]) == tokens(ref["b"])


def test_lone_row_and_ineligible_rows_are_not_preempted(pre, dense):
    sched = scheduler(dense)
    r = request(prompts([10], 51)[0], 30, "solo")
    hits = force(sched, "solo")
    out = drive(sched, [r])
    assert hits[0] == 0 and r.preempts == 0 and len(tokens(out["solo"])) == 30
    # the eligibility rules, on a row of a running batch
    sched = scheduler(dense)
    reqs = [request(p, 30, f"x{i}") for i, p in enumerate(prompts([10, 10], 52))]
    seen = []

    def on_tick(t, s):
        if len(s.rows) == 2 and not seen and not any(r.first for r in s.rows):
            row = s.rows[0]
            seen.append(s._preemptable(row))
            row.req.preempts = worker.PREEMPT
            seen.append(s._preemptable(row))
            row.req.preempts = 0
            fed, row.fed = row.fed, None             # prompt_cache "read" / cache off
            seen.append(s._preemptable(row))
            row.fed = fed
    drive(sched, reqs, on_tick=on_tick)
    assert seen == [True, False, False]


def test_real_budget_preempts_instead_of_length(pre, dense, caplog):
    """No forcing: a budget that holds both rows at admission but not the
    first one's growth. Without QWEN_PREEMPT it ends with "length"; with it
    it is preempted, resumes after the other row's end and answers in full."""
    caplog.set_level(logging.INFO, logger="qwen.worker")
    ps = prompts([20, 14], 61)

    def make():
        return [request(ps[0], 200, "a"), request(ps[1], 12, "b")]
    probe = scheduler(dense)
    ra, rb = make()
    room = probe._reserve(ra) + probe._reserve(rb)
    big = probe._bytes(len(ps[0]) + 201) - probe._reserve(ra)
    budget = room + big // 2
    full = drive(scheduler(dense), make())
    with pytest.MonkeyPatch.context() as m:
        m.setattr(worker, "PREEMPT", 0)
        m.setattr(worker, "PROMPT_CACHE", False)         # nothing else in the budget
        cut = drive(scheduler(dense, budget=budget), make())
    assert cut["a"][-1] == ("done", "length") and len(tokens(cut["a"])) < 200
    sched = scheduler(dense, budget=budget)
    reqs = make()
    got = drive(sched, reqs)
    assert reqs[0].preempts >= 1
    assert tokens(got["a"]) == tokens(full["a"]) and len(tokens(got["a"])) == 200
    assert tokens(got["b"]) == tokens(full["b"])
    assert sched.reserved == 0
