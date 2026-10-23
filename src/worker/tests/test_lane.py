"""The prefill lane (QWEN_PREFILL_LANE): a short text request is prefilled
between the slabs of a long running prefill instead of waiting for it.

Every answer must be what it is without the lane (and what a cold serial run
gives: hit == cold), the long prefill must be cut into the same slabs, the
lane never runs more than QWEN_PREFILL_LANE_BURST prefills in a row, it
overtakes only when the budget also holds the head, and the entry the long
prefill resumed from is never evicted by the lane's store."""
import pytest

import worker
from conftest import IM_START, prompts, request, run, scheduler, vision_ids, vision_inputs

GEN = [IM_START, 11, 12, 13, 12]


@pytest.fixture
def small(monkeypatch):
    """As test_prompt_cache.small (cache on, slabs of 8, reply KV off), the lane on."""
    monkeypatch.setattr(worker, "PROMPT_CACHE", True)
    monkeypatch.setattr(worker, "PROMPT_CACHE_REPLY", False)
    monkeypatch.setattr(worker, "PREFILL_STEP", 8)
    monkeypatch.setattr(worker, "PREFILL_MIN_STEP", 4)
    monkeypatch.setattr(worker, "PROMPT_CACHE_STRIDE", 16)
    monkeypatch.setattr(worker, "PROMPT_CACHE_CKPTS", 12)
    monkeypatch.setattr(worker, "PROMPT_CACHE_KEEP_TOKENS", 8)
    monkeypatch.setattr(worker, "PROMPT_CACHE_KEEP_WAIT_S", float("inf"))
    monkeypatch.setattr(worker, "PREFILL_LANE", 1)
    monkeypatch.setattr(worker, "PREFILL_LANE_BURST", 4)


def conv(seed):
    """A short conversation's first prompt (stored by a warm-up run) and its
    next turn: the same prompt + a few tokens (a hit, then <= 1 slab)."""
    a = [IM_START] + prompts([20], seed)[0] + [IM_START] + prompts([6], seed + 1)[0] + GEN
    return a, a + prompts([3], seed + 2)[0]


def trace(sched):
    """Record which prefill each prefill tick ran (main / lane) and the
    main prefill's cache offset after each of its slabs."""
    seen, offsets = [], []
    main, lane = sched._advance, sched._advance_lane

    def adv():
        row = sched.task_row
        seen.append(("main", row.req.id))
        main()
        if row.cache is not None:
            offsets.append((row.req.id, row.cache[sched.fa_layers[0]].offset))

    def adv_lane():
        seen.append(("lane", sched.lane_row.req.id))
        lane()
    sched._advance, sched._advance_lane = adv, adv_lane
    return seen, offsets


def cold(lm, ids, n, rid, **kw):
    return run(scheduler(lm, max_batch=1, **kw), [request(ids, n, rid)])[rid]


def scenario(lm, lane, n_short=3, **kw):
    """Warm-up turns, then a long cold prompt; the short next turns arrive
    while it prefills. -> (outputs, scheduler, trace, offsets, joined:
    {id: was a row while the long prompt was still prefilling})."""
    sched = scheduler(lm, **kw)
    convs = [conv(10 + 3 * i) for i in range(n_short)]
    run(sched, [request(a, 2, f"w{i}") for i, (a, _) in enumerate(convs)])
    long = request(prompts([120], 3)[0], 5, "long")
    shorts = [request(b, 6, f"s{i}") for i, (_, b) in enumerate(convs)]
    seen, offsets = trace(sched)
    joined = {}

    def on_tick(t, s):
        if s.task_row is not None and s.task_row.req is long:
            for r in s.rows:
                joined.setdefault(r.req.id, True)
    with pytest.MonkeyPatch.context() as m:
        m.setattr(worker, "PREFILL_LANE", lane)
        got = run(sched, [long] + shorts, arrive={2: shorts}, on_tick=on_tick)
    return got, sched, seen, offsets, joined, convs, shorts


def test_lane_runs_short_hits_beside_a_long_prefill(dense, small):
    got, sched, seen, offs, joined, convs, shorts = scenario(dense, 1)
    off, _, seen0, offs0, joined0, _, _ = scenario(dense, 0)
    assert got == off                                   # every answer as without the lane
    assert got["long"] == cold(dense, prompts([120], 3)[0], 5, "long")
    for i, (_, b) in enumerate(convs):
        assert got[f"s{i}"] == cold(dense, b, 6, f"s{i}"), i
    assert all(r.reused > 0 for r in shorts)            # hits
    # the long prompt is cut into the same slabs, with the lane between them
    longs = [o for o in offs if o[0] == "long"]
    assert longs == [o for o in offs0 if o[0] == "long"] and len(longs) >= 10
    lanes = [x for x in seen if x[0] == "lane"]
    assert {i for _, i in lanes} == {r.id for r in shorts} and not any(x[0] == "lane" for x in seen0)
    first_lane = seen.index(lanes[0])
    assert any(x == ("main", "long") for x in seen[first_lane:])    # the long one goes on after
    assert all(joined.get(r.id) for r in shorts)         # decoding while the long one prefilled
    assert not any(joined0.get(r.id) for r in shorts)    # without the lane: after it
    assert sched.reserved == 0 and sched.lane is None and sched.task is None


def test_lane_burst_never_starves_the_main_prefill(dense, small, monkeypatch):
    monkeypatch.setattr(worker, "PREFILL_LANE_BURST", 2)
    got, sched, seen, offs, joined, convs, shorts = scenario(dense, 1, n_short=5, max_batch=8)
    run_ = 0
    for kind, rid in seen:
        if kind == "lane":
            run_ += 1
            assert run_ <= 2, seen
        else:
            run_ = 0
    assert sum(k == "lane" for k, _ in seen) == 5


def test_lane_overtakes_only_when_the_head_fits_too(dense, small):
    sched = scheduler(dense, max_batch=4)
    a, b = conv(40)
    run(sched, [request(a, 2, "warm")])
    long = request(prompts([120], 41)[0], 3, "long")
    head = request(prompts([60], 42)[0], 3, "head")       # cold, several slabs: no lane
    short = request(b, 3, "short")
    seen, _ = trace(sched)
    assert sched._lane_slabs(head) > 1 and sched._lane_slabs(short) <= 1
    run(sched, [long, head, short], arrive={2: [head, short]})
    assert [rid for k, rid in seen if k == "lane"] == ["short"]
    assert [rid for k, rid in seen if k == "main"].index("head") > 0
    # a budget for the long row + the short one but not the head too: no overtaking
    need = sched._reserve
    long, head, short = (request(prompts([120], 41)[0], 3, "long"),
                         request(prompts([60], 42)[0], 3, "head"), request(b, 3, "short"))
    sched2 = scheduler(dense, max_batch=4, budget=need(long) + need(short) + need(head) - 1)
    run(sched2, [request(a, 2, "warm")])
    seen, _ = trace(sched2)
    run(sched2, [long, head, short], arrive={2: [head, short]})
    assert not any(k == "lane" for k, _ in seen)
    main = [rid for k, rid in seen if k == "main"]
    assert main.index("head") < main.index("short")      # FIFO as without the lane


def test_eligible_head_that_does_not_fit_blocks_the_lane(dense, small):
    """The first eligible request is the only candidate (FIFO among lane
    requests): it does not fit, nothing behind it is taken."""
    a, b = conv(50)
    c, d = conv(53)
    long = request(prompts([120], 51)[0], 3, "long")
    big = request(b, 400, "big")                           # eligible, reserves a lot
    small_ = request(d, 3, "small")
    sched = scheduler(dense, max_batch=4)
    run(sched, [request(a, 2, "w0"), request(c, 2, "w1")])
    sched.budget = sched._reserve(long) + sched._reserve(small_) + 1
    seen, _ = trace(sched)
    run(sched, [long, big, small_], arrive={2: [big, small_]})
    assert not any(k == "lane" for k, _ in seen)


def test_vision_main_text_lane_and_no_vision_in_the_lane(dense, small):
    def vreq(rid):
        ids = vision_ids(20)
        r = request(ids, 8, rid)
        r.images, r.vision = ["img.png"], vision_inputs(ids)
        return r
    one = run(scheduler(dense, max_batch=1), [vreq("v")])["v"]
    t_ids = prompts([6], 60)[0]                           # cold, one slab
    ref = cold(dense, t_ids, 8, "t")
    sched = scheduler(dense)
    seen, _ = trace(sched)
    v, t, v2 = vreq("v"), request(t_ids, 8, "t"), vreq("v2")
    assert sched._lane_slabs(v2) is None
    got = run(sched, [v, t, v2], arrive={2: [t, v2]})
    assert got["v"] == one and got["v2"][0] == one[0] and got["t"] == ref
    assert [rid for k, rid in seen if k == "lane"] == ["t"]


def test_lane_store_never_evicts_the_main_prefills_parent(dense, small, monkeypatch):
    """The long prefill resumed from an entry (a hit, many slabs left); the
    lane's store must keep that entry (its pieces are held until the long
    prefill stores) whatever QWEN_PROMPT_CACHE_GB says."""
    sched = scheduler(dense)
    pc = sched.eng._pcache
    base = [IM_START] + prompts([40], 70)[0] + [IM_START] + prompts([4], 71)[0]
    run(sched, [request(base, 2, "base")])
    (parent,) = pc.entries
    long = request(base + prompts([90], 72)[0], 3, "long")
    short = request(prompts([6], 73)[0], 3, "short")
    monkeypatch.setattr(worker, "PROMPT_CACHE_GB", 1e-9)  # every store evicts all it may
    kept = []

    def on_tick(t, s):
        if s.task_row is not None and s.task_row.req is long and s.task_row.parent is not None:
            kept.append(any(e is parent for e in pc.entries))
    seen, _ = trace(sched)
    run(sched, [long, short], arrive={2: [short]}, on_tick=on_tick)
    assert [rid for k, rid in seen if k == "lane"] == ["short"]
    assert long.reused > 0 and kept and all(kept)
