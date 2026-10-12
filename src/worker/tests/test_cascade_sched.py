"""Cascade groups in the worker (plan Part B, steps B2 + B3; cascade.py itself:
test_cascade.py). Logits level: RowsKV with a step plan -- grouped rows close
to their per-row attention, ungrouped rows of the same batch bit-identical,
release exact. Scheduler level (tiny random models, CPU; QWEN_CASCADE on with
the thresholds lowered): formation around the cold producer, later rows
joining, finished rows leaving, the long-suffix leave rule, dissolution at
B <= 2 (and the continuation after it bitwise equal to a never-grouped row
holding the same cache), the memory rule with its dissolve floor and the
budget guard with the prompt cache's bytes (in bytes, 9B layout, and in the
scheduler), optional releases checked with the group alive and staged one
member per decode step, the accounting, the exclusions (vision, DFlash,
quantized KV, B <= 2), the failure paths (a failed step, a failed join, a
failed tidy), no reference cycles (a dropped group's arrays are freed by
refcounting alone), cancel / "ended at" of a grouped row, two groups in one
batch, no duplicate group on a refused row's prefix, a member resumed deeper
than P, back-to-back joins, the step plan's identity order, the
QWEN_CASCADE_CHECK refusal, the default (on for fp16 KV only) and
QWEN_CASCADE=0 = the per-row code path. The wave and mixed-batch tests run with QWEN_CASCADE_CHECK on and
off (off is production: the check's evals could hide laziness) and compare
the grouped rows' per-step logits with an ungrouped batch (fp32, <= 1e-5)."""
import gc
import logging
import types
import weakref

import mlx.core as mx
import numpy as np
import pytest
from mlx_vlm.models.cache import ArraysCache, KVCache

import cascade
import worker
from conftest import IM_START, Cold, prompts, request, run, scheduler, vision_inputs
from test_scheduler import AllowAll, FakeDraft, no_metal_verify

GEN = [IM_START, 11, 12, 13, 12]          # <|im_start|>assistant\n<think>\n stand-in


REAL_WINS = worker._Scheduler._group_wins


def any_three(self, P, rows):
    return len(rows) >= 3


@pytest.fixture
def casc(monkeypatch):
    """Prompt cache on with 8-token slabs (as test_prompt_cache's `small`),
    cascade on: 3 rows, 16-token prefixes, chunk and stack step 16, check on.
    The memory rule is off (any 3 rows win): the tiny model's 2 FA layers and
    16-token stack steps make a 150-token prefix lose to its rows' caches at
    3 rows (the 9B's 4K prefix wins there); the rule's own tests put
    REAL_WINS back."""
    for k, v in dict(PROMPT_CACHE=True, PREFILL_STEP=8, PREFILL_MIN_STEP=4,
                     PROMPT_CACHE_STRIDE=16, PROMPT_CACHE_CKPTS=12, PROMPT_CACHE_KEEP_TOKENS=8,
                     PROMPT_CACHE_KEEP_WAIT_S=float("inf"), CASCADE=True, CASCADE_MIN_ROWS=3,
                     CASCADE_MIN_PREFIX=16, CASCADE_CHUNK=16, CASCADE_STEP=16,
                     CASCADE_CHECK=True).items():
        monkeypatch.setattr(worker, k, v)
    monkeypatch.setattr(worker._Scheduler, "_group_wins", any_three)
    cascade.clear_traces()
    yield
    cascade.clear_traces()


def chats(n, sys_len=150, seed=60, user=lambda i: 6 + i):
    """n prompts on one system prompt: [<im_start> system][<im_start> user_i][GEN].
    The shared prefix ends at the second <im_start> (a slab end of each)."""
    system = [IM_START] + prompts([sys_len], seed)[0]
    users = prompts([user(i) for i in range(n)], seed + 1)
    return [system + [IM_START] + u + GEN for u in users]


def serial(lm, reqs):
    out = {}
    for r in reqs:
        out.update(run(scheduler(lm, max_batch=1), [r]))
    return out


class Spy:
    """Records group formations / joins / grouped attention calls."""

    def __init__(self, monkeypatch):
        self.builds, self.adds, self.calls, self.groups = [], [], 0, []
        build, add, rows = (worker._Scheduler._group_build, worker._Scheduler._group_add,
                            worker._cascade_rows)

        def spy_build(sched, P, rws):
            self.builds.append((P, [(r.req.id, r.req.reused) for r in rws]))
            return build(sched, P, rws)

        def spy_add(sched, g, row):
            self.adds.append((g.P, row.req.id))
            return add(sched, g, row)

        def spy_rows(*a):
            self.calls += 1
            return rows(*a)
        monkeypatch.setattr(worker._Scheduler, "_group_build", spy_build)
        monkeypatch.setattr(worker._Scheduler, "_group_add", spy_add)
        monkeypatch.setattr(worker, "_cascade_rows", spy_rows)

    def tick(self, t, sched):
        self.groups.append([[r.req.id for r in g.rows] for g in sched.groups])
        check_accounting(sched)


def check_accounting(sched):
    """reserved = live ungrouped rows' + members' own (recurrent state) +
    every group's charge (= _group_bytes of its rows) + the prefilling row."""
    member = sched._member_bytes()
    want = sum(r.reserved for r in sched.rows if not r.done)
    want += sum(g.reserved for g in sched.groups)
    if sched.task_row is not None and not sched.task_row.done:
        want += sched.task_row.reserved
    assert sched.reserved == want
    for g in sched.groups:
        assert g.reserved == sched._group_bytes(g.P, g.rows)
        assert g.pg.nbytes <= g.reserved and g.pg.n == len(g.rows)
        for r in g.rows:
            assert r.group is g and (r.done or r.reserved == member)
            assert all(isinstance(r.cache[i], worker._GroupRef) for i in sched.fa_layers)
    assert (sched.n_groups, sched.n_grouped) == (len(sched.groups),
                                                 sum(len(g.rows) for g in sched.groups))


def one_per_tick(reqs, start=0):
    return {start + i: [r] for i, r in enumerate(reqs)}


def row_of(sched, rid):
    return next((r for r in sched.rows if r.req.id == rid), None)


class Logits:
    """Every row's logits as the scheduler hands them to the sampler, per run
    (start() opens a new run): {request id: [step logits, fp32 numpy]}."""

    def __init__(self, monkeypatch):
        self.runs = []
        logits = worker._Scheduler._logits

        def rec(sched, row, x):
            if self.runs:
                self.runs[-1].setdefault(row.req.id, []).append(np.array(x.astype(mx.float32)))
            return logits(sched, row, x)
        monkeypatch.setattr(worker._Scheduler, "_logits", rec)

    def start(self):
        self.runs.append({})

    def max_diff(self, a=0, b=1):
        x, y = self.runs[a], self.runs[b]
        assert x.keys() == y.keys()
        d = 0.0
        for k in x:
            assert len(x[k]) == len(y[k]), k
            d = max([d] + [float(np.abs(u - v).max()) for u, v in zip(x[k], y[k])])
        return d


class Plans:
    """The step plans _plan returned: (batch size, plan)."""

    def __init__(self, monkeypatch):
        self.seen = []
        plan = worker._Scheduler._plan

        def spy(sched, rows):
            p = plan(sched, rows)
            self.seen.append((len(rows), p))
            return p
        monkeypatch.setattr(worker._Scheduler, "_plan", spy)

    def whole(self):
        """Plans of a batch that is one group: identity order, one slice."""
        return [(B, p) for B, p in self.seen if not p.single and len(p.groups) == 1]


class FakeRow:
    """What the memory rule and the tidy decision read of a row."""

    def __init__(self, pos, tokens, reserved=0, rid="f"):
        self.pos, self.tokens, self.reserved, self.done = pos, tokens, reserved, False
        self.req = types.SimpleNamespace(id=rid)
        self.group = None


def nine_b(lm, monkeypatch):
    """A scheduler whose byte formulas use the 9B layout: fp16 KV 32 KiB per
    token over 8 FA layers, 16 q heads, chunk 1024, stack step 256, 64 MiB
    slack; the real memory rule."""
    s = scheduler(lm, max_batch=16)
    s.kv_fp16 = 32 << 10
    s.fa_layers = list(range(3, 32, 4))
    s.lm = types.SimpleNamespace(args=types.SimpleNamespace(num_attention_heads=16))
    for k, v in dict(CASCADE_CHUNK=1024, CASCADE_STEP=256, CASCADE_SLACK=64 << 20).items():
        monkeypatch.setattr(worker, k, v)
    monkeypatch.setattr(worker._Scheduler, "_group_wins", REAL_WINS)
    return s


GIB = 1 << 30


# ------------------------------------------------------------------ pure rules


def test_knobs():
    assert worker._cascade_knobs({}) == (True, 8, 4096, False)            # on by default
    assert worker._cascade_knobs({"QWEN_CASCADE": "1", "QWEN_CASCADE_MIN_ROWS": "2",
                                  "QWEN_CASCADE_MIN_PREFIX": "100",
                                  "QWEN_CASCADE_CHECK": "1"}) == (True, 3, 100, True)
    assert worker._cascade_knobs({"QWEN_CASCADE": "0", "QWEN_CASCADE_MIN_ROWS": "0",
                                  "QWEN_CASCADE_MIN_PREFIX": "-5"}) == (False, 0, 0, False)
    assert worker._cascade_knobs({"QWEN_CASCADE": "", "QWEN_CASCADE_MIN_ROWS": ""})[:2] == (True, 8)
    for off in ("0", "false", "off", "no", "true", "2"):       # "1" only, as the other switches
        assert worker._cascade_knobs({"QWEN_CASCADE": off})[0] is False


def test_groupref_of_a_gone_row():
    """_GroupRef reads its row's position; a row gone (unreachable: unlink
    clears the entries first) is a clear error, not an AttributeError."""
    row = FakeRow(7, 8)
    ref = worker._GroupRef(row)
    assert ref.offset == 7
    del row
    gc.collect()
    with pytest.raises(RuntimeError, match="outlived its row"):
        ref.offset


def test_default_on_for_fp16_kv_only(dense, monkeypatch):
    """The production default (QWEN_CASCADE unset): on for fp16 KV with the
    prompt cache and max_batch >= 3; off by itself for quantized KV, for
    max_batch < 3 (also when fit_max_batch turns batching off) and without
    the prompt cache."""
    on, rows, prefix, check = worker._cascade_knobs({})
    for k, v in dict(CASCADE=on, CASCADE_MIN_ROWS=rows, CASCADE_MIN_PREFIX=prefix,
                     CASCADE_CHECK=check, PROMPT_CACHE=True).items():
        monkeypatch.setattr(worker, k, v)
    s = scheduler(dense, max_batch=16)
    assert s.cascade and s.cascade_state() == "on: >= 8 rows sharing >= 4096 prompt tokens"
    for kv_bits in (4, 8):
        s = scheduler(dense, max_batch=16, kv_bits=kv_bits)
        assert not s.cascade and "quantized KV" in s.cascade_state()
    for mb in (1, 2):
        s = scheduler(dense, max_batch=mb)
        assert not s.cascade and "max_batch < 3" in s.cascade_state()
    s = scheduler(dense, max_batch=16, budget=1)
    s.fit_max_batch()
    assert s.max_batch == 1 and not s.cascade
    monkeypatch.setattr(worker, "PROMPT_CACHE", False)
    assert not scheduler(dense, max_batch=16).cascade


def test_memory_rule_bytes(dense, casc, monkeypatch):
    """The charge, the rule and the guard in bytes, 9B layout (review
    cascade_b2b3 finding 2's table): rows of prompt P + 150 with the default
    4096 reserved generation tokens."""
    s = nine_b(dense, monkeypatch)
    member, kv = s._member_bytes(), 32 << 10

    def rows(P, n, S, reserve=4096):
        return [FakeRow(P + S, P + S + reserve, s._bytes(P + S + reserve)) for _ in range(n)]

    # _group_bytes term by term: P 8K, 16 rows, S_res 4246 -> 4352 (17 steps)
    st = 4352
    assert s._group_bytes(8192, rows(8192, 16, 150)) == (
        8192 * 2 * kv + int(16 * st * 2 * kv * 1.05) + 8 * 8 * 16 * 16 * (8192 + st)
        + (64 << 20) + 2 * 16 * st * 2 * kv // 8)

    def delta(P, n):               # what forming moves into `reserved`, GiB
        rs = rows(P, n, 150)
        return round((s._group_bytes(P, rs) - sum(r.reserved - member for r in rs)) / GIB, 2)
    assert [delta(8192, n) for n in (5, 8, 16)] == [0.71, 0.35, -0.10]
    assert [delta(32768, n) for n in (3, 16)] == [1.66, -10.14]
    assert [delta(4096, n) for n in (3, 8, 16)] == [0.83, 1.0, 1.69]
    # the rule, at the real lengths (pos + 1): ~150-token suffixes win from 5
    # rows (4 from a 32K prefix), never at 2..3 (the floor, below)
    for P in (4096, 8192, 32768):
        for n in (5, 8, 16):
            assert s._group_wins(P, rows(P, n, 150)), (P, n)
        for n in (2, 3):
            assert not s._group_wins(P, rows(P, n, 150)), (P, n)
        assert s._group_wins(P, rows(P, 4, 150)) == (P == 32768), P
    # break-even: 4K x 5 rows at a 256-token suffix (the stack's 2nd step),
    # 4K x 8 at 1280, 8K x 8 at 3072, 8K x 16 at 4352 (past the P // 2
    # leave), 32K x 8 at 13056
    for P, n, S in ((4096, 5, 256), (4096, 8, 1280), (8192, 8, 3072), (8192, 16, 4352),
                    (32768, 8, 13056)):
        assert s._group_wins(P, rows(P, n, S - 1)) and not s._group_wins(P, rows(P, n, S))
    # the budget guard of a formation / join: reserved + the prompt cache's
    # live bytes (review cascade_b2b3_fixes B)
    pc = s.eng._pcache
    s.reserved, s.budget = 10 * GIB, 10 * GIB + 100
    assert s._charge_fits(100) and not s._charge_fits(101) and s._charge_fits(-5 * GIB)
    pc.bytes = 40                               # the cache makes the difference
    assert s._charge_fits(60) and not s._charge_fits(61) and s._charge_fits(0)
    pc.bytes = 0
    s.reserved = 11 * GIB                       # already past it (growth): savings only
    assert not s._charge_fits(1) and s._charge_fits(0)


def test_dissolve_floor_bytes(dense, casc, monkeypatch):
    """_group_bytes never charges less than the prefix + stacks + 2
    contiguous fp16 rows (the unchecked forced release of the last 2 members
    builds them beside the live group: review cascade_b2b3_fixes C). 9B
    layout, 3 rows over 32K / 64K / 96K with ~150-token real suffixes and
    short answers: the floor is the charge, and the group loses to its rows
    (4.2 / 8.3 / 12.4 GiB vs 3.2 / 6.3 / 9.5); without it, it won (2.2 /
    4.3 / 6.4 GiB). 16 rows over 32K still win by far."""
    s = nine_b(dense, monkeypatch)
    kv, member = 32 << 10, s._member_bytes()
    for P, want, own in ((32768, 4.17, 3.16), (65536, 8.27, 6.31), (98304, 12.37, 9.46)):
        rs = [FakeRow(P + 150, P + 400, s._bytes(P + 400)) for _ in range(3)]
        real = {r: r.pos + 1 for r in rs}
        st = 256                                # 151 real suffix tokens: one stack step
        floor = P * 2 * kv + int(3 * st * 2 * kv * 1.05) + 2 * int((P + st) * kv * 1.05)
        assert s._group_bytes(P, rs, real) == floor
        assert round(floor / GIB, 2) == want
        assert round((sum(s._bytes(n) for n in real.values()) - 3 * member) / GIB, 2) == own
        assert not s._group_wins(P, rs)
        # the charge at the reservations holds the forced release's peak:
        # the group's arrays + both members' contiguous caches (P + 400 each)
        arrays = P * 2 * kv + int(3 * 512 * 2 * kv * 1.05)
        assert s._group_bytes(P, rs) >= arrays + 2 * (s._bytes(P + 400) - member)
    rs = [FakeRow(32768 + 150, 32768 + 4246, s._bytes(32768 + 4246)) for _ in range(16)]
    assert s._group_wins(32768, rs)


def test_release_waits_for_headroom(dense, casc, monkeypatch):
    """_tidy_choice, 9B layout: an optional release happens only while
    `reserved` + the released rows' contiguous caches + the prompt cache fit
    budget + prefill reserve, with nothing credited for the group's shrink
    (the group is alive while release builds the caches, take is lazy:
    review cascade_b2b3_fixes A + B), one member per tidy (the last < 3
    together); else the rows stay grouped. Forced releases go anyway."""
    s = nine_b(dense, monkeypatch)
    P, member = 8192, s._member_bytes()
    pc = s.eng._pcache

    def setup(live, extra=0):
        g = types.SimpleNamespace(P=P, reserved=s._group_bytes(P, live))
        s.rows, s.prefill_reserve, pc.bytes = list(live) + [FakeRow(P, P)] * extra, 0, 0
        s.reserved = g.reserved + len(live) * member
        return g

    def need(out):                 # what the release materializes beside the group
        return sum(s._bytes(r.tokens) - r.reserved for r in out)

    def credited(g, out, stay):    # the post-state delta (the old, too lenient check)
        return need(out) + (s._group_bytes(P, stay) if stay else 0) - g.reserved

    # memory: 15 short members + one at 4097 lose at real lengths -> staged,
    # the cheapest member first (here all reserve alike: the first of them)
    short = [FakeRow(P + 150, P + 4246, member, f"m{i}") for i in range(15)]
    long = FakeRow(P + 4097, P + 4246, member, "long")
    live = short + [long]
    g = setup(live)
    assert not s._group_wins(P, live)
    n1 = need(short[:1])
    assert n1 > 0 and credited(g, live, []) < n1      # the old check credited the group
    s.budget = s.reserved + n1 - 1
    assert s._tidy_choice(g, live) == ([], live, None)
    s.prefill_reserve = 1                        # the headroom counts the prefill reserve
    assert s._tidy_choice(g, live) == (short[:1], short[1:] + [long], "memory")
    pc.bytes = 1                                 # ... and the prompt cache's live bytes
    assert s._tidy_choice(g, live) == ([], live, None)
    # a whole dissolve at once would put 16 caches beside the group: more
    # than the group's own charge
    assert need(live) == 16 * n1 > g.reserved
    # the P // 2 leave (memory rule off): one leaver's cache, +0.28 GiB
    monkeypatch.setattr(worker._Scheduler, "_group_wins", any_three)
    g = setup(live)
    d = need([long])
    assert credited(g, [long], short) < 64 << 20 < d
    s.budget = s.reserved + d - 1
    assert s._tidy_choice(g, live) == ([], live, None)
    s.budget += 1
    assert s._tidy_choice(g, live) == ([long], short, "suffix past P/2")
    # several leavers in one tidy: only the longest goes
    longer = FakeRow(P + 4200, P + 4246, member, "longer")
    rs = short[:4] + [long, longer]
    g = setup(rs)
    s.budget = 1 << 50
    assert s._tidy_choice(g, rs) == ([longer], short[:4] + [long], "suffix past P/2")
    # a leave that would leave < 3 members releases them too, all checked
    three = short[:2] + [long]
    g = setup(three)
    s.budget = s.reserved + need(three) - 1
    assert s._tidy_choice(g, three) == ([], three, None)
    s.budget += 1
    out, stay, why = s._tidy_choice(g, three)
    assert out == three and stay == [] and why == "suffix past P/2"
    # switched off: one member per tidy, the cheapest first, inside the check
    s.cascade = False
    rs = short[:3] + [FakeRow(P + 150, P + 300, member, "cheap")]
    g = setup(rs)
    s.budget = s.reserved + need(rs[3:]) - 1
    assert s._tidy_choice(g, rs) == ([], rs, None)
    s.budget += 1
    assert s._tidy_choice(g, rs) == (rs[3:], rs[:3], "switched off")
    s.cascade = True
    # forced: 2 live members (always so at a batch of 2) -- whatever the headroom
    s.budget = 0
    g = setup(short[:2], extra=1)
    pc.bytes = 1 << 40
    assert s._tidy_choice(g, short[:2]) == (short[:2], [], "2 members")


def test_staged_memory_dissolve(dense, casc, monkeypatch):
    """A memory dissolve of a 16-member group (9B layout) goes one member
    per tidy, the cheapest first: each tidy materializes one contiguous
    cache beside the group, not 16; the members left are re-judged every
    tidy (here they lose until 3 are left, which go together: a group of 3
    never wins with the dissolve floor)."""
    s = nine_b(dense, monkeypatch)
    P, member = 8192, s._member_bytes()
    live = [FakeRow(P + 4400 + i, P + 5000 - i, member, f"m{i}") for i in range(16)]
    g = types.SimpleNamespace(P=P, reserved=s._group_bytes(P, live))
    s.rows, s.prefill_reserve, s.budget = list(live), 0, 1 << 50
    s.reserved = g.reserved + len(live) * member
    seen = []
    while live:
        out, stay, why = s._tidy_choice(g, live)
        assert out and why == "memory"
        seen.append([r.req.id for r in out])
        for r in out:                           # as _tidy_group re-accounts them
            s.reserved += s._bytes(r.tokens) - r.reserved
            r.reserved = s._bytes(r.tokens)
        s.reserved += (s._group_bytes(P, stay) if stay else 0) - g.reserved
        g.reserved = s._group_bytes(P, stay) if stay else 0
        live = stay
    assert seen == [[f"m{i}"] for i in range(15, 2, -1)] + [["m0", "m1", "m2"]]
    # one optional release per decode step (the last one's take is freed by
    # the step that consumes it); a forced one goes anyway
    live = [FakeRow(P + 4400 + i, P + 5000 - i, member, f"n{i}") for i in range(4)]
    g = types.SimpleNamespace(P=P, reserved=s._group_bytes(P, live), released_at=s.steps)
    assert s._tidy_choice(g, live) == ([], live, None)
    s.steps += 1
    assert s._tidy_choice(g, live) == (live[3:], live[:3], "memory")
    s.steps -= 1
    assert s._tidy_choice(g, live[:2]) == (live[:2], [], "2 members")


def test_cascade_switch(dense, casc, monkeypatch):
    assert scheduler(dense).cascade
    assert scheduler(dense).cascade_state().startswith("on")
    assert not scheduler(dense, kv_bits=8).cascade                 # quantized KV
    assert not scheduler(dense, max_batch=2).cascade               # B <= 2 only
    s = scheduler(dense, budget=1)
    s.fit_max_batch()                                              # batching off -> no groups
    assert s.max_batch == 1 and not s.cascade
    monkeypatch.setattr(worker, "PROMPT_CACHE", False)
    assert "no prompt cache" in scheduler(dense).cascade_state()
    monkeypatch.setattr(worker, "PROMPT_CACHE", True)
    monkeypatch.setattr(worker, "CASCADE", False)
    s = scheduler(dense)
    assert not s.cascade and "QWEN_CASCADE=0" in s.cascade_state()


# ------------------------------------------------------------------ logits level


def prefix_rows(lm, shared, tails):
    """Rows' caches holding the same KV bits [0, len(shared)): the prefix
    prefilled once, each row's own tokens on a copy (as a prompt-cache hit)."""
    lm._position_ids, lm._rope_deltas = None, None
    base = lm.make_cache()
    lm(mx.array([shared]), cache=base, skip_logits=True)
    mx.eval([c.state for c in base])
    rows = []
    for t in tails:
        c = worker._copy_cache(base)
        lm._rope_deltas, lm._position_ids = mx.zeros((1, 1), dtype=mx.int32), None
        lm(mx.array([t]), cache=c, skip_logits=True)
        mx.eval([x.state for x in c])
        rows.append(c)
    return base, rows


@pytest.mark.parametrize("model,tol", [("dense", 1e-4), ("dense_fused", 3e-2)])
@pytest.mark.parametrize("P", [32, 36])                   # whole chunks / padded
def test_grouped_logits(model, tol, P, request, monkeypatch):
    """6 rows sharing a 36-token prefix, rows 0..2 grouped over [0, P): their
    logits stay within fp rounding of the per-row path (teacher-forced, top-1
    equal every step); rows 3..5 of the same batch are bit-identical to a run
    without groups; release gives [0, P) bit for bit and the first FA layer's
    whole cache bit for bit (its K/V do not depend on attention)."""
    lm = request.getfixturevalue(model)
    shared = prompts([36], 70)[0]
    tails = prompts([2, 5, 3, 6, 1, 4], 71)
    feed = prompts([7] * 6, 72)
    layers = lm.model.layers
    fa = [i for i, x in enumerate(layers) if not x.is_linear]
    dt = lm.model.embed_tokens(mx.array([[0]])).dtype

    def go(grouped):
        base, caches = prefix_rows(lm, shared, tails)
        gdn = worker._gdn_stack(lm, caches, dt)
        pg = None
        if grouped:
            pg = cascade.PrefixGroup.build(caches[0], caches[:3], P, fa, chunk=16, step=16)
        pos = [36 + len(t) for t in tails]
        out = []
        for s in range(7):
            plan = None
            if pg is not None:
                plan = worker._StepPlan([3, 4, 5], [(pg, mx.array([0, 1, 2]))],
                                        mx.array([3, 4, 5, 0, 1, 2]), [])
                pg.begin_step()
            cache = [gdn[i] if i in gdn else worker.RowsKV([c[i] for c in caches], i, plan)
                     for i in range(len(layers))]
            lg = lm(mx.array([f[s] for f in feed])[:, None], cache=cache,
                    position_ids=mx.array([[p] for p in pos], dtype=mx.int32)).logits[:, -1, :]
            if pg is not None:
                pg.end_step()
            mx.eval(lg)
            out.append(lg)
            pos = [p + 1 for p in pos]
        return base, caches, pg, out

    _, ref_caches, _, ref = go(False)
    base, caches, pg, got = go(True)
    for a, b in zip(ref, got):
        assert mx.array_equal(a[3:], b[3:]).item()                  # ungrouped: bitwise
        d = mx.abs(a[:3].astype(mx.float32) - b[:3].astype(mx.float32)).max().item()
        assert d <= tol, d
        assert mx.array_equal(mx.argmax(a[:3], axis=-1), mx.argmax(b[:3], axis=-1)).item()
    assert pg.lens == tuple(36 + len(t) - P + 7 for t in tails[:3])
    for j in range(3):
        rel = pg.release(j)
        for i in fa:
            k, v = rel[i].keys, rel[i].values
            assert rel[i].offset == 36 + len(tails[j]) + 7
            assert mx.array_equal(k[..., :P, :], base[i].keys[..., :P, :]).item()
            assert mx.array_equal(v[..., :P, :], base[i].values[..., :P, :]).item()
            n0 = 36 + len(tails[j])                       # the row's own tokens before grouping
            assert mx.array_equal(k[..., P:n0, :], caches[j][i].keys[..., P:n0, :]).item()
        first = fa[0]                                     # attention-independent K/V
        r = ref_caches[j][first]
        n = r.offset
        assert mx.array_equal(rel[first].keys[..., :n, :], r.keys[..., :n, :]).item()
        assert mx.array_equal(rel[first].values[..., :n, :], r.values[..., :n, :]).item()


# ------------------------------------------------------------------ scheduler


@pytest.mark.parametrize("check", [True, False], ids=["check", "nocheck"])
def test_wave_forms_joins_and_dissolves(dense, casc, monkeypatch, caplog, check):
    """6 requests on one system prompt arriving one per tick: the group forms
    at the 3rd join and converts the earlier rows (the cold producer among
    them), the later rows join it, every answer equals the request alone
    (fp32: the cascade differs from the per-row kernel by ~1e-7), every
    row's logits at every step stay within 1e-5 of the same batch without
    groups (catches a wrong slot / a diluted prefix that flips no token),
    the members are in batch order (one slice, no output take), and at the
    end no group is left and nothing is reserved."""
    monkeypatch.setattr(worker, "CASCADE_CHECK", check)
    caplog.set_level(logging.INFO, logger="qwen.worker")
    ids = chats(6)
    ref = serial(dense, [request(p, 20, f"r{i}") for i, p in enumerate(ids)])
    lg = Logits(monkeypatch)
    monkeypatch.setattr(worker, "CASCADE", False)
    lg.start()
    reqs = [request(p, 20, f"r{i}") for i, p in enumerate(ids)]
    assert run(scheduler(dense, max_batch=8), reqs, arrive=one_per_tick(reqs)) == ref
    monkeypatch.setattr(worker, "CASCADE", True)
    lg.start()
    spy = Spy(monkeypatch)
    plans = Plans(monkeypatch)
    sched = scheduler(dense, max_batch=8)
    reqs = [request(p, 20, f"r{i}") for i, p in enumerate(ids)]
    got = run(sched, reqs, arrive=one_per_tick(reqs), on_tick=spy.tick)
    assert got == ref
    d = lg.max_diff()
    assert 0 < d <= 1e-5, d                    # grouped steps ran, close to the per-row path
    whole = plans.whole()
    assert len(whole) > 10
    for B, p in whole:
        assert p.inv is None and p.groups[0][1] == slice(0, B), (B, p)
    (P, members), = spy.builds
    assert P == 151 and len(members) == 3 and {r for _, r in members} >= {0}   # cold producer
    assert [rid for _, rid in spy.adds] == ["r3", "r4", "r5"]
    assert max(len(g[0]) for g in spy.groups if g) == 6 and spy.calls > 0
    assert sched.groups == [] and sched.reserved == 0 and sched.n_grouped == 0
    assert any("cascade: group formed: P=151, 3 rows" in m for m in caplog.messages)
    assert any("dissolved (finished)" in m for m in caplog.messages)


@pytest.mark.parametrize("check", [True, False], ids=["check", "nocheck"])
def test_sync_mode_and_mixed_batch(dense, casc, monkeypatch, check):
    """A guided row (the whole batch in sync mode: no step-ahead) and rows of
    another prompt in the same batch: the shared-prefix rows are grouped, the
    others decode per row, every answer equals the request alone."""
    monkeypatch.setattr(worker, "CASCADE_CHECK", check)
    ids = chats(4)
    other = prompts([30, 22], 80)

    def make():
        return ([request(p, 16, f"s{i}") for i, p in enumerate(ids)]
                + [request(other[0], 16, "o0"), request(other[1], 16, "o1"),
                   request(chats(1, seed=90)[0], 16, "g", guide=AllowAll())])
    ref = serial(dense, make())
    spy = Spy(monkeypatch)
    sched = scheduler(dense, max_batch=8)
    reqs = make()
    got = run(sched, reqs, arrive=one_per_tick(reqs), on_tick=spy.tick)
    assert got == ref
    assert spy.calls > 0
    grouped = {rid for g in spy.groups for m in g for rid in m}
    assert grouped == {"s0", "s1", "s2", "s3"}


def test_off_is_bitwise(dense, casc, monkeypatch):
    """QWEN_CASCADE=0 (set explicitly: the default is on): no group, no
    plan, cascade never runs -- the per-row code path, bit for bit, with
    every other cascade condition met (fp16 KV, prompt cache, max_batch 8,
    a shared 151-token prefix over the lowered thresholds)."""
    on = worker._cascade_knobs({"QWEN_CASCADE": "0"})[0]
    assert on is False
    monkeypatch.setattr(worker, "CASCADE", on)
    ids = chats(6)
    ref = serial(dense, [request(p, 12, f"r{i}") for i, p in enumerate(ids)])
    spy = Spy(monkeypatch)
    plans = []
    monkeypatch.setattr(worker._Scheduler, "_plan", lambda *a: plans.append(1))
    monkeypatch.setattr(cascade, "attention", lambda *a: pytest.fail("cascade ran"))
    sched = scheduler(dense, max_batch=8)
    reqs = [request(p, 12, f"r{i}") for i, p in enumerate(ids)]
    assert run(sched, reqs, arrive=one_per_tick(reqs), on_tick=spy.tick) == ref
    assert spy.builds == spy.adds == plans == [] and spy.calls == 0


def test_two_rows_never_grouped(dense, casc, monkeypatch):
    """B <= 2: two rows sharing a prefix are not grouped (min rows lowered to
    3 is the floor), even with max_batch 8."""
    monkeypatch.setattr(worker, "CASCADE_MIN_ROWS", 3)
    spy = Spy(monkeypatch)
    ids = chats(2)
    reqs = [request(p, 12, f"r{i}") for i, p in enumerate(ids)]
    ref = serial(dense, [request(p, 12, f"r{i}") for i, p in enumerate(ids)])
    assert run(scheduler(dense, max_batch=8), reqs, on_tick=spy.tick) == ref
    assert spy.builds == [] and spy.calls == 0


def snap(sched, r, b):
    """A dissolved row's state: its caches (new objects), next input, position,
    tokens sent."""
    cache = []
    for i, c in enumerate(r.cache):
        if i in sched.fa_layers:
            n = KVCache()
            n.keys, n.values, n.offset = c.keys[..., :c.offset, :], c.values[..., :c.offset, :], c.offset
        else:
            n = ArraysCache(size=2)
            n.cache = ([a[b:b + 1] for a in sched.gdn[i].cache] if sched.gdn is not None
                       else list(c.cache))
        cache.append(n)
    return dict(cache=cache, y=r.y, pos=r.pos, n=r.n, sampling=r.req.sampling)


def spy_released(sched, out):
    """Snapshot every row the cascade tidy-up gives its own caches back."""
    tidy = sched._cascade_tidy

    def spy(*a, **k):
        before = {id(r) for g in sched.groups for r in g.rows if not r.done}
        tidy(*a, **k)
        now = {id(r) for g in sched.groups for r in g.rows}
        for b, r in enumerate(sched.rows):
            if id(r) in before and id(r) not in now:
                out[r.req.id] = snap(sched, r, b)
    sched._cascade_tidy = spy


def replay(lm, s, n):
    """n greedy tokens decoded alone (the B = 1 library call) from a snapshot."""
    sampler = worker.make_sampler(s["sampling"])
    y, pos, toks = s["y"], s["pos"], []
    for _ in range(n):
        lg = lm(y[None], cache=s["cache"], position_ids=mx.array([[pos]], dtype=mx.int32))
        y = sampler(lg.logits[:, -1, :])
        toks.append(y.item())
        pos += 1
    return toks


def check_continuation(lm, got, snaps, ids_of, P):
    """From the release on, every released row decodes exactly as a row that
    was never grouped and holds the same cache; its [0, P) are the cold bits."""
    cold = Cold(lm)(ids_of[next(iter(snaps))][:P])
    for rid, s in snaps.items():
        toks = got[rid][0]
        assert toks[s["n"]] == s["y"].item()
        rest = toks[s["n"] + 1:]
        assert rest and replay(lm, s, len(rest)) == rest, rid
        for i in (i for i, c in enumerate(cold) if isinstance(c, KVCache)):
            assert mx.array_equal(s["cache"][i].keys[..., :P, :], cold[i].keys[..., :P, :]).item()
            assert mx.array_equal(s["cache"][i].values[..., :P, :],
                                  cold[i].values[..., :P, :]).item()


@pytest.mark.parametrize("model", ["dense", "dense_fused"])
def test_dissolves_at_two_rows(model, request, casc, monkeypatch, caplog):
    """4 rows with max_tokens 6 / 8 / 30 / 30 (grouped at the 3rd join, the
    4th joins): once two are left the group dissolves, every later step takes
    the per-row path, and the two continue exactly as rows that were never
    grouped and hold the same caches (B <= 2 rows are bit-exact). fp16
    (dense_fused): the answers before the release may flip near-ties."""
    dense = request.getfixturevalue(model)
    caplog.set_level(logging.INFO, logger="qwen.worker")
    ids = chats(4)
    lens = [6, 8, 30, 30]
    ref = serial(dense, [worker_req(p, n, f"r{i}") for i, (p, n) in enumerate(zip(ids, lens))])
    spy = Spy(monkeypatch)
    sched = scheduler(dense, max_batch=8)
    snaps = {}
    spy_released(sched, snaps)
    calls = []
    reqs = [worker_req(p, n, f"r{i}") for i, (p, n) in enumerate(zip(ids, lens))]

    def tick(t, s):
        spy.tick(t, s)
        calls.append((len(s.rows), len(s.groups), spy.calls))
    got = run(sched, reqs, on_tick=tick)
    if model == "dense":
        assert got == ref
    assert len(spy.builds) == 1 and len(spy.builds[0][1]) == 3 and len(spy.adds) == 1
    assert set(snaps) == {"r2", "r3"}
    assert any("dissolved (2 members)" in m for m in caplog.messages)
    last = max(i for i, c in enumerate(calls) if c[1])    # the last tick with a group
    after = calls[last + 1:]
    assert after and all(b <= 2 and g == 0 for b, g, _ in after)
    assert after[0][2] == after[-1][2]                    # no grouped attention after it
    check_continuation(dense, got, snaps, {f"r{i}": p for i, p in enumerate(ids)}, 151)
    assert [len(got[f"r{i}"][0]) for i in range(4)] == lens


worker_req = request                      # `request` is pytest's fixture inside parametrized tests


def test_long_suffix_leaves(dense, casc, monkeypatch, caplog):
    """A member whose own suffix passes P // 2 leaves the group (released),
    the others stay grouped; every answer equals the request alone and the
    released row continues as a never-grouped row would. (The memory rule is
    off here: with a suffix near P / 2 it would keep r4 out of a 5-row group.)"""
    caplog.set_level(logging.INFO, logger="qwen.worker")
    ids = chats(5, user=lambda i: 60 if i == 4 else 6 + i)    # r4: suffix 66 at join, P // 2 = 75
    ref = serial(dense, [request(p, 30, f"r{i}") for i, p in enumerate(ids)])
    spy = Spy(monkeypatch)
    sched = scheduler(dense, max_batch=8)
    snaps = {}
    spy_released(sched, snaps)
    reqs = [request(p, 30, f"r{i}") for i, p in enumerate(ids)]
    got = run(sched, reqs, arrive=one_per_tick(reqs), on_tick=spy.tick)
    assert got == ref
    assert ("r4" in {rid for g in spy.groups for m in g for rid in m})
    assert any("r4 left (suffix past P/2), 4 rows stay" in m for m in caplog.messages)
    assert set(snaps) >= {"r4"}
    check_continuation(dense, got, {"r4": snaps["r4"]}, {"r4": ids[4]}, 151)


def test_memory_rule_in_the_scheduler(dense, casc, monkeypatch, caplog):
    """The real rule on the tiny model (2 FA layers, 16-token stack steps, no
    fixed slack): 4 rows over the 151-token prefix would take more as a group
    (with the floor of 2 contiguous rows) than in their own caches and never
    form one; 5 do (at the 5th join, the rows prefilled back to back: <= 16-
    token suffixes) and, once their real suffixes reach the second stack
    step, dissolve in stages ("memory"): one member per tidy -- re-judged
    each time -- until the last 3 go together. Every answer is the
    request's own."""
    caplog.set_level(logging.INFO, logger="qwen.worker")
    monkeypatch.setattr(worker._Scheduler, "_group_wins", REAL_WINS)
    monkeypatch.setattr(worker, "CASCADE_SLACK", 0)
    ids = chats(5)
    spy = Spy(monkeypatch)
    run(scheduler(dense, max_batch=8), [request(p, 10, f"r{i}") for i, p in enumerate(ids[:4])],
        on_tick=spy.tick)
    assert spy.builds == [] and spy.adds == []
    ref = serial(dense, [request(p, 20, f"m{i}") for i, p in enumerate(ids)])
    tidies = []                     # releases per _tidy_group call
    release, tidy = cascade.PrefixGroup.release, worker._Scheduler._tidy_group

    def spy_release(pg, j, *a, **k):
        tidies[-1] += 1
        return release(pg, j, *a, **k)

    def spy_tidy(sched, g):
        tidies.append(0)
        return tidy(sched, g)
    monkeypatch.setattr(cascade.PrefixGroup, "release", spy_release)
    monkeypatch.setattr(worker._Scheduler, "_tidy_group", spy_tidy)
    sched = scheduler(dense, max_batch=8)
    got = run(sched, [request(p, 20, f"m{i}") for i, p in enumerate(ids)], on_tick=spy.tick)
    assert got == ref and len(spy.builds) == 1 and len(spy.builds[0][1]) == 5
    left = [m for m in caplog.messages if "left (memory)" in m]
    assert len(left) == 2 and any("dissolved (memory): 3 rows released" in m
                                  for m in caplog.messages)
    assert [n for n in tidies if n] == [1, 1, 3], tidies
    assert sched.groups == [] and sched.reserved == 0


@pytest.mark.parametrize("room", ["short", "fits", "pc-blind"])
@pytest.mark.parametrize("what", ["form", "join"])
def test_budget_guard(dense, casc, monkeypatch, room, what):
    """Grouping that raises `reserved` (forming: the fixed 64 MiB slack;
    joining: a stack slot as wide as r0's long reservation) is taken only
    while `reserved` + the prompt cache's live bytes stay within the budget
    (review cascade_b2b3_fixes B): one byte short, no group at P = 151 (a
    shallower P may still fit) / no join; exactly enough, it forms / joins;
    a budget that holds the charge but not the cache's bytes too (the
    cache holds the rows' entries) refuses it -- nothing is evicted for it.
    `reserved` + the cache never pass the budget; a refused row decodes
    ungrouped; every answer is the request's own."""
    ids = chats(4)
    n = 3 if what == "form" else 4

    def make():
        return [request(p, m, f"r{i}") for i, (p, m) in enumerate(zip(ids[:n], [40, 12, 12, 12]))]
    ref = serial(dense, make())
    spy = Spy(monkeypatch)
    sched = scheduler(dense, max_batch=8)
    pc = sched.eng._pcache
    join, member, seen = sched._cascade_join, sched._member_bytes(), []

    def guarded(row):
        if len(sched.rows) != n:
            return join(row)
        rows = list(sched.rows)
        if what == "form":
            delta = sched._group_bytes(151, rows) - sum(r.reserved - member for r in rows)
        else:
            g, = sched.groups
            delta = (sched._group_bytes(151, g.rows + [row]) - g.reserved
                     - (row.reserved - member))
        cached, before = pc.bytes, sched.reserved
        sched.budget = before + delta + {"short": cached - 1, "fits": cached,
                                         "pc-blind": 0}[room]
        join(row)
        # the budget set here holds the state before the join only with the
        # cache counted ("short" / "fits"): then it holds after it too
        ok = (sched.reserved + pc.bytes <= sched.budget if room != "pc-blind"
              else sched.reserved == before)
        seen.append((delta, cached, pc.bytes, [g.P for g in sched.groups], ok,
                     row.group is not None))
        sched.budget = 1 << 40
    sched._cascade_join = guarded
    got = run(sched, make(), on_tick=spy.tick)
    assert got == ref
    (delta, cached, after, groups, ok, grouped), = seen
    assert delta > 0 and cached > 0 and after == cached and ok
    if what == "form":
        assert (151 in groups) == (room == "fits")
    else:
        assert groups == [151] and grouped == (room == "fits")


def test_accounting_numbers(dense, casc, monkeypatch):
    """Formation moves the rows' KV charge to the group (members keep their
    recurrent state); a growing member grows the group's charge; dissolving
    gives the rows _bytes(tokens) back."""
    monkeypatch.setattr(worker, "RESERVE_TOKENS", 4)
    ids = chats(5)
    sched = scheduler(dense, max_batch=8)
    seen = []
    build, tidy = sched._group_build, sched._cascade_tidy

    def spy_build(P, rows):
        before = (sched.reserved, sum(r.reserved for r in rows))
        build(P, rows)
        g = sched.groups[-1]
        seen.append(("build", sched.reserved - before[0],
                     g.reserved + len(rows) * sched._member_bytes() - before[1]))

    def spy_tidy(*a, **k):
        live = [r for g in sched.groups for r in g.rows if not r.done]
        tidy(*a, **k)
        for r in live:
            if r.group is None:
                assert r.reserved == sched._bytes(r.tokens)
                seen.append(("release", r.req.id))
    sched._group_build, sched._cascade_tidy = spy_build, spy_tidy
    charges = []

    def tick(t, s):
        check_accounting(s)
        charges.extend(g.reserved for g in s.groups)
    reqs = [request(p, n, f"r{i}") for i, (p, n) in enumerate(zip(ids, [40, 40, 30, 20, 12]))]
    run(sched, reqs, on_tick=tick)
    assert seen[0][0] == "build" and seen[0][1] == seen[0][2]
    assert max(charges) > min(charges)                    # _regrow grew the group's charge
    assert any(s[0] == "release" for s in seen)
    assert sched.groups == [] and sched.reserved == 0


def test_suffix_stack_grows(dense, casc, monkeypatch):
    """Generation passes the stack's capacity (step 16) several times."""
    ids = chats(4, sys_len=300)
    ref = serial(dense, [request(p, 40, f"r{i}") for i, p in enumerate(ids)])
    caps = set()
    spy = Spy(monkeypatch)

    def tick(t, s):
        spy.tick(t, s)
        caps.update(g.pg.cap for g in s.groups)
    reqs = [request(p, 40, f"r{i}") for i, p in enumerate(ids)]
    assert run(scheduler(dense, max_batch=8), reqs, on_tick=tick) == ref
    assert len(caps) >= 3 and spy.calls > 0


def test_excludes_vision_and_dflash(dense, casc, monkeypatch):
    """A vision row with the same leading text and a (demoted) DFlash row are
    never grouped; the text rows around them are, and every answer is the
    request's own."""
    no_metal_verify(monkeypatch)
    ids = chats(4)
    plain = run(scheduler(dense, max_batch=1), [request(ids[0], 30, "d")])["d"]
    v_ids = ids[1][:152] + [worker_vision_start()] + [500] * 16 + prompts([9], 81)[0] + GEN
    vreq = request(v_ids, 12, "v")
    vreq.images, vreq.vision = ["img.png"], vision_inputs(v_ids)
    alone = run(scheduler(dense, max_batch=1), [vreq])["v"]
    texts = [request(p, 12, f"t{i}") for i, p in enumerate(ids[1:])]
    ref = serial(dense, [request(p, 12, f"t{i}") for i, p in enumerate(ids[1:])])
    spy = Spy(monkeypatch)
    sched = scheduler(dense, max_batch=8)
    sched.eng.draft = FakeDraft(plain[0])
    d = request(ids[0], 30, "d")
    vreq = request(v_ids, 12, "v")
    vreq.images, vreq.vision = ["img.png"], vision_inputs(v_ids)
    spec = []
    join = sched._join
    sched._join = lambda row: (spec.append((row.req.id, row.spec)), join(row))
    got = run(sched, [d, vreq] + texts, arrive={26: [vreq], 27: texts[:1], 28: texts[1:2],
                                                29: texts[2:]}, on_tick=spy.tick)
    assert got["d"] == plain and got["v"] == alone
    for k in ref:
        assert got[k] == ref[k], k
    grouped = {rid for g in spy.groups for m in g for rid in m}
    assert grouped == {"t0", "t1", "t2"} and len(sched.eng.draft.accept_lens) > 0
    assert ("d", True) in spec                  # demoted into the batch, a DFlash row


def worker_vision_start():
    from conftest import VISION_START
    return VISION_START


def test_quantized_kv_never_groups(dense, casc, monkeypatch):
    spy = Spy(monkeypatch)
    ids = chats(4)
    sched = scheduler(dense, max_batch=8, kv_bits=8, kv_start=40)
    run(sched, [request(p, 8, f"r{i}") for i, p in enumerate(ids)], on_tick=spy.tick)
    assert not sched.cascade and spy.builds == [] and spy.calls == 0


def test_failed_step_drops_groups(dense, casc, monkeypatch):
    """A grouped step that fails: its rows fail, the groups go, the
    accounting is clean, and the next request runs."""
    ids = chats(4)
    fail = {"left": 1}
    attention = cascade.attention

    def boom(*a, **k):
        if fail["left"]:
            fail["left"] -= 1
            raise RuntimeError("cascade boom")
        return attention(*a, **k)
    monkeypatch.setattr(cascade, "attention", boom)
    sched = scheduler(dense, max_batch=8)
    tick = sched.tick

    def tick_as_run():                          # _Scheduler._run's handling of a failed tick
        try:
            tick()
        except Exception as e:  # noqa: BLE001
            sched._fail_rows(e)
            sched.failures += 1
    sched.tick = tick_as_run
    reqs = [request(p, 10, f"r{i}") for i, p in enumerate(ids)]
    got = run(sched, reqs, on_tick=lambda t, s: check_accounting(s))
    assert all(got[r.id][2][:3] == ("error", "RuntimeError", "cascade boom") for r in reqs)
    assert sched.groups == [] and sched.reserved == 0 and sched.n_groups == 0
    assert sched.failures == 1 and sched.closed is None and fail["left"] == 0
    nxt = request(ids[0], 6, "next")
    assert run(sched, [nxt])["next"] == serial(dense, [request(ids[0], 6, "next")])["next"]


@pytest.mark.parametrize("victim", ["r1", "r4"], ids=["at-formation", "at-join"])
def test_check_refuses_other_bits(dense, casc, monkeypatch, caplog, victim):
    """QWEN_CASCADE_CHECK: a row whose KV [0, P) differs from the group's (here
    corrupted after its prefill) is logged and left out; the others group."""
    caplog.set_level(logging.WARNING, logger="qwen.worker")
    ids = chats(5)
    spy = Spy(monkeypatch)
    sched = scheduler(dense, max_batch=8)
    join = sched._join

    def corrupt(row):
        if row.req.id == victim:
            c = row.cache[sched.fa_layers[0]]
            c.keys[..., 5:6, :] = c.keys[..., 5:6, :] + 1
        join(row)
    sched._join = corrupt
    reqs = [request(p, 10, f"r{i}") for i, p in enumerate(ids)]
    got = run(sched, reqs, arrive=one_per_tick(reqs), on_tick=spy.tick)
    grouped = {rid for g in spy.groups for m in g for rid in m}
    assert victim not in grouped and grouped == {f"r{i}" for i in range(5)} - {victim}
    assert any("but not" in m and victim in m for m in caplog.messages)
    assert all(got[r.id][2] == ("done", "length") for r in reqs)


def test_set_cascade(dense, casc, monkeypatch):
    """The tools' switch: off dissolves now, on groups the running rows again."""
    ids = chats(4)
    ref = serial(dense, [request(p, 24, f"r{i}") for i, p in enumerate(ids)])
    spy = Spy(monkeypatch)
    sched = scheduler(dense, max_batch=8)
    states = []

    def tick(t, s):
        if s.groups and t % 3 == 0:
            s.set_cascade(False)
        elif not s.groups and len(s.rows) == 4 and t % 3 == 1:
            s.set_cascade(True)
        states.append(len(s.groups))
        spy.tick(t, s)
    reqs = [request(p, 24, f"r{i}") for i, p in enumerate(ids)]
    assert run(sched, reqs, on_tick=tick) == ref
    assert len(spy.builds) >= 2 and 0 in states and 1 in states


# ------------------------------------------------------------------ lifecycle edges


class Refs:
    """Weak references to every group's objects and arrays (the group, its
    cascade.PrefixGroup, the fp32 prefix, the stacks as of each tick)."""

    def __init__(self, monkeypatch):
        self.refs = []
        build = worker._Scheduler._group_build

        def spy(sched, P, rows):
            build(sched, P, rows)
            g = sched.groups[-1]
            self.refs += [weakref.ref(g), weakref.ref(g.pg)]
            self.refs += [weakref.ref(a) for a in (*g.pg.kc.values(), *g.pg.vc.values())]
        monkeypatch.setattr(worker._Scheduler, "_group_build", spy)

    def tick(self, t, sched):
        for g in sched.groups:
            self.refs += [weakref.ref(a) for a in (*g.pg.sk.values(), *g.pg.sv.values())]

    def alive(self):
        return [r() for r in self.refs if r() is not None]


@pytest.fixture
def no_gc():
    """Only refcounting frees objects: a reference cycle keeps them."""
    gc.collect()
    gc.disable()
    yield
    gc.enable()


@pytest.mark.parametrize("end", ["dissolve", "failed-step", "close"])
def test_dropped_group_freed_without_gc(dense, casc, monkeypatch, no_gc, end):
    """A group's fp32 prefix and stacks go with the group (review
    cascade_b2b3 finding 1): with the cyclic GC off, after a normal
    dissolution, after a failed grouped step (_fail_rows) and after _close,
    the group, its PrefixGroup and its arrays are dead -- no row <-> group
    or row -> _GroupRef -> row cycle holds them."""
    ids = chats(4)
    refs = Refs(monkeypatch)
    sched = scheduler(dense, max_batch=8)
    if end == "failed-step":
        attention, left = cascade.attention, [1]

        def boom(*a, **k):
            if left[0]:
                left[0] -= 1
                raise RuntimeError("cascade boom")
            return attention(*a, **k)
        monkeypatch.setattr(cascade, "attention", boom)
        tick = sched.tick

        def tick_as_run():                      # _Scheduler._run's handling of a failed tick
            try:
                tick()
            except Exception as e:  # noqa: BLE001
                sched._fail_rows(e)
        sched.tick = tick_as_run

    def on_tick(t, s):
        refs.tick(t, s)
        if end == "close" and s.groups and t > 40:
            s._close("shutdown", "test")
    reqs = [request(p, 30, f"r{i}") for i, p in enumerate(ids)]
    got = run(sched, reqs, on_tick=on_tick)
    assert len(refs.refs) > 10
    want = {"dissolve": "length", "failed-step": "RuntimeError", "close": "test"}[end]
    assert all(want in (got[r.id][2] or r.out.get_nowait()) for r in reqs), got
    assert sched.groups == [] and sched.reserved == 0
    assert refs.alive() == []


def test_two_groups_in_one_batch(dense, casc, monkeypatch):
    """Two system prompts, three rows each: two groups decode in the same
    steps (each its own slice of the batch), every answer is the request's
    own."""
    a, b = chats(3, seed=60), chats(3, sys_len=170, seed=90)
    make = lambda: ([request(p, 16, f"a{i}") for i, p in enumerate(a)]  # noqa: E731
                    + [request(p, 16, f"b{i}") for i, p in enumerate(b)])
    ref = serial(dense, make())
    spy = Spy(monkeypatch)
    plans = Plans(monkeypatch)
    reqs = make()
    got = run(scheduler(dense, max_batch=8), reqs, arrive=one_per_tick(reqs), on_tick=spy.tick)
    assert got == ref
    assert len(spy.builds) == 2 and spy.builds[0][0] == 151 and spy.builds[1][0] > 151
    both = [p for _, p in plans.seen if len(p.groups) == 2]
    assert both and all(p.inv is None and not p.single for p in both)
    assert max(len(g) for g in spy.groups) == 2


def test_refused_join_forms_no_duplicate_group(dense, casc, monkeypatch):
    """A row a group refuses (here the memory rule: only 3-row groups win)
    forms no second group on that group's prefix with the other refused
    rows -- the same prefix held twice; a shallower slab end may still
    group them. Every answer is the request's own."""
    monkeypatch.setattr(worker._Scheduler, "_group_wins",
                        lambda self, P, rows: len(rows) == 3)
    ids = chats(6)
    ref = serial(dense, [request(p, 20, f"r{i}") for i, p in enumerate(ids)])
    spy = Spy(monkeypatch)
    reqs = [request(p, 20, f"r{i}") for i, p in enumerate(ids)]
    sched = scheduler(dense, max_batch=8)
    seen = []

    def tick(t, s):
        spy.tick(t, s)
        seen.append([(g.P, tuple(g.ids)) for g in s.groups])
    got = run(sched, reqs, arrive=one_per_tick(reqs), on_tick=tick)
    assert got == ref
    assert spy.builds[0][0] == 151 and spy.adds == []
    assert all(len(set(gs)) == len(gs) for gs in seen)         # never one prefix twice
    assert all(P < 151 for P, _ in spy.builds[1:]), spy.builds
    assert sched.groups == [] and sched.reserved == 0


def test_member_resumed_deeper_than_P(dense, casc, monkeypatch):
    """A second turn resumes from its first turn's entry (deeper than the
    system prompt) and joins the system-prompt group: its suffix [P, pos)
    holds restored tokens too. Every answer is the request's own (hit ==
    cold)."""
    ids = chats(4)
    sched = scheduler(dense, max_batch=8)
    first = run(sched, [request(ids[3], 6, "t1")])["t1"][0]
    turn2 = ids[3] + first + [IM_START] + prompts([4], 83)[0] + GEN
    make = lambda: ([request(p, 12, f"r{i}") for i, p in enumerate(ids[:3])]  # noqa: E731
                    + [request(turn2, 12, "t2")])
    ref = serial(dense, make())
    spy = Spy(monkeypatch)
    reqs = make()
    got = run(sched, reqs, arrive=one_per_tick(reqs), on_tick=spy.tick)
    assert got == ref
    assert reqs[3].reused > 151 and (151, "t2") in spy.adds


def test_cancel_grouped_row(dense, casc, monkeypatch):
    """A grouped row cancelled mid-generation leaves the group (take); the
    others stay grouped and finish as alone; nothing stays reserved."""
    ids = chats(4)
    ref = serial(dense, [request(p, 30, f"r{i}") for i, p in enumerate(ids)])
    spy = Spy(monkeypatch)
    sched = scheduler(dense, max_batch=8)
    reqs = [request(p, 30, f"r{i}") for i, p in enumerate(ids)]
    after = []

    def tick(t, s):
        spy.tick(t, s)
        r1 = row_of(s, "r1")
        if r1 is not None and r1.group is not None and r1.n >= 5 and not reqs[1].cancel.is_set():
            reqs[1].cancel.set()
        elif reqs[1].cancel.is_set() and s.groups:
            after.append([r.req.id for r in s.groups[0].rows])
    got = run(sched, reqs, on_tick=tick)
    toks, _, ev = got["r1"]
    assert ev == ("done", "cancelled") and 5 <= len(toks) < 30 and toks == ref["r1"][0][:len(toks)]
    for k in ("r0", "r2", "r3"):
        assert got[k] == ref[k], k
    assert after and all("r1" not in m and len(m) == 3 for m in after[1:])
    assert sched.groups == [] and sched.reserved == 0


def test_grouped_row_ended_at_headroom(dense, casc, monkeypatch, caplog):
    """_regrow of a grouped row past the headroom: the group's charge would
    grow, so the row ends with "length" ("ended at"), its tokens so far are
    the request's own, the group goes on without it, the accounting holds."""
    caplog.set_level(logging.WARNING, logger="qwen.worker")
    monkeypatch.setattr(worker, "RESERVE_TOKENS", 4)
    ids = chats(4)
    ref = serial(dense, [request(p, 20, f"r{i}") for i, p in enumerate(ids)])
    spy = Spy(monkeypatch)
    sched = scheduler(dense, max_batch=8)
    regrow, hit = sched._regrow, []

    def tight(row):
        if row.group is not None and not hit and row.pos >= row.tokens:
            hit.append(row.req.id)
            budget, sched.budget = sched.budget, sched.reserved - sched.prefill_reserve
            regrow(row)
            sched.budget = budget
            return
        regrow(row)
    sched._regrow = tight
    got = run(sched, [request(p, 20, f"r{i}") for i, p in enumerate(ids)], on_tick=spy.tick)
    victim, = hit
    toks, _, ev = got[victim]
    assert ev == ("done", "length") and len(toks) < 20 and toks == ref[victim][0][:len(toks)]
    for k in ref:
        if k != victim:
            assert got[k] == ref[k], k
    assert any(f"id={victim} ended at" in m for m in caplog.messages)
    assert sched.groups == [] and sched.reserved == 0


def test_join_failure_leaves_row_ungrouped(dense, casc, monkeypatch, caplog):
    """An exception while a row joins a group (here inside PrefixGroup.add)
    is logged; the row decodes ungrouped, the group is unchanged (_group_add
    is atomic: pg and g.rows agree), a later row joins it, every answer is
    the request's own."""
    caplog.set_level(logging.ERROR, logger="qwen.worker")
    ids = chats(5)
    ref = serial(dense, [request(p, 16, f"r{i}") for i, p in enumerate(ids)])
    add, left = cascade.PrefixGroup.add, [1]

    def bad_add(self, *a, **k):
        if left[0]:
            left[0] -= 1
            raise RuntimeError("add boom")
        return add(self, *a, **k)
    monkeypatch.setattr(cascade.PrefixGroup, "add", bad_add)
    spy = Spy(monkeypatch)
    reqs = [request(p, 16, f"r{i}") for i, p in enumerate(ids)]
    got = run(scheduler(dense, max_batch=8), reqs, arrive=one_per_tick(reqs), on_tick=spy.tick)
    assert got == ref
    assert any("id=r3 stays ungrouped" in m for m in caplog.messages)
    grouped = {rid for g in spy.groups for m in g for rid in m}
    assert grouped == {"r0", "r1", "r2", "r4"}


@pytest.mark.parametrize("check", [True, False], ids=["check", "nocheck"])
def test_back_to_back_joins(dense, casc, monkeypatch, check):
    """Five rows prefilled back to back: the group forms at the 3rd join and
    the 4th and 5th are added with no decode step in between (each add's
    copies stack lazily on the previous one's); every answer is the
    request's own."""
    monkeypatch.setattr(worker, "CASCADE_CHECK", check)
    ids = chats(5)
    ref = serial(dense, [request(p, 16, f"r{i}") for i, p in enumerate(ids)])
    spy = Spy(monkeypatch)
    calls = []
    add = worker._Scheduler._group_add

    def spy_add(sched, g, row):
        calls.append(spy.calls)
        return add(sched, g, row)
    monkeypatch.setattr(worker._Scheduler, "_group_add", spy_add)
    got = run(scheduler(dense, max_batch=8), [request(p, 16, f"r{i}") for i, p in enumerate(ids)],
              on_tick=spy.tick)
    assert got == ref
    assert len(spy.builds) == 1 and calls == [0, 0] and spy.calls > 0


def test_tidy_failure_in_join_fails_the_group_only(dense, casc, monkeypatch, caplog):
    """A tidy that raises inside a join (a release, e.g. an OOM) fails that
    group's live rows -- their KV is only in the group -- and drops it; the
    joining request and the rest of the batch go on."""
    caplog.set_level(logging.ERROR, logger="qwen.worker")
    ids = chats(3)
    other = prompts([20], 85)[0]
    ref = serial(dense, [request(other, 10, "j")])
    sched = scheduler(dense, max_batch=8)
    join = sched._join

    def joining(row):
        if row.req.id == "j":
            assert sched.groups and len(sched.groups[0].rows) == 3
            sched._finish(row_of(sched, "r0"), "cancelled")     # 2 members left: dissolve
            monkeypatch.setattr(cascade.PrefixGroup, "release",
                                lambda *a, **k: (_ for _ in ()).throw(RuntimeError("release boom")))
        join(row)
    sched._join = joining
    reqs = [request(p, 40, f"r{i}") for i, p in enumerate(ids)] + [request(other, 10, "j")]
    got = run(sched, reqs, arrive={12: [reqs[3]]}, on_tick=lambda t, s: check_accounting(s))
    assert got["j"] == ref["j"]
    assert got["r0"][2] == ("done", "cancelled")
    assert got["r1"][2] == got["r2"][2] == ("error", "RuntimeError", "release boom")
    assert any("tidy failed" in m for m in caplog.messages)
    assert sched.groups == [] and sched.reserved == 0


def test_set_cascade_off_releases_in_stages(dense, casc, monkeypatch, caplog):
    """set_cascade(False) releases one member per group and decode step
    (the cheapest first), not every member at once; the last three go
    together (< 3 would stay); every answer is the request's own."""
    caplog.set_level(logging.INFO, logger="qwen.worker")
    ids = chats(6)
    ref = serial(dense, [request(p, 24, f"r{i}") for i, p in enumerate(ids)])
    spy = Spy(monkeypatch)
    sched = scheduler(dense, max_batch=8)
    sizes = []

    def tick(t, s):
        spy.tick(t, s)
        if s.cascade and s.groups and len(s.groups[0].rows) == 6 and t > 30:
            s.set_cascade(False)
        if not s.cascade:
            sizes.append(len(s.groups[0].rows) if s.groups else 0)
    got = run(sched, [request(p, 24, f"r{i}") for i, p in enumerate(ids)], on_tick=tick)
    assert got == ref
    end = sizes.index(0)
    assert sizes[0] == 5 and set(sizes[end:]) == {0} and sizes[end - 1] == 3, sizes
    steps = [a - b for a, b in zip(sizes[:end], sizes[1:end])]
    assert all(0 <= d <= 1 for d in steps), sizes           # one per decode step
    assert sum(1 for m in caplog.messages if "left (switched off)" in m) >= 2
    assert any("dissolved (switched off)" in m for m in caplog.messages)
