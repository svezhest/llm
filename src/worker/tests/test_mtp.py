"""The MTP head as a speculative drafter (qwen_worker/mtp.py) on the tiny random
model (CPU): a real mlx-vlm Qwen3_5MTPDraftModel (random weights) bound to the
tiny target. Greedy output == the plain loop's for every draft count, with
the head's own drafts (random: nearly all rejected) and with an oracle that
proposes the known answer (accepted prefixes, rejections, the chained
entries kept or recomputed); the head's KV cache after a request == the head
run once over every (target hidden, next token) pair; hybrid with a DFlash
stand-in; the thinking budget; the prompt cache's drafter context."""
import mlx.core as mx
import math

import pytest

import worker
from conftest import COMMON, FakeTok, prompts, request, run, scheduler
from test_scheduler import FakeDraft, no_metal_verify


def head(lm):
    from mlx_vlm.speculative.drafters.qwen3_5_mtp import ModelConfig, Model
    mx.random.seed(5)
    m = Model(ModelConfig.from_dict({"model_type": "qwen3_5_mtp", "text_config": dict(
        model_type="qwen3_5_text", intermediate_size=256, mtp_num_hidden_layers=1, **COMMON)}))
    m.apply(lambda a: a.astype(mx.float32))
    mx.eval(m.parameters())
    return m


class Oracle(worker.MTPDraft):
    """The real head (its cache, chained entries) but the drafts are the
    known answer `full` (prompt + plain answer), the last one wrong every
    other round."""

    def __init__(self, model, lm, full, wrong=True):
        super().__init__(model, lm)
        self.full, self.wrong = full, wrong

    def draft(self, n):
        super().draft(n)
        at = self.pos - self.appended + 1       # the position the seed predicts
        d = (self.full[at:at + n] + [0] * n)[:n]
        if self.wrong and len(self.accept_lens) % 2:
            d[-1] = (d[-1] + 1) % 400
        return mx.array([d], dtype=mx.int32)


@pytest.fixture
def spec(monkeypatch):
    no_metal_verify(monkeypatch)


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    """The process-wide round times (mtp.CLOCK): a fresh table per test."""
    c = worker.Clock()
    monkeypatch.setattr(worker, "CLOCK", c)
    return c


def plain(lm, ids, n, **kw):
    return run(scheduler(lm, max_batch=1), [request(ids, n, "a", **kw)])["a"]


def with_mtp(lm, ids, n, mt, draft=None, **kw):
    sched = scheduler(lm, max_batch=1)
    sched.eng.mtp, sched.eng.draft = mt, draft
    return run(sched, [request(ids, n, "a", **kw)])["a"]


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_own_drafts_exact(dense, spec, monkeypatch, k):
    monkeypatch.setattr(worker, "MTP_K", k)
    for ids in prompts([25, 3, 40], 30):
        ref = plain(dense, ids, 24)
        mt = worker.MTPDraft(head(dense), dense)
        assert with_mtp(dense, ids, 24, mt) == ref
        assert mt.accept_lens and max(mt.draft_lens) == k   # (the last round: cut to max_tokens)


@pytest.mark.parametrize("refeed", [False, True])
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_oracle_exact_and_the_heads_cache(dense, spec, monkeypatch, k, refeed):
    """Accepted prefixes and rejections: the answer is plain greedy's; with
    QWEN_MTP_REFEED every head entry is the head over the target's real
    hidden state and the next token, as one forward over the whole sequence
    gives (the chained entries never survive)."""
    monkeypatch.setattr(worker, "MTP_K", k)
    monkeypatch.setattr(worker, "MTP_REFEED", refeed)
    ids = prompts([25], 31)[0]
    ref = plain(dense, ids, 30)
    mt = Oracle(head(dense), dense, ids + ref[0])
    assert with_mtp(dense, ids, 30, mt) == ref
    assert max(mt.accept_lens) == k and min(mt.accept_lens) < k   # (k = 1: wrong every other)
    full = ids + ref[0]
    n = mt.cache[0].offset
    assert n >= len(full) - 2 * (k + 1)
    if refeed:
        h = dense.model(mx.array([full[:n]]), cache=dense.make_cache())   # post-norm
        from mlx_vlm.models.cache import KVCache
        c = [KVCache()]
        mt.model._forward_hidden(dense.model.embed_tokens(mx.array([full[1:n + 1]])), h, c,
                                 mx.arange(n)[None])
        assert mx.allclose(c[0].keys[..., :n, :], mt.cache[0].keys[..., :n, :], atol=1e-4).item()
        assert mx.allclose(c[0].values[..., :n, :], mt.cache[0].values[..., :n, :], atol=1e-4).item()


def test_hybrid_exact_takes_both(dense, spec, monkeypatch):
    """Both drafters (QWEN_DRAFT_HYBRID=1): each round one of them; DFlash's
    context collects the MTP rounds' positions; plain greedy's answer."""
    monkeypatch.setattr(worker, "MTP_K", 2)
    ids = prompts([25], 32)[0]
    ref = plain(dense, ids, 60)
    seen = []

    class Rec(FakeDraft):
        def draft_block(self, b, hidden, cache, bs, sampler, dtype):
            seen.append(hidden.shape)
            # FakeDraft counts emitted tokens from its own rounds only
            return mx.array([(ref[0][len(emitted):] + [0] * bs)[:bs - 1]], dtype=dtype)

    emitted = []
    mt = Oracle(head(dense), dense, ids + ref[0])
    orig = mt.emit
    mt.emit = lambda t: (emitted.append(t), orig(t))
    d = Rec(ref[0])
    out = with_mtp(dense, ids, 60, mt, draft=d)
    assert out == ref
    assert mt.accept_lens and d.accept_lens
    W = (len(d.config.target_layer_ids) + 0) * COMMON["hidden_size"]
    assert all(s[-1] == W for s in seen)                 # DFlash gets its layers only
    for force in ("dflash", "mtp"):
        monkeypatch.setattr(worker, "DRAFT_HYBRID", force)
        emitted.clear()
        mt = Oracle(head(dense), dense, ids + ref[0])
        orig = mt.emit
        mt.emit = lambda t, orig=orig: (emitted.append(t), orig(t))
        d = Rec(ref[0])
        assert with_mtp(dense, ids, 60, mt, draft=d) == ref
        assert bool(mt.accept_lens) == (force == "mtp") and bool(d.accept_lens) == (force == "dflash")


class Head:
    """MTPDraft's side of the chooser: expect() / learn() / evidence()."""

    def __init__(self, tok, ms, e=10.0):
        self.tok, self.ms, self.e, self.learned, self.now = tok, ms, e, [], 0.0

    def expect(self):
        return self.tok, self.ms

    def evidence(self):
        return self.e

    def learn(self, accepted, drafted):
        self.learned.append((accepted, drafted))


def test_clock_follows_the_drift(clock):
    """A kind that is not measured follows the machine's drift measured on
    the others; a pause is clamped."""
    clock.add("dflash", 180.0)
    clock.add("plain", 80.0)
    assert clock.ms("dflash") == 180.0 and clock.ms("plain") == 80.0 and clock.ms(("mtp", 1)) is None
    for _ in range(60):
        clock.add("plain", 104.0)                        # the machine 30 % slower
    assert clock.ms("plain") == pytest.approx(104.0, rel=0.01)
    assert clock.ms("dflash") == pytest.approx(180.0 * 1.3, rel=0.08)
    before = clock.ms("plain")
    clock.add("plain", 5000.0)                           # a pause: one bounded step
    assert clock.ms("plain") < before * 2 ** 0.26


def test_clock_per_request_class(clock):
    """Round times per class of requests (context size, sampling, budget): a
    new class takes the times of one with the same context size, else none."""
    a = clock.at((0, False, False))
    assert a is clock.at((0, False, False)) and a is not clock and a.ms("plain") is None
    a.add("plain", 80.0)
    a.add("dflash", 180.0)
    b = clock.at((0, True, False))                       # sampling, the same context class
    assert b is not a and b.ms("plain") == 80.0 and b.ms("dflash") == 180.0
    b.add("plain", 160.0)
    assert a.ms("plain") == 80.0 and b.ms("plain") > 80.0
    c = clock.at((2, False, False))                      # a long context: its own probes
    assert c.ms("plain") is None and c.ms("dflash") is None


def test_avg_weights_samples_by_age():
    """The mean with a prior; old samples count less, `e` = what it knows of now."""
    a = worker.Avg(3.0, 2)
    assert a.mean == 3.0 and a.e == 0.0
    a.add(6.0, 0.0, 1000.0)
    assert a.mean == pytest.approx(4.0) and a.e == 1.0
    a.add(6.0, 0.0, 1000.0)
    assert a.mean == pytest.approx(4.5) and a.e == 2.0
    assert a.age(1000.0, 1000.0) == pytest.approx(2.0 / 2.718281828)
    assert a.mean == pytest.approx(4.5)                  # ageing alone moves nothing
    a.add(1.0, 5000.0, 1000.0)                           # the old ones are almost gone
    assert a.mean == pytest.approx(1.0, abs=0.1) and a.e == pytest.approx(1.0, abs=0.05)


def test_chooser_rate_margin_and_probe(clock):
    """An arm never timed is taken PROBE rounds; a request starts on the head;
    it is left only for an arm MARGIN better that is known from recent samples
    (a plain step needs none)."""
    c = worker.Chooser(("dflash", "mtp", "plain"), Head(1.8, None))
    P = c.PROBE
    got = []
    for _ in range(P):                                   # nothing timed: dflash first
        got.append(c.pick())
        assert c.shadows() == [] and c.steps() == c.SHORT
        c.done(3, 7)
    clock.add("dflash", 180.0)
    for _ in range(P):
        got.append(c.pick())
        c.done(1, 1)
    c.mtp.ms = 115.0
    for _ in range(P):
        got.append(c.pick())
        c.done(0)
    clock.add("plain", 80.0)
    assert got == ["dflash"] * P + ["mtp"] * P + ["plain"] * P

    def dflash(tok, e=10.0):
        c.tok["dflash"].h, c.tok["dflash"].w, c.tok["dflash"].e = tok * 10.0, 10.0, e

    c = worker.Chooser(("dflash", "mtp", "plain"), Head(1.8, 115.0))
    dflash(4.8)                                          # 26.7 vs 15.7 vs 12.5 tok/s
    assert c.pick() == "mtp"                             # the start, whatever the estimates
    assert c.pick() == "dflash"
    dflash(1.9)                                          # 10.6: the head
    assert c.pick() == "mtp"
    dflash(1.8 * 180 / 115 * 1.04)                       # 4 % better: not worth a switch
    assert c.pick() == "mtp"
    dflash(1.8 * 180 / 115 * 1.06, e=1.0)                # better, from one sample: not yet
    assert c.pick() == "mtp" and c.shadows() == ["dflash"]       # (a trial: drafted every round)
    dflash(1.8 * 180 / 115 * 1.06)
    assert c.pick() == "dflash"
    dflash(1.0)
    c.mtp.tok = 1.0                                      # both below a plain step
    assert c.pick() == "plain"
    assert c.shadows() == ["dflash", "mtp"] and c.steps() == c.SHORT     # not drafted yet
    c.shadowed.update(dflash=c.WARM, mtp=c.WARM)
    assert c.pick() == "plain" and c.shadows() == [] and c.steps() == c.BURST
    assert c.pick(force="mtp") == "mtp" and c.shadows() == []
    assert c.switches == 5
    assert worker.Chooser(("dflash", "plain")).pick() == "dflash"


def test_chooser_shadows_cadence(clock):
    """Shadow drafts of the arm not taken: every round until it has WARM
    drafts, then as often as their cost is NEAR (an arm close to the current
    one) or FAR of the time."""
    clock.add("dflash", 180.0)
    c = worker.Chooser(("dflash", "mtp"), Head(1.8, 115.0))
    t, rounds = 0.0, []

    def go(n):
        nonlocal t
        out = []
        for _ in range(n):
            assert c.pick(force=None, now=t) in ("mtp", "dflash")
            c.cur = "dflash"                             # (the test's arm)
            sh = c.shadows()
            out.append(bool(sh))
            c.done(4, 7, shadows=sh)
            for x in sh:
                c.sample(x, 1, 2)
            c.tok["dflash"].h = 4.8 * c.tok["dflash"].w
            t += 180.0
        return out

    c.pick(now=0.0)
    c.cur = "dflash"
    got = go(200)
    assert got[:c.WARM] == [True] * c.WARM               # 15.7 < 0.8 * 26.7 tok/s: far
    far = math.ceil(c.COST["mtp"] * 115.0 / c.FAR / 180.0)       # rounds between two drafts
    assert sum(got[c.WARM:]) == pytest.approx((200 - c.WARM) / far, abs=2)
    c.mtp.tok = 2.9                                      # 25.2: close
    got = go(200)
    near = math.ceil(c.COST["mtp"] * 115.0 / c.NEAR / 180.0)
    assert sum(got) == pytest.approx(200 / near, abs=2) and near < far
    assert c.n["mtp"] == 0 and c.pend["mtp"] == 0 and len(c.mtp.learned) == c.shadowed["mtp"]


def test_chooser_noisy_drafter_keeps_its_rounds(clock):
    """The old chooser's failure: a drafter whose (noisy) tokens per round beat
    a plain step lost its rounds to plain whenever its average dipped, and the
    frozen estimate kept it out. Now the arm not taken is sampled by shadow
    drafts, the means are slow and a switch needs a margin."""
    import random
    rng = random.Random(3)
    clock.add("dflash", 180.0)
    clock.add("plain", 80.0)

    def accepted():                                      # mean 2.1 of 7: 3.1 tok / 180 ms vs 1 / 80
        a = 0
        while a < 7 and rng.random() < 0.69:
            a += 1
        return a

    c = worker.Chooser(("dflash", "plain"))
    t, took = 0.0, {"dflash": 0.0, "plain": 0.0}
    while t < 120e3:
        k = c.pick(now=t)
        sh = c.shadows()
        n = c.steps() if k == "plain" else 1
        c.done(accepted() if k == "dflash" else 0, 7 if k == "dflash" else 0, shadows=sh, steps=n)
        for x in sh:
            c.sample(x, accepted(), 7)
        dt = 180.0 if k == "dflash" else 80.0 * n
        took[k] += dt
        t += dt + 18.0 * len(sh)
    assert took["plain"] < 0.1 * t


def test_chooser_switches_on_shadow_samples(clock):
    """The arm not taken wins by its shadow drafts alone (no round spent on
    it), and loses again when the text changes; a single good draft of it
    starts a trial (drafted every round), not a switch."""
    clock.add("dflash", 180.0)
    c = worker.Chooser(("dflash", "mtp"), Head(1.85, 115.0))     # the head: 16.1 tok/s
    t, took, drafted = 0.0, [], []

    def go(n, dflash_accepts):
        nonlocal t
        for _ in range(n):
            k = c.pick(now=t)
            took.append(k)
            sh = c.shadows()
            drafted.append(bool(sh))
            c.done(dflash_accepts if k == "dflash" else 1, 7 if k == "dflash" else 1, shadows=sh)
            for x in sh:
                if x == "dflash":
                    c.sample(x, dflash_accepts, 7)
            t += 180.0 if k == "dflash" else 115.0

    go(40, 1)                                            # 2 tok / 180 ms = 11.1
    assert set(took) == {"mtp"} and drafted[:c.WARM] == [True] * c.WARM
    assert sum(drafted) <= c.WARM + 2                    # far behind: a draft now and then
    i = len(took)
    go(200, 5)                                           # 6 tok / 180 ms = 33.3 by shadows alone
    first = took.index("dflash")
    assert took[-1] == "dflash" and first - i < 64       # (the next far draft, then the trial)
    j = i + drafted[i:].index(True)
    assert drafted[j:first] == [True] * (first - j) and 2 <= first - j <= 4
    go(40, 0)                                            # 1 tok / 180 ms: the head again
    assert took[-1] == "mtp" and took[-10:].count("dflash") == 0
    assert c.switches == 2


def test_shadows_score_against_the_emitted_tokens():
    s = worker.Shadows()
    for t in (5, 6):
        s.emit(t)
    s.add("dflash", [7, 8, 9])                           # after the anchor 6
    s.add("mtp", [7, 1])
    s.add("mtp", [])                                     # nothing drafted: nothing to score
    assert s.settle() == []
    s.emit(7)
    assert s.settle() == []                              # both still right
    s.emit(8)
    assert s.settle() == [("mtp", 1, 2)]                 # its 2nd draft is wrong
    s.add("mtp", [9, 4])
    s.emit(9)
    assert s.settle() == [("dflash", 3, 3)]              # every position emitted
    s.emit(3)
    assert s.settle() == [("mtp", 1, 2)]
    for t in range(600):                                 # the token window is trimmed
        s.emit(t)
        s.settle()
    s.add("dflash", [600, 601])
    s.emit(600)
    s.emit(0)
    assert s.settle() == [("dflash", 1, 2)] and len(s.toks) < 400


@pytest.mark.parametrize("mode", ["chooser", "all", "refeed"])
def test_hybrid_shadow_drafts_exact(dense, spec, monkeypatch, clock, mode):
    """Shadow drafts (the arm not taken drafts without a verify) leave the
    answer plain greedy's and every drafter's state as its own rounds would:
    the head's cache == the head over the real (hidden, next token) pairs
    (its shadow entries are trimmed), DFlash's context takes every position
    once. "all" (QWEN_SPEC_SHADOW=all + plain forced): both drafters' drafts
    at every position of the answer, in the trace."""
    monkeypatch.setattr(worker, "MTP_K", 2)
    monkeypatch.setattr(worker, "SPEC_TRACE", True)
    monkeypatch.setattr(worker, "MTP_REFEED", mode != "chooser")
    if mode == "chooser":                                # (the request's first arm: the head)
        monkeypatch.setattr(worker.Chooser, "START", ("dflash", "mtp", "plain"))
    if mode == "all":
        monkeypatch.setattr(worker, "SPEC_PLAIN", True)
        monkeypatch.setattr(worker, "DRAFT_HYBRID", "plain")
        monkeypatch.setattr(worker, "SPEC_SHADOW", "all")
    for k, ms in (("dflash", 180.0), (("mtp", 2), 60.0 if mode == "refeed" else 200.0), ("plain", 80.0)):
        clock.add(k, ms)                                 # every arm timed: no probe rounds
    ids = prompts([25], 32)[0]
    ref = plain(dense, ids, 60)
    full = ids + ref[0]
    fed = []

    class Rec(FakeDraft):
        def draft_block(self, b, hidden, cache, bs, sampler, dtype):
            fed.append(hidden.shape[1])
            return mx.array([(ref[0][len(emitted):] + [0] * bs)[:bs - 1]], dtype=dtype)

    emitted = []
    mt = Oracle(head(dense), dense, full, wrong=False)
    d = Rec(ref[0])
    sched = scheduler(dense, max_batch=1)
    sched.eng.mtp, sched.eng.draft = mt, d
    stop = sched.eng.stopping
    sched.eng.stopping = lambda t: emitted.append(t) or stop(t)     # every emitted token
    x = request(ids, 60, "a")
    assert run(sched, [x])["a"] == ref
    tr = x.spec_stats["trace"]
    shadows = [(t[0], s[0], s[1], t[7]) for t in tr for s in t[6]]
    assert shadows
    win = d.config.sliding_window - 1
    if mode == "all":
        assert [t[0] for t in tr] == ["plain"] * (len(ref[0]) - 1)
        assert fed == [win] + [1] * (len(fed) - 1) and len(fed) >= len(tr) - 4   # every position once
        for i, t in enumerate(tr[:-4]):                  # both drafters drafted after every token
            assert dict((a, s) for a, s in t[6]) == {"dflash": ref[0][i + 1:i + 4], "mtp": ref[0][i + 1:i + 5]}
    else:
        took = {t[0] for t in tr}
        assert took == ({"mtp"} if mode == "refeed" else {"dflash"})     # the other arm: shadows only
        if mode == "chooser":                            # the context: the positions each round kept
            assert fed == [win] + [t[2] + 1 for t in tr[:-1]]
        else:                                            # ... since the last shadow draft
            at = [i for i, t in enumerate(tr) if t[6]]
            assert fed == [win] + [sum(t[2] + 1 for t in tr[a:b]) for a, b in zip(at, at[1:])]
        assert {s[1] for s in shadows} == {"dflash", "mtp"} - took
        assert bool(d.accept_lens) == (mode == "chooser") and bool(mt.accept_lens) == (mode == "refeed")
    if mode != "chooser":                                # real entries only: the head over the pairs
        n = mt.cache[0].offset
        assert n >= len(full) - 6
        h = dense.model(mx.array([full[:n]]), cache=dense.make_cache())
        from mlx_vlm.models.cache import KVCache
        c = [KVCache()]
        mt.model._forward_hidden(dense.model.embed_tokens(mx.array([full[1:n + 1]])), h, c,
                                 mx.arange(n)[None])
        assert mx.allclose(c[0].keys[..., :n, :], mt.cache[0].keys[..., :n, :], atol=1e-4).item()
        assert mx.allclose(c[0].values[..., :n, :], mt.cache[0].values[..., :n, :], atol=1e-4).item()


@pytest.mark.parametrize("arms", ["mtp", "dflash", "both"])
def test_plain_arm_exact(dense, spec, monkeypatch, arms):
    """QWEN_SPEC_PLAIN: plain steps mixed into the rounds (each arm is taken
    at least once: fresh estimates first); the answer is plain greedy's."""
    monkeypatch.setattr(worker, "SPEC_PLAIN", True)
    monkeypatch.setattr(worker, "MTP_K", 2)
    ids = prompts([25], 36)[0]
    ref = plain(dense, ids, 40)
    mt = Oracle(head(dense), dense, ids + ref[0]) if arms != "dflash" else None
    emitted = []

    class Rec(FakeDraft):
        def draft_block(self, b, hidden, cache, bs, sampler, dtype):
            return mx.array([(ref[0][len(emitted):] + [0] * bs)[:bs - 1]], dtype=dtype)
    d = Rec(ref[0]) if arms != "mtp" else None
    taken = []
    pick = worker.Chooser.pick
    monkeypatch.setattr(worker.Chooser, "pick", lambda self, force=None, now=None: taken.append(pick(self, force, now))
                        or taken[-1])
    sched = scheduler(dense, max_batch=1)
    sched.eng.mtp, sched.eng.draft = mt, d
    stop = sched.eng.stopping
    sched.eng.stopping = lambda t: emitted.append(t) or stop(t)     # every emitted token
    assert run(sched, [request(ids, 40, "a")])["a"] == ref
    assert "plain" in taken and len(set(taken)) == (3 if arms == "both" else 2)


def forced(monkeypatch, order, took):
    """The chooser takes the arms of `order` ("d" / "m" / "p") in turn, with
    shadow drafts of both drafters every round."""
    seq = {"d": "dflash", "m": "mtp", "p": "plain"}

    def pick(self, force=None, now=None):
        self.rounds += 1
        self.cur = seq[order[len(took) % len(order)]]
        self.r = {}
        took.append(self.cur)
        return self.cur
    monkeypatch.setattr(worker.Chooser, "pick", pick)
    monkeypatch.setattr(worker.Chooser, "shadows",
                        lambda self: [x for x in ("dflash", "mtp") if x != self.cur])


@pytest.mark.parametrize("order", ["p", "pmd", "dpm", "ppd"])
@pytest.mark.parametrize("burst", [1, 3, 8])
def test_plain_bursts_penalties_exact(dense, spec, monkeypatch, order, burst):
    """The plain arm's bursts (step-ahead: a token is fed before it is read)
    between drafter rounds, with penalties (a fed token is counted once: by
    the step that feeds it, or at its emission when the burst ends on it) -
    the plain loop's tokens, greedy with a presence + repetition penalty."""
    from test_kvq8 import spec_arms
    monkeypatch.setattr(worker, "MTP_K", 2)
    monkeypatch.setattr(worker, "SPEC_PLAIN", True)
    monkeypatch.setattr(worker, "DRAFT_SAMPLING", True)
    monkeypatch.setattr(worker.Chooser, "BURST", burst)
    took = []
    forced(monkeypatch, order, took)
    ids = prompts([25], 33)[0]
    kw = dict(presence_penalty=1.5, repetition_penalty=1.3, frequency_penalty=0.2)
    ref = plain(dense, ids, 40, **kw)
    assert ref != plain(dense, ids, 40)                  # the penalties move the answer
    sched = scheduler(dense, max_batch=1)
    d, mt = spec_arms(sched, ref[0], ids, ("dflash", "mtp"))
    x = request(ids, 40, "a", **kw)
    assert run(sched, [x])["a"] == ref
    assert x.spec_stats["plain_steps"] >= (39 if order == "p" else 4)
    assert ("d" not in order or d.accept_lens) and ("m" not in order or mt.accept_lens)


def test_sampling_takes_every_arm(dense, spec, monkeypatch):
    """A sampling request through the three arms (QWEN_DRAFT=auto: sampling
    requests are speculative): it runs to its length; the same seed gives the
    same tokens again (the draws: the request's key, split per call)."""
    from test_kvq8 import spec_arms
    monkeypatch.setattr(worker, "MTP_K", 2)
    monkeypatch.setattr(worker, "SPEC_PLAIN", True)
    monkeypatch.setattr(worker, "DRAFT_SAMPLING", True)
    monkeypatch.setattr(worker.Chooser, "BURST", 3)
    ids = prompts([25], 34)[0]
    kw = dict(temperature=0.8, top_p=0.95, top_k=20, seed=11)
    ref = plain(dense, ids, 40, temperature=0.0)
    outs = []
    for _ in range(2):
        took = []
        forced(monkeypatch, "dmp", took)
        sched = scheduler(dense, max_batch=1)
        spec_arms(sched, ref[0], ids, ("dflash", "mtp"))
        x = request(ids, 40, "a", **kw)
        outs.append(run(sched, [x])["a"])
        assert len(outs[-1][0]) == 40 and {"dflash", "mtp", "plain"} <= set(took)
        assert x.spec_stats["plain_steps"] >= 6
    assert outs[0] == outs[1] and outs[0] != ref


def test_auto_drafters(tmp_path):
    """QWEN_DRAFT=auto: the drafter dirs next to the model dir, by their
    config.json (target hidden size / layers / vocabulary); explicit dirs and
    "none" win; a model with one drafter or none gets what there is."""
    import json
    from qwen_worker.engine import _auto_drafters

    def put(name, cfg):
        d = tmp_path / name
        d.mkdir()
        (d / "config.json").write_text(json.dumps(cfg))
        return str(d)

    text = dict(hidden_size=5120, num_hidden_layers=64, vocab_size=248320)
    model = dict(model_type="qwen3_5", text_config=text)
    big = put("m27-mlx-4bit", model)
    put("m27-mlx-5bit", model)                           # another build of the model: no drafter
    d27 = put("m27-dflash2", dict(model_type="qwen3", hidden_size=5120, num_target_layers=64,
                                  vocab_size=248320, dflash_config=dict(block_size=8)))
    put("a-dflash", dict(model_type="qwen3", hidden_size=5120, num_target_layers=64,
                         vocab_size=248320, dflash_config=dict(block_size=8)))
    put("m9-dflash", dict(model_type="qwen3", hidden_size=4096, num_target_layers=32,
                          vocab_size=248320, dflash_config=dict(block_size=16)))
    h27 = put("m27-mtp", dict(model_type="qwen3_5_mtp", text_config=text))
    put("m9-mtp", dict(model_type="qwen3_5_mtp", text_config=dict(text, hidden_size=4096, num_hidden_layers=32)))
    (tmp_path / "notes").mkdir()                         # no config.json
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "config.json").write_text("{")
    assert _auto_drafters(big, model) == (d27, h27)      # the longest common name prefix, not "a-dflash"
    assert _auto_drafters(big + "/", model) == (d27, h27)
    assert _auto_drafters(big, model, dflash="none") == ("", h27)
    assert _auto_drafters(big, model, dflash="/x/d", mtp="none") == ("/x/d", "")
    small = dict(hidden_size=4096, text_config=dict(text, hidden_size=4096, num_hidden_layers=32))
    m9 = put("m9-mlx-8bit", small)
    assert _auto_drafters(m9, small) == (str(tmp_path / "m9-dflash"), str(tmp_path / "m9-mtp"))
    moe = dict(text_config=dict(text, hidden_size=2048, num_hidden_layers=40))
    assert _auto_drafters(put("m35", moe), moe) == ("", "")
    assert _auto_drafters(str(tmp_path / "nowhere" / "m"), model) == ("", "")


def test_auto_setting(dense, spec, monkeypatch):
    """QWEN_DRAFT=auto's defaults (config._auto) and what a request's
    spec_stats say of its rounds."""
    from qwen_worker import config
    assert config._auto("QWEN_NO_SUCH_KNOB", "a", "d") == ("a" if config.DRAFT == "auto" else "d")
    monkeypatch.setattr(config, "DRAFT", "auto")
    assert config._auto("QWEN_NO_SUCH_KNOB", "a", "d") == "a"
    monkeypatch.setenv("QWEN_NO_SUCH_KNOB", "x")
    assert config._auto("QWEN_NO_SUCH_KNOB", "a", "d") == "x"
    from test_kvq8 import spec_arms
    monkeypatch.setattr(worker, "SPEC_PLAIN", True)
    took = []
    forced(monkeypatch, "dmp", took)
    ids = prompts([25], 35)[0]
    ref = plain(dense, ids, 40)
    sched = scheduler(dense, max_batch=1)
    spec_arms(sched, ref[0], ids, ("dflash", "mtp"))
    x = request(ids, 40, "a")
    assert run(sched, [x])["a"] == ref
    arms = x.spec_stats["arms"]
    assert set(arms) == {"dflash", "mtp", "plain"}
    # every token but the first, once (+ the stop token, read but not emitted)
    assert sum(v[1] for v in arms.values()) == len(ref[0]) - 1 + ("stop" in ref[2])
    assert sum(v[0] for v in arms.values()) == len(took)
    assert set(x.spec_stats["shadows"]) == {"dflash", "mtp", "plain"}


@pytest.mark.parametrize("plain_arm", [False, True])
def test_thinking_budget(dense, spec, monkeypatch, plain_arm):
    """A thinking budget inside MTP rounds: the same tokens as the plain loop
    with that budget (the forced tokens go through plain steps; the head's
    pending position takes the forced token as its successor)."""
    from mlx_vlm.utils import ThinkingBudgetCriteria
    monkeypatch.setattr(worker, "MTP_K", 3)
    monkeypatch.setattr(worker, "SPEC_PLAIN", plain_arm)
    NL, CLOSE, OPEN = 11, 12, 13
    tok = FakeTok({"</think>": [CLOSE], "<think>": [OPEN], "\n": [NL]})
    p = prompts([25], 30)[0] + [OPEN]

    def crit(budget):
        return ThinkingBudgetCriteria(tokenizer=tok, thinking_budget=budget, thinking_end_token="</think>",
                                      thinking_start_token="<think>", enable_thinking=True,
                                      prompt_preopens_thinking=True)

    free = plain(dense, p, 24)
    rounds = 0
    for budget, n in [(0, 24), (1, 24), (2, 24), (3, 24), (5, 24), (9, 24), (3, 5), (40, 24)]:
        ref = plain(dense, p, n, thinking=crit(budget))
        for full in (free[0], ref[0]):
            mt = Oracle(head(dense), dense, p + full, wrong=False)
            assert with_mtp(dense, p, n, mt, thinking=crit(budget)) == ref, (budget, n)
            rounds += len(mt.accept_lens)
    assert rounds > 20


@pytest.mark.parametrize("order", ["p", "pm", "dpm"])
@pytest.mark.parametrize("burst", [1, 2, 8])
def test_plain_bursts_thinking_budget(dense, spec, monkeypatch, order, burst):
    """A thinking budget inside the plain arm's bursts: the burst ends at the
    token after which the budget forces one (the step queued ahead has fed
    it: its own token is dropped), the forced tokens follow; the plain loop's
    tokens with that budget."""
    from mlx_vlm.utils import ThinkingBudgetCriteria
    from test_kvq8 import spec_arms
    monkeypatch.setattr(worker, "MTP_K", 2)
    monkeypatch.setattr(worker, "SPEC_PLAIN", True)
    monkeypatch.setattr(worker.Chooser, "BURST", burst)
    NL, CLOSE, OPEN = 11, 12, 13
    tok = FakeTok({"</think>": [CLOSE], "<think>": [OPEN], "\n": [NL]})
    p = prompts([25], 30)[0] + [OPEN]

    def crit(budget):
        return ThinkingBudgetCriteria(tokenizer=tok, thinking_budget=budget, thinking_end_token="</think>",
                                      thinking_start_token="<think>", enable_thinking=True,
                                      prompt_preopens_thinking=True)

    steps = 0
    for budget, n in [(0, 24), (1, 24), (2, 24), (3, 24), (4, 24), (5, 24), (9, 24), (3, 5), (40, 24)]:
        ref = plain(dense, p, n, thinking=crit(budget))
        took = []
        forced(monkeypatch, order, took)
        sched = scheduler(dense, max_batch=1)
        spec_arms(sched, ref[0], p, ("dflash", "mtp"))
        x = request(p, n, "a", thinking=crit(budget))
        assert run(sched, [x])["a"] == ref, (budget, n)
        steps += x.spec_stats["plain_steps"]
    assert steps > 30


def test_sampling_exact_distribution_path(dense, spec, monkeypatch):
    """QWEN_DRAFT_SAMPLING with MTP drafts: the seeded sample-match answer
    equals the plain loop's seeded answer (the drafts only decide how many
    target samples a round keeps)."""
    monkeypatch.setattr(worker, "DRAFT_SAMPLING", True)
    monkeypatch.setattr(worker, "MTP_K", 3)
    ids = prompts([25], 33)[0]
    kw = dict(temperature=0.7, top_p=0.95, top_k=20, seed=11)
    ref = plain(dense, ids, 24, **kw)
    mt = Oracle(head(dense), dense, ids + ref[0], wrong=False)
    out = with_mtp(dense, ids, 24, mt, **kw)
    assert mt.accept_lens
    # the spec loop draws its samples per verify row (a different RNG walk
    # than the plain loop's): only the length and the vocabulary range hold
    assert len(out[0]) == len(ref[0]) and all(0 <= t < 512 for t in out[0])


def test_prompt_cache_context(dense, spec, monkeypatch):
    """With the prompt cache: the entry's states carry the final hidden
    states (the head's context, QWEN_DRAFT_TAIL), an exact repeat starts the
    head from them, and the answer is the cold one."""
    from test_prompt_cache import small  # noqa: F401
    monkeypatch.setattr(worker, "PROMPT_CACHE", True)
    monkeypatch.setattr(worker, "PROMPT_CACHE_REPLY", False)
    monkeypatch.setattr(worker, "PREFILL_STEP", 8)
    monkeypatch.setattr(worker, "PREFILL_MIN_STEP", 4)
    monkeypatch.setattr(worker, "DRAFT_TAIL", 16)
    monkeypatch.setattr(worker, "DRAFT_TAIL_END", 16)
    monkeypatch.setattr(worker, "MTP_K", 2)
    ids = prompts([60], 34)[0]
    ref = plain(dense, ids, 16)
    sched = scheduler(dense, max_batch=1)
    sched.eng.mtp = mt = Oracle(head(dense), dense, ids + ref[0])
    assert run(sched, [request(ids, 16, "x")])["x"] == ref
    (e,) = sched.eng._pcache.entries
    t = worker._state_tail(e.end, e.layout)
    assert t is not None and t.shape == (1, 16, COMMON["hidden_size"])
    starts = []
    orig = mt.start
    mt.start = lambda at: (starts.append(at), orig(at))
    x = request(ids, 16, "y")
    assert run(sched, [x])["y"] == ref
    assert x.reused == len(ids) - 1 and starts == [len(ids) - 1 - 16]


def test_adaptive_k(dense, spec, monkeypatch):
    """QWEN_MTP_K=0: high acceptance -> the deepest block, none -> one draft
    (+ one every EXPLORE-th round); the answer stays plain greedy's."""
    monkeypatch.setattr(worker, "MTP_K", 0)
    mt = worker.MTPDraft(head(dense), dense)
    mt.start(0)
    for _ in range(20):
        mt.record(4, 4)
    assert mt.next_k() == 4
    mt.start(0)
    for _ in range(19):
        mt.record(0, mt.next_k())
    assert mt.next_k() == 1
    for _ in range(4):
        mt.record(0, 1)
    assert mt.next_k() == 2                              # the 24th round explores
    # measured round times (the 27B's: 2-token verify on exact qmv, 3.. on mrqmm)
    T4 = {("mtp", 1): 66.0, ("mtp", 2): 88.0, ("mtp", 3): 92.0, ("mtp", 4): 97.0}
    monkeypatch.setattr(worker, "CLOCK", worker.Clock(T4))
    mt.start(0)
    mt.p, mt.seen = [0.46] * 4, [1] * 4
    assert mt.next_k() == 1                              # 1.46 / 66 > 1.77 / 92
    mt.p = [0.85] * 4
    assert mt.next_k() == 4                              # 3.70 / 97 > 1.85 / 66
    mt.p, mt.seen = [0.9, 0.5, 0.5, 0.5], [9, 0, 0, 0]                  # only k = 1 drafted so far:
    assert mt.next_k() == 4                              # deeper depths inherit 0.9, not the prior
    for x in mt.st:                                      # the means over TAU: the chooser's side
        x.h, x.w = 0.9, 1.0
    assert mt.expect() == (pytest.approx(1 + 0.9 + 0.81 + 0.729 + 0.6561), pytest.approx(97.0))
    monkeypatch.setattr(worker, "CLOCK", worker.Clock({("mtp", 1): 66.0}))
    assert mt._ms(3) == pytest.approx(66.0 * (1 + 3 * worker.MTP_COST) / (1 + worker.MTP_COST))
    ids = prompts([25], 35)[0]
    ref = plain(dense, ids, 40)
    mt = Oracle(head(dense), dense, ids + ref[0], wrong=False)
    assert with_mtp(dense, ids, 40, mt) == ref
    assert max(mt.draft_lens) == 4
    # QWEN_MTP_KMAX: the ceiling of the adaptive count
    monkeypatch.setattr(worker, "MTP_KMAX", 7)
    monkeypatch.setattr(worker, "CLOCK", worker.Clock())
    mt = worker.MTPDraft(head(dense), dense)
    mt.start(0)
    for _ in range(20):
        mt.record(7, 7)
    assert mt.KMAX == 7 and len(mt.p) == 7 and mt.next_k() == 7


def test_token_bytes(dense):
    """The head's own fp16 KV per position (charged to fp16 rows by the scheduler)."""
    from conftest import COMMON
    mt = worker.MTPDraft(head(dense), dense)
    kv, hd = COMMON["num_key_value_heads"], COMMON["head_dim"]
    assert mt.token_bytes == 2 * 2 * kv * hd * 1
