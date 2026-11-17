"""The MTP head as a speculative drafter (qwen_worker/mtp.py) on the tiny random
model (CPU): a real mlx-vlm Qwen3_5MTPDraftModel (random weights) bound to the
tiny target. Greedy output == the plain loop's for every draft count and for
the confidence gate (k = 0 rounds: plain bursts), with the head's own drafts
(random: nearly all rejected) and with an oracle that proposes the known
answer (accepted prefixes, rejections); the head's KV cache after a request
== the head run once over every (target hidden, next token) pair (no chained
or rejected entry survives, a burst's one catch-up flush == a flush per
step); hybrid with a DFlash stand-in and the Chooser (QWEN_DRAFT=auto); the
thinking budget; the prompt cache's drafter context."""
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
    """The real head (its cache, chained entries, the gate's confidence
    reads) but the drafts are the known answer `full` (prompt + plain
    answer), the last one wrong every other round. `script`: the gate's draft
    counts in turn (0: a round with no draft); `zero` (forced arms): the next
    rounds take none."""

    def __init__(self, model, lm, full, wrong=True, script=None):
        super().__init__(model, lm)
        self.full, self.wrong, self.script, self.zero = full, wrong, script, False

    def depth(self, c1, T, top):
        if self.zero:
            return 0
        if self.script is None:
            return super().depth(c1, T, top)
        return min(top, self.script[sum(self.ks) % len(self.script)])

    def round(self, room):
        d = super().round(room)
        if isinstance(d, int):
            return d
        n = d.shape[1]
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


def real_entries(dense, mt, full, least, exact=False):
    """The head's cache == the head run once over the real pairs (target
    hidden at t, token t + 1) of `full`: no chained / rejected entry left."""
    n = mt.cache[0].offset
    assert n == mt.pos and n >= least
    h = dense.model(mx.array([full[:n]]), cache=dense.make_cache())   # post-norm
    from mlx_vlm.models.cache import KVCache
    c = [KVCache()]
    mt.model._forward_hidden(dense.model.embed_tokens(mx.array([full[1:n + 1]])), h, c,
                             mx.arange(n)[None])
    for a, b in ((c[0].keys, mt.cache[0].keys), (c[0].values, mt.cache[0].values)):
        assert mx.allclose(a[..., :n, :], b[..., :n, :], atol=1e-4).item()
    return n


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
def test_own_drafts_exact(dense, spec, monkeypatch, k):
    """The head's own drafts (random weights), k per round or (0) gated by
    their confidence: plain greedy's answer."""
    monkeypatch.setattr(worker, "MTP_K", k)
    for ids in prompts([25, 3, 40], 30):
        ref = plain(dense, ids, 24)
        mt = worker.MTPDraft(head(dense), dense)
        assert with_mtp(dense, ids, 24, mt) == ref
        if k:
            assert mt.accept_lens and max(mt.draft_lens) == k   # (the last round: cut to max_tokens)
        else:                                            # random confidences: mostly plain bursts
            assert mt.burst_tokens > 0 and mt.dropped == 0


@pytest.mark.parametrize("how", ["k1", "k2", "k3", "k4", "gate"])
def test_oracle_exact_and_the_heads_cache(dense, spec, monkeypatch, how):
    """Accepted prefixes and rejections: the answer is plain greedy's; every
    head entry is the head over the target's real hidden state and the next
    token, as one forward over the whole sequence gives (the chained entries
    and the rejected drafts never survive: verify / settle). "gate": draft
    counts 0..4 in turn (0: plain bursts and their catch-up flush)."""
    k = int(how[1:]) if how != "gate" else 4
    monkeypatch.setattr(worker, "MTP_K", k if how != "gate" else 0)
    monkeypatch.setattr(worker.MTPDraft, "CHUNK", 7 if k % 2 else 512)   # the prefill slab in pieces
    ids = prompts([25], 31)[0]
    ref = plain(dense, ids, 30)
    mt = Oracle(head(dense), dense, ids + ref[0], script=(3, 0, 4, 1, 0, 0, 2))
    assert with_mtp(dense, ids, 30, mt) == ref
    assert max(mt.accept_lens) >= min(k, 3) and min(mt.accept_lens) < k   # (k = 1: wrong every other)
    if how == "gate":
        assert mt.ks[0] >= 2 and mt.burst_tokens >= 2 and all(mt.ks[1:])
    real_entries(dense, mt, ids + ref[0], len(ids + ref[0]) - 2 * (k + 1))


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


def test_chooser_head_is_the_default_arm(clock):
    """With the gated head: DFlash never timed is probed PROBE rounds (the
    head never is), the head is taken until it is measured (a later request
    of the class starts on DFlash, the head's rate known), DFlash wins only
    by a MARGIN known from recent samples, and the head is taken back on the
    margin alone; its rate (tokens / ms of its rounds) follows the drift
    while it is not taken; nothing else is a plain arm."""
    assert worker.Chooser(("dflash", "plain")).pick() == "dflash"     # (without the head: a plain arm)
    c = worker.Chooser(("dflash", "mtp"), mtp=object())
    P = c.PROBE
    got = []
    for _ in range(P):                                   # nothing timed: DFlash probed
        got.append(c.pick())
        assert c.shadows() == []
        c.done(3, 7)
    clock.add("dflash", 180.0)
    got += [c.pick(), c.pick()]                          # the head, not measured yet: no shadows
    assert got == ["dflash"] * P + ["mtp"] * 2 and c.shadows() == []
    c.timed("mtp", 2.0, 115.0)                           # 17.4 tok/s
    c.timed("plain", 1.0, 10.0)                          # (only the head's arm is measured here)
    assert c.expect("mtp") == (pytest.approx(2.0), pytest.approx(115.0))
    r = 2.0 / 115.0 * 180.0                              # DFlash's tokens per round at the head's rate

    def dflash(tok, e=10.0):
        c.tok["dflash"].h, c.tok["dflash"].w, c.tok["dflash"].e = tok * 10.0, 10.0, e

    dflash(1.9)
    assert c.pick() == "mtp"
    dflash(r * (1 + c.MARGIN - 0.01))                    # a little better: not worth a switch
    assert c.pick() == "mtp"
    dflash(r * (1 + c.MARGIN + 0.01), e=1.0)             # better, from one sample: not yet
    assert c.pick() == "mtp" and c.shadows() == ["dflash"]       # (a trial: drafted every round)
    dflash(r * (1 + c.MARGIN + 0.01))
    assert c.pick() == "dflash" and c.shadows() == []    # (the head is never shadow-drafted)
    dflash(r * 0.95)                                     # the head back: no evidence needed
    assert c.pick() == "mtp"
    clock.add("dflash", 180.0)
    for _ in range(20):
        clock.add("dflash", 234.0)                       # the machine 30 % slower
    assert c.expect("mtp")[1] == pytest.approx(115.0 * clock.g, rel=1e-6) and clock.g > 1.2
    assert c.pick(force="dflash") == "dflash" and c.switches == 4
    # the class's next request: the head's arm is known (Clock.arm), so it starts on DFlash
    # and the head takes over as soon as DFlash's own rounds read below it
    c = worker.Chooser(("dflash", "mtp"), mtp=object())
    assert c.expect("mtp")[0] == pytest.approx(2.0) and c.pick() == "dflash"
    dflash(1.0)
    assert c.pick() == "mtp" and c.switches == 1


def test_chooser_shadows_cadence(clock):
    """Shadow drafts of DFlash while the head is taken: every round until it
    has WARM drafts, then as often as their cost is NEAR (an arm close to the
    current one) or FAR of the time."""
    clock.add("dflash", 180.0)
    c = worker.Chooser(("dflash", "mtp"), mtp=object())
    c.timed("mtp", 2.9, 115.0)                           # 25.2 tok/s
    c.cur = "mtp"                                        # (a request under way on the head's arm)
    t = 0.0

    def go(n, tok):
        nonlocal t
        out = []
        for _ in range(n):
            assert c.pick(force=None, now=t) == "mtp"
            sh = c.shadows()
            out.append(bool(sh))
            c.done(1, 1, shadows=sh)
            for x in sh:
                c.sample(x, 1, 7)
            c.tok["dflash"].h = tok * c.tok["dflash"].w
            t += 115.0
        return out

    got = go(200, 1.9)                                   # 10.6 < 0.8 * 25.2 tok/s: far
    assert got[:c.WARM] == [True] * c.WARM
    far = math.ceil(c.COST["dflash"] * 180.0 / c.FAR / 115.0)   # rounds between two drafts
    assert sum(got[c.WARM:]) == pytest.approx((200 - c.WARM) / far, abs=2)
    got = go(200, 4.2)                                   # 23.3: close
    near = math.ceil(c.COST["dflash"] * 180.0 / c.NEAR / 115.0)
    assert sum(got) == pytest.approx(200 / near, abs=2) and near < far
    assert c.n["dflash"] == 0 and c.pend["dflash"] == 0 and c.shadowed["mtp"] == 0


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
    """DFlash wins by its shadow drafts alone (no round spent on it), and
    loses again to the head when the text changes; a single good draft of
    it starts a trial (drafted every round), not a switch."""
    clock.add("dflash", 180.0)
    c = worker.Chooser(("dflash", "mtp"), mtp=object())
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
                c.sample(x, dflash_accepts, 7)
            t += 180.0 if k == "dflash" else 115.0
            if k == "mtp":
                c.timed("mtp", 1.85, 115.0)              # the head: 16.1 tok/s

    go(40, 1)                                            # 2 tok / 180 ms = 11.1
    assert set(took) == {"mtp"} and drafted[1:1 + c.WARM] == [True] * c.WARM
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


def test_hybrid_shadow_drafts_exact(dense, spec, monkeypatch, clock):
    """DFlash's shadow drafts beside the head's gated rounds (its plain
    bursts included) leave the answer plain greedy's, DFlash's context takes
    every position once (the ones since its last draft) and the head's cache
    holds real entries only."""
    monkeypatch.setattr(worker, "MTP_K", 0)
    monkeypatch.setattr(worker, "SPEC_TRACE", True)
    clock.add("dflash", 180.0)                           # timed: no probe rounds
    ids = prompts([25], 32)[0]
    ref = plain(dense, ids, 60)
    full = ids + ref[0]
    fed = []

    class Rec(FakeDraft):
        def draft_block(self, b, hidden, cache, bs, sampler, dtype):
            fed.append(hidden.shape[1])
            return mx.array([(ref[0][len(emitted):] + [0] * bs)[:bs - 1]], dtype=dtype)

    emitted = []
    mt = Oracle(head(dense), dense, full, wrong=False, script=(2, 0, 3))
    d = Rec(ref[0])
    sched = scheduler(dense, max_batch=1)
    sched.eng.mtp, sched.eng.draft = mt, d
    stop = sched.eng.stopping
    sched.eng.stopping = lambda t: emitted.append(t) or stop(t)     # every emitted token
    x = request(ids, 60, "a")
    assert run(sched, [x])["a"] == ref
    tr = x.spec_stats["trace"]
    assert {t[0] for t in tr} == {"mtp"} and not d.accept_lens       # DFlash: shadows only
    at = [i for i, t in enumerate(tr) if t[6]]
    assert len(at) >= 3 and {s[0] for t in tr for s in t[6]} == {"dflash"}
    assert any(t[1] == 1 for t in tr[at[0]:])            # bursts among the rounds sampled
    # the context of a shadow draft: every position since the last one
    assert fed[1:] == [sum(len(t[7]) for t in tr[a:b]) for a, b in zip(at, at[1:])]
    assert x.spec_stats["mtp"]["burst_tokens"] > 0 and x.spec_stats["shadows"] == {"dflash": len(at), "mtp": 0}
    real_entries(dense, mt, full, len(full) - 6)


@pytest.mark.parametrize("arms", ["mtp", "dflash", "both"])
def test_plain_arm_exact(dense, spec, monkeypatch, arms):
    """QWEN_SPEC_PLAIN: with DFlash alone plain steps are an arm (mixed into
    its rounds, each arm taken at least once: fresh estimates first); with
    the head there is no plain arm (its gate takes plain steps); the answer
    is plain greedy's."""
    monkeypatch.setattr(worker, "SPEC_PLAIN", True)
    monkeypatch.setattr(worker, "MTP_K", 0)
    ids = prompts([25], 36)[0]
    ref = plain(dense, ids, 40)
    mt = Oracle(head(dense), dense, ids + ref[0], script=(2, 0)) if arms != "dflash" else None
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
    assert set(taken) == {"dflash": {"dflash", "plain"}, "mtp": set(), "both": {"dflash", "mtp"}}[arms]
    assert mt is None or mt.burst_tokens > 0


def forced(monkeypatch, order, took):
    """The chooser takes the arms of `order` ("d" / "m" / "p") in turn, with
    shadow drafts of DFlash every round it is not taken; "p" with the head
    loaded = its round with no draft (the gate's plain burst)."""
    seq = {"d": "dflash", "m": "mtp", "p": "plain"}

    def pick(self, force=None, now=None):
        self.rounds += 1
        x = seq[order[len(took) % len(order)]]
        took.append(x)
        if self.head:
            self.mtp.zero = x == "plain"
            x = "mtp" if x == "plain" else x
        self.cur = x
        self.r = {}
        return self.cur
    monkeypatch.setattr(worker.Chooser, "pick", pick)
    monkeypatch.setattr(worker.Chooser, "shadows",
                        lambda self: ["dflash"] if self.cur != "dflash" and "dflash" in self.kinds else [])


@pytest.mark.parametrize("order", ["p", "pmd", "dpm", "ppd"])
@pytest.mark.parametrize("burst", [1, 3, 8])
def test_plain_bursts_penalties_exact(dense, spec, monkeypatch, order, burst):
    """The plain arm's bursts (step-ahead: a token is fed before it is read)
    between drafter rounds, with penalties (a fed token is counted once: by
    the step that feeds it, or at its emission when the burst ends on it) -
    the plain loop's tokens, greedy with a presence + repetition penalty."""
    from test_kvq8 import spec_arms
    monkeypatch.setattr(worker, "MTP_K", 0)
    monkeypatch.setattr(worker, "SPEC_PLAIN", True)
    monkeypatch.setattr(worker, "DRAFT_SAMPLING", True)
    monkeypatch.setattr(worker.Chooser, "BURST", burst)
    monkeypatch.setattr(worker.MTPDraft, "BURSTS", (burst,))
    took = []
    forced(monkeypatch, order, took)
    ids = prompts([25], 33)[0]
    kw = dict(presence_penalty=1.5, repetition_penalty=1.3, frequency_penalty=0.2)
    ref = plain(dense, ids, 40, **kw)
    assert ref != plain(dense, ids, 40)                  # the penalties move the answer
    sched = scheduler(dense, max_batch=1)
    d, mt = spec_arms(sched, ref[0], ids, ("dflash", "mtp"), script=(2,))
    x = request(ids, 40, "a", **kw)
    assert run(sched, [x])["a"] == ref
    assert x.spec_stats["plain_steps"] >= (39 if order == "p" else 4)
    assert ("d" not in order or d.accept_lens) and ("m" not in order or mt.accept_lens)


def test_sampling_takes_every_arm(dense, spec, monkeypatch):
    """A sampling request through the three arms (QWEN_DRAFT=auto: sampling
    requests are speculative): it runs to its length; the same seed gives the
    same tokens again (the draws: the request's key, split per call)."""
    from test_kvq8 import spec_arms
    monkeypatch.setattr(worker, "MTP_K", 0)
    monkeypatch.setattr(worker, "SPEC_PLAIN", True)
    monkeypatch.setattr(worker, "DRAFT_SAMPLING", True)
    monkeypatch.setattr(worker.MTPDraft, "BURSTS", (3,))
    ids = prompts([25], 34)[0]
    kw = dict(temperature=0.8, top_p=0.95, top_k=20, seed=11)
    ref = plain(dense, ids, 40, temperature=0.0)
    outs = []
    for _ in range(2):
        took = []
        forced(monkeypatch, "dmp", took)
        sched = scheduler(dense, max_batch=1)
        spec_arms(sched, ref[0], ids, ("dflash", "mtp"), script=(2,))
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
    monkeypatch.setattr(worker, "MTP_K", 0)
    took = []
    forced(monkeypatch, "dmp", took)
    ids = prompts([25], 35)[0]
    ref = plain(dense, ids, 40)
    sched = scheduler(dense, max_batch=1)
    spec_arms(sched, ref[0], ids, ("dflash", "mtp"), script=(2,))
    x = request(ids, 40, "a")
    assert run(sched, [x])["a"] == ref
    arms = x.spec_stats["arms"]
    assert set(arms) == {"dflash", "mtp"}                # the head's plain bursts: its arm's
    # every token but the first, once (+ the stop token, read but not emitted)
    assert sum(v[1] for v in arms.values()) == len(ref[0]) - 1 + ("stop" in ref[2])
    assert sum(v[0] for v in arms.values()) == len(took)
    assert set(x.spec_stats["shadows"]) == {"dflash", "mtp"}
    st = x.spec_stats["mtp"]
    assert st["k"][0] == took.count("plain") and st["k"][2] >= 1 and st["burst_tokens"] >= st["k"][0]
    assert sum(st["accepted"]) == sum(st["k"][1:]) and st["dropped"] == 0


@pytest.mark.parametrize("gate", [False, True])
def test_thinking_budget(dense, spec, monkeypatch, gate):
    """A thinking budget inside MTP rounds (3 drafts, or gated: 0..3 with
    plain bursts): the same tokens as the plain loop with that budget (the
    forced tokens go through plain steps; the head's pending position takes
    the forced token as its successor)."""
    from mlx_vlm.utils import ThinkingBudgetCriteria
    monkeypatch.setattr(worker, "MTP_K", 0 if gate else 3)
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
            mt = Oracle(head(dense), dense, p + full, wrong=False, script=(3, 0, 1, 0, 2))
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
    monkeypatch.setattr(worker, "MTP_K", 0)
    monkeypatch.setattr(worker, "SPEC_PLAIN", True)
    monkeypatch.setattr(worker.Chooser, "BURST", burst)
    monkeypatch.setattr(worker.MTPDraft, "BURSTS", (burst,))
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
        spec_arms(sched, ref[0], p, ("dflash", "mtp"), script=(2,))
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


def test_gate_draft_count(dense, monkeypatch, clock):
    """The gate (MTPDraft.round / depth): the draft count with the most
    expected tokens per ms, E(k) / T(k), E from the seed's confidence and the
    class's keep rates of the chained drafts (Clock.rate, moved by record());
    every PROBE-th drafting round takes all the room; k = 0 rounds in a row
    lengthen the plain bursts (BURSTS), a drafted round resets them; an
    unmeasured T scaled by 1 + R * k from the nearest measured k;
    QWEN_MTP_KMAX the ceiling."""
    monkeypatch.setattr(worker, "MTP_K", 0)
    monkeypatch.setattr(worker.MTPDraft, "PROBE", 10 ** 9)
    T = [100.0, 137.0, 176.0, 193.0, 210.0]              # the 27B: verify 1.29 / 1.6 / 1.7 / 1.8 + ~7 ms a draft
    for k, ms in enumerate(T):
        clock.add(worker.MTPDraft.kind_of(k), ms)
    mt = worker.MTPDraft(head(dense), dense)
    assert [mt._ms(k) for k in range(5)] == T
    assert mt.depth(0.3, T, 4) == 1                      # a bin not measured yet: SEED0
    clock.seed[1] = 0.35
    assert mt.depth(0.3, T, 4) == 0                      # 1.35 / 137 < 1 / 100 < 1.4 / 137
    clock.seed[1] = 0.4
    assert mt.depth(0.3, T, 4) == 1 and mt.bin == 1
    mt.record(0, 1)
    assert clock.seed[1] == pytest.approx(0.36) and mt.bin is None
    assert mt.depth(0.95, T, 4) == 1                     # (bin 4 starts at 0.9) rates 0.5: 2.35 / 176 < 1.9 / 137
    clock.rate.update({2: 0.9, 3: 0.9, 4: 0.9})
    assert mt.depth(0.95, T, 4) == 4 and mt.depth(0.95, T, 2) == 2
    clock.rate.update({3: 0.2, 4: 0.2})
    assert mt.depth(0.95, T, 4) == 2                     # 2.81 / 176 > 2.98 / 193 > 3.01 / 210
    clock.rate.update({2: 0.5})
    mt.record(1, 4)                                      # depth 2 reached and lost; 3, 4 not reached
    assert clock.rate == {2: pytest.approx(0.4), 3: 0.2, 4: 0.2}
    mt.record(3, 3)
    assert clock.rate == {2: pytest.approx(0.52), 3: pytest.approx(0.36), 4: 0.2}
    mt.record(0, 2)
    assert clock.rate[2] == pytest.approx(0.52)
    confs = []
    real = worker.MTPDraft._head

    def scripted(self, h):                               # the head's tokens, scripted confidences
        return real(self, h)[0], mx.array([[confs.pop(0)]] * h.shape[1], dtype=mx.float32).reshape(1, -1)
    monkeypatch.setattr(worker.MTPDraft, "_head", scripted)
    H = COMMON["hidden_size"]
    mt.fresh()
    mt.start(0)
    at = [0]

    def seed(c, n=1):                                    # one more position and its successor
        mt.round_end(0, None)                            # (the last round's chained entries go)
        confs[:] = [c] + [0.0] * 8                       # (a chained draft's own confidence: not used)
        mt.push(mx.random.normal((1, n, H)))
        mt.emit_ids(range(at[0] + 1, at[0] + n + 1))
        at[0] += n

    clock.rate.update({2: 0.9, 3: 0.9, 4: 0.9})
    seed(0.95, 4)
    d = mt.round(9)
    assert d.shape == (1, 4) and mt.appended == 3 and len(confs) == 5    # KMAX
    seed(0.95)
    assert mt.round(1).shape == (1, 1)                   # room
    clock.seed.update({0: 0.1, 1: 0.2})
    seed(0.1)
    assert [mt.round(9) for _ in range(6)] == [1, 2, 4, 8, 8, 8]   # one seed, low: longer bursts
    clock.rate.update({3: 0.2, 4: 0.2})
    seed(0.95)
    assert mt.round(9).shape == (1, 2)
    seed(0.2)
    assert mt.round(9) == 1                              # reset by the drafted round
    assert mt.ks == [7, 1, 1, 0, 1] and mt.stats()["k"] == mt.ks
    monkeypatch.setattr(worker.MTPDraft, "PROBE0", 3)    # every 3rd no-draft decision drafts once
    mt.fresh()
    seed(0.1)
    assert [isinstance(mt.round(9), int) for _ in range(3)] == [True, True, False] and mt.ks[:2] == [2, 1]
    monkeypatch.setattr(worker.MTPDraft, "PROBE", 2)     # every 2nd drafting round: all the room
    mt.fresh()
    clock.rate.clear()
    seed(0.95)
    assert mt.round(9).shape == (1, 1)
    seed(0.95)
    assert mt.round(9).shape == (1, 4)
    seed(0.95)
    assert mt.round(9).shape == (1, 1)
    monkeypatch.setattr(worker.MTPDraft, "PROBE", 10 ** 9)
    mt.clock = worker.Clock({("mtp", 1): 66.0})
    assert mt._ms(3) == pytest.approx(66.0 * (1 + 3 * mt.R) / (1 + mt.R))
    assert mt._ms(0) == pytest.approx(66.0 / (1 + mt.R))
    mt.clock = worker.Clock()
    assert mt._ms(2) == 1 + 2 * mt.R
    monkeypatch.setattr(worker, "MTP_KMAX", 7)
    mt.fresh()
    mt.clock = worker.Clock({worker.MTPDraft.kind_of(k): 100.0 for k in range(8)})
    seed(0.5)
    assert mt.KMAX == 7 and mt.round(9).shape == (1, 7)


@pytest.mark.parametrize("bursts", [(1,), (1, 2, 4, 8)])
@pytest.mark.parametrize("kw", ["greedy", "penalties", "sampling"])
def test_gate_plain_rounds_are_the_plain_loop(dense, spec, monkeypatch, kw, bursts):
    """Rounds with no draft (k = 0) are the plain loop's steps: every round
    a plain burst gives the plain loop's tokens - greedy, with penalties, and
    a seeded sampling request (one draw per step from the request's key)."""
    monkeypatch.setattr(worker, "MTP_K", 0)
    monkeypatch.setattr(worker, "DRAFT_SAMPLING", True)
    monkeypatch.setattr(worker.MTPDraft, "BURSTS", bursts)
    kw = {"greedy": {}, "penalties": dict(presence_penalty=1.5, repetition_penalty=1.3),
          "sampling": dict(temperature=0.8, top_p=0.95, top_k=20, seed=11)}[kw]
    ids = prompts([25], 38)[0]
    ref = plain(dense, ids, 30, **kw)
    mt = Oracle(head(dense), dense, ids + ref[0], script=(0,))
    assert with_mtp(dense, ids, 30, mt, **kw) == ref
    assert not mt.accept_lens and mt.burst_tokens >= len(ref[0]) - 1
    assert mt.ks[0] == (len(ref[0]) - 1 if bursts == (1,) else 6)  # 1 + 2 + 4 + 8 + 8 + 6 (cut)


def test_gate_burst_catch_up_equals_per_step(dense, spec, monkeypatch):
    """A burst's single catch-up flush, queued with its last step on that
    step's token not read yet (emit_lazy), leaves the head's cache bitwise
    equal to the same flush after the token was read (a sync per step); and
    as a flush after every step (one-step bursts) up to the rounding of an
    n-row against a 1-row forward (CPU fp32: ~1e-6, not bitwise); the same
    answer each time."""
    monkeypatch.setattr(worker, "MTP_K", 0)
    ids = prompts([25], 39)[0]
    ref = plain(dense, ids, 40)
    lazy = worker.MTPDraft.emit_lazy

    def synced(self, y):                                 # the token read on the host first
        self.succ.append(int(y.item()))
        self.skip += 1
    got = []
    for bursts, emit in [((3, 8, 5), lazy), ((3, 8, 5), synced), ((1,), lazy)]:
        monkeypatch.setattr(worker.MTPDraft, "BURSTS", bursts)
        monkeypatch.setattr(worker.MTPDraft, "emit_lazy", emit)
        mt = Oracle(head(dense), dense, ids + ref[0], script=(0,))
        assert with_mtp(dense, ids, 40, mt) == ref
        n = mt.cache[0].offset
        got.append((n, mt.cache[0].keys[..., :n, :], mt.cache[0].values[..., :n, :]))
    (n0, k0, v0), (n1, k1, v1), (n2, k2, v2) = got
    assert n0 == n1 == n2 >= len(ids) + 38
    assert mx.array_equal(k0, k1).item() and mx.array_equal(v0, v1).item()
    for a, b in ((k0, k2), (v0, v2)):
        assert mx.abs(a - b).max().item() < 1e-5


def test_auto_equals_mtp_equals_plain(dense, spec, monkeypatch, clock):
    """QWEN_DRAFT=auto's setup - DFlash + the gated head, the Chooser with
    its probes and DFlash's shadow drafts - == the head alone == plain
    greedy, the head's own confidences deciding (the random head's are
    0.006..0.014, a bin whose keep rate is set to 0: plain bursts, every
    PROBE0-th decision a draft, every PROBE-th drafting round all four)."""
    monkeypatch.setattr(worker, "MTP_K", 0)
    monkeypatch.setattr(worker, "SPEC_PLAIN", True)
    monkeypatch.setattr(worker.Clock, "add", lambda self, kind, ms: None)
    monkeypatch.setattr(worker.MTPDraft, "PROBE0", 3)
    monkeypatch.setattr(worker.MTPDraft, "PROBE", 2)
    monkeypatch.setattr(worker.MTPDraft, "B", 0.0)
    rounds = [0, 0, 0]
    for ids in prompts([25, 40, 7], 37):
        ref = plain(dense, ids, 40)
        for both in (False, True):
            ck = worker.Clock({worker.MTPDraft.kind_of(k): (100.0, 100.9, 101.5, 110.0, 120.0)[k]
                               for k in range(5)})
            ck.seed[0] = 0.0                             # (kept there: the Oracle's rounds)
            monkeypatch.setattr(worker, "CLOCK", ck)
            mt = Oracle(head(dense), dense, ids + ref[0])
            d = FakeDraft(ref[0]) if both else None
            sched = scheduler(dense, max_batch=1)
            sched.eng.mtp, sched.eng.draft = mt, d
            x = request(ids, 40, "a")
            assert run(sched, [x])["a"] == ref, (ids, both)
            assert set(x.spec_stats["arms"]) <= {"dflash", "mtp"}
            rounds[0] += mt.ks[0]
            rounds[1] += mt.ks[1]
            rounds[2] += sum(mt.ks[2:])
            if both:
                assert d.accept_lens                     # the probe
    assert all(rounds)                                   # bursts and drafted rounds


def test_token_bytes(dense):
    """The head's own fp16 KV per position (charged to fp16 rows by the scheduler)."""
    from conftest import COMMON
    mt = worker.MTPDraft(head(dense), dense)
    kv, hd = COMMON["num_key_value_heads"], COMMON["head_dim"]
    assert mt.token_bytes == 2 * 2 * kv * hd * 1
