"""The MTP head as a speculative drafter (qwen_worker/mtp.py) on the tiny random
model (CPU): a real mlx-vlm Qwen3_5MTPDraftModel (random weights) bound to the
tiny target. Greedy output == the plain loop's for every draft count, with
the head's own drafts (random: nearly all rejected) and with an oracle that
proposes the known answer (accepted prefixes, rejections, the chained
entries kept or recomputed); the head's KV cache after a request == the head
run once over every (target hidden, next token) pair; hybrid with a DFlash
stand-in; the thinking budget; the prompt cache's drafter context."""
import mlx.core as mx
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
    """MTPDraft's side of the chooser: expect() / learn()."""

    def __init__(self, tok, ms):
        self.tok, self.ms, self.learned = tok, ms, []

    def expect(self):
        return self.tok, self.ms

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


def test_chooser_rate_margin_and_probe(clock):
    """An arm never timed is taken PROBE rounds; then the arm with the most
    expected tokens per ms, left only for one MARGIN better."""
    c = worker.Chooser(("dflash", "mtp", "plain"), Head(1.8, None))
    P = c.PROBE
    got = []
    for _ in range(P):                                   # nothing timed: dflash first
        got.append(c.pick())
        assert c.shadows() == []
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
    c.tok["dflash"] = 4.8                                # 26.7 vs 15.7 vs 12.5 tok/s
    assert c.pick() == "dflash"
    c.tok["dflash"] = 1.9                                # 10.6: the head
    assert c.pick() == "mtp"
    c.tok["dflash"] = 1.8 * 180 / 115 * 1.04             # 4 % better: not worth a switch
    assert c.pick() == "mtp"
    c.tok["dflash"] = 1.8 * 180 / 115 * 1.06
    assert c.pick() == "dflash"
    c.tok["dflash"], c.mtp.tok = 1.0, 1.0                # both below a plain step
    assert c.pick() == "plain"
    assert c.pick(force="mtp") == "mtp" and c.shadows() == []


def test_chooser_shadows_cadence(clock):
    """Shadow drafts of the arms not taken: every round of the first WARM,
    then every NEAR-th (an arm close to the current one) or FAR-th round."""
    clock.add("dflash", 180.0)
    c = worker.Chooser(("dflash", "mtp"), Head(1.8, 115.0))
    c.tok["dflash"] = 4.8
    rounds = []
    for i in range(c.WARM + 2 * c.FAR + 4):
        assert c.pick() == "dflash"
        sh = c.shadows()
        if sh:
            rounds.append(i)
        c.done(4, 7, shadows=sh)
        c.tok["dflash"] = 4.8
    assert rounds[:c.WARM] == list(range(c.WARM))        # 15.7 < 0.8 * 26.7: far
    assert [b - a for a, b in zip(rounds[c.WARM - 1:], rounds[c.WARM:])] == [c.FAR + 1] * 2
    c.mtp.tok = 2.9                                      # 25.2: close
    rounds = []
    for i in range(4 * c.NEAR + 4):
        c.pick()
        sh = c.shadows()
        rounds += [i] if sh else []
        c.done(4, 7, shadows=sh)
        c.tok["dflash"] = 4.8
    assert all(b - a == c.NEAR + 1 for a, b in zip(rounds, rounds[1:])) and len(rounds) >= 3
    assert c.shadowed["mtp"] == c.WARM + 2 + len(rounds) and c.n["mtp"] == 0


def test_chooser_noisy_drafter_keeps_its_rounds(clock):
    """The old chooser's failure: a drafter whose (noisy) tokens per round beat
    a plain step lost its rounds to plain whenever its average dipped, and the
    frozen estimate kept it out. Now the arm not taken is sampled by shadow
    drafts, the averages are slow and a switch needs a margin."""
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
    took = []
    for _ in range(400):
        k = c.pick()
        took.append(k)
        sh = c.shadows()
        c.done(accepted() if k == "dflash" else 0, 7 if k == "dflash" else 0, shadows=sh)
        for x in sh:
            c.sample(x, accepted(), 7)
    assert took.count("plain") < 0.1 * len(took)


def test_chooser_switches_on_shadow_samples(clock):
    """The arm not taken wins by its shadow drafts alone (no round spent on
    it), and loses again when the text changes."""
    clock.add("dflash", 180.0)
    c = worker.Chooser(("dflash", "mtp"), Head(1.85, 115.0))     # the head: 16.1 tok/s
    took = []

    def go(n, dflash_accepts):
        for _ in range(n):
            k = c.pick()
            took.append(k)
            sh = c.shadows()
            c.done(dflash_accepts if k == "dflash" else 1, 7 if k == "dflash" else 1, shadows=sh)
            for x in sh:
                if x == "dflash":
                    c.sample(x, dflash_accepts, 7)

    go(12, 1)                                            # 2 tok / 180 ms = 11.1: the prior, then the head
    assert took[0] == "dflash" and took[-1] == "mtp" and took.index("mtp") <= 4
    n0 = c.n["dflash"]
    go(120, 5)                                           # 6 tok / 180 ms = 33.3 by shadows alone
    assert took[-1] == "dflash" and c.shadowed["dflash"] >= 2
    assert c.n["dflash"] - n0 < 120 and c.n["mtp"] >= 8


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
    monkeypatch.setattr(worker.Chooser, "pick", lambda self, force=None: taken.append(pick(self, force))
                        or taken[-1])
    sched = scheduler(dense, max_batch=1)
    sched.eng.mtp, sched.eng.draft = mt, d
    stop = sched.eng.stopping
    sched.eng.stopping = lambda t: emitted.append(t) or stop(t)     # every emitted token
    assert run(sched, [request(ids, 40, "a")])["a"] == ref
    assert "plain" in taken and len(set(taken)) == (3 if arms == "both" else 2)


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
    mt.ps = [0.9] * 4                                    # the slow averages: the chooser's side
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
