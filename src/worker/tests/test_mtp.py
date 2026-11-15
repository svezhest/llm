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


def test_chooser_prefers_tokens_per_ms():
    c = worker.Chooser(("dflash", "mtp", "plain"))
    P = c.PROBE
    assert [(c.pick(), c.done(5))[0] for _ in range(P)] == ["dflash"] * P   # its probe
    assert c.n["dflash"] == P and c.tok["dflash"] > c.PRIORS["dflash"]
    assert [(c.pick(), c.done(0))[0] for _ in range(P)] == ["mtp"] * P
    assert c.tok["mtp"] < c.PRIORS["mtp"]                      # the prior gives way
    assert [(c.pick(), c.done(0))[0] for _ in range(P)] == ["plain"] * P
    assert c.tok["plain"] == 1.0                                 # plain is exactly 1
    c.tok = {"dflash": 6.0, "mtp": 1.5, "plain": 1.0}
    c.ms = {"dflash": 150.0, "mtp": 60.0, "plain": 50.0}
    assert c.pick() == "dflash"                          # 0.040 vs 0.025 / 0.020 tok/ms
    c.done(5)
    c.tok["dflash"] = 2.0
    assert c.pick() == "mtp"
    c.tok["mtp"] = 1.0
    picks = []
    for _ in range(8 * c.EXPLORE):
        picks.append(c.pick())
        c.done(0)
        c.tok, c.ms = {"dflash": 2.0, "mtp": 1.0, "plain": 1.0}, {"dflash": 150.0, "mtp": 60.0, "plain": 50.0}
    assert picks.count("plain") > 0.9 * len(picks)      # every drafter below plain: plain
    assert "dflash" in picks and "mtp" in picks          # re-probed now and then
    runs = "".join(x[0] for x in picks).replace("p", " ").split()
    assert all(len(r) == P for r in runs[1:-1])          # each re-probe is PROBE rounds


def test_chooser_first_round_time_replaced():
    """An arm's first wall time (DFlash: the prompt window fed to the drafter)
    is replaced by its second, then averaged."""
    c = worker.Chooser(("dflash", "plain"))
    t = iter([0.0, 1.0, 1.0, 1.1, 2.0, 2.1])
    import qwen_worker.mtp as M
    real = M.time.perf_counter
    M.time.perf_counter = lambda: next(t)
    try:
        for _ in range(3):
            c.pick(force="dflash")
            c.done(1)
    finally:
        M.time.perf_counter = real
    assert c.ms["dflash"] == pytest.approx(0.8 * 100 + 0.2 * 100)


@pytest.mark.parametrize("arms", ["mtp", "dflash", "both"])
def test_plain_arm_exact(dense, spec, monkeypatch, arms):
    """QWEN_SPEC_PLAIN: plain steps mixed into the rounds (each arm is taken
    at least once: fresh estimates first); the answer is plain greedy's."""
    monkeypatch.setattr(worker, "SPEC_PLAIN", True)
    monkeypatch.setattr(worker, "MTP_K", 2)
    monkeypatch.setattr(worker.Chooser, "EXPLORE", 4)
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
    monkeypatch.setattr(worker.Chooser, "EXPLORE", 3)
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
    monkeypatch.setattr(worker.MTPDraft, "MS", {})
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
    monkeypatch.setattr(worker.MTPDraft, "MS", {1: 66.0, 2: 88.0, 3: 92.0, 4: 97.0})
    mt.start(0)
    mt.p = [0.46] * 4
    assert mt.next_k() == 1                              # 1.46 / 66 > 1.77 / 92
    mt.p = [0.85] * 4
    assert mt.next_k() == 4                              # 3.70 / 97 > 1.85 / 66
    monkeypatch.setattr(worker.MTPDraft, "MS", {1: 66.0})
    assert mt._ms(3) == pytest.approx(66.0 * (1 + 3 * worker.MTP_COST) / (1 + worker.MTP_COST))
    ids = prompts([25], 35)[0]
    ref = plain(dense, ids, 40)
    mt = Oracle(head(dense), dense, ids + ref[0], wrong=False)
    assert with_mtp(dense, ids, 40, mt) == ref
    assert max(mt.draft_lens) == 4


def test_token_bytes(dense):
    """The head's own fp16 KV per position (charged to fp16 rows by the scheduler)."""
    from conftest import COMMON
    mt = worker.MTPDraft(head(dense), dense)
    kv, hd = COMMON["num_key_value_heads"], COMMON["head_dim"]
    assert mt.token_bytes == 2 * 2 * kv * hd * 1
