"""Sampling keyed by (seed, absolute position) (sampler.POS_KEYS): every
sampled token is drawn with noise hashed from the request's seed and the
token's position, so a seeded request gives the same text plain, through
speculative decoding (MTP head at k = 1 / 3, DFlash at any block size,
lookup rounds, both drafters with the plain arm), beside other rows in the
batch, after a demotion and after a prompt-cache hit. On the tiny random
model (CPU, fp32: verify rows and batched rows round like the lone step to
~1e-6, so the draws match). Units: a block's row i == the lone draw at
pos + i == sample_rows beside other requests, on both draw paths; seeds
differ; greedy ignores it; a draw from another thread is the same draw."""
import threading

import mlx.core as mx
import pytest

import worker
from conftest import prompts, request, run, scheduler
from test_lookup import FakeLookup
from test_mtp import Oracle, head
from test_prompt_cache import small  # noqa: F401 (a fixture)
from test_scheduler import FakeDraft, no_metal_verify

CARD = dict(temperature=0.7, top_p=0.95, top_k=20, seed=11)
CFGS = [dict(temperature=0.6, top_p=0.95, top_k=20), dict(temperature=1.0),
        dict(temperature=0.8, min_p=0.05), dict(temperature=0.9, top_k=5, top_p=0.5)]


@pytest.fixture(autouse=True)
def fresh_clock(monkeypatch):
    """The process-wide round times (mtp.CLOCK): a fresh table per test."""
    monkeypatch.setattr(worker, "CLOCK", worker.Clock())


@pytest.fixture
def spec(monkeypatch):
    no_metal_verify(monkeypatch)
    monkeypatch.setattr(worker, "DRAFT_SAMPLING", True)


def make(monkeypatch, cfg, seed, fast=False):
    monkeypatch.setattr(worker, "SAMPLER_TOPK_FAST", fast)
    return worker.make_sampler(dict(worker.SAMPLING_KWARGS, **cfg), seed=seed)


def plain(lm, ids, n, **kw):
    return run(scheduler(lm, max_batch=1), [request(ids, n, "a", **kw)])["a"]


# --- the sampler ----------------------------------------------------------------------
@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("cfg", CFGS)
def test_block_rows_are_positions(monkeypatch, cfg, fast):
    """Row i of a block drawn at pos == the lone draw at pos + i == that row
    in sample_rows beside other requests (other seeds, positions, settings);
    the order of draws does not matter (nothing is carried)."""
    x = (mx.random.normal((6, 512), key=mx.random.key(3)) * 3).astype(mx.float16)
    block = make(monkeypatch, cfg, 5, fast)(x, 40).tolist()
    s = make(monkeypatch, cfg, 5, fast)
    lone = [s(x[i:i + 1], 40 + i).item() for i in reversed(range(6))][::-1]
    assert block == lone
    others = [make(monkeypatch, cfg, 99, fast), make(monkeypatch, CFGS[(CFGS.index(cfg) + 1) % 4], 5, fast)]
    for i in range(6):
        mine = make(monkeypatch, cfg, 5, fast)
        ys = worker.sample_rows([others[0], mine, others[1]], [x[(i + 1) % 6:(i + 1) % 6 + 1],
                                x[i:i + 1], x[i:i + 1]], [7, 40 + i, 40 + i])
        assert ys[1].item() == block[i], i


@pytest.mark.parametrize("fast", [False, True])
def test_seeds_and_positions_differ(monkeypatch, fast):
    """Another seed, or the same seed at other positions: another stream;
    the same (seed, positions): the same tokens."""
    x = mx.zeros((64, 512))                                 # uniform: every token as likely
    cfg = dict(temperature=1.0, top_k=100)
    a = make(monkeypatch, cfg, 5, fast)(x, 0).tolist()
    assert a == make(monkeypatch, cfg, 5, fast)(x, 0).tolist()
    assert a != make(monkeypatch, cfg, 6, fast)(x, 0).tolist()
    assert a != make(monkeypatch, cfg, 5, fast)(x, 64).tolist()
    assert len(set(a)) > 40


def test_greedy_ignores_positions(monkeypatch):
    x = (mx.random.normal((3, 512), key=mx.random.key(4)) * 3).astype(mx.float16)
    want = mx.argmax(x - mx.logsumexp(x, axis=-1, keepdims=True), axis=-1).tolist()
    for fast in (False, True):
        g = make(monkeypatch, dict(temperature=0.0, top_k=20), 5, fast)
        assert g(x).tolist() == g(x, 0).tolist() == g(x, 1234).tolist() == want


def test_draw_from_another_thread(monkeypatch):
    """MLX random state is per thread; the draw has none: a handler-like
    thread draws what the main thread draws for the same (seed, position),
    and samplers without a seed get different ones."""
    x = (mx.random.normal((4, 512), key=mx.random.key(5)) * 3)
    want = make(monkeypatch, CARD, 11)(x, 9).tolist()
    got = {}

    def draw():
        got["seeded"] = worker.make_sampler(dict(worker.SAMPLING_KWARGS, **CARD), seed=11)(x, 9).tolist()
        got["seeds"] = [worker.make_sampler(dict(worker.SAMPLING_KWARGS, **CARD)).seed for _ in range(2)]
    t = threading.Thread(target=draw)
    t.start()
    t.join()
    assert got["seeded"] == want
    assert got["seeds"][0] != got["seeds"][1]


# --- the scheduler: the same seeded text on every path ----------------------------------
@pytest.mark.parametrize("drafts", ["oracle", "own"])
@pytest.mark.parametrize("k", [1, 3])
def test_mtp_sampled_equals_plain(dense, spec, monkeypatch, k, drafts):
    """The MTP head, k drafts a round: drafts of the sampled answer (the last
    one wrong every other round) or the head's own (random: nearly all
    rejected, so most verify rows are discarded and their positions drawn
    again next round): the plain loop's seeded text."""
    monkeypatch.setattr(worker, "MTP_K", k)
    ids = prompts([25], 33)[0]
    ref = plain(dense, ids, 32, **CARD)
    mt = (Oracle(head(dense), dense, ids + ref[0]) if drafts == "oracle"
          else worker.MTPDraft(head(dense), dense))
    sched = scheduler(dense, max_batch=1)
    sched.eng.mtp = mt
    x = request(ids, 32, "a", **CARD)
    assert run(sched, [x])["a"] == ref
    assert x.spec_stats is not None and mt.accept_lens
    if drafts == "oracle":
        assert max(mt.accept_lens) >= 1 and min(mt.accept_lens) < k
    assert ref != plain(dense, ids, 32, **dict(CARD, seed=12))


@pytest.mark.parametrize("answer", [True, False])
@pytest.mark.parametrize("block", [2, 4, 8])
def test_dflash_sampled_equals_plain(dense, spec, monkeypatch, block, answer):
    """DFlash at block 2 / 4 / 8, drafting the sampled answer (accepted
    prefixes, a wrong last draft every other round) or junk (every round
    keeps only the target's own token): the plain loop's seeded text."""
    monkeypatch.setattr(worker, "SPEC_PLAIN", False)
    ids = prompts([25], 34)[0]
    ref = plain(dense, ids, 30, **CARD)
    sched = scheduler(dense, max_batch=1)
    sched.eng.draft = d = FakeDraft(ref[0] if answer else [(t + 1) % 400 for t in ref[0]], block=block)
    assert run(sched, [request(ids, 30, "a", **CARD)])["a"] == ref
    assert d.accept_lens and max(d.draft_lens) == block - 1
    assert (max(d.accept_lens) >= 1) == answer


def test_lookup_rounds_sampled_equal_plain(dense, spec, monkeypatch):
    """Prompt lookup without a drafter: lookup rounds (the sampled answer's
    continuation, a wrong last draft now and then) between plain step-ahead
    bursts: the plain loop's seeded text (the stand-in checks every emitted
    token against it as it goes)."""
    monkeypatch.setattr(worker, "_Lookup", FakeLookup)
    monkeypatch.setattr(worker, "LOOKUP_K", 3)
    FakeLookup.every, FakeLookup.made = 2, None
    ids = prompts([25], 35)[0]
    ref = plain(dense, ids, 40, **CARD)
    FakeLookup.answer = ref[0]
    monkeypatch.setattr(worker, "LOOKUP", "1")
    x = request(ids, 40, "a", **CARD)
    assert run(scheduler(dense, max_batch=1), [x])["a"] == ref
    st = x.spec_stats
    assert st["lookup_rounds"] >= 2 and st["plain_steps"] > 0


@pytest.mark.parametrize("order", ["dmp", "mpd", "pdm"])
def test_both_drafters_and_plain_arm_sampled_equal_plain(dense, spec, monkeypatch, order):
    """QWEN_DRAFT=auto's shape: DFlash (block 4), the MTP head (2 drafts) and
    the head's no-draft rounds (plain step-ahead bursts) in turn, with
    DFlash's shadow drafts: the plain loop's seeded text."""
    from test_kvq8 import spec_arms
    from test_mtp import forced
    monkeypatch.setattr(worker, "MTP_K", 0)
    monkeypatch.setattr(worker, "SPEC_PLAIN", True)
    monkeypatch.setattr(worker.MTPDraft, "BURSTS", (3,))
    ids = prompts([25], 34)[0]
    ref = plain(dense, ids, 40, **CARD)
    took = []
    forced(monkeypatch, order, took)
    sched = scheduler(dense, max_batch=1)
    d, mt = spec_arms(sched, ref[0], ids, ("dflash", "mtp"), script=(2,))
    x = request(ids, 40, "a", **CARD)
    assert run(sched, [x])["a"] == ref
    assert {"dflash", "mtp", "plain"} <= set(took)
    assert d.sizes and mt.accept_lens and x.spec_stats["plain_steps"] > 0


def _neighbour_runs(lm):
    ps = prompts([20, 14], 41)
    alone = plain(lm, ps[0], 24, **CARD)
    out = []
    for first in (True, False):
        a = request(ps[0], 24, "a", **CARD)
        b = request(ps[1], 30, "b", temperature=0.9, seed=3)
        reqs, later = ([a, b], b) if first else ([b, a], a)
        out.append(run(scheduler(lm, max_batch=2), reqs, arrive={5: [later]})["a"])
    return alone, out


@pytest.mark.parametrize("model", ["dense", "dense_fused"])
def test_beside_another_row(model, request):        # pytest's own `request` fixture
    """B = 2 (bit-exact rows): the seeded request alone == beside another
    sampled row, joining it or joined by it mid-answer."""
    alone, out = _neighbour_runs(request.getfixturevalue(model))
    assert out == [alone, alone]


def test_speculative_then_demoted(dense, spec):
    """A seeded request on DFlash, demoted to a batch row when another one
    arrives: its text is the plain loop's alone."""
    ps = prompts([25, 12], 42)
    ref = plain(dense, ps[0], 30, **CARD)
    sched = scheduler(dense, max_batch=2)
    sched.eng.draft = d = FakeDraft(ref[0])
    a, b = request(ps[0], 30, "a", **CARD), request(ps[1], 20, "b", temperature=0.8, seed=5)
    got = run(sched, [a, b], arrive={4: [b]})
    assert got["a"] == ref and got["b"] == plain(dense, ps[1], 20, temperature=0.8, seed=5)
    assert 0 < len(d.accept_lens) < 10                   # rounds, then a batch row


def test_prompt_cache_hit_equals_cold(dense, small, monkeypatch):
    """A hit (an exact repeat; a longer prompt resuming inside the entry) is
    drawn at the same absolute positions as the cold prefill: the same
    seeded text; with the MTP head on the hit too."""
    no_metal_verify(monkeypatch)
    monkeypatch.setattr(worker, "DRAFT_SAMPLING", True)
    ids = prompts([40], 50)[0]
    longer = ids + prompts([9], 51)[0]
    cold, cold_longer = plain(dense, ids, 20, **CARD), plain(dense, longer, 20, **CARD)
    sched = scheduler(dense, max_batch=1)
    x, y, z = (request(ids, 20, "x", **CARD), request(ids, 20, "y", **CARD),
               request(longer, 20, "z", **CARD))
    assert run(sched, [x])["x"] == cold
    assert run(sched, [y])["y"] == cold and y.reused > 0
    assert run(sched, [z])["z"] == cold_longer and z.reused > 0
    monkeypatch.setattr(worker, "MTP_K", 2)
    sched.eng.mtp = worker.MTPDraft(head(dense), dense)
    w = request(ids, 20, "w", **CARD)
    assert run(sched, [w])["w"] == cold and w.reused > 0 and w.spec_stats is not None


def test_greedy_untouched_by_the_switch(dense, spec, monkeypatch):
    """Greedy never draws: plain and MTP greedy are the same tokens with
    POS_KEYS on and off."""
    monkeypatch.setattr(worker, "MTP_K", 2)
    ids = prompts([25], 36)[0]
    outs = []
    for on in (True, False):
        monkeypatch.setattr(worker, "POS_KEYS", on)
        outs.append(plain(dense, ids, 24))
        sched = scheduler(dense, max_batch=1)
        sched.eng.mtp = worker.MTPDraft(head(dense), dense)
        outs.append(run(sched, [request(ids, 24, "a")])["a"])
    assert all(o == outs[0] for o in outs)
