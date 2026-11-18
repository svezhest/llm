"""The prefill of a speculative row with its drafters (sched_spec.PREFILL_NO_SEED:
a slab's eval leaves out the MTP head's seed, which nothing reads;
sched_spec.PREFILL_DEFER: the drafters' share of the last slab and the stored
context's copies are evaluated after the first token is emitted). Each switch
moves only when the GPU computes what the prefill builds: with either one on or
off the tokens, every stored prompt-cache entry (KV, recurrent states, the
drafter's context) and every drafter context (each DFlash block's context and
the drafter's K / V before it, the head's cache at each round) are the same
bit for bit: cold multi-slab and one-slab prompts, a conversation's hits
through checkpoints and reply KV, an exact repeat (no slab), a request that
ends at its first token, q8 KV; greedy and sampled; the head, DFlash and both
(tiny random drafters, CPU fp32)."""
import mlx.core as mx
import pytest

import worker
from conftest import V, request, run, scheduler
from test_mtp import head
from test_partial_store import prompt
from test_prompt_cache import conversation, small  # noqa: F401
from test_scheduler import no_metal_verify

# (PREFILL_NO_SEED, PREFILL_DEFER); the first is the reference (both off)
SWITCHES = [(False, False), (True, False), (False, True), (True, True)]
CARD = dict(temperature=0.7, top_p=0.95, top_k=20, seed=11)


def tiny_dflash(lm, window=16, block=4):
    """A real mlx-vlm DFlash drafter with random weights for the tiny target
    (hidden 128, 8 layers, vocab 512), bound by reset(): a sliding layer and
    a full-attention one."""
    from mlx_vlm.speculative.drafters.qwen3_dflash.config import DFlashConfig
    from mlx_vlm.speculative.drafters.qwen3_dflash.dflash import DFlashDraftModel
    mx.random.seed(17)
    d = DFlashDraftModel(DFlashConfig(
        hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, vocab_size=V, max_position_embeddings=4096,
        block_size=block, mask_token_id=V - 1, target_layer_ids=[0, 3, 5], num_target_layers=8,
        layer_types=["sliding_attention", "full_attention"], sliding_window=window,
        rope_theta=10000.0))
    d.apply(lambda a: a.astype(mx.float32))
    mx.eval(d.parameters())
    d.prefer_requested_block_size = True                # the fixed block (as QWEN_VERIFY_MRQMM)
    return d


@pytest.fixture(autouse=True)
def spec(monkeypatch):
    no_metal_verify(monkeypatch)
    monkeypatch.setattr(worker, "DRAFT_SAMPLING", True)
    monkeypatch.setattr(worker, "SPEC_PLAIN", False)
    # every round's depth fixed (the head's gate reads wall-clock round times)
    monkeypatch.setattr(worker, "MTP_K", 2)


def arrays(x):
    """The arrays of a nested state (lists / tuples / dicts by key), in order."""
    if x is None:
        return []
    if isinstance(x, mx.array):
        return [x]
    if isinstance(x, dict):
        return [a for k in sorted(x) for a in arrays(x[k])]
    return [a for y in x for a in arrays(y)]


def entries(pc):
    """Every entry of the prompt cache: ids, checkpoint positions, reply,
    then all of its arrays (KV pieces, end state with the drafter's context,
    checkpoint states)."""
    out = []
    for e in sorted(pc.entries, key=lambda e: (len(e.ids), e.ids)):
        kv = [a[..., lo - s.a:hi - s.a, :] for s, lo, hi in e.chain for a in s.arrays]
        out.append((tuple(e.ids), tuple(sorted(e.ckpts)), e.reply, kv + arrays(e.end) + arrays(e.ckpts)))
    return out


def own(x):
    """A copy of x as it is now (a cache array is overwritten in place later)."""
    y = mx.array(x)
    mx.eval(y)
    return y


def spy(d, m):
    """Each DFlash block's context: (each layer's offset, the context, the
    drafter's K / V before the call, the drafts); each head round: (its
    position, its cache's K / V, the drafts)."""
    rec = []
    if d is not None:
        orig = d.draft_block

        def draft_block(b, hidden, cache, bs, sampler, dtype):
            at = [c.offset for c in cache]
            kv = [own(a) for c in cache if c.keys is not None for a in c.state[:2]]
            out = orig(b, hidden, cache, bs, sampler, dtype)
            rec.append(("dflash", at, own(hidden), kv, own(out)))
            return out
        d.draft_block = draft_block
    if m is not None:
        rnd = m.round

        def round_(room):
            kv = [own(a[..., :c.offset, :]) for c in m.cache if c.keys is not None for a in (c.keys, c.values)]
            out = rnd(room)
            rec.append(("mtp", m.pos, kv, out if isinstance(out, int) else own(out)))
            return out
        m.round = round_
    return rec


def session(dense, monkeypatch, sw, kind, reqs, window=16, **kw):
    """`reqs` [(ids, max_tokens, params)] in order on one scheduler with the
    switches `sw` -> per request (tokens, reused, the cache's entries after
    it), the drafter contexts."""
    monkeypatch.setattr(worker, "PREFILL_NO_SEED", sw[0])
    monkeypatch.setattr(worker, "PREFILL_DEFER", sw[1])
    monkeypatch.setattr(worker, "CLOCK", worker.Clock())
    sched = scheduler(dense, max_batch=1, **kw)
    d = tiny_dflash(dense, window=window) if kind != "head" else None
    m = worker.MTPDraft(head(dense), dense) if kind != "dflash" else None
    monkeypatch.setattr(worker, "DRAFT_HYBRID", {"both": "dflash", "both-mtp": "mtp"}.get(kind, "1"))
    sched.eng.draft, sched.eng.mtp = d, m
    rec = spy(d, m)
    out = []
    for i, (ids, n, params) in enumerate(reqs):
        r = request(ids, n, f"r{i}", **params)
        out.append((run(sched, [r])[r.id][0], r.reused, entries(sched.eng._pcache)))
    if m is not None:                                       # the head's cache at the end
        rec.append(("head", m.pos, [own(a[..., :c.offset, :]) for c in m.cache if c.keys is not None
                                    for a in (c.keys, c.values)]))
    return out, rec


def equal(a, b):
    """Nested records equal, arrays bit for bit (shape, dtype, values)."""
    if isinstance(a, mx.array) or isinstance(b, mx.array):
        return (isinstance(a, mx.array) and isinstance(b, mx.array) and a.shape == b.shape
                and a.dtype == b.dtype and mx.array_equal(a, b).item())
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def plain(dense, monkeypatch, reqs, **kw):
    monkeypatch.setattr(worker, "CLOCK", worker.Clock())
    sched = scheduler(dense, max_batch=1, **kw)
    out = []
    for i, (ids, n, params) in enumerate(reqs):
        r = request(ids, n, f"p{i}", **params)
        out.append(run(sched, [r])[r.id][0])
    return out


def check(dense, monkeypatch, kind, reqs, window=16, **kw):
    ref = plain(dense, monkeypatch, reqs, **kw)
    base = None
    for sw in SWITCHES:
        got = session(dense, monkeypatch, sw, kind, reqs, window, **kw)
        assert [o[0] for o in got[0]] == ref, sw
        if base is None:
            base = got
            assert any(r[0] in ("dflash", "mtp") for r in got[1])     # (rounds were drafted)
            continue
        for i, (x, y) in enumerate(zip(base[0], got[0])):
            assert x[1] == y[1], (sw, i)                            # the same resume points
            assert equal(x[2], y[2]), (sw, i, "entries")
        assert len(base[1]) == len(got[1]), sw
        for j, (x, y) in enumerate(zip(base[1], got[1])):
            assert equal(x, y), (sw, j, x[0])
    return base


KINDS = ["head", "dflash", "both", "both-mtp"]


@pytest.mark.parametrize("win", [16, 256], ids=["window16", "window256"])
@pytest.mark.parametrize("mode", ["greedy", "sampled"])
@pytest.mark.parametrize("kind", KINDS)
def test_conversation_with_the_cache(dense, small, monkeypatch, kind, mode, win):
    """Cache on, reply KV on, slabs of 8: three turns (each a hit through
    the previous turn's checkpoints or reply), an exact repeat of the first
    (nothing prefilled), a request that ends at its first token. The
    drafters' window (DFlash's sliding window, the head's DRAFT_TAIL_END):
    16, the last two slabs deferred; 256, every slab of these prompts and
    the checkpoints' contexts inside them."""
    monkeypatch.setattr(worker, "PROMPT_CACHE_REPLY", True)
    monkeypatch.setattr(worker, "DRAFT_TAIL", 16)
    monkeypatch.setattr(worker, "DRAFT_TAIL_END", win)
    p = CARD if mode == "sampled" else {}
    a = conversation(3, 611, system=30, user=lambda i: 20, answer=lambda i: 8)
    reqs = [(ids, 10, p) for ids in a] + [(a[0], 10, p), (prompt(612, user=40), 1, p)]
    base = check(dense, monkeypatch, kind, reqs, window=win)
    assert [o[1] > 0 for o in base[0]] == [False, True, True, True, False]


@pytest.mark.parametrize("step", [8, 4096], ids=["slabs", "one-slab"])
@pytest.mark.parametrize("kind", KINDS)
def test_cold_without_the_cache(dense, monkeypatch, kind, step):
    """Cache off: a prompt in many slabs and in one; greedy and sampled."""
    monkeypatch.setattr(worker, "PROMPT_CACHE", False)
    monkeypatch.setattr(worker, "PREFILL_STEP", step)
    monkeypatch.setattr(worker, "PREFILL_MIN_STEP", 4)
    reqs = [(prompt(613, user=60), 14, {}), (prompt(614, user=50), 14, CARD)]
    check(dense, monkeypatch, kind, reqs)


@pytest.mark.parametrize("kind", ["head", "both"])
def test_q8_kv(dense, small, monkeypatch, kind):
    """q8 KV from position 40 (the prompt crosses it): cold, then the next
    turn's hit."""
    monkeypatch.setattr(worker, "DRAFT_TAIL", 16)
    monkeypatch.setattr(worker, "DRAFT_TAIL_END", 16)
    a = conversation(2, 615, system=30, user=lambda i: 20, answer=lambda i: 8)
    check(dense, monkeypatch, kind, [(ids, 10, {}) for ids in a], kv_bits=8, kv_start=40)


def flat(x):
    return len(arrays(x if isinstance(x, (list, tuple)) else [x]))


@pytest.mark.parametrize("defer", [False, True])
def test_the_first_token_goes_out_before_the_drafters_work(dense, monkeypatch, defer):
    """The mechanism (one-slab prompt, both drafters, cache off): with
    PREFILL_DEFER the last eval before the first token takes the token alone
    and the drafters' share is queued (one async_eval) right after it is
    emitted; off: the token with the drafter's window. PREFILL_NO_SEED: the
    head's state in a slab's eval without the seed."""
    monkeypatch.setattr(worker, "PROMPT_CACHE", False)
    monkeypatch.setattr(worker, "PREFILL_STEP", 4096)
    monkeypatch.setattr(worker, "PREFILL_DEFER", defer)
    monkeypatch.setattr(worker, "DRAFT_HYBRID", "dflash")
    sched = scheduler(dense, max_batch=1)
    sched.eng.draft, sched.eng.mtp = tiny_dflash(dense), worker.MTPDraft(head(dense), dense)
    log = []
    ev, aev = mx.eval, mx.async_eval
    monkeypatch.setattr(mx, "eval", lambda *a: (log.append(("eval", flat(a))), ev(*a))[1])
    monkeypatch.setattr(mx, "async_eval", lambda *a: (log.append(("async", flat(a))), aev(*a))[1])
    seeds = []
    st = worker.MTPDraft.state
    monkeypatch.setattr(worker.MTPDraft, "state", lambda self, seed=True: (seeds.append(seed), st(self, seed))[1])
    r = request(prompt(616, user=60), 6, "x")

    class Out:
        """The request's queue, the first token's emission in the log."""
        def __init__(self, q):
            self.q = q

        def put(self, e):
            if e[0] == "tok" and ("tok",) not in log:
                log.append(("tok",))
            self.q.put(e)

        def __getattr__(self, name):
            return getattr(self.q, name)
    r.out = Out(r.out)
    run(sched, [r])
    t = log.index(("tok",))
    assert log[t - 1] == ("eval", 1 if defer else 2)
    assert log[t + 1][0] == "async" and (log[t + 1][1] > 1 if defer else True)
    if not defer:
        assert seeds[:1] == [False]                         # the one slab's eval (PREFILL_NO_SEED)


def test_no_drafter_takes_the_old_path(dense, monkeypatch):
    """A speculative row without a drafter (prompt lookup only: no hidden
    states captured) evaluates the same arrays in the same calls with the
    switches on as with both off, and gives the plain loop's tokens."""
    monkeypatch.setattr(worker, "LOOKUP", "1")
    monkeypatch.setattr(worker, "PROMPT_CACHE", False)
    monkeypatch.setattr(worker, "PREFILL_STEP", 8)
    monkeypatch.setattr(worker, "PREFILL_MIN_STEP", 4)
    reqs = [(prompt(617, user=40), 12, {}), (prompt(618, user=30), 12, CARD)]
    ref = plain(dense, monkeypatch, [(ids, n, p) for ids, n, p in reqs])
    ev, aev = mx.eval, mx.async_eval
    logs = []
    for sw in [(False, False), (True, True)]:
        monkeypatch.setattr(worker, "PREFILL_NO_SEED", sw[0])
        monkeypatch.setattr(worker, "PREFILL_DEFER", sw[1])
        log = []
        monkeypatch.setattr(mx, "eval", lambda *a: (log.append(("eval", flat(a))), ev(*a))[1])
        monkeypatch.setattr(mx, "async_eval", lambda *a: (log.append(("async", flat(a))), aev(*a))[1])
        sched = scheduler(dense, max_batch=1)
        assert sched.eng.draft is None and getattr(sched.eng, "mtp", None) is None
        out = []
        for i, (ids, n, p) in enumerate(reqs):
            r = request(ids, n, f"n{i}", **p)
            out.append(run(sched, [r])[r.id][0])
            assert r.spec_stats is not None                 # (the row went through _spec)
        monkeypatch.setattr(mx, "eval", ev)
        monkeypatch.setattr(mx, "async_eval", aev)
        assert out == ref, sw
        logs.append(log)
    assert logs[0] == logs[1] and logs[0]
