"""The scheduler end to end on a tiny random fp16 model (fp32 on the CPU, where a
batched step matches a lone one to ~1e-6): every request's tokens must not
depend on what else runs (joins, leaves, interleaved prefill, sync mode),
and the admission / cancel / error rules must hold."""
import threading
import types

import mlx.core as mx
import pytest

import worker
from conftest import IM_START, FakeTok, prompts, request, run, scheduler, vision_ids, vision_inputs


@pytest.fixture(params=["dense", "dense_fused"])
def dense_or_fused(request):
    return request.getfixturevalue(request.param)


def serial(lm, reqs_fn, **kw):
    """Each request alone (max_batch 1, submitted one after the other)."""
    out = {}
    for r in reqs_fn():
        out.update(run(scheduler(lm, max_batch=1, **kw), [r]))
    return out


def mixed(seed=1):
    ps = prompts([1, 9, 23, 5, 40, 14], seed)
    return lambda: [
        request(ps[0], 10, "one-token"),
        request(ps[1], 1, "max1"),
        request(ps[2], 17, "long"),
        request(ps[3], 0, "max0"),
        request(ps[4], 25, "sampled", temperature=0.9, top_p=0.9, top_k=40, seed=7),
        request(ps[5], 20, "penalties", presence_penalty=0.7, frequency_penalty=0.3,
                repetition_penalty=1.3),
    ]


@pytest.mark.parametrize("model", ["dense", "moe", "dense_fused"])
def test_batched_tokens_equal_solo(model, request):   # pytest's own `request` fixture
    lm = request.getfixturevalue(model)
    make = mixed()
    ref = serial(lm, make)
    reqs = make()
    # staggered arrivals: rows join a running batch and prefill interleaves
    sizes = []
    got = run(scheduler(lm, max_batch=4), reqs, arrive={3: reqs[2:4], 9: reqs[4:]},
              on_tick=lambda t, s: sizes.append(len(s.rows)))
    assert max(sizes) >= 3
    reqs = make()                       # prefill interleaved with decode by time share
    assert run(scheduler(lm, max_batch=4, share=0.3), reqs,
               arrive={3: reqs[2:4], 9: reqs[4:]}) == got
    assert got.keys() == ref.keys()
    for k in ref:
        assert got[k] == ref[k], k
    assert ref["max0"][0] == [] and ref["max0"][2] == ("done", "length")
    assert len(ref["max1"][0]) == 1 and ref["long"][2] == ("done", "length")
    assert len(ref["long"][0]) == 17


def test_kv_quantized_mid_batch(dense):
    """q8 KV that crosses quantized_kv_start while batched (and q4)."""
    for bits in (8, 4):
        make = mixed(2)
        ref = serial(dense, make, kv_bits=bits, kv_start=30)
        reqs = make()
        mixed_kv = []

        def watch(t, s):
            kinds = {hasattr(r.cache[3], "bits") for r in s.rows}
            mixed_kv.append(len(s.rows) >= 2 and True in kinds)
        got = run(scheduler(dense, kv_bits=bits, kv_start=30), reqs,
                  arrive={0: reqs[:3], 2: reqs[3:]}, on_tick=watch)
        assert any(mixed_kv)                # quantized rows decoded in a batch
        for k in ref:
            assert got[k] == ref[k], (bits, k)


def test_stop_ids_ignore_eos_min_tokens(dense):
    ps = prompts([12, 12, 12], 3)
    probe = run(scheduler(dense, max_batch=1), [request(ps[0], 15, "probe")])["probe"][0]
    stop_tok = probe[5]

    def make():
        return [request(ps[0], 15, "stop", stop_token_ids=[stop_tok]),
                request(ps[1], 15, "ignore", ignore_eos=True,
                        stop_token_ids=[], allowed_token_ids=list(range(0, 64))),
                request(ps[2], 15, "min", min_tokens=6, bad_words=["ab"])]
    ref = serial(dense, make)
    got = run(scheduler(dense), make())
    assert got == ref
    toks, shown, end = ref["stop"]
    assert end == ("done", "stop") and toks == probe[:probe.index(stop_tok)]
    assert all(t < 64 for t in ref["ignore"][0])


def test_ignore_eos_is_not_text(dense):
    """An EOS under ignore_eos is sent but marked not shown, and generation goes on."""
    ps = prompts([10], 4)
    lone = run(scheduler(dense, max_batch=1), [request(ps[0], 8, "probe")])["probe"][0]
    r = request(ps[0], 8, "x", ignore_eos=True)
    r.ctl.eos = {lone[2]}                               # its 3rd token is an EOS now
    toks, shown, end = run(scheduler(dense), [r])["x"]
    assert toks == lone and end == ("done", "length")
    assert shown == [t != lone[2] for t in lone]


class AllowAll:
    """A guide that masks nothing: puts its row in sync mode (no step-ahead);
    `bad` makes consume() fail at that token count (grammar rejected)."""

    def __init__(self, bad=None):
        self.seen, self.bad = [], bad

    def consume(self, tok):
        self.seen.append(tok)
        if self.bad is not None and len(self.seen) == self.bad:
            raise RuntimeError(f"grammar: token {tok} rejected")

    def sample(self, logits, sampler):
        return sampler(logits)


def test_sync_mode_and_row_error(dense):
    make = mixed(5)
    ref = serial(dense, make)
    reqs = make()
    g = request(prompts([8], 6)[0], 30, "guided", guide=AllowAll())
    bad = request(prompts([8], 7)[0], 30, "bad", guide=AllowAll(bad=4))
    got = run(scheduler(dense, max_batch=8), reqs + [g, bad],
              arrive={0: reqs[:3] + [g, bad], 6: reqs[3:]})
    for k in ref:
        assert got[k] == ref[k], k
    assert got["bad"][2][0] == "error" and "rejected" in got["bad"][2][2]
    assert len(got["bad"][0]) == 3
    assert g.guide.seen == got["guided"][0]
    lone = run(scheduler(dense, max_batch=1),
               [request(prompts([8], 6)[0], 30, "guided", guide=AllowAll())])
    assert lone["guided"] == got["guided"]


def test_thinking_budget_forced_tokens(dense):
    from mlx_vlm.utils import ThinkingBudgetCriteria
    NL, CLOSE, OPEN = 11, 12, 13
    tok = FakeTok({"</think>": [CLOSE], "<think>": [OPEN], "\n": [NL]})

    def make():
        crit = ThinkingBudgetCriteria(tokenizer=tok, thinking_budget=4, thinking_end_token="</think>",
                                      thinking_start_token="<think>", enable_thinking=True,
                                      prompt_preopens_thinking=True)
        return [request(prompts([9], 8)[0] + [OPEN], 12, "think", thinking=crit)] + mixed(8)()[:3]
    ref = serial(dense, make)
    got = run(scheduler(dense), make())
    assert got == ref
    toks = ref["think"][0]
    assert toks[5:7] == [NL, CLOSE], toks       # 5 thinking tokens > budget 4 -> \n</think>


def test_thinking_budget_with_grammar_row(dense):
    """A row with both a thinking budget and a grammar (sync mode + forced
    tokens) in a batch with plain rows: every row as when alone."""
    from mlx_vlm.utils import ThinkingBudgetCriteria
    NL, CLOSE, OPEN = 11, 12, 13
    tok = FakeTok({"</think>": [CLOSE], "<think>": [OPEN], "\n": [NL]})

    def make():
        crit = ThinkingBudgetCriteria(tokenizer=tok, thinking_budget=3, thinking_end_token="</think>",
                                      thinking_start_token="<think>", enable_thinking=True,
                                      prompt_preopens_thinking=True)
        both = request(prompts([9], 49)[0] + [OPEN], 14, "both", thinking=crit, guide=AllowAll())
        return [both] + mixed(50)()[:3]
    ref = serial(dense, make)
    reqs = make()
    got = run(scheduler(dense), reqs, arrive={2: reqs[:1]})
    assert got == ref
    assert ref["both"][0][4:6] == [NL, CLOSE]
    assert reqs[0].guide.seen == got["both"][0]


def test_cancel_waiting_prefill_decode(dense, monkeypatch):
    monkeypatch.setattr(worker, "PREFILL_STEP", 4)
    monkeypatch.setattr(worker, "PREFILL_MIN_STEP", 2)
    make = mixed(9)
    ref = serial(dense, make)
    reqs = make()
    ps = prompts([30, 10, 10], 10)
    in_prefill = request(ps[0], 10, "c-prefill")
    in_decode = request(ps[1], 40, "c-decode")
    waiting = request(ps[2], 10, "c-waiting")
    sched = scheduler(dense, max_batch=3)
    slabs = []

    def on_tick(t, s):
        if s.task_row is not None and s.task_row.req is in_prefill:
            slabs.append(t)
            if len(slabs) == 3:                         # two slabs in
                in_prefill.cancel.set()
        if t == 1:
            waiting.cancel.set()
        if t == 12:
            in_decode.cancel.set()
    got = run(sched, reqs + [in_prefill, in_decode, waiting],
              arrive={0: [in_decode] + reqs[:3] + [in_prefill, waiting], 20: reqs[3:]},
              on_tick=on_tick)
    for k in ref:
        assert got[k] == ref[k], k
    for k in ("c-prefill", "c-decode", "c-waiting"):
        assert got[k][2] == ("done", "cancelled"), k
    assert got["c-waiting"][0] == [] and got["c-prefill"][0] == []
    assert 0 < len(got["c-decode"][0]) < 40
    assert sched.reserved == 0 and sched.rows == [] and sched.gdn is None


def test_admission(dense):
    ps = prompts([20] * 6, 11)
    sched = scheduler(dense, max_batch=3)
    seen = []
    reqs = [request(p, 30, f"r{i}") for i, p in enumerate(ps)]
    run(sched, reqs, on_tick=lambda t, s: seen.append(len(s.rows) + (s.task is not None)))
    assert max(seen) == 3

    # budget: room for one row; the batch always takes the head when empty,
    # and a later small request never overtakes the waiting head (FIFO)
    need = scheduler(dense)._reserve(reqs[0])
    sched = scheduler(dense, budget=need + 1)
    big = request(ps[0], 300, "big")
    small = request(ps[1], 2, "small")
    first = request(ps[2], 20, "first")
    rows = []
    run(sched, [first, big, small], on_tick=lambda t, s: rows.append(len(s.rows)))
    assert max(rows) == 1
    assert first.t0 < big.t0 < small.t0
    assert sched.reserved == 0

    # live memory guard: a cap below the active memory admits one at a time
    sched = scheduler(dense, cap=0)
    seen.clear()
    run(sched, [request(p, 5, f"m{i}") for i, p in enumerate(ps[:3])],
        on_tick=lambda t, s: seen.append(len(s.rows) + (s.task is not None)))
    assert max(seen) == 1


def test_stats(dense):
    ps = prompts([10, 10, 10], 12)
    reqs = [request(p, 20, f"s{i}") for i, p in enumerate(ps)]
    run(scheduler(dense, max_batch=1), reqs[:1])
    # the first step (last prompt token) + one per token, the last being the
    # look-ahead whose token is never read
    assert reqs[0].steps == 21 and reqs[0].batch_mean == 1.0
    run(scheduler(dense, max_batch=4), reqs[1:])
    assert all(1.0 < r.batch_mean <= 2.0 for r in reqs[1:])
    assert all(r.queue_ms >= 0 and r.t0 >= r.t_arrival for r in reqs)


def test_prompt_cache_while_batching(dense_or_fused, monkeypatch):
    """Requests sharing a system prompt hit the prefix cache of the one before
    them, and their output equals a cold run (slab cuts depend only on the
    prompt)."""
    dense = dense_or_fused
    monkeypatch.setattr(worker, "PROMPT_CACHE", True)
    monkeypatch.setattr(worker, "PREFILL_STEP", 8)
    monkeypatch.setattr(worker, "PREFILL_MIN_STEP", 4)
    system = [IM_START] + prompts([30], 13)[0]
    tails = prompts([6, 9, 12], 14)

    def make():
        return [request(system + [IM_START] + t, 12, f"c{i}") for i, t in enumerate(tails)]
    cold = serial(dense, make)
    sched = scheduler(dense)
    reqs = make()
    got = run(sched, reqs, arrive={0: reqs[:1], 2: reqs[1:]})
    assert got == cold
    assert reqs[0].reused == 0 and all(r.reused > 0 for r in reqs[1:])


def test_scheduler_thread(dense):
    """The real thread: submissions from several threads, idle and wake-up."""
    make = mixed(15)
    ref = serial(dense, make)
    sched = scheduler(dense)
    sched.start()
    out = {}

    def client(r):
        sched.submit(r)
        toks, shown = [], []
        while True:
            ev = r.out.get(timeout=60)
            if ev[0] == "tok":
                toks.append(ev[1])
                shown.append(ev[2])
            else:
                out[r.id] = (toks, shown, ev)
                return
    for wave in (make()[:3], make()[3:]):
        ts = [threading.Thread(target=client, args=(r,)) for r in wave]
        for t in ts:
            t.start()
        for t in ts:
            t.join(60)
    assert out == ref
    assert mx.default_device().type == mx.cpu


def vision_request(seed, rid, max_tokens=12):
    ids = vision_ids(seed)
    r = request(ids, max_tokens, rid)
    r.images, r.vision = ["img.png"], vision_inputs(ids)
    return r


def test_vision_rows(dense, monkeypatch):
    """An image prompt: counted on the scheduler, prefilled over several slabs
    with its mRoPE positions (the same answer as one slab), then decoded with
    its own position delta inside a text batch (the same answer as alone)."""
    one = run(scheduler(dense, max_batch=1), [vision_request(20, "v")])["v"]
    monkeypatch.setattr(worker, "PREFILL_STEP", 8)
    monkeypatch.setattr(worker, "PREFILL_MIN_STEP", 4)
    sched = scheduler(dense, max_batch=1)
    v = vision_request(20, "v")
    slabs = run(sched, [v])["v"]
    assert slabs == one and v.prompt_tokens == 57 and v.vision is None
    make = mixed(21)
    ref = serial(dense, make)
    reqs = make()
    v = vision_request(20, "v")
    deltas = []
    got = run(scheduler(dense), reqs + [v], arrive={2: [v]},
              on_tick=lambda t, s: deltas.append([r.delta for r in s.rows]))
    assert got["v"] == one
    for k in ref:
        assert got[k] == ref[k], k
    assert any(len(d) >= 3 and any(d) for d in deltas)       # a delta row inside a batch


def no_metal_verify(monkeypatch):
    """The multi-token verify forward's Metal-only kernels -> their generic
    paths (mlx-vlm 0.7.6: speculative.ops.linear)."""
    from mlx_vlm.speculative.ops import linear
    monkeypatch.setattr(linear, "_use_target_verify_dense", lambda *a: False)


class FakeDraft:
    """A DFlash stand-in that proposes the known greedy answer, with a wrong
    last draft every other round (accepted prefix + rollback exercised)."""

    def __init__(self, answer, block=4):
        self.answer = answer
        self.config = types.SimpleNamespace(target_layer_ids=[0, 3], sliding_window=8,
                                            block_size=block, runtime_block_size=None)
        self.prefer_requested_block_size = True
        self.accept_lens, self.draft_lens = [], []

    def reset(self, model):
        return None

    def draft_block(self, b, hidden, cache, bs, sampler, dtype):
        at = 1 + sum(int(a) + 1 for a in self.accept_lens)    # tokens emitted so far
        d = (self.answer[at:at + bs - 1] + [0] * bs)[:bs - 1]
        if len(self.accept_lens) % 2:
            d[-1] = (d[-1] + 1) % 400
        return mx.array([d], dtype=dtype)


def test_dflash_exact_and_demoted(dense_or_fused, monkeypatch):
    """The speculative path is exact for greedy: the same tokens as plain
    decoding (also checks the first target call's position, appendix A1),
    and a second request demotes it to a batch row mid-answer. dense_fused:
    the verify forward runs over the fused model's row-view weights."""
    dense = dense_or_fused
    no_metal_verify(monkeypatch)
    ps = prompts([25, 12], 30)
    plain = run(scheduler(dense, max_batch=1), [request(ps[0], 30, "a")])["a"]
    other = run(scheduler(dense, max_batch=1), [request(ps[1], 20, "b")])["b"]
    sched = scheduler(dense, max_batch=1)
    sched.eng.draft = FakeDraft(plain[0])
    assert run(sched, [request(ps[0], 30, "a")])["a"] == plain
    assert len(sched.eng.draft.accept_lens) > 3 and max(sched.eng.draft.accept_lens) >= 2

    sched = scheduler(dense, max_batch=4)
    sched.eng.draft = FakeDraft(plain[0])
    a, b = request(ps[0], 30, "a"), request(ps[1], 20, "b")
    fa, P, kv = dense.model.fa_idx, len(ps[0]), {}
    join = sched._join

    def spy(row):                       # the demoted row's keys of the prompt
        if row.req is a:
            kv["spec"] = row.cache[fa].keys[..., :P, :]
        join(row)
    sched._join = spy
    got = run(sched, [a, b], arrive={4: [b]})
    assert got == {"a": plain, "b": other}
    assert 0 < len(sched.eng.draft.accept_lens) < 10      # rounds, then a plain batch row
    # the last prompt token's key went in at position P-1 (not 0)
    c = dense.make_cache()
    dense(mx.array([ps[0][:-1]]), cache=c)
    dense(mx.array([ps[0][-1:]]), cache=c, position_ids=mx.array([[P - 1]], dtype=mx.int32))
    assert mx.abs(kv["spec"] - c[fa].keys[..., :P, :]).max().item() < 1e-4


def test_dflash_not_demoted_for_a_request_that_cannot_join(dense, monkeypatch):
    """A waiting request that would not be admitted next to the DFlash row
    (here: over budget) does not cost that row its speculative decoding."""
    no_metal_verify(monkeypatch)
    ps = prompts([25, 12], 30)
    plain = run(scheduler(dense, max_batch=1), [request(ps[0], 30, "a")])["a"]
    probe = scheduler(dense)
    sched = scheduler(dense, max_batch=4, budget=probe._reserve(request(ps[0], 30, "a")))
    sched.eng.draft = FakeDraft(plain[0])
    a, b = request(ps[0], 30, "a"), request(ps[1], 20, "b")
    joined = []
    sched._join = (lambda join: lambda row: (joined.append(row.req.id), join(row)))(sched._join)
    got = run(sched, [a, b], arrive={4: [b]})
    assert got["a"] == plain and got["b"][2] == ("done", "length")
    assert "a" not in joined                     # never demoted: DFlash ran to the end
    assert b.t0 > a.t0 and len(sched.eng.draft.accept_lens) >= 8


def test_sample_rows_equal_per_row():
    """sample_rows (rows with equal settings filtered as one stack, drawn with
    their own keys) == each row's sampler alone, over steps, mixed dtypes."""
    cfgs = [dict(temperature=0.0), dict(temperature=0.6, top_p=0.95, top_k=20),
            dict(temperature=0.7, top_p=0.9), dict(temperature=0.8, min_p=0.05)]
    rows = [(dict(worker.SAMPLING_KWARGS, **cfgs[i % 4]), 11 * i,
             mx.float32 if i % 3 == 0 else mx.float16) for i in range(10)]
    a = [worker.make_sampler(c, seed=s) for c, s, _ in rows]
    b = [worker.make_sampler(c, seed=s) for c, s, _ in rows]
    for step in range(6):
        lg = (mx.random.normal((10, 512), key=mx.random.key(step)) * 3).astype(mx.float16)
        xs = [lg[i:i + 1].astype(dt) for i, (_, _, dt) in enumerate(rows)]
        want = [a[i](xs[i]).tolist() for i in range(10)]
        assert [y.tolist() for y in worker.sample_rows(b, xs)] == want


def test_grouped_sampling_rows_equal_solo(dense):
    """Several seeded rows with the same sampler settings (one group in the
    batched draw) next to greedy and penalized rows: tokens == solo."""
    ps = prompts([7, 12, 5, 9, 16, 3], seed=4)

    def make():
        card = dict(temperature=0.6, top_p=0.95, top_k=20)
        return [request(ps[0], 14, "c0", seed=1, **card), request(ps[1], 14, "c1", seed=2, **card),
                request(ps[2], 14, "c2", seed=3, **card), request(ps[3], 14, "g0"),
                request(ps[4], 14, "g1"), request(ps[5], 14, "p0", presence_penalty=0.8)]
    ref = serial(dense, make)
    sizes = []
    got = run(scheduler(dense, max_batch=6), make(), on_tick=lambda t, s: sizes.append(len(s.rows)))
    assert max(sizes) == 6
    assert got == ref


def test_sample_rows_error_fails_its_group_only():
    """An exception while building one group's draw: on_error for that
    group's rows (their entries None); the other groups are drawn as alone."""
    card = dict(worker.SAMPLING_KWARGS, temperature=0.6, top_p=0.95, top_k=20)
    hot = dict(worker.SAMPLING_KWARGS, temperature=0.9, min_p=0.1)
    greedy = dict(worker.SAMPLING_KWARGS, temperature=0.0)
    cfgs = [card, hot, card, greedy, hot, greedy]
    a = [worker.make_sampler(c, seed=i) for i, c in enumerate(cfgs)]
    b = [worker.make_sampler(c, seed=i) for i, c in enumerate(cfgs)]

    def boom(x):
        raise RuntimeError("boom")
    b[1].filtered = boom                     # the "hot" group's first sampler
    lg = mx.random.normal((6, 512), key=mx.random.key(3)).astype(mx.float16)
    xs = [lg[i:i + 1] for i in range(6)]
    failed = []
    got = worker.sample_rows(b, xs, on_error=lambda i, e: failed.append((i, str(e))))
    assert failed == [(1, "boom"), (4, "boom")]
    assert got[1] is None and got[4] is None
    for i in (0, 2, 3, 5):
        assert got[i].tolist() == a[i](xs[i]).tolist(), i
    with pytest.raises(RuntimeError):
        worker.sample_rows(b, xs)
