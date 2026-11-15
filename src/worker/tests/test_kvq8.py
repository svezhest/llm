"""Quantized KV (QWEN_KV_BITS=8) as a single-stream option without losing
the features that needed fp16 KV (2026-11-15, docs/perf/2026-11-15/design/
kv_q8.md): one KV copy (QWEN_PROMPT_CACHE_MOVE) over quantized segments,
DFlash over a quantized cache, the dequantized fused prefill
(QWEN_KVQ_PREFILL_FUSED) and the memory accounting.

kv_start 0: quantized from the first slab; kv_start 40: an fp16 warm-up, then
quantized (in a prefill slab or a decode step), the only resume point below it
an fp16 entry."""
import logging

import mlx.core as mx
import pytest
from mlx_vlm.models import base
from mlx_vlm.models.cache import KVCache, QuantizedKVCache

import worker
import test_reply_kv as rk
from conftest import Cold, check_cold, prompts, request, run, scheduler
from test_partial_store import cancel_after
from test_pcache_move import (TH, THC, cancel_at_lookup, parity, Play, play, small,  # noqa: F401
                              think_first, think_next)
from test_scheduler import FakeDraft, no_metal_verify

KV = {"q0": dict(kv_bits=8, kv_start=0), "q40": dict(kv_bits=8, kv_start=40)}


@pytest.fixture(params=sorted(KV))
def kv(request):
    return KV[request.param]


def grow(turns=4, seed=300, n=lambda t: 5 + t):
    """A conversation that starts below kv_start 40 (23 tokens) and grows past
    it (fp16 entry, then quantized ones)."""
    def script(p):
        ids = rk.first(seed, system=8, user=6)
        for t in range(turns):
            toks = p.do(ids, n(t), f"t{t}")
            ids = rk.nxt(ids, toks, seed + 10 + t, user=6)
    return script


# --- one KV copy (QWEN_PROMPT_CACHE_MOVE) over quantized KV ------------------------
def test_move_chat_parity(dense, small, monkeypatch, kv, caplog):
    """Copies vs move: same answers, reused tokens and cache contents (ids,
    checkpoints, low, every resumable position's caches bit for bit), with
    the reply KV; quantized entries move into the row."""
    caplog.set_level(logging.INFO, logger="qwen.worker")
    a, b = parity(monkeypatch, dense, grow(), **kv)
    assert all(r > 0 for _, _, r in b.trace[1:])
    assert any(e.quantized for e in b.pc.entries)
    assert any("(moved: the entry's KV is the row's now)" in m for m in caplog.messages)
    if kv["kv_start"]:
        # the fp16 floor (the first turn's prompt) stays beside the quantized chain
        assert any(not e.quantized for e in b.pc.entries)


def test_move_chat_parity_fused(dense_fused, small, monkeypatch):
    """The production layout (8-bit weights, fp16 scales, fused GDN), q8 KV
    crossing the start."""
    parity(monkeypatch, dense_fused, grow(3), **KV["q40"])


def test_move_without_reply_kv(dense, small, monkeypatch, kv):
    monkeypatch.setattr(worker, "PROMPT_CACHE_REPLY", False)
    a, b = parity(monkeypatch, dense, grow(5), **kv)
    assert any(e.chained for e in b.pc.entries)


def test_move_reply_crossing_the_start(dense, small, monkeypatch):
    """A prompt below kv_start whose answer decodes past it: the row is fp16
    at its prefill's end, so it stores as without the mode (the fp16 prompt
    entry, then a quantized reply entry); the next turns move the quantized
    one."""
    a, b = parity(monkeypatch, dense, grow(3, seed=310, n=lambda t: 30 - 10 * t),
                  **KV["q40"], eos=())
    assert sorted(e.quantized for e in b.pc.entries) == [False, True]


def test_move_thinking_replies(dense, small, monkeypatch, kv):
    def script(p):
        ids = think_first(320, system=12, user=6)
        for t in range(4):
            toks = p.do(ids, 6 + t, f"t{t}")
            ids = think_next(ids, toks[3:], 321 + t, user=6)
    parity(monkeypatch, dense, script, think=[TH, THC], **kv)


@pytest.mark.parametrize("kind", ["miss", "hit"])
def test_move_cancelled_prefill(dense, small, monkeypatch, kv, kind):
    def script(p):
        ids = rk.first(330, system=8, user=40 if kind == "miss" else 6)
        if kind == "hit":
            toks = p.do(ids, 5, "t0")
            ids = rk.nxt(ids, toks, 331, user=40)
        assert p.do(ids, 6, "x", on_tick=lambda x: cancel_after(x, 3)) == []
        p.do(ids, 6, "y")
    a, b = parity(monkeypatch, dense, script, **kv)
    assert b.trace[-1][2] > 0


def test_move_cancel_before_the_first_slab(dense, small, monkeypatch, kv):
    def script(p):
        ids = rk.first(340, system=30, user=10)
        toks = p.do(ids, 5, "t0")
        ids = rk.nxt(ids, toks, 341)
        cancel_at_lookup(p, "x")
        assert p.do(ids, 6, "x") == []
        p.do(ids, 6, "y")
    a, b = parity(monkeypatch, dense, script, **kv)
    assert b.trace[1][2] == b.trace[2][2] > 0


def test_move_dflash_rows(dense, small, monkeypatch, kv):
    """DFlash rows over quantized KV with the move: the plain answers, the
    same entries (and drafter contexts) as with the copies."""
    def script(p):
        ids = rk.first(350, system=8, user=8)
        for t in range(3):
            plain = run(scheduler(dense, max_batch=1, **kv), [request(ids, 9 + t, "ref")])["ref"][0]
            assert p.do(ids, 9 + t, f"t{t}", draft=plain) == plain
            ids = rk.nxt(ids, plain, 360 + t, user=10)
    a, b = parity(monkeypatch, dense, script, **kv)
    assert len(b.ctx) == 3 and all(r > 0 for _, _, r in b.trace[1:])


def test_fp16_entry_stays_quantized_entry_moves(dense, small, monkeypatch, caplog):
    """kv_start 40: the fp16 entry of a short first turn is copied by the
    next (longer, quantized) turn -- it stays in the cache while that row
    runs; the quantized entry that turn stores moves into the turn after."""
    caplog.set_level(logging.INFO, logger="qwen.worker")
    monkeypatch.setattr(worker, "PROMPT_CACHE_MOVE", True)
    p = Play(dense, True, **KV["q40"])
    ids = rk.first(370, system=8, user=6)
    toks = p.do(ids, 5, "t0")
    (floor,) = p.pc.entries
    assert not floor.quantized
    seen = []

    def watch(x):
        def on_tick(t, s):
            if s.rows or s.task:
                seen.append(tuple(e.quantized for e in p.pc.entries))
        return on_tick
    ids = rk.nxt(ids, toks, 371, user=20)
    toks = p.do(ids, 5, "t1", on_tick=watch)
    assert p.trace[-1][2] > 0 and set(seen) == {(False,)}       # copied, not moved
    assert sorted(e.quantized for e in p.pc.entries) == [False, True]
    seen.clear()
    p.do(rk.nxt(ids, toks, 372), 5, "t2", on_tick=watch)
    assert set(seen[1:]) == {(False,)}                          # the quantized one is in the row
    assert sum("(moved: the entry's KV is the row's now)" in m for m in caplog.messages) == 1


# --- DFlash over quantized KV ------------------------------------------------------------
class Blocks(FakeDraft):
    def __init__(self, answer, block=4):
        super().__init__(answer, block)
        self.sizes = []

    def draft_block(self, b, hidden, cache, bs, sampler, dtype):
        self.sizes.append(bs)
        return super().draft_block(b, hidden, cache, bs, sampler, dtype)


def final_kv(sched, rid):
    """The row's KV when it ends: {rid: [dequantized (k, v) per FA layer]}."""
    got = {}
    fin = sched._finish

    def spy(row, why):
        if row.req.id == rid and rid not in got:
            out = []
            for c in row.cache:
                if hasattr(c, "cache"):
                    continue
                if hasattr(c, "bits"):
                    k, v = (mx.dequantize(*(y[..., :c.offset, :] for y in x), group_size=c.group_size,
                                          bits=c.bits) for x in (c.keys, c.values))
                else:
                    k, v = c.keys[..., :c.offset, :], c.values[..., :c.offset, :]
                out.append((k.astype(mx.float32), v.astype(mx.float32), hasattr(c, "bits")))
            mx.eval(out)
            got[rid] = out
        return fin(row, why)
    sched._finish = spy
    return got


@pytest.mark.parametrize("start", [0, 12, 26, 27, 29, 33])
def test_dflash_quantized_equals_plain(dense, monkeypatch, start):
    """DFlash over a q8 cache: the plain answer, and the cache it ends with
    holds the plain loop's KV (to the verify-vs-decode rounding of the fp32
    CPU model, far below what one position attending the other KV format
    would move): quantized at the same point (a prefill slab past the start,
    or the step at offset == start), no block crossing the start while the
    cache is fp16 (cut there; a lone token at start - 1: one plain step)."""
    no_metal_verify(monkeypatch)
    p = prompts([25], 30)[0]
    sp = scheduler(dense, max_batch=1, kv_bits=8, kv_start=start)
    ref = final_kv(sp, "a")
    plain = run(sp, [request(p, 30, "a")])["a"]
    sched = scheduler(dense, max_batch=1, kv_bits=8, kv_start=start)
    d = sched.eng.draft = Blocks(plain[0])
    got = final_kv(sched, "a")
    assert run(sched, [request(p, 30, "a")])["a"] == plain
    assert len(d.accept_lens) > 3
    if 25 < start:                                      # the start falls inside the decode
        # cut there: a shorter block, or (one fp16 position left) a plain step
        # that records a round without a draft_block call
        assert min(d.sizes) < 4 or len(d.accept_lens) > len(d.sizes) or start - 25 <= 1
    assert len(got["a"]) == len(ref["a"])
    for (k, v, q), (k0, v0, q0) in zip(got["a"], ref["a"]):
        # (a "length" row of the plain loop fed its last token too: one more)
        m = min(k.shape[2], k0.shape[2])
        assert q and q0 and m >= len(p) + 28
        k, v, k0, v0 = (x[:, :, :m] for x in (k, v, k0, v0))
        assert mx.abs(k - k0).max().item() < 1e-4 and mx.abs(v - v0).max().item() < 1e-4


def test_dflash_quantized_thinking_budget(dense, monkeypatch):
    """A thinking budget inside DFlash rounds (plain close steps) over a cache
    that is quantized on the way: the plain loop's tokens."""
    from mlx_vlm.utils import ThinkingBudgetCriteria
    from conftest import FakeTok
    no_metal_verify(monkeypatch)
    NL, CLOSE, OPEN = 11, 12, 13
    tok = FakeTok({"</think>": [CLOSE], "<think>": [OPEN], "\n": [NL]})
    p = prompts([25], 30)[0] + [OPEN]

    def crit(budget):
        return ThinkingBudgetCriteria(tokenizer=tok, thinking_budget=budget, thinking_end_token="</think>",
                                      thinking_start_token="<think>", enable_thinking=True,
                                      prompt_preopens_thinking=True)
    for start, budget in ((28, 3), (30, 4), (33, 9)):
        kw = dict(max_batch=1, kv_bits=8, kv_start=start)
        plain = run(scheduler(dense, **kw), [request(p, 24, "a", thinking=crit(budget))])["a"]
        sched = scheduler(dense, **kw)
        sched.eng.draft = FakeDraft(plain[0])
        assert run(sched, [request(p, 24, "a", thinking=crit(budget))])["a"] == plain, (start, budget)


def test_dflash_quantized_demoted(dense, monkeypatch):
    """A DFlash row over a q8 cache demoted to a batch row: both answers as
    alone."""
    no_metal_verify(monkeypatch)
    ps = prompts([25, 12], 30)
    kw = dict(kv_bits=8, kv_start=20)
    plain = run(scheduler(dense, max_batch=1, **kw), [request(ps[0], 30, "a")])["a"]
    other = run(scheduler(dense, max_batch=1, **kw), [request(ps[1], 20, "b")])["b"]
    sched = scheduler(dense, max_batch=4, **kw)
    sched.eng.draft = FakeDraft(plain[0])
    got = run(sched, [request(ps[0], 30, "a"), request(ps[1], 20, "b")], arrive={4: []})
    assert got["a"] == plain
    sched = scheduler(dense, max_batch=4, **kw)
    sched.eng.draft = FakeDraft(plain[0])
    a, b = request(ps[0], 30, "a"), request(ps[1], 20, "b")
    assert run(sched, [a, b], arrive={4: [b]}) == {"a": plain, "b": other}
    assert 0 < len(sched.eng.draft.accept_lens) < 10


def test_verify_attention_kernel_skips_quantized_caches(dense, monkeypatch):
    """QWEN_VERIFY_ATT (the fp16 vector kernel): a verify block over a
    QuantizedKVCache goes to the library's one-query quantized calls; over
    the fp16 warm-up it still takes the kernel."""
    from mlx_vlm.models.qwen3_5 import speculative_verifier as sv
    no_metal_verify(monkeypatch)
    v = sv.Qwen3_5BatchInvariantForward
    lib, kinds, blocks = v._attention, [], []
    monkeypatch.setattr(worker, "_VERIFY_ATTENTION",
                        lambda *a: kinds.append(type(a[4]).__name__) or lib(*a))
    monkeypatch.setattr(v, "_attention", worker._verify_attention)
    monkeypatch.setattr(worker, "_VERIFY_ATT_Q", 4)

    def unfused(q, k, v, scale):
        blocks.append(q.shape[2])
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
    monkeypatch.setattr(worker, "_block_sdpa", unfused)
    p = prompts([25], 31)[0]
    plain = run(scheduler(dense, max_batch=1, kv_bits=8, kv_start=0), [request(p, 20, "a")])["a"]
    sched = scheduler(dense, max_batch=1, kv_bits=8, kv_start=0)
    sched.eng.draft = FakeDraft(plain[0])
    assert run(sched, [request(p, 20, "a")])["a"] == plain
    assert blocks == [] and kinds and set(kinds) == {"QuantizedKVCache"}
    kinds.clear()
    sched = scheduler(dense, max_batch=1, kv_bits=8, kv_start=40)       # fp16 up to 40
    plain = run(scheduler(dense, max_batch=1, kv_bits=8, kv_start=40), [request(p, 30, "a")])["a"]
    sched.eng.draft = FakeDraft(plain[0])
    assert run(sched, [request(p, 30, "a")])["a"] == plain
    assert blocks and set(kinds) <= {"QuantizedKVCache"} and kinds


# --- the dequantized fused prefill (QWEN_KVQ_PREFILL_FUSED) ----------------------------
def _qkv(q_len, kv_len, bits=8, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((1, 4, q_len, 64)).astype(mx.float16)
    k = (2 * mx.random.normal((1, 2, kv_len, 64))).astype(mx.float16)
    v = mx.random.normal((1, 2, kv_len, 64)).astype(mx.float16)
    return q, mx.quantize(k, group_size=64, bits=bits), mx.quantize(v, group_size=64, bits=bits)


@pytest.fixture
def fused_spy(monkeypatch):
    """mx.fast SDPA calls (force_fused?) with the CPU's fallback standing in
    for the fused kernel."""
    calls = []
    real = mx.fast.scaled_dot_product_attention

    def spy(*a, **kw):
        calls.append(kw.get("force_fused", False))
        kw.pop("force_fused", None)
        return real(*a, **kw)
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", spy)
    monkeypatch.setattr(worker, "_SDPA", base.scaled_dot_product_attention)
    monkeypatch.setattr(worker, "SDPA_FUSED", True)
    monkeypatch.setattr(worker, "SDPA_FUSED_MIN_Q", 8)
    monkeypatch.setattr(worker, "SDPA_FUSED_MIN_KV", 32)
    return calls


def test_kvq_fused_gate(fused_spy, monkeypatch):
    """Knob on: a long slab over a QuantizedKVCache takes the fused kernel over
    mx.dequantize'd keys / values (exactly that call's result); short slabs,
    few keys, decode, an array mask, fp32 queries: the library's quantized
    SDPA. Knob off: always the library's."""
    calls = fused_spy
    qc = QuantizedKVCache(group_size=64, bits=8)
    q, kq, vq = _qkv(8, 40)
    monkeypatch.setattr(worker, "KVQ_PREFILL_FUSED", False)
    calls.clear()
    worker._fused_sdpa(q * 1, kq, vq, cache=qc, scale=0.125, mask="causal")
    assert calls == []
    monkeypatch.setattr(worker, "KVQ_PREFILL_FUSED", True)
    calls.clear()
    out = worker._fused_sdpa(q, kq, vq, cache=qc, scale=0.125, mask="causal")
    kd, vd = (mx.dequantize(*x, group_size=64, bits=8) for x in (kq, vq))
    want = mx.fast.scaled_dot_product_attention(q, kd, vd, scale=0.125, mask="causal")
    assert calls == [True, False] and kd.dtype == mx.float16
    assert mx.array_equal(out, want).item()
    for (ql, kl), mask, dt in (((7, 40), "causal", mx.float16), ((8, 31), "causal", mx.float16),
                               ((1, 40), None, mx.float16), ((8, 40), "array", mx.float16),
                               ((8, 40), "causal", mx.float32)):
        q, kq, vq = _qkv(ql, kl)
        m = mx.ones((ql, kl), mx.bool_) if mask == "array" else mask
        calls.clear()
        worker._fused_sdpa(q.astype(dt), kq, vq, cache=qc, scale=0.125, mask=m)
        assert calls == [], (ql, kl, mask, dt)
    # a fp16 KVCache is unchanged by the knob
    q, k, v = (mx.random.normal((1, n, 8 if i == 0 else 40, 64)).astype(mx.float16)
               for i, n in enumerate((4, 2, 2)))
    calls.clear()
    worker._fused_sdpa(q, k, v, cache=KVCache(), scale=0.125, mask="causal")
    assert calls == [True]


def test_kvq_fused_falls_back(monkeypatch):
    """No fused kernel: the library's quantized SDPA, the gate off once; a
    slab whose scores the budget does not allow fails instead."""
    real = mx.fast.scaled_dot_product_attention

    def nofused(*a, **kw):
        if kw.get("force_fused"):
            raise ValueError("no fused kernel")
        return real(*a, **kw)
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", nofused)
    monkeypatch.setattr(worker, "_SDPA", base.scaled_dot_product_attention)
    monkeypatch.setattr(worker, "SDPA_FUSED", True)
    monkeypatch.setattr(worker, "KVQ_PREFILL_FUSED", True)
    monkeypatch.setattr(worker, "SDPA_FUSED_MIN_Q", 8)
    monkeypatch.setattr(worker, "SDPA_FUSED_MIN_KV", 32)
    qc = QuantizedKVCache(group_size=64, bits=8)
    q, kq, vq = _qkv(8, 40)
    ref = base.scaled_dot_product_attention(q * 1, kq, vq, cache=qc, scale=0.125, mask="causal")
    out = worker._fused_sdpa(q * 1, kq, vq, cache=qc, scale=0.125, mask="causal")
    assert mx.array_equal(out, ref).item() and worker.SDPA_FUSED is False
    monkeypatch.setattr(worker, "PREFILL_BUDGET", 8 * 39)
    with pytest.raises(RuntimeError, match="QWEN_PREFILL_BUDGET"):
        worker._fused_sdpa(q * 1, kq, vq, cache=qc, scale=0.125, mask="causal")


@pytest.mark.parametrize("bits", [8, 4])
def test_kvq_fused_closer_to_the_dequantized_reference(fused_spy, monkeypatch, bits):
    """The reference: fp32 attention over the dequantized keys / values (what
    the stored cache means). The fused path (fp16 inputs, scores never
    rounded to fp16) is closer to it than the library's quantized SDPA
    (fp16 scores and probabilities between its two quantized matmuls), on
    max and mean error."""
    monkeypatch.setattr(worker, "KVQ_PREFILL_FUSED", True)
    qc = QuantizedKVCache(group_size=64, bits=bits)
    for i, (ql, kl) in enumerate(((16, 64), (64, 512), (128, 2048))):
        q, kq, vq = _qkv(ql, kl, bits, seed=i)
        kd, vd = (mx.dequantize(*x, group_size=64, bits=bits).astype(mx.float32) for x in (kq, vq))
        ref = mx.fast.scaled_dot_product_attention(q.astype(mx.float32), kd, vd, scale=0.125, mask="causal")
        new = worker._fused_sdpa(q * 1, kq, vq, cache=qc, scale=0.125, mask="causal")
        # (the library scales its queries argument in place: a copy)
        old = base.scaled_dot_product_attention(q * 1, kq, vq, cache=qc, scale=0.125, mask="causal")
        en, eo = (mx.abs(x.astype(mx.float32) - ref) for x in (new, old))
        assert en.max().item() < eo.max().item() and en.mean().item() < eo.mean().item(), (ql, kl)


def test_kvq_fused_prefill_runs_and_hits_equal_cold(dense_fused, small, fused_spy, monkeypatch):
    """A whole q8 prefill with the knob on: long slabs past the threshold take
    the dequantized fused path, the answer is the plain path's to the
    attention rounding (greedy, short), and a prompt-cache hit is still a cold
    prefill bit for bit (the slab policy is fixed: _bounds determinism)."""
    calls = fused_spy
    monkeypatch.setattr(worker, "_FIXED_SLAB", None)
    monkeypatch.setattr(worker, "FP16", True)
    monkeypatch.setattr(worker, "SDPA_FUSED_MIN_Q", 4)
    monkeypatch.setattr(worker, "SDPA_FUSED_MIN_KV", 16)
    monkeypatch.setattr(worker, "PREFILL_BUDGET", 8 * 16)
    from mlx_vlm.models.qwen3_5 import language as q35
    monkeypatch.setattr(q35, "scaled_dot_product_attention", worker._fused_sdpa)
    dense = dense_fused                                         # fp16 queries
    monkeypatch.setattr(worker, "PROMPT_CACHE_REPLY", False)    # (Cold: prompt entries)
    kw = dict(kv_bits=8, kv_start=0)
    ids = rk.first(380, system=40, user=10)
    monkeypatch.setattr(worker, "KVQ_PREFILL_FUSED", False)
    assert worker._step(100) < 8                                # shrunk: the quantized SDPA's scores
    off = run(scheduler(dense, max_batch=1, **kw), [request(ids, 6, "a")])["a"]
    assert not any(calls)
    monkeypatch.setattr(worker, "KVQ_PREFILL_FUSED", True)
    assert worker._step(100) == 8                               # fixed past the threshold
    calls.clear()
    sched = scheduler(dense, max_batch=1, **kw)
    on = run(sched, [request(ids, 6, "a")])["a"]
    assert sum(calls) >= 2 and on[2] == off[2]
    toks = on[0]
    nxt = rk.nxt(ids, toks, 381, user=20)
    run(sched, [request(nxt, 4, "b")])
    assert check_cold(sched.eng._pcache, Cold(dense, **kw)) >= 3


# --- accounting -----------------------------------------------------------------------------
def test_fixed_slab_and_reserve(monkeypatch):
    monkeypatch.setattr(worker, "_FIXED_SLAB", None)
    monkeypatch.setattr(worker, "PREFILL_STEP", 2048)
    monkeypatch.setattr(worker, "PREFILL_BUDGET", 2048 * 32768)
    monkeypatch.setattr(worker, "PREFILL_RESERVE_GB", 0)
    monkeypatch.setattr(worker, "FP16", True)
    monkeypatch.setattr(worker, "SDPA_FUSED", True)
    monkeypatch.setattr(worker, "MAX_CONTEXT", 131072)
    monkeypatch.setattr(worker, "KV_KWARGS", dict(worker.KV_KWARGS, kv_bits=8))
    heads, kvh, d = 24, 4, 256                                     # the 27B
    base_ = 2048 * 32768 * heads * 2 + (400 << 20)
    monkeypatch.setattr(worker, "KVQ_PREFILL_FUSED", False)
    assert worker._step(122880) == 546 and worker._prefill_reserve(heads, kvh, d) == base_
    monkeypatch.setattr(worker, "KVQ_PREFILL_FUSED", True)
    assert worker._step(122880) == 2048 and worker._step(81920) == 2048
    # one layer's K + V at 128K (512 MiB) x 1.6, the measured transient
    assert worker._prefill_reserve(heads, kvh, d) == base_ + (512 << 20) * 8 // 5
    monkeypatch.setattr(worker, "SDPA_FUSED", False)                         # no fused kernel: as before
    assert worker._step(122880) == 546 and worker._prefill_reserve(heads, kvh, d) == base_
    monkeypatch.setattr(worker, "SDPA_FUSED", True)
    monkeypatch.setattr(worker, "KV_KWARGS", dict(worker.KV_KWARGS, kv_bits=None))
    assert worker._prefill_reserve(heads, kvh, d) == base_                   # fp16: nothing to dequantize


def test_kv_token_bytes_27b_layout():
    """16 full-attention layers x 4 KV heads x head_dim 256 (K and V): fp16
    64 KiB/token, q8 34 KiB (1 B + an fp16 scale and bias per 64 values),
    q4 18 KiB."""
    import types
    layers = [types.SimpleNamespace(is_linear=i % 4 != 3) for i in range(64)]
    lm = types.SimpleNamespace(model=types.SimpleNamespace(layers=layers),
                               args=types.SimpleNamespace(num_key_value_heads=4, head_dim=256))
    assert worker._kv_token_bytes(lm, 0) == 64 << 10
    assert worker._kv_token_bytes(lm, 8) == 34 << 10
    assert worker._kv_token_bytes(lm, 4) == 18 << 10


def test_reservation_drops_the_fp16_copy(dense, small, monkeypatch):
    """Past kv_start a row reserves its quantized KV + the fp16 copy of the
    tokens up to the crossing slab (the peak while its cache is quantized),
    and only the quantized KV once it is: after the crossing slab, or from
    the lookup for a row restored from a quantized entry. The accounting
    ends at 0."""
    s = scheduler(dense, max_batch=1, kv_bits=8, kv_start=40)
    fp, q = s.kv_fp16, s.kv_q
    assert q == fp // 2 + fp // 32                                  # (1 + 4/64) B per value vs 2
    n = 100
    assert s._bytes(n) - s._bytes(n, True) == int((n * q + 48 * fp) * 1.05) - int(n * q * 1.05)
    assert s._bytes(30) == s._bytes(30, True)                       # never quantized
    seen = []

    def watch(t, sched):
        r = sched.task_row or (sched.rows[0] if sched.rows else None)
        if r is not None:
            seen.append((r.req.id, r.kvq, r.reserved == sched._bytes(r.tokens, r.kvq),
                         r.cache is not None and hasattr(r.cache[3], "bits")))
    ids = rk.first(390, system=40, user=10)                         # 58 tokens: crosses at 40
    toks = run(s, [request(ids, 5, "a")], on_tick=watch)["a"][0]
    assert s.reserved == 0
    a = [x for x in seen if x[0] == "a"]
    assert all(ok for _, _, ok, _ in a) and a[0][1] is False and a[-1][1] is True
    assert all(kvq == quant for _, kvq, _, quant in a)
    seen.clear()
    marks = []
    mark = s._kvq_mark
    s._kvq_mark = lambda row, known=False: (marks.append(known), mark(row, known))
    run(s, [request(rk.nxt(ids, toks, 391), 5, "b")], on_tick=watch)
    assert marks[0] is True                                         # at the lookup: a quantized entry
    assert all(kvq for _, kvq, _, _ in seen) and s.reserved == 0


def test_step_bytes_quantized(dense):
    s = scheduler(dense, max_batch=4, kv_bits=8, kv_start=40)
    import types
    rows = [types.SimpleNamespace(pos=30), types.SimpleNamespace(pos=100)]
    assert s._step_bytes(rows) == s.w_bytes + 30 * s.kv_fp16 + 100 * s.kv_q + 4 * s.gdn_row
