"""Batched decode plumbing at the logits level (tiny random models, CPU):
RowsKV + stacked recurrent state + explicit positions give every row the
logits it gets decoding alone; rows join, leave and drop back to the plain
B=1 call; the memory formulas match the real caches."""
import mlx.core as mx
import pytest
from mlx_vlm.generate.common import maybe_quantize_kv_cache
from mlx_vlm.models.cache import KVCache

import worker
from conftest import build_lm, prompts


def prefill(lm, ids, kv_bits):
    lm._position_ids = None
    lm._rope_deltas = None
    cache = lm.make_cache()
    if len(ids) > 1:                       # a 1-token prompt leaves every cache empty
        lm(mx.array([ids[:-1]]), cache=cache, skip_logits=True)
        if kv_bits:
            for i, c in enumerate(cache):
                if isinstance(c, KVCache):
                    cache[i] = c.to_quantized(group_size=64, bits=kv_bits)
        mx.eval([c.state for c in cache])
    return cache


def solo(lm, ids, cache, toks, delta=0, quantize=None):
    """Decode `toks` one by one on the row's own caches -> logits per step."""
    out, pos = [], len(ids) - 1
    for t in toks:
        lg = lm(mx.array([[t]]), cache=cache,
                position_ids=mx.array([[pos + delta]], dtype=mx.int32)).logits[:, -1]
        if quantize:
            quantize(cache)
        mx.eval(lg)
        out.append(lg[0])
        pos += 1
    return out


def batched(lm, caches, gdn, toks, pos):
    layers = lm.model.layers
    cache = [gdn[i] if i in gdn else worker.RowsKV([c[i] for c in caches])
             for i in range(len(layers))]
    lg = lm(mx.array(toks)[:, None], cache=cache,
            position_ids=mx.array([[p] for p in pos], dtype=mx.int32)).logits[:, -1, :]
    mx.eval(lg)
    return lg


CONFIGS = [("dense", None, None, mx.float32, 1e-4), ("dense", 8, 8, mx.float32, 1e-2),
           ("dense", 4, 4, mx.float32, 1e-2), ("dense", 8, None, mx.float16, 2e-2),
           ("moe", None, None, mx.float32, 1e-4), ("moe", 8, 8, mx.float32, 1e-2),
           ("moe", 4, 4, mx.float32, 1e-2), ("moe", 8, None, mx.float16, 2e-2)]


@pytest.mark.parametrize("kind,wbits,kvbits,dtype,tol", CONFIGS)
def test_rows_equal_solo(kind, wbits, kvbits, dtype, tol):
    lm = build_lm(kind, wbits, dtype)
    ids = prompts([1, 13, 21], 2)                  # a 1-token prompt: no recurrent state yet
    deltas = [0, 5, 0]                             # row 1: a vision-like mRoPE delta
    steps = 7
    quant = None
    if kvbits:                                     # crosses the start mid-batch
        quant = lambda c: maybe_quantize_kv_cache(c, 16, 64, kvbits)  # noqa: E731
    feed = [[(i * 37 + 11 * s) % 400 for s in range(steps)] for i in range(3)]
    ref = [solo(lm, ids[i], prefill(lm, ids[i], None), feed[i], deltas[i], quant)
           for i in range(3)]
    caches = [prefill(lm, ids[i], None) for i in range(3)]
    dt = lm.model.embed_tokens(mx.array([[0]])).dtype
    at = [0, 0, 0]
    worst, agree, total = 0.0, 0, 0

    def go(rows, gdn):
        nonlocal worst, agree, total
        lg = batched(lm, [caches[r] for r in rows], gdn, [feed[r][at[r]] for r in rows],
                     [len(ids[r]) - 1 + at[r] + deltas[r] for r in rows])
        for i, r in enumerate(rows):
            if quant:
                quant(caches[r])
            x, y = lg[i].astype(mx.float32), ref[r][at[r]].astype(mx.float32)
            worst = max(worst, float(mx.abs(x - y).max().item()))
            agree += int(mx.argmax(x).item() == mx.argmax(y).item())
            total += 1
            at[r] += 1

    rows = [0, 1]
    gdn = worker._gdn_stack(lm, [caches[r] for r in rows], dt)
    for _ in range(2):
        go(rows, gdn)
    gdn = worker._gdn_concat(gdn, worker._gdn_stack(lm, [caches[2]], dt))       # row 2 joins
    rows = [0, 1, 2]
    for _ in range(2):
        go(rows, gdn)
    gdn, rows = worker._gdn_take(gdn, [1, 2]), [1, 2]                           # row 0 leaves
    for _ in range(2):
        go(rows, gdn)
    worker._gdn_split(gdn, rows.index(1), caches[1])                            # row 1 alone
    lg = lm(mx.array([[feed[1][at[1]]]]), cache=caches[1],
            position_ids=mx.array([[len(ids[1]) - 1 + at[1] + deltas[1]]], dtype=mx.int32))
    back = float(mx.abs(lg.logits[0, -1].astype(mx.float32)
                        - ref[1][at[1]].astype(mx.float32)).max().item())
    assert worst <= tol and back <= tol, (worst, back)
    assert agree == total
    if kvbits:
        assert any(hasattr(c, "bits") for c in caches[2])


def test_explicit_positions_bit_identical():
    """B=1 with explicit positions == the positions LanguageModel computes."""
    for kind in ("dense", "moe"):
        lm = build_lm(kind, 8, mx.float16)
        ids = prompts([13], 3)[0]
        a = prefill(lm, ids, None)
        b = prefill(lm, ids, None)
        lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int32)
        lm._position_ids = None
        auto = lm(mx.array([[ids[-1]]]), cache=a).logits
        expl = lm(mx.array([[ids[-1]]]), cache=b,
                  position_ids=mx.array([[len(ids) - 1]], dtype=mx.int32)).logits
        assert mx.array_equal(auto, expl).item()


def test_memory_formulas():
    lm = build_lm("dense", 8, mx.float16)
    n = 40
    cache = prefill(lm, prompts([n + 1], 4)[0], None)
    fa = [c for c in cache if isinstance(c, KVCache)]
    real = sum(c.keys[..., :c.offset, :].nbytes + c.values[..., :c.offset, :].nbytes for c in fa)
    assert worker._kv_token_bytes(lm, 0) * n == real
    q = [c.to_quantized(group_size=64, bits=8) for c in fa]
    real_q = sum(sum(x[..., :n, :].nbytes for x in (*c.keys, *c.values)) for c in q)
    assert worker._kv_token_bytes(lm, 8) * n == real_q
    gdn = sum(a.nbytes for c in cache if hasattr(c, "cache") for a in c.cache)
    assert worker._gdn_row_bytes(lm) == gdn

    from conftest import engine
    s = worker._Scheduler(engine(lm), max_batch=4, budget=0, cap=0, prefill_reserve=0,
                          kv={"kv_bits": 8, "kv_group_size": 64, "quantized_kv_start": 5000})
    r = worker._Req(id="x", p=None, ids=[0] * 4000, max_tokens=900, sampling={}, ctl=None)
    base = 2 * s.gdn_row + (8 << 20)
    assert s._reserve(r) == int(4900 * s.kv_fp16 * 1.05) + base          # under the start: fp16
    r.max_tokens = 2000                                                   # 6000 >= 5000
    # until its cache is quantized the row holds the fp16 KV of up to start +
    # a slab tokens beside their quantized copy; after (_kvq_mark) n x q only
    assert s._reserve(r) == int((min(6000, 5000 + worker.PREFILL_STEP) * s.kv_fp16
                                 + 6000 * s.kv_q) * 1.05) + base
    assert s._bytes(6000, kvq=True) == int(6000 * s.kv_q * 1.05) + base
    assert s._bytes(20000) == int(((5000 + worker.PREFILL_STEP) * s.kv_fp16 + 20000 * s.kv_q) * 1.05) + base
    r.max_tokens = 120000                                                 # no max_tokens: capped
    assert s._reserve(r) == s._bytes(4000 + worker.RESERVE_TOKENS)
