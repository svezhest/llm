"""QWEN_SAMPLER_TOPK_FAST (_Sampler.survivors / _TopK): min_p / top_p and the
draw over the top_k survivors. Against filtered() (the vLLM-order path) on a
small vocabulary: the full-vocab distribution it implies equals
softmax(filtered(x)) — the same support, probabilities to fp32 rounding —
with ties of the k-th value (inside and outside the k, cut by top_p),
min_p / top_p boundaries, -inf rows (grammar masks) and fewer finite logits
than k; seeded draws are reproducible and their frequencies match; the knob
off leaves the old path (and its random stream) bit for bit; sample_rows'
grouped draw == each row alone. CPU (the Metal check at the real vocabulary
is tests/test_gpu_verify_sampler.py, VERIFY_GPU=1)."""
import mlx.core as mx
import numpy as np
import pytest

import worker

CARD = dict(temperature=0.6, top_p=0.95, top_k=20)


def sampler(monkeypatch, fast=True, seed=0, **kw):
    monkeypatch.setattr(worker, "SAMPLER_TOPK_FAST", fast)
    return worker.make_sampler(dict(worker.SAMPLING_KWARGS, **kw), seed=seed)


def full_probs(t, V):
    """The full-vocab distribution a _TopK row draws from."""
    z = np.array(t.z, dtype=np.float64)
    ids, order = np.array(t.ids), np.array(t.order)
    start, extra = np.array(t.start), np.array(t.extra)
    out = np.zeros((z.shape[0], V))
    for r in range(z.shape[0]):
        m = np.max(z[r])
        e = np.exp(z[r] - m) if np.isfinite(m) else np.zeros_like(z[r])
        p = e / e.sum()
        k = ids.shape[-1]
        out[r, ids[r]] += p[:k]
        n = int(extra[r, 0])
        if n:
            s = int(start[r, 0])
            out[r, order[r, s:s + n]] += p[k] / n
    return out


def ref_probs(s, x):
    f = np.array(s.filtered(x), dtype=np.float64)
    m = f.max(axis=-1, keepdims=True)
    e = np.exp(f - m)
    return e / e.sum(axis=-1, keepdims=True)


def boundary(s, x, eps=1e-5):
    """Tokens exactly at a min_p / top_p boundary (float64, the vLLM order in
    exact arithmetic, ties by index): filtered() and survivors() round
    their sums differently there, so either side is right."""
    x = np.array(x.astype(mx.float32), dtype=np.float64) / s.temp
    out = np.zeros(x.shape, dtype=bool)
    for r in range(x.shape[0]):
        v = x[r].copy()
        if not np.isfinite(v).any():
            continue
        rel = np.exp(v - v.max())
        if s.min_p > 0:
            out[r] |= np.abs(rel - s.min_p) <= eps * s.min_p
            v[rel < s.min_p] = -np.inf
        if 0 < s.top_k < len(v):
            v[v < np.sort(v)[-s.top_k]] = -np.inf
        if s.top_p < 1.0:
            o = np.argsort(v, kind="stable")
            e = np.exp(v[o] - v.max())
            c = np.cumsum(e) / e.sum()
            out[r, o] |= np.abs(c - (1 - s.top_p)) <= eps
    return out


def check_same(s, x, tol=2e-6):
    V = x.shape[-1]
    t = s.survivors(x)
    got, want = full_probs(t, V), ref_probs(s, x)
    amb = boundary(s, x)
    for r in range(x.shape[0]):
        if amb[r].any():                                   # either side of the boundary
            ok = amb[r] | ((got[r] > 0) == (want[r] > 0))
            assert ok.all(), (r, np.nonzero(~ok))
            continue
        np.testing.assert_array_equal(got[r] > 0, want[r] > 0)      # the same support
        np.testing.assert_allclose(got[r], want[r], rtol=1e-5, atol=tol)
    return got


CFGS = [dict(temperature=0.6, top_p=0.95, top_k=20), dict(temperature=1.0, top_k=50),
        dict(temperature=0.8, top_k=20, min_p=0.05), dict(temperature=0.7, top_k=5, top_p=0.5),
        dict(temperature=1.3, top_k=100, top_p=0.9, min_p=0.01), dict(temperature=1.0, top_k=1),
        dict(temperature=0.5, top_k=511, top_p=0.99)]


@pytest.mark.parametrize("cfg", CFGS)
@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_random_logits(monkeypatch, cfg, dtype):
    s = sampler(monkeypatch, **cfg)
    for seed in range(4):
        x = (mx.random.normal((6, 512), key=mx.random.key(seed)) * 3).astype(dtype)
        check_same(s, x)


@pytest.mark.parametrize("cfg", CFGS)
def test_ties(monkeypatch, cfg):
    """Ties at the k-th value outside the k (filtered keeps them all), a tie
    block cut by top_p, ties with the maximum, a flat fp16 plateau."""
    s = sampler(monkeypatch, **cfg)
    k = cfg["top_k"]
    x = (mx.random.normal((8, 512), key=mx.random.key(7)) * 2).astype(mx.float32)
    x = np.array(x)
    srt = np.sort(x, axis=-1)[:, ::-1]
    for r, n in enumerate([3, 40, 200]):                   # n tokens tied at the k-th value
        x[r, 300:300 + n] = srt[r, k - 1]
    x[3, 10:60] = srt[3, 0] + 1                            # 50 tied maxima
    x[4, :] = 0.0                                          # flat: every token ties
    x[5, 100:164] = 9.0                                    # 64 equal top logits
    x[6, 5:30] = x[6, 5]                                   # a tie block around the cut
    x[6, 400:420] = x[6, 5]
    xm = mx.array(x).astype(mx.float16)
    check_same(s, xm)
    check_same(s, mx.array(x))


def test_top_p_cuts_a_tie_group(monkeypatch):
    """top_p drops part of the k-th tie group, inside and outside the k:
    the lowest vocab indices first, as filtered()'s stable argsort."""
    s = sampler(monkeypatch, temperature=1.0, top_k=4, top_p=0.6)
    x = np.full((1, 64), -30.0, dtype=np.float32)
    x[0, 50] = 3.0                                         # the top token
    x[0, [5, 9, 13, 40, 41, 60]] = 1.0                    # 6 tied at the 4th value
    got = check_same(s, mx.array(x))
    want = ref_probs(s, mx.array(x))
    kept = sorted(np.nonzero(want[0])[0].tolist())
    assert kept == [50, 60]          # 5 of the 6 ties go: the 3 outside the k, then 40, 41
    assert np.array_equal(got[0] > 0, want[0] > 0)


def test_masked_rows(monkeypatch):
    """-inf logits (grammar / allowed_token_ids masks), fewer finite than k."""
    for cfg in CFGS:
        s = sampler(monkeypatch, **cfg)
        x = np.array(mx.random.normal((4, 512), key=mx.random.key(1)) * 3)
        x[0, 10:] = -np.inf                               # 10 finite < every k but 1/5
        x[1, ::2] = -np.inf
        x[2, :500] = -np.inf
        x[3, 3:] = -np.inf
        x[3, :3] = 2.0                                    # 3 tied finite
        check_same(s, mx.array(x))


def test_draw_frequencies_and_seed(monkeypatch):
    """Seeded draws reproduce; frequencies over many rows follow the
    distribution (bucket of outside ties included)."""
    cfg = dict(temperature=1.0, top_k=3, top_p=0.97)
    x = np.full((1, 32), -5.0, dtype=np.float32)
    x[0, 7] = 1.0
    x[0, [2, 20, 25, 30]] = 0.0                          # top 3 = 7 + two of four tied
    s = sampler(monkeypatch, seed=5, **cfg)
    want = ref_probs(s, mx.array(x))[0]
    rows = mx.broadcast_to(mx.array(x), (40000, 32))
    a = np.array(s(rows))
    b = np.array(sampler(monkeypatch, seed=5, **cfg)(rows))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, np.array(sampler(monkeypatch, seed=6, **cfg)(rows)))
    freq = np.bincount(a, minlength=32) / len(a)
    assert set(np.nonzero(freq)[0]) <= set(np.nonzero(want)[0])
    np.testing.assert_allclose(freq, want, atol=0.012)
    # the outside ties are drawn too (filtered() keeps all four)
    assert all(freq[i] > 0.05 for i in (2, 20, 25, 30))


def test_off_is_the_old_path(monkeypatch):
    """Knob off: prepare() is filtered(), the draw the full-vocab categorical
    with the same key stream — the tokens equal the pre-knob sampler's."""
    x = (mx.random.normal((3, 512), key=mx.random.key(2)) * 3).astype(mx.float16)
    s = sampler(monkeypatch, fast=False, seed=9, **CARD)
    assert not s.fast
    ref_key = mx.random.key(9)
    out = []
    for _ in range(4):
        ref_key, sub = mx.random.split(ref_key)
        want = mx.random.categorical(s.filtered(x), key=sub)
        out.append((s(x).tolist(), want.tolist()))
    assert all(a == b for a, b in out)
    assert isinstance(s.prepare(x), mx.array)


def test_fast_only_with_top_k(monkeypatch):
    s = sampler(monkeypatch, temperature=0.7, top_p=0.9)          # top_k 0
    assert not s.fast
    s = sampler(monkeypatch, temperature=0.7, top_k=600)          # >= V: filtered() too
    x = mx.random.normal((1, 512), key=mx.random.key(0))
    assert s.fast and isinstance(s.prepare(x), mx.array)
    g = sampler(monkeypatch, temperature=0.0, top_k=20)
    assert g.greedy and g(x).tolist() == mx.argmax(x, axis=-1).tolist()


def test_sample_rows_fast_equals_solo(monkeypatch):
    """The grouped draw (one survivors() over the stack, per-row keys) ==
    each row's sampler alone, over steps, mixed settings."""
    monkeypatch.setattr(worker, "SAMPLER_TOPK_FAST", True)
    cfgs = [dict(temperature=0.0), CARD, dict(temperature=0.7, top_p=0.9),
            dict(temperature=0.8, top_k=10, min_p=0.05)]
    rows = [dict(worker.SAMPLING_KWARGS, **cfgs[i % 4]) for i in range(10)]
    a = [worker.make_sampler(c, seed=3 * i) for i, c in enumerate(rows)]
    b = [worker.make_sampler(c, seed=3 * i) for i, c in enumerate(rows)]
    for step in range(5):
        lg = (mx.random.normal((10, 512), key=mx.random.key(step)) * 3).astype(mx.float16)
        lg[:, 11] = lg[:, 12]                                     # some ties
        xs = [lg[i:i + 1] for i in range(10)]
        want = [a[i](xs[i]).tolist() for i in range(10)]
        assert [y.tolist() for y in worker.sample_rows(b, xs)] == want
