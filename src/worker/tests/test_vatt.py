"""vatt (one-pass verify-block attention, QWEN_VERIFY_ATT=2): the Metal kernel
against an fp32 reference and MLX's own per-query SDPA (the library's verify
calls), on the 27B / 9B head layouts (G 6 / 4, head_dim 256), block sizes 1..16
(one or two query blocks), lengths around the chunk and key-block boundaries
(the causal tail inside one key block, across two, across a chunk boundary),
buffers with spare capacity and garbage past n (1000 in K, NaN in V: never
read), a strided view (heads not adjacent: the token / head strides are read
from the array, nothing is copied), flat and peaky scores, forced chunks.
Runs only with VATT_GPU=1 (it takes the GPU):

    VATT_GPU=1 .venv/bin/python -m pytest -q tests/test_vatt.py

conftest makes the CPU the default device; these tests switch to the GPU and
back. The CPU fallback (`reference`) runs in the CPU suite (last tests)."""
import os

import mlx.core as mx
import pytest

import vatt

GPU = os.environ.get("VATT_GPU") == "1" and mx.metal.is_available()
gpu_only = pytest.mark.skipif(not GPU, reason="VATT_GPU=1 and Metal")
D = 256


@pytest.fixture
def gpu():
    prev = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(prev)


def kv(n, hkv, spare=37, seed=0, strided=False):
    mx.random.seed(seed)
    cap = n + spare
    shape = (1, 2 * hkv if strided else hkv, cap, D)
    k = (mx.random.normal(shape) * 0.5).astype(mx.float16)
    v = (mx.random.normal(shape) * 0.5).astype(mx.float16)
    k[..., n:, :] = 1000.0          # garbage past n: must never be read
    v[..., n:, :] = float("nan")
    if strided:                     # every other head of a bigger buffer: a non-contiguous view
        k, v = k[:, ::2], v[:, ::2]
    mx.eval(k, v)
    return k, v


def rows_mlx(q, k, v, n, scale):
    """The library's verify attention: one SDPA call per query over its keys."""
    T = q.shape[2]
    return mx.concatenate([mx.fast.scaled_dot_product_attention(
        q[:, :, i:i + 1], k[:, :, :n - T + i + 1], v[:, :, :n - T + i + 1], scale=scale, mask=None)
        for i in range(T)], axis=2)


def rel_rms(a, ref):
    d = a.astype(mx.float32) - ref
    return (mx.sqrt(mx.mean(d * d)) / mx.sqrt(mx.mean(ref * ref))).item()


CASES = [  # (n, T, hq, hkv)
    (1, 1, 24, 4), (8, 8, 24, 4), (9, 8, 24, 4), (100, 8, 24, 4), (700, 8, 24, 4),
    (2049, 8, 24, 4), (4096 + 3, 8, 24, 4), (12345, 8, 24, 4), (33000, 8, 24, 4),
    (5000, 1, 24, 4), (5000, 2, 24, 4), (5000, 3, 24, 4), (5000, 5, 24, 4), (5000, 7, 24, 4),
    (5000, 9, 24, 4), (5000, 16, 24, 4), (3000, 8, 16, 4), (3000, 16, 16, 4), (3000, 4, 16, 4),
    (3000, 8, 16, 2), (60000, 8, 24, 4),
]


@gpu_only
@pytest.mark.parametrize("n,T,hq,hkv", CASES)
@pytest.mark.parametrize("mul", [1.0, 8.0])
def test_matches_reference(gpu, n, T, hq, hkv, mul):
    k, v = kv(n, hkv, seed=n % 7)
    mx.random.seed(T)
    q = (mx.random.normal((1, hq, T, D)) * mul).astype(mx.float16)
    scale = D ** -0.5
    out = vatt.attention(q, k, v, n, scale)
    ref = vatt.reference(q, k, v, n, scale)
    own = rows_mlx(q, k, v, n, scale)
    mx.eval(out, ref, own)
    assert out.shape == q.shape and out.dtype == mx.float16
    assert not mx.any(mx.isnan(out)).item()
    e, e_mlx = rel_rms(out, ref), rel_rms(own, ref)
    # fp32 throughout: at most fp16 output rounding, never worse than MLX's own kernel
    assert e <= max(1.05 * e_mlx, 4e-4), (e, e_mlx)


@gpu_only
@pytest.mark.parametrize("n,chunk", [(3000, 48), (3000, 96), (1000, 960), (4100, 4080), (517, 2016)])
def test_chunks_and_tails(gpu, n, chunk):
    """Forced chunks: the tail of the block (last 8 keys) crossing a chunk end
    and a key-block end; a chunk fully past some rows' limits (weight 0)."""
    k, v = kv(n, 4, seed=1)
    q = mx.random.normal((1, 24, 8, D)).astype(mx.float16)
    for nn in (n, n - 3, n - 5):
        out = vatt.attention(q, k, v, nn, D ** -0.5, chunk=chunk)
        assert rel_rms(out, vatt.reference(q, k, v, nn, D ** -0.5)) < 4e-4, (nn, chunk)


@gpu_only
def test_strided_view_no_copy(gpu):
    """A view (every other head): strides are read from the array."""
    k, v = kv(3000, 4, seed=2, strided=True)
    q = mx.random.normal((1, 24, 8, D)).astype(mx.float16)
    out = vatt.attention(q, k, v, 2990, D ** -0.5)
    assert rel_rms(out, vatt.reference(q, k, v, 2990, D ** -0.5)) < 4e-4


@gpu_only
def test_transposed_queries(gpu):
    """The verify path hands over (1, Hq, T, D) as a transposed view of (1, T, Hq, D)."""
    k, v = kv(1500, 4, seed=3)
    q = mx.random.normal((1, 8, 24, D)).astype(mx.float16).transpose(0, 2, 1, 3)
    out = vatt.attention(q, k, v, 1500, D ** -0.5)
    assert rel_rms(out, vatt.reference(q, k, v, 1500, D ** -0.5)) < 4e-4


def test_rejects_bad_shapes():
    k = mx.zeros((1, 4, 64, D), mx.float16)
    for q, n in ((mx.zeros((1, 24, 17, D), mx.float16), 40),     # T > MAX_T
                 (mx.zeros((1, 24, 8, D), mx.float32), 40),
                 (mx.zeros((1, 24, 8, D), mx.float16), 65),      # n > cap
                 (mx.zeros((1, 24, 8, D), mx.float16), 7),       # n < T
                 (mx.zeros((1, 24, 8, 128), mx.float16), 40),
                 (mx.zeros((2, 24, 8, D), mx.float16), 40)):
        with pytest.raises(ValueError):
            vatt.attention(q, k, k, n, 0.0625)


def test_cpu_reference_is_causal_tail():
    """The CPU fallback = per-query softmax over [0, n - T + t + 1) (fp32)."""
    mx.random.seed(0)
    q = mx.random.normal((1, 4, 3, D)).astype(mx.float16)
    k = mx.random.normal((1, 2, 20, D)).astype(mx.float16)
    v = mx.random.normal((1, 2, 20, D)).astype(mx.float16)
    out = vatt.attention(q, k, v, 15, 0.0625)
    for t in range(3):
        lim = 15 - 3 + t + 1
        for hq in range(4):
            s = (q[0, hq, t].astype(mx.float32) * 0.0625) @ k[0, hq // 2, :lim].astype(mx.float32).T
            want = mx.softmax(s, precise=True) @ v[0, hq // 2, :lim].astype(mx.float32)
            assert mx.abs(out[0, hq, t].astype(mx.float32) - want).max().item() < 2e-3
