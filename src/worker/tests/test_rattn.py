"""rattn (ragged multi-row decode attention): the Metal kernel against an fp32
reference and MLX's own SDPA per row, on ragged lengths around the chunk
boundaries (1, chunk - 1, chunk, chunk + 1, ...), buffers with spare capacity
(cap > len: only [0, len) is attended), batches split over launches
(> MAX_ROWS rows), GQA 4 and 8, flat and peaky scores.
Runs only with RATTN_GPU=1 (it takes the GPU, like PFQMM_GPU=1 for test_pfqmm):

    RATTN_GPU=1 uv run --group dev pytest -q tests/test_rattn.py

conftest makes the CPU the default device; these tests switch to the GPU and back."""
import os

import mlx.core as mx
import pytest

import rattn

pytestmark = pytest.mark.skipif(
    not (os.environ.get("RATTN_GPU") == "1" and mx.metal.is_available()),
    reason="RATTN_GPU=1 and Metal")
D = 256


@pytest.fixture(autouse=True)
def gpu():
    prev = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(prev)


def rows(lens, hkv, spare=37, seed=0):
    mx.random.seed(seed)
    ks, vs = [], []
    for n in lens:
        cap = n + spare
        k = (mx.random.normal((1, hkv, cap, D)) * 0.5).astype(mx.float16)
        v = (mx.random.normal((1, hkv, cap, D)) * 0.5).astype(mx.float16)
        # garbage past len: must never be read
        k[..., n:, :] = 1000.0
        v[..., n:, :] = float("nan")
        ks.append(k)
        vs.append(v)
    mx.eval(ks, vs)
    return ks, vs


def reference(q, ks, vs, lens, scale):
    B, hq = q.shape[0], q.shape[1]
    out = []
    for i in range(B):
        hkv = ks[i].shape[1]
        qg = q[i].astype(mx.float32).reshape(hkv, hq // hkv, D)
        k = ks[i][0, :, :lens[i]].astype(mx.float32)
        v = vs[i][0, :, :lens[i]].astype(mx.float32)
        p = mx.softmax((qg * scale) @ k.transpose(0, 2, 1), axis=-1)
        out.append((p @ v).reshape(1, hq, 1, D))
    return mx.concatenate(out, axis=0)


def mlx_rows(q, ks, vs, lens, scale):
    return mx.concatenate([mx.fast.scaled_dot_product_attention(
        q[i:i + 1], ks[i][..., :lens[i], :], vs[i][..., :lens[i], :], scale=scale)
        for i in range(q.shape[0])], axis=0)


def rel_rms(a, ref):
    d = a.astype(mx.float32) - ref
    return (mx.sqrt(mx.mean(d * d)) / mx.sqrt(mx.mean(ref * ref))).item()


C = rattn.CHUNK
CASES = [
    ([1, 2, 3], 4),
    ([C - 1, C, C + 1, 3 * C + 5], 4),
    ([5000, 8192, 300, 12345, 1, 777, 2048, 4097], 4),
    ([1000 + 97 * i for i in range(11)], 4),           # 11 rows: two launches
    ([4096, 33, 9000, 1500, 2047, 513, 64, 100, 3000, 7, 8191, 4500, 250, 1024, 6000, 16], 4),
    ([600, 5000, 1], 8),                               # GQA 8 (the 35B's 16 / 2)
]


@pytest.mark.parametrize("lens,g", CASES)
@pytest.mark.parametrize("mul", [1.0, 8.0])
def test_matches_reference(lens, g, mul):
    hq = 16
    hkv = hq // g
    ks, vs = rows(lens, hkv)
    mx.random.seed(len(lens))
    q = (mx.random.normal((len(lens), hq, 1, D)) * mul).astype(mx.float16)
    scale = D ** -0.5
    out = rattn.attention(q, ks, vs, lens, scale)
    ref = reference(q, ks, vs, lens, scale)
    own = mlx_rows(q, ks, vs, lens, scale)
    mx.eval(out, ref, own)
    assert out.shape == q.shape and out.dtype == mx.float16
    assert not mx.any(mx.isnan(out)).item()
    e_rattn, e_mlx = rel_rms(out, ref), rel_rms(own, ref)
    # fp32 accumulation: at most fp16 output rounding, never worse than MLX's own kernel
    assert e_rattn <= max(1.05 * e_mlx, 4e-4), (e_rattn, e_mlx)


@pytest.mark.parametrize("chunk,ns", [(256, 4), (512, 2), (1024, 4), (64, 1)])
def test_configs_agree(chunk, ns):
    lens = [3000, 129, 70, 5000]
    ks, vs = rows(lens, 4, seed=3)
    q = mx.random.normal((4, 16, 1, D)).astype(mx.float16)
    ref = reference(q, ks, vs, lens, D ** -0.5)
    out = rattn.attention(q, ks, vs, lens, D ** -0.5, chunk=chunk, ns=ns)
    assert rel_rms(out, ref) < 4e-4


def test_rejects_bad_shapes():
    ks, vs = rows([10, 10], 4)
    with pytest.raises(ValueError):
        rattn.attention(mx.zeros((2, 16, 2, D), mx.float16), ks, vs, [10, 10], 0.0625)
    with pytest.raises(ValueError):
        rattn.attention(mx.zeros((2, 16, 1, D), mx.float32), ks, vs, [10, 10], 0.0625)
    with pytest.raises(ValueError):
        rattn.attention(mx.zeros((3, 16, 1, D), mx.float16), ks, vs, [10, 10], 0.0625)
