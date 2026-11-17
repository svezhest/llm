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


# ---- 8-bit quantized KV (vatt.attention_q8, QWEN_VATT_Q8) ----

def kvq(n, hkv, spare=37, seed=0, strided=False):
    """An 8-bit QuantizedKVCache's buffers: (packed, scales, biases) for K and
    V with spare slots past n holding garbage (random bytes, K scales 1000, V
    scales NaN): never read."""
    k, v = kv(n, hkv, spare, seed, strided=False)
    k, v = k[..., :n, :], v[..., :n, :]
    pad = [(0, 0), (0, 0), (0, spare), (0, 0)]
    kq = [mx.pad(a, pad) for a in mx.quantize(k, 64, 8)]
    vq = [mx.pad(a, pad) for a in mx.quantize(v, 64, 8)]
    kq[0][..., n:, :] = 0x7f3a91c5
    vq[0][..., n:, :] = 0x12ffee01
    kq[1][..., n:, :] = 1000.0
    vq[1][..., n:, :] = float("nan")
    if strided:                     # every other head of a bigger buffer: non-contiguous views
        kq = [mx.concatenate([a, a], axis=1)[:, ::2] for a in kq]
        vq = [mx.concatenate([a, a], axis=1)[:, ::2] for a in vq]
    mx.eval(kq, vq)
    return tuple(kq), tuple(vq)


def rows_q8_lib(q, kq, vq, n, scale):
    """The library's verify attention over a quantized cache: one quantized
    SDPA call per query."""
    from mlx_vlm.models.base import quantized_scaled_dot_product_attention as qsdpa
    T = q.shape[2]
    return mx.concatenate([qsdpa(q[:, :, i:i + 1], tuple(x[..., :n - T + i + 1, :] for x in kq),
                                 tuple(x[..., :n - T + i + 1, :] for x in vq), scale, None, 64, 8)
                           for i in range(T)], axis=2)


CASES_Q8 = [  # (n, T, hq, hkv)
    (1, 1, 24, 4), (8, 8, 24, 4), (9, 8, 24, 4), (100, 8, 24, 4), (700, 8, 24, 4),
    (2049, 8, 24, 4), (4096 + 3, 8, 24, 4), (12345, 8, 24, 4), (33000, 8, 24, 4),
    (5000, 1, 24, 4), (5000, 2, 24, 4), (5000, 5, 24, 4), (5000, 9, 24, 4), (5000, 16, 24, 4),
    (3000, 8, 16, 4), (3000, 1, 16, 4), (3000, 8, 16, 2), (60000, 8, 24, 4), (60000, 1, 24, 4),
]


# Q8Loader into the fp16 kernel's tiles / fragments in registers (vattq_p1) / the same with scale
# and bias out of the sums: the affine values unrounded, so its reference is the exact one
KERNELS_Q8 = ["tile", "reg", "regf"]


def ref_q8(kernel, q, kq, vq, n, scale):
    return vatt.reference_q8(q, kq, vq, n, scale, exact=kernel == "regf")


@gpu_only
@pytest.mark.parametrize("n,T,hq,hkv", CASES_Q8)
@pytest.mark.parametrize("mul", [1.0, 8.0])
@pytest.mark.parametrize("kernel", KERNELS_Q8)
def test_q8_matches_reference(gpu, n, T, hq, hkv, mul, kernel):
    kq, vq = kvq(n, hkv, seed=n % 7)
    mx.random.seed(T)
    q = (mx.random.normal((1, hq, T, D)) * mul).astype(mx.float16)
    scale = D ** -0.5
    out = vatt.attention_q8(q, kq, vq, n, scale, kernel=kernel)
    ref = ref_q8(kernel, q, kq, vq, n, scale)
    lib = rows_q8_lib(q, kq, vq, n, scale)
    mx.eval(out, ref, lib)
    assert out.shape == q.shape and out.dtype == mx.float16
    assert not mx.any(mx.isnan(out)).item()
    e, e_lib = rel_rms(out, ref), rel_rms(lib, ref)
    assert e <= max(1.05 * e_lib, 4e-4), (e, e_lib)


@gpu_only
@pytest.mark.parametrize("T", [1, 8])
@pytest.mark.parametrize("kernel", KERNELS_Q8)
def test_q8_error_gate(gpu, kernel, T):
    """The microbench's shape (8K keys, |q| x 4): rms rel error <= 2.5e-4 (fp16 rounding of the
    output is 2.0-2.1e-4), tile / reg vs fp32 over mx.dequantize's values, regf vs fp32 over the
    affine values."""
    n = 8192
    kq, vq = kvq(n, 4, seed=6)
    q = (mx.random.normal((1, 24, T, D)) * 4).astype(mx.float16)
    out = vatt.attention_q8(q, kq, vq, n, D ** -0.5, kernel=kernel)
    assert rel_rms(out, ref_q8(kernel, q, kq, vq, n, D ** -0.5)) <= 2.5e-4


@gpu_only
@pytest.mark.parametrize("kernel", ["tile", "reg"])
def test_q8_dequantizes_like_mlx(gpu, kernel):
    """The in-register dequantization == mx.dequantize: the q8 kernel over the
    packed KV equals the fp16 kernel over mx.dequantize's fp16 KV (the same
    tiles after the loaders; "reg": the same fp16 values in its fragments) to
    fp16 rounding of the dequantized values."""
    n = 3000
    kq, vq = kvq(n, 4, seed=5)
    q = mx.random.normal((1, 24, 8, D)).astype(mx.float16)
    a = vatt.attention_q8(q, kq, vq, n, D ** -0.5, kernel=kernel)
    b = vatt.attention(q, vatt.dequantized(kq, n), vatt.dequantized(vq, n), n, D ** -0.5)
    assert mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item() <= 2e-3
    assert rel_rms(a, b.astype(mx.float32)) < 2e-4


@gpu_only
@pytest.mark.parametrize("n,chunk", [(3000, 48), (3000, 96), (1000, 960), (4100, 4080), (517, 2016), (3000, 8),
                                     (3000, 40)])
@pytest.mark.parametrize("kernel", KERNELS_Q8)
def test_q8_chunks_and_tails(gpu, n, chunk, kernel):
    if kernel == "tile" and chunk % 16:      # its key blocks are 16 keys, "reg"'s 8 (default tiles)
        pytest.skip("chunk not a multiple of the tile kernel's key block")
    kq, vq = kvq(n, 4, seed=1)
    q = mx.random.normal((1, 24, 8, D)).astype(mx.float16)
    for nn in (n, n - 3, n - 5):
        out = vatt.attention_q8(q, kq, vq, nn, D ** -0.5, chunk=chunk, kernel=kernel)
        assert rel_rms(out, ref_q8(kernel, q, kq, vq, nn, D ** -0.5)) < 4e-4, (nn, chunk)


@gpu_only
@pytest.mark.parametrize("kernel", KERNELS_Q8)
def test_q8_strided_view_no_copy(gpu, kernel):
    kq, vq = kvq(3000, 4, seed=2, strided=True)
    q = mx.random.normal((1, 24, 8, D)).astype(mx.float16)
    out = vatt.attention_q8(q, kq, vq, 2990, D ** -0.5, kernel=kernel)
    assert rel_rms(out, ref_q8(kernel, q, kq, vq, 2990, D ** -0.5)) < 4e-4


REG_TILES = [(1, 2, 1), (1, 4, 1), (2, 4, 1), (3, 4, 1), (2, 2, 1), (1, 2, 2), (2, 4, 2), (1, 1, 1)]


@gpu_only
@pytest.mark.parametrize("tile", REG_TILES)
@pytest.mark.parametrize("n,T", [(5, 1), (5, 5), (700, 1), (700, 3), (4099, 8), (9000, 16)])
@pytest.mark.parametrize("kernel", ["reg", "regf"])
def test_q8_reg_tiles(gpu, monkeypatch, tile, n, T, kernel):
    """Every "reg" tile (MB 8-row query blocks a simdgroup x WN simdgroups x KF 8-key
    fragments a key block) for one query and for blocks: partly filled row blocks, the
    causal tail, keys past n never read."""
    monkeypatch.setattr(vatt, "Q8R_DECODE", tile)
    monkeypatch.setattr(vatt, "Q8R_VERIFY", tile)
    kq, vq = kvq(n, 4, seed=3)
    mx.random.seed(n + T)
    q = (mx.random.normal((1, 24, T, D)) * 2).astype(mx.float16)
    out = vatt.attention_q8(q, kq, vq, n, D ** -0.5, kernel=kernel)
    assert not mx.any(mx.isnan(out)).item()
    assert rel_rms(out, ref_q8(kernel, q, kq, vq, n, D ** -0.5)) < 4e-4


@gpu_only
def test_q8_reg_close_to_tile(gpu):
    """Both kernels attend the same fp16 values (one fma a value): they differ by the
    order of the fp32 sums only."""
    n = 6000
    kq, vq = kvq(n, 4, seed=4)
    for T in (1, 8):
        q = (mx.random.normal((1, 24, T, D)) * 2).astype(mx.float16)
        a = vatt.attention_q8(q, kq, vq, n, D ** -0.5, kernel="tile").astype(mx.float32)
        b = vatt.attention_q8(q, kq, vq, n, D ** -0.5, kernel="reg").astype(mx.float32)
        assert rel_rms(b, a) < 2e-4            # fp16 output rounding of nearly equal fp32 values


@gpu_only
@pytest.mark.parametrize("kernel", ["tile", "reg"])
@pytest.mark.parametrize("timing", [1, 2, 3])
def test_q8_timing_variants_run(gpu, kernel, timing):
    """The microbench's cost-split kernels compile and run (their numbers are wrong by design)."""
    kq, vq = kvq(2000, 4, seed=1)
    for T in (1, 8):
        q = mx.random.normal((1, 24, T, D)).astype(mx.float16)
        out = vatt.attention_q8(q, kq, vq, 2000, D ** -0.5, kernel=kernel, timing=timing)
        mx.eval(out)
        assert out.shape == q.shape


@gpu_only
def test_fp16_path_unchanged_by_the_q8_kernels(gpu):
    """vatt.attention's kernel source does not depend on the q8 additions: the same bits
    before and after a q8 call compiled its kernels (the fp16 / q8 kernels share the header)."""
    k, v = kv(5000, 4, seed=9)
    q = mx.random.normal((1, 24, 8, D)).astype(mx.float16)
    a = vatt.attention(q, k, v, 5000, D ** -0.5)
    mx.eval(a)
    kq, vq = kvq(700, 4, seed=1)
    for kernel in KERNELS_Q8:
        mx.eval(vatt.attention_q8(q, kq, vq, 700, D ** -0.5, kernel=kernel))
    assert mx.array_equal(a, vatt.attention(q, k, v, 5000, D ** -0.5)).item()


def test_q8_rejects_bad_inputs():
    k = mx.quantize(mx.zeros((1, 4, 64, D), mx.float16), 64, 8)
    q = mx.zeros((1, 24, 8, D), mx.float16)
    for args in ((q, k, k, 40, 64, 4), (q, k, k, 40, 32, 8), (q, k, k, 65, 64, 8),
                 (q.astype(mx.float32), k, k, 40, 64, 8), (q, k[:2], k, 40, 64, 8),
                 (q, (k[0], k[1].astype(mx.float32), k[2]), k, 40, 64, 8)):
        with pytest.raises(ValueError):
            vatt.attention_q8(args[0], args[1], args[2], args[3], 0.0625, args[4], args[5])
    with pytest.raises(ValueError):
        vatt.attention_q8(q, k, k, 40, 0.0625, kernel="tiles")


def test_q8_cpu_reference_is_dequantized_attention():
    mx.random.seed(0)
    q = mx.random.normal((1, 4, 3, D)).astype(mx.float16)
    k = mx.random.normal((1, 2, 20, D)).astype(mx.float16)
    v = mx.random.normal((1, 2, 20, D)).astype(mx.float16)
    kq, vq = mx.quantize(k, 64, 8), mx.quantize(v, 64, 8)
    out = vatt.attention_q8(q, kq, vq, 15, 0.0625)
    want = vatt.attention(q, mx.dequantize(*kq, 64, 8), mx.dequantize(*vq, 64, 8), 15, 0.0625)
    assert mx.array_equal(out, want).item()
    assert mx.array_equal(vatt.attention_q8(q, kq, vq, 15, 0.0625, kernel="reg"), want).item()
    # "regf" attends the affine values scale * code + bias unrounded: the exact reference
    f = vatt.attention_q8(q, kq, vq, 15, 0.0625, kernel="regf")
    assert mx.array_equal(f, vatt.reference_q8(q, kq, vq, 15, 0.0625, exact=True).astype(mx.float16)).item()
    assert mx.abs(f.astype(mx.float32) - want.astype(mx.float32)).max().item() < 2e-2


def test_q8_exact_reference_is_the_affine_values():
    """reference_q8(exact=True) = attention over scale * code + bias computed in fp32."""
    mx.random.seed(1)
    q = mx.random.normal((1, 4, 2, D)).astype(mx.float16)
    kq = mx.quantize(mx.random.normal((1, 2, 12, D)).astype(mx.float16), 64, 8)
    vq = mx.quantize(mx.random.normal((1, 2, 12, D)).astype(mx.float16), 64, 8)

    def affine(x):
        w, s, b = x
        codes = mx.stack([(w >> (8 * i)) & 0xff for i in range(4)], axis=-1).reshape(1, 2, 12, D).astype(mx.float32)
        return codes * mx.repeat(s.astype(mx.float32), 64, axis=-1) + mx.repeat(b.astype(mx.float32), 64, axis=-1)
    want = vatt.reference(q, affine(kq), affine(vq), 12, 0.0625)
    got = vatt.reference_q8(q, kq, vq, 12, 0.0625, exact=True)
    assert mx.abs(got - want).max().item() < 1e-5


def test_q8_kernel_and_timing_plumbing(monkeypatch):
    """CPU: attention_q8 hands the kernel name (None = Q8_KERNEL), the reg tile, the timing
    variant and the launch geometry to the kernels (fakes; Metal: the VATT_GPU tests)."""
    seen = {}

    class Fake:
        def __init__(self, name):
            self.name = name

        def __call__(self, inputs, grid, threadgroup, output_shapes, output_dtypes):
            seen[self.name] = dict(grid=grid, threadgroup=threadgroup, n_in=len(inputs))
            return [mx.zeros(s, d) for s, d in zip(output_shapes, output_dtypes)]

    def reg(D_, hkv, scale, mb, wn, kf, fact=False, timing=0):
        seen["reg"] = (mb, wn, kf, fact, timing)
        return Fake("p1"), Fake("p2")

    def tile(D_, hkv, scale, bq, bk, wm, wn, q8=False, timing=0):
        seen["tile"] = (bq, bk, wm, wn, q8, timing)
        return Fake("p1"), Fake("p2")
    monkeypatch.setattr(vatt, "use_kernel", lambda: True)
    monkeypatch.setattr(vatt, "_kernels_reg", reg)
    monkeypatch.setattr(vatt, "_kernels", tile)
    monkeypatch.setattr(vatt, "Q8R_DECODE", (1, 2, 1))
    monkeypatch.setattr(vatt, "Q8R_VERIFY", (2, 4, 1))
    n = 5000
    kq = mx.quantize(mx.zeros((1, 4, n + 8, D), mx.float16), 64, 8)
    for T, kernel, timing, want, tg, blocks, bk in ((8, "reg", None, (2, 4, 1, False, 0), 128, 3, 8),
                                                    (8, "regf", 1, (2, 4, 1, True, 1), 128, 3, 8),
                                                    (1, "reg", 3, (1, 2, 1, False, 3), 64, 1, 8),
                                                    (16, "regf", None, (2, 4, 1, True, 0), 128, 6, 8)):
        seen.clear()
        q = mx.zeros((1, 24, T, D), mx.float16)
        out = vatt.attention_q8(q, kq, kq, n, 0.0625, kernel=kernel, timing=timing)
        assert out.shape == q.shape and seen["reg"] == want and "tile" not in seen
        chunk = vatt.pick_chunk(n, 4, blocks, bk)
        assert chunk % bk == 0
        assert seen["p1"] == dict(grid=(tg * -(-n // chunk) * blocks, 4, 1), threadgroup=(tg, 1, 1), n_in=8)
        assert seen["p2"]["grid"] == (D * 6 * T, 4, 1)
    seen.clear()
    vatt.attention_q8(mx.zeros((1, 24, 8, D), mx.float16), kq, kq, n, 0.0625, kernel="tile", timing=2)
    assert seen["tile"] == (16, 16, 2, 2, True, 2) and "reg" not in seen
    for name in vatt.Q8_KERNELS:               # the default kernel is the module's Q8_KERNEL
        seen.clear()
        monkeypatch.setattr(vatt, "Q8_KERNEL", name)
        vatt.attention_q8(mx.zeros((1, 24, 1, D), mx.float16), kq, kq, n, 0.0625)
        assert ("tile" in seen) == (name == "tile") and (name == "tile" or seen["reg"][3] == (name == "regf"))
    seen.clear()                               # the fp16 path never takes a q8 kernel or a timing variant
    k = mx.zeros((1, 4, n + 8, D), mx.float16)
    vatt.attention(mx.zeros((1, 24, 8, D), mx.float16), k, k, n, 0.0625)
    assert seen["tile"] == (16, 16, 2, 2, False, 0)


def test_q8_kernel_sources_render():
    """The Metal sources as text (no compiler here): the reg kernel's call and constants, one
    header per timing variant, balanced brackets in the new function."""
    import re
    assert "vattq_p1<MB, WN, KF, FACT>(q, kv, po, pml" in vatt._P1R and "KVQ8 kv;" in vatt._P1R
    assert "threadgroup float2 X[2 * WN * MB * KF * 32];" in vatt._P1R
    for t in range(4):
        assert f"#define VATT_Q8_TIMING {t}\n" in vatt._header(t)
    assert vatt._header(0) is vatt._header(0) and vatt._header(0) != vatt._header(1)
    f = vatt._IMPL[vatt._IMPL.index("template <int MB, int WN, int KF, bool FACT>"):]
    code = re.sub(r"//.*", "", f)
    for a, b in ("()", "{}", "[]"):
        assert code.count(a) == code.count(b), (a, code.count(a), code.count(b))
    assert vatt.Q8_KERNEL in vatt.Q8_KERNELS and vatt.Q8_KERNELS[0] == "tile"
