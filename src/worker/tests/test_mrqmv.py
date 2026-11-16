"""mrqmv (QWEN_VERIFY_QMV: multi-row 4 / 5 / 6 / 8-bit qmv for short verify blocks): every
row of a T-row call is MLX's solo qmv of that row bit for bit, at the 27B's
shapes (random 4-bit g64 weights) and small ones; install() marks modules and
fused sets only where the self-test passed; the verify hooks route (1, T, K)
blocks there and leave everything else alone. Needs Metal (skipped
otherwise; conftest makes the CPU the default device, these tests switch to
the GPU and back) - except the CPU fallback test at the end.
"""
import mlx.core as mx
import mlx.nn as nn
import pytest

import mrqmv

metal = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


@pytest.fixture
def gpu(monkeypatch):
    prev = mx.default_device()
    mx.set_default_device(mx.gpu)
    monkeypatch.setattr(mrqmv, "_OK", {})
    yield
    mx.set_default_device(prev)


def q4(N, K, seed=0):
    wf = (mx.random.normal((N, K), key=mx.random.key(seed)) * K ** -0.5).astype(mx.float16)
    return mx.quantize(wf, group_size=64, bits=4)


def solo(x, w, s, b):
    return mx.concatenate([mx.quantized_matmul(x[t:t + 1], w, s, b, transpose=True, group_size=64, bits=4)
                           for t in range(x.shape[0])])


# the 27B's 4-bit linears (K, N): q/k/v/o, GDN in/out, MLP, lm_head (a slice)
SHAPES = [(5120, 12288), (5120, 1024), (6144, 5120), (5120, 17408), (17408, 5120), (5120, 8192),
          (5120, 16384), (512, 8), (1024, 24)]


@metal
@pytest.mark.parametrize("K,N", SHAPES)
@pytest.mark.parametrize("T", [2, 3, 4, 5, 8])
def test_rows_equal_solo_qmv(gpu, K, N, T):
    w, s, b = q4(N, K, seed=K + N)
    x = (mx.random.normal((T, K), key=mx.random.key(T)) * 0.7).astype(mx.float16)
    assert mrqmv.shape_ok(w, s, b)
    assert mx.array_equal(mrqmv.rows(x, w, s, b), solo(x, w, s, b)).item()


@metal
def test_outlier_rows_and_selftest(gpu):
    w, s, b = q4(1024, 2048, seed=3)
    x = (mx.random.normal((4, 2048), key=mx.random.key(9)) * mx.array([[0.01], [1.0], [60.0], [3.0]])
         ).astype(mx.float16)
    assert mx.array_equal(mrqmv.rows(x, w, s, b), solo(x, w, s, b)).item()
    assert mrqmv.tested(w, s, b, 6) and mrqmv._OK[(2048, 1024, 4)] is None


def qb(N, K, bits, seed=0):
    wf = (mx.random.normal((N, K), key=mx.random.key(seed)) * K ** -0.5).astype(mx.float16)
    return mx.quantize(wf, group_size=64, bits=bits)


# the oQ4e's 5 / 6 / 8-bit shapes (K, N): attention o / q, MLP down, GDN out, lm_head (a slice)
SHAPES_B = [(6144, 5120), (5120, 12288), (17408, 5120), (5120, 1024), (5120, 4096), (512, 8)]


@metal
@pytest.mark.parametrize("bits", [5, 6, 8])
@pytest.mark.parametrize("K,N", SHAPES_B)
@pytest.mark.parametrize("T", [2, 3])
def test_rows_equal_solo_qmv_other_bits(gpu, bits, K, N, T):
    """5 / 6 / 8 bits (MLX's load_vector / qdot per row): == the solo qmv."""
    w, s, b = qb(N, K, bits, seed=K + N + bits)
    x = (mx.random.normal((T, K), key=mx.random.key(T)) * mx.array([[0.7], [30.0], [0.01]][:T])
         ).astype(mx.float16)
    assert mrqmv.shape_ok(w, s, b) and mrqmv.bits_of(w, s) == bits
    one = mx.concatenate([mx.quantized_matmul(x[t:t + 1], w, s, b, transpose=True, group_size=64,
                                              bits=bits) for t in range(T)])
    assert mx.array_equal(mrqmv.rows(x, w, s, b), one).item()
    assert mrqmv.tested(w, s, b, 3)


def test_shape_rules():
    w, s, b = mx.quantize(mx.zeros((64, 1024), dtype=mx.float16), group_size=64, bits=4)
    assert mrqmv.shape_ok(w, s, b)
    assert not mrqmv.shape_ok(w, s, b, bits=8)
    for bits in (5, 6, 8):
        wb, sb, bb = mx.quantize(mx.zeros((64, 1024), dtype=mx.float16), group_size=64, bits=bits)
        assert mrqmv.shape_ok(wb, sb, bb) and mrqmv.bits_of(wb, sb) == bits
    w7, s7, b7 = mx.quantize(mx.zeros((64, 1024), dtype=mx.float16), group_size=64, bits=3)
    assert not mrqmv.shape_ok(w7, s7, b7)                  # 3 bits: not taken
    assert not mrqmv.shape_ok(w, s, None)
    w2, s2, b2 = mx.quantize(mx.zeros((64, 576), dtype=mx.float16), group_size=64, bits=4)
    assert not mrqmv.shape_ok(w2, s2, b2)                  # K % 512
    w3, s3, b3 = mx.quantize(mx.zeros((64, 1024), dtype=mx.bfloat16), group_size=64, bits=4)
    assert not mrqmv.shape_ok(w3, s3, b3)                  # bf16 scales


@metal
def test_install_and_verify_hooks(gpu, monkeypatch):
    """install marks 4 / 5 / 6 / 8-bit g64 modules; _verify_linear(s) send 2..N-token
    blocks through mrqmv (== the solo rows) and anything else to the old path."""
    from qwen_worker import config as cfg
    from qwen_worker import kernels
    m = nn.Module()
    m.a = nn.QuantizedLinear(1024, 256, bias=False, group_size=64, bits=4)
    m.b = nn.QuantizedLinear(1024, 128, bias=False, group_size=64, bits=4)
    m.c = nn.QuantizedLinear(1024, 128, bias=False, group_size=64, bits=3)   # not taken
    m.d = nn.QuantizedLinear(1024, 128, bias=False, group_size=64, bits=8)
    for name in ("a", "b", "c", "d"):
        lin = getattr(m, name)
        lin.scales = lin.scales.astype(mx.float16)
        lin.biases = lin.biases.astype(mx.float16)
    mx.eval(m.parameters())
    n, nf = mrqmv.install(m, 4)
    assert (n, nf) == (3, 0) and m.a.__dict__.get("_qmv") and m.d.__dict__.get("_qmv")
    assert not m.c.__dict__.get("_qmv")
    monkeypatch.setattr(cfg, "VERIFY_QMV", 4)
    calls = []
    monkeypatch.setattr(kernels, "_VERIFY_LINEAR", lambda self, lin, x: calls.append(x.shape) or lin(x))
    monkeypatch.setattr(kernels, "_VERIFY_LINEARS", lambda self, ls, x: tuple(lin(x) for lin in ls))
    monkeypatch.setattr(kernels, "VERIFY_MR", False)
    for T in (1, 2, 4, 5):
        x = (mx.random.normal((1, T, 1024), key=mx.random.key(T))).astype(mx.float16)
        y = kernels._verify_linear(None, m.a, x)
        ref = solo(x[0], m.a["weight"], m.a["scales"], m.a["biases"])[None]
        assert mx.array_equal(y, ref).item()
        ya, yb = kernels._verify_linears(None, (m.a, m.b), x)
        assert mx.array_equal(yb, solo(x[0], m.b["weight"], m.b["scales"], m.b["biases"])[None]).item()
    assert calls == [(1, 1, 1024), (1, 5, 1024)]           # T = 1 and 5: the old path


def test_cpu_selftest_fails_closed(monkeypatch):
    """Without a Metal kernel (CPU device) the self-test fails and nothing is
    marked: the knob degrades to the old verify path."""
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mrqmv, "_OK", {})
    try:
        lin = nn.QuantizedLinear(1024, 64, bias=False, group_size=64, bits=4)
        lin.scales, lin.biases = lin.scales.astype(mx.float16), lin.biases.astype(mx.float16)
        m = nn.Module()
        m.l = lin
        assert mrqmv.install(m, 4) == (0, 0) and not lin.__dict__.get("_qmv")
    finally:
        mx.set_default_device(prev)
