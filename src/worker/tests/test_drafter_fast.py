"""QWEN_DRAFT_MRQMM / QWEN_DRAFT_COMPILE on the CPU: the row prescale, the
routing of the drafter's quantized linears (mrqmm_wide replaced by a CPU
stand-in with the kernel's arithmetic: fp16 x, fp16 dequantized weights, fp32
sums), the install, the compiled DFlash2 convolution. The Metal kernel itself:
tests/test_mrqmm.py::test_wide_*."""
import importlib.util
import os
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import worker
from qwen_worker import kernels

MRQMM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mrqmm.py")


@pytest.fixture
def mr(monkeypatch):
    """mrqmm in sys.modules, Metal "available", its self-test passed, the wide
    kernel replaced by its arithmetic on the CPU; calls recorded."""
    spec = importlib.util.spec_from_file_location("mrqmm", MRQMM)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "enabled", lambda: True)
    monkeypatch.setattr(m, "_selftest", lambda *a: None)
    m.calls = []

    def wide(x, w, sc, b, rs, cfg=None):
        K = x.shape[-1]
        bits = m._bits_of(K, w.shape[1])
        assert x.dtype == mx.float16 and sc.dtype == mx.float16 and rs.dtype == mx.float32
        m.calls.append((tuple(x.shape), w.shape[0], bits))
        deq = mx.dequantize(w, sc, b, group_size=64, bits=bits)          # fp16, as the kernel's A side
        y = x.reshape(-1, K).astype(mx.float32) @ deq.astype(mx.float32).T
        return (y * rs.reshape(-1, 1)).reshape(*x.shape[:-1], w.shape[0])
    monkeypatch.setattr(m, "mrqmm_wide", wide)
    monkeypatch.setitem(sys.modules, "mrqmm", m)
    return m


def qlin(K, N, bits, seed=0, scale=0.05):
    mx.random.seed(seed)
    lin = nn.Linear(K, N, bias=False)
    lin.weight = (mx.random.normal((N, K)) * scale).astype(mx.bfloat16)
    q = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=bits)
    # as _widen_drafter leaves them: bf16 -> fp32
    q.scales, q.biases = q.scales.astype(mx.float32), q.biases.astype(mx.float32)
    mx.eval(q.parameters())
    return q


def test_prescale_is_exact_power_of_two():
    x = mx.array([[[3.0, -100.5, 0.25, 1e-5]], [[0.0, 0.0, 0.0, 0.0]], [[1e-30, -2e-30, 0.0, 5e-31]],
                  [[60000.0, 1.0, -7e4, 3.0]]], dtype=mx.float32)
    xs, rs = kernels._prescale_rows(x)
    assert xs.dtype == mx.float16 and rs.dtype == mx.float32 and rs.shape == (4,)
    r = np.array(rs)
    assert np.all(np.log2(r) == np.round(np.log2(r)))                  # powers of two
    mx16 = np.abs(np.array(xs.astype(mx.float32))).max(axis=-1).ravel()
    assert np.all((mx16[[0, 2, 3]] >= 1024) & (mx16[[0, 2, 3]] < 2048)) and mx16[1] == 0
    back = np.array(xs.astype(mx.float32)) * r[:, None, None]
    xn = np.array(x)
    assert np.all(np.abs(back - xn) <= 2.0 ** -11 * np.abs(xn) + 1e-45)
    xc, rc = kernels._prescale_c(x)
    assert mx.array_equal(xc, xs).item() and mx.array_equal(rc, rs).item()


def test_install_and_routing(mr, monkeypatch):
    m = nn.Module()
    m.a = qlin(256, 64, 4)
    m.b = qlin(256, 128, 5, seed=1)
    m.c = qlin(320, 64, 6, seed=2)              # no kernel for 6 bits
    m.d = nn.Linear(64, 64)                      # not quantized
    ref = {k: getattr(m, k) for k in "abc"}
    x8 = (mx.random.normal((1, 8, 256), key=mx.random.key(3)) * 40).astype(mx.float32)
    off = {k: ref[k](x8 if k != "c" else mx.zeros((1, 8, 320))) for k in "ab"}
    mx.eval(off)
    assert kernels._install_drafter_mrqmm(m, keep_fp32=True) == 2
    assert type(m.a) is kernels._DraftQuantizedLinear and type(m.c) is nn.QuantizedLinear
    assert m.a.scales.dtype == mx.float32 and m.a.__dict__["_mr_wide"][1].dtype == mx.float16
    monkeypatch.setattr(worker, "DRAFT_MRQMM", False)
    for k in "ab":                               # off: quantized_matmul, bit for bit
        assert mx.array_equal(getattr(m, k)(x8), off[k]).item()
    assert mr.calls == []
    monkeypatch.setattr(worker, "DRAFT_MRQMM", True)
    for k, bits in (("a", 4), ("b", 5)):
        y = getattr(m, k)(x8)
        assert y.dtype == mx.float32 and y.shape == off[k].shape
        err = mx.abs(y - off[k]).max().item()
        assert err <= 2e-3 * mx.abs(off[k]).max().item(), (k, err)
    assert mr.calls == [((8, 1, 256), 64, 4), ((8, 1, 256), 128, 5)]
    for T in (1, 2, 16):                         # 1..16 rows on the kernel
        m.a(mx.ones((1, T, 256)))
    assert [c[0][0] for c in mr.calls[2:]] == [1, 2, 16]
    n = len(mr.calls)
    x17 = mx.random.normal((1, 17, 256), key=mx.random.key(4))
    y = m.a(x17)                                 # 17 rows: quantized_matmul
    assert len(mr.calls) == n and mx.array_equal(y, nn.QuantizedLinear.__call__(m.a, x17)).item()
    y16 = m.a(mx.ones((2, 4, 256), mx.float16))  # fp16 input: no prescale (rs = 1)
    assert y16.dtype == mx.float32 and mr.calls[-1][0] == (8, 1, 256)


def test_install_replaces_scales_and_skips_inexact(mr):
    m = nn.Module()
    m.a = qlin(256, 64, 8)
    m.b = qlin(256, 64, 8, seed=1)
    m.b.scales = m.b.scales * (1 + 2.0 ** -20)    # not representable in fp16
    x = mx.random.normal((1, 5, 256), key=mx.random.key(5))
    before = m.a(x)
    assert kernels._install_drafter_mrqmm(m, keep_fp32=False) == 1
    assert m.a.scales.dtype == mx.float16 and m.a.biases.dtype == mx.float16
    assert type(m.b) is nn.QuantizedLinear and m.b.scales.dtype == mx.float32
    # the off path with fp16 scales (promoted) == the fp32 scales' result
    assert mx.array_equal(nn.QuantizedLinear.__call__(m.a, x), before).item()


def test_compiled_conv_matches_library(monkeypatch):
    from mlx_vlm.speculative.drafters.dflash2 import dflash2 as d2
    assert kernels._install_drafter_compile()
    assert d2._grouped_dynamic_convolve is kernels._grouped_conv
    conv = d2.GroupedDynamicCausalConv(64, 2, 16)
    conv.base_kernel = mx.random.normal((2, 2, 64), key=mx.random.key(1))
    h = mx.random.normal((1, 8, 64), key=mx.random.key(2)) * 1000
    monkeypatch.setattr(worker, "DRAFT_COMPILE", False)
    p0, k0 = conv.prepare(h)
    f0 = conv.finish(p0, k0)
    lib = kernels._GROUPED_CONV(p0, k0, conv.base_kernel[1], 16)
    assert mx.array_equal(f0, lib).item()
    monkeypatch.setattr(worker, "DRAFT_COMPILE", True)
    p1, k1 = conv.prepare(h)
    f1 = conv.finish(p1, k1)
    assert mx.allclose(p1, p0, rtol=1e-6, atol=1e-3).item() and mx.allclose(f1, f0, rtol=1e-6, atol=1e-3).item()
