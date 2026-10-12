"""Prefill speed paths: pfqmm's dispatch rules (which calls take the 64-row
kernel) and the fixed fp16 slab (_step) — CPU only. The Metal kernel itself (bit-identical to
mx.quantized_matmul on 3..8-bit weights, a tiny quantized model's prefill)
runs only with PFQMM_GPU=1 (it takes the GPU):

    PFQMM_GPU=1 uv run --group dev pytest -q tests/test_pfqmm.py
"""
import os

import mlx.core as mx
import mlx.nn as nn
import pytest
from conftest import build_lm, prompts

import pfqmm
import worker


@pytest.mark.parametrize("M,K,N,uses", [
    (16, 4096, 12288, False),     # qmv
    (32, 4096, 12288, True),      # 384 tiles: split 1
    (64, 4096, 4096, False),      # 256 tiles: split-K 2
    (128, 4096, 4096, True),
    (256, 4096, 1024, False),     # split-K 2
    (512, 4096, 1024, True),
    (2048, 4096, 32, False),      # the GDN b/a projections: split-K
    (2048, 12288, 4096, True),
])
def test_mlx_uses_qmm(M, K, N, uses):
    assert pfqmm._mlx_uses_qmm(M, K, N, 64) is uses


def _qlinear(K, N, bits=8, dtype=mx.float16):
    m = nn.QuantizedLinear(K, N, bias=False, group_size=64, bits=bits)
    m.scales, m.biases = m.scales.astype(dtype), m.biases.astype(dtype)
    return m


@pytest.mark.parametrize("M,use", [(96, False), (128, True), (288, False), (2000, True),
                                   (2048, True), (100000, True)])
def test_use_rows(M, use):
    m = _qlinear(256, 4096)
    assert pfqmm._use(m, mx.zeros((1, M, 256), mx.float16)) is use


def test_use_batched_x_falls_back():
    """Leading dims > 1: MLX may dispatch by shape[-2] (batched qmm / qmv), not
    the qmm_t mirrored here, so pfqmm never takes it."""
    m = _qlinear(256, 4096)
    assert pfqmm._use(m, mx.zeros((1, 1, 2048, 256), mx.float16))
    assert pfqmm._use(m, mx.zeros((2048, 256), mx.float16))
    assert not pfqmm._use(m, mx.zeros((2, 1024, 256), mx.float16))
    assert not pfqmm._use(m, mx.zeros((2, 1, 2048, 256), mx.float16))


def test_use_dtype_and_supported():
    m = _qlinear(256, 4096)
    assert pfqmm.supported(m)
    assert not pfqmm._use(m, mx.zeros((1, 2048, 256), mx.bfloat16))
    assert not pfqmm.supported(_qlinear(256, 4096, dtype=mx.bfloat16))
    assert not pfqmm.supported(_qlinear(256, 4080))               # N % 32 != 0
    assert not pfqmm.supported(nn.Linear(256, 4096))
    assert pfqmm.supported(_qlinear(256, 4096, bits=5))
    assert not pfqmm.supported(_qlinear(256, 4096, bits=2))       # untested: left to MLX
    g128 = nn.QuantizedLinear(256, 4096, bias=False, group_size=128, bits=8)
    g128.scales, g128.biases = g128.scales.astype(mx.float16), g128.biases.astype(mx.float16)
    assert not pfqmm.supported(g128)


def test_test_rows_never_vacuous():
    """The self-test's rows: MLX runs qmm_t there and 8 rows further (checked
    against the dispatch mirror), or None when the shape never leaves split-K
    in range — never a row count it would skip."""
    for K, N in [(4096, 248320), (4096, 12288), (2048, 256), (4096, 1024), (128, 256),
                 (5120, 640), (4096, 32), (2048, 32)]:
        M = pfqmm._test_rows(K, N, 64)
        if M is None:
            assert not any(pfqmm._mlx_uses_qmm(r, K, N, 64) and pfqmm._mlx_uses_qmm(r + 8, K, N, 64)
                           for r in range(pfqmm.TEST_MIN_ROWS, pfqmm.TEST_MAX_ROWS + 1, 64))
            continue
        assert M % 64 == 0 and pfqmm.TEST_MIN_ROWS <= M <= pfqmm.TEST_MAX_ROWS
        assert pfqmm._mlx_uses_qmm(M, K, N, 64) and pfqmm._mlx_uses_qmm(M + 8, K, N, 64)
    assert pfqmm._test_rows(4096, 248320, 64) == 256                # 9B lm_head
    assert pfqmm._test_rows(2048, 256, 64) == 1088                  # 35B router: split-K below
    assert pfqmm._test_rows(4096, 32, 64) is None                   # GDN b/a: always split-K
    with pytest.raises(ValueError):                                  # a split-K row count: refused
        pfqmm._selftest(_qlinear(2048, 256), 256)


class _Holder(nn.Module):
    def __init__(self, **mods):
        super().__init__()
        for k, v in mods.items():
            setattr(self, k, v)


def _install_cpu(monkeypatch, model, selftest):
    """install() with the kernel stubbed out (no Metal needed)."""
    monkeypatch.setattr(pfqmm, "enabled", lambda: True)
    monkeypatch.setattr(pfqmm, "_selftest", selftest)
    return pfqmm.install(model)


def test_install_per_key(monkeypatch, caplog):
    """One (bits, gs) key fails its self-test, one raises, one has no module
    MLX would run as qmm_t (untested), one passes: only the last is covered;
    the self-test runs on the first module of a key that qualifies."""
    # named_modules walks children last-set first: a8 (never qmm_t) is met
    # before b8, so the 8-bit key's self-test has to walk on to b8
    m = _Holder(e6=_qlinear(4096, 32, bits=6), d5=_qlinear(256, 4096, bits=5),
                c4=_qlinear(256, 4096, bits=4), b8=_qlinear(4096, 1024), a8=_qlinear(4096, 32))
    assert [n for n, _ in m.named_modules()].index("a8") < [n for n, _ in m.named_modules()].index("b8")
    seen = []

    def selftest(mod, M):
        seen.append((mod.bits, mod["weight"].shape[0], M))
        if mod.bits == 4:
            return "differs"
        if mod.bits == 5:
            raise RuntimeError("no headers")
        return None
    caplog.set_level("INFO", logger="worker.pfqmm")
    try:
        assert _install_cpu(monkeypatch, m, selftest) == 2            # the 8-bit key: a8 and b8
        on = [k for k in ("a8", "b8", "c4", "d5", "e6") if getattr(m, k).__dict__.get(pfqmm._ATTR)]
        assert on == ["a8", "b8"]
    finally:
        pfqmm.uninstall(m)
    assert sorted(seen) == [(4, 4096, pfqmm._test_rows(256, 4096, 64)),
                            (5, 4096, pfqmm._test_rows(256, 4096, 64)),
                            (8, 1024, pfqmm._test_rows(4096, 1024, 64))]
    text = caplog.text
    assert "4-bit g64: failed" in text and "5-bit g64: failed" in text and "RuntimeError" in text
    assert "6-bit g64: untested" in text and "8-bit g64: tested" in text


def test_install_untested_key_stays_off(monkeypatch):
    """The vacuous case: every module of the key always splits K, so nothing
    is compared — and nothing is enabled."""
    m = _Holder(a=_qlinear(4096, 32), b=_qlinear(2048, 32))

    called = []
    assert _install_cpu(monkeypatch, m, lambda mod, M: called.append(M)) == 0
    assert not called
    assert not any(x.__dict__.get(pfqmm._ATTR) for x in (m.a, m.b))


def test_install_logs_when_nothing_supported(monkeypatch, caplog):
    caplog.set_level("INFO", logger="worker.pfqmm")
    m = _Holder(a=_qlinear(256, 4096, dtype=mx.bfloat16))          # QWEN_FP16=0
    assert _install_cpu(monkeypatch, m, lambda mod, M: None) == 0
    assert "QWEN_FP16=0" in caplog.text


def test_step_fixed_for_fp16_kv(monkeypatch):
    monkeypatch.setattr(worker, "_FIXED_SLAB", None)
    monkeypatch.setattr(worker, "PREFILL_STEP", 2048)
    monkeypatch.setattr(worker, "PREFILL_BUDGET", 2048 * 32768)
    monkeypatch.setattr(worker, "FP16", True)
    monkeypatch.setattr(worker, "SDPA_FUSED", True)
    monkeypatch.setattr(worker, "KV_KWARGS", dict(worker.KV_KWARGS, kv_bits=None))
    assert worker._step(122880) == 2048
    monkeypatch.setattr(worker, "SDPA_FUSED", False)            # unfused SDPA: the budget again
    assert worker._step(122880) == 546
    monkeypatch.setattr(worker, "SDPA_FUSED", True)
    monkeypatch.setattr(worker, "KV_KWARGS", dict(worker.KV_KWARGS, kv_bits=8))
    assert worker._step(122880) == 546
    assert worker._step(16384) == 2048


def test_step_budget_below_fused_threshold(monkeypatch):
    """The fixed fp16 slab keeps QWEN_PREFILL_BUDGET while the slab would still
    run the unfused SDPA (pos + step < 32K keys); the policy is frozen at load;
    the prefill reserve covers a full step under the threshold."""
    monkeypatch.setattr(worker, "_FIXED_SLAB", None)
    monkeypatch.setattr(worker, "FP16", True)
    monkeypatch.setattr(worker, "SDPA_FUSED", True)
    monkeypatch.setattr(worker, "KV_KWARGS", dict(worker.KV_KWARGS, kv_bits=None))
    monkeypatch.setattr(worker, "PREFILL_BUDGET", 2048 * 32768)
    monkeypatch.setattr(worker, "PREFILL_STEP", 4096)
    assert worker._step(27000) == 2048 * 32768 // 27000          # 2485: unfused, budget holds
    assert worker._step(28672) == 4096                           # reaches 32K keys: fused
    assert worker._step(120000) == 4096
    monkeypatch.setattr(worker, "PREFILL_STEP", 2048)
    monkeypatch.setattr(worker, "PREFILL_BUDGET", 512 * 32768)   # lowered budget is honoured
    assert worker._step(16384) == 1024 and worker._step(31000) == 2048
    # defaults: never shrinks (identical cuts to before)
    monkeypatch.setattr(worker, "PREFILL_BUDGET", 2048 * 32768)
    assert all(worker._step(p) == 2048 for p in range(0, 140000, 997))
    # reserve: the budget's scores, at least PREFILL_STEP x 32K with the fixed slab
    heads = 16
    monkeypatch.setattr(worker, "PREFILL_BUDGET", 512 * 32768)
    assert worker._prefill_reserve(heads) == 2048 * 32768 * heads * 2 + (400 << 20)
    monkeypatch.setattr(worker, "_FIXED_SLAB", False)            # frozen at load: unfused policy
    assert worker._prefill_reserve(heads) == 512 * 32768 * heads * 2 + (400 << 20)
    assert worker._step(122880) == 256                           # max(MIN_STEP, 16M / 120K)
    monkeypatch.setattr(worker, "_FIXED_SLAB", True)
    monkeypatch.setattr(worker, "SDPA_FUSED", False)             # a runtime flip changes nothing
    assert worker._step(122880) == 2048


def test_fused_sdpa_fallback_is_bounded(monkeypatch):
    """No fused kernel at run time (the CPU): a slab whose unfused scores
    would pass PREFILL_BUDGET raises instead of materializing them."""
    from mlx_vlm.models.base import scaled_dot_product_attention
    from mlx_vlm.models.cache import KVCache
    monkeypatch.setattr(worker, "_SDPA", worker._SDPA or scaled_dot_product_attention)
    monkeypatch.setattr(worker, "SDPA_FUSED", True)
    monkeypatch.setattr(worker, "SDPA_FUSED_MIN_Q", 8)
    monkeypatch.setattr(worker, "SDPA_FUSED_MIN_KV", 32)
    monkeypatch.setattr(worker, "PREFILL_BUDGET", 8 * 40)
    q = mx.zeros((1, 4, 16, 64), mx.float16)
    k = mx.zeros((1, 2, 40, 64), mx.float16)
    with pytest.raises(RuntimeError, match="PREFILL_BUDGET"):
        worker._fused_sdpa(q, k, k, cache=KVCache(), scale=0.125, mask="causal")
    assert worker.SDPA_FUSED is False
    with pytest.raises(RuntimeError):                           # stays bounded after the flip
        worker._fused_sdpa(q, k, k, cache=KVCache(), scale=0.125, mask="causal")
    worker._fused_sdpa(q[:, :, :8], k, k, cache=KVCache(), scale=0.125, mask="causal")   # 8 x 40: fits


# --- Metal (PFQMM_GPU=1) -----------------------------------------------------

gpu = pytest.mark.skipif(not (os.environ.get("PFQMM_GPU") == "1" and mx.metal.is_available()),
                         reason="PFQMM_GPU=1 and Metal")


@pytest.fixture
def on_gpu():
    prev = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(prev)


@gpu
@pytest.mark.parametrize("bits", [3, 4, 5, 6, 8])
@pytest.mark.parametrize("K,N", [(4096, 4096), (4096, 1024), (5120, 640)])
def test_kernel_bit_identical(on_gpu, bits, K, N):
    w = (mx.random.normal((N, K), key=mx.random.key(bits)) * 0.02).astype(mx.float16)
    wq, s, b = mx.quantize(w, group_size=64, bits=bits)
    s, b = s.astype(mx.float16), b.astype(mx.float16)
    m = _qlinear(K, N, bits)
    m.weight, m.scales, m.biases = wq, s, b
    for M in (128, 200, 256, 640, 2048):
        x = (mx.random.normal((1, M, K), key=mx.random.key(M)) * 2).astype(mx.float16)
        qmm = pfqmm._mlx_uses_qmm(M, K, N, 64)
        assert pfqmm._use(m, x) == (qmm and M != 200)
        if not qmm:                  # MLX runs split-K there: other rounding, never taken
            continue
        ref = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=bits)
        y = pfqmm.pfqmm(x, wq, s, b, bits, 64)
        assert mx.array_equal(y, ref).item(), (bits, K, N, M)


@gpu
def test_tiny_model_prefill(on_gpu):
    """2112 rows: enough for the tiny model's narrow linears (K = 128/256,
    N = 128/256) to leave split-K, so every linear takes the kernel."""
    lm = build_lm("dense", wbits=8)
    assert pfqmm.install(lm) > 0
    ids = mx.array([prompts([2112], 1)[0]])
    calls, real = [], pfqmm.pfqmm

    def spy(*a, **k):
        calls.append(a[0].shape)
        return real(*a, **k)

    pfqmm.pfqmm = spy
    try:
        outs = []
        for active in (False, True):
            pfqmm.ACTIVE = active
            from mlx_vlm.models.cache import make_prompt_cache
            outs.append(lm(ids, cache=make_prompt_cache(lm)).logits)
            mx.eval(outs[-1])
        assert mx.array_equal(*outs).item()
        assert len(calls) >= 8 * 4             # every layer's projections + MLP
    finally:
        pfqmm.pfqmm = real
        pfqmm.ACTIVE = True
        pfqmm.uninstall(lm)


@gpu
def test_selftest_compares_on_a_split_k_first_module(on_gpu):
    """The 35B case: the 8-bit key's first module is the MoE router (K=2048,
    N=256: split-K at 256 rows). install compares it at rows where MLX runs
    qmm_t (1088 / 1096) instead of passing without a comparison."""
    w = (mx.random.normal((256, 2048), key=mx.random.key(3)) * 0.02).astype(mx.float16)
    m = _qlinear(2048, 256)
    wq, s, b = mx.quantize(w, group_size=64, bits=8)
    m.weight, m.scales, m.biases = wq, s.astype(mx.float16), b.astype(mx.float16)
    calls, real = [], pfqmm.pfqmm

    def spy(x, *a, **k):
        calls.append(x.shape[-2])
        return real(x, *a, **k)
    pfqmm.pfqmm = spy
    try:
        assert pfqmm.install(_Holder(gate=m)) == 1
    finally:
        pfqmm.pfqmm = real
        pfqmm.uninstall(_Holder(gate=m))
    assert calls == [1088, 1096]
