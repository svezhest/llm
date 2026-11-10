"""mrqmm by bit width (QWEN_MRQMM5: the 5-bit kernel for the 27B), CPU part:
the 8-bit kernel's source is pinned (the generalization must not move it), the
5-bit unpack scheme against mx.dequantize (numpy mirror of the kernel's
integer ops), and the dispatch: which weights `supported` / `install` take with
the knob off (default: 8-bit only, as before), on, and in batch-invariant
mode (never 5-bit), how `mrqmm` finds the bit width, and the worker's users
(fused GDN in-projection, DFlash verify block, the invariant check) on a tiny
5-bit model with the Metal kernel replaced by quantized_matmul.
The module is loaded under its name for the test only (the CPU suite must
not see it in sys.modules). Metal: tests/test_mrqmm.py."""
import hashlib
import importlib.util
import os
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from conftest import build_lm
from mlx_vlm.models.qwen3_5 import speculative_verifier
from mlx_vlm.speculative.ops import linear as vlinear

import worker

MRQMM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mrqmm.py")
# sha256 of the 8-bit kernel source as of d976b57 (before the bit-width split)
PIN8 = "1c42e9884a2f2823ac0639e11d072d7aa565fb5a21455e66dbad77113012483b"


def load():
    spec = importlib.util.spec_from_file_location("mrqmm", MRQMM)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def raw(monkeypatch):
    """The real module, Metal "available", nothing compiled."""
    m = load()
    monkeypatch.setattr(m, "enabled", lambda: True)
    monkeypatch.delenv("QWEN_MRQMM5", raising=False)
    return m


@pytest.fixture
def mr(raw, monkeypatch):
    """`raw` in sys.modules with its kernel -> quantized_matmul at the bit
    width mrqmm itself would pick; calls recorded as (x.shape, N, bits)."""
    raw.calls = []

    def kernel(x, w, s, b, cfg=None):
        bits = raw._bits_of(x.shape[-1], w.shape[1])
        raw.calls.append((tuple(x.shape), w.shape[0], bits))
        return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=bits)
    monkeypatch.setattr(raw, "mrqmm", kernel)
    monkeypatch.setattr(raw, "warm", lambda *a: None)
    monkeypatch.setattr(raw, "MIN_N", 8)                 # tiny layers
    monkeypatch.setitem(sys.modules, "mrqmm", raw)
    return raw


def qlinear(K, N, bits, dtype=mx.float16):
    lin = nn.QuantizedLinear(K, N, bias=False, group_size=64, bits=bits)
    lin.scales, lin.biases = lin.scales.astype(dtype), lin.biases.astype(dtype)
    mx.eval(lin.parameters())
    return lin


class Holder(nn.Module):
    def __init__(self, **mods):
        super().__init__()
        for k, v in mods.items():
            setattr(self, k, v)


def test_8bit_source_is_pinned(raw, monkeypatch):
    """bits = 8 renders the kernel text it had before the 5-bit split, under
    the same kernel name and constants; 5 bits is another kernel."""
    assert hashlib.sha256(raw._src(8).encode()).hexdigest() == PIN8
    assert raw._SRC == raw._src(8) and raw._src(5) != raw._src(8)
    made = []
    monkeypatch.setattr(mx.fast, "metal_kernel", lambda **kw: made.append(kw) or object())
    raw._get_kernel(128, 64, 1, 2, 2)
    raw._get_kernel(128, 64, 2, 4, 4, 8)
    raw._get_kernel(128, 64, 1, 2, 2, 5)
    assert all("@" not in raw._src(b) for b in raw.KERNEL_BITS) and len({raw._src(b) for b in raw.KERNEL_BITS}) == 3
    consts = "using T = half;\n    constexpr int K = 128, N = 64, MB = {}, NA = {}, KS = {};\n"
    assert [k["name"] for k in made] == ["mrqmm8_g64_k128_n64_mb1_a2_s2", "mrqmm8_g64_k128_n64_mb2_a4_s4",
                                         "mrqmm5_g64_k128_n64_mb1_a2_s2"]
    assert made[0]["source"] == consts.format(1, 2, 2) + raw._src(8)
    assert made[1]["source"] == consts.format(2, 4, 4) + raw._src(8)
    assert made[2]["source"] == consts.format(1, 2, 2) + raw._src(5)
    assert all(k["input_names"] == ["x", "w", "scales", "biases", "m0"] and k["header"] == raw._HEADER
               for k in made)
    assert "@" not in raw._src(5) and "@" not in raw._src(8)


def test_5bit_unpack_scheme():
    """The integer path of _WSIDE[5] in numpy: for every row, 64-group, thread
    (fn) and MMA step, the two lanes of `q` are the weights at k = 8fn + st
    and 8fn + 8 + st of MLX's 5-bit packing (mx.dequantize with scale 1)."""
    N, K = 16, 256
    mx.random.seed(0)
    w, s, b = mx.quantize(mx.random.normal((N, K)), group_size=64, bits=5)
    want = np.array(mx.dequantize(w, mx.ones_like(s), mx.zeros_like(b), group_size=64, bits=5))
    want = want.round().astype(np.int64)
    assert want.min() == 0 and want.max() == 31
    wn = np.array(w).astype(np.uint32)
    assert wn.shape == (N, K * 5 // 32)
    m32 = 0xffffffff

    def insert_bits(base, ins, off, bits):
        mask = ((1 << bits) - 1) << off
        return (base & ~mask & m32) | ((ins << off) & mask)

    got = np.full_like(want, -1)
    for r in range(N):
        for g in range(K // 64):
            for fn in (0, 2, 4, 6):
                p = (5 * fn) // 4 + 10 * g
                a0, a1, a2 = (int(wn[r, p + i]) for i in range(3))
                odd = (fn // 2) % 2 != 0
                v0 = ((a0 >> 16) | (a1 << 16)) & m32 if odd else a0
                v1 = ((a1 >> 16) | (a2 << 16)) & m32 if odd else a1
                v2 = a2 >> 16 if odd else a2
                wq = [insert_bits(v0, v1 >> 8, 16, 16),
                      insert_bits(v0 >> 15, ((v1 >> 23) | (v2 << 9)) & m32, 16, 16),
                      insert_bits(((v0 >> 30) | (v1 << 2)) & m32, v2 >> 6, 16, 16)]
                for st in range(8):
                    q = (wq[st // 3] >> (5 * (st % 3))) & 0x001f001f
                    got[r, 64 * g + 8 * fn + st] = q & 0xffff
                    got[r, 64 * g + 8 * fn + 8 + st] = q >> 16
    assert np.array_equal(got, want)


def test_supported_by_bits(raw, monkeypatch):
    a8, a5, a4 = qlinear(256, 64, 8), qlinear(256, 64, 5), qlinear(256, 64, 4)

    def ok(m):
        return raw.supported(m.weight, m.scales, m.biases, m.bits, m.group_size)
    assert raw._bits_of(256, 64) == 8 and raw._bits_of(256, 40) == 5
    assert raw._bits_of(256, 32) == 4 and raw._bits_of(5120, 800) == 5 and raw._bits_of(5120, 1280) == 8
    assert raw._bits_of(256, 48) is None and raw._bits_of(256, 24) is None          # 6-bit, 3-bit
    assert ok(a8) and not ok(a5) and not ok(a4)                 # default: 8-bit only
    assert not raw.bits5()
    monkeypatch.setenv("QWEN_MRQMM5", "1")
    assert ok(a8) and ok(a5) and not ok(a4)
    assert not ok(qlinear(256, 64, 5, mx.bfloat16)) and not ok(qlinear(256, 60, 5))
    raw.set_invariant(True)                                      # never 5-bit in batch-invariant mode
    assert ok(a8) and not ok(a5)
    raw.set_invariant(False)
    monkeypatch.setenv("QWEN_MRQMM5", "0")
    assert ok(a8) and not ok(a5)


def test_install_by_bits(raw, monkeypatch):
    def holder():
        return Holder(a=qlinear(256, 512, 8), b=qlinear(256, 512, 5), c=qlinear(256, 512, 4),
                      d=qlinear(256, 256, 5), e=qlinear(256, 512, 5, mx.bfloat16))

    def kinds(m):
        return "".join(k for k in "abcde" if type(getattr(m, k)) is raw.MRQuantizedLinear)
    m = holder()
    assert raw.install(m, warmup=False) == 1 and kinds(m) == "a"       # knob off: as before
    monkeypatch.setenv("QWEN_MRQMM5", "1")
    m = holder()
    assert raw.install(m, warmup=False) == 2 and kinds(m) == "ab"      # d: N < MIN_N, e: bf16 scales
    assert raw.uninstall(m) == 2 and kinds(m) == ""
    warmed = []
    monkeypatch.setattr(raw, "warm", lambda w, s, b: warmed.append(w.shape))
    m = holder()
    raw.install(m)
    assert sorted(warmed) == [(512, 40), (512, 64)]                   # one per (bits, shape)
    raw.set_invariant(True)
    try:
        m = holder()
        assert raw.install(m, warmup=False) == 1 and kinds(m) == "a"
        assert raw.uncovered(m) == {"QuantizedLinear 5-bit g64": 3, "QuantizedLinear 4-bit g64": 1}
    finally:
        raw.set_invariant(False)


def test_call_finds_the_bit_width(raw, monkeypatch):
    """mrqmm(x, w, ...) takes the kernel of the packed row's width (w words
    per K values); anything else, or scales of another K, is refused."""
    seen = []

    def call(K, N, rows, cfg=None, bits=8):
        seen.append((K, N, rows, bits))
        return (lambda inputs, **kw: [mx.zeros(kw["output_shapes"][0], mx.float16)]), {}
    monkeypatch.setattr(raw, "_call", call)
    x = mx.zeros((4, 1, 256), mx.float16)
    for bits in (8, 5, 4):                 # 4: callable (the bench), never dispatched
        lin = qlinear(256, 64, bits)
        assert raw.mrqmm(x, lin.weight, lin.scales, lin.biases).shape == (4, 1, 64)
    assert seen == [(256, 64, 4, 8), (256, 64, 4, 5), (256, 64, 4, 4)]
    for bits in (6, 3):
        lin = qlinear(256, 64, bits)
        with pytest.raises(ValueError):
            raw.mrqmm(x, lin.weight, lin.scales, lin.biases)
    # an 8-bit weight of K = 320 has the words of a 5-bit K = 512 one: the scales say which
    lin = qlinear(320, 64, 8)
    with pytest.raises(ValueError):
        raw.mrqmm(mx.zeros((4, 1, 512), mx.float16), lin.weight, lin.scales, lin.biases)
    assert len(seen) == 3


def test_config_default_bits(raw):
    """_config's bits argument defaults to 8 (tools call it with three)."""
    for K, N, MB in ((4096, 12288, 1), (4096, 12288, 2), (4096, 1024, 2), (640, 200, 1)):
        assert raw._config(K, N, MB) == raw._config(K, N, MB, 8)


def model5(mr):
    lm = build_lm("dense", 5, mx.float16)
    n = mr.install(lm, warmup=False)
    return lm, n, worker._install_gdn_fuse(lm)


def decode(lm, B, seed=0):
    ids = mx.random.randint(0, 500, (B, 1), key=mx.random.key(seed))
    return lm(ids, cache=lm.make_cache()).logits.astype(mx.float32)


def test_worker_knob_off_leaves_5bit_alone(mr):
    """Default: nothing of a 5-bit model is covered, the fused GDN
    in-projection stays on quantized_matmul, no kernel call at any B, the
    verify setup is off."""
    import types
    lm, n, fused = model5(mr)
    assert n == 0 and fused == sum(1 for x in lm.model.layers if x.is_linear)
    assert all(m.__dict__["_fused_in"][7] is None for x in lm.model.layers if x.is_linear
               for m in [x.linear_attn])
    for B in (1, 2, 5, 16):
        decode(lm, B)
    assert mr.calls == []
    draft = types.SimpleNamespace(prefer_requested_block_size=False)
    assert worker._setup_verify_mrqmm(lm, draft) == "off: nothing covered"


@pytest.mark.parametrize("B", [1, 2, 3, 8, 16])
def test_worker_decode_on_5bit(mr, monkeypatch, B):
    """QWEN_MRQMM5=1: a 3..16-row decode step of a 5-bit model runs every
    linear and the fused GDN in-projection through the kernel at 5 bits;
    B <= 2 never does (the knob-off worker's bits)."""
    ref_lm, n0, _ = model5(mr)                    # knob off: fused GDN on quantized_matmul
    want = decode(ref_lm, B)
    assert n0 == 0 and mr.calls == []
    monkeypatch.setenv("QWEN_MRQMM5", "1")
    lm, n, fused = model5(mr)
    gdn = sum(1 for x in lm.model.layers if x.is_linear)
    assert n > 0 and fused == gdn
    assert all(x.linear_attn.__dict__["_fused_in"][7] is mr for x in lm.model.layers if x.is_linear)
    got = decode(lm, B)
    if B <= 2:
        assert mr.calls == [] and mx.array_equal(got, want).item()
        return
    assert {c[2] for c in mr.calls} == {5} and {c[0][:2] for c in mr.calls} == {(B, 1)}
    a = lm.model.layers[0].linear_attn
    ns = [c[1] for c in mr.calls]
    assert ns.count(sum(a[k].weight.shape[0] for k in worker._GDN_IN)) == gdn
    n_fa = len(lm.model.layers) - gdn
    assert len(ns) == 4 * n_fa + 2 * gdn + 3 * len(lm.model.layers) + 1
    assert mx.allclose(got, want, atol=2e-2, rtol=1e-2).item()


@pytest.mark.parametrize("T", [2, 8, 16])
def test_worker_verify_block_on_5bit(mr, monkeypatch, T):
    """QWEN_MRQMM5=1 + QWEN_VERIFY_MRQMM=1: a DFlash verify block of a 5-bit
    model goes through the kernel as T decode rows (the GDN in-projections
    fused), the drafter's lm_head too; 1..2-token blocks keep the old path."""
    import types
    monkeypatch.setattr(vlinear, "_use_target_verify_dense", lambda *a: False)
    v = speculative_verifier.Qwen3_5BatchInvariantForward
    monkeypatch.setattr(worker, "_VERIFY_LINEAR", v._linear)
    monkeypatch.setattr(worker, "_VERIFY_LINEARS", v._linears)
    monkeypatch.setattr(v, "_linear", worker._verify_linear)
    monkeypatch.setattr(v, "_linears", worker._verify_linears)
    monkeypatch.setattr(worker, "VERIFY_MRQMM", "1")
    monkeypatch.setattr(worker, "VERIFY_MR", True)
    monkeypatch.setattr(worker, "VERIFY_MR_HEAD", True)
    monkeypatch.setattr(worker, "DRAFT_BLOCK", 0)
    monkeypatch.setenv("QWEN_MRQMM5", "1")
    ids = mx.random.randint(0, 500, (1, T), key=mx.random.key(T))
    ref_lm = build_lm("dense", 5, mx.float16)     # nothing installed: the library's modules
    want = ref_lm(ids, cache=ref_lm.make_cache(), speculative_verify=True).logits.astype(mx.float32)
    lm, n, fused = model5(mr)
    draft = types.SimpleNamespace(prefer_requested_block_size=False)
    how = worker._setup_verify_mrqmm(lm, draft)
    assert how.startswith("verify blocks of 3..16 tokens") and worker.DRAFT_BLOCK == 8
    got = lm(ids, cache=lm.make_cache(), speculative_verify=True).logits.astype(mx.float32)
    if T <= 2:
        assert mr.calls == []
    else:
        assert {c[2] for c in mr.calls} == {5} and {c[0][:2] for c in mr.calls} == {(T, 1)}
        gdn = sum(1 for x in lm.model.layers if x.is_linear)
        assert len(mr.calls) == 4 * (len(lm.model.layers) - gdn) + 2 * gdn + 3 * len(lm.model.layers) + 1
    assert mx.allclose(got, want, atol=2e-2, rtol=1e-2).item()
    head = worker._verify_head(lm.lm_head)
    mr.calls.clear()
    head(mx.zeros((1, 7, 128), mx.float16))
    assert mr.calls == [((7, 1, 128), 512, 5)]


def test_batch_invariant_ignores_the_knob(mr, monkeypatch):
    """QWEN_BATCH_INVARIANT=1 with QWEN_MRQMM5=1: a 5-bit model is as
    uncovered as before (max_batch 2, no module swapped, GDN fused onto
    quantized_matmul)."""
    monkeypatch.setenv("QWEN_MRQMM5", "1")
    monkeypatch.setattr(worker, "BATCH_INVARIANT", True)
    mr.set_invariant(True)
    try:
        lm, n, fused = model5(mr)
        assert n == 0 and fused > 0
        assert all(x.linear_attn.__dict__["_fused_in"][7] is None for x in lm.model.layers if x.is_linear)
        assert worker._invariant_check(lm, 16) == 2
        decode(lm, 2), decode(lm, 1)
        assert mr.calls == []
    finally:
        mr.set_invariant(False)
