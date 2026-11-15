"""mrqmm (multi-row 8-bit quantized matmul for batched decode): the Metal
kernel against mx.quantized_matmul and an fp32 reference on synthetic 8-bit
g64 weights (rows 1..16, tile tails, K not a multiple of the K split,
outliers), the dispatch boundaries (rows <= 2 / > 16 / prefill-shaped / bf16 /
other bits stay on quantized_matmul, bit-identical), install() coverage, a
tiny qwen3_5 forward, worker._install_gdn_fuse (and its kernel warm-up), and
worker.sample_rows at the real vocabulary (grouped draw == solo per row).
Needs Metal (skipped otherwise; conftest makes the CPU the default device,
these tests switch to the GPU and back).

The 5-bit kernel (QWEN_MRQMM5, the 27B): the same checks on 5-bit weights,
the unpack of every k position exactly (one-hot x on unit scales), rows
independent of the row count and of their neighbours, the knob (off: 5-bit
modules untouched), a tiny 5-bit model's decode step and the worker's fused
GDN in-projection. The CPU part (dispatch, pinned 8-bit source) is
tests/test_mrqmm_bits.py.

The 4-bit kernel (QWEN_MRQMM4, the 4-bit 27B): the same, at the 27B's shapes
(random weights), plus the load self-test (mrqmm._selftest) on them, lm_head
included.

Real weights (slow, optional): MRQMM_REAL_WEIGHTS=1 pytest tests/test_mrqmm.py
(9B: weights/ornith15-9b-mlx-8bit; 27B: weights/qwen38-27b-mlx-5bit or
MRQMM_WEIGHTS_27B=<dir>; 27B 4-bit: weights/qwen38-27b-mlx-4bit or
MRQMM_WEIGHTS_27B4=<dir>)
"""
import json
import math
import os

import mlx.core as mx
import mlx.nn as nn
import pytest

import mrqmm

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")

W9B = os.path.join(os.path.dirname(__file__), "..", "..", "..", "weights", "ornith15-9b-mlx-8bit")
W27B = os.environ.get("MRQMM_WEIGHTS_27B") or os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "weights", "qwen38-27b-mlx-5bit")
W27B4 = os.environ.get("MRQMM_WEIGHTS_27B4") or os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "weights", "qwen38-27b-mlx-4bit")


@pytest.fixture(autouse=True)
def gpu():
    prev = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(prev)


@pytest.fixture(autouse=True)
def narrow(monkeypatch):
    """The test layers are tiny: let install() take N >= 8 (default MIN_N 512).
    QWEN_MRQMM5 / QWEN_MRQMM4 are off unless a test sets them; the 4-bit
    self-test verdict is per test."""
    monkeypatch.setattr(mrqmm, "MIN_N", 8)
    monkeypatch.delenv("QWEN_MRQMM5", raising=False)
    monkeypatch.delenv("QWEN_MRQMM4", raising=False)
    monkeypatch.setattr(mrqmm, "_TESTED", {})


def qweights(N, K, seed=0, outlier_rows=0, bits=8, dtype=mx.float16):
    mx.random.seed(seed)
    wf = mx.random.normal((N, K)) * K ** -0.5
    if outlier_rows:
        wf[:outlier_rows] = wf[:outlier_rows] * 50
    wq, s, b = mx.quantize(wf.astype(dtype), group_size=64, bits=bits)
    return wq, s.astype(dtype), b.astype(dtype)


def ref32(x, w, s, b, bits=8):
    deq = mx.dequantize(w, s.astype(mx.float32), b.astype(mx.float32), group_size=64, bits=bits)
    return x.astype(mx.float32) @ deq.T


def qmm(x, w, s, b, bits=8):
    return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=bits)


def check_close(y, x, w, s, b, bits=8):
    """Within fp16 output rounding of the fp32 reference, and no worse than
    quantized_matmul itself (+ one fp16 ulp)."""
    ref = ref32(x, w, s, b, bits)
    q = qmm(x, w, s, b, bits).astype(mx.float32)
    y = y.astype(mx.float32)
    assert y.shape == ref.shape
    assert mx.isfinite(y).all().item()
    mref = mx.abs(ref).max().item()
    err = mx.abs(y - ref).max().item()
    qerr = mx.abs(q - ref).max().item()
    ulp = 2.0 ** (math.ceil(math.log2(max(mref, 1e-6))) - 11)
    assert err <= 1.5e-3 * mref + 1e-4, (err, mref)
    assert err <= qerr + ulp, (err, qerr, ulp)


SHAPES = [(128, 64), (256, 200), (640, 1032), (4096, 1024), (2048, 8)]


@pytest.mark.parametrize("K,N", SHAPES)
@pytest.mark.parametrize("M", list(range(1, 17)))
def test_kernel_matches_reference(K, N, M):
    w, s, b = qweights(N, K, seed=K + N)
    x = (mx.random.normal((M, 1, K)) * 0.5).astype(mx.float16)
    check_close(mrqmm.mrqmm(x, w, s, b), x, w, s, b)


@pytest.mark.parametrize("cfg", [(1, 1), (2, 2), (4, 4), (2, 8), (4, 8), (1, 8)])
@pytest.mark.parametrize("M", [3, 8, 9, 16])
def test_every_config_with_tails(cfg, M):
    # N = 8 * 25: every NA leaves a partial tile; K = 640: 10 groups, not a multiple of KS
    K, N = 640, 200
    w, s, b = qweights(N, K, seed=3)
    x = (mx.random.normal((M, 1, K))).astype(mx.float16)
    check_close(mrqmm.mrqmm(x, w, s, b, cfg=cfg), x, w, s, b)


@pytest.mark.parametrize("M", [3, 7, 16])
def test_outliers(M):
    K, N = 1024, 512
    w, s, b = qweights(N, K, seed=5, outlier_rows=4)
    x = mx.random.normal((M, 1, K))
    x[..., ::97] = x[..., ::97] * 40.0
    x[0, 0, :8] = 200.0
    x = x.astype(mx.float16)
    check_close(mrqmm.mrqmm(x, w, s, b), x, w, s, b)


def test_too_many_rows():
    w, s, b = qweights(64, 128)
    with pytest.raises(ValueError):
        mrqmm.mrqmm(mx.zeros((17, 1, 128), mx.float16), w, s, b)
    with pytest.raises(ValueError):               # the kernel reads x as fp16
        mrqmm.mrqmm(mx.zeros((4, 1, 128), mx.bfloat16), w, s, b)


def test_wrong_k_after_a_cached_call():
    """The call cache is keyed by the weight's shape: a second weight with the
    same N but another K is checked, not read out of bounds."""
    w, s, b = qweights(64, 128)
    mx.eval(mrqmm.mrqmm(mx.zeros((4, 1, 128), mx.float16), w, s, b))
    w2, s2, b2 = qweights(64, 256)
    with pytest.raises(ValueError):
        mrqmm.mrqmm(mx.zeros((4, 1, 128), mx.float16), w2, s2, b2)


def test_strided_and_view_inputs():
    """A strided x (copied by metal_kernel) and a contiguous view that does not
    start at the buffer's first row give the same result as a fresh copy."""
    K, N = 512, 256
    w, s, b = qweights(N, K, seed=9)
    big = (mx.random.normal((7, 3, K))).astype(mx.float16)
    flat = (mx.random.normal((11, 1, K))).astype(mx.float16)
    mx.eval(big, flat)
    for x in (big[:, 1:2, :], big[2:6, 2:3, :], flat[3:10], flat[5:8]):
        y = mrqmm.mrqmm(x, w, s, b)
        assert mx.array_equal(y, mrqmm.mrqmm(mx.array(x.tolist(), dtype=mx.float16), w, s, b)).item()
        check_close(y, x, w, s, b)


def make_linear(K, N, bits=8, dtype=mx.float16, bias=False):
    lin = nn.QuantizedLinear(K, N, bias=bias, group_size=64, bits=bits)
    w, s, b = qweights(N, K, seed=N, bits=bits, dtype=dtype)
    lin.weight, lin.scales, lin.biases = w, s, b
    if bias:
        lin.bias = mx.random.normal((N,)).astype(dtype)
    mx.eval(lin.parameters())
    return lin


class Holder(nn.Module):
    def __init__(self, **mods):
        super().__init__()
        for k, v in mods.items():
            setattr(self, k, v)


@pytest.mark.parametrize("M", list(range(1, 18)))
def test_dispatch_boundaries(M):
    K, N = 256, 128
    lin = make_linear(K, N)
    x = mx.random.normal((M, 1, K)).astype(mx.float16)
    before = lin(x)
    assert mrqmm.install(Holder(l=lin), warmup=False) == 1
    assert type(lin) is mrqmm.MRQuantizedLinear
    after = lin(x)
    if 3 <= M <= 16:
        assert mx.array_equal(after, mrqmm.mrqmm(x, lin.weight, lin.scales, lin.biases)).item()
        check_close(after, x, lin.weight, lin.scales, lin.biases)
    else:   # M <= 2 (bit-identical to solo) and M > 16: quantized_matmul untouched
        assert mx.array_equal(after, before).item()
    # prefill-shaped (1, M, K) and bf16 inputs never take the kernel
    xs = mx.random.normal((1, M, K)).astype(mx.float16)
    assert mx.array_equal(lin(xs), qmm(xs, lin.weight, lin.scales, lin.biases)).item()
    mrqmm.uninstall(Holder(l=lin))
    assert type(lin) is nn.QuantizedLinear


def test_bias_and_bf16_input():
    lin = make_linear(256, 64, bias=True)
    x = mx.random.normal((5, 1, 256)).astype(mx.float16)
    ref = lin(x)
    mrqmm.install(Holder(l=lin), warmup=False)
    y = lin(x)
    assert mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max().item() < 5e-3
    xb = x.astype(mx.bfloat16)          # not fp16: falls back (and is not an error)
    assert lin(xb).dtype == qmm(xb, lin.weight, lin.scales, lin.biases).dtype


def test_install_skips_unsupported(monkeypatch):
    m = Holder(a=make_linear(256, 64), b=make_linear(256, 64, bits=4), c=make_linear(256, 64, bits=5),
               d=make_linear(256, 64, bits=6), e=make_linear(256, 64, bits=3),
               f=make_linear(256, 64, dtype=mx.bfloat16), g=nn.Linear(256, 64))
    assert mrqmm.install(m, warmup=True) == 1
    assert type(m.a) is mrqmm.MRQuantizedLinear
    for k in "bcdef":
        assert type(getattr(m, k)) is nn.QuantizedLinear, k
    assert type(m.g) is nn.Linear
    m2 = Holder(b=make_linear(256, 64, bits=4))
    assert mrqmm.install(m2) == 0                      # 27B (5-bit) / 35B experts: no-op
    monkeypatch.setenv("QWEN_MRQMM", "0")
    m3 = Holder(a=make_linear(256, 64))
    assert mrqmm.install(m3) == 0 and type(m3.a) is nn.QuantizedLinear


def test_min_n(monkeypatch):
    """Default: layers narrower than 512 outputs (the 35B MoE router) stay
    on quantized_matmul."""
    monkeypatch.setattr(mrqmm, "MIN_N", 512)
    m = Holder(a=make_linear(256, 256), b=make_linear(256, 512))
    assert mrqmm.install(m, warmup=False) == 1
    assert type(m.a) is nn.QuantizedLinear and type(m.b) is mrqmm.MRQuantizedLinear
    mrqmm.uninstall(m)


def tiny_lm(wbits=8):
    """Tiny qwen3_5 text model whose GDN in-projections are all 8-bit
    (linear_num_value_heads 8: in_proj_b/a have N = 8), so the worker's fused
    decode in-projection takes it too."""
    from conftest import VCFG
    from mlx_vlm.models.qwen3_5 import language as q35
    from mlx_vlm.models.qwen3_5.config import TextConfig
    mx.random.seed(0)
    lm = q35.LanguageModel(TextConfig(
        model_type="qwen3_5_text", intermediate_size=256, hidden_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64, linear_num_value_heads=8,
        linear_num_key_heads=2, linear_key_head_dim=64, linear_value_head_dim=32,
        linear_conv_kernel_dim=4, rms_norm_eps=1e-6, vocab_size=512, max_position_embeddings=4096,
        full_attention_interval=4,
        rope_parameters={"type": "default", "mrope_section": [3, 3, 2], "rope_theta": 100000,
                         "partial_rotary_factor": 0.25}), VCFG)
    nn.quantize(lm, group_size=64, bits=wbits,
                class_predicate=lambda p, m: hasattr(m, "to_quantized")
                and getattr(m, "weight", mx.zeros((1, 64))).shape[-1] % 64 == 0)
    lm.apply(lambda a: a.astype(mx.float16) if a.dtype in (mx.float32, mx.bfloat16) else a)
    mx.eval(lm.parameters())
    return lm


@pytest.mark.parametrize("B", [2, 5, 16])
def test_tiny_model_decode_step(B):
    lm = tiny_lm()
    ids = mx.random.randint(0, 512, (B, 1))
    ref = lm(ids, cache=lm.make_cache()).logits.astype(mx.float32)
    n = mrqmm.install(lm)
    assert n > 0
    calls = []
    orig = mrqmm.mrqmm
    mrqmm.mrqmm = lambda x, w, *a, **k: calls.append(w.shape[0]) or orig(x, w, *a, **k)
    try:
        y = lm(ids, cache=lm.make_cache()).logits.astype(mx.float32)
    finally:
        mrqmm.mrqmm = orig
        mrqmm.uninstall(lm)
    if B < 3:
        assert not calls and mx.array_equal(y, ref).item()
    else:
        # 3 GDN layers: qkv / z / b / a + out_proj (the library calls them one by
        # one; worker._install_gdn_fuse fuses them, test_worker_gdn_fuse); 1
        # attention layer: q/k/v/o; 4 MLPs: gate/up/down; lm_head.
        assert len(calls) == 3 * (4 + 1) + 4 + 4 * 3 + 1, calls
        assert mx.abs(y - ref).max().item() <= 2e-2 * max(1.0, mx.abs(ref).max().item())
        assert (y.argmax(-1) == ref.argmax(-1)).all().item()


@pytest.mark.skipif(os.environ.get("MRQMM_REAL_WEIGHTS") != "1" or not os.path.isdir(W9B),
                    reason="slow: set MRQMM_REAL_WEIGHTS=1 (needs weights/ornith15-9b-mlx-8bit)")
def test_real_9b_weights():
    """Layer 2 (GDN + MLP), layer 3 (attention) and lm_head of the 9B, loaded
    lazily from the safetensors (~1.5 GB, not the model)."""
    idx = json.load(open(os.path.join(W9B, "model.safetensors.index.json")))["weight_map"]
    files = {}

    def lin(prefix):
        out = []
        for p in ("weight", "scales", "biases"):
            f = idx[f"{prefix}.{p}"]
            files.setdefault(f, mx.load(os.path.join(W9B, f)))
            a = files[f][f"{prefix}.{p}"]
            out.append(a if p == "weight" else a.astype(mx.float16))
        return out

    P = "language_model.model.layers."
    names = [P + "3.self_attn." + n for n in ("q_proj", "k_proj", "v_proj", "o_proj")]
    names += [P + "2.linear_attn.out_proj"] + [P + "2.mlp." + n for n in ("gate_proj", "up_proj", "down_proj")]
    names += ["language_model.lm_head"]
    ws = [lin(n) for n in names]
    gdn = [lin(P + "2.linear_attn." + p) for p in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")]
    ws.append([mx.concatenate([g[i] for g in gdn]) for i in range(3)])
    mx.random.seed(1)
    for w, s, b in ws:
        assert mrqmm.supported(w, s, b, 8, 64)
        K = w.shape[1] * 4
        for M in (3, 8, 9, 16):
            x = mx.random.normal((M, 1, K))
            x[..., ::211] = x[..., ::211] * 30.0
            x = x.astype(mx.float16)
            check_close(mrqmm.mrqmm(x, w, s, b), x, w, s, b)


@pytest.mark.parametrize("B", [1, 2, 5, 16])
def test_worker_gdn_fuse(B):
    """worker._install_gdn_fuse (QWEN_GDN_FUSE): one matmul for the four GDN
    in-projections of a decode step over row views of their concatenation.
    qmv rows (B <= 2) are bit-identical to the unfused model; 3..16 rows take
    mrqmm over the whole N = 784 (b/a ride along); prefill is unchanged."""
    import worker
    lm = tiny_lm()
    ids = mx.random.randint(0, 512, (B, 1))
    pre = mx.random.randint(0, 512, (1, 9))
    ref = lm(ids, cache=lm.make_cache()).logits.astype(mx.float32)
    ref_pre = lm(pre, cache=lm.make_cache()).logits
    mrqmm.install(lm)
    for k in [k for k in mrqmm._KERNELS if k[1] == 784]:      # the fused N = 512 + 256 + 8 + 8
        del mrqmm._KERNELS[k]
    for k in [k for k in mrqmm._CALLS if k[2][0] == 784]:
        del mrqmm._CALLS[k]
    try:
        assert worker._install_gdn_fuse(lm) == 3
        # compiled at install for both row buckets, not in the first batched step
        assert {k[2] for k in mrqmm._KERNELS if k[:2] == (128, 784)} == {1, 2}
        calls = []
        orig = mrqmm.mrqmm
        mrqmm.mrqmm = lambda x, w, *a, **k: calls.append(w.shape[0]) or orig(x, w, *a, **k)
        try:
            y = lm(ids, cache=lm.make_cache()).logits.astype(mx.float32)
        finally:
            mrqmm.mrqmm = orig
        assert mx.array_equal(lm(pre, cache=lm.make_cache()).logits, ref_pre).item()
    finally:
        mrqmm.uninstall(lm)
    if B < 3:
        assert not calls and mx.array_equal(y, ref).item()
    else:
        assert calls.count(784) == 3, calls
        assert mx.abs(y - ref).max().item() <= 2e-2 * max(1.0, mx.abs(ref).max().item())
        assert (y.argmax(-1) == ref.argmax(-1)).all().item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_sample_rows_full_vocab_stack_equals_solo(dtype):
    """worker.sample_rows on Metal at the real vocabulary: a (16, 248320)
    stack takes MLX's multi-block sort and looped softmax / logsumexp kernels
    (V=512 on the CPU suite does not); every row's filtered logits and token
    must be bit-identical to that row alone. Ties injected (equal maxima, a
    tied top-k boundary, a flat tail), seeded, over steps."""
    import worker
    V, B = 248320, 16
    cfgs = [dict(temperature=0.0), dict(temperature=0.6, top_p=0.95, top_k=20),
            dict(temperature=0.7, top_p=0.9), dict(temperature=0.8, min_p=0.05),
            dict(temperature=1.0, top_k=50)]
    for c in cfgs:
        sampling = dict(worker.SAMPLING_KWARGS, **c)
        solo = [worker.make_sampler(sampling, seed=100 + i) for i in range(B)]
        grouped = [worker.make_sampler(sampling, seed=100 + i) for i in range(B)]
        for step in range(3):
            lg = mx.random.normal((B, V), key=mx.random.key(17 * step + 1)) * 4
            lg[:, 7] = lg.max(axis=-1)                       # ties with the maximum
            lg[:4, 1000:1064] = 9.0                          # 64 equal top logits
            lg[4:8, 5:30] = lg[4:8, 5:6]                     # a tied block around the top-k cut
            lg[8:, V - 4096:] = -3.0                         # a flat tail
            lg = lg.astype(dtype)
            xs = [lg[i:i + 1] for i in range(B)]
            if not grouped[0].greedy:
                stack = grouped[0].filtered(mx.concatenate(xs))
                for i in range(B):
                    assert mx.array_equal(stack[i:i + 1], solo[i].filtered(xs[i])).item(), (c, step, i)
            want = [solo[i](xs[i]).tolist() for i in range(B)]
            got = [y.tolist() for y in worker.sample_rows(grouped, xs)]
            assert got == want, (c, step)


# --- 5 bits (QWEN_MRQMM5) -----------------------------------------------------

SHAPES5 = [(128, 64), (256, 200), (640, 1032), (5120, 1024), (2048, 8)]


@pytest.mark.parametrize("K,N", SHAPES5)
@pytest.mark.parametrize("M", [1, 2, 3, 5, 8, 9, 13, 16])
def test_kernel5_matches_reference(K, N, M):
    w, s, b = qweights(N, K, seed=K + N, bits=5)
    assert w.shape == (N, K * 5 // 32)
    x = (mx.random.normal((M, 1, K)) * 0.5).astype(mx.float16)
    check_close(mrqmm.mrqmm(x, w, s, b), x, w, s, b, bits=5)


@pytest.mark.parametrize("N", [64, 200])
def test_kernel5_unpacks_every_k(N):
    """Unit scales, zero biases, one-hot x rows: y[m, n] is the 5-bit weight
    of (n, k_m) itself, exactly, for every k of a 192-wide row (3 groups: KS
    does not divide them) and every row slot of both buckets."""
    K = 192
    w, s, b = qweights(N, K, seed=N, bits=5)
    one, zero = mx.ones_like(s), mx.zeros_like(b)
    q = mx.dequantize(w, one.astype(mx.float32), zero.astype(mx.float32), group_size=64, bits=5)
    assert q.min().item() == 0 and q.max().item() == 31
    for M in (8, 16, 5):
        for k0 in range(0, K, M):
            ks = [(k0 + i) % K for i in range(M)]
            x = mx.zeros((M, 1, K), mx.float16)
            for m, k in enumerate(ks):
                x[m, 0, k] = 1.0
            y = mrqmm.mrqmm(x, w, one, zero).reshape(M, N)
            assert mx.array_equal(y.astype(mx.float32), q[:, ks].T).item(), (M, k0)


@pytest.mark.parametrize("cfg", [(1, 1), (2, 2), (4, 4), (2, 8), (4, 8), (1, 8)])
@pytest.mark.parametrize("M", [3, 8, 9, 16])
def test_kernel5_every_config_with_tails(cfg, M):
    K, N = 640, 200
    w, s, b = qweights(N, K, seed=3, bits=5)
    x = (mx.random.normal((M, 1, K))).astype(mx.float16)
    check_close(mrqmm.mrqmm(x, w, s, b, cfg=cfg), x, w, s, b, bits=5)


@pytest.mark.parametrize("M", [3, 7, 16])
def test_kernel5_outliers(M):
    K, N = 1024, 512
    w, s, b = qweights(N, K, seed=5, outlier_rows=4, bits=5)
    x = mx.random.normal((M, 1, K))
    x[..., ::97] = x[..., ::97] * 40.0
    x[0, 0, :8] = 200.0
    x = x.astype(mx.float16)
    check_close(mrqmm.mrqmm(x, w, s, b), x, w, s, b, bits=5)


@pytest.mark.parametrize("K,N", [(5120, 4096), (640, 200)])
def test_kernel5_rows_do_not_depend_on_the_batch(K, N):
    """A row's bits are the same at every row count 1..16 (both row buckets
    split K over the same KS at 5 bits, mrqmm._config) and whatever the other
    rows hold."""
    w, s, b = qweights(N, K, seed=11, bits=5)
    X = mx.random.normal((16, 1, K)).astype(mx.float16)
    Z = mx.random.normal((16, 1, K)).astype(mx.float16)
    assert mrqmm._config(K, N, 1, 5)[1] == mrqmm._config(K, N, 2, 5)[1]
    for rows in ((1, 2, 3, 4, 6, 8, 9, 12, 16),):
        base = mrqmm.mrqmm(X[:rows[-1]], w, s, b)
        for M in rows:
            if M < 3:
                assert mx.array_equal(mrqmm.mrqmm(X[:M], w, s, b), base[:M]).item(), M
                continue
            y = mrqmm.mrqmm(X[:M], w, s, b)
            assert mx.array_equal(y, base[:M]).item(), M
            y2 = mrqmm.mrqmm(mx.concatenate([Z[:1], X[1:2], Z[2:M]]), w, s, b)
            assert mx.array_equal(y2[1], base[1]).item(), M


@pytest.mark.parametrize("M", list(range(1, 18)))
def test_dispatch_boundaries5(M, monkeypatch):
    K, N = 256, 128
    lin = make_linear(K, N, bits=5)
    x = mx.random.normal((M, 1, K)).astype(mx.float16)
    before = lin(x)
    assert mrqmm.install(Holder(l=lin), warmup=False) == 0          # knob off: untouched
    assert type(lin) is nn.QuantizedLinear
    monkeypatch.setenv("QWEN_MRQMM5", "1")
    assert mrqmm.install(Holder(l=lin), warmup=False) == 1
    assert type(lin) is mrqmm.MRQuantizedLinear
    after = lin(x)
    if 3 <= M <= 16:
        assert mx.array_equal(after, mrqmm.mrqmm(x, lin.weight, lin.scales, lin.biases)).item()
        check_close(after, x, lin.weight, lin.scales, lin.biases, bits=5)
    else:   # M <= 2 (bit-identical to solo) and M > 16: quantized_matmul untouched
        assert mx.array_equal(after, before).item()
    xs = mx.random.normal((1, M, K)).astype(mx.float16)
    assert mx.array_equal(lin(xs), qmm(xs, lin.weight, lin.scales, lin.biases, bits=5)).item()
    mrqmm.uninstall(Holder(l=lin))
    assert type(lin) is nn.QuantizedLinear


def test_install5_knob_and_invariant(monkeypatch):
    def holder():
        return Holder(a=make_linear(256, 64), c=make_linear(256, 64, bits=5),
                      f=make_linear(256, 64, bits=5, dtype=mx.bfloat16), b=make_linear(256, 64, bits=4))
    monkeypatch.setenv("QWEN_MRQMM5", "1")
    m = holder()
    assert mrqmm.install(m, warmup=True) == 2
    assert type(m.a) is type(m.c) is mrqmm.MRQuantizedLinear
    assert type(m.f) is type(m.b) is nn.QuantizedLinear
    assert {k[5] for k in mrqmm._KERNELS if k[:2] == (256, 64)} == {8, 5}      # warmed per bit width
    mrqmm.set_invariant(True)
    try:
        m = holder()
        assert mrqmm.install(m, warmup=False) == 1 and type(m.c) is nn.QuantizedLinear
    finally:
        mrqmm.set_invariant(False)
        mrqmm.uninstall(m)


@pytest.mark.parametrize("B", [2, 5, 16])
def test_tiny_model_decode_step5(B, monkeypatch):
    lm = tiny_lm(wbits=5)
    ids = mx.random.randint(0, 512, (B, 1))
    ref = lm(ids, cache=lm.make_cache()).logits.astype(mx.float32)
    assert mrqmm.install(lm) == 0
    monkeypatch.setenv("QWEN_MRQMM5", "1")
    assert mrqmm.install(lm) > 0
    calls = []
    orig = mrqmm.mrqmm
    mrqmm.mrqmm = lambda x, w, *a, **k: calls.append((x.shape[-1], w.shape)) or orig(x, w, *a, **k)
    try:
        y = lm(ids, cache=lm.make_cache()).logits.astype(mx.float32)
    finally:
        mrqmm.mrqmm = orig
        mrqmm.uninstall(lm)
    if B < 3:
        assert not calls and mx.array_equal(y, ref).item()
    else:
        assert len(calls) == 3 * (4 + 1) + 4 + 4 * 3 + 1, calls
        assert all(mrqmm._bits_of(K, ws[1]) == 5 for K, ws in calls)
        assert mx.abs(y - ref).max().item() <= 3e-2 * max(1.0, mx.abs(ref).max().item())
        assert (y.argmax(-1) == ref.argmax(-1)).all().item()


@pytest.mark.parametrize("B", [1, 2, 5, 16])
def test_worker_gdn_fuse5(B, monkeypatch):
    """worker._install_gdn_fuse on a 5-bit model with QWEN_MRQMM5=1: the fused
    in-projection (N = 784) takes the 5-bit kernel for 3..16 rows, its kernels
    compiled at install; B <= 2 is bit-identical to the unfused model."""
    import worker
    monkeypatch.setenv("QWEN_MRQMM5", "1")
    lm = tiny_lm(wbits=5)
    ids = mx.random.randint(0, 512, (B, 1))
    ref = lm(ids, cache=lm.make_cache()).logits.astype(mx.float32)
    mrqmm.install(lm)
    for k in [k for k in mrqmm._KERNELS if k[1] == 784 and k[5] == 5]:
        del mrqmm._KERNELS[k]
    for k in [k for k in mrqmm._CALLS if k[2] == (784, 20)]:
        del mrqmm._CALLS[k]
    try:
        assert worker._install_gdn_fuse(lm) == 3
        assert all(x.linear_attn.__dict__["_fused_in"][7] is mrqmm for x in lm.model.layers if x.is_linear)
        assert {k[2] for k in mrqmm._KERNELS if k[:2] == (128, 784) and k[5] == 5} == {1, 2}
        calls = []
        orig = mrqmm.mrqmm
        mrqmm.mrqmm = lambda x, w, *a, **k: calls.append(w.shape) or orig(x, w, *a, **k)
        try:
            y = lm(ids, cache=lm.make_cache()).logits.astype(mx.float32)
        finally:
            mrqmm.mrqmm = orig
    finally:
        mrqmm.uninstall(lm)
    if B < 3:
        assert not calls and mx.array_equal(y, ref).item()
    else:
        assert calls.count((784, 20)) == 3, calls
        assert mx.abs(y - ref).max().item() <= 3e-2 * max(1.0, mx.abs(ref).max().item())
        assert (y.argmax(-1) == ref.argmax(-1)).all().item()


@pytest.mark.skipif(os.environ.get("MRQMM_REAL_WEIGHTS") != "1" or not os.path.isdir(W27B),
                    reason="slow: set MRQMM_REAL_WEIGHTS=1 (needs weights/qwen38-27b-mlx-5bit "
                           "or MRQMM_WEIGHTS_27B)")
def test_real_27b_weights():
    """Layer 2 (GDN + MLP), layer 3 (attention) and lm_head of the 27B, loaded
    lazily from the safetensors (~1.3 GB, not the model); scales / biases cast
    bf16 -> fp16 as the worker does."""
    real_27b(W27B, 5)


@pytest.mark.skipif(os.environ.get("MRQMM_REAL_WEIGHTS") != "1" or not os.path.isdir(W27B4),
                    reason="slow: set MRQMM_REAL_WEIGHTS=1 (needs weights/qwen38-27b-mlx-4bit "
                           "or MRQMM_WEIGHTS_27B4)")
def test_real_27b_4bit_weights():
    """test_real_27b_weights on the 4-bit 27B, + the load self-test per shape."""
    real_27b(W27B4, 4)


def real_27b(W27B, bits):
    idx = json.load(open(os.path.join(W27B, "model.safetensors.index.json")))["weight_map"]
    files = {}

    def lin(prefix):
        out = []
        for p in ("weight", "scales", "biases"):
            f = idx[f"{prefix}.{p}"]
            files.setdefault(f, mx.load(os.path.join(W27B, f)))
            a = files[f][f"{prefix}.{p}"]
            out.append(a if p == "weight" else a.astype(mx.float16))
        return out

    P = "language_model.model.layers."
    names = [P + "3.self_attn." + n for n in ("q_proj", "k_proj", "v_proj", "o_proj")]
    names += [P + "2.linear_attn.out_proj"] + [P + "2.mlp." + n for n in ("gate_proj", "up_proj", "down_proj")]
    names += ["language_model.lm_head"]
    ws = [lin(n) for n in names]
    gdn = [lin(P + "2.linear_attn." + p) for p in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")]
    ws.append([mx.concatenate([g[i] for g in gdn]) for i in range(3)])
    mx.random.seed(1)
    for w, s, b in ws:
        K = w.shape[1] * 32 // bits
        assert mrqmm._bits_of(K, w.shape[1]) == bits and s.shape == (w.shape[0], K // 64)
        if bits in mrqmm.SELFTEST_BITS:
            assert mrqmm._selftest(w, s, b, bits) is None
        for M in (3, 8, 9, 16):
            x = mx.random.normal((M, 1, K))
            x[..., ::211] = x[..., ::211] * 30.0
            x = x.astype(mx.float16)
            y = mrqmm.mrqmm(x, w, s, b)
            for lo in range(0, w.shape[0], 32768):         # the fp32 reference in chunks (lm_head)
                hi = lo + 32768
                check_close(y[..., lo:hi], x, w[lo:hi], s[lo:hi], b[lo:hi], bits=bits)


# --- 4 bits (kernel only: never dispatched, scripts/perf/dflash2/kern5_bench.py --bits 4) ---

@pytest.mark.parametrize("K,N", [(128, 64), (640, 200), (5120, 1024)])
@pytest.mark.parametrize("M", [1, 3, 8, 9, 16])
def test_kernel4_matches_reference(K, N, M):
    w, s, b = qweights(N, K, seed=K + N, bits=4)
    assert w.shape == (N, K // 8)
    x = (mx.random.normal((M, 1, K)) * 0.5).astype(mx.float16)
    check_close(mrqmm.mrqmm(x, w, s, b), x, w, s, b, bits=4)


def test_kernel4_unpacks_every_k():
    K, N = 192, 200
    w, s, b = qweights(N, K, seed=N, bits=4)
    one, zero = mx.ones_like(s), mx.zeros_like(b)
    q = mx.dequantize(w, one.astype(mx.float32), zero.astype(mx.float32), group_size=64, bits=4)
    assert q.min().item() == 0 and q.max().item() == 15
    for M in (8, 16):
        for k0 in range(0, K, M):
            x = mx.zeros((M, 1, K), mx.float16)
            for m in range(M):
                x[m, 0, k0 + m] = 1.0
            y = mrqmm.mrqmm(x, w, one, zero).reshape(M, N)
            assert mx.array_equal(y.astype(mx.float32), q[:, k0:k0 + M].T).item(), (M, k0)


def test_4bit_needs_its_knob(monkeypatch):
    """QWEN_MRQMM5 does not take 4-bit modules; QWEN_MRQMM4 does (self-tested
    first), and not in batch-invariant mode."""
    monkeypatch.setenv("QWEN_MRQMM5", "1")
    m = Holder(b=make_linear(256, 64, bits=4))
    assert mrqmm.install(m, warmup=False) == 0 and type(m.b) is nn.QuantizedLinear
    assert not mrqmm.supported(m.b.weight, m.b.scales, m.b.biases, 4, 64) and mrqmm._TESTED == {}
    monkeypatch.setenv("QWEN_MRQMM4", "1")
    assert mrqmm.install(m, warmup=True) == 1 and type(m.b) is mrqmm.MRQuantizedLinear
    assert mrqmm._TESTED[4][0], mrqmm._TESTED
    mrqmm.uninstall(m)
    mrqmm.set_invariant(True)
    try:
        assert mrqmm.install(m, warmup=False) == 0
    finally:
        mrqmm.set_invariant(False)


# the 4-bit 27B (weights/qwen38-27b-mlx-4bit): (K, N) of every linear a decode
# step runs; the fused GDN in-projection is qkv + z + b + a = 10240 + 6144 + 48 + 48
SHAPES27 = [(5120, 17408), (17408, 5120), (5120, 16480), (6144, 5120), (5120, 12288), (5120, 1024)]


@pytest.mark.parametrize("K,N", SHAPES27)
@pytest.mark.parametrize("M", [3, 4, 8, 9, 16])
def test_kernel4_27b_shapes(K, N, M):
    w, s, b = qweights(N, K, seed=K + N + M, bits=4)
    x = mx.random.normal((M, 1, K))
    x[..., ::211] = x[..., ::211] * 30.0
    x = x.astype(mx.float16)
    check_close(mrqmm.mrqmm(x, w, s, b), x, w, s, b, bits=4)


@pytest.mark.parametrize("K,N", SHAPES27 + [(5120, 248320)])
def test_selftest4_27b_shapes(K, N):
    """The load self-test passes at every 27B shape, lm_head included (random
    packed words: every 4-bit value occurs)."""
    mx.random.seed(N)
    w = mx.random.randint(0, 256, (N, K // 2), dtype=mx.uint8).view(mx.uint32)
    s = (mx.random.uniform(shape=(N, K // 64)) * 2e-3 + 1e-4).astype(mx.float16)
    b = (mx.random.normal((N, K // 64)) * 3e-2).astype(mx.float16)
    assert mrqmm._selftest(w, s, b, 4) is None


@pytest.mark.parametrize("K,N", [(5120, 4096), (640, 200)])
def test_kernel4_rows_do_not_depend_on_the_batch(K, N):
    """As at 5 bits: KS 8 in both row buckets, a row's bits are the same at
    every row count 3..16 and whatever sits beside it."""
    w, s, b = qweights(N, K, seed=13, bits=4)
    X = mx.random.normal((16, 1, K)).astype(mx.float16)
    Z = mx.random.normal((16, 1, K)).astype(mx.float16)
    assert mrqmm._config(K, N, 1, 4)[1] == mrqmm._config(K, N, 2, 4)[1]
    base = mrqmm.mrqmm(X, w, s, b)
    for M in (3, 4, 6, 8, 9, 12, 16):
        assert mx.array_equal(mrqmm.mrqmm(X[:M], w, s, b), base[:M]).item(), M
        y2 = mrqmm.mrqmm(mx.concatenate([Z[:1], X[1:2], Z[2:M]]), w, s, b)
        assert mx.array_equal(y2[1], base[1]).item(), M


@pytest.mark.parametrize("M", list(range(1, 18)))
def test_dispatch_boundaries4(M, monkeypatch):
    K, N = 256, 128
    lin = make_linear(K, N, bits=4)
    x = mx.random.normal((M, 1, K)).astype(mx.float16)
    before = lin(x)
    assert mrqmm.install(Holder(l=lin), warmup=False) == 0          # knob off: untouched
    monkeypatch.setenv("QWEN_MRQMM4", "1")
    assert mrqmm.install(Holder(l=lin), warmup=False) == 1
    after = lin(x)
    if 3 <= M <= 16:
        assert mx.array_equal(after, mrqmm.mrqmm(x, lin.weight, lin.scales, lin.biases)).item()
        check_close(after, x, lin.weight, lin.scales, lin.biases, bits=4)
    else:
        assert mx.array_equal(after, before).item()
    xs = mx.random.normal((1, M, K)).astype(mx.float16)
    assert mx.array_equal(lin(xs), qmm(xs, lin.weight, lin.scales, lin.biases, bits=4)).item()
    mrqmm.uninstall(Holder(l=lin))


@pytest.mark.parametrize("B", [2, 5, 16])
def test_tiny_model_decode_step4(B, monkeypatch):
    lm = tiny_lm(wbits=4)
    ids = mx.random.randint(0, 512, (B, 1))
    ref = lm(ids, cache=lm.make_cache()).logits.astype(mx.float32)
    assert mrqmm.install(lm) == 0
    monkeypatch.setenv("QWEN_MRQMM4", "1")
    assert mrqmm.install(lm) > 0 and mrqmm._TESTED[4][0]
    calls = []
    orig = mrqmm.mrqmm
    mrqmm.mrqmm = lambda x, w, *a, **k: calls.append((x.shape[-1], w.shape)) or orig(x, w, *a, **k)
    try:
        y = lm(ids, cache=lm.make_cache()).logits.astype(mx.float32)
    finally:
        mrqmm.mrqmm = orig
        mrqmm.uninstall(lm)
    if B < 3:
        assert not calls and mx.array_equal(y, ref).item()
    else:
        assert len(calls) == 3 * (4 + 1) + 4 + 4 * 3 + 1, calls
        assert all(mrqmm._bits_of(K, ws[1]) == 4 for K, ws in calls)
        assert mx.abs(y - ref).max().item() <= 3e-2 * max(1.0, mx.abs(ref).max().item())
        assert (y.argmax(-1) == ref.argmax(-1)).all().item()


@pytest.mark.parametrize("B", [1, 2, 5, 16])
def test_worker_gdn_fuse4(B, monkeypatch):
    """worker._install_gdn_fuse on a 4-bit model with QWEN_MRQMM4=1: the fused
    in-projection takes the 4-bit kernel for 3..16 rows; B <= 2 is
    bit-identical to the unfused model."""
    import worker
    monkeypatch.setenv("QWEN_MRQMM4", "1")
    lm = tiny_lm(wbits=4)
    ids = mx.random.randint(0, 512, (B, 1))
    ref = lm(ids, cache=lm.make_cache()).logits.astype(mx.float32)
    mrqmm.install(lm)
    try:
        assert worker._install_gdn_fuse(lm) == 3
        assert all(x.linear_attn.__dict__["_fused_in"][7] is mrqmm for x in lm.model.layers if x.is_linear)
        calls = []
        orig = mrqmm.mrqmm
        mrqmm.mrqmm = lambda x, w, *a, **k: calls.append(w.shape) or orig(x, w, *a, **k)
        try:
            y = lm(ids, cache=lm.make_cache()).logits.astype(mx.float32)
        finally:
            mrqmm.mrqmm = orig
    finally:
        mrqmm.uninstall(lm)
    if B < 3:
        assert not calls and mx.array_equal(y, ref).item()
    else:
        assert calls.count((784, 16)) == 3, calls
        assert mx.abs(y - ref).max().item() <= 3e-2 * max(1.0, mx.abs(ref).max().item())
        assert (y.argmax(-1) == ref.argmax(-1)).all().item()


# The wide variant (mrqmm_wide, QWEN_DRAFT_MRQMM: the fp32 DFlash2 drafter).
@pytest.mark.parametrize("bits", [4, 5, 8])
@pytest.mark.parametrize("M", [1, 3, 8, 9, 16])
def test_wide_equals_narrow_sums_scaled(bits, M):
    """mrqmm_wide = the same fp32 sums as mrqmm, written in fp32 times rs[m]
    (power-of-two rs: exact), on the drafter's shapes (K 5120 / 17408)."""
    for K, N in ((5120, 1280), (17408, 5120)):
        w, s, b = qweights(N, K, seed=K + M, bits=bits)
        x = (mx.random.normal((M, 1, K), key=mx.random.key(M)) * 300).astype(mx.float16)
        rs = mx.array([2.0 ** (-3 - i) for i in range(M)], dtype=mx.float32)
        y = mrqmm.mrqmm_wide(x, w, s, b, rs)
        assert y.dtype == mx.float32 and y.shape == (M, 1, N)
        ref = ref32(x.reshape(M, K), w, s, b, bits) * rs[:, None]
        err = mx.abs(y.reshape(M, N) - ref).max().item()
        assert err <= 2e-4 * mx.abs(ref).max().item(), (K, N, err)
        # the fp16 kernel's rows are these sums rounded to fp16 (unscaled: rs = 1)
        y1 = mrqmm.mrqmm_wide(x, w, s, b, mx.ones((M,), mx.float32))
        assert mx.array_equal(y1.astype(mx.float16), mrqmm.mrqmm(x, w, s, b)).item()


def test_wide_drafter_linear_past_fp16_range(monkeypatch):
    """kernels._drafter_linear: fp32 rows whose values and outputs overflow
    fp16 (max |x| 1e5) come out within fp16 input rounding of
    quantized_matmul in fp32 — the drafter's fp32 path (QWEN_DRAFT_FP32)."""
    import sys
    from qwen_worker import kernels
    monkeypatch.setitem(sys.modules, "mrqmm", mrqmm)
    K, N = 5120, 4096
    wf = (mx.random.normal((N, K), key=mx.random.key(1)) * 0.05).astype(mx.bfloat16)
    w, s, b = mx.quantize(wf, group_size=64, bits=4)
    s32, b32 = s.astype(mx.float32), b.astype(mx.float32)
    lin = nn.QuantizedLinear(K, N, bias=False, group_size=64, bits=4)
    lin.weight, lin.scales, lin.biases = w, s32, b32
    lin.__dict__["_mr_wide"] = (w, s.astype(mx.float16), b.astype(mx.float16))
    for M, mag in ((8, 1e5), (5, 1e-3), (1, 3.0)):
        x = mx.random.normal((1, M, K), key=mx.random.key(M)) * mag
        x[0, 0, 0] = 0.0
        y = kernels._drafter_linear(lin, x).reshape(1, M, N)
        ref = mx.quantized_matmul(x, w, s32, b32, transpose=True, group_size=64, bits=4)
        assert mx.isfinite(y).all().item()
        err = (mx.abs(y - ref).max() / mx.abs(ref).max()).item()
        assert err < 2e-3, (M, mag, err)
