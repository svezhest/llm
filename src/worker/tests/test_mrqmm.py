"""mrqmm (multi-row 8-bit quantized matmul for batched decode): the Metal
kernel against mx.quantized_matmul and an fp32 reference on synthetic 8-bit
g64 weights (rows 1..16, tile tails, K not a multiple of the K split,
outliers), the dispatch boundaries (rows <= 2 / > 16 / prefill-shaped / bf16 /
other bits stay on quantized_matmul, bit-identical), install() coverage, a
tiny qwen3_5 forward, worker._install_gdn_fuse (and its kernel warm-up), and
worker.sample_rows at the real vocabulary (grouped draw == solo per row).
Needs Metal (skipped otherwise; conftest makes the CPU the default device,
these tests switch to the GPU and back).

Real 9B weights (slow, optional): MRQMM_REAL_WEIGHTS=1 pytest tests/test_mrqmm.py
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


@pytest.fixture(autouse=True)
def gpu():
    prev = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(prev)


@pytest.fixture(autouse=True)
def narrow(monkeypatch):
    """The test layers are tiny: let install() take N >= 8 (default MIN_N 512)."""
    monkeypatch.setattr(mrqmm, "MIN_N", 8)


def qweights(N, K, seed=0, outlier_rows=0, bits=8, dtype=mx.float16):
    mx.random.seed(seed)
    wf = mx.random.normal((N, K)) * K ** -0.5
    if outlier_rows:
        wf[:outlier_rows] = wf[:outlier_rows] * 50
    wq, s, b = mx.quantize(wf.astype(dtype), group_size=64, bits=bits)
    return wq, s.astype(dtype), b.astype(dtype)


def ref32(x, w, s, b):
    deq = mx.dequantize(w, s.astype(mx.float32), b.astype(mx.float32), group_size=64, bits=8)
    return x.astype(mx.float32) @ deq.T


def qmm(x, w, s, b, bits=8):
    return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=bits)


def check_close(y, x, w, s, b):
    """Within fp16 output rounding of the fp32 reference, and no worse than
    quantized_matmul itself (+ one fp16 ulp)."""
    ref = ref32(x, w, s, b)
    q = qmm(x, w, s, b).astype(mx.float32)
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
