"""Metal checks of QWEN_VERIFY_MRQMM and QWEN_SAMPLER_TOPK_FAST (small shapes,
< 1 GB, seconds of GPU): mrqmm on verify-shaped rows (1, T, K) -> (T, 1, K)
against the fp32 reference, quantized_matmul and mlx-vlm's exact verify
kernel; a tiny model's verify forward with the knob on vs off (fused GDN
in-projections included); the fast sampler at the real 248K vocabulary
(distribution == filtered()'s, grouped == solo bit for bit).

    VERIFY_GPU=1 pytest tests/test_gpu_verify_sampler.py

Skipped otherwise (the CPU suite must not touch Metal or import mrqmm)."""
import os

import mlx.core as mx
import numpy as np
import pytest

if os.environ.get("VERIFY_GPU") != "1" or not mx.metal.is_available():
    pytest.skip("Metal checks: set VERIFY_GPU=1", allow_module_level=True)

import mrqmm  # noqa: E402
from mlx_vlm.models.qwen3_5 import speculative_verifier  # noqa: E402
from mlx_vlm.speculative.ops import linear as vlinear  # noqa: E402
from test_mrqmm import Holder, check_close, make_linear, qmm, tiny_lm  # noqa: E402
from test_sampler_topk import boundary, full_probs, ref_probs  # noqa: E402

import worker  # noqa: E402


@pytest.fixture(autouse=True)
def gpu():
    prev = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(prev)


def knob(monkeypatch, value):
    monkeypatch.setattr(worker, "VERIFY_MRQMM", value)
    monkeypatch.setattr(worker, "VERIFY_MR", value == "1")
    monkeypatch.setattr(worker, "VERIFY_MR_HEAD", value in ("1", "head"))


@pytest.mark.parametrize("K,N", [(4096, 1024), (1024, 4096), (4096, 776)])
@pytest.mark.parametrize("T", [3, 4, 8, 9, 12, 16])
def test_verify_rows(monkeypatch, K, N, T):
    """A verify block as T decode rows: == mrqmm of those rows, within qmm's
    own error of the fp32 reference, and close to the exact verify kernel
    (whose rows are bit-identical to a 1-token qmv)."""
    knob(monkeypatch, "1")
    monkeypatch.setattr(mrqmm, "MIN_N", 8)
    lin = make_linear(K, N)
    x = (mx.random.normal((1, T, K), key=mx.random.key(T)) * 0.5).astype(mx.float16)
    exact = vlinear._target_verify_linear(lin, x)              # before the class swap
    assert mrqmm.install(Holder(l=lin), warmup=False) == 1
    try:
        assert worker._verify_mr(lin, x)
        y = worker._verify_linear(None, lin, x)
        assert y.shape == (1, T, N)
        assert mx.array_equal(y, mrqmm.mrqmm(x.reshape(T, 1, K), lin.weight, lin.scales,
                                             lin.biases).reshape(1, T, N)).item()
        check_close(y[0], x[0], lin.weight, lin.scales, lin.biases)
        ref = qmm(x, lin.weight, lin.scales, lin.biases).astype(mx.float32)
        e = mx.abs(y.astype(mx.float32) - exact.astype(mx.float32)).max().item()
        assert e <= 2 * mx.abs(ref - exact.astype(mx.float32)).max().item() + 2e-3, e
    finally:
        mrqmm.uninstall(Holder(l=lin))


@pytest.mark.parametrize("T", [4, 8, 16])
def test_verify_forward_tiny(monkeypatch, T):
    """A tiny 8-bit model's verify forward through the worker's hooks: knob
    off == the library's verify bit for bit (up to QWEN_VERIFY_EXACT_MAX_T); on: every linear on mrqmm (GDN
    in-projections fused), logits within the batched-decode tolerance."""
    lm = tiny_lm()
    v = speculative_verifier.Qwen3_5BatchInvariantForward
    monkeypatch.setattr(worker, "_VERIFY_LINEAR", v._linear)
    monkeypatch.setattr(worker, "_VERIFY_LINEARS", v._linears)
    ids = mx.random.randint(0, 512, (1, T), key=mx.random.key(1))

    def run():
        return lm(ids, cache=lm.make_cache(), speculative_verify=True).logits.astype(mx.float32)
    lib = run()
    monkeypatch.setattr(mrqmm, "MIN_N", 8)
    assert mrqmm.install(lm) > 0
    try:
        assert worker._install_gdn_fuse(lm) == 3
        monkeypatch.setattr(v, "_linear", worker._verify_linear)
        monkeypatch.setattr(v, "_linears", worker._verify_linears)
        knob(monkeypatch, "0")
        off = run()
        if T <= worker.VERIFY_EXACT_MAX_T:               # past it: quantized_matmul, as before
            assert mx.array_equal(off, lib).item()
        knob(monkeypatch, "1")
        assert worker._install_verify_mrqmm(lm) == 3
        calls = []
        orig = mrqmm.mrqmm
        monkeypatch.setattr(mrqmm, "mrqmm", lambda x, w, *a, **k: calls.append(w.shape[0]) or orig(x, w, *a, **k))
        y = run()
        # 3 GDN layers: fused in + out; 1 attention layer: q/k/v/o; 4 MLPs; lm_head
        assert len(calls) == 3 * 2 + 4 + 4 * 3 + 1, calls
        for ref in (lib, off):
            assert mx.abs(y - ref).max().item() <= 2e-2 * max(1.0, mx.abs(ref).max().item())
            assert (y.argmax(-1) == ref.argmax(-1)).all().item()
    finally:
        mrqmm.uninstall(lm)


def test_sample_rows_topk_fast_full_vocab(monkeypatch):
    """The fast sampler on Metal at V = 248320 (multi-block sort): its
    distribution == softmax(filtered()) per row (ties at the k-th value,
    a flat tail, tied maxima), and a (16, V) stack gives every row
    bit-for-bit what it alone gets (survivors and tokens)."""
    monkeypatch.setattr(worker, "SAMPLER_TOPK_FAST", True)
    V, B = 248320, 16
    cfgs = [dict(temperature=0.6, top_p=0.95, top_k=20), dict(temperature=1.0, top_k=50),
            dict(temperature=0.8, top_k=20, min_p=0.05), dict(temperature=1.0, top_p=0.95, top_k=20)]
    for c in cfgs:
        sampling = dict(worker.SAMPLING_KWARGS, **c)
        solo = [worker.make_sampler(sampling, seed=100 + i) for i in range(B)]
        grouped = [worker.make_sampler(sampling, seed=100 + i) for i in range(B)]
        for step in range(2):
            lg = mx.random.normal((B, V), key=mx.random.key(17 * step + 1)) * 4
            lg[:, 7] = lg.max(axis=-1)                       # ties with the maximum
            lg[:4, 1000:1064] = 9.0                          # 64 equal top logits
            top = mx.sort(lg[4:8], axis=-1)[:, -c["top_k"]:-c["top_k"] + 1]
            lg[4:8, 5:30] = top                              # 25 ties at the k-th value
            lg[8:, V - 4096:] = -3.0                         # a flat tail
            lg = lg.astype(mx.float16)
            xs = [lg[i:i + 1] for i in range(B)]
            s0 = grouped[0]
            st = s0.survivors(mx.concatenate(xs))
            got, want, amb = full_probs(st, V), ref_probs(s0, lg), boundary(s0, lg)
            for r in range(B):
                ok = amb[r] | ((got[r] > 0) == (want[r] > 0))
                assert ok.all(), (c, step, r)
                if not amb[r].any():
                    np.testing.assert_allclose(got[r], want[r], rtol=1e-4, atol=1e-6)
                one = solo[r].survivors(xs[r])
                for a in st.__slots__:
                    assert mx.array_equal(getattr(st, a)[r:r + 1], getattr(one, a)).item(), (c, r, a)
            want_t = [solo[i](xs[i]).tolist() for i in range(B)]
            got_t = [y.tolist() for y in worker.sample_rows(grouped, xs)]
            assert got_t == want_t, (c, step)


def test_position_noise_full_vocab(monkeypatch):
    """Noise keyed by (seed, position) on Metal at V = 248320: the hash's
    uniforms == the CPU's bit for bit (integer ops, exact conversion), and a
    verify block's draw (T rows at positions pos..pos + T - 1, one kernel)
    == each row drawn alone at its position, full-vocab and fast paths."""
    V = 248320
    keys = mx.array([worker._words(11, p) for p in (0, 1, 4095, 1 << 20)], dtype=mx.uint32)
    gpu_u = worker._uniform(keys, worker._cols(V))
    with mx.stream(mx.cpu):
        cpu_u = worker._uniform_rows(keys, worker._cols(V))
        mx.eval(cpu_u)
    assert mx.array_equal(gpu_u, cpu_u).item()
    for fast in (False, True):
        monkeypatch.setattr(worker, "SAMPLER_TOPK_FAST", fast)
        for c in [dict(temperature=0.6, top_p=0.95, top_k=20), dict(temperature=1.0),
                  dict(temperature=0.8, min_p=0.05)]:
            s = worker.make_sampler(dict(worker.SAMPLING_KWARGS, **c), seed=11)
            for T in (2, 4, 8, 16):
                lg = (mx.random.normal((T, V), key=mx.random.key(T)) * 4).astype(mx.float16)
                lg[:, 7] = lg.max(axis=-1)                   # ties with the maximum
                block = s(lg, 100).tolist()
                assert block == [s(lg[i:i + 1], 100 + i).item() for i in range(T)], (fast, c, T)
