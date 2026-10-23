"""Batch-invariant mode (QWEN_BATCH_INVARIANT) on the GPU: mrqmm's row
invariance at the 9B's K / N (rows 1..16, other rows beside it or not, ==),
and the scheduler on a tiny 8-bit fp16 qwen3_5 with the production kernels
(mrqmm in the mode, the fused GDN in-projection on it, Metal RoPE / GDN /
SDPA): every row of a batch of 4 with mixed lengths (shrinking to the B = 1
library call) gives the tokens and log-probabilities it gives alone, bit for
bit, also with forced (golden) tokens. The CPU cannot show this: its
vectorized fp16 / compiled elementwise kernels round an element by its
position in the array (tests/test_batch_invariant.py checks the rest there).

Light (< 0.3 GB, seconds) but it takes the GPU, like tests/test_pfqmm.py:
    BATCH_INV_GPU=1 pytest -q tests/test_batch_invariant_metal.py
"""
import os

import mlx.core as mx
import pytest
from mlx_vlm.models import rope_utils

import conftest
import mrqmm
import worker
from conftest import EOS, prompts, request, scheduler
from test_logprobs import drive, gen_req

pytestmark = pytest.mark.skipif(
    not (os.environ.get("BATCH_INV_GPU") == "1" and mx.metal.is_available()),
    reason="BATCH_INV_GPU=1 and Metal")


@pytest.fixture(autouse=True)
def gpu(monkeypatch):
    prev = mx.default_device()
    mx.set_default_device(mx.gpu)
    monkeypatch.setattr(rope_utils, "_HAS_METAL", True)     # the production RoPE kernels
    yield
    mx.set_default_device(prev)


@pytest.fixture
def inv(monkeypatch):
    monkeypatch.setattr(worker, "BATCH_INVARIANT", True)
    mrqmm.set_invariant(True)
    yield
    mrqmm.set_invariant(False)


def qweights(N, K, seed):
    """Random 8-bit codes and fp16 scales / biases of a plausible size: the
    bits of every output depend on them all the same."""
    mx.random.seed(seed)
    w = mx.random.randint(0, 256, (N, K), dtype=mx.uint8).view(mx.uint32)
    s = (mx.random.uniform(shape=(N, K // 64)) * 2e-3 + 1e-4).astype(mx.float16)
    b = (mx.random.normal((N, K // 64)) * 3e-2).astype(mx.float16)
    mx.eval(w, s, b)
    return w, s, b


# the 9B's (K, N): q 8192, k/v 1024, o / GDN out 4096, MLP gate/up 12288 and
# down (12288 -> 4096), the fused GDN in-projection 12352, b/a 32, and a wide N
# for the lm_head's many tiles (its 248320 x 4096 codes alone are 1 GB; N
# changes only how many tiles there are, not a tile's order)
SHAPES = [(4096, 8192), (4096, 1024), (4096, 4096), (4096, 12288), (12288, 4096),
          (4096, 12352), (4096, 32), (4096, 32768)]


@pytest.mark.parametrize("K,N", SHAPES)
def test_rows_invariant(K, N, inv):
    w, s, b = qweights(N, K, K + N)
    X = (mx.random.normal((16, 1, K)) * 2).astype(mx.float16)
    Z = (mx.random.normal((16, 1, K)) * 5).astype(mx.float16)      # other companions
    solo = [mrqmm.mrqmm(X[i:i + 1], w, s, b) for i in range(16)]
    mx.eval(solo)
    for M in range(1, 17):
        Y = mrqmm.mrqmm(X[:M], w, s, b)
        # row i among other rows: Z's rows around X[i]
        W = [mrqmm.mrqmm(mx.concatenate([Z[:i], X[i:i + 1], Z[i + 1:M]]), w, s, b)
             for i in range(M)]
        mx.eval(Y, W)
        for i in range(M):
            assert mx.array_equal(Y[i:i + 1], solo[i]).item(), (M, i)
            assert mx.array_equal(W[i][i:i + 1], solo[i]).item(), (M, i, "companions")


def test_default_mode_is_not_invariant():
    """The test above has teeth: by default the 8- and 16-row buckets split K
    differently (and B <= 2 runs qmv), so some output differs."""
    K, N = 12288, 4096
    w, s, b = qweights(N, K, 7)
    X = mx.random.normal((16, 1, K)).astype(mx.float16)
    a, c = mrqmm.mrqmm(X[:8], w, s, b)[0], mrqmm.mrqmm(X[:9], w, s, b)[0]
    assert not mx.array_equal(a, c).item()


@pytest.fixture
def lm(inv, monkeypatch):
    """conftest's tiny dense model, 8-bit fp16 (the QWEN_FP16 layout), mrqmm
    installed in the mode (every linear; lm_head N=512) + the fused GDN
    in-projection (N = 512 + 256 + 4 + 4 = 776, on mrqmm)."""
    monkeypatch.setenv("QWEN_MRQMM", "1")
    m = conftest.build_lm("dense", 8, mx.float16)
    assert mrqmm.install(m) > 0
    assert worker._install_gdn_fuse(m) == sum(1 for x in m.model.layers if x.is_linear)
    assert worker._invariant_uncovered(m) == {}
    yield m
    mrqmm.uninstall(m)


def mixed(golden):
    ps = prompts([31, 9, 22, 14, 17], 71)
    reqs = [gen_req(ps[0], 14, "a", logprobs=True, top_logprobs=3),
            gen_req(ps[1], 5, "b", logprobs=True, top_logprobs=3),
            request(ps[2], 9, "c", temperature=0.8, top_k=40, seed=3, logprobs=True),
            gen_req(ps[3], 12, "d", logprobs=True, top_logprobs=1),
            gen_req(ps[4], 7, "e", logprobs=True)]
    if golden:
        reqs[0].golden = [t for t in prompts([14], 72)[0] if t != EOS]
        reqs[3].golden = [t for t in prompts([6], 73)[0] if t != EOS]
    return reqs


@pytest.mark.parametrize("golden", [False, True])
@pytest.mark.parametrize("max_batch", [4, 2])
def test_batch_row_equals_solo(lm, golden, max_batch, monkeypatch):
    """5 requests, max_batch 4 (the 5th waits, then joins a running batch) or
    2: each row's (tokens, logprobs, end) == its run alone at max_batch 1."""
    monkeypatch.setattr(worker, "PREFILL_STEP", 8)
    monkeypatch.setattr(worker, "PREFILL_MIN_STEP", 4)
    both = drive(scheduler(lm, max_batch=max_batch), mixed(golden))
    for r in mixed(golden):
        solo = drive(scheduler(lm, max_batch=1), [r])[r.id]
        assert solo == both[r.id], r.id
        assert all(x is not None for x in solo[1])


def test_batch_differs_without_mode(lm, monkeypatch):
    """The test above has teeth: with the mode off the batched rows (3..4 on
    quantized_matmul's qmm here, N < 512; qmv alone) differ from solo."""
    monkeypatch.setattr(worker, "PREFILL_STEP", 8)
    monkeypatch.setattr(worker, "PREFILL_MIN_STEP", 4)
    mrqmm.set_invariant(False)
    monkeypatch.setattr(worker, "BATCH_INVARIANT", False)
    both = drive(scheduler(lm, max_batch=4), mixed(False))
    diff = [r.id for r in mixed(False)
            if drive(scheduler(lm, max_batch=1), [r])[r.id] != both[r.id]]
    assert diff
