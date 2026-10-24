"""QWEN_VERIFY_MRQMM: DFlash verify blocks / the drafter's lm_head through
mrqmm (worker._verify_linear / _verify_linears / _install_verify_mrqmm /
_verify_head). CPU: the real mrqmm module (loaded under its name for the
test only, never left in sys.modules: the CPU suite must not see it) with
its kernel replaced by quantized_matmul over the same (T, 1, K) rows, so the
dispatch is what is tested: which calls take it (3..16-token blocks on
covered modules, the GDN in-projections as one call over their fused
concatenation, lm_head), which keep the old path (1..2 tokens, > 16,
uncovered modules, knob off: the library's verify forward bit for bit).
Metal: tests/test_gpu_verify_sampler.py (VERIFY_GPU=1)."""
import importlib.util
import os
import sys

import mlx.core as mx
import mlx.nn as nn
import pytest
from conftest import build_lm
from mlx_vlm.models.qwen3_5 import speculative_verifier
from mlx_vlm.speculative.ops import linear as vlinear

import worker

MRQMM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mrqmm.py")


@pytest.fixture
def mr(monkeypatch):
    """The real mrqmm module, its Metal kernel -> quantized_matmul (CPU),
    every kernel call recorded as (x.shape, N)."""
    spec = importlib.util.spec_from_file_location("mrqmm", MRQMM)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.calls = []

    def kernel(x, w, s, b, cfg=None):
        m.calls.append((tuple(x.shape), w.shape[0]))
        return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=8)
    monkeypatch.setattr(m, "mrqmm", kernel)
    monkeypatch.setattr(m, "warm", lambda *a: None)
    monkeypatch.setattr(m, "enabled", lambda: True)
    monkeypatch.setattr(m, "MIN_N", 8)                 # tiny layers
    monkeypatch.setitem(sys.modules, "mrqmm", m)
    # the verify forward's Metal-only kernels -> their generic paths
    monkeypatch.setattr(vlinear, "_use_target_verify_dense", lambda *a: False)
    v = speculative_verifier.Qwen3_5BatchInvariantForward
    monkeypatch.setattr(worker, "_VERIFY_LINEAR", v._linear)
    monkeypatch.setattr(worker, "_VERIFY_LINEARS", v._linears)
    monkeypatch.setattr(v, "_linear", worker._verify_linear)
    monkeypatch.setattr(v, "_linears", worker._verify_linears)
    return m


def knob(monkeypatch, value):
    monkeypatch.setattr(worker, "VERIFY_MRQMM", value)
    monkeypatch.setattr(worker, "VERIFY_MR", value == "1")
    monkeypatch.setattr(worker, "VERIFY_MR_HEAD", value in ("1", "head"))


def model(mr):
    """8-bit fp16-scale tiny model, mrqmm installed, GDN in-projections fused."""
    lm = build_lm("dense", 8, mx.float16)
    assert mr.install(lm, warmup=False) > 0
    n = worker._install_gdn_fuse(lm)
    assert n == sum(1 for x in lm.model.layers if x.is_linear)
    return lm


def verify(lm, T, seed=0):
    ids = mx.random.randint(0, 500, (1, T), key=mx.random.key(seed))
    out = lm(ids, cache=lm.make_cache(), speculative_verify=True)
    return out.logits.astype(mx.float32)


def test_off_is_the_library_forward(mr, monkeypatch):
    """Knob off: the hooks change nothing (no mrqmm call, logits == the
    library's own verify methods bit for bit), at every block size."""
    knob(monkeypatch, "0")
    lm = model(mr)
    assert worker._install_verify_mrqmm(lm) == 0
    v = speculative_verifier.Qwen3_5BatchInvariantForward
    for T in (2, 3, 8, 9, 16, 17):
        got = verify(lm, T, seed=T)
        with monkeypatch.context() as m2:
            m2.setattr(v, "_linear", worker._VERIFY_LINEAR)
            m2.setattr(v, "_linears", worker._VERIFY_LINEARS)
            want = verify(lm, T, seed=T)
        assert mx.array_equal(got, want).item(), T
    assert mr.calls == []
    assert worker._verify_head(lm.lm_head) is lm.lm_head


@pytest.mark.parametrize("T", [3, 6, 8, 9, 16])
def test_on_routes_blocks_to_mrqmm(mr, monkeypatch, T):
    """3..16 tokens: every covered linear as T decode rows, the GDN
    in-projections as one call over N = qkv + z + b + a (b/a, N = 4, are
    not covered alone), lm_head included; logits close to the off path."""
    knob(monkeypatch, "0")
    lm = model(mr)
    want = verify(lm, T)
    knob(monkeypatch, "1")
    gdn = worker._install_verify_mrqmm(lm)
    assert gdn == sum(1 for x in lm.model.layers if x.is_linear)
    got = verify(lm, T)
    rows = {c[0] for c in mr.calls}
    assert rows == {(T, 1, 128), (T, 1, 256)}               # hidden 128, MLP 256
    ns = [c[1] for c in mr.calls]
    a = lm.model.layers[0].linear_attn
    fused = sum(a[n].weight.shape[0] for n in worker._GDN_IN)
    assert ns.count(fused) == gdn and 4 not in ns
    n_fa = len(lm.model.layers) - gdn
    # per FA layer q, k, v, o; per GDN layer fused in + out; per layer gate, up, down; lm_head
    assert len(ns) == 4 * n_fa + 2 * gdn + 3 * len(lm.model.layers) + 1
    assert mx.allclose(got, want, atol=2e-2, rtol=1e-2).item()


@pytest.mark.parametrize("T", [1, 2, 17])
def test_on_keeps_short_and_long_blocks(mr, monkeypatch, T):
    knob(monkeypatch, "0")
    lm = model(mr)
    want = verify(lm, T)
    knob(monkeypatch, "1")
    worker._install_verify_mrqmm(lm)
    assert mx.array_equal(verify(lm, T), want).item()
    assert mr.calls == []


def test_linears_mixed_and_uncovered(mr, monkeypatch):
    """A tuple with covered and uncovered modules: the covered ones on
    mrqmm, the rest through the old path, in order; nothing covered = old
    path (exact up to QWEN_VERIFY_EXACT_MAX_T, then the module itself)."""
    knob(monkeypatch, "1")
    seen = []
    monkeypatch.setattr(worker, "_VERIFY_LINEAR", lambda self, l, x: seen.append(("one", l)) or l(x))
    monkeypatch.setattr(worker, "_VERIFY_LINEARS",
                        lambda self, ls, x: seen.append(("many", tuple(ls))) or tuple(l(x) for l in ls))

    def q(N, bits=8):
        lin = nn.QuantizedLinear(128, N, bias=False, group_size=64, bits=bits)
        lin.scales, lin.biases = lin.scales.astype(mx.float16), lin.biases.astype(mx.float16)
        return lin
    a, b, c = q(64), q(64, bits=4), q(32)
    a.__class__ = c.__class__ = mr.MRQuantizedLinear
    x = mx.random.normal((1, 5, 128)).astype(mx.float16)
    ys = worker._verify_linears(None, (a, b, c), x)
    assert [y.shape for y in ys] == [(1, 5, 64), (1, 5, 64), (1, 5, 32)]
    assert seen == [("many", (b,))] and [n for _, n in mr.calls] == [64, 32]
    for y, lin in zip(ys, (a, b, c)):
        assert mx.allclose(y, nn.QuantizedLinear.__call__(lin, x), atol=1e-2).item()
    seen.clear(), mr.calls.clear()
    worker._verify_linear(None, b, x)                       # uncovered: exact path
    assert seen == [("one", b)] and mr.calls == []
    seen.clear()
    worker._verify_linear(None, b, mx.zeros((1, worker.VERIFY_EXACT_MAX_T + 1, 128), mx.float16))
    assert seen == [] and mr.calls == []                    # past the cap: the module itself
    worker._verify_linear(None, a, x.astype(mx.float32))     # not fp16: never the kernel
    assert mr.calls == [] and seen == [("one", a)]


@pytest.mark.parametrize("value", ["head", "1"])
def test_drafter_head(mr, monkeypatch, value):
    """"head" / "1": the drafter's lm_head (the target's) takes mrqmm for
    3..16-row blocks; "head" leaves the verify forward alone."""
    knob(monkeypatch, value)
    lm = model(mr)
    h = worker._verify_head(lm.lm_head)
    assert h is not lm.lm_head
    for T in (1, 2, 3, 15, 16, 17):
        x = mx.random.normal((1, T, 128)).astype(mx.float16)
        mr.calls.clear()
        y = h(x)
        assert y.shape == (1, T, lm.lm_head.weight.shape[0])
        assert mr.calls == ([((T, 1, 128), 512)] if 3 <= T <= 16 else [])
        assert mx.allclose(y, lm.lm_head(x), atol=2e-2).item()
    mr.calls.clear()
    if value == "head":
        worker._install_verify_mrqmm(lm)
        verify(lm, 6)
        assert mr.calls == []
    # a drafter bound to the module again (draft.reset) is wrapped again
    d = nn.Module()
    d.lm_head = lm.lm_head
    d.lm_head = worker._verify_head(d.lm_head)
    assert "lm_head" not in d and callable(d.lm_head)
    d.lm_head = lm.lm_head                                   # what bind() does
    assert d.lm_head is lm.lm_head


def test_dflash_rounds_on_mrqmm(mr, monkeypatch):
    """The scheduler's DFlash loop with the knob on: verify blocks go through
    mrqmm, and on the CPU (quantized_matmul computes each row alone) the
    answer is still the plain greedy one; the drafter's head is rebound and
    wrapped at every reset. Without _install_verify_mrqmm here: the fused
    GDN call hands strided views of one output to the CPU's elementwise
    kernels, which round differently from contiguous inputs (logits ~6e-3
    apart; Metal's do not, and decode's _gdn_call passes the same views)."""
    from conftest import prompts, request, run, scheduler
    from test_scheduler import FakeDraft
    knob(monkeypatch, "1")
    lm = model(mr)
    p = prompts([25], 30)[0]
    plain = run(scheduler(lm, max_batch=1), [request(p, 30, "a")])["a"]
    assert mr.calls == []                         # B = 1 decode never takes it

    class Draft(FakeDraft):
        def reset(self, model):
            self.lm_head = lm.lm_head             # as bind(): the target's module
            return None
    sched = scheduler(lm, max_batch=1)
    sched.eng.draft = Draft(plain[0], block=6)
    assert run(sched, [request(p, 30, "a")])["a"] == plain
    assert mr.calls and all(3 <= c[0][0] <= 16 for c in mr.calls)
    assert callable(sched.eng.draft.lm_head) and sched.eng.draft.lm_head is not lm.lm_head


def test_setup_at_load(mr, monkeypatch):
    """Engine.load's step: nothing covered -> off (block untouched); "head" ->
    the head only; "1" -> the fused GDN verify + the fixed block 8 unless
    QWEN_DRAFT_BLOCK is set."""
    import types
    draft = types.SimpleNamespace(prefer_requested_block_size=False)
    knob(monkeypatch, "1")
    monkeypatch.setattr(worker, "DRAFT_BLOCK", 0)
    plain = build_lm("dense", 8, mx.float16)              # mrqmm not installed: nothing covered
    assert worker._setup_verify_mrqmm(plain, draft) == "off: nothing covered"
    assert worker.DRAFT_BLOCK == 0 and not draft.prefer_requested_block_size
    lm = model(mr)
    knob(monkeypatch, "head")
    assert worker._setup_verify_mrqmm(lm, draft) == "drafter lm_head only"
    assert worker.DRAFT_BLOCK == 0 and not draft.prefer_requested_block_size
    knob(monkeypatch, "1")
    monkeypatch.setattr(worker, "DRAFT_BLOCK", 12)          # explicit: kept
    assert "block 12" in worker._setup_verify_mrqmm(lm, draft)
    assert worker.DRAFT_BLOCK == 12 and not draft.prefer_requested_block_size
    monkeypatch.setattr(worker, "DRAFT_BLOCK", 0)
    how = worker._setup_verify_mrqmm(lm, draft)
    assert worker.DRAFT_BLOCK == worker.VERIFY_MR_BLOCK == 8 and draft.prefer_requested_block_size
    assert "6 GDN layers" in how and "block 8" in how


def test_default_is_on():
    if "QWEN_VERIFY_MRQMM" not in os.environ:
        assert worker.VERIFY_MRQMM == "1" and worker.VERIFY_MR and worker.VERIFY_MR_HEAD
