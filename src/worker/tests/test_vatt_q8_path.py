"""QWEN_VATT_Q8 in the worker (CPU): over an 8-bit QuantizedKVCache, DFlash
verify blocks and decode steps attend through vatt.attention_q8 on the
cache's own (packed, scales, biases) buffers (not the [:n] views), with n =
the keys after the append; the greedy answers are the library's; the fp16
warm-up (below quantized_kv_start) keeps the fp16 paths; a failing call turns
the path off (the library's quantized calls from then on, the failing block
too); the load-time probe turns it on only where its self-test passes. The
kernel itself: tests/test_vatt.py (Metal, VATT_GPU=1)."""
import mlx.core as mx
import pytest

import vatt
import worker
from conftest import prompts, request, run, scheduler
from test_scheduler import FakeDraft, no_metal_verify


@pytest.fixture
def q8_calls(monkeypatch):
    """vatt.attention_q8 -> the library's quantized one-query calls over the
    buffers it is handed (so routing is checked exactly: the answers must be
    the library's bit for bit), recording (T, n, cap)."""
    from mlx_vlm.models.base import quantized_scaled_dot_product_attention as qsdpa
    calls = []

    def fake(q, kq, vq, n, scale, group_size=64, bits=8, chunk=None):
        assert type(kq) in (tuple, list) and len(kq) == 3 and kq[0].dtype == mx.uint32
        assert kq[0].shape[2] >= n and vq[0].shape == kq[0].shape and bits == 8 and group_size == 64
        calls.append((q.shape[2], n, kq[0].shape[2]))
        T = q.shape[2]
        return mx.concatenate([qsdpa(q[:, :, i:i + 1], tuple(x[..., :n - T + i + 1, :] for x in kq),
                                     tuple(x[..., :n - T + i + 1, :] for x in vq), scale, None, group_size, bits)
                               for i in range(T)], axis=2)
    monkeypatch.setattr(vatt, "attention_q8", fake)
    return calls


@pytest.fixture
def overrides(monkeypatch):
    """The load-time overrides: the SDPA global (decode) and the verifier's _attention."""
    from mlx_vlm.models.qwen3_5 import language as q35, speculative_verifier as sv
    no_metal_verify(monkeypatch)
    monkeypatch.setattr(worker, "_SDPA", q35.scaled_dot_product_attention)
    monkeypatch.setattr(q35, "scaled_dot_product_attention", worker._fused_sdpa)
    v = sv.Qwen3_5BatchInvariantForward
    monkeypatch.setattr(worker, "_VERIFY_ATTENTION", v._attention)
    monkeypatch.setattr(v, "_attention", worker._verify_attention)


@pytest.mark.parametrize("start", [0, 40])
def test_dflash_and_decode_through_q8_kernel(dense_fused, overrides, q8_calls, monkeypatch, start):
    p = prompts([25], 31)[0]
    kw = dict(max_batch=1, kv_bits=8, kv_start=start)
    monkeypatch.setattr(worker, "_VATT_Q8", False)
    monkeypatch.setattr(worker, "_VATT_Q8_DEC", False)
    plain = run(scheduler(dense_fused, **kw), [request(p, 30, "a")])["a"]
    assert not q8_calls

    # decode steps
    monkeypatch.setattr(worker, "_VATT_Q8_DEC", True)
    assert run(scheduler(dense_fused, **kw), [request(p, 30, "a")])["a"] == plain
    assert q8_calls and all(T == 1 for T, _, _ in q8_calls)
    # only once the cache is quantized (past the start); n = the keys after the append
    assert min(n for _, n, _ in q8_calls) == (len(p) if start == 0 else start + 1)
    assert all(cap >= n for _, n, cap in q8_calls)

    # DFlash rounds
    q8_calls.clear()
    monkeypatch.setattr(worker, "_VATT_Q8", True)
    monkeypatch.setattr(worker, "_VATT_Q8_DEC", False)
    sched = scheduler(dense_fused, **kw)
    sched.eng.draft = FakeDraft(plain[0])
    assert run(sched, [request(p, 30, "a")])["a"] == plain
    assert len(sched.eng.draft.accept_lens) > 3
    assert q8_calls and all(T >= 2 for T, _, _ in q8_calls)
    assert min(n for _, n, _ in q8_calls) > start


def test_batch_rows_decode_through_q8_kernel(dense_fused, overrides, q8_calls, monkeypatch):
    """Each row of a batch over its own quantized cache (rows attention calls
    the SDPA per row): the same answers as the library's quantized calls."""
    ps = prompts([40, 25, 33], 7)
    kw = dict(max_batch=3, kv_bits=8, kv_start=0)
    reqs = lambda: [request(p, 12, f"r{i}") for i, p in enumerate(ps)]   # noqa: E731
    monkeypatch.setattr(worker, "_VATT_Q8_DEC", False)
    want = run(scheduler(dense_fused, **kw), reqs())
    monkeypatch.setattr(worker, "_VATT_Q8_DEC", True)
    assert run(scheduler(dense_fused, **kw), reqs()) == want
    assert q8_calls and all(T == 1 for T, _, _ in q8_calls)


def test_failing_calls_turn_it_off(dense_fused, overrides, monkeypatch):
    p = prompts([25], 31)[0]
    kw = dict(max_batch=1, kv_bits=8, kv_start=0)
    monkeypatch.setattr(worker, "_VATT_Q8", False)
    monkeypatch.setattr(worker, "_VATT_Q8_DEC", False)
    plain = run(scheduler(dense_fused, **kw), [request(p, 30, "a")])["a"]

    def broken(*a, **k):
        raise ValueError("no kernel")
    monkeypatch.setattr(vatt, "attention_q8", broken)
    monkeypatch.setattr(worker, "_VATT_Q8", True)
    monkeypatch.setattr(worker, "_VATT_Q8_DEC", True)
    sched = scheduler(dense_fused, **kw)
    sched.eng.draft = FakeDraft(plain[0])
    assert run(sched, [request(p, 30, "a")])["a"] == plain
    assert worker._VATT_Q8 is False and worker._VATT_Q8_DEC is False


def test_fp16_cache_keeps_fp16_paths(dense, overrides, q8_calls, monkeypatch):
    """Without KV quantization nothing goes to the q8 kernel."""
    p = prompts([25], 31)[0]
    monkeypatch.setattr(worker, "_VATT_Q8", True)
    monkeypatch.setattr(worker, "_VATT_Q8_DEC", True)
    plain = run(scheduler(dense, max_batch=1), [request(p, 20, "a")])["a"]
    sched = scheduler(dense, max_batch=1)
    sched.eng.draft = FakeDraft(plain[0])
    assert run(sched, [request(p, 20, "a")])["a"] == plain
    assert not q8_calls


def test_probe(dense, monkeypatch):
    """On only with QWEN_KV_BITS=8 + QWEN_VATT_Q8, a head_dim the kernel
    takes, fp16 activations and a passing self-test (CPU: the reference
    stands in for the kernel)."""
    from mlx_vlm.models.qwen3_5 import language as q35
    monkeypatch.setattr(worker, "_SDPA", q35.scaled_dot_product_attention)
    monkeypatch.setattr(worker, "VATT_Q8_VERIFY", True)
    monkeypatch.setattr(worker, "VATT_Q8_DECODE", True)
    monkeypatch.setattr(worker.cfg, "KV_BITS", 0)
    assert worker._probe_vatt_q8(dense) == (False, False)
    monkeypatch.setattr(worker.cfg, "KV_BITS", 8)
    assert worker._probe_vatt_q8(dense) == (False, False)        # head_dim 64, fp32 activations
    monkeypatch.setattr(vatt, "D_OK", (64,))
    monkeypatch.setattr(worker, "_act_dtype", lambda lm: mx.float16)
    assert worker._probe_vatt_q8(dense) == (True, True) and worker._VATT_Q8 and worker._VATT_Q8_DEC
    monkeypatch.setattr(worker, "VATT_Q8_DECODE", False)
    assert worker._probe_vatt_q8(dense) == (True, False)
    monkeypatch.setattr(worker, "VATT_Q8_DECODE", True)
    monkeypatch.setattr(worker, "VATT_Q8_VERIFY", False)
    assert worker._probe_vatt_q8(dense) == (False, True)
    monkeypatch.setattr(worker, "VATT_Q8_VERIFY", True)
    monkeypatch.setattr(worker, "_SDPA", None)
    assert worker._probe_vatt_q8(dense) == (True, False)          # no SDPA override: no decode
    monkeypatch.setattr(worker, "_SDPA", q35.scaled_dot_product_attention)
    real = vatt.attention_q8
    monkeypatch.setattr(vatt, "attention_q8", lambda *a, **k: real(*a, **k) * 1.01)   # 1 % off
    assert worker._probe_vatt_q8(dense) == (False, False)
    monkeypatch.setattr(vatt, "attention_q8", lambda *a, **k: real(*a, **k) * float("nan"))
    assert worker._probe_vatt_q8(dense) == (False, False)
    assert worker._VATT_Q8 is False and worker._VATT_Q8_DEC is False


@pytest.mark.parametrize("kernel", ["tile", "reg", "regf"])
def test_probe_with_the_kernel_knob(dense, monkeypatch, kernel):
    """QWEN_VATT_Q8_KERNEL (vatt.Q8_KERNEL) picks the kernel inside vatt.attention_q8 only:
    the worker's calls and the probe pass no kernel, and the probe's reference and bound are
    the same for every kernel (fp32 over mx.dequantize's values; "regf" attends the affine
    values unrounded, so on the CPU its stand-in is the exact reference and its error vs the
    probe's reference is the largest of the three)."""
    from mlx_vlm.models.qwen3_5 import language as q35
    monkeypatch.setattr(worker, "_SDPA", q35.scaled_dot_product_attention)
    monkeypatch.setattr(worker, "VATT_Q8_VERIFY", True)
    monkeypatch.setattr(worker, "VATT_Q8_DECODE", True)
    monkeypatch.setattr(worker.cfg, "KV_BITS", 8)
    monkeypatch.setattr(vatt, "D_OK", (64,))
    monkeypatch.setattr(worker, "_act_dtype", lambda lm: mx.float16)
    monkeypatch.setattr(vatt, "Q8_KERNEL", kernel)
    exacts, real = [], vatt.reference_q8

    def ref(*a, exact=False, **k):
        exacts.append(exact)
        return real(*a, exact=exact, **k)
    monkeypatch.setattr(vatt, "reference_q8", ref)
    from qwen_worker import kernels
    why, worst = kernels._vatt_q8_selftest(dense, (1, 8))
    assert set(exacts) == ({False, True} if kernel == "regf" else {False})
    assert worker._probe_vatt_q8(dense) == ((not why), (not why))
    if kernel != "regf":
        assert not why and worst < 4e-4
    else:
        assert worst > 4e-4                     # not mx.dequantize's values: the probe's bound decides
    monkeypatch.setattr(vatt, "Q8_KERNEL", "tiles")
    assert worker._probe_vatt_q8(dense) == (False, False)        # an unknown name: off, the library's calls
