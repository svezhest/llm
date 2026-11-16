"""QWEN_VERIFY_ATT=2 in the worker (CPU): a DFlash verify block attends through
vatt.attention over the cache's own buffers (not the [:n] views, nothing
copied), with n = the keys after the block's append; the logits match the
library's per-query calls to rounding and the greedy answer is the plain
one; a failing call turns the path off (the vector kernel's path from then
on); the load-time self-test turns it on only where it passes. The kernel
itself: tests/test_vatt.py (Metal)."""
import mlx.core as mx

import vatt
import worker


def _one_pass_calls(monkeypatch):
    calls = []

    def fake(q, k, v, n, scale, chunk=None):
        calls.append((q.shape[2], n, k.shape[2]))
        assert k.shape[2] >= n and v.shape == k.shape and k.ndim == 4
        return vatt.reference(q, k, v, n, scale).astype(q.dtype)
    monkeypatch.setattr(vatt, "attention", fake)
    return calls


def test_verify_block_through_one_pass_kernel(dense, monkeypatch):
    from mlx_vlm.models.qwen3_5 import speculative_verifier as sv
    from mlx_vlm.speculative.cache_state import abort_speculative_round
    from conftest import prompts, request, run, scheduler
    from test_scheduler import FakeDraft, no_metal_verify
    no_metal_verify(monkeypatch)
    v = sv.Qwen3_5BatchInvariantForward
    lib, lib_calls = v._attention, []
    monkeypatch.setattr(worker, "_VERIFY_ATTENTION", lambda *a: lib_calls.append(a[2].shape[1]) or lib(*a))
    monkeypatch.setattr(v, "_attention", worker._verify_attention)
    calls = _one_pass_calls(monkeypatch)
    p = prompts([40], 30)[0]
    n_fa = sum(1 for x in dense.model.layers if not x.is_linear)

    logits = {}
    for mode in ("lib", "vatt"):
        monkeypatch.setattr(worker, "_VATT", mode == "vatt")
        monkeypatch.setattr(worker, "_VERIFY_ATT_Q", 0)
        calls.clear(), lib_calls.clear()
        c = dense.make_cache()
        dense(mx.array([p[:30]]), cache=c)
        out = dense(mx.array([p[30:36]]), cache=c, speculative_verify=True)
        logits[mode] = out.logits.astype(mx.float32)
        mx.eval(logits[mode])
        abort_speculative_round(out.gdn_states)
        if mode == "vatt":
            assert len(lib_calls) == 0 and len(calls) == n_fa
            assert all(T == 6 and n == 36 and cap >= 36 for T, n, cap in calls), calls
        else:
            assert len(lib_calls) == n_fa and not calls
    assert mx.abs(logits["vatt"] - logits["lib"]).max().item() < 1e-3
    assert mx.array_equal(mx.argmax(logits["vatt"], -1), mx.argmax(logits["lib"], -1)).item()

    # the DFlash run: the same greedy answer as plain decoding, every round's block through vatt
    monkeypatch.setattr(worker, "_VATT", False)
    plain = run(scheduler(dense, max_batch=1), [request(p, 30, "a")])["a"]
    monkeypatch.setattr(worker, "_VATT", True)
    calls.clear()
    sched = scheduler(dense, max_batch=1)
    sched.eng.draft = FakeDraft(plain[0])
    assert run(sched, [request(p, 30, "a")])["a"] == plain
    assert len(sched.eng.draft.accept_lens) > 3 and calls
    assert all(T >= 2 for T, _, _ in calls)


def test_failing_call_turns_it_off(dense, monkeypatch):
    from mlx_vlm.models.qwen3_5 import speculative_verifier as sv
    from mlx_vlm.speculative.cache_state import abort_speculative_round
    from conftest import prompts
    from test_scheduler import no_metal_verify
    no_metal_verify(monkeypatch)
    v = sv.Qwen3_5BatchInvariantForward
    monkeypatch.setattr(v, "_attention", worker._verify_attention)
    blocks = []

    def broken(*a, **k):
        raise ValueError("no kernel")

    def unfused(q, k, v, scale):
        blocks.append(q.shape[2])
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
    monkeypatch.setattr(vatt, "attention", broken)
    monkeypatch.setattr(worker, "_block_sdpa", unfused)
    monkeypatch.setattr(worker, "_VATT", True)
    monkeypatch.setattr(worker, "_VERIFY_ATT_Q", 4)
    p = prompts([40], 30)[0]
    c = dense.make_cache()
    dense(mx.array([p[:30]]), cache=c)
    out = dense(mx.array([p[30:36]]), cache=c, speculative_verify=True)
    mx.eval(out.logits)
    abort_speculative_round(out.gdn_states)
    assert worker._VATT is False
    n_fa = sum(1 for x in dense.model.layers if not x.is_linear)
    assert blocks == [4, 2] * n_fa          # the vector kernel's chunks in every layer, the first too


def test_probe(dense, monkeypatch):
    """On only with QWEN_VERIFY_ATT=2, a head_dim the kernel takes, fp16
    activations and a passing self-test (CPU: the reference path stands in)."""
    monkeypatch.setattr(worker, "VERIFY_ATT2", True)
    # the test model: head_dim 64, fp32 activations -> off
    assert worker._probe_vatt(dense) is False and worker._VATT is False
    monkeypatch.setattr(vatt, "D_OK", (64,))
    monkeypatch.setattr(worker, "_act_dtype", lambda lm: mx.float16)
    assert worker._probe_vatt(dense) is True and worker._VATT is True
    worker._probe_verify_attention(dense)          # the load call runs it too
    assert worker._VATT is True
    real = vatt.attention
    monkeypatch.setattr(vatt, "attention", lambda *a, **k: real(*a, **k) * 1.01)   # 1 % off
    assert worker._probe_vatt(dense) is False and worker._VATT is False
    monkeypatch.setattr(vatt, "attention", lambda *a, **k: real(*a, **k) * float("nan"))
    assert worker._probe_vatt(dense) is False

    def missing(*a, **k):
        raise ValueError("no kernel")
    monkeypatch.setattr(vatt, "attention", missing)
    assert worker._probe_vatt(dense) is False
    monkeypatch.setattr(vatt, "attention", real)
    monkeypatch.setattr(worker, "VERIFY_ATT2", False)
    assert worker._probe_vatt(dense) is False


def test_decode_att(dense_fused, monkeypatch):
    """QWEN_DECODE_ATT: a decode step's attention over >= DECODE_ATT_MIN_KV
    fp16 keys goes through vatt.attention on the row's own buffers (a lone row
    and every row of a batch), prefill slabs and shorter KV do not; the greedy
    answers are the ones without it (CPU: the reference stands in for the
    kernel); a failing call turns it off."""
    from mlx_vlm.models.qwen3_5 import language as q35
    from conftest import prompts, request, run, scheduler
    calls = _one_pass_calls(monkeypatch)
    monkeypatch.setattr(worker, "_SDPA", q35.scaled_dot_product_attention)
    monkeypatch.setattr(q35, "scaled_dot_product_attention", worker._fused_sdpa)
    ps = prompts([40, 25, 33], 7)
    want = {}
    for mb in (1, 3):
        monkeypatch.setattr(worker, "_VATT_DEC", False)
        want[mb] = run(scheduler(dense_fused, max_batch=mb),
                       [request(p, 12, f"r{i}") for i, p in enumerate(ps[:mb])])
        assert not calls
    monkeypatch.setattr(worker, "DECODE_ATT_MIN_KV", 30)
    for mb in (1, 3):
        monkeypatch.setattr(worker, "_VATT_DEC", True)
        calls.clear()
        got = run(scheduler(dense_fused, max_batch=mb), [request(p, 12, f"r{i}") for i, p in enumerate(ps[:mb])])
        assert got == want[mb]
        assert calls and all(T == 1 and n >= 30 for T, n, _ in calls)
        # the 25-token prompt's first decode steps are under the threshold: not through it
        assert min(n for _, n, _ in calls) == (40 if mb == 1 else 30)

    def broken(*a, **k):
        raise ValueError("no kernel")
    monkeypatch.setattr(vatt, "attention", broken)
    assert run(scheduler(dense_fused, max_batch=1), [request(ps[0], 12, "r0")]) == want[1]
    assert worker._VATT_DEC is False


def test_probe_decode_att(dense_fused, monkeypatch):
    from mlx_vlm.models.qwen3_5 import language as q35
    monkeypatch.setattr(worker, "DECODE_ATT_MIN_KV", 0)
    assert worker._probe_decode_att(dense_fused) is False
    monkeypatch.setattr(worker, "DECODE_ATT_MIN_KV", 4096)
    assert worker._probe_decode_att(dense_fused) is False        # head_dim 64
    monkeypatch.setattr(vatt, "D_OK", (64,))
    monkeypatch.setattr(worker, "_SDPA", None)
    assert worker._probe_decode_att(dense_fused) is False        # no override installed
    monkeypatch.setattr(worker, "_SDPA", q35.scaled_dot_product_attention)
    assert worker._probe_decode_att(dense_fused) is True and worker._VATT_DEC is True
