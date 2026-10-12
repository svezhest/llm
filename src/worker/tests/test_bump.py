"""mlx-vlm 0.7 / mlx 0.32 specifics (CPU): the enable_thinking default in
_render, the force_fused SDPA gate (QWEN_SDPA_FUSED) and the
mx.fast.gated_delta_update override (QWEN_GDN_FAST)."""
import json
import types

import mlx.core as mx
import pytest
from mlx_vlm.models import base
from mlx_vlm.models.cache import KVCache, QuantizedKVCache
from mlx_vlm.models.qwen3_5 import gated_delta

import worker
import worker_pb2 as pb


def _render_kwargs(monkeypatch, template_vars, ctk=None):
    import mlx_vlm.prompt_utils as pu
    seen = {}

    def fake(processor, config, messages, add_generation_prompt=True, num_images=0, **kw):
        seen.update(kw)
        return "prompt"
    monkeypatch.setattr(pu, "apply_chat_template", fake)
    eng = types.SimpleNamespace(config={"vision_config": {}}, vocab=512, _template_vars=template_vars,
                                _developer=False, grammar_for=lambda p: None, processor=None,
                                model_id="m")
    p = pb.Params(chat_template_kwargs_json=json.dumps(ctk) if ctk is not None else "")
    req = pb.Request(id="r", params=p, messages=[pb.Message(role="user", content="hi")])
    worker.Engine._render(eng, req, [])
    return seen


def test_render_enable_thinking_default(monkeypatch):
    """Unset = the template's default (thinking on): mlx-vlm >= 0.6.3 would
    pass False. The request's value wins; a template without the variable
    gets nothing."""
    assert _render_kwargs(monkeypatch, {"enable_thinking"})["enable_thinking"] is True
    assert _render_kwargs(monkeypatch, {"enable_thinking"},
                          {"enable_thinking": False})["enable_thinking"] is False
    assert "enable_thinking" not in _render_kwargs(monkeypatch, {"other"})
    assert "enable_thinking" not in _render_kwargs(monkeypatch, set(), {"enable_thinking": True})


def _qkv(q_len, kv_len, dtype=mx.float16):
    mx.random.seed(0)
    q = mx.random.normal((1, 4, q_len, 64)).astype(dtype)
    k = mx.random.normal((1, 2, kv_len, 64)).astype(dtype)
    v = mx.random.normal((1, 2, kv_len, 64)).astype(dtype)
    return q, k, v


def test_fused_sdpa_gate(monkeypatch):
    calls = []
    real = mx.fast.scaled_dot_product_attention

    def spy(*a, **kw):
        calls.append(kw.get("force_fused", False))
        if kw.get("force_fused"):                     # CPU has no fused kernel: as the GPU would
            kw = {k: v for k, v in kw.items() if k != "force_fused"}
        return real(*a, **kw)
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", spy)
    monkeypatch.setattr(worker, "_SDPA", base.scaled_dot_product_attention)
    monkeypatch.setattr(worker, "SDPA_FUSED", True)
    monkeypatch.setattr(worker, "SDPA_FUSED_MIN_Q", 8)
    monkeypatch.setattr(worker, "SDPA_FUSED_MIN_KV", 32)
    q, k, v = _qkv(8, 40)
    ref = base.scaled_dot_product_attention(q, k, v, cache=KVCache(), scale=0.125, mask="causal")
    calls.clear()
    out = worker._fused_sdpa(q, k, v, cache=KVCache(), scale=0.125, mask="causal")
    assert calls == [True] and mx.allclose(out, ref).item()
    # below the thresholds, decode, an array mask, fp32, a quantized cache: the library's call
    for args, cache, mask in (((7, 40), KVCache(), "causal"), ((8, 31), KVCache(), "causal"),
                              ((1, 40), KVCache(), None), ((8, 40), KVCache(), mx.ones((8, 40), mx.bool_))):
        calls.clear()
        worker._fused_sdpa(*_qkv(*args), cache=cache, scale=0.125, mask=mask)
        assert calls == [False], (args, mask)
    calls.clear()
    worker._fused_sdpa(*_qkv(8, 40, mx.float32), cache=KVCache(), scale=0.125, mask="causal")
    assert calls == [False]
    lib = []
    monkeypatch.setattr(worker, "_SDPA", lambda *a, **kw: lib.append(kw["cache"]))
    calls.clear()
    qc = QuantizedKVCache(group_size=64, bits=8)
    worker._fused_sdpa(q, k, v, cache=qc, scale=0.125, mask="causal")
    assert calls == [] and lib == [qc]
    # what a quantized cache really hands over: (packed, scales, biases) tuples
    # (a long prompt on the 27B/35B, q8/q4 KV past 5000 tokens, failed on keys.shape)
    lib.clear()
    kq, vq = mx.quantize(k, group_size=64, bits=8), mx.quantize(v, group_size=64, bits=8)
    worker._fused_sdpa(q, kq, vq, cache=qc, scale=0.125, mask="causal")
    assert calls == [] and lib == [qc]


def test_fused_sdpa_falls_back_once(monkeypatch):
    """No fused kernel (the CPU here): the library's result, and the gate turns
    itself off with one warning."""
    monkeypatch.setattr(worker, "_SDPA", base.scaled_dot_product_attention)
    monkeypatch.setattr(worker, "SDPA_FUSED", True)
    monkeypatch.setattr(worker, "SDPA_FUSED_MIN_Q", 8)
    monkeypatch.setattr(worker, "SDPA_FUSED_MIN_KV", 32)
    q, k, v = _qkv(8, 40)
    ref = base.scaled_dot_product_attention(q, k, v, cache=KVCache(), scale=0.125, mask="causal")
    out = worker._fused_sdpa(q, k, v, cache=KVCache(), scale=0.125, mask="causal")
    assert mx.array_equal(out, ref).item() and worker.SDPA_FUSED is False


@pytest.mark.parametrize("T", [1, 15, 16, 48])
def test_fast_gdn_kernel(monkeypatch, T):
    """mx.fast.gated_delta_update for >= 16 tokens: the recurrence of the
    library kernel (here its ops reference: the CPU has no Metal kernel) within
    fp16 rounding; shorter slabs and decode stay on the library kernel."""
    lib = []
    monkeypatch.setattr(worker, "_GDN_KERNEL",
                        lambda *a: lib.append(1) or gated_delta.gated_delta_ops(*a))
    mx.random.seed(1)
    B, hk, hv, dk, dv = 1, 2, 4, 64, 64
    q = mx.random.normal((B, T, hk, dk)).astype(mx.float16)
    k = mx.random.normal((B, T, hk, dk)).astype(mx.float16)
    q = (dk ** -1) * mx.fast.rms_norm(q, None, 1e-6)
    k = (dk ** -0.5) * mx.fast.rms_norm(k, None, 1e-6)
    v = mx.random.normal((B, T, hv, dv)).astype(mx.float16)
    a = mx.random.normal((B, T, hv)).astype(mx.float16)
    b = mx.random.normal((B, T, hv)).astype(mx.float16)
    g, beta = gated_delta._compute_g_beta(mx.log(mx.random.uniform(1, 16, (hv,))), a, b,
                                          mx.ones((hv,), mx.float16))
    state = mx.random.normal((B, hv, dv, dk)) * 0.1
    y, s = worker._fast_gdn_kernel(q, k, v, g, beta, state)
    y0, s0 = gated_delta.gated_delta_ops(q, k, v, g, beta, state)
    assert bool(lib) == (T < worker.GDN_FAST_MIN_T)
    assert y.dtype == y0.dtype and s.dtype == mx.float32 and s.shape == s0.shape
    assert mx.abs(y - y0).max().item() < 2e-3 and mx.abs(s - s0).max().item() < 2e-3


def test_rows_attention_needs_positions(dense):
    """A RowsKV step always passes explicit positions (rows sit at different
    offsets); without them there is no single cache offset to fall back on."""
    attn = dense.model.layers[dense.model.fa_idx].self_attn
    x = mx.zeros((2, 1, dense.args.hidden_size))
    with pytest.raises(ValueError):
        worker._rows_attention(attn, x, cache=worker.RowsKV([KVCache(), KVCache()]))


def test_verify_exact_cap(monkeypatch):
    """QWEN_VERIFY_EXACT_MAX_T: verify blocks longer than the cap skip the
    exact verify linears (quantized_matmul, as 0.6.2 for one row)."""
    exact = []
    monkeypatch.setattr(worker, "_VERIFY_LINEAR", lambda self, l, x: exact.append(x.shape[1]) or l(x))
    monkeypatch.setattr(worker, "_VERIFY_LINEARS",
                        lambda self, ls, x: exact.append(x.shape[1]) or tuple(l(x) for l in ls))
    monkeypatch.setattr(worker, "VERIFY_EXACT_MAX_T", 8)
    lin = lambda x: x + 1  # noqa: E731
    for T in (1, 8, 9, 16):
        x = mx.zeros((1, T, 4))
        assert mx.array_equal(worker._verify_linear(None, lin, x), x + 1).item()
        assert mx.array_equal(worker._verify_linears(None, (lin, lin), x)[1], x + 1).item()
    assert exact == [1, 1, 8, 8]


@pytest.mark.parametrize("cap", [0, 2, 16])
def test_dflash_exact_with_verify_cap(dense, monkeypatch, cap):
    """DFlash greedy = plain greedy whichever verify path a block takes."""
    from mlx_vlm.models.qwen3_5 import speculative_verifier as sv
    from conftest import prompts, request, run, scheduler
    from test_scheduler import FakeDraft, no_metal_verify
    no_metal_verify(monkeypatch)
    v = sv.Qwen3_5BatchInvariantForward
    monkeypatch.setattr(worker, "_VERIFY_LINEAR", v._linear)
    monkeypatch.setattr(worker, "_VERIFY_LINEARS", v._linears)
    monkeypatch.setattr(v, "_linear", worker._verify_linear)
    monkeypatch.setattr(v, "_linears", worker._verify_linears)
    monkeypatch.setattr(worker, "VERIFY_EXACT_MAX_T", cap)
    p = prompts([25], 30)[0]
    plain = run(scheduler(dense, max_batch=1), [request(p, 30, "a")])["a"]
    sched = scheduler(dense, max_batch=1)
    sched.eng.draft = FakeDraft(plain[0])
    assert run(sched, [request(p, 30, "a")])["a"] == plain
    assert len(sched.eng.draft.accept_lens) > 3
