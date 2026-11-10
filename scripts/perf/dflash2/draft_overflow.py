"""Where does the fp16 DFlash2 drafter break? Stage-by-stage nan/inf/absmax."""
import os
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm import load
from mlx_vlm.models.cache import make_prompt_cache
from mlx_vlm.speculative.drafters import load_drafter

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, f"{ROOT}/scripts/perf/prof")
from wtest import TASKS  # noqa: E402

model, proc = load(f"{ROOT}/weights/qwen38-27b-mlx-5bit")
lm = model.language_model
lm.apply(lambda a: a.astype(mx.float16) if a.dtype == mx.bfloat16 else a)
mx.eval(lm.parameters())
tok = getattr(proc, "tokenizer", proc)
argmax = lambda x: mx.argmax(x, axis=-1)  # noqa: E731


def fresh(dtype):
    d, _ = load_drafter(f"{ROOT}/weights/qwen38-27b-dflash2")
    if dtype is not None:
        d.apply(lambda a: a.astype(dtype) if a.dtype == mx.bfloat16 else a)
    mx.eval(d.parameters())
    return d


def st(name, a):
    a32 = a.astype(mx.float32)
    bad = mx.any(mx.isnan(a32)).item() or mx.any(mx.isinf(a32)).item()
    fin = mx.where(mx.isfinite(a32), mx.abs(a32), 0)
    print(f"   {name:34s} {str(a.dtype):22s} absmax(finite) {mx.max(fin).item():10.4g} "
          f"nan {int(mx.sum(mx.isnan(a32)).item()):8d} inf {int(mx.sum(mx.isinf(a32)).item()):8d}"
          + ("   <-- BAD" if bad else ""), flush=True)


d0 = fresh(None)
layers = list(d0.config.target_layer_ids)
s = tok.apply_chat_template([{"role": "user", "content": TASKS[0]}], tokenize=False,
                            add_generation_prompt=True, enable_thinking=False)
ids = tok.encode(s)
cache = make_prompt_cache(lm)
lm._rope_deltas = lm._position_ids = None
out = lm(mx.array([ids]), cache=cache)
y = mx.argmax(out.logits[:, -1, :], axis=-1)
gen = [y.item()]
for _ in range(47):
    out = lm(y[None], cache=cache)
    y = mx.argmax(out.logits[:, -1, :], axis=-1)
    gen.append(y.item())
seq = ids + gen
cache = make_prompt_cache(lm)
sink = []
lm._rope_deltas = lm._position_ids = None
lm.model(mx.array([seq]), cache=cache, capture_layer_ids=layers, hidden_sink=sink)
H = mx.concatenate(sink, axis=-1)
mx.eval(H)
n0 = len(ids)


def probe(d, H, block=8):
    lens = []
    for p in range(n0, len(seq) - block, 4):
        got = d.draft_block(int(seq[p]), H[:, :p], d.reset(model), block, argmax, mx.int32)
        got = [int(t) for t in got[0].tolist()]
        ref = seq[p + 1:p + block]
        k = 0
        while k < len(got) and got[k] == ref[k]:
            k += 1
        lens.append(k)
    return f"mean accepted prefix {sum(lens) / len(lens):.2f} of 7, lens {lens}"


def trace(d, H, label):
    print(f"-- {label}")
    d.reset(model)
    p = n0 + 20
    ctx = H[:, :p]
    x = d._embed_input_tokens(mx.array([[seq[p]] + [d.config.mask_token_id] * 7]))
    st("embed", x)
    f = d.fc(ctx)
    st("fc(H)", f)
    h_ctx = d.hidden_norm(f)
    st("hidden_norm", h_ctx)
    h = x
    for i, (layer, c) in enumerate(zip(d.layers, d.make_cache())):
        a_in = layer.input_layernorm(h)
        st(f"L{i} input_layernorm", a_in)
        a, kern = layer.attention_conv.prepare(a_in)
        st(f"L{i} attn_conv.prepare", a)
        st(f"L{i} attn kernel", kern)
        att = layer.self_attn(a, h_ctx, d.rope, c, None)
        st(f"L{i} self_attn", att)
        h = h + layer.attention_conv.finish(att, kern)
        st(f"L{i} after attn", h)
        m_in, kern = layer.mlp_conv.prepare(layer.post_attention_layernorm(h))
        st(f"L{i} mlp_conv.prepare", m_in)
        g, u = layer.mlp.gate_proj(m_in), layer.mlp.up_proj(m_in)
        st(f"L{i} gate", g)
        st(f"L{i} up", u)
        mm = layer.mlp(m_in)
        st(f"L{i} mlp", mm)
        h = h + layer.mlp_conv.finish(mm, kern)
        st(f"L{i} out", h)
    h = d.norm(h)
    st("final norm", h)
    st("logits", d._logits(h[:, 1:]))


print("bf16 drafter, bf16 H:", probe(d0, H.astype(mx.bfloat16)), flush=True)
trace(d0, H.astype(mx.bfloat16), "bf16 drafter (fresh), bf16 H")
print("bf16 drafter, fp16 H:", probe(fresh(None), H), flush=True)
del d0
d16 = fresh(mx.float16)
trace(d16, H, "fp16 drafter (fresh), fp16 H")
print("fp16 drafter, fp16 H:", probe(d16, H), flush=True)
del d16
d32 = fresh(mx.float32)
print("fp32 drafter, fp32 H:", probe(d32, H.astype(mx.float32)), flush=True)
print("peak GB", mx.get_peak_memory() / 1e9)
