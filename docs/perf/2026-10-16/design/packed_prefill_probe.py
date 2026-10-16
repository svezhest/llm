"""E2: variant (b) on CPU, whole model: R rows at different cache offsets,
each prefilling its own next slab; every row-local op (embedding, norms,
all Linear / QuantizedLinear GEMMs, MLP, residuals) runs ONCE on the rows'
tokens packed into (1, sum T_i [+ alignment filler], H); the sequence-coupled
ops (GDN conv + recurrence + gated norm, attention cache update + SDPA)
run per row on its own slice with exactly the solo call shapes.
Reference: the solo slab per row, `lm(ids, cache=..., skip_logits=True)` as
_Scheduler._prefill runs it on a prompt-cache hit (zero rope deltas).
Compared bitwise: every cache array of every layer after the slab.

usage: packed_model.py [tiny|wide]
  tiny: tests/conftest.py's dense model (fp32, and 8-bit fp16 scales)
  wide: 9B widths (hidden 4096, GDN 16/32 heads x 128, FA 16/4 heads x 256,
        MLP 12288), 4 layers (3 GDN + 1 FA), vocab 512, 8-bit fp16.
"""
import sys

import mlx.core as mx

mx.set_default_device(mx.cpu)
import mlx.nn as nn  # noqa: E402

REPO = "/Users/user/Projects/llm"
import os
PER_ROW_BA = os.environ.get("PER_ROW_BA") == "1"
sys.path[:0] = [f"{REPO}/src/worker", f"{REPO}/src/worker/tests"]
import conftest  # noqa: E402,F401  (CPU device, Metal RoPE off)
from conftest import build_lm  # noqa: E402
import worker  # noqa: E402
from mlx_vlm.models.base import create_attention_mask  # noqa: E402,F401
from mlx_vlm.models.cache import make_prompt_cache  # noqa: E402
from mlx_vlm.models.qwen3_5 import language as q35  # noqa: E402
from mlx_vlm.models.qwen3_5.config import TextConfig  # noqa: E402


def build_wide():
    mx.random.seed(0)
    cfg = TextConfig(model_type="qwen3_5_text", hidden_size=4096, intermediate_size=12288,
                     num_hidden_layers=4, num_attention_heads=16, num_key_value_heads=4,
                     head_dim=256, linear_num_value_heads=32, linear_num_key_heads=16,
                     linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
                     rms_norm_eps=1e-6, vocab_size=512, max_position_embeddings=262144,
                     full_attention_interval=4,
                     rope_parameters={"type": "default", "mrope_section": [11, 11, 10],
                                      "rope_theta": 10000000, "partial_rotary_factor": 0.25})
    lm = q35.LanguageModel(cfg, conftest.VCFG)
    nn.quantize(lm, group_size=64, bits=8,
                class_predicate=lambda p, m: hasattr(m, "to_quantized")
                and getattr(m, "weight", mx.zeros((1, 64))).shape[-1] % 64 == 0)
    lm.apply(lambda a: a.astype(mx.float16) if a.dtype in (mx.float32, mx.bfloat16) else a)
    mx.eval(lm.parameters())
    assert worker._install_rows_attention(lm)
    return lm


def prefix_cache(lm, ids):
    cache = make_prompt_cache(lm)
    if ids:
        lm._rope_deltas, lm._position_ids = None, None
        lm(mx.array([ids]), cache=cache, skip_logits=True)
        mx.eval([c.state for c in cache])
    return cache


def solo(lm, ids, cache, record):
    """The worker's hit slab; records the position_ids the model gets."""
    lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int32)
    lm._position_ids = None
    inner = lm.model.__call__.__func__ if hasattr(lm.model.__call__, "__func__") else None
    orig = type(lm.model).__call__

    def spy(self, *a, **k):
        record.append(k.get("position_ids"))
        return orig(self, *a, **k)
    type(lm.model).__call__ = spy
    try:
        emb = lm.model.embed_tokens(mx.array([ids]))
        lm(mx.array([ids]), inputs_embeds=emb, cache=cache, skip_logits=True)
    finally:
        type(lm.model).__call__ = orig
    mx.eval([c.state for c in cache])
    del inner


def gdn_row(m, mixed_qkv, z, b, a, cache):
    """Qwen3_5GatedDeltaNet.__call__ (mlx-vlm 0.7.6, language.py 1058-1121)
    after the in-projections, for one (1, T) row, mask None; returns the
    gated-norm output before out_proj."""
    B, S, _ = mixed_qkv.shape
    z = z.reshape(B, S, -1, m.head_v_dim)
    conv_state = cache[0] if cache[0] is not None else mx.zeros(
        (B, m.conv_kernel_size - 1, m.conv_dim), dtype=mixed_qkv.dtype)
    conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
    cache.update_window(0, conv_input, m.conv_kernel_size - 1, lengths=cache.lengths)
    conv_out = nn.silu(m.conv1d(conv_input))
    q, k, v = [t.reshape(B, S, h, d) for t, h, d in zip(
        mx.split(conv_out, [m.key_dim, 2 * m.key_dim], -1),
        [m.num_k_heads, m.num_k_heads, m.num_v_heads],
        [m.head_k_dim, m.head_k_dim, m.head_v_dim])]
    q, k = m._normalize_qk(q, k)
    out, _ = q35.gated_delta_update(q, k, v, a, b, m.A_log, m.dt_bias, mask=None,
                                    use_kernel=True, cache=cache)
    return m.norm(out, z).reshape(B, S, -1)


def attn_row(at, q, k, v, cache, pos, pe, mask):
    queries, keys, values, gate, mask = at._prepare_projected_qkv(q, k, v, cache, pos, pe, mask)
    out = q35.scaled_dot_product_attention(queries, keys, values, cache=cache, scale=at.scale,
                                           mask=mask)
    B, L = q.shape[:2]
    return out.transpose(0, 2, 1, 3).reshape(B, L, -1) * mx.sigmoid(gate)


def packed(lm, rows, align):
    """rows: [(ids, cache, position_ids)]. One packed pass (variant b)."""
    model = lm.model
    spans, parts, at = [], [], 0
    for ids, _, _ in rows:
        T = len(ids)
        start = -(-at // align) * align if align else at
        if start > at:
            parts.append(mx.zeros((start - at,), dtype=mx.int32))       # filler rows
        parts.append(mx.array(ids))
        spans.append((start, start + T))
        at = start + T
    flat = mx.concatenate(parts)[None]
    h = model.embed_tokens(flat)
    # per-row masks / rope tables, from the row's FA cache before the layers
    fa = [r[1][model.fa_idx] for r in rows]
    masks = [q35._create_qwen3_5_attention_mask(h[:, s:e], c) for (s, e), c in zip(spans, fa)]
    pes = []
    for (s, e), (_, _, pos) in zip(spans, rows):
        pe = None
        for layer in model.layers:
            if not layer.is_linear:
                if not layer.self_attn.rotary_emb.fused_apply:
                    pe = layer.self_attn.rotary_emb(h[:, s:e], pos)
                break
        pes.append(pe)
    for li, layer in enumerate(model.layers):
        x = layer.input_layernorm(h)
        outs = []
        if layer.is_linear:
            m = layer.linear_attn
            mixed, z = m.in_proj_qkv(x), m.in_proj_z(x)
            b, a = m._project_gates(x)
            for (s, e), (_, cache, _) in zip(spans, rows):
                if PER_ROW_BA:                  # the N=32 b/a GEMMs per row (split-K on Metal)
                    bi, ai = m._project_gates(x[:, s:e])
                else:
                    bi, ai = b[:, s:e], a[:, s:e]
                outs.append((s, e, gdn_row(m, mixed[:, s:e], z[:, s:e], bi, ai, cache[li])))
            width = m.value_dim
            proj = m.out_proj
        else:
            at_ = layer.self_attn
            q, k, v = at_.q_proj(x), at_.k_proj(x), at_.v_proj(x)
            for i, ((s, e), (_, cache, pos)) in enumerate(zip(spans, rows)):
                outs.append((s, e, attn_row(at_, q[:, s:e], k[:, s:e], v[:, s:e], cache[li], pos,
                                            pes[i], masks[i])))
            width = at_.num_attention_heads * at_.head_dim
            proj = at_.o_proj
        # back into the packed layout (filler rows: zeros)
        seq, last = [], 0
        for s, e, o in outs:
            if s > last:
                seq.append(mx.zeros((1, s - last, width), dtype=o.dtype))
            seq.append(o)
            last = e
        r = proj(mx.concatenate(seq, axis=1))
        h = h + r
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    mx.eval([c.state for _, cache, _ in rows for c in cache])


def copy(lm, c):
    return worker._copy_cache(c) if c[lm.model.fa_idx].keys is not None else make_prompt_cache(lm)


def arrays(cache):
    return conftest.view(cache)


def main(kind):
    if kind == "tiny":
        variants = [("dense fp32", build_lm("dense", None, mx.float32)),
                    ("dense 8-bit fp16", build_lm("dense", 8, mx.float16))]
        offs, lens = [37, 0, 150], [61, 120, 9]
    else:
        variants = [("wide 8-bit fp16", build_wide())]
        offs, lens = [150, 0, 70], [121, 40, 77]
    for name, lm in variants:
        V = lm.model.embed_tokens.weight.shape[0] if hasattr(lm.model.embed_tokens, "weight") else 512
        mx.random.seed(7)
        pre = [[int(t) for t in mx.random.randint(10, 400, (n,)).tolist()] for n in offs]
        new = [[int(t) for t in mx.random.randint(10, 400, (n,)).tolist()] for n in lens]
        base = [prefix_cache(lm, p) for p in pre]
        ref, poss = [], []
        for ids, c in zip(new, base):
            cc = copy(lm, c)
            rec = []
            solo(lm, ids, cc, rec)
            ref.append(cc)
            poss.append(rec[0])
        for align in (0, 8, 64):
            caches = [copy(lm, c) for c in base]
            packed(lm, [(ids, c, p) for ids, c, p in zip(new, caches, poss)], align)
            res = []
            for r, c in zip(ref, caches):
                a, b = arrays(r), arrays(c)
                ok = [mx.array_equal(x, y).item() for x, y in zip(a, b)]
                d = max(mx.abs(x.astype(mx.float32) - y.astype(mx.float32)).max().item()
                        for x, y in zip(a, b))
                res.append(f"{sum(ok)}/{len(ok)} eq, max|d| {d:.3g}")
            print(f"{name} align={align} offsets={offs} T={lens}: {res}", flush=True)
        del V


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tiny")
