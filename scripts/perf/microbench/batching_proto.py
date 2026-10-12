"""Prototype of the batched-decode plumbing from design/batching.md (section 2).

Pure helpers meant to be lifted into src/worker/worker.py:

  - RowsKV                : the per-row KV caches of ONE full-attention layer,
                            handed to the library model as that layer's cache.
  - install_rows_attention: class-level dispatch on Qwen3_5Attention.__call__
                            (shared by qwen3_5 and qwen3_5_moe): a RowsKV cache
                            runs the projections/norms/RoPE batched and the
                            cache update + SDPA per row; anything else goes to
                            the original method untouched.
  - stack_gdn / take_gdn / split_gdn : one ArraysCache per linear-attention
                            layer holding the recurrent + conv state of every
                            row stacked on axis 0.
  - batch_logits          : one decode forward for B rows (B >= 2).

No weights are needed to exercise this: see test_batching_proto.py (tiny random
qwen3_5 / qwen3_5_moe models on the CPU device).
"""
import mlx.core as mx
from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.qwen3_5 import language as q35


class RowsKV:
    """Per-row KV caches (KVCache or QuantizedKVCache, one per row) of one
    full-attention layer. No `left_padding`, no `make_mask`, no `_idx`: the
    library then builds no mask (decode, L=1) and never takes its
    batch-cache / left-padding branches (Qwen3_5Model.__call__, language.py
    1782-1876; LanguageModel.__call__ 2402-2423)."""

    def __init__(self, caches):
        self.caches = caches

    @property
    def offset(self):
        # LanguageModel.__call__ reads c0.offset (cache_offset); with explicit
        # position_ids the value is not used for positions.
        return max(int(c.offset) for c in self.caches)


_ORIG_ATTN_CALL = None


def _rows_attention(self, x, mask=None, cache=None, position_ids=None,
                    position_embeddings=None, target_verify=False):
    if not isinstance(cache, RowsKV):
        kw = dict(mask=mask, cache=cache, position_ids=position_ids,
                  position_embeddings=position_embeddings)
        if target_verify:              # mlx-vlm 0.6.2 only (0.7.x dropped the kwarg)
            kw["target_verify"] = True
        return _ORIG_ATTN_CALL(self, x, **kw)
    if hasattr(self, "_prepare_projected_qkv"):   # mlx-vlm 0.7.x
        B, L, _ = x.shape
        assert L == 1 and position_ids is not None
        queries, keys, values, gate, _ = self._prepare_projected_qkv(
            self.q_proj(x), self.k_proj(x), self.v_proj(x), None, position_ids,
            position_embeddings, None)
        outs = []
        for i, c in enumerate(cache.caches):
            k, v = c.update_and_fetch(keys[i:i + 1], values[i:i + 1])
            outs.append(q35.scaled_dot_product_attention(queries[i:i + 1], k, v, cache=c,
                                                         scale=self.scale, mask=None))
        out = mx.concatenate(outs, axis=0) if B > 1 else outs[0]
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out * mx.sigmoid(gate))
    # Same ops as Qwen3_5Attention.__call__ (language.py 1393-1495) for L == 1,
    # target_verify False; only update_and_fetch + SDPA run per row.
    B, L, _ = x.shape
    assert L == 1 and not target_verify and position_ids is not None
    q_out, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    queries, gate = mx.split(q_out.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1)
    gate = gate.reshape(B, L, -1)
    queries = self.q_norm(queries).transpose(0, 2, 1, 3)
    keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
    if position_embeddings is None:
        queries, keys = self.rotary_emb.apply_rotary(queries, keys, position_ids, unsqueeze_dim=1)
    else:
        cos, sin = position_embeddings
        queries, keys = q35.apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)
    outs = []
    for i, c in enumerate(cache.caches):
        k, v = c.update_and_fetch(keys[i:i + 1], values[i:i + 1])
        outs.append(q35.scaled_dot_product_attention(queries[i:i + 1], k, v, cache=c,
                                                     scale=self.scale, mask=None))
    out = mx.concatenate(outs, axis=0) if B > 1 else outs[0]
    out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(out * mx.sigmoid(gate))


def install_rows_attention():
    """Idempotent. Qwen3_5MoeAttention is the same class object (imported
    alias in qwen3_5_moe/language.py:8), so this covers the 35B too."""
    global _ORIG_ATTN_CALL
    if _ORIG_ATTN_CALL is None:
        _ORIG_ATTN_CALL = q35.Qwen3_5Attention.__call__
        q35.Qwen3_5Attention.__call__ = _rows_attention


def gdn_zero_state(layer, dtype):
    """What Qwen3_5GatedDeltaNet builds when a row has no state yet
    (a 1-token prompt: nothing was prefilled)."""
    la = layer.linear_attn
    conv = mx.zeros((1, la.conv_kernel_size - 1, la.conv_dim), dtype=dtype)
    state = mx.zeros((1, la.num_v_heads, la.head_v_dim, la.head_k_dim), dtype=mx.float32)
    return [conv, state]


def stack_gdn(lm, row_caches, dtype):
    """{layer index: ArraysCache with every row's (conv, recurrent) state
    concatenated on axis 0}. The rows' own ArraysCache objects are emptied so
    no stale copy survives (the batch one is the live state)."""
    out = {}
    for i, layer in enumerate(lm.model.layers):
        if not layer.is_linear:
            continue
        parts = []
        for rc in row_caches:
            c = rc[i]
            parts.append(list(c.cache) if c.cache[0] is not None else gdn_zero_state(layer, dtype))
            c.cache = [None] * len(c.cache)
        b = ArraysCache(size=2)
        b.cache = [mx.concatenate([p[j] for p in parts], axis=0) for j in range(2)]
        out[i] = b
    return out


def take_gdn(gdn, keep):
    """Keep rows `keep` (list of int) -> new dict (new arrays: a gather)."""
    idx = mx.array(keep, dtype=mx.int32)
    out = {}
    for i, c in gdn.items():
        b = ArraysCache(size=2)
        b.cache = [a[idx] for a in c.cache]
        out[i] = b
    return out


def concat_gdn(a, b):
    out = {}
    for i in a:
        c = ArraysCache(size=2)
        c.cache = [mx.concatenate([x, y], axis=0) for x, y in zip(a[i].cache, b[i].cache)]
        out[i] = c
    return out


def split_gdn(gdn, row, row_cache):
    """Give row `row` its state back (batch -> B=1 path)."""
    for i, c in gdn.items():
        row_cache[i].cache = [a[row:row + 1] for a in c.cache]


def batch_logits(lm, tokens, row_caches, gdn, positions):
    """One decode step for B rows. tokens: (B,) int32, positions: (B, 1)
    int32 = each row's absolute position (KV length + its mRoPE delta).
    row_caches[r][i] is row r's cache for layer i (full-attention layers
    used, linear ones replaced by gdn[i]). Returns logits (B, V)."""
    cache = [gdn[i] if layer.is_linear else RowsKV([rc[i] for rc in row_caches])
             for i, layer in enumerate(lm.model.layers)]
    out = lm(tokens[:, None], cache=cache, position_ids=positions)
    return out.logits[:, -1, :]
