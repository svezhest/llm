"""Batched decode over per-row caches: RowsKV, the patched attention, cascade
step plans, the stacked GDN state."""

import typing
import weakref

import mlx.core as mx

import cascade


class _GroupRef:
    """Stands in for a grouped row's full-attention KVCache in row.cache: the
    KV lives in the group. Only RowsKV.offset reads it; the B = 1 library call
    never sees one (groups dissolve at B <= 2). Holds the row by a weak
    reference and not the group: row -> row.cache -> _GroupRef -> row (or ->
    the group's arrays) would be a reference cycle, so a dropped group's fp32
    prefix and stacks (GBs) would wait for Python's cyclic GC (review
    cascade_b2b3 finding 1)."""
    __slots__ = ("_row",)

    def __init__(self, row):
        self._row = weakref.ref(row)

    @property
    def offset(self):
        row = self._row()
        if row is None:
            raise RuntimeError("cascade: a grouped KV entry outlived its row")
        return row.pos


class _PrefixGroup:
    """Scheduler side of a cascade.PrefixGroup: running rows sharing prompt ids
    [0, P). `pg` holds the KV (the prefix once, fp32; the suffixes stacked),
    `rows` the members in stack order (pg's member j is rows[j]; batch order
    at formation, joins append in both), `ids` the shared tokens, `reserved`
    the bytes charged for it in _Scheduler.reserved (_group_bytes; each
    member keeps only _member_bytes). rows <-> row.group is a cycle while the
    group lives: whoever drops a group clears both (_Scheduler._unlink)."""

    def __init__(self, pg, ids, rows):
        self.pg, self.ids, self.rows = pg, ids, rows
        self.P = pg.P
        self.reserved = 0
        self.released_at = -1           # _Scheduler.steps of the last optional release


class _StepPlan(typing.NamedTuple):
    """One decode step's split of the batch (_Scheduler._plan), shared by every
    FA layer: `single` the ungrouped rows' batch indices, `groups` (pg, member
    batch indices) per group -- a slice when they are consecutive (members
    are kept in batch order, so a wave's group is one), else an int32 array
    (a gather) --, `inv` the take back to batch order (None when the parts
    are already in order), `owners` the groups."""
    single: list
    groups: list
    inv: object
    owners: list


class RowsKV:
    """The per-row KV caches of one full-attention layer in a batched decode
    step. No left_padding / make_mask / _idx: the library then builds no mask
    and takes none of its batch-cache branches. `layer` / `plan`: the FA layer
    index and the step's _StepPlan when rows are grouped (QWEN_CASCADE), else
    None: today's per-row loop."""

    def __init__(self, caches, layer=None, plan=None):
        self.caches = caches
        self.layer, self.plan = layer, plan

    @property
    def offset(self):                 # read by LanguageModel; positions are explicit
        return max(c.offset for c in self.caches)


_Q35 = None                           # mlx_vlm.models.qwen3_5.language, once installed
_ATTN_CALL = None


def _rows_attention(self, x, mask=None, cache=None, position_ids=None,
                    position_embeddings=None, **kw):
    """Qwen3_5Attention.__call__ (mlx-vlm 0.7.6, language.py 857-898) for a
    RowsKV cache: projections, norms and RoPE batched (the library's
    _prepare_projected_qkv without a cache: positions are always explicit),
    cache update + SDPA per row with the same calls a lone row makes. Any
    other cache goes to the library method. Diff against the library's
    __call__ / _prepare_projected_qkv when bumping mlx-vlm."""
    if not isinstance(cache, RowsKV):
        return _ATTN_CALL(self, x, mask=mask, cache=cache, position_ids=position_ids,
                          position_embeddings=position_embeddings, **kw)
    if position_ids is None:
        raise ValueError("a RowsKV step needs explicit position_ids")
    B, L, _ = x.shape
    queries, keys, values, gate, _ = self._prepare_projected_qkv(
        self.q_proj(x), self.k_proj(x), self.v_proj(x), None, position_ids,
        position_embeddings, None)
    if cache.plan is None:
        outs = []
        for i, c in enumerate(cache.caches):
            k, v = c.update_and_fetch(keys[i:i + 1], values[i:i + 1])
            outs.append(_Q35.scaled_dot_product_attention(queries[i:i + 1], k, v, cache=c,
                                                          scale=self.scale, mask=None))
        out = mx.concatenate(outs, axis=0)
    else:
        out = _cascade_rows(self, cache, queries, keys, values)
    out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(out * mx.sigmoid(gate))


def _cascade_rows(attn, cache, queries, keys, values):
    """The attention of a RowsKV step with groups (cache.plan): ungrouped rows
    take today's per-row calls verbatim; each group writes its members' K/V
    into its stack (cascade append) and attends them over the shared prefix
    + their suffixes at once (cascade.attention). Back to batch order with
    one take (exact copies: the ungrouped rows' outputs keep their bits)."""
    plan, layer = cache.plan, cache.layer
    parts = []
    for b in plan.single:
        c = cache.caches[b]
        k, v = c.update_and_fetch(keys[b:b + 1], values[b:b + 1])
        parts.append(_Q35.scaled_dot_product_attention(queries[b:b + 1], k, v, cache=c,
                                                       scale=attn.scale, mask=None))
    for pg, idx in plan.groups:        # idx: a slice (a view) or an int32 array (a gather)
        pg.append(layer, keys[idx], values[idx])
        parts.append(cascade.attention(queries[idx], pg, layer, attn.scale))
    if len(parts) == 1 and plan.inv is None:   # one group of the whole batch, in order
        return parts[0]
    out = mx.concatenate(parts, axis=0)
    return out if plan.inv is None else out[plan.inv]


def _install_rows_attention(lm):
    """Patch Qwen3_5Attention (also the MoE model's attention: the same class)
    once. -> False when `lm` is another architecture (no batching then)."""
    global _Q35, _ATTN_CALL
    from mlx_vlm.models.qwen3_5 import language as q35
    layers = getattr(getattr(lm, "model", None), "layers", None) or []
    if not layers or not all(hasattr(x, "is_linear") and (x.is_linear or isinstance(
            x.self_attn, q35.Qwen3_5Attention)) for x in layers):
        return False
    if _ATTN_CALL is None:
        _Q35, _ATTN_CALL = q35, q35.Qwen3_5Attention.__call__
        q35.Qwen3_5Attention.__call__ = _rows_attention
    return True


def _gdn_zero(layer, dtype):
    """The state Qwen3_5GatedDeltaNet starts from (a 1-token prompt has none yet)."""
    la = layer.linear_attn
    return [mx.zeros((1, la.conv_kernel_size - 1, la.conv_dim), dtype=dtype),
            mx.zeros((1, la.num_v_heads, la.head_v_dim, la.head_k_dim), dtype=mx.float32)]


def _gdn_stack(lm, row_caches, dtype):
    """{layer: ArraysCache holding the rows' recurrent states on axis 0}. The
    rows' own ArraysCaches are emptied (the stacked one is the live state),
    but only once every layer is built: a failure leaves the rows intact."""
    from mlx_vlm.models.cache import ArraysCache
    out = {}
    for i, layer in enumerate(lm.model.layers):
        if not layer.is_linear:
            continue
        parts = [list(rc[i].cache) if rc[i].cache[0] is not None else _gdn_zero(layer, dtype)
                 for rc in row_caches]
        b = ArraysCache(size=2)
        b.cache = [mx.concatenate([p[j] for p in parts], axis=0) for j in range(2)]
        out[i] = b
    for i in out:
        for rc in row_caches:
            rc[i].cache = [None] * len(rc[i].cache)
    return out


def _gdn_take(gdn, keep):
    from mlx_vlm.models.cache import ArraysCache
    idx = mx.array(keep, dtype=mx.int32)
    out = {}
    for i, c in gdn.items():
        out[i] = ArraysCache(size=2)
        out[i].cache = [a[idx] for a in c.cache]
    return out


def _gdn_concat(a, b):
    from mlx_vlm.models.cache import ArraysCache
    out = {}
    for i in a:
        out[i] = ArraysCache(size=2)
        out[i].cache = [mx.concatenate([x, y], axis=0) for x, y in zip(a[i].cache, b[i].cache)]
    return out


def _gdn_split(gdn, row, row_cache):
    """Give row `row` its state back (the batch shrinks to it alone)."""
    for i, c in gdn.items():
        row_cache[i].cache = [a[row:row + 1] for a in c.cache]
