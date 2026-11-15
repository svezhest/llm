"""mlx-vlm patches and fast paths: fused SDPA, fast GDN, the DFlash verify
(mrqmm, vector attention), the drafter's fp32 mode, the fused GDN
in-projection, batch-invariant checks."""

import sys
from typing import NamedTuple

import mlx.core as mx

from . import config as cfg
from .config import (DRAFT_FP32, GDN_FAST, GDN_FAST_MIN_T, GDN_FUSE, GIB, log, PREFILL_BUDGET,
                     PROMPT_CACHE, PROMPT_CACHE_REPLY, SDPA_FUSED_MIN_KV, SDPA_FUSED_MIN_Q,
                     VERIFY_ATT, VERIFY_EXACT_MAX_T, VERIFY_MR, VERIFY_MR_BLOCK, VERIFY_MR_HEAD)
from .sizing import _act_dtype


_SDPA = None                          # the library's qwen3_5 scaled_dot_product_attention
_GDN_KERNEL = None                    # the library's gated_delta_kernel
_VERIFY_LINEAR = _VERIFY_LINEARS = None   # Qwen3_5BatchInvariantForward._linear / _linears
_VERIFY_ATTENTION = None                  # ... ._attention
_VERIFY_ATT_Q = 0                         # queries per call; 0 = off (set at load)


def _fused_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
    """qwen3_5.language.scaled_dot_product_attention with force_fused for
    long fp16 prefill slabs (QWEN_SDPA_FUSED); everything else unchanged. If
    the fused kernel fails at run time, the gate turns off; a slab whose
    unfused scores would pass PREFILL_BUDGET (_step sized it for the fused
    kernel) then fails its request instead of materializing them."""
    # cache type first: a quantized cache hands over (packed, scales, biases)
    # tuples as keys/values
    if (type(cache).__name__ == "KVCache" and isinstance(keys, mx.array)
            and sinks is None and isinstance(mask, str) and mask == "causal"
            and queries.dtype == mx.float16 and keys.shape[2] >= SDPA_FUSED_MIN_KV
            and (queries.shape[2] >= SDPA_FUSED_MIN_Q
                 # a short slab (message cut, last slab) whose scores would
                 # exceed the budget _step relies on
                 or (queries.shape[2] > 16 and queries.shape[2] * keys.shape[2] > PREFILL_BUDGET))):
        if cfg.SDPA_FUSED:
            try:
                return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale,
                                                            mask=mask, force_fused=True)
            except (ValueError, RuntimeError) as e:    # no fused kernel for this shape / device
                cfg.SDPA_FUSED = False
                log.warning("QWEN_SDPA_FUSED off: %s", e)
        q, k = queries.shape[2], keys.shape[2]
        if q * k > PREFILL_BUDGET:
            raise RuntimeError(
                f"fused SDPA unavailable for a {q} x {k} prefill slab; unfused it would hold "
                f"{q * k * queries.shape[1] * 2 / GIB:.1f} GiB of scores (> QWEN_PREFILL_BUDGET)")
    return _SDPA(queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks)


def _probe_fused_sdpa():
    """QWEN_SDPA_FUSED off at load unless the fused kernel runs a causal
    head_dim-256 fp16 slab here: _step sizes fp16 slabs for it, and the
    unfused fallback at a long context would materialize the scores."""
    try:
        q = mx.zeros((1, 4, 64, 256), dtype=mx.float16)
        k = mx.zeros((1, 1, 128, 256), dtype=mx.float16)
        mx.eval(mx.fast.scaled_dot_product_attention(q, k, k, scale=0.0625, mask="causal",
                                                     force_fused=True))
    except (ValueError, RuntimeError, TypeError) as e:
        cfg.SDPA_FUSED = False
        log.warning("QWEN_SDPA_FUSED off (no fused SDPA kernel): %s", e)


def _fast_gdn_kernel(q, k, v, g, beta, state, mask=None):
    """qwen3_5.gated_delta.gated_delta_kernel (the Metal path of
    gated_delta_update without a speculative history) through
    mx.fast.gated_delta_update for slabs of >= GDN_FAST_MIN_T tokens."""
    if mask is None and g.ndim == 3 and q.shape[1] >= GDN_FAST_MIN_T:
        y, s = mx.fast.gated_delta_update(q, k, v, g, beta, initial_state=state)
        return y, s.astype(state.dtype)
    return _GDN_KERNEL(q, k, v, g, beta, state, mask)


def _verify_rows(x):
    """A verify block mrqmm takes as decode rows: (1, T, K) fp16 with
    MIN_ROWS..MAX_ROWS tokens (mrqmm loaded)."""
    mr = sys.modules.get("mrqmm")
    return (mr is not None and x.ndim == 3 and x.shape[0] == 1
            and mr.MIN_ROWS <= x.shape[1] <= mr.MAX_ROWS and x.dtype == mx.float16)


def _verify_mr(linear, x):
    """A verify / drafter-head call that takes mrqmm: _verify_rows on a module
    mrqmm.install covered."""
    mr = sys.modules.get("mrqmm")
    return mr is not None and type(linear) is mr.MRQuantizedLinear and _verify_rows(x)


def _mr_rows(linear, x):
    """(1, T, K) as T decode rows through the covered module (mrqmm)."""
    _, T, K = x.shape
    return linear(x.reshape(T, 1, K)).reshape(1, T, -1)


def _verify_linear(self, linear, x):
    if VERIFY_MR and _verify_mr(linear, x):
        return _mr_rows(linear, x)
    if x.shape[1] > VERIFY_EXACT_MAX_T:
        return linear(x)
    return _VERIFY_LINEAR(self, linear, x)


def _block_sdpa(queries, keys, values, scale):
    """Causal attention of the LAST queries.shape[2] positions over `keys`
    (query i sees keys [0, n - T + i]) through MLX's vector kernel; raises
    where there is none for the shape / dtype / device."""
    return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask="causal",
                                                force_fused=True)


def _rows_sdpa(queries, keys, values, scale, a, b):
    """The library's verify attention for queries [a, b) of a block of T (the
    last T positions of `keys`): one call per query over the keys up to its
    own position. -> list of (1, H, 1, D)."""
    n, T = keys.shape[2], queries.shape[2]
    return [mx.fast.scaled_dot_product_attention(
        queries[:, :, i:i + 1], keys[:, :, :n - T + i + 1], values[:, :, :n - T + i + 1],
        scale=scale, mask=None) for i in range(a, b)]


def _block_attention(queries, keys, values, scale, step):
    """A verify block's attention, `step` queries per vector-kernel call; a
    call that fails turns QWEN_VERIFY_ATT off and its queries (and the rest)
    go one by one."""
    global _VERIFY_ATT_Q
    n, T, out = keys.shape[2], queries.shape[2], []
    for a in range(0, T, step):
        b = min(T, a + step)
        if _VERIFY_ATT_Q:
            try:
                out.append(_block_sdpa(queries[:, :, a:b], keys[:, :, :n - T + b],
                                       values[:, :, :n - T + b], scale))
                continue
            except (ValueError, RuntimeError, TypeError) as e:
                _VERIFY_ATT_Q = 0
                log.warning("QWEN_VERIFY_ATT off: %s", e)
        out += _rows_sdpa(queries, keys, values, scale, a, b)
    return out[0] if len(out) == 1 else mx.concatenate(out, axis=2)


def _probe_verify_attention(lm):
    """At load with a drafter: -> queries per vector-kernel call (0 = off:
    QWEN_VERIFY_ATT=0, no kernel, or its output is not bit-equal to the
    one-query calls' on a random block of this model's heads)."""
    global _VERIFY_ATT_Q
    _VERIFY_ATT_Q = 0
    a = lm.args
    H, HK, D = a.num_attention_heads, a.num_key_value_heads, a.head_dim
    T = min(32 // max(1, H // HK), 16)
    if not VERIFY_ATT or T < 2:
        return 0
    try:
        dt = _act_dtype(lm) or mx.float16
        key = mx.random.key(0)
        q, k, v = (mx.random.normal(shape, key=mx.random.split(key, 3)[i]).astype(dt)
                   for i, shape in enumerate(((1, H, T, D), (1, HK, 700, D), (1, HK, 700, D))))
        k, v = k[:, :, :600], v[:, :, :600]           # views of longer buffers, as a cache's
        one = _block_sdpa(q, k, v, D ** -0.5)
        rows = mx.concatenate(_rows_sdpa(q, k, v, D ** -0.5, 0, T), axis=2)
        if mx.array_equal(one, rows).item():
            _VERIFY_ATT_Q = T
        else:
            log.warning("QWEN_VERIFY_ATT off: the vector attention kernel's %d-query output differs "
                        "from the one-query calls'", T)
    except (ValueError, RuntimeError, TypeError) as e:
        log.info("QWEN_VERIFY_ATT off (no vector attention kernel): %s", e)
    return _VERIFY_ATT_Q


def _verify_attention(self, attention, x, mask, cache, position_ids, position_embeddings):
    """Qwen3_5BatchInvariantForward._attention; a verify block (1, T >= 2, K)
    of a lone row over a plain fp16 KVCache attends _VERIFY_ATT_Q queries per
    call (QWEN_VERIFY_ATT) instead of one. Decided before the cache is
    touched; anything else is the library's call."""
    T = x.shape[1]
    if not (_VERIFY_ATT_Q and x.shape[0] == 1 and T >= 2
            and (mask is None or (isinstance(mask, str) and mask == "causal"))
            and type(cache).__name__ == "KVCache" and not hasattr(cache, "bits")):
        return _VERIFY_ATTENTION(self, attention, x, mask, cache, position_ids, position_embeddings)
    q, k, v = self._linears((attention.q_proj, attention.k_proj, attention.v_proj), x)
    queries, keys, values, gate, _ = attention._prepare_projected_qkv(
        q, k, v, cache, position_ids, position_embeddings, mask)
    out = _block_attention(queries, keys, values, attention.scale, _VERIFY_ATT_Q)
    out = out.transpose(0, 2, 1, 3).reshape(1, T, -1)
    return self._linear(attention.o_proj, out * mx.sigmoid(gate))


def _verify_linears(self, linears, x):
    if VERIFY_MR:
        f = linears[0].__dict__.get("_verify_fused") if linears else None
        if f is not None and tuple(map(id, linears)) == f[0] and _verify_rows(x):
            # the GDN in-projections (qkv, z, b, a): one mrqmm per width group
            # on the kernel (_install_verify_mrqmm; a uniform-width layer: one
            # over all four), as _gdn_call in decode; the others below
            _, T, K = x.shape
            out = [None] * len(linears)
            for idx, w, sc, b, split, mr in f[1]:
                y = mr.mrqmm(x.reshape(T, 1, K), w, sc, b).reshape(1, T, -1)
                for i, o in zip(idx, mx.split(y, split, axis=-1)):
                    out[i] = o
            rest = [linear for linear, o in zip(linears, out) if o is None]
            if not rest:
                return tuple(out)
            rest = iter(_verify_linears(self, rest, x))
            return tuple(next(rest) if o is None else o for o in out)
        on = [_verify_mr(linear, x) for linear in linears]
        if any(on):
            rest = [linear for linear, m in zip(linears, on) if not m]
            rest = iter(_verify_linears(self, rest, x) if rest else ())
            return tuple(_mr_rows(linear, x) if m else next(rest) for linear, m in zip(linears, on))
    if x.shape[1] > VERIFY_EXACT_MAX_T:
        return tuple(linear(x) for linear in linears)
    return _VERIFY_LINEARS(self, linears, x)


def _install_verify_mrqmm(lm):
    """QWEN_VERIFY_MRQMM=1: point each GDN layer's in_proj_qkv at the fused
    in-projection groups _install_gdn_fuse built for mrqmm (no copy), for
    _verify_linears. After _install_gdn_fuse. -> number of layers with one."""
    if not VERIFY_MR:
        return 0
    n = 0
    for _, m in lm.named_modules():
        gs = m.__dict__.get("_fused_in")
        on = [g for g in gs or () if g.mr is not None]
        if not on:
            continue
        ls = [m[a] for a in _GDN_IN]
        # ids of the four modules (no reference back to them: no cycle); per
        # group on the kernel: its members' positions in _GDN_IN + its weights
        ls[0].__dict__["_verify_fused"] = (tuple(map(id, ls)), tuple(
            (tuple(_GDN_IN.index(a) for a in g.names), g.w, g.s, g.b, g.split, g.mr) for g in on))
        n += 1
    return n


def _setup_verify_mrqmm(lm, draft):
    """At load, with a drafter and QWEN_VERIFY_MRQMM 1 / head: nothing when
    mrqmm covers no module (QWEN_MRQMM=0, QWEN_FP16=0, the 5-bit 27B without
    QWEN_MRQMM5=1, the 4-bit 27B without QWEN_MRQMM4=1 or with its self-test
    failed); else
    the fused GDN verify (1) and, unless QWEN_DRAFT_BLOCK is set, the fixed
    block VERIFY_MR_BLOCK. -> what the load log says."""
    mr = sys.modules.get("mrqmm")
    if mr is None or not any(type(m) is mr.MRQuantizedLinear for _, m in lm.named_modules()):
        return "off: nothing covered"
    if not VERIFY_MR:
        return "drafter lm_head only"
    gdn = _install_verify_mrqmm(lm)
    if not cfg.DRAFT_BLOCK:
        cfg.DRAFT_BLOCK = VERIFY_MR_BLOCK
        draft.prefer_requested_block_size = True
    return (f"verify blocks of 3..16 tokens + drafter lm_head ({gdn} GDN layers fused), "
            f"block {cfg.DRAFT_BLOCK}")


def _draft_wide(draft):
    """QWEN_DRAFT_FP32: does this drafter run fp32 activations?"""
    if DRAFT_FP32 in ("0", "1"):
        return DRAFT_FP32 == "1"
    return type(draft).__name__.startswith("DFlash2")


def _widen_drafter(draft, quantized):
    """QWEN_DRAFT_FP32, at load after the drafter's quantization: every bf16
    parameter left (quantization scales / biases, norms, conv kernels) to
    fp32, the selector's codebooks (embedding lookups, 2 x vocab x rank) to
    fp16, and the block's input embeddings (the target's, fp16) cast to fp32
    so the residual stream is fp32 whatever the promotion rules. Unquantized
    weights are left as loaded (bf16). _spec casts the target's hidden states
    on the way in and the drafter's on the way to the target's lm_head."""
    if quantized:
        from mlx.utils import tree_flatten, tree_unflatten
        draft.update(tree_unflatten(
            [(k, v.astype(mx.float16 if "codebook" in k else mx.float32)
              if v.dtype == mx.bfloat16 else v) for k, v in tree_flatten(draft.parameters())]))
    embed = draft._embed_input_tokens
    draft._embed_input_tokens = lambda ids: embed(ids).astype(mx.float32)


def _verify_head(head):
    """QWEN_VERIFY_MRQMM 1 / head: the drafter's lm_head (bound to the
    target's at every draft.reset) with 3..16-row blocks on mrqmm."""
    mr = sys.modules.get("mrqmm")
    if not VERIFY_MR_HEAD or mr is None or type(head) is not mr.MRQuantizedLinear:
        return head
    return lambda x: _mr_rows(head, x) if _verify_mr(head, x) else head(x)
_GDN_IN = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
_GDN_CALL = None                      # the library's Qwen3_5GatedDeltaNet.__call__


class _InGroup(NamedTuple):
    """In-projections of one GDN layer quantized alike, run as one matmul
    over their concatenated weights (_install_gdn_fuse)."""
    names: tuple                      # members, in _GDN_IN order
    w: mx.array                       # concatenated packed weights / scales / biases
    s: mx.array
    b: mx.array
    split: list                       # mx.split indices of the output (ends but the last)
    gs: int
    bits: int
    mode: str
    mr: object                        # the mrqmm module for 3..16 rows, or None


class _Given:
    """Stands in for an in-projection during one fused decode call: returns
    its precomputed output."""
    __slots__ = ("y",)

    def __init__(self, y):
        self.y = y

    def __call__(self, x):
        return self.y


def _gdn_call(self, inputs, mask=None, cache=None):
    """Qwen3_5GatedDeltaNet.__call__ with the fused decode in-projection: each
    width group's outputs are computed in one matmul and handed to the library
    call in place of the modules (restored right after; the scheduler thread
    is the only caller of the model). A projection in no group (its width
    shared with no other) is the library's own call."""
    gs = self.__dict__.get("_fused_in")
    if gs is None or not GDN_FUSE or inputs.ndim != 3 or inputs.shape[1] != 1 or inputs.dtype != gs[0].s.dtype:
        return _GDN_CALL(self, inputs, mask=mask, cache=cache)
    given = {}
    for g in gs:
        if g.mr is not None and g.mr._use(inputs):
            y = g.mr.mrqmm(inputs, g.w, g.s, g.b)
        else:
            y = mx.quantized_matmul(inputs, g.w, scales=g.s, biases=g.b, transpose=True,
                                    group_size=g.gs, bits=g.bits, mode=g.mode)
        given.update(zip(g.names, mx.split(y, g.split, axis=-1)))
    mods = {n: self[n] for n in given}
    for n, o in given.items():
        dict.__setitem__(self, n, _Given(o))
    try:
        return _GDN_CALL(self, inputs, mask=mask, cache=cache)
    finally:
        for n, m in mods.items():
            dict.__setitem__(self, n, m)


def _gdn_groups(ls):
    """Indices (into _GDN_IN) of the in-projections that can share a matmul:
    quantized, no bias, fp16 scales and biases, grouped by (bits, group size,
    mode); groups of >= 2 only, in order of their first member. A uniform-
    width layer: one group [0, 1, 2, 3]."""
    import mlx.nn as nn
    by = {}
    for i, l in enumerate(ls):
        if (isinstance(l, nn.QuantizedLinear) and "bias" not in l and l.get("biases") is not None
                and l.scales.dtype == mx.float16 and l.biases.dtype == mx.float16):
            by.setdefault((l.bits, l.group_size, getattr(l, "mode", "affine")), []).append(i)
    return [ix for ix in by.values() if len(ix) >= 2]


def _group_label(gs):
    """'qkv+z 4-bit, b+a 5-bit' (+ ', a alone')."""
    short = lambda n: n[len("in_proj_"):]  # noqa: E731
    seen = {n for g in gs for n in g.names}
    parts = [f"{'+'.join(map(short, g.names))} {g.bits}-bit" for g in gs]
    parts += [f"{short(n)} alone" for n in _GDN_IN if n not in seen]
    return ", ".join(parts)


def _install_gdn_fuse(lm):
    """QWEN_GDN_FUSE: per GDN layer, concatenate the in-projections quantized
    alike (_gdn_groups: by bits / group size / mode; fp16 scales/biases, no
    bias) into one weight per group, point the modules at row views of it,
    route the class __call__. A uniform-width layer is one group of all four
    (the fusion as it always was); a mixed-width one (oQ4e: qkv+z 4-bit, b+a
    5-bit) fuses each width group, a projection alone at its width keeps its
    own weights. A group takes mrqmm for 3..16 rows where mrqmm.supported
    takes its width and N >= mrqmm.MIN_N (any N in batch-invariant mode), as
    mrqmm.install's rule for a module. After the fp16 cast and mrqmm.install.
    -> number of layers with a group."""
    global _GDN_CALL
    if not GDN_FUSE:
        return 0
    from mlx_vlm.models.qwen3_5 import language as q35
    mr = sys.modules.get("mrqmm")
    n, warm, kinds = 0, {}, {}
    for _, m in lm.named_modules():
        if not isinstance(m, q35.Qwen3_5GatedDeltaNet) or "_fused_in" in m.__dict__:
            continue
        ls = [m[a] for a in _GDN_IN]
        groups = []
        for ix in _gdn_groups(ls):
            gl = [ls[i] for i in ix]
            f = gl[0]
            w, s, b = (mx.concatenate([l[k] for l in gl], axis=0) for k in ("weight", "scales", "biases"))
            mx.eval(w, s, b)
            offs, o = [], 0
            for l in gl:
                offs.append((o, o + l.weight.shape[0]))
                o += l.weight.shape[0]
            for l, (lo, hi) in zip(gl, offs):     # row views: the old buffers are freed
                l.weight, l.scales, l.biases = w[lo:hi], s[lo:hi], b[lo:hi]
                mx.eval(l.weight, l.scales, l.biases)
            mode = getattr(f, "mode", "affine")
            # the kernel's bit widths: 8, 5 / 4 / 6 with QWEN_MRQMM5 / 4 / 6 = 1
            # once self-tested (mr.supported); narrow groups (b+a, N = 96 on the
            # 27B) stay on quantized_matmul as narrow modules do
            use_mr = (mr is not None and mr.enabled() and f.group_size == mr.GROUP
                      and (mr.INVARIANT or w.shape[0] >= mr.MIN_N)
                      and mr.supported(w, s, b, f.bits, f.group_size, mode))
            groups.append(_InGroup(tuple(_GDN_IN[i] for i in ix), w, s, b, [hi for _, hi in offs[:-1]],
                                   f.group_size, f.bits, mode, mr if use_mr else None))
            if use_mr:
                warm.setdefault(w.shape, (w, s, b))
        if not groups:
            continue
        m.__dict__["_fused_in"] = tuple(groups)
        k = _group_label(groups)
        kinds[k] = kinds.get(k, 0) + 1
        n += 1
    # the concatenated shapes (9B K=4096, 35B K=2048 -> N=12352) are new to
    # mrqmm: compile their 3..8 / 9..16-row kernels now, not in the first
    # batched decode step on the scheduler thread
    for ws in warm.values():
        mr.warm(*ws)
    warm.clear()
    if n and _GDN_CALL is None:
        _GDN_CALL, q35.Qwen3_5GatedDeltaNet.__call__ = q35.Qwen3_5GatedDeltaNet.__call__, _gdn_call
    if n:
        mx.clear_cache()
    if any(not k.startswith("qkv+z+b+a ") for k in kinds):     # mixed widths: say how
        log.info("GDN in-projections fused by width: %s",
                 "; ".join(f"{v} x [{k}]" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])))
    return n


def _invariant_uncovered(lm):
    """mrqmm.uncovered for the language model, the fused GDN in-projections
    counted as their group's one call: on mrqmm (covered: the members are
    never called alone), or on quantized_matmul (one entry per group)."""
    import mrqmm
    skip, out = set(), {}
    for _, m in lm.named_modules():
        for g in m.__dict__.get("_fused_in") or ():
            skip.update(id(m[n]) for n in g.names)
            if g.mr is None:
                k = "fused GDN in-projection on quantized_matmul"
                out[k] = out.get(k, 0) + 1
    for k, v in mrqmm.uncovered(lm, skip).items():
        out[k] = out.get(k, 0) + v
    return out


def _invariant_check(lm, max_batch):
    """QWEN_BATCH_INVARIANT at load, before the scheduler is built -> the
    max_batch the guarantee holds at. Modules outside the invariant kernel
    (_invariant_uncovered: 27B / 35B non-8-bit layers, QWEN_MRQMM=0, QWEN_FP16=0)
    keep MLX's row-count rules, under which only B <= 2 is exact: quantized
    layers (and the experts' unsorted gather, B * top_k < 64) run one qmv per
    row there, mrqmm gives 1 and 2 rows the same bits; so max_batch -> 2. An
    unquantized nn.Linear has no such rule (MLX's 1-row gemv vs a gemm at 2
    rows): the mode is refused (off, with an error; the drafter, skipped at
    load, stays off)."""
    miss = _invariant_uncovered(lm)
    what = ", ".join(f"{k} x{v}" for k, v in sorted(miss.items()))
    if any(k.startswith("Linear") for k in miss):
        log.error("QWEN_BATCH_INVARIANT=1 refused: unquantized layers (%s) are not row-exact "
                  "at any B; the mode is off", what)
        cfg.BATCH_INVARIANT = False
        mr = sys.modules.get("mrqmm")
        if mr is not None:
            mr.set_invariant(False)
        return max_batch
    if miss and max_batch > 2:
        log.warning("batch-invariant: %d modules not covered (%s): exact only at B <= 2 there, "
                    "max_batch %d -> 2", sum(miss.values()), what, max_batch)
        return 2
    return max_batch


def _log_invariant(lm, max_batch):
    """QWEN_BATCH_INVARIANT's load log (after fit_max_batch: the real
    max_batch), and what still makes runs differ."""
    if max_batch <= 2:
        log.info("batch-invariant (QWEN_BATCH_INVARIANT): max_batch %d (B <= 2: exact per row); "
                 "DFlash and cascade off", max_batch)
    else:
        import mrqmm
        n = sum(1 for _, m in lm.named_modules() if type(m) is mrqmm.MRQuantizedLinear)
        log.info("batch-invariant (QWEN_BATCH_INVARIANT): max_batch %d, %d modules on mrqmm for "
                 "1..%d rows (K split %d), %d fused GDN in-projections; DFlash and cascade off",
                 max_batch, n, mrqmm.MAX_ROWS, mrqmm.INVARIANT_KS, sum(
                     1 for _, m in lm.named_modules() if "_fused_in" in m.__dict__))
    if PROMPT_CACHE and PROMPT_CACHE_REPLY:
        log.warning("batch-invariant with QWEN_PROMPT_CACHE_REPLY=1: a hit through a reply's KV "
                    "is not a cold prefill's bits; reproducible runs want REPLY=0")


def _install_fast_paths():
    """QWEN_SDPA_FUSED / QWEN_GDN_FAST / QWEN_VERIFY_EXACT_MAX_T as overrides
    of the qwen3_5 module globals / verifier methods (the MoE model uses the
    same modules). Once; -> what is on."""
    global _SDPA, _GDN_KERNEL, _VERIFY_LINEAR, _VERIFY_LINEARS, _VERIFY_ATTENTION
    from mlx_vlm.models.qwen3_5 import gated_delta, language as q35, speculative_verifier
    if cfg.SDPA_FUSED and _SDPA is None:
        _probe_fused_sdpa()
    if cfg.SDPA_FUSED and _SDPA is None:
        _SDPA, q35.scaled_dot_product_attention = q35.scaled_dot_product_attention, _fused_sdpa
    if GDN_FAST and _GDN_KERNEL is None and hasattr(mx.fast, "gated_delta_update"):
        _GDN_KERNEL, gated_delta.gated_delta_kernel = gated_delta.gated_delta_kernel, _fast_gdn_kernel
    v = speculative_verifier.Qwen3_5BatchInvariantForward
    if _VERIFY_LINEAR is None:
        _VERIFY_LINEAR, _VERIFY_LINEARS = v._linear, v._linears
        v._linear, v._linears = _verify_linear, _verify_linears
    if _VERIFY_ATTENTION is None:
        _VERIFY_ATTENTION, v._attention = v._attention, _verify_attention
    return {"sdpa_fused": _SDPA is not None, "gdn_fast": _GDN_KERNEL is not None,
            "verify_exact_max_t": VERIFY_EXACT_MAX_T}
