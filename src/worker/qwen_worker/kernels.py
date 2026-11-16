"""mlx-vlm patches and fast paths: fused SDPA, fast GDN, the DFlash verify
(mrqmm, vector attention), the drafter's fp32 mode, the fused GDN
in-projection, batch-invariant checks."""

import sys
from typing import NamedTuple

import mlx.core as mx

from . import config as cfg
from .config import (DECODE_ATT_MIN_KV, DRAFT_FP32, GDN_FAST, GDN_FAST_MIN_T, GDN_FUSE, GIB, log, PREFILL_BUDGET,
                     PROMPT_CACHE, PROMPT_CACHE_REPLY, SDPA_FUSED_MIN_KV, SDPA_FUSED_MIN_Q,
                     VERIFY_ATT, VERIFY_ATT2, VERIFY_EXACT_MAX_T, VERIFY_MR, VERIFY_MR_BLOCK,
                     VERIFY_MR_HEAD)
from .sizing import _act_dtype


_SDPA = None                          # the library's qwen3_5 scaled_dot_product_attention
_GDN_KERNEL = None                    # the library's gated_delta_kernel
_VERIFY_LINEAR = _VERIFY_LINEARS = None   # Qwen3_5BatchInvariantForward._linear / _linears
_VERIFY_ATTENTION = None                  # ... ._attention
_VERIFY_ATT_Q = 0                         # queries per call; 0 = off (set at load)
_VATT = False                             # QWEN_VERIFY_ATT=2: one-pass kernel (vatt.py), set at load
_VATT_DEC = False                         # QWEN_DECODE_ATT: decode steps through it, set at load


def _fused_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
    """qwen3_5.language.scaled_dot_product_attention with force_fused for
    long fp16 prefill slabs (QWEN_SDPA_FUSED); everything else unchanged. If
    the fused kernel fails at run time, the gate turns off; a slab whose
    unfused scores would pass PREFILL_BUDGET (_step sized it for the fused
    kernel) then fails its request instead of materializing them.

    Quantized KV (QWEN_KVQ_PREFILL_FUSED): the same slabs over a
    QuantizedKVCache take the fused kernel over the layer's keys / values
    dequantized to fp16 (_dequantized_kv); the cache keeps its packed arrays.
    Off: the library's quantized SDPA, as before."""
    global _VATT_DEC
    if (_VATT_DEC and queries.shape[2] == 1 and queries.shape[0] == 1 and mask is None
            and sinks is None and type(cache).__name__ == "KVCache" and isinstance(keys, mx.array)
            and keys.shape[2] >= DECODE_ATT_MIN_KV and queries.dtype == mx.float16):
        # QWEN_DECODE_ATT: a decode step's query (a lone row's or one row of
        # a batch: _rows_attention calls this per row) over a long fp16 KV
        # through the one-pass kernel, on the cache's own buffers
        try:
            return sys.modules["vatt"].attention(queries, cache.keys, cache.values, keys.shape[2], scale)
        except (ValueError, RuntimeError, TypeError) as e:
            _VATT_DEC = False
            log.warning("QWEN_DECODE_ATT off: %s", e)
    # cache type first: a quantized cache hands over (packed, scales, biases)
    # tuples as keys/values
    plain = type(cache).__name__ == "KVCache" and isinstance(keys, mx.array)
    deq = (not plain and cfg.KVQ_PREFILL_FUSED and type(cache).__name__ == "QuantizedKVCache"
           and type(keys) in (tuple, list) and len(keys) == 3)
    if ((plain or deq) and sinks is None and isinstance(mask, str) and mask == "causal"
            and queries.dtype == mx.float16):
        q, k = queries.shape[2], (keys if plain else keys[0]).shape[2]
        if (k >= SDPA_FUSED_MIN_KV
                and (q >= SDPA_FUSED_MIN_Q
                     # a short slab (message cut, last slab) whose scores would
                     # exceed the budget _step relies on
                     or (q > 16 and q * k > PREFILL_BUDGET))):
            if cfg.SDPA_FUSED:
                try:
                    kk, vv = (keys, values) if plain else _dequantized_kv(keys, values, cache)
                    return mx.fast.scaled_dot_product_attention(queries, kk, vv, scale=scale,
                                                                mask=mask, force_fused=True)
                except (ValueError, RuntimeError) as e:    # no fused kernel for this shape / device
                    cfg.SDPA_FUSED = False
                    log.warning("QWEN_SDPA_FUSED off: %s", e)
            if q * k > PREFILL_BUDGET:
                raise RuntimeError(
                    f"fused SDPA unavailable for a {q} x {k} prefill slab; unfused it would hold "
                    f"{q * k * queries.shape[1] * 2 / GIB:.1f} GiB of scores (> QWEN_PREFILL_BUDGET)")
    return _SDPA(queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks)


def _dequantized_kv(keys, values, cache):
    """A QuantizedKVCache's (packed, scales, biases) keys / values -> fp16
    arrays (the scales' dtype): mx.dequantize, the values the quantized
    matmuls of the library's quantized SDPA compute with. One layer's
    transient (_prefill_reserve sizes it at QWEN_MAX_CONTEXT keys)."""
    def dq(x):
        return mx.dequantize(*x, group_size=cache.group_size, bits=cache.bits)
    return dq(keys), dq(values)


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


def _qmv_t(x):
    """QWEN_VERIFY_QMV: a (1, T, K) fp16 verify block with 2 <= T <= the knob."""
    return (cfg.VERIFY_QMV and x.ndim == 3 and x.shape[0] == 1 and 2 <= x.shape[1] <= cfg.VERIFY_QMV
            and x.dtype == mx.float16)


def _qmv_rows(w, sc, b, x):
    _, T, K = x.shape
    return sys.modules["mrqmv"].rows(x.reshape(T, K), w, sc, b).reshape(1, T, -1)


def _verify_linear(self, linear, x):
    if linear.__dict__.get("_qmv") and _qmv_t(x):
        return _qmv_rows(linear["weight"], linear["scales"], linear["biases"], x)
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
    one-query calls' on a random block of this model's heads). With
    QWEN_VERIFY_ATT=2 also the one-pass kernel's self-test (_probe_vatt)."""
    global _VERIFY_ATT_Q
    _VERIFY_ATT_Q = 0
    _probe_vatt(lm)
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


def _rel_rms(a, ref):
    d = a.astype(mx.float32) - ref
    return (mx.sqrt(mx.mean(d * d)) / mx.sqrt(mx.mean(ref * ref))).item()


def _vatt_selftest(lm, Ts):
    """The one-pass kernel (vatt.py) on random blocks of this model's heads,
    every block size in Ts (each compiles its tile's pipelines now, not on the
    scheduler's first round), over a buffer with garbage past its keys and
    several chunks: -> (why it is unusable or "", the worst rms relative error).
    Usable only if every output is finite and within max(1.05 x the per-query
    calls' error, 4e-4) of an fp32 reference."""
    import vatt
    a = lm.args
    H, HK, D = a.num_attention_heads, a.num_key_value_heads, a.head_dim
    if D not in vatt.D_OK or H % HK:
        return f"head_dim {D}, {H} / {HK} heads", 0.0
    if (_act_dtype(lm) or mx.float16) != mx.float16:
        return f"activations {_act_dtype(lm)}", 0.0
    worst = 0.0
    try:
        ks = mx.random.split(mx.random.key(1), 3)
        qkeys = mx.random.split(ks[0], vatt.MAX_T + 1)
        n, cap = 600, 700
        k = mx.random.normal((1, HK, cap, D), key=ks[1]).astype(mx.float16)
        v = mx.random.normal((1, HK, cap, D), key=ks[2]).astype(mx.float16)
        k[..., n:, :] = 1000.0
        v[..., n:, :] = float("nan")
        for T in Ts:
            q = (mx.random.normal((1, H, T, D), key=qkeys[T]) * 2).astype(mx.float16)
            out = vatt.attention(q, k, v, n, D ** -0.5)
            ref = vatt.reference(q, k, v, n, D ** -0.5)
            rows = mx.concatenate(_rows_sdpa(q, k[:, :, :n], v[:, :, :n], D ** -0.5, 0, T), axis=2)
            e, e_rows = _rel_rms(out, ref), _rel_rms(rows, ref)
            worst = max(worst, e)
            if mx.any(mx.isnan(out)).item() or not e <= max(1.05 * e_rows, 4e-4):
                return f"T={T}: rms rel error {e:.2e} vs the per-query calls' {e_rows:.2e}", worst
    except (ValueError, RuntimeError, TypeError) as e:
        return f"{type(e).__name__}: {e}", worst
    return "", worst


def _probe_vatt(lm):
    """QWEN_VERIFY_ATT=2: on (-> _VATT) only where _vatt_selftest passes for
    every block size 2..vatt.MAX_T, else off (the vector kernel's path, logged)."""
    global _VATT
    _VATT = False
    if not VERIFY_ATT2:
        return False
    import vatt
    why, worst = _vatt_selftest(lm, range(2, vatt.MAX_T + 1))
    if why:
        log.warning("QWEN_VERIFY_ATT=2 off (one-pass kernel self-test failed: %s); the vector kernel's path",
                    why)
        return False
    _VATT = True
    log.info("DFlash verify attention: one-pass kernel (QWEN_VERIFY_ATT=2, vatt), blocks of 2..%d, "
             "self-test rms rel error <= %.2e", vatt.MAX_T, worst)
    return True


def _probe_decode_att(lm):
    """QWEN_DECODE_ATT: on (-> _VATT_DEC) only where _vatt_selftest passes for
    one query, else off (MLX's decode SDPA, logged)."""
    global _VATT_DEC
    _VATT_DEC = False
    if not DECODE_ATT_MIN_KV:
        return False
    why, worst = _vatt_selftest(lm, (1,))
    if why:
        log.warning("QWEN_DECODE_ATT off (one-pass kernel self-test failed: %s)", why)
        return False
    if _SDPA is None:
        log.warning("QWEN_DECODE_ATT off: the SDPA override is not installed")
        return False
    _VATT_DEC = True
    log.info("decode attention from %d keys: one-pass kernel (QWEN_DECODE_ATT, vatt), self-test rms rel "
             "error %.2e", DECODE_ATT_MIN_KV, worst)
    return True


def _verify_attention(self, attention, x, mask, cache, position_ids, position_embeddings):
    """Qwen3_5BatchInvariantForward._attention; a verify block (1, T >= 2, K)
    of a lone row over a plain fp16 KVCache attends through the one-pass
    kernel (QWEN_VERIFY_ATT=2: vatt over the cache's own buffers) or
    _VERIFY_ATT_Q queries per call (QWEN_VERIFY_ATT) instead of one. Decided
    before the cache is touched; anything else is the library's call. A
    one-pass call that raises turns it off (the vector kernel's path from then on)."""
    global _VATT
    T = x.shape[1]
    if not ((_VERIFY_ATT_Q or _VATT) and x.shape[0] == 1 and T >= 2
            and (mask is None or (isinstance(mask, str) and mask == "causal"))
            and type(cache).__name__ == "KVCache" and not hasattr(cache, "bits")):
        return _VERIFY_ATTENTION(self, attention, x, mask, cache, position_ids, position_embeddings)
    q, k, v = self._linears((attention.q_proj, attention.k_proj, attention.v_proj), x)
    queries, keys, values, gate, _ = attention._prepare_projected_qkv(
        q, k, v, cache, position_ids, position_embeddings, mask)
    out = None
    if _VATT and T <= sys.modules["vatt"].MAX_T:
        try:
            # the cache's own buffers (keys / values are their [..., :n, :] views)
            out = sys.modules["vatt"].attention(queries, cache.keys, cache.values, keys.shape[2],
                                                attention.scale)
        except (ValueError, RuntimeError, TypeError) as e:
            _VATT = False
            log.warning("QWEN_VERIFY_ATT=2 off: %s", e)
    if out is None:
        out = _block_attention(queries, keys, values, attention.scale, _VERIFY_ATT_Q or T)
    out = out.transpose(0, 2, 1, 3).reshape(1, T, -1)
    return self._linear(attention.o_proj, out * mx.sigmoid(gate))


def _verify_linears(self, linears, x):
    if linears and _qmv_t(x):
        # QWEN_VERIFY_QMV: the GDN in-projections' 4-bit width groups over
        # their concatenation, then every marked module (mrqmv.install); the
        # rest below
        out = [None] * len(linears)
        f = linears[0].__dict__.get("_qmv_fused")
        if f is not None and tuple(map(id, linears)) == f[0]:
            for idx, w, sc, b, split in f[1]:
                for i, o in zip(idx, mx.split(_qmv_rows(w, sc, b, x), split, axis=-1)):
                    out[i] = o
        for i, linear in enumerate(linears):
            if out[i] is None and linear.__dict__.get("_qmv"):
                out[i] = _verify_linear(self, linear, x)
        if any(o is not None for o in out):
            rest = [linear for linear, o in zip(linears, out) if o is None]
            rest = iter(_verify_linears(self, rest, x) if rest else ())
            return tuple(next(rest) if o is None else o for o in out)
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
    if draft is None:                   # an MTP head alone (QWEN_MTP_K + 1) / QWEN_LOOKUP's rounds
        return f"verify blocks of 3..16 tokens ({gdn} GDN layers fused)"
    if not cfg.DRAFT_BLOCK:
        cfg.DRAFT_BLOCK = VERIFY_MR_BLOCK
        draft.prefer_requested_block_size = True
    return (f"verify blocks of 3..16 tokens + drafter lm_head ({gdn} GDN layers fused), "
            f"block {cfg.DRAFT_BLOCK}")


def _setup_verify_qmv(lm):
    """QWEN_VERIFY_QMV=N, at load with a drafter (after _setup_verify_mrqmm):
    verify blocks of 2..N tokens through mrqmv on every 4 / 5 / 6 / 8-bit g64 module (and
    fused GDN in-projection) whose shape passes its self-test (each row ==
    MLX's solo qmv bit for bit). -> what the load log says (None: off)."""
    if not cfg.VERIFY_QMV:
        return None
    import mrqmv
    fused = []
    for _, m in lm.named_modules():
        gs = m.__dict__.get("_fused_in")
        if not gs:
            continue
        ls = [m[a] for a in _GDN_IN]
        groups = [(tuple(_GDN_IN.index(a) for a in g.names), g.w, g.s, g.b, g.split) for g in gs
                  if g.bits in mrqmv.BITS_OK and g.gs == 64 and g.mode == "affine"]
        if groups:
            fused.append((ls[0], groups, tuple(map(id, ls))))
    n, nf = mrqmv.install(lm, cfg.VERIFY_QMV, fused)
    cfg.VERIFY_QMV = mrqmv.MAX_ROWS if n else 0
    return (f"verify blocks of 2..{mrqmv.MAX_ROWS} tokens on mrqmv: {n} modules, {nf} fused GDN "
            f"in-projections" if n else "off: no 4 / 5 / 6 / 8-bit g64 module passed")


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


# QWEN_DRAFT_MRQMM: a row's prescale puts max |x| in [2^PRESCALE, 2^(PRESCALE+1)):
# far from fp16's max (a product with a dequantized weight stays finite even if
# the MMA rounded it to fp16 for |w| < 32) and from its subnormals (elements
# 2^-24 below the row's max still keep 11 bits).
_PRESCALE = 10.0


def _prescale_rows(x):
    """(rows, 1, K) fp32 -> (fp16 x * 2^e per row, fp32 2^-e per row): e such
    that the row's max |x| * 2^e is in [2^10, 2^11) — the power of two of the
    max's exponent field, so scaling and un-scaling are exact."""
    m = mx.max(mx.abs(x), axis=-1, keepdims=True)
    p = mx.maximum(mx.view(mx.view(m, mx.uint32) & 0x7F800000, mx.float32), 2.0 ** -100)
    return (x * (2.0 ** _PRESCALE / p)).astype(mx.float16), (p * 2.0 ** -_PRESCALE).reshape(-1)


_prescale_c = mx.compile(_prescale_rows)


def _drafter_linear(m, x):
    """QWEN_DRAFT_MRQMM: x (..., K) fp32 (or fp16) with 1..16 rows through
    mrqmm_wide on the module's fp16 copies of its scales / biases; else None."""
    f = m.__dict__.get("_mr_wide")
    K = x.shape[-1]
    rows = x.size // K
    mr = sys.modules.get("mrqmm")
    if f is None or mr is None or not 1 <= rows <= mr.MAX_ROWS or x.dtype not in (mx.float32, mx.float16):
        return None
    w, sc, b = f
    x = x.reshape(rows, 1, K)
    if x.dtype == mx.float32:
        xs, rs = _prescale_c(x)
    else:
        xs, rs = x, mx.ones((rows,), mx.float32)
    return mr.mrqmm_wide(xs, w, sc, b, rs)


def _drafter_call(self, x):
    """__call__ of a drafter's quantized linear (_install_drafter_mrqmm)."""
    y = _drafter_linear(self, x) if cfg.DRAFT_MRQMM else None
    if y is None:
        return type(self).__mro__[1].__call__(self, x)
    y = y.reshape(*x.shape[:-1], y.shape[-1])
    if "bias" in self:
        y = y + self["bias"]
    return y


def _install_drafter_mrqmm(draft, keep_fp32=True):
    """At load, after _widen_drafter: every quantized linear of the drafter
    mrqmm has a kernel for (4 / 5 / 8 bits, g64, N % 8 == 0; a 4-bit width
    once its self-test passed) whose scales and biases are exactly fp16
    (checked) gets fp16 copies of them and a __call__ that takes mrqmm_wide
    while cfg.DRAFT_MRQMM is on (read per call). keep_fp32: the fp32 scales
    stay the module's parameters (the off path unchanged; +2 B per 64
    weights); else they are replaced by the fp16 ones (quantized_matmul
    promotes them back for the rows mrqmm does not take). -> modules covered."""
    mr = sys.modules.get("mrqmm")
    if mr is None:
        return 0
    import mlx.nn as nn
    n = 0
    for _, m in draft.named_modules():
        if type(m) is not nn.QuantizedLinear or getattr(m, "mode", "affine") != "affine":
            continue
        sc, b = m["scales"], m.get("biases")
        if b is None:
            continue
        sc16, b16 = sc.astype(mx.float16), b.astype(mx.float16)
        if not (mx.array_equal(sc16.astype(sc.dtype), sc).item()
                and mx.array_equal(b16.astype(b.dtype), b).item()):
            log.info("drafter mrqmm: %s scales / biases not exact in fp16: left on quantized_matmul",
                     tuple(m["weight"].shape))
            continue
        if not (mr._layout_ok(m["weight"], sc16, b16, m.bits, m.group_size)
                and mr._tested(m.bits, m["weight"], sc16, b16, "drafter")):
            continue
        if not keep_fp32:
            m["scales"], m["biases"] = sc16, b16
        m.__dict__["_mr_wide"] = (m["weight"], sc16, b16)
        m.__class__ = _DraftQuantizedLinear
        n += 1
    return n


def _draft_linear_class():
    import mlx.nn as nn

    class _DraftQL(nn.QuantizedLinear):
        __call__ = _drafter_call
    _DraftQL.__name__ = _DraftQL.__qualname__ = "_DraftQuantizedLinear"
    return _DraftQL


_DraftQuantizedLinear = _draft_linear_class()


_GROUPED_CONV = None                  # mlx-vlm dflash2 _grouped_dynamic_convolve
_CONV_C = {}                          # group_size -> its compiled graph


def _grouped_conv(hidden, dynamic, base, group_size):
    """dflash2._grouped_dynamic_convolve, compiled per group size while
    cfg.DRAFT_COMPILE is on (the library's graph: ~8 elementwise ops per
    kernel tap, 20 calls per block on the 27B's drafter, fused into one)."""
    if not cfg.DRAFT_COMPILE:
        return _GROUPED_CONV(hidden, dynamic, base, group_size)
    f = _CONV_C.get(group_size)
    if f is None:
        import functools
        f = _CONV_C[group_size] = mx.compile(functools.partial(_GROUPED_CONV, group_size=group_size))
    return f(hidden, dynamic, base)


def _install_drafter_compile():
    """Point mlx-vlm's dflash2 module at _grouped_conv (once; it reads the
    knob per call). -> whether the module has the function."""
    global _GROUPED_CONV
    try:
        from mlx_vlm.speculative.drafters.dflash2 import dflash2 as d2
    except ImportError:
        return False
    if _GROUPED_CONV is None:
        if not hasattr(d2, "_grouped_dynamic_convolve"):
            return False
        _GROUPED_CONV = d2._grouped_dynamic_convolve
        d2._grouped_dynamic_convolve = _grouped_conv
    return True


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
    if (cfg.SDPA_FUSED or DECODE_ATT_MIN_KV) and _SDPA is None:
        # QWEN_DECODE_ATT rides on the same override (its prefill branches
        # need cfg.SDPA_FUSED; without it _step never sized a slab for them)
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
