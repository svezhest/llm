"""Prefill slab policy (_step / _bounds) and the memory budget arithmetic."""

from . import config as cfg
from .config import (BATCH_MEM_GB, CACHE_LIMIT_GB, CLEAR_CACHE, FP16, GIB, KV_KWARGS,
                     PREFILL_BUDGET, PREFILL_MIN_STEP, PREFILL_RESERVE_GB, PREFILL_STEP,
                     SDPA_FUSED_MIN_KV)


_FIXED_SLAB = None    # frozen at load (Engine.load); None = derived from the knobs (tests)


def _fixed_slab():
    """fp16 KV + fp16 queries + the fused SDPA (probed at load): slabs past
    SDPA_FUSED_MIN_KV keys hold no score matrix."""
    return KV_KWARGS["kv_bits"] is None and cfg.SDPA_FUSED and FP16


def _step(pos):
    """Slab size at cache offset `pos`. Shrunk with context where the attention
    materializes its scores (slab x context x heads: a 2048 slab at 120K
    peaked at 26.7 GB on the 9B), so that slab x pos <= PREFILL_BUDGET: the
    quantized-KV SDPA, and the default head_dim-256 SDPA below
    SDPA_FUSED_MIN_KV keys. With the fused kernel (fp16 KV, QWEN_SDPA_FUSED)
    a slab that reaches the threshold (pos + PREFILL_STEP >= 32K keys) takes
    it and holds no score matrix, so it stays PREFILL_STEP at any context;
    below it the budget rule still holds (at the defaults, 2048 and 2048 x
    32K, it never shrinks there). 9B, slab 2048 vs the old shrunk slab (546
    at 120K): x1.28 per token at 120K. The policy is frozen at load
    (_FIXED_SLAB): _bounds must cut a prompt the same way for the process's
    life (prompt-cache hits vs cold runs), whatever happens to SDPA_FUSED."""
    fixed = _FIXED_SLAB if _FIXED_SLAB is not None else _fixed_slab()
    if fixed and pos + PREFILL_STEP >= SDPA_FUSED_MIN_KV:
        return max(PREFILL_MIN_STEP, PREFILL_STEP)
    return max(PREFILL_MIN_STEP, min(PREFILL_STEP, PREFILL_BUDGET // max(pos, 1)))


def _prefill_reserve(heads):
    """Bytes kept outside the batch budget for a prefill slab's activations:
    the unfused attention scores (PREFILL_BUDGET x q heads x 2 B; with the
    fixed fp16 slab, at least PREFILL_STEP x SDPA_FUSED_MIN_KV — a short slab
    just under the fused threshold) + ~0.4 GiB of MLP. QWEN_PREFILL_RESERVE_GB
    replaces the estimate (measured peaks of a model + drafter)."""
    if PREFILL_RESERVE_GB > 0:
        return int(PREFILL_RESERVE_GB * GIB)
    fixed = _FIXED_SLAB if _FIXED_SLAB is not None else _fixed_slab()
    scores = max(PREFILL_BUDGET, PREFILL_STEP * SDPA_FUSED_MIN_KV) if fixed else PREFILL_BUDGET
    return scores * heads * 2 + (400 << 20)


def _slab(pos, left, cuts):
    """Prefill slab at absolute position `pos`, ended on the next cut."""
    n = _step(pos)
    nxt = min((c for c in cuts if c > pos), default=pos + n)
    return min(n, left, nxt - pos)


def _bounds(ids, mark):
    """Slab ends for the prefill of ids[:-1]: a message start (`mark`) at least
    PREFILL_MIN_STEP past the previous end, else a full step.

    The kernels round differently depending on the slab shape (split-K in the
    thin fp16 matmuls, SDPA over a key length that is not a multiple of the key
    block), so the same prompt cut into other slabs gives other logits, and a
    greedy answer can flip. Every end here depends only on the tokens before
    it, so a prompt resumed from a cached prefix is cut exactly as a cold one
    and the output does not depend on what the prompt cache holds."""
    out, last = [], 0
    for i in range(1, len(ids) - 1):
        if i - last >= _step(last) or (ids[i] == mark and i - last >= PREFILL_MIN_STEP):
            out.append(i)
            last = i
    return out


def _batch_cap(kernel, wired_gb):
    """Memory the batch may use: QWEN_WIRED_LIMIT_GB clamped to the kernel's
    max, else the device's recommended working set (`kernel`), which is the
    iogpu.wired_limit_mb sysctl when raised, else macOS's default share of RAM."""
    return min(int(wired_gb * GIB), kernel) if wired_gb > 0 else kernel


def _batch_budget(cap, active, prefill_reserve):
    """Bytes the running rows may reserve: QWEN_BATCH_MEM_GB, else cap - what
    is resident (weights + drafter, `active` after load) - the prefill
    activations (the attention scratch PREFILL_BUDGET x q heads x 2 B: the
    head_dim-256 SDPA fallback materializes the scores, + ~0.4 GiB of MLP) -
    1 GiB slack (logits, KV growth steps) - the buffer pool when it is kept
    across requests (QWEN_CLEAR_CACHE=0). The headroom outside the budget
    (prefill reserve + 1 GiB) is what absorbs rows that outgrow their
    QWEN_BATCH_RESERVE_TOKENS reservation. The prompt cache is not taken off
    here: its live bytes count against the budget beside the rows'
    reservations, and it gives way on demand (_Scheduler._fits / _regrow)."""
    if BATCH_MEM_GB > 0:
        return int(BATCH_MEM_GB * GIB)
    budget = (cap - active - prefill_reserve - GIB
              - (0 if CLEAR_CACHE else int(CACHE_LIMIT_GB * GIB)))
    return max(0, budget)


def _act_dtype(lm):
    """The language model's compute (and KV) dtype: its final norm's weight
    (fp16 after QWEN_FP16's cast, else bf16); None when it has none."""
    w = getattr(getattr(getattr(lm, "model", None), "norm", None), "weight", None)
    return getattr(w, "dtype", None)


def _kv_token_bytes(lm, bits):
    """Full-attention KV bytes per token of one row: fp16, or `bits`-bit values
    in uint32 words plus an fp16 scale and bias per 64-element group."""
    n_fa = sum(1 for x in lm.model.layers if not x.is_linear)
    elems = 2 * n_fa * lm.args.num_key_value_heads * lm.args.head_dim
    return elems * 2 if not bits else int(elems * (bits / 8 + 4 / 64))


def _gdn_row_bytes(lm):
    """Recurrent state of one row: conv state (fp16) + delta-rule state (fp32)."""
    n = 0
    for layer in lm.model.layers:
        if layer.is_linear:
            la = layer.linear_attn
            n += (la.conv_kernel_size - 1) * la.conv_dim * 2 \
                + la.num_v_heads * la.head_v_dim * la.head_k_dim * 4
    return n


def _weight_bytes(lm):
    """Weight bytes a decode step reads: all of the language model but the
    embedding table (a gather), unless the head is tied to it."""
    from mlx.utils import tree_flatten
    params = tree_flatten(lm.parameters())
    tied = not any(k.startswith("lm_head") for k, _ in params)
    return sum(v.nbytes for k, v in params if tied or "embed_tokens" not in k)
