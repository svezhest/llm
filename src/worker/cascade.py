"""Cascade decode attention for batch rows that share a token prefix.

A batched decode step today attends every row over its own contiguous
KVCache: per row `mx.fast.scaled_dot_product_attention(q_i (1,Hq,1,D),
K_i (1,Hkv,P+S_i,D), V_i)`, so B rows sharing a P-token prefix (a common system
prompt restored from the prompt cache) read that prefix B times. A *group*
holds the prefix once and reads it once for all its rows; each row keeps only
its own suffix. Design: docs/perf/2026-10-12/design/cascade_plan.md (Part B),
measured as `b32c` / `b32u` in docs/perf/2026-10-12/cascade_microbench.md
(9B, B=16, P=8K, S=512: attention + update 22-24 -> ~9 ms per 8-layer step).

Formulation (`cascade`, one full-attention layer, B_g rows, rep = Hq // Hkv):

    qg = fp32(q) * scale                           (B_g, Hkv, rep, D)   q head h -> kv head h // rep
    qp = qg -> (Hkv, 1, B_g*rep, D)                 row index b*rep + r (kv head major)
    prefix, chunked (flash-decoding split, chunk Pc = 1024):
      s     = qp @ Kc^T [+ pmask]                  (Hkv, c, B_g*rep, Pc)  fp32
      lse_c = logsumexp(s);  o_c = softmax(s) @ Vc (Hkv, c, B_g*rep, 1 / D)
      lse_p = logsumexp_c(lse_c);  o_p = sum_c exp(lse_c - lse_p) o_c
    suffix, stacked per row with a length mask:
      ss    = qg @ Ks^T + smask                    (B_g, Hkv, rep, cap)  fp32
      lse_s, o_s the same two ops (o_s := 0 when a row's suffix is empty)
    merge: m = max(lse_p, lse_s); w = exp(lse - m)
      out = (w_p o_p + w_s o_s) / (w_p + w_s) -> (B_g, Hq, 1, D) in q's dtype

The whole function runs under `mx.compile` (fuses the merge chain; the
microbench: 5-15 % at B=16).

Data (`PrefixGroup`), per full-attention layer key `i` (any hashable, the
worker uses the model's layer index):
  * kc[i], vc[i]: the prefix [0, P) as fp32 (Hkv, c, Pc, D) = a reshape of one
    contiguous (1, Hkv, Ppad, D) buffer, Ppad = c * Pc. When P is not a chunk
    multiple the tail [P, Ppad) holds zeros and `pmask` (1, c, 1, Pc) fp32 is
    -inf there (0 elsewhere); pmask is None when P % Pc == 0. Built ONCE per
    group from one source row's cache.
  * sk[i], sv[i]: the members' suffixes [P, P + S_j) stacked as fp32
    (n, Hkv, cap, D), slots >= S_j hold zeros. cap is a multiple of `step`
    (256, as KVCache) with room for at least one more token; `grow` adds
    zeros, `take` re-takes rows (drop / reorder) and trims cap.
  * lens: the S_j (Python ints; `g.lens` is a read-only tuple). It is a second
    copy of the scheduler's `row.pos - P`: B2's `_plan` asserts the two agree
    with `g.check_lens([r.pos - g.P for r in g.rows])` every step.
Per decode step (the B2 caller, `_rows_attention`):

    g.begin_step()                        # once, before the forward is built: grows cap if
                                          #   needed, checks the bounds, builds the write
                                          #   index + suffix mask
    per FA layer i:                       #   (the mask attends [0, S_j + 1): the new token)
        g.append(i, keys[idx], values[idx])          # (n, Hkv, 1, D) in kv_dtype: one put_along_axis per K/V
        out = attention(queries[idx], g, i, scale)   # (n, Hq, 1, D)
    g.end_step()                          # after the forward is built: S_j += 1
    (g.abort_step() if building the step failed on the host: zeroes the slot again)

The protocol is enforced with a per-step set of appended layers (pure Python,
free): a second `append` of a layer, `attention` on a layer not yet appended in
this step, and `end_step` with a layer missing all raise RuntimeError. A
skipped append would otherwise leave a zero key in a slot the mask attends:
score 0, weight exp(0 - m) on a zero value, silently diluting that row's
attention in every later step (no NaN; the unused-slot checks cannot see it,
the slot counts as used). After a refused `end_step` the step is still open:
`abort_step` it. `grow` / `add` / `take` / `release` refuse inside a step.

Bounds: `put_along_axis` does not check its indices -- an out-of-range write
index killed the CPU test process with SIGBUS and on Metal would silently
overwrite a neighbouring buffer (another layer's KV, weights, GDN state). Safe
by construction (begin_step grows to max(lens) + 1), and asserted anyway:
`begin_step` raises before anything is written unless max(lens) < cap and
every stack is (n, Hkv, cap, D).

`abort_step` covers HOST-side build errors only (a Python exception while the
step's graph is being built). It writes zeros over the appended slots on top
of the step's lazy graph and evaluates the stacks at once, so the aborted
forward up to the last appended layer runs once more (wasted GPU time, no
harm). If that eval raises -- the failure was inside the graph the keys depend
on (OOM, an eval-time error) -- every later step would depend on the same
poisoned graph: `abort_step` raises, the group refuses further steps, and the
caller drops it (the scheduler's `_fail_rows` drops every group anyway, plan
B.4, so B3 may never call `abort_step`). A snapshot of the pre-step stacks
would avoid the re-run but costs every step its donation (a second reference).

Conversion: `PrefixGroup.build(source, members, P, layers)` from rows' contiguous
KVCaches ([0, P) from `source`, [P, offset) of each member);
`release(j)` gives row j back as contiguous KVCaches in the rows' dtype:
prefix (fp32 -> kv dtype) + suffix (fp32 -> kv dtype) + `step` zeros of
capacity. Both casts are exact: the prefix was fp16 (or the row's dtype) before
the widening, and every suffix value -- copied from the row's cache at build or
written by `append` from the projections' fp16 output -- was a kv-dtype value
before it was stored as fp32. So a row that leaves a group has the bits it
would have had in its own cache (tests/test_cascade.py checks it bitwise).

Why fp32 (the microbench, numerics section):
  * fp16 scores fail: the matmul output rounds s to fp16 (ulp ~0.005 at |s| ~ 10)
    and that error goes straight into the softmax weights (`b16`: 2x today's
    error on flat scores, 4-10x on peaked ones).
  * The statistics (scores, LSE, merge weights) must be fp32: an fp16 LSE has an
    ulp of ~0.008 at 8-16, a 0.4-0.8 % merge-weight error (`mx.logsumexp` of an
    fp16 input returns fp16, so the inputs are cast first).
  * Partials are normalized (o_c, o_s + their LSE), not unnormalized sums, which
    would overflow fp16 and lose range in the merge.
  * Products are identical to MLX's decode kernel (fp16 -> fp32 is exact, the
    scale 1/sqrt(D) is a power of two for D = 64 / 256); only the summation
    order differs, and the single final rounding to fp16 leaves the result
    closer to an fp32 reference than the per-row kernel (rms rel ~2.1e-4 vs
    ~2.8e-4 on the 9B layout). Not bit-equal to a lone row: B <= 2 must stay on
    the per-row path (the worker's "B <= 2 bit-exact" invariant).
Storage is fp32 too (64 KiB/token on the 9B instead of 32): an fp16 store
would need an fp32 cast of the whole buffer every layer and step.

Pitfalls (handled here, keep them when changing this file):
  * GQA layout: MLX maps q head h to kv head h // rep (contiguous repeats); the
    stacked queries are (kv head, b*rep + r) and are transposed back the same
    way. A wrong order still looks plausible -- tests compare against per-row
    SDPA on random per-head data.
  * Unused slots (prefix padding, suffix slots >= S_j) must hold ZEROS, not
    garbage: the additive -inf mask gives them weight exactly 0, but 0 * NaN =
    NaN in P.V and in the scores. The masks make any FINITE garbage weightless
    (bitwise), but NaN / inf would leak, so the holders never copy anything
    past a row's offset (whatever a source KVCache holds there) and only ever
    write zeros or real K/V into their buffers. `abort_step` zeroes the slot a
    failed step may have written.
  * An empty suffix (S_j = 0 outside a step) makes the suffix softmax all -inf
    (NaN); its weight is exp(-inf) = 0 and o_s is replaced by 0 for it.
  * mx.compile is shape-specialized: a new (B_g, c, cap) retraces. One compiled
    wrapper per shape key lives in an LRU of MAX_TRACES (64); evicting the
    wrapper drops its trace (mlx erases a compiled function's cache when the
    function is destroyed; a new closure may reuse the evicted one's address,
    so its `fun_id`, and still gets a fresh trace -- tests run MAX_TRACES = 1
    alternating 4 scales). Distinct keys over a group's life: B_g changes on
    joins / leaves (<= max_batch values), c is fixed per group, cap moves in
    `step` (256) increments -- they follow a path (cap grows while B_g
    shrinks), ~31 keys over a 16-row group decoding 4K tokens; a new key costs
    ~0.2 ms of host trace time. The key includes the default DEVICE, not the
    stream: the worker traces and runs everything under one stream on one
    thread; a tool calling `attention` under another stream shares the trace.
  * Donation: `append` rebinds sk[i] / sv[i] to the put_along_axis result and
    nothing else may hold the old array, so MLX updates the buffer in place.
    An extra live reference (a stray `g.sk[i]` kept by a caller, a lazy
    `release(evaluate=False)` not yet evaluated) makes that append copy the
    whole layer stack (+8 ms per step in the microbench's run 1).
    Under the scheduler's step-ahead (step k+1 built and queued while step k
    runs on the GPU) donation still works and stays correct, but NOT because
    step k+1 depends on step k's token: donation is decided on the host at
    encode time (`array::is_donatable`: one reference to the array and to its
    data), and MLX's Metal eval keeps every input's data referenced from the
    command buffer until that buffer completes. While step k's buffer reading
    sk_k is in flight, sk_k is not donatable and step k+1's scatter COPIES
    (slower, never corrupt) -- the mechanism that also protects KVCache's
    slice updates. MLX keeps <= 10 command buffers (~50 ops each on M1 Max)
    in flight, so layer i's reader of step k has completed when step k+1's
    scatter is encoded only if a step spans more ops than that window.
    Measured (M1 Max, B_g 16, P 8K, 8 FA layers, elementwise filler ops per
    layer standing in for the rest of the model): filler 0 -> every stack
    copied, peak +1.06 GB vs +0.17 sync, 14.0 vs 9.6 ms/step; 40 -> +0.47 GB;
    80 / 160 -> donated (+128 / +85 MiB vs +117 / +72 sync; step-ahead then
    faster than sync, 11.3 vs 12.4 / 14.0 vs 15.5 ms). The 9B step (3 GDN
    layers + 4 MLPs between two FA layers) is far past that window; a harness
    timing attention alone with step-ahead (B4 step_prof) is not, and measures
    the copies. Tests: step-ahead outputs bitwise equal to a synchronous run;
    the GPU smoke asserts the step-ahead peak (with filler) within one layer's
    K+V stack of the synchronous one.
  * Transient per layer and step: the prefix scores and probabilities, (Hkv,
    c, B_g*rep, Pc) fp32 each = 4 * Hq * B_g * Ppad bytes, plus the suffix's
    (~27 MiB per layer at 9B/16 rows/8K/cap 4352, so ~0.2 GB per 8-layer step;
    ~0.6 GB at 32K). The worker charges it in its group reservation (plan B.5).
  * Transient of membership changes (9B: fp32 K+V = 64 KiB/token; a 16-row
    stack at cap 4352 = 4.25 GiB over 8 layers, 272 MiB per K or V per layer):
    `take` / `add` / `grow` build each layer's new stack from the old one.
    They are LAZY by default (`evaluate=False`; `add` still evaluates the
    joining row's own S-token fp32 copies, so they do not pin the row's
    contiguous cache until the next step): the old stack is then
    referenced by the graph only, the copy is donatable, and the next step's
    `append` of each layer consumes it layer by layer -- the transient is about
    one layer's old + new K and V, and nothing waits for the step in flight.
    `evaluate=True` (tests) evaluates every layer while sk / sv still hold the
    old arrays: old and new stacks of ALL layers live at once (+4 GiB at 16
    rows x cap 4352; a scaled CPU copy measured +448 vs +48 MiB lazy), and
    the host waits for the in-flight step whose output the copy reads. Lazy
    means the old stacks live until that next step is evaluated, and a lazy
    `release(evaluate=False)` keeps sk[i] referenced (and costs the next
    append its donation) until the released caches are evaluated -- which is
    why `release` evaluates by default (it is rare). `begin_step`'s `grow` is
    lazy too: per layer the old K+V is alive during the concat (~0.5 GiB at 16
    rows, cap 4096 -> 4352). Plan B.5 must charge one layer's old + new K+V on
    top of `_group_bytes` (its 5 % slack is ~0.2 GiB). With lazy copies the
    "an exception leaves the group as it was" promise covers the host-side
    checks only: an OOM in a copy surfaces in the step's eval (the scheduler
    then drops every group, `_fail_rows`).
  * Memory break-even (input for B3's dissolve / leave rule, not this
    module's): grouped = P * 64 KiB + B * S * 64 KiB (fp32 prefix + fp32
    stack, 9B; S = the stack's cap, ragged rows pay the longest), per row =
    B * (P + S) * 32 KiB. A group saves memory only while S < P * (B - 2) / B:
    0.875 P at B = 16 but P / 3 at B = 3. The plan's leave rule (S_j > P // 2)
    and dissolve-at-2 keep 3..7-row groups in the band that uses MORE memory
    than per-row caches -- where speed is also weakest (suffix GEMM M = rep,
    few rows to amortise the prefix). Accounting stays correct (B.5 charges
    it: efficiency, not OOM). B3: make the rule memory-aware
    (`_group_bytes(g) > sum(_bytes(r) for r in g.rows)` -> dissolve) or raise
    the dissolve threshold above 2 rows.
  * The suffix part has M = rep rows per GEMM (padded to 64-row tiles), so it
    is slower than the per-row kernel over the same tokens: group only when the
    prefix dominates (plan: >= 8 rows, P >= 4096, leave when S_j > P // 2).

Kernel mode (`build(..., kernel=True)`, the worker's QWEN_CASCADE_KERNEL;
docs/perf/2026-10-24/design/cascade_mma.md): the prefix stays fp16 and its
attention is cascade_mma's Metal kernel (simdgroup MMA, fp32 accumulation,
online softmax, split over the prefix + a merge pass) returning (o_p, lse_p);
the suffix part and the merge are `merge` (the same ops as `cascade`, under
mx.compile). kc[i] / vc[i] are then the source row's own (1, Hkv, cap, D)
KVCache buffers, aliased when cap - P <= `slack` (ALIAS_SLACK: no formation
copy, no transient; the group pins the buffer, its [P, cap) unused), else a
fresh (1, Hkv, P + 1, D) copy; c = 0, Ppad = P, pmask None. Only the stacks,
lens, masks and the step protocol are shared with the fp32 mode, so append /
take / grow / release behave the same; `release` and `prefix_kv` hand out the
fp16 prefix as is (exact). Memory per token of prefix: 32 KiB on the 9B
instead of 64, and no (Hkv, c, B_g*rep, Pc) fp32 scores (the kernel's
partials are (items, Hkv, B_g*rep, D + 2) fp32 per layer instead). fp16 KV
only (bf16 / fp32 raise at build). On a CPU default device cascade_mma falls
back to mx ops (`cascade_mma.reference`): the CPU tests run this mode's
bookkeeping; the kernel itself is tests/test_cascade_mma.py.

Env knobs the worker reserves (plan B.1; this module takes explicit
parameters and reads no environment): QWEN_CASCADE (switch),
QWEN_CASCADE_MIN_ROWS (8), QWEN_CASCADE_MIN_PREFIX (4096), QWEN_CASCADE_CHECK;
the prefix chunk (CHUNK = 1024, the microbench's best of 512..4096) and the
suffix growth step (STEP = 256) are this module's defaults.
"""
from __future__ import annotations

import math
import threading
from collections import OrderedDict

import mlx.core as mx
from mlx_vlm.models.cache import KVCache

import cascade_mma

F32 = mx.float32
CHUNK = 1024          # prefix chunk (flash-decoding split); P.V unchunked is ~1.7x slower at 32K
STEP = 256            # suffix stack growth, as KVCache.step
ALIAS_SLACK = 1024    # kernel mode: a donor buffer with <= this many slots past P is aliased, else copied
MAX_TRACES = 64       # compiled wrappers kept (one per shape key)

_NEG = -math.inf


def _round_up(n: int, m: int) -> int:
    return -(-n // m) * m


# ---------------------------------------------------------------- the kernel


def cascade(q, kc, vc, pmask, sk, sv, smask, scale):
    """One layer's attention for B_g rows over a shared chunked prefix and
    their stacked suffixes (uncompiled; `attention` runs it under mx.compile).

    q      (B_g, Hq, 1, D) any float dtype (queries after norm + RoPE)
    kc/vc  (Hkv, c, Pc, D) fp32 prefix chunks, zeros past P
    pmask  (1, c, 1, Pc) fp32 additive (0 / -inf past P) or None
    sk/sv  (B_g, Hkv, cap, D) fp32 suffix stack, zeros in unused slots
    smask  (B_g, 1, 1, cap) fp32 additive (0 for slots < the row's length)
    Returns (B_g, Hq, 1, D) in q's dtype."""
    B, Hq, L, D = q.shape
    Hkv = kc.shape[0]
    rep = Hq // Hkv
    qg = (q.astype(F32) * scale).reshape(B, Hkv, rep, D)                 # head h -> kv h // rep
    qp = qg.transpose(1, 0, 2, 3).reshape(Hkv, 1, B * rep, D)            # row b*rep + r
    s = qp @ kc.swapaxes(-1, -2)                                         # (Hkv, c, B*rep, Pc)
    if pmask is not None:
        s = s + pmask
    lse_c = mx.logsumexp(s, axis=-1, keepdims=True)
    o_c = mx.softmax(s, axis=-1, precise=True) @ vc                      # (Hkv, c, B*rep, D)
    lse_p = mx.logsumexp(lse_c, axis=1, keepdims=True)                   # (Hkv, 1, B*rep, 1)
    o_p = (mx.exp(lse_c - lse_p) * o_c).sum(axis=1, keepdims=True)
    ss = qg @ sk.swapaxes(-1, -2) + smask                                # (B, Hkv, rep, cap)
    lse_s = mx.logsumexp(ss, axis=-1, keepdims=True)
    o_s = mx.softmax(ss, axis=-1, precise=True) @ sv                     # (B, Hkv, rep, D)
    o_s = mx.where(lse_s == _NEG, 0.0, o_s)                              # empty suffix: NaN -> 0
    o_s = o_s.transpose(1, 0, 2, 3).reshape(Hkv, 1, B * rep, D)
    lse_s = lse_s.transpose(1, 0, 2, 3).reshape(Hkv, 1, B * rep, 1)
    m = mx.maximum(lse_p, lse_s)
    wp, ws = mx.exp(lse_p - m), mx.exp(lse_s - m)
    out = (wp * o_p + ws * o_s) / (wp + ws)                              # (Hkv, 1, B*rep, D)
    out = out.reshape(Hkv, B, rep, D).transpose(1, 0, 2, 3)
    return out.reshape(B, Hq, L, D).astype(q.dtype)


def merge(q, op, lse, sk, sv, smask, scale):
    """QWEN_CASCADE_KERNEL's half of `cascade` (uncompiled; `attention` runs
    it under mx.compile): the prefix part comes from cascade_mma as o_p
    (Hkv, B_g*rep, D) fp32 + lse_p (Hkv, B_g*rep) fp32; the suffix part and
    the merge are `cascade`'s own ops, line for line."""
    B, Hq, L, D = q.shape
    Hkv = sk.shape[1]
    rep = Hq // Hkv
    qg = (q.astype(F32) * scale).reshape(B, Hkv, rep, D)
    o_p = op.reshape(Hkv, 1, B * rep, D)
    lse_p = lse.reshape(Hkv, 1, B * rep, 1)
    ss = qg @ sk.swapaxes(-1, -2) + smask                                # (B, Hkv, rep, cap)
    lse_s = mx.logsumexp(ss, axis=-1, keepdims=True)
    o_s = mx.softmax(ss, axis=-1, precise=True) @ sv                     # (B, Hkv, rep, D)
    o_s = mx.where(lse_s == _NEG, 0.0, o_s)                              # empty suffix: NaN -> 0
    o_s = o_s.transpose(1, 0, 2, 3).reshape(Hkv, 1, B * rep, D)
    lse_s = lse_s.transpose(1, 0, 2, 3).reshape(Hkv, 1, B * rep, 1)
    m = mx.maximum(lse_p, lse_s)
    wp, ws = mx.exp(lse_p - m), mx.exp(lse_s - m)
    out = (wp * o_p + ws * o_s) / (wp + ws)                              # (Hkv, 1, B*rep, D)
    out = out.reshape(Hkv, B, rep, D).transpose(1, 0, 2, 3)
    return out.reshape(B, Hq, L, D).astype(q.dtype)


_FNS: OrderedDict = OrderedDict()
_LOCK = threading.Lock()


def _key(scale, q, kc, pmask, sk):
    return (float(scale), pmask is not None, str(mx.default_device()), tuple(q.shape),
            q.dtype, tuple(kc.shape), tuple(sk.shape))


def compiled(scale, q, kc, pmask, sk):
    """The mx.compile'd `cascade` for these shapes (LRU of MAX_TRACES wrappers,
    one trace each; see the module docstring)."""
    key = _key(scale, q, kc, pmask, sk)
    with _LOCK:
        fn = _FNS.get(key)
        if fn is not None:
            _FNS.move_to_end(key)
            return fn
        if pmask is None:
            fn = mx.compile(lambda q, kc, vc, sk, sv, sm: cascade(q, kc, vc, None, sk, sv, sm, scale))
        else:
            fn = mx.compile(lambda q, kc, vc, pm, sk, sv, sm: cascade(q, kc, vc, pm, sk, sv, sm, scale))
        _FNS[key] = fn
        while len(_FNS) > MAX_TRACES:
            _FNS.popitem(last=False)
        return fn


def compiled_merge(scale, q, sk):
    """The mx.compile'd `merge` for these shapes (same LRU as `compiled`)."""
    key = ("merge", float(scale), str(mx.default_device()), tuple(q.shape), q.dtype, tuple(sk.shape))
    with _LOCK:
        fn = _FNS.get(key)
        if fn is not None:
            _FNS.move_to_end(key)
            return fn
        fn = mx.compile(lambda q, op, lse, sk, sv, sm: merge(q, op, lse, sk, sv, sm, scale))
        _FNS[key] = fn
        while len(_FNS) > MAX_TRACES:
            _FNS.popitem(last=False)
        return fn


def traces() -> int:
    """Compiled wrappers currently kept (tests, logs)."""
    return len(_FNS)


def clear_traces():
    with _LOCK:
        _FNS.clear()


def attention(queries, group: "PrefixGroup", layer, scale):
    """One FA layer of a group: queries (B_g, Hq, 1, D) in member order ->
    (B_g, Hq, 1, D) in the queries' dtype. Inside a step (`begin_step`) the
    mask attends each row's suffix plus the slot `append` just wrote."""
    if queries.ndim != 4 or queries.shape[2] != 1 or queries.shape[0] != group.n:
        raise ValueError(f"cascade.attention: queries {tuple(queries.shape)} for a group "
                         f"of {group.n} rows (decode only: (B_g, Hq, 1, D))")
    if queries.shape[1] % group.Hkv or queries.shape[3] != group.D:
        raise ValueError(f"cascade.attention: {queries.shape[1]} q heads / head_dim "
                         f"{queries.shape[3]} vs {group.Hkv} kv heads / {group.D}")
    if group._pending and layer not in group._appended:
        raise RuntimeError(f"cascade: attention() on layer {layer!r} before its append() in this step")
    kc, vc, sk, sv = group.kc[layer], group.vc[layer], group.sk[layer], group.sv[layer]
    smask = group.smask()
    if group.kernel:
        op, lse = cascade_mma.prefix_attention(queries, kc, vc, group.P, scale)
        return compiled_merge(scale, queries, sk)(queries, op, lse, sk, sv, smask)
    fn = compiled(scale, queries, kc, group.pmask, sk)
    if group.pmask is None:
        return fn(queries, kc, vc, sk, sv, smask)
    return fn(queries, kc, vc, group.pmask, sk, sv, smask)


# ---------------------------------------------------------------- the group


def _fa(cache, P):
    """A member's FA cache: (keys, values, offset) after the checks."""
    if not isinstance(cache, KVCache) or cache.keys is None:
        raise ValueError(f"cascade: a member's FA cache is {type(cache).__name__}, not a filled KVCache")
    if cache.keys.shape[0] != 1:
        raise ValueError(f"cascade: a member cache has batch {cache.keys.shape[0]}, not 1")
    if cache.offset < P:
        raise ValueError(f"cascade: a member cache holds {cache.offset} tokens < prefix {P}")
    return cache.keys, cache.values, cache.offset


class PrefixGroup:
    """The KV of rows sharing a token prefix [0, P): the prefix once (fp32,
    chunked), the members' suffixes stacked (fp32). Layout and invariants in
    the module docstring. Every mutating method checks and builds its new
    arrays first and assigns them at the end, so a host-side exception leaves
    the group as it was; with lazy copies (the default for take / add / grow)
    an eval-time failure (OOM) surfaces in the next step's eval instead.

    Attributes: P, chunk (Pc), c, Ppad, step, layers (FA layer keys), Hkv, D,
    kv_dtype (the rows' KV dtype), kc / vc / sk / sv (dict layer -> array),
    pmask, lens (read-only tuple: suffix lengths in member order), cap, n,
    kernel / aliased (kernel mode, module docstring)."""

    def __init__(self):            # use PrefixGroup.build
        self.P = self.chunk = self.c = self.Ppad = self.step = 0
        self.layers: list = []
        self.Hkv = self.D = 0
        self.kv_dtype = None
        self.kc: dict = {}
        self.vc: dict = {}
        self.sk: dict = {}
        self.sv: dict = {}
        self.pmask = None
        self.kernel = False             # QWEN_CASCADE_KERNEL: fp16 prefix, cascade_mma
        self.aliased = False            # kernel mode: kc / vc are the donor row's own buffers
        self._slack = 0
        self._lens: list[int] = []
        self.cap = 0
        self._pending = False
        self._appended: set = set()     # layers appended in the open step
        self._failed = False            # abort_step's eval failed: the group is unusable
        self._upos = None
        self._smask = None

    # ---- construction

    @classmethod
    def build(cls, source, members, P: int, layers=None, chunk: int = CHUNK,
              step: int = STEP, evaluate: bool = True, kernel: bool = False,
              slack: int = ALIAS_SLACK) -> "PrefixGroup":
        """A group over `members` (each a row's cache list, indexable by the
        layer keys) with the prefix [0, P) taken from `source` (one row's
        cache list, typically members[0]; the other members' [0, P) are NOT
        read -- they must hold the same bits, see `prefix_equal`). `layers`:
        the FA layer keys (default: every index of `source`). Nothing past a
        cache's offset (nor past P in `source`) is read. Evaluates by default:
        there is no old stack to keep alive, and the rows' caches can only be
        released once the copies exist (formation transient, plan B.10).

        `kernel` (QWEN_CASCADE_KERNEL; fp16 KV only): the prefix stays fp16
        and is attended by cascade_mma. kc[i] / vc[i] are then `source`'s own
        (1, Hkv, cap, D) buffers, aliased (no copy: nothing is allocated and
        no formation transient) when at most `slack` slots lie past P,
        else a fresh (1, Hkv, P + 1, D) copy of [0, P) (+ one zero slot) --
        either way the kernel reads only [0, P). `chunk` and `pmask` are
        unused (c = 0, Ppad = P)."""
        if P <= 0:
            raise ValueError("cascade: empty prefix")
        if not members:
            raise ValueError("cascade: a group needs members")
        layers = list(range(len(source)) if layers is None else layers)
        if not layers:
            raise ValueError("cascade: no layers")
        g = cls()
        k0, _, _ = _fa(source[layers[0]], P)
        g.P, g.chunk, g.step, g.layers = P, chunk, step, layers
        g.Hkv, g.D, g.kv_dtype = k0.shape[1], k0.shape[3], k0.dtype
        g.kernel = bool(kernel)
        if g.kernel:
            if g.kv_dtype != mx.float16:
                raise ValueError(f"cascade: the MMA kernel takes an fp16 KV, not {g.kv_dtype}")
            g.chunk, g.c, g.Ppad = 0, 0, P
            g._slack = slack
            g.aliased = source[layers[0]].keys.shape[2] - P <= slack
        else:
            g.c = -(-P // chunk)
            g.Ppad = g.c * chunk
        lens = []
        for mem in members:
            offs = {_fa(mem[i], P)[2] for i in layers}
            if len(offs) != 1:
                raise ValueError(f"cascade: a member's FA caches disagree on the offset: {sorted(offs)}")
            lens.append(offs.pop() - P)
        cap = _round_up(max(lens) + 1, step)
        kc, vc, sk, sv = {}, {}, {}, {}
        for i in layers:
            ks, vs, _ = _fa(source[i], P)
            g._check_shape(ks, vs)
            kc[i] = g._prefix16(ks) if g.kernel else g._prefix32(ks)
            vc[i] = g._prefix16(vs) if g.kernel else g._prefix32(vs)
            sk[i] = g._stack([_fa(m[i], P) for m in members], lens, cap, keys=True)
            sv[i] = g._stack([_fa(m[i], P) for m in members], lens, cap, keys=False)
        pmask = None
        if not g.kernel and g.Ppad != P:
            pmask = mx.where(mx.arange(g.Ppad) < P, 0.0, _NEG).astype(F32).reshape(1, g.c, 1, chunk)
        if evaluate:
            mx.eval(kc, vc, sk, sv, pmask)
        g.kc, g.vc, g.sk, g.sv, g.pmask = kc, vc, sk, sv, pmask
        g._lens, g.cap = lens, cap
        return g

    def _check_shape(self, k, v):
        if k.shape[1] != self.Hkv or k.shape[3] != self.D or k.dtype != self.kv_dtype \
                or v.shape[:2] != k.shape[:2] or v.shape[3] != self.D or v.dtype != self.kv_dtype:
            raise ValueError(f"cascade: cache shape/dtype {tuple(k.shape)} {k.dtype} / "
                             f"{tuple(v.shape)} {v.dtype} vs heads {self.Hkv}, head_dim "
                             f"{self.D}, {self.kv_dtype}")

    def _prefix32(self, x):
        """[0, P) of a (1, Hkv, cap, D) buffer -> fp32 (Hkv, c, Pc, D), zeros
        past P. Always a fresh buffer (a same-dtype astype of a slice would be
        a view pinning the source row's buffer)."""
        x = x[..., :self.P, :]
        pad = mx.zeros((1, self.Hkv, self.Ppad - self.P, self.D), x.dtype)
        x = mx.concatenate([x, pad], axis=2)
        return x.astype(F32).reshape(self.Hkv, self.c, self.chunk, self.D)

    def _prefix16(self, x):
        """Kernel mode: [0, P) of a (1, Hkv, cap, D) fp16 buffer for
        cascade_mma. The buffer itself when `aliased` (cap - P <= slack at
        build; every FA layer's buffer of a KVCache has the same cap)
        (mx.contiguous of a row-contiguous array shares its buffer: a KVCache
        buffer is one), so the group pins the donor row's buffer instead of
        copying it; else a (1, Hkv, P + 1, D) copy (a concatenate always
        writes a fresh buffer; the extra slot is a zero) so that a long donor
        suffix is not kept alive. Bits unchanged either way."""
        if self.aliased:
            if x.shape[2] - self.P > self._slack:
                raise ValueError(f"cascade: a donor buffer of {x.shape[2]} slots past "
                                 f"{self.P} + {self._slack}")
            return mx.contiguous(x)
        z = mx.zeros((1, self.Hkv, 1, self.D), x.dtype)
        return mx.concatenate([x[..., :self.P, :], z], axis=2)

    def _stack(self, fas, lens, cap, keys):
        """Members' [P, offset) of one layer -> fp32 (n, Hkv, cap, D), zeros
        past each length (slice updates into one zeros buffer, donated)."""
        out = mx.zeros((len(fas), self.Hkv, cap, self.D), F32)
        for j, ((k, v, _), S) in enumerate(zip(fas, lens)):
            self._check_shape(k, v)
            if S:
                x = k if keys else v
                out[j:j + 1, :, :S, :] = x[..., self.P:self.P + S, :].astype(F32)
        return out

    # ---- properties

    @property
    def n(self) -> int:
        return len(self._lens)

    @property
    def lens(self) -> tuple:
        """The members' suffix lengths S_j (member order), read-only."""
        return tuple(self._lens)

    def check_lens(self, expected):
        """Raise unless the group's suffix lengths equal `expected` (the
        scheduler's [r.pos - P for r in members]): the two copies of the
        write index must never drift apart (module docstring, Bounds)."""
        if list(expected) != self._lens:
            raise RuntimeError(f"cascade: group lengths {self._lens} != the rows' {list(expected)}")

    @property
    def nbytes(self) -> int:
        """Bytes of the group's own arrays (prefix + stacks + mask)."""
        arrs = [*self.kc.values(), *self.vc.values(), *self.sk.values(), *self.sv.values()]
        return sum(a.nbytes for a in arrs) + (self.pmask.nbytes if self.pmask is not None else 0)

    def _no_step(self, what):
        if self._pending:
            raise RuntimeError(f"cascade: {what} inside a decode step")

    def _changed(self):
        self._upos = self._smask = None

    # ---- membership

    def add(self, member, evaluate: bool = False) -> int:
        """Append a row (its cache list) to the stack; returns its index. Its
        [0, P) is not read (see `prefix_equal`). The row's own suffix [P,
        offset) is copied to fp32 and EVALUATED here (S tokens per layer,
        small): left lazy, those slices would keep the row's whole contiguous
        cache alive until the next step evaluated the stack (~1 GB per row
        at 32K on the 9B, uncharged once the scheduler moved the row's charge
        to the group, and chained over back-to-back joins; review
        cascade_b2b3 finding 3). The stack concat is lazy by default: the
        next step's appends consume the new stacks layer by layer (module
        docstring, transient of membership changes); `evaluate=True`
        evaluates it now too."""
        self._no_step("add")
        offs = {_fa(member[i], self.P)[2] for i in self.layers}
        if len(offs) != 1:
            raise ValueError(f"cascade: a member's FA caches disagree on the offset: {sorted(offs)}")
        S = offs.pop() - self.P
        cap = max(self.cap, _round_up(S + 1, self.step))
        # S + 1 slots (the last one zero): an update covering the whole
        # zeros buffer may come back as the source slice itself (a view of
        # the row's buffer, measured with an fp32 cache); a partial one is
        # always a fresh buffer
        own = {}
        for i in self.layers:
            fa = _fa(member[i], self.P)
            own[i] = (self._stack([fa], [S], S + 1, keys=True),
                      self._stack([fa], [S], S + 1, keys=False))
        mx.eval(own)                    # nothing of the row's is referenced after this
        sk, sv = {}, {}
        for i in self.layers:
            new_k, new_v = (self._pad(x, cap) for x in own[i])
            sk[i] = mx.concatenate([self._pad(self.sk[i], cap), new_k], axis=0)
            sv[i] = mx.concatenate([self._pad(self.sv[i], cap), new_v], axis=0)
        if evaluate:
            mx.eval(sk, sv)
        self.sk, self.sv, self.cap = sk, sv, cap
        self._lens = self._lens + [S]
        self._changed()
        return self.n - 1

    def _pad(self, x, cap):
        if x.shape[2] == cap:
            return x
        return mx.concatenate([x, mx.zeros((*x.shape[:2], cap - x.shape[2], x.shape[3]), F32)], axis=2)

    def take(self, keep, evaluate: bool = False):
        """Keep the members `keep` (indices, in that new order); cap is trimmed
        to round_up(max length + 1, step). Copies the stacks, lazily by
        default: the old stacks stay referenced by the graph until the next
        step's appends consume the copies layer by layer (~one layer's old +
        new K/V of transient; module docstring). `evaluate=True` evaluates
        now: old and new stacks of every layer at once, and a wait for the
        step in flight."""
        self._no_step("take")
        keep = list(keep)
        if not keep:
            raise ValueError("cascade: take() of no rows (drop the group instead)")
        if any(not 0 <= j < self.n for j in keep) or len(set(keep)) != len(keep):
            raise ValueError(f"cascade: bad take {keep} of {self.n} rows")
        lens = [self._lens[j] for j in keep]
        cap = min(self.cap, _round_up(max(lens) + 1, self.step))
        idx = mx.array(keep, dtype=mx.int32)
        sk, sv = {}, {}
        for i in self.layers:
            sk[i] = mx.contiguous(mx.take(self.sk[i], idx, axis=0)[:, :, :cap, :])
            sv[i] = mx.contiguous(mx.take(self.sv[i], idx, axis=0)[:, :, :cap, :])
        if evaluate:
            mx.eval(sk, sv)
        self.sk, self.sv, self._lens, self.cap = sk, sv, lens, cap
        self._changed()

    def grow(self, need: int, evaluate: bool = False):
        """Capacity for `need` suffix tokens per row (cap -> round_up(need,
        step), zeros appended). No-op when it fits. Lazy by default, like
        `take`. Not inside a step (`begin_step` grows before it opens one)."""
        self._no_step("grow")
        if need <= self.cap:
            return
        cap = _round_up(need, self.step)
        sk = {i: self._pad(self.sk[i], cap) for i in self.layers}
        sv = {i: self._pad(self.sv[i], cap) for i in self.layers}
        if evaluate:
            mx.eval(sk, sv)
        self.sk, self.sv, self.cap = sk, sv, cap
        self._changed()

    # ---- conversion back

    def prefix_kv(self, layer):
        """The prefix [0, P) of one layer as (1, Hkv, P, D) in the rows' dtype
        (exact: the fp32 copy was widened from these values; kernel mode: a
        view of the fp16 buffer)."""
        if self.kernel:
            return self.kc[layer][..., :self.P, :], self.vc[layer][..., :self.P, :]
        k = self.kc[layer].reshape(1, self.Hkv, self.Ppad, self.D)[..., :self.P, :]
        v = self.vc[layer].reshape(1, self.Hkv, self.Ppad, self.D)[..., :self.P, :]
        return k.astype(self.kv_dtype), v.astype(self.kv_dtype)

    def prefix_equal(self, member) -> bool:
        """True when a row's FA keys/values [0, P) equal the group's prefix
        bitwise in every layer (the worker's QWEN_CASCADE_CHECK). Evaluates."""
        for i in self.layers:
            ks, vs, _ = _fa(member[i], self.P)
            pk, pv = self.prefix_kv(i)
            if ks.dtype != self.kv_dtype or not (mx.array_equal(ks[..., :self.P, :], pk).item()
                                                 and mx.array_equal(vs[..., :self.P, :], pv).item()):
                return False
        return True

    def release(self, j: int, evaluate: bool = True) -> dict:
        """Row j as contiguous KVCaches {layer: KVCache}, offset P + S_j, in
        the rows' dtype, with `step` zeros of spare capacity. Bit-exact (module
        docstring). The row stays in the group: `take` it out afterwards.
        Evaluates by default: lazy caches would keep every sk[i] / sv[i]
        referenced, so the next append of each layer would copy its stack."""
        self._no_step("release")
        if not 0 <= j < self.n:
            raise ValueError(f"cascade: release({j}) of {self.n} rows")
        S = self._lens[j]
        out = {}
        for i in self.layers:
            pk, pv = self.prefix_kv(i)
            z = mx.zeros((1, self.Hkv, self.step, self.D), self.kv_dtype)
            sk = self.sk[i][j:j + 1, :, :S, :].astype(self.kv_dtype)
            sv = self.sv[i][j:j + 1, :, :S, :].astype(self.kv_dtype)
            c = KVCache()
            c.keys = mx.concatenate([pk, sk, z], axis=2)
            c.values = mx.concatenate([pv, sv, z], axis=2)
            c.offset = self.P + S
            out[i] = c
        if evaluate:
            mx.eval([(c.keys, c.values) for c in out.values()])
        return out

    # ---- decode step

    def _check_bounds(self):
        """max(lens) < cap == every stack's slot count (put_along_axis writes
        at lens without a bounds check: module docstring, Bounds)."""
        want = (self.n, self.Hkv, self.cap, self.D)
        bad = [(i, tuple(a.shape)) for i in self.layers for a in (self.sk[i], self.sv[i])
               if tuple(a.shape) != want]
        if not max(self._lens) < self.cap or bad:
            raise RuntimeError(f"cascade: bounds broken: max length {max(self._lens)}, cap "
                               f"{self.cap}, stacks {bad[:2]} vs {want}")

    def begin_step(self):
        """Start a decode step: room for one more token per row, the bounds
        check, the write index (the rows' lengths) and the mask over
        [0, S_j + 1)."""
        self._no_step("begin_step")
        if self._failed:
            raise RuntimeError("cascade: the group failed in abort_step; drop it")
        self.grow(max(self._lens) + 1)
        self._check_bounds()
        lens = mx.array(self._lens, dtype=mx.int32)
        self._upos = mx.broadcast_to(lens[:, None, None, None], (self.n, self.Hkv, 1, self.D))
        self._appended = set()
        self._pending = True
        self._smask = None

    def append(self, layer, keys, values):
        """Write this step's token (keys/values (n, Hkv, 1, D) in member order
        and in `kv_dtype` -- widened to fp32 exactly, which `release`'s
        bit-exactness relies on) of one layer at each row's length. Exactly
        once per layer per step, before that layer's `attention`."""
        if not self._pending:
            raise RuntimeError("cascade: append() outside begin_step/end_step")
        if layer not in self.sk:
            raise KeyError(f"cascade: append() to unknown layer {layer!r}")
        if layer in self._appended:
            raise RuntimeError(f"cascade: second append() of layer {layer!r} in one step")
        want = (self.n, self.Hkv, 1, self.D)
        if tuple(keys.shape) != want or tuple(values.shape) != want:
            raise ValueError(f"cascade: append {tuple(keys.shape)} / {tuple(values.shape)}, want {want}")
        if keys.dtype != self.kv_dtype or values.dtype != self.kv_dtype:
            raise ValueError(f"cascade: append {keys.dtype} / {values.dtype}, the group's KV is "
                             f"{self.kv_dtype} (a wider value would round on release)")
        self.sk[layer] = mx.put_along_axis(self.sk[layer], self._upos, keys.astype(F32), axis=2)
        self.sv[layer] = mx.put_along_axis(self.sv[layer], self._upos, values.astype(F32), axis=2)
        self._appended.add(layer)

    def end_step(self):
        """The step's forward is built: every row's suffix is one longer.
        Refuses (the step stays open: `abort_step` it) unless every layer was
        appended."""
        if not self._pending:
            raise RuntimeError("cascade: end_step() without begin_step()")
        missing = [i for i in self.layers if i not in self._appended]
        if missing:
            raise RuntimeError(f"cascade: end_step() with layers {missing} not appended "
                               "(abort_step the step)")
        self._lens = [s + 1 for s in self._lens]
        self._pending = False
        self._appended = set()
        self._changed()

    def abort_step(self):
        """Building the step failed ON THE HOST (a Python exception while the
        graph was built): zero the slots `append` wrote (keeps "unused slots
        are zeros"), leave the lengths as they were, and evaluate the stacks
        now. If that eval raises (the failure was inside the graph), the group
        is unusable: RuntimeError, and the caller drops it (module docstring)."""
        if not self._pending:
            return
        z = mx.zeros((self.n, self.Hkv, 1, self.D), F32)
        for i in self._appended:
            self.sk[i] = mx.put_along_axis(self.sk[i], self._upos, z, axis=2)
            self.sv[i] = mx.put_along_axis(self.sv[i], self._upos, z, axis=2)
        self._pending = False
        self._appended = set()
        self._changed()
        try:
            mx.eval(self.sk, self.sv)
        except Exception as e:
            self._failed = True
            raise RuntimeError("cascade: abort_step could not restore the stacks; drop the group") from e

    def smask(self):
        """(n, 1, 1, cap) fp32 additive suffix mask: 0 for slots < S_j (+1
        inside a step), -inf past. Cached until the lengths / cap change."""
        if self._smask is None:
            n_att = mx.array(self._lens, dtype=mx.int32) + (1 if self._pending else 0)
            m = mx.arange(self.cap)[None, :] < n_att[:, None]
            self._smask = mx.where(m, 0.0, _NEG).astype(F32).reshape(self.n, 1, 1, self.cap)
        return self._smask
