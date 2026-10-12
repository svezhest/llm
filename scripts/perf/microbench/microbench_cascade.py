#!/usr/bin/env python3
"""Grouped shared-prefix ("cascade") decode attention vs today's per-row SDPA.
NEEDS THE GPU (nothing else computing on it). Synthetic fp16 tensors with the
9B's full-attention shape: 16 q heads, 4 kv heads, head_dim 256, 8 full-attention
layers of 32 (no model load). Decides whether docs/perf/2026-10-12/design/
prompt_cache_opt.md proposal 1 is worth building.

B rows share a prefix of P tokens; each row owns a suffix of len_i = S - (i % 8)
tokens (ragged, as real rows are); one query token per row.

Variants (all produce (B, 16, 1, 256) fp16, the layer's attention output):
  a      TODAY. Per row: mx.fast.scaled_dot_product_attention(q_i, K_i, V_i)
         over the row's own contiguous [prefix | suffix] KVCache buffer, outputs
         concatenated -- _rows_attention without the cache update.
  a_u    a + the per-row KVCache.update_and_fetch (exactly _rows_attention).
  b16    cascade, fp16 storage. Prefix ONCE for all rows: queries stacked
         (4 kv heads, 1, 4B, 256) (row-major: index b*4 + rep), prefix keys viewed
         as (4, c, Pc, 256) chunks (flash-decoding split, Pc = MC_PC), scores by
         one fp16 matmul, per-chunk softmax(precise) + fp32 logsumexp, P.V per
         chunk, chunks merged by their LSE in fp32. Suffix: stacked
         (B, 4, S, 256) buffer, the same manual formulation with an additive
         ragged-length mask. Final merge of prefix and suffix by LSE in fp32.
  b32    the same with fp32 storage (prefix + suffix): scores, softmax, P.V all
         fp32 -- the same fp32 products as MLX's decode kernel, other order.
  b16c / b32c   b16 / b32 under mx.compile (fuses the elementwise merge chain).
  b2     b32c's prefix; suffix by the fused SDPA (fp16 suffix buffer, bool mask)
         plus a separate fp32 scores matmul + logsumexp for its LSE.
  b3     b32c's prefix; suffix by the fused SDPA with the prefix as an
         attention SINK (rows folded into heads so sinks are per row): two
         calls (V, then V = ones with 8 channels), no suffix LSE at all.
  b3v    b3 in one call: V stored with 8 ones channels appended (d_v 264).
  b32m   b32c with a prefix of P - 7 tokens (not a multiple of the chunk): the
         buffer is padded to c*Pc and the scores get a static additive mask.
  b32u   b32c + the suffix update (one put_along_axis per K and V for all rows).
  (a_u and b32u run min(8, copies) chained layers per eval, scaled to 8; see
  Bench.step.)

A "step" = 8 layer calls, chained (layer l's query depends on layer l-1's
output, as in the model), one mx.eval + mx.synchronize. Layer data cycles over
enough distinct copies that the working set exceeds the 48 MB SLC. Variants are
interleaved within each round (order rotated per round), MC_ROUNDS rounds of
MC_STEPS timed steps after MC_WARM warm-up steps; medians.

Also: (roof) the prefix QK and PV matmuls alone (GB/s, TFLOPS), (num) numerics
vs a fp32 reference of (a) on two data regimes (flat scores: q ~ N(0,1);
peaked: q ~ N(0, 16) -> score std ~4), (mem) resident bytes per row + the
transient peak of one step.

Usage (from src/worker, with its venv; setopt NO_BG_NICE first):
  uv run python ../../scripts/perf/microbench/microbench_cascade.py
  MC_B="4,8,16" MC_P="1024,8192,32768" MC_S="64,512,2048" MC_ROUNDS=5
  MC_STEPS=6 MC_WARM=2 MC_PC=1024 MC_MEM_GB=6 MC_VARIANTS=a,a_u,b16,...
  --smoke : CPU, tiny shapes (code-path + numerics check only)
Output: $PERF_OUT/cascade_microbench.json (+ .txt), PERF_OUT default /tmp/perf.
"""
import json
import math
import os
import statistics
import sys
import time

SMOKE = "--smoke" in sys.argv
import mlx.core as mx  # noqa: E402

if SMOKE:
    mx.set_default_device(mx.cpu)

from mlx_vlm.models.cache import KVCache  # noqa: E402

F16, F32 = mx.float16, mx.float32
NQ, NKV, HD, NLAYERS = 16, 4, 256, 8
REP = NQ // NKV
SCALE = HD ** -0.5                     # 1/16: a power of two, so q*scale is exact in fp16


def _ints(name, default):
    return [int(x) for x in os.environ.get(name, default).split(",") if x]


BS = _ints("MC_B", "4" if SMOKE else "4,8,16")
PS = _ints("MC_P", "256" if SMOKE else "1024,8192,32768")
SS = _ints("MC_S", "16,40" if SMOKE else "64,512,2048")
ROUNDS = int(os.environ.get("MC_ROUNDS", "2" if SMOKE else "5"))
STEPS = int(os.environ.get("MC_STEPS", "2" if SMOKE else "6"))
WARM = int(os.environ.get("MC_WARM", "1" if SMOKE else "2"))
PC = int(os.environ.get("MC_PC", "64" if SMOKE else "1024"))
MEM_GB = float(os.environ.get("MC_MEM_GB", "6"))
SLC = 256 << 20                        # target working set per variant family (> 48 MB SLC)
ALL = ["a", "a_u", "b16", "b32", "b16c", "b32c", "b2", "b3", "b3v", "b32m", "b32u"]
VARIANTS = os.environ.get("MC_VARIANTS", ",".join(ALL)).split(",")
OUT = os.environ.get("PERF_OUT", "/tmp/perf")
PAD = 7                                # b32m: prefix length P - PAD
UPDATES = ("a_u", "b32u")


def sync():
    if not SMOKE:
        mx.synchronize()


# ---------------------------------------------------------------- cascade core

def _chunks(K, V, Pc):
    """(1, 4, Ppad, 256) -> (4, c, Pc, 256) views (Ppad % Pc == 0)."""
    n = K.shape[2]
    return (K.reshape(NKV, n // Pc, Pc, HD), V.reshape(NKV, n // Pc, Pc, HD))


def _to_prefix_layout(x):
    """(B, 4, 4, d) per-row layout -> (4, 1, 4B, d): kv head major, row b*4+rep."""
    B = x.shape[0]
    return x.transpose(1, 0, 2, 3).reshape(NKV, 1, B * REP, x.shape[-1])


def _from_prefix_layout(x, B):
    """(4, 1, 4B, d) -> (B, 4, 4, d)."""
    return x.reshape(NKV, B, REP, x.shape[-1]).transpose(1, 0, 2, 3)


def _partial(q, K, V, mask, dt):
    """Normalized partial attention + its LSE over the last key axis.
    q (..., M, 256) dt (already scaled), K/V (..., N, 256) dt, mask additive fp32
    broadcastable to the scores or None. Returns o (..., M, 256) dt, lse fp32."""
    s = q @ K.swapaxes(-1, -2)
    if mask is not None:
        s = s.astype(F32) + mask
    lse = mx.logsumexp(s.astype(F32), axis=-1, keepdims=True)
    p = mx.softmax(s, axis=-1, precise=True).astype(dt)
    return p @ V, lse


def cascade(q, Kc, Vc, pmask, Ks, Vs, smask, dt, sdpa_suffix=None, sinks_suffix=None):
    """q (B,16,1,256) fp16. Kc/Vc (4,c,Pc,256) shared prefix chunks, dt.
    pmask (1,c,1,Pc) fp32 additive or None. Ks/Vs (B,4,S,256) dt, smask
    (B,1,1,S) fp32 additive. sdpa_suffix=(Ks16, Vs16, bmask): suffix output by
    the fused kernel instead of the manual P.V (LSE still from Ks/smask).
    sinks_suffix=(Ks16, Vs, bmask, ones, nv): the suffix by the fused kernel
    with the prefix as an attention sink, no suffix LSE needed (_sinks_merge)."""
    B = q.shape[0]
    qg = (q.astype(dt) * SCALE).reshape(B, NKV, REP, HD)          # (B,4,4,256)
    qp = _to_prefix_layout(qg)                                    # (4,1,4B,256)
    o_c, lse_c = _partial(qp, Kc, Vc, pmask, dt)                  # (4,c,4B,256), (4,c,4B,1)
    lse_p = mx.logsumexp(lse_c, axis=1, keepdims=True)            # (4,1,4B,1)
    o_p = (mx.exp(lse_c - lse_p) * o_c.astype(F32)).sum(axis=1, keepdims=True)
    if sinks_suffix is not None:
        return _sinks_merge(q, o_p, lse_p, *sinks_suffix)
    if sdpa_suffix is None:
        o_s, lse_s = _partial(qg, Ks, Vs, smask, dt)              # (B,4,4,256), (B,4,4,1)
    else:
        Ks16, Vs16, bmask = sdpa_suffix
        o_s = mx.fast.scaled_dot_product_attention(q, Ks16, Vs16, scale=SCALE, mask=bmask)
        o_s = o_s.reshape(B, NKV, REP, HD)
        s = (qg @ Ks.swapaxes(-1, -2)).astype(F32) + smask
        lse_s = mx.logsumexp(s, axis=-1, keepdims=True)
    o_s = _to_prefix_layout(o_s.astype(F32))
    lse_s = _to_prefix_layout(lse_s)
    m = mx.maximum(lse_p, lse_s)
    wp, ws = mx.exp(lse_p - m), mx.exp(lse_s - m)
    out = (wp * o_p + ws * o_s) / (wp + ws)                       # (4,1,4B,256) fp32
    return _from_prefix_layout(out, B).reshape(B, NQ, 1, HD).astype(F16)


def _sinks_merge(q, o_p, lse_p, Ks16, Vs, bmask, ones, nv):
    """The suffix through mx.fast.scaled_dot_product_attention with the prefix
    folded in as an attention sink. Rows become heads -- q (1, B*16, 1, 256),
    K/V (1, B*4, S, d) -- so the per-head `sinks` (shape (H,), fp16 only) is
    per (row, head). With sink = lse_p the kernel returns
        A = Z_s o_s / (Z_s + Z_p)
    and the same call over V = ones returns w_s = Z_s / (Z_s + Z_p), so
        out = A + (1 - w_s) o_p   -- no suffix LSE.
    fp16 sinks would round lse_p (~10: ulp 0.008 -> 0.4 % weight error), so
    c = fp16(lse_p) moves into the additive mask (exact: score - c in fp32 in
    the kernel) and the sink is the tiny residual lse_p - c.
    ones=None: Vs already carries nv extra ones channels (one call, d = 256 +
    nv); else a second call over `ones` (1, B*4, S, nv)."""
    B = q.shape[0]
    S = Ks16.shape[2]
    lse = _from_prefix_layout(lse_p, B).reshape(B * NQ)               # fp32
    c = lse.astype(F16)
    sink = (lse - c.astype(F32)).astype(F16)
    fmask = mx.where(bmask.reshape(B, 1, 1, S), -c.reshape(B, NQ, 1, 1), -math.inf) \
        .astype(F16).reshape(1, B * NQ, 1, S)
    qh = q.reshape(1, B * NQ, 1, HD)
    Kh = Ks16.reshape(1, B * NKV, S, HD)
    A = mx.fast.scaled_dot_product_attention(qh, Kh, Vs.reshape(1, B * NKV, S, -1),
                                             scale=SCALE, mask=fmask, sinks=sink)
    if ones is None:
        A, w = A[..., :HD], A[..., HD:HD + 1]
    else:
        w = mx.fast.scaled_dot_product_attention(qh, Kh, ones, scale=SCALE, mask=fmask,
                                                 sinks=sink)[..., :1]
    o_p = _from_prefix_layout(o_p, B).reshape(1, B * NQ, 1, HD)
    out = A.astype(F32) + (1.0 - w.astype(F32)) * o_p
    return out.reshape(B, NQ, 1, HD).astype(F16)


# ---------------------------------------------------------------- data

class Group:
    """One layer's data for one (B, P, S): today's per-row caches and the
    cascade's shared prefix + stacked suffixes, same contents."""

    def __init__(self, B, P, S, need, qstd=1.0, rows=True):
        self.B, self.P, self.S = B, P, S
        lens = [S - (i % 8) for i in range(B)]
        self.lens = lens
        Kp = mx.random.normal((1, NKV, P, HD)).astype(F16)
        Vp = mx.random.normal((1, NKV, P, HD)).astype(F16)
        Ks = mx.random.normal((B, NKV, S, HD)).astype(F16)
        Vs = mx.random.normal((B, NKV, S, HD)).astype(F16)
        valid = mx.arange(S)[None, :] < mx.array(lens)[:, None]          # (B,S)
        self.bmask = valid[:, None, None, :]
        self.smask = mx.where(self.bmask, 0.0, -math.inf).astype(F32)
        Ks = mx.where(self.bmask.reshape(B, 1, S, 1), Ks, 0)             # unused slots zeroed
        Vs = mx.where(self.bmask.reshape(B, 1, S, 1), Vs, 0)
        self.qstd = qstd
        self.caches = []
        if rows:                       # today's layout: one contiguous buffer per row
            for i in range(B):
                n = P + lens[i]
                cap = (n + 1 + 255) // 256 * 256
                c = KVCache()
                z = mx.zeros((1, NKV, cap - n, HD), F16)
                c.keys = mx.concatenate([Kp, Ks[i:i + 1, :, :lens[i]], z], axis=2)
                c.values = mx.concatenate([Vp, Vs[i:i + 1, :, :lens[i]], z], axis=2)
                c.offset = n
                mx.eval(c.keys, c.values)
                self.caches.append(c)
        Pc = min(PC, P)
        self.Pc = Pc
        if {"b16", "b16c"} & need:
            self.Kc16, self.Vc16 = _chunks(Kp, Vp, Pc)
        if {"b32", "b32c", "b2", "b3", "b3v", "b32u", "b32m"} & need:
            self.Kc32, self.Vc32 = _chunks(Kp.astype(F32), Vp.astype(F32), Pc)
        if {"b3", "b3v"} & need:
            self.ones8 = mx.ones((1, B * NKV, S, 8), F16)
        if "b3v" in need:
            self.Vs264 = mx.concatenate([Vs, mx.ones((B, NKV, S, 8), F16)], axis=-1)
        if {"b16", "b16c", "b2", "b3", "b3v"} & need:
            self.Ks16, self.Vs16 = Ks, Vs
        if {"b32", "b32c", "b2", "b32u", "b32m"} & need:
            self.Ks32, self.Vs32 = Ks.astype(F32), Vs.astype(F32)
        if "b32m" in need:
            # prefix of P - PAD real tokens in a buffer padded to c * Pc
            self.pmask = mx.where(mx.arange(P) < P - PAD, 0.0, -math.inf).astype(F32) \
                .reshape(1, P // Pc, 1, Pc)
        self.Kp, self.Vp = Kp, Vp      # kept for numerics only (shared with Kc16 views)
        self.upos = mx.broadcast_to(mx.array(lens)[:, None, None, None] - 1, (B, NKV, 1, HD))
        mx.eval([v for v in vars(self).values() if isinstance(v, mx.array)])

    def bytes_rows(self):
        return sum(c.keys.nbytes + c.values.nbytes for c in self.caches)

    def bytes_cascade(self):
        return sum(v.nbytes for k, v in vars(self).items()
                   if isinstance(v, mx.array) and k[:2] in ("Kc", "Vc", "Ks", "Vs"))


# ---------------------------------------------------------------- variants

def _rows(g, q, update):
    outs = []
    for i, c in enumerate(g.caches):
        if update:
            kn = q[i:i + 1, :NKV]       # any (1,4,1,256) fp16 row: written at offset
            k, v = c.update_and_fetch(kn, kn)
        else:
            k, v = c.keys[..., :c.offset, :], c.values[..., :c.offset, :]
        outs.append(mx.fast.scaled_dot_product_attention(q[i:i + 1], k, v, scale=SCALE, mask=None))
    return mx.concatenate(outs, axis=0)


_COMPILED = {}


def _compiled(key, fn):
    if key not in _COMPILED:
        _COMPILED[key] = mx.compile(fn)
    return _COMPILED[key]


def layer_call(name, g, q):
    """One full-attention layer of variant `name` on group g. Returns
    (out, extra arrays to evaluate)."""
    if name == "a":
        return _rows(g, q, False), []
    if name == "a_u":
        out = _rows(g, q, True)
        return out, [x for c in g.caches for x in (c.keys, c.values)]
    if name == "b16":
        return cascade(q, g.Kc16, g.Vc16, None, g.Ks16, g.Vs16, g.smask, F16), []
    if name == "b32":
        return cascade(q, g.Kc32, g.Vc32, None, g.Ks32, g.Vs32, g.smask, F32), []
    if name == "b16c":
        f = _compiled("b16c", lambda q, K, V, Ks, Vs, sm: cascade(q, K, V, None, Ks, Vs, sm, F16))
        return f(q, g.Kc16, g.Vc16, g.Ks16, g.Vs16, g.smask), []
    if name == "b32c":
        f = _compiled("b32c", lambda q, K, V, Ks, Vs, sm: cascade(q, K, V, None, Ks, Vs, sm, F32))
        return f(q, g.Kc32, g.Vc32, g.Ks32, g.Vs32, g.smask), []
    if name == "b32m":
        f = _compiled("b32m", lambda q, K, V, pm, Ks, Vs, sm: cascade(q, K, V, pm, Ks, Vs, sm, F32))
        return f(q, g.Kc32, g.Vc32, g.pmask, g.Ks32, g.Vs32, g.smask), []
    if name == "b3":
        f = _compiled("b3", lambda q, K, V, Ks, Vs, bm, on:
                      cascade(q, K, V, None, None, None, None, F32, sinks_suffix=(Ks, Vs, bm, on, 8)))
        return f(q, g.Kc32, g.Vc32, g.Ks16, g.Vs16, g.bmask, g.ones8), []
    if name == "b3v":
        f = _compiled("b3v", lambda q, K, V, Ks, Vs, bm:
                      cascade(q, K, V, None, None, None, None, F32, sinks_suffix=(Ks, Vs, bm, None, 8)))
        return f(q, g.Kc32, g.Vc32, g.Ks16, g.Vs264, g.bmask), []
    if name == "b2":
        f = _compiled("b2", lambda q, K, V, Ks, Vs, sm, K16, V16, bm:
                      cascade(q, K, V, None, Ks, Vs, sm, F32, sdpa_suffix=(K16, V16, bm)))
        return f(q, g.Kc32, g.Vc32, g.Ks32, g.Vs32, g.smask, g.Ks16, g.Vs16, g.bmask), []
    if name == "b32u":
        kn = q[:, :NKV].astype(F32)     # (B,4,1,256): the new token's K/V, one slot per row
        g.Ks32 = mx.put_along_axis(g.Ks32, g.upos, kn, axis=2)
        g.Vs32 = mx.put_along_axis(g.Vs32, g.upos, kn, axis=2)
        f = _compiled("b32c", lambda q, K, V, Ks, Vs, sm: cascade(q, K, V, None, Ks, Vs, sm, F32))
        return f(q, g.Kc32, g.Vc32, g.Ks32, g.Vs32, g.smask), [g.Ks32, g.Vs32]
    raise ValueError(name)


def family(name):
    return "rows" if name.startswith("a") else "cascade"


class Bench:
    def __init__(self, B, P, S, need):
        self.B = B
        one = Group(B, P, S, need)
        self.qs = [mx.random.normal((B, NQ, 1, HD)).astype(F16) for _ in range(NLAYERS)]
        mx.eval(self.qs)
        br, bc = one.bytes_rows(), one.bytes_cascade()
        budget = MEM_GB * 1e9
        n = max(2, min(8, math.ceil(SLC / max(1, min(br, bc)))))
        while n > 2 and n * (br + bc) > budget:
            n -= 1
        self.groups = [one] + [Group(B, P, S, need) for _ in range(n - 1)]
        self.n = n
        self.bytes_rows, self.bytes_cascade = br, bc
        self.k = 0

    def step(self, name):
        """8 chained layer calls -> (wall ms, host build ms). The update
        variants (a_u, b32u) run min(8, copies) layers and scale to 8: a copy
        updated twice in one graph is read in between, so its buffer is not
        donatable and the second update copies it -- an artifact of cycling
        copies (in the model every layer's cache is updated once per step)."""
        nl = min(NLAYERS, self.n) if name in UPDATES else NLAYERS
        t0 = time.perf_counter()
        prev, extra = None, []
        for layer in range(nl):
            g = self.groups[(self.k + layer) % self.n]
            q = self.qs[layer] if prev is None else self.qs[layer] + prev * 1e-3
            prev, ex = layer_call(name, g, q)
            extra += ex
        self.k += nl
        t1 = time.perf_counter()
        mx.eval(prev, extra)
        sync()
        t2 = time.perf_counter()
        f = NLAYERS / nl
        return (t2 - t0) * 1e3 * f, (t1 - t0) * 1e3 * f


# ---------------------------------------------------------------- numerics

def numerics(B, P, S, qstd, need):
    g = Group(B, P, S, need, qstd=qstd)
    q = (mx.random.normal((B, NQ, 1, HD)) * qstd).astype(F16)
    mx.eval(q)
    # fp32 reference of (a): the fused kernel on fp32 copies of each row
    ref = mx.concatenate([
        mx.fast.scaled_dot_product_attention(
            q[i:i + 1].astype(F32), c.keys[..., :c.offset, :].astype(F32),
            c.values[..., :c.offset, :].astype(F32), scale=SCALE, mask=None)
        for i, c in enumerate(g.caches)], axis=0)
    outs = {"a": _rows(g, q, False)}
    for name in need:
        if family(name) == "cascade" and name not in ("b32u", "b32m"):
            outs[name] = layer_call(name, g, q)[0]
    if "b32m" in need:
        # b32m reference: the rows with the last PAD prefix tokens dropped
        ref_m = mx.concatenate([
            mx.fast.scaled_dot_product_attention(
                q[i:i + 1].astype(F32),
                mx.concatenate([g.Kp[..., :P - PAD, :], c.keys[..., P:c.offset, :]], axis=2).astype(F32),
                mx.concatenate([g.Vp[..., :P - PAD, :], c.values[..., P:c.offset, :]], axis=2).astype(F32),
                scale=SCALE, mask=None)
            for i, c in enumerate(g.caches)], axis=0)
        outs["b32m"] = layer_call("b32m", g, q)[0]
    mx.eval(ref, outs)
    res = {}
    for name, o in outs.items():
        r = ref_m if name == "b32m" else ref
        d = o.astype(F32) - r
        da = o.astype(F32) - outs["a"].astype(F32)
        rms_ref = mx.sqrt(mx.mean(r * r)).item()
        res[name] = {
            "max_abs_vs_ref": mx.max(mx.abs(d)).item(),
            "rms_vs_ref": mx.sqrt(mx.mean(d * d)).item(),
            "rel_rms_vs_ref": mx.sqrt(mx.mean(d * d)).item() / rms_ref,
            "max_abs_vs_a": None if name == "b32m" else mx.max(mx.abs(da)).item(),
            "frac_neq_a": None if name == "b32m" else mx.mean(o != outs["a"]).item(),
        }
    # transient peak of one step (8 layers, same group) per variant
    peaks = {}
    if not SMOKE:
        for name in ["a"] + [v for v in need if family(v) == "cascade" and v != "b32u"]:
            mx.eval(layer_call(name, g, q)[0])  # compile / warm
            mx.clear_cache()
            sync()
            base = mx.get_active_memory()
            mx.reset_peak_memory()
            prev = None
            for layer in range(NLAYERS):
                qq = q if prev is None else q + prev * 1e-3
                prev = layer_call(name, g, qq)[0]
            mx.eval(prev)
            sync()
            peaks[name] = (mx.get_peak_memory() - base) / 1e6
    del g
    mx.clear_cache()
    return res, peaks


# ---------------------------------------------------------------- roof

def roof(B, P, dt, n=4, reps=8):
    """Prefix QK and PV alone, chunked as in the cascade: ms per call, GB/s of
    the K (V) read, TFLOPS."""
    Pc = min(PC, P)
    c = P // Pc
    Ks = [mx.random.normal((NKV, c, Pc, HD)).astype(dt) for _ in range(n)]
    Vs = [mx.random.normal((NKV, c, Pc, HD)).astype(dt) for _ in range(n)]
    qp = mx.random.normal((NKV, 1, REP * B, HD)).astype(dt)
    p = mx.softmax(mx.random.normal((NKV, c, REP * B, Pc)), axis=-1).astype(dt)
    mx.eval(Ks, Vs, qp, p)

    def t(fn):
        mx.eval(fn())
        sync()
        ts = []
        for _ in range(7):
            a = time.perf_counter()
            mx.eval(fn())
            sync()
            ts.append(time.perf_counter() - a)
        return statistics.median(ts) / reps * 1e3

    qk = t(lambda: [qp @ Ks[i % n].swapaxes(-1, -2) for i in range(reps)])
    pv = t(lambda: [p @ Vs[i % n] for i in range(reps)])
    byt = NKV * P * HD * (2 if dt == F16 else 4)
    fl = 2 * NKV * REP * B * P * HD
    del Ks, Vs
    mx.clear_cache()
    return {"qk_ms": qk, "pv_ms": pv, "qk_GBs": byt / qk / 1e6, "pv_GBs": byt / pv / 1e6,
            "qk_TF": fl / qk / 1e9, "pv_TF": fl / pv / 1e9}


# ---------------------------------------------------------------- main

def fmt(x, w=7, p=3):
    return f"{x:{w}.{p}f}" if isinstance(x, (int, float)) else f"{'-':>{w}}"


def main():
    os.makedirs(OUT, exist_ok=True)
    need = set(VARIANTS)
    mx.random.seed(0)
    t_start = time.time()
    out = {"mlx": mx.__version__, "pc": PC, "rounds": ROUNDS, "steps": STEPS,
           "variants": VARIANTS, "configs": [], "roof": [], "numerics": []}
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f"mlx {mx.__version__}  Pc={PC}  rounds={ROUNDS} steps={STEPS} warm={WARM}  "
        f"variants={','.join(VARIANTS)}")

    # numerics first (one layer each, cheap)
    log("\n== numerics (one layer; vs fp32 reference of (a); b32m vs its own reference) ==")
    for qstd, regime in ((1.0, "flat"), (4.0, "peaked")):
        for B in BS:
            for P in PS:
                for S in SS:
                    res, peaks = numerics(B, P, S, qstd, need)
                    out["numerics"].append({"B": B, "P": P, "S": S, "regime": regime,
                                            "res": res, "peak_MB": peaks})
                    log(f"{regime:6s} B={B:2d} P={P:5d} S={S:4d} " + "  ".join(
                        f"{k}:{v['rel_rms_vs_ref']:.2e}/{v['max_abs_vs_ref']:.1e}" for k, v in res.items()))
                    if peaks and qstd == 1.0:
                        log("        transient MB/step: " + "  ".join(f"{k}:{v:.0f}" for k, v in peaks.items()))

    log("\n== roof: prefix matmuls alone (chunked, Pc=%d) ==" % PC)
    for B in BS:
        for P in PS:
            for dt, dn in ((F16, "fp16"), (F32, "fp32")):
                r = roof(B, P, dt)
                out["roof"].append({"B": B, "P": P, "dtype": dn, **r})
                log(f"B={B:2d} P={P:5d} {dn}: QK {r['qk_ms']:.3f} ms {r['qk_GBs']:4.0f} GB/s "
                    f"{r['qk_TF']:.2f} TF | PV {r['pv_ms']:.3f} ms {r['pv_GBs']:4.0f} GB/s {r['pv_TF']:.2f} TF")

    log("\n== timing: ms per 8-layer step (median of round medians); /8 = per layer ==")
    log("B   P     S    | " + " ".join(f"{v:>7s}" for v in VARIANTS) + " | copies")
    for B in BS:
        for P in PS:
            for S in SS:
                bench = Bench(B, P, S, need)
                for v in VARIANTS:              # warm-up (compiles, first-step growth)
                    for _ in range(WARM):
                        bench.step(v)
                per = {v: [] for v in VARIANTS}
                build = {v: [] for v in VARIANTS}
                for r in range(ROUNDS):
                    order = VARIANTS[r % len(VARIANTS):] + VARIANTS[:r % len(VARIANTS)]
                    for v in order:
                        ts = [bench.step(v) for _ in range(STEPS)]
                        per[v].append(statistics.median(t[0] for t in ts))
                        build[v].append(statistics.median(t[1] for t in ts))
                med = {v: statistics.median(per[v]) for v in VARIANTS}
                bmed = {v: statistics.median(build[v]) for v in VARIANTS}
                spread = {v: (min(per[v]), max(per[v])) for v in VARIANTS}
                rec = {"B": B, "P": P, "S": S, "copies": bench.n, "step_ms": med,
                       "build_ms": bmed, "round_min_max": spread, "rounds": per,
                       "bytes_rows_layer": bench.bytes_rows, "bytes_cascade_layer": bench.bytes_cascade}
                out["configs"].append(rec)
                log(f"{B:2d} {P:5d} {S:4d}  | " + " ".join(fmt(med[v], 7, 2) for v in VARIANTS)
                    + f" | {bench.n}   build a {bmed.get('a', 0):.2f} / b32c {bmed.get('b32c', 0):.2f} ms")
                del bench
                mx.clear_cache()
                with open(os.path.join(OUT, "cascade_microbench.json"), "w") as f:
                    json.dump(out, f, indent=1)
    out["wall_s"] = time.time() - t_start
    log(f"\nwall {out['wall_s']:.0f} s")
    with open(os.path.join(OUT, "cascade_microbench.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(OUT, "cascade_microbench.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
