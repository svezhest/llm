"""cascade.py (grouped shared-prefix decode attention, plan Part B / step B1):
the formulation against per-row SDPA over each row's contiguous [prefix | suffix]
cache (fp32 reference, GQA order, padded prefix, ragged / empty suffixes,
B_g = 1), the masks against garbage in unused slots, the holders (build, add,
take, grow, release round trip, decode steps, abort, compile-trace LRU), the
step protocol and bounds guards, step-ahead (step k+1 built before step k is
read) and release-then-continue bitwise checks, and the error paths.

CPU by default (conftest). `CASCADE_GPU=1 uv run --group dev pytest
tests/test_cascade.py` runs the same tests on Metal, plus the GPU-only ones:
the 9B-shape rms criterion (cascade <= 2.3e-4 rms rel vs an fp32 reference in
every case, and <= the per-row kernel's own error in >= 95 % of the (case,
query draw) pairs and in aggregate) and a timing smoke (B_g 16, P 8K, S 512,
8 layers: cascade update + attention vs today's per-row update_and_fetch +
SDPA, interleaved, medians; prints the ms; then step-ahead vs synchronous
steps with model-like ops between the FA layers: the step-ahead peak must stay
within one layer's K+V stack of the synchronous one -- donation holds)."""
import math
import os
import statistics
import time

import mlx.core as mx
import pytest
from mlx_vlm.models.base import scaled_dot_product_attention as lib_sdpa
from mlx_vlm.models.cache import KVCache

import cascade

GPU = os.environ.get("CASCADE_GPU") == "1"
F16, F32 = mx.float16, mx.float32
gpu_only = pytest.mark.skipif(not (GPU and mx.metal.is_available()),
                              reason="CASCADE_GPU=1 and Metal only")


@pytest.fixture(autouse=True)
def device():
    if GPU and not mx.metal.is_available():
        pytest.skip("CASCADE_GPU=1 needs Metal")
    prev = mx.default_device()
    mx.set_default_device(mx.gpu if GPU else mx.cpu)
    cascade.clear_traces()
    yield
    mx.set_default_device(prev)
    cascade.clear_traces()


# ---------------------------------------------------------------- data


def _cache(k, v, garbage=None):
    """A KVCache holding k/v (1, Hkv, n, D) as a prefill leaves it (capacity
    rounded up to 256); `garbage` fills the slots past the offset."""
    c = KVCache()
    c.update_and_fetch(k, v)
    if garbage is not None:
        n = c.offset
        c.keys[..., n:, :] = garbage
        c.values[..., n:, :] = garbage
    mx.eval(c.keys, c.values)
    return c


class Rows:
    """B rows sharing a P-token prefix, ragged suffixes `lens`, n_layers FA
    layers (keys 3, 7, ... like the model's FA layer indices), random data.
    rows[b][layer] is row b's contiguous KVCache (a dict, as the worker's
    row.cache list is indexed by layer)."""

    def __init__(self, P, lens, Hkv, D, n_layers=1, dtype=F16, seed=0, garbage=None):
        mx.random.seed(seed)
        self.P, self.lens, self.Hkv, self.D, self.dtype = P, list(lens), Hkv, D, dtype
        self.layers = [4 * i + 3 for i in range(n_layers)]
        self.rows = [{} for _ in lens]
        for i in self.layers:
            kp = mx.random.normal((1, Hkv, P, D)).astype(dtype)
            vp = mx.random.normal((1, Hkv, P, D)).astype(dtype)
            for b, S in enumerate(lens):
                ks = mx.random.normal((1, Hkv, S, D)).astype(dtype)
                vs = mx.random.normal((1, Hkv, S, D)).astype(dtype)
                self.rows[b][i] = _cache(mx.concatenate([kp, ks], axis=2),
                                         mx.concatenate([vp, vs], axis=2), garbage)

    def group(self, chunk, step=16, members=None):
        mem = self.rows if members is None else [self.rows[b] for b in members]
        return cascade.PrefixGroup.build(mem[0], mem, self.P, self.layers, chunk=chunk, step=step)


def per_row(q, caches, scale, fp32=False):
    """Today's path: per-row SDPA over each row's contiguous cache (fp32=True:
    the same op on fp32 copies -- the reference)."""
    outs = []
    for b, c in enumerate(caches):
        k, v = c.keys[..., :c.offset, :], c.values[..., :c.offset, :]
        qb = q[b:b + 1]
        if fp32:
            qb, k, v = qb.astype(F32), k.astype(F32), v.astype(F32)
        outs.append(mx.fast.scaled_dot_product_attention(qb, k, v, scale=scale, mask=None))
    return mx.concatenate(outs, axis=0)


def rms_rel(out, ref):
    d = out.astype(F32) - ref.astype(F32)
    return (mx.sqrt(mx.mean(d * d)) / mx.sqrt(mx.mean(ref.astype(F32) ** 2))).item()


def queries(B, Hq, D, qstd=1.0, dtype=F16):
    return (mx.random.normal((B, Hq, 1, D)) * qstd).astype(dtype)


def assert_unused_zero(g):
    """The holders' invariant: prefix padding and suffix slots past each
    row's length are exactly zero."""
    for i in g.layers:
        for arr in (g.kc[i], g.vc[i]):
            flat = arr.reshape(g.Hkv, g.Ppad, g.D)
            assert mx.all(flat[:, g.P:, :] == 0).item()
        for arr in (g.sk[i], g.sv[i]):
            for j, S in enumerate(g.lens):
                assert mx.all(arr[j, :, S:, :] == 0).item()


LAYOUTS = [(4, 2, 64), (16, 4, 256)]       # the tiny test model, the 9B


# ---------------------------------------------------------------- numerics


@pytest.mark.parametrize("Hq,Hkv,D", LAYOUTS)
@pytest.mark.parametrize("P", [48, 40, 70])          # 3 chunks exact; padded
@pytest.mark.parametrize("B", [3, 5])
@pytest.mark.parametrize("qstd", [1.0, 4.0])         # flat / peaked scores
def test_matches_rows(Hq, Hkv, D, P, B, qstd):
    d = Rows(P, [20 - 3 * b for b in range(B)], Hkv, D, seed=P + B)
    g = d.group(chunk=16)
    assert (g.pmask is None) == (P % 16 == 0)
    q = queries(B, Hq, D, qstd)
    scale = D ** -0.5
    caches = [r[d.layers[0]] for r in d.rows]
    out = cascade.attention(q, g, d.layers[0], scale)
    today = per_row(q, caches, scale)
    ref = per_row(q, caches, scale, fp32=True)
    assert out.shape == (B, Hq, 1, D) and out.dtype == F16
    assert mx.all(mx.isfinite(out)).item()
    e_c, e_t = rms_rel(out, ref), rms_rel(today, ref)
    assert e_c <= 3e-4 and e_c <= 1.1 * e_t, (e_c, e_t)
    # max error: within the plan's bound vs the reference (vs `today` it is
    # about today's own error: ~5e-3 on CPU's unfused fp16 path, peaked)
    assert mx.max(mx.abs(out.astype(F32) - ref)).item() <= 4e-3
    # the uncompiled function agrees with the compiled one up to fusion rounding
    raw = cascade.cascade(q, g.kc[d.layers[0]], g.vc[d.layers[0]], g.pmask,
                          g.sk[d.layers[0]], g.sv[d.layers[0]], g.smask(), scale)
    assert rms_rel(raw, ref) <= 3e-4


@pytest.mark.parametrize("P", [2048, 2000])
def test_matches_rows_9b_chunk1024(P):
    """9B layout at the real chunk (one padded chunk at 2000)."""
    lens = [64 - (b % 8) for b in range(8)]
    d = Rows(P, lens, 4, 256, seed=1)
    g = d.group(chunk=1024, step=256)
    assert g.c == 2 and g.cap == 256
    q = queries(8, 16, 256, 4.0)
    caches = [r[d.layers[0]] for r in d.rows]
    out = cascade.attention(q, g, d.layers[0], 1 / 16)
    e_c = rms_rel(out, per_row(q, caches, 1 / 16, fp32=True))
    e_t = rms_rel(per_row(q, caches, 1 / 16), per_row(q, caches, 1 / 16, fp32=True))
    assert e_c <= 3e-4 and e_c <= 1.1 * e_t, (e_c, e_t)


def test_fp32_rows_close():
    """The CPU test model runs fp32 KV: then the cascade is fp32 throughout."""
    d = Rows(40, [5, 0, 9], 2, 64, dtype=F32)
    g = d.group(chunk=16)
    q = queries(3, 4, 64, dtype=F32)
    out = cascade.attention(q, g, d.layers[0], 0.125)
    ref = per_row(q, [r[d.layers[0]] for r in d.rows], 0.125, fp32=True)
    assert out.dtype == F32
    assert mx.allclose(out, ref, atol=2e-5, rtol=1e-4).item()


@pytest.mark.parametrize("Hq,Hkv,D", LAYOUTS)
def test_empty_suffix_and_exact_chunk_multiple(Hq, Hkv, D):
    """P = 2 chunks exactly (no pmask); rows with S = 0, outside a step (an
    all -inf suffix row) and inside one (the new token is the whole suffix)."""
    d = Rows(32, [0, 7, 0], Hkv, D, n_layers=1, seed=3)
    g = d.group(chunk=16)
    assert g.pmask is None and g.c == 2 and g.lens == (0, 7, 0)
    i = d.layers[0]
    q = queries(3, Hq, D)
    out = cascade.attention(q, g, i, D ** -0.5)
    caches = [r[i] for r in d.rows]
    ref = per_row(q, caches, D ** -0.5, fp32=True)
    assert mx.all(mx.isfinite(out)).item()
    assert rms_rel(out, ref) <= 3e-4
    # one step: the new token is attended
    k, v = mx.random.normal((3, Hkv, 1, D)).astype(F16), mx.random.normal((3, Hkv, 1, D)).astype(F16)
    g.begin_step()
    g.append(i, k, v)
    out = cascade.attention(q, g, i, D ** -0.5)
    g.end_step()
    for b, c in enumerate(caches):
        c.update_and_fetch(k[b:b + 1], v[b:b + 1])
    assert g.lens == (1, 8, 1)
    assert rms_rel(out, per_row(q, caches, D ** -0.5, fp32=True)) <= 3e-4


@pytest.mark.parametrize("Hq,Hkv,D", LAYOUTS)
def test_single_row(Hq, Hkv, D):
    """B_g = 1 works (the scheduler never groups one row, but the kernel is general)."""
    d = Rows(40, [11], Hkv, D, seed=4)
    g = d.group(chunk=16)
    q = queries(1, Hq, D)
    out = cascade.attention(q, g, d.layers[0], D ** -0.5)
    ref = per_row(q, [d.rows[0][d.layers[0]]], D ** -0.5, fp32=True)
    assert out.shape == (1, Hq, 1, D)
    assert rms_rel(out, ref) <= 3e-4


# ---------------------------------------------------------------- unused slots


@pytest.mark.parametrize("Hq,Hkv,D", LAYOUTS)
def test_unused_slots_finite_garbage_is_weightless(Hq, Hkv, D):
    """Large finite garbage in the prefix padding and in every row's slots
    past its length changes no output bit: the -inf masks give those slots
    weight exactly 0 (scores and P.V)."""
    d = Rows(40, [3, 9, 0, 14], Hkv, D, seed=5)
    g = d.group(chunk=16)
    i = d.layers[0]
    q = queries(4, Hq, D, 4.0)
    clean = cascade.attention(q, g, i, D ** -0.5)
    mx.eval(clean)
    pad = mx.arange(g.Ppad)[None, :, None] >= g.P
    for name in ("kc", "vc"):
        flat = getattr(g, name)[i].reshape(g.Hkv, g.Ppad, g.D)
        getattr(g, name)[i] = mx.where(pad, 1e4, flat).reshape(g.Hkv, g.c, g.chunk, g.D)
    used = mx.arange(g.cap)[None, :] < mx.array(list(g.lens))[:, None]
    used = used[:, None, :, None]
    g.sk[i] = mx.where(used, g.sk[i], -3e4)
    g.sv[i] = mx.where(used, g.sv[i], 3e4)
    dirty = cascade.attention(q, g, i, D ** -0.5)
    assert mx.array_equal(clean, dirty).item()


def test_garbage_in_source_caches_never_copied():
    """NaN in the rows' KVCache buffers past their offsets (and in the source
    row's prefix-padding range) does not reach the group: the holders copy
    only [0, offset) and pad with zeros, so the output is finite and bitwise
    the clean build's, through build, steps with growth, add and take."""
    lens = [5, 0, 12]
    clean = Rows(40, lens, 2, 64, n_layers=2, seed=6)
    dirty = Rows(40, lens, 2, 64, n_layers=2, seed=6, garbage=math.nan)
    q = queries(3, 4, 64)
    gs = [clean.group(chunk=16), dirty.group(chunk=16)]
    assert_unused_zero(gs[1])
    mx.random.seed(60)
    toks = [(mx.random.normal((3, 2, 1, 64)).astype(F16),) * 2 for _ in range(20)]
    for s in range(20):                     # crosses cap (16 -> 32)
        outs = []
        for g in gs:
            g.begin_step()
            for i in clean.layers:
                g.append(i, *toks[s])
                outs.append(cascade.attention(q, g, i, 0.125))
            g.end_step()
        assert all(mx.all(mx.isfinite(o)).item() for o in outs)
        assert all(mx.array_equal(a, b).item() for a, b in zip(outs[:2], outs[2:]))
    assert gs[1].cap == 32 and gs[1].lens == (25, 20, 32)
    extra = Rows(40, [3], 2, 64, n_layers=2, seed=6, garbage=math.nan)
    for g in gs:
        g.add(extra.rows[0])
        g.take([3, 0, 2])
    assert_unused_zero(gs[1])
    q = queries(3, 4, 64)
    a, b = (cascade.attention(q, g, clean.layers[1], 0.125) for g in gs)
    assert mx.all(mx.isfinite(b)).item() and mx.array_equal(a, b).item()


# ---------------------------------------------------------------- holders


@pytest.mark.parametrize("dtype", [F16, F32, mx.bfloat16])
def test_release_round_trip_bit_exact(dtype):
    """build -> release gives every row its cache back bitwise ([0, offset)),
    also after grouped decode steps whose tokens were stored as fp32."""
    d = Rows(40, [6, 0, 17], 2, 64, n_layers=2, dtype=dtype, seed=7)
    g = d.group(chunk=16)
    for j, row in enumerate(d.rows):
        back = g.release(j)
        for i in d.layers:
            c, o = back[i], row[i]
            assert c.offset == o.offset and c.keys.dtype == dtype
            assert c.keys.shape[2] == c.offset + g.step
            assert mx.array_equal(c.keys[..., :c.offset, :], o.keys[..., :o.offset, :]).item()
            assert mx.array_equal(c.values[..., :c.offset, :], o.values[..., :o.offset, :]).item()
            assert mx.all(c.keys[..., c.offset:, :] == 0).item()
    # steps: the projections' tokens (row dtype) appended both ways
    mx.random.seed(70)
    for _ in range(19):                     # crosses cap (32 -> 48)
        g.begin_step()
        for i in d.layers:
            k = (mx.random.normal((3, 2, 1, 64)) * 3).astype(dtype)
            v = mx.random.normal((3, 2, 1, 64)).astype(dtype)
            g.append(i, k, v)
            for b, row in enumerate(d.rows):
                row[i].update_and_fetch(k[b:b + 1], v[b:b + 1])
        g.end_step()
    assert g.lens == (25, 19, 36)
    for j, row in enumerate(d.rows):
        back = g.release(j)
        for i in d.layers:
            n = row[i].offset
            assert back[i].offset == n
            assert mx.array_equal(back[i].keys[..., :n, :], row[i].keys[..., :n, :]).item()
            assert mx.array_equal(back[i].values[..., :n, :], row[i].values[..., :n, :]).item()
    # a released cache decodes on like any KVCache
    c = g.release(0)[d.layers[0]]
    k, _ = c.update_and_fetch(mx.ones((1, 2, 1, 64), dtype), mx.ones((1, 2, 1, 64), dtype))
    assert k.shape[2] == 40 + 25 + 1


@pytest.mark.parametrize("Hq,Hkv,D", LAYOUTS)
def test_decode_steps_match_rows(Hq, Hkv, D):
    """begin_step / append / attention / end_step over 2 layers and 20 steps
    (the stack grows twice at step 16) against the per-row KVCache path at
    every step and layer, chained like the model (layer 2's query depends on
    layer 1's output)."""
    d = Rows(40, [9, 2, 15, 0], Hkv, D, n_layers=2, seed=8)
    g = d.group(chunk=16, step=16)
    assert g.cap == 16
    scale = D ** -0.5
    caps = set()
    mx.random.seed(80)
    for s in range(20):
        q0 = queries(4, Hq, D)
        g.begin_step()
        caps.add(g.cap)
        q = q0
        for i in d.layers:
            k = mx.random.normal((4, Hkv, 1, D)).astype(F16)
            v = mx.random.normal((4, Hkv, 1, D)).astype(F16)
            g.append(i, k, v)
            out = cascade.attention(q, g, i, scale)
            for b, row in enumerate(d.rows):
                row[i].update_and_fetch(k[b:b + 1], v[b:b + 1])
            caches = [row[i] for row in d.rows]
            ref = per_row(q, caches, scale, fp32=True)
            today = per_row(q, caches, scale)
            e_c, e_t = rms_rel(out, ref), rms_rel(today, ref)
            assert e_c <= 3e-4 and e_c <= 1.1 * e_t + 1e-5, (s, i, e_c, e_t)
            q = (q0 + ref * 0.5).astype(F16)    # chained like the model's layers
        g.end_step()
    assert caps == {16, 32, 48} and g.lens == (29, 22, 35, 20)
    assert_unused_zero(g)


def test_take_add_equal_fresh_build():
    """add / take / grow give the same stacks (bitwise) as building the group
    from scratch over the same rows, so attention over them is the same."""
    d = Rows(40, [3, 21, 8, 0, 5], 2, 64, n_layers=2, seed=9)
    g = d.group(chunk=16, members=[0, 1, 2])
    assert g.cap == 32
    j = g.add(d.rows[3])                    # lazy (the default)
    j2 = g.add(d.rows[4], evaluate=True)
    assert (j, j2) == (3, 4) and g.lens == (3, 21, 8, 0, 5)
    fresh = d.group(chunk=16)
    for i in d.layers:
        assert mx.array_equal(g.sk[i], fresh.sk[i]).item()
        assert mx.array_equal(g.sv[i], fresh.sv[i]).item()
    g.take([4, 0, 2], evaluate=True)        # the 21-token row leaves: cap 32 -> 16
    assert g.lens == (5, 3, 8) and g.cap == 16
    fresh = d.group(chunk=16, members=[4, 0, 2])
    for i in d.layers:
        assert mx.array_equal(g.sk[i], fresh.sk[i]).item()
        assert mx.array_equal(g.kc[i], fresh.kc[i]).item()
    q = queries(3, 4, 64)
    assert mx.array_equal(cascade.attention(q, g, d.layers[1], 0.125),
                          cascade.attention(q, fresh, d.layers[1], 0.125)).item()
    g.grow(40)
    assert g.cap == 48
    assert_unused_zero(g)
    assert g.nbytes == sum(a.nbytes for a in (*g.kc.values(), *g.vc.values(),
                                              *g.sk.values(), *g.sv.values(), g.pmask))


def test_abort_step_zeroes_the_slot():
    d = Rows(40, [4, 6, 1], 2, 64, n_layers=2, seed=10)
    g = d.group(chunk=16)
    before = {i: (g.sk[i], g.sv[i]) for i in d.layers}
    g.begin_step()
    g.append(d.layers[0], mx.full((3, 2, 1, 64), math.nan, F16), mx.ones((3, 2, 1, 64), F16))
    g.abort_step()
    assert g.lens == (4, 6, 1)
    for i in d.layers:
        assert mx.array_equal(g.sk[i], before[i][0]).item()
        assert mx.array_equal(g.sv[i], before[i][1]).item()
    assert_unused_zero(g)
    out = cascade.attention(queries(3, 4, 64), g, d.layers[0], 0.125)
    assert mx.all(mx.isfinite(out)).item()


def test_prefix_equal():
    d = Rows(40, [4, 6, 1], 2, 64, n_layers=2, seed=11)
    g = d.group(chunk=16)
    assert all(g.prefix_equal(r) for r in d.rows)
    c = d.rows[2][d.layers[1]]
    c.keys[..., 17, 5] = c.keys[..., 17, 5] + 1
    assert not g.prefix_equal(d.rows[2])
    assert g.prefix_equal(d.rows[1])


def test_compile_traces_bounded(monkeypatch):
    """One compiled wrapper per shape key, LRU-bounded; evicted shapes are
    re-traced and still correct; the same shape reuses its wrapper."""
    monkeypatch.setattr(cascade, "MAX_TRACES", 2)
    d = Rows(40, [4, 6, 1], 2, 64, seed=12)
    g = d.group(chunk=16, step=16)
    i = d.layers[0]
    q = queries(3, 4, 64)
    ref = per_row(q, [r[i] for r in d.rows], 0.125, fp32=True)
    first = cascade.attention(q, g, i, 0.125)
    assert cascade.traces() == 1
    cascade.attention(q, g, i, 0.125)
    assert cascade.traces() == 1
    for need in (17, 33, 49):               # three new caps
        g.grow(need)
        out = cascade.attention(q, g, i, 0.125)
        assert rms_rel(out, ref) <= 3e-4
        assert cascade.traces() <= 2
    g.take([0, 1, 2])                       # back to cap 16: the first shape again, re-traced
    again = cascade.attention(q, g, i, 0.125)
    assert mx.array_equal(first, again).item()
    assert cascade.traces() == 2


def test_errors_leave_group_unchanged():
    d = Rows(40, [4, 6], 2, 64, n_layers=2, seed=13)
    with pytest.raises(ValueError):         # a member shorter than the prefix
        short = Rows(30, [0], 2, 64, n_layers=2, seed=13)
        cascade.PrefixGroup.build(d.rows[0], [d.rows[0], short.rows[0]], 40, d.layers, chunk=16)
    g = d.group(chunk=16)

    def state():
        return ([id(g.sk[i]) for i in d.layers], [id(g.sv[i]) for i in d.layers], list(g.lens), g.cap)

    before = state()
    bad = Rows(40, [100], 2, 64, n_layers=2, dtype=F32, seed=13)
    with pytest.raises(ValueError):         # dtype mismatch, after a cap growth was planned
        g.add(bad.rows[0])
    assert state() == before
    with pytest.raises(ValueError):
        g.take([])
    with pytest.raises(ValueError):
        g.take([0, 0])
    with pytest.raises(RuntimeError):
        g.append(d.layers[0], mx.zeros((2, 2, 1, 64), F16), mx.zeros((2, 2, 1, 64), F16))
    g.begin_step()
    with pytest.raises(RuntimeError):
        g.take([0])
    with pytest.raises(RuntimeError):
        g.release(0)
    with pytest.raises(ValueError):
        g.append(d.layers[0], mx.zeros((1, 2, 1, 64), F16), mx.zeros((1, 2, 1, 64), F16))
    g.abort_step()
    with pytest.raises(ValueError):         # queries for another group size
        cascade.attention(queries(3, 4, 64), g, d.layers[0], 0.125)


# ---------------------------------------------------------------- step protocol, bounds


def _tok(n, Hkv=2, D=64, dtype=F16):
    return mx.random.normal((n, Hkv, 1, D)).astype(dtype)


def test_step_protocol_enforced():
    """append exactly once per layer per step, before that layer's attention,
    and every layer before end_step; a refused end_step leaves the step open
    for abort_step."""
    d = Rows(40, [4, 6, 1], 2, 64, n_layers=2, seed=14)
    g = d.group(chunk=16)
    a, b = d.layers
    q = queries(3, 4, 64)
    mx.random.seed(140)
    cascade.attention(q, g, a, 0.125)               # outside a step: allowed
    g.begin_step()
    with pytest.raises(RuntimeError, match="before its append"):
        cascade.attention(q, g, a, 0.125)
    g.append(a, _tok(3), _tok(3))
    with pytest.raises(RuntimeError, match="second append"):
        g.append(a, _tok(3), _tok(3))
    cascade.attention(q, g, a, 0.125)
    with pytest.raises(RuntimeError, match="before its append"):
        cascade.attention(q, g, b, 0.125)
    with pytest.raises(RuntimeError, match="not appended"):
        g.end_step()
    with pytest.raises(KeyError):
        g.append(99, _tok(3), _tok(3))
    assert g._pending and g.lens == (4, 6, 1)      # still open
    g.abort_step()
    assert g.lens == (4, 6, 1)
    assert_unused_zero(g)
    for _ in range(2):                              # complete steps pass; the set resets per step
        g.begin_step()
        for i in d.layers:
            g.append(i, _tok(3), _tok(3))
            cascade.attention(q, g, i, 0.125)
        g.end_step()
    assert g.lens == (6, 8, 3)
    g.begin_step()
    with pytest.raises(RuntimeError, match="before its append"):
        cascade.attention(q, g, a, 0.125)           # last step's append does not count
    # dtype: only kv_dtype values (release's bit-exactness)
    with pytest.raises(ValueError, match="KV is"):
        g.append(a, _tok(3, dtype=F32), _tok(3))
    with pytest.raises(ValueError, match="KV is"):
        g.append(a, _tok(3), _tok(3, dtype=mx.bfloat16))
    g.abort_step()
    assert g.lens == (6, 8, 3)
    assert_unused_zero(g)


def test_bounds_guard_fires_before_any_write():
    """begin_step refuses (before a write index exists or anything is
    written) when max(lens) reaches cap or a stack is not (n, Hkv, cap, D).
    The guard is driven by a grow that forgets to grow: the mutation that
    killed the process with SIGBUS before. grow refuses mid-step; lens is
    read-only; check_lens compares the second copy of the write index."""
    d = Rows(40, [13, 2], 2, 64, n_layers=2, seed=15)
    g = d.group(chunk=16, step=16)
    assert g.cap == 16
    mx.random.seed(150)

    def step():
        g.begin_step()
        for i in d.layers:
            g.append(i, _tok(2), _tok(2))
        g.end_step()

    g.grow = lambda need, evaluate=False: None     # the bug: no growth
    for _ in range(3):                              # lengths 13, 14, 15 still fit cap 16
        step()
    assert g.lens == (16, 5) and g.cap == 16
    before = [id(a) for i in d.layers for a in (g.sk[i], g.sv[i])]
    with pytest.raises(RuntimeError, match="bounds"):
        g.begin_step()                              # would write slot 16 of 16
    assert not g._pending and g._upos is None
    assert [id(a) for i in d.layers for a in (g.sk[i], g.sv[i])] == before
    del g.grow                                      # the real grow again
    step()
    assert g.lens == (17, 6) and g.cap == 32
    assert_unused_zero(g)
    # a stack of the wrong width / row count
    good = g.sk[d.layers[1]]
    for bad in (g._pad(good, 48), good[:1]):
        g.sk[d.layers[1]] = bad
        with pytest.raises(RuntimeError, match="bounds"):
            g.begin_step()
        assert not g._pending
    g.sk[d.layers[1]] = good
    # grow is not allowed inside a step (it would reset the write index)
    g.begin_step()
    with pytest.raises(RuntimeError, match="grow inside"):
        g.grow(100)
    g.abort_step()
    assert g.cap == 32
    # lens: read-only, and checked against the rows' positions
    with pytest.raises(AttributeError):
        g.lens = [0, 0]
    g.check_lens([17, 6])
    g.check_lens((17, 6))
    with pytest.raises(RuntimeError, match="lengths"):
        g.check_lens([17, 7])
    with pytest.raises(RuntimeError, match="lengths"):
        g.check_lens([17])


def test_abort_step_eval_failure_marks_group(monkeypatch):
    """abort_step evaluates the restored stacks; if that eval raises (the
    step failed inside the graph), it raises and the group refuses further
    steps (the caller drops it)."""
    d = Rows(40, [4, 6], 2, 64, n_layers=2, seed=16)
    g = d.group(chunk=16)
    g.begin_step()
    g.append(d.layers[0], _tok(2), _tok(2))

    def boom(*a, **k):
        raise RuntimeError("[metal] out of memory (simulated)")

    monkeypatch.setattr(mx, "eval", boom)
    with pytest.raises(RuntimeError, match="drop the group"):
        g.abort_step()
    monkeypatch.undo()
    assert not g._pending and g.lens == (4, 6)
    with pytest.raises(RuntimeError, match="drop it"):
        g.begin_step()


# ---------------------------------------------------------------- step-ahead, release, traces


def test_step_ahead_equals_synchronous():
    """The scheduler's step-ahead: step k+1's graph (begin_step, append,
    attention; its queries and keys depend on step k's output) is built and
    queued before step k's output is read. 22 steps across two cap boundaries
    (16 -> 32 -> 48) and a lazy take while a step is in flight: every output
    bitwise equal to the synchronous run (donation is decided at encode time;
    a buffer still read by a queued step is copied, never overwritten)."""
    lens = [9, 2, 15, 0]

    def run(ahead):
        d = Rows(40, lens, 2, 64, n_layers=2, seed=17)
        g = d.group(chunk=16, step=16)
        mx.random.seed(170)
        q0s = [queries(4, 4, 64) for _ in range(22)]
        toks = [[(_tok(4), _tok(4)) for _ in d.layers] for _ in range(22)]
        mx.eval(q0s, toks)
        order, prev, outs, caps = [0, 1, 2, 3], None, [], set()
        for s in range(22):
            if s == 12:                             # a row leaves (lazy take) with step 11 queued
                keep = [3, 0, 2]
                g.take(keep)
                order = [order[j] for j in keep]
                prev = prev[mx.array(keep)]
            sel = mx.array(order)
            g.begin_step()
            caps.add(g.cap)
            q = q0s[s][sel]
            if prev is not None:
                q = (q + prev * 0.5).astype(F16)
            for li, i in enumerate(d.layers):
                k, v = (t[sel] for t in toks[s][li])
                if prev is not None:
                    k = (k + prev[:, :2] * 0.25).astype(F16)
                g.append(i, k, v)
                q = cascade.attention(q, g, i, 0.125)
            g.end_step()
            if ahead:
                mx.async_eval(q)                    # queued; read only after step s+1 is built
                if outs:
                    mx.eval(outs[-1])
            else:
                mx.eval(q)
            outs.append(q)
            prev = q
        mx.eval(outs)
        return outs, caps, g.lens

    sync, ahead = run(False), run(True)
    assert sync[1] == ahead[1] == {16, 32, 48}
    assert sync[2] == ahead[2] == (22, 31, 37)
    for s, (a, b) in enumerate(zip(sync[0], ahead[0])):
        assert mx.all(mx.isfinite(a)).item()
        assert mx.array_equal(a, b).item(), s


def test_release_then_continue_bitwise():
    """A row released from a group decodes on exactly like the row that was
    never grouped: 300 per-row update_and_fetch + library SDPA steps (both
    caches cross a capacity boundary; their capacities differ: P + S + step
    vs multiples of 256) give bitwise equal outputs at every step. Suffixes
    0 / 9 / 37 / 255 / 256 (empty, short, stack-cap edges)."""
    P, lens = 300, [0, 9, 37, 255, 256]
    d = Rows(P, lens, 2, 64, n_layers=1, seed=18)
    g = d.group(chunk=128, step=256)
    i = d.layers[0]
    mx.random.seed(180)
    for _ in range(3):                              # a few grouped steps first
        g.begin_step()
        k, v = _tok(5), _tok(5)
        g.append(i, k, v)
        g.end_step()
        for b, row in enumerate(d.rows):
            row[i].update_and_fetch(k[b:b + 1], v[b:b + 1])
    mine = [g.release(j)[i] for j in range(5)]
    theirs = [row[i] for row in d.rows]
    caps0 = [(a.keys.shape[2], b.keys.shape[2]) for a, b in zip(mine, theirs)]
    assert [a for a, _ in caps0] == [P + S + 3 + 256 for S in lens]
    for _ in range(300):
        q, k, v = queries(5, 4, 64), _tok(5), _tok(5)
        same = []
        for j in range(5):
            o = []
            for c in (mine[j], theirs[j]):
                kk, vv = c.update_and_fetch(k[j:j + 1], v[j:j + 1])
                o.append(lib_sdpa(q[j:j + 1], kk, vv, cache=c, scale=0.125, mask=None))
            same.append(mx.array_equal(*o))
        assert all(x.item() for x in same)
    for (a0, b0), a, b in zip(caps0, mine, theirs):
        assert a.offset == b.offset
        assert a.keys.shape[2] > a0 and b.keys.shape[2] > b0     # both grew


def test_compile_eviction_alternating_scales(monkeypatch):
    """MAX_TRACES = 1 and 4 scales in turn: every call evicts the previous
    wrapper (whose closure address, so fun_id, may be reused by the next
    one); every result is the right scale's (vs the fp32 reference) and
    bitwise the first result for that scale -- an evicted trace is erased,
    never run for another scale."""
    monkeypatch.setattr(cascade, "MAX_TRACES", 1)
    d = Rows(40, [4, 6, 1], 2, 64, seed=19)
    g = d.group(chunk=16)
    i = d.layers[0]
    q = queries(3, 4, 64, 4.0)
    caches = [r[i] for r in d.rows]
    scales = [0.125, 0.1, 0.0625, 0.2]
    refs = {s: per_row(q, caches, s, fp32=True) for s in scales}
    first = {}
    for n in range(120):
        s = scales[n % 4]
        out = cascade.attention(q, g, i, s)
        assert cascade.traces() == 1
        assert rms_rel(out, refs[s]) <= 3e-4, (n, s)
        if s in first:
            assert mx.array_equal(out, first[s]).item(), (n, s)
        else:
            first[s] = out


# ---------------------------------------------------------------- GPU only


@gpu_only
def test_gpu_rms_criterion_9b():
    """Plan B.7's unit criterion at 9B shapes on Metal (B_g 16; P 8192 and
    8150 (padded); S 64 / 512; flat and peaked queries; 4 query draws per
    case = 32 pairs): rms rel vs the fp32 reference <= 2.3e-4 in every pair,
    and <= the per-row fused kernel's own error in >= 95 % of the pairs and
    in aggregate (mean over the pairs)."""
    B, DRAWS = 16, 4
    pairs = []
    for P in (8192, 8150):
        for S in (64, 512):
            d = Rows(P, [S - (b % 8) for b in range(B)], 4, 256, seed=P + S)
            g = d.group(chunk=1024, step=256)
            caches = [r[d.layers[0]] for r in d.rows]
            for qstd in (1.0, 4.0):
                for _ in range(DRAWS):
                    q = queries(B, 16, 256, qstd)
                    out = cascade.attention(q, g, d.layers[0], 1 / 16)
                    ref = per_row(q, caches, 1 / 16, fp32=True)
                    e_c, e_t = rms_rel(out, ref), rms_rel(per_row(q, caches, 1 / 16), ref)
                    assert mx.all(mx.isfinite(out)).item()
                    pairs.append((P, S, qstd, e_c, e_t))
            del d, g, caches
    for P, S, qstd, e_c, e_t in pairs:
        print(f"\nP={P} S={S} qstd={qstd}: cascade {e_c:.3e}  per-row {e_t:.3e}", end="")
    wins = sum(e_c <= e_t for *_, e_c, e_t in pairs)
    mc = statistics.mean(p[3] for p in pairs)
    mt = statistics.mean(p[4] for p in pairs)
    print(f"\ncascade <= per-row in {wins}/{len(pairs)}; mean {mc:.3e} vs {mt:.3e}")
    assert all(p[3] <= 2.3e-4 for p in pairs), max(p[3] for p in pairs)
    assert wins >= 0.95 * len(pairs), (wins, len(pairs))
    assert mc <= mt, (mc, mt)


@gpu_only
def test_gpu_timing_smoke():
    """B_g 16, P 8K, S ~512 (ragged), 8 FA layers of the 9B: one decode step
    = per layer the update + attention, layers chained, one eval + sync.
    Today = per-row KVCache.update_and_fetch + the library SDPA (exactly
    _rows_attention's loop); cascade = append + attention. Interleaved rounds,
    medians; the cascade must be faster."""
    B, P, S, NL = 16, 8192, 512, 8
    d = Rows(P, [S - (b % 8) for b in range(B)], 4, 256, n_layers=NL, seed=21)
    g = d.group(chunk=1024, step=256)
    for row in d.rows:                      # the rows' caches stay only for "today"
        for i in d.layers:
            mx.eval(row[i].keys)
    mx.random.seed(22)
    qs = [queries(B, 16, 256) for _ in range(NL)]
    kv = [mx.random.normal((B, 4, 1, 256)).astype(F16) for _ in range(NL)]
    mx.eval(qs, kv)

    def step_today():
        prev = None
        for li, i in enumerate(d.layers):
            q = qs[li] if prev is None else qs[li] + prev * 1e-3
            outs = []
            for b, row in enumerate(d.rows):
                c = row[i]
                k, v = c.update_and_fetch(kv[li][b:b + 1], kv[li][b:b + 1])
                outs.append(lib_sdpa(q[b:b + 1], k, v, cache=c, scale=1 / 16, mask=None))
            prev = mx.concatenate(outs, axis=0)
        mx.eval(prev, [(r[i].keys, r[i].values) for r in d.rows for i in d.layers])

    def step_cascade():
        prev = None
        g.begin_step()
        for li, i in enumerate(d.layers):
            q = qs[li] if prev is None else qs[li] + prev * 1e-3
            g.append(i, kv[li], kv[li])
            prev = cascade.attention(q, g, i, 1 / 16)
        g.end_step()
        mx.eval(prev, list(g.sk.values()), list(g.sv.values()))

    def timed(fn):
        mx.synchronize()
        t = time.perf_counter()
        fn()
        mx.synchronize()
        return (time.perf_counter() - t) * 1e3

    for _ in range(3):
        step_today()
        step_cascade()
    rounds = {"today": [], "cascade": []}
    for r in range(6):
        order = [("today", step_today), ("cascade", step_cascade)]
        for name, fn in (order if r % 2 == 0 else order[::-1]):
            rounds[name].append(statistics.median(timed(fn) for _ in range(6)))
    med = {k: statistics.median(v) for k, v in rounds.items()}
    print(f"\nB_g={B} P={P} S~{S} {NL} layers, ms/step (median of 6 round medians): "
          f"today {med['today']:.2f} (rounds {', '.join(f'{x:.1f}' for x in rounds['today'])}) "
          f"cascade {med['cascade']:.2f} (rounds {', '.join(f'{x:.1f}' for x in rounds['cascade'])})"
          f"  traces {cascade.traces()}")
    assert med["cascade"] < med["today"], med

    # step-ahead, as the scheduler runs: step k+1 is built and queued before
    # step k's output is read. Donation is decided when step k+1's scatter is
    # ENCODED: if step k's buffer reading that stack is still in flight, the
    # scatter copies the layer's stack. MLX keeps <= 10 command buffers of ~50
    # ops in flight, so that only works when a step spans more ops than that
    # window; the real 9B has 3 GDN layers + 4 MLPs between two FA layers.
    # FILL elementwise ops per FA layer stand in for them (an 8-layer step of
    # attention alone, FILL 0, copies every stack: ~+1 GB, step x1.4 -- printed,
    # not asserted). The step-ahead peak must stay within one layer's K+V stack
    # of the synchronous run's (all copies would be ~8 layers).
    cap0 = g.cap
    layer_kv = g.sk[d.layers[0]].nbytes + g.sv[d.layers[0]].nbytes
    del rounds

    def steps(n, ahead, fill):
        prev = None
        for _ in range(n):
            x = prev
            g.begin_step()
            for li, i in enumerate(d.layers):
                q = qs[li] if x is None else qs[li] + x * 1e-3
                g.append(i, kv[li], kv[li])
                x = cascade.attention(q, g, i, 1 / 16)
                for _ in range(fill):
                    x = x * 1.0
            g.end_step()
            if ahead:
                mx.async_eval(x)
                if prev is not None:
                    mx.eval(prev)               # read step k after step k+1 is queued
            else:
                mx.eval(x)
            prev = x
        mx.eval(prev)

    def peak(ahead, fill, n=12):
        mx.synchronize()
        mx.clear_cache()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        t = time.perf_counter()
        steps(n, ahead, fill)
        mx.synchronize()
        return mx.get_peak_memory() - base, (time.perf_counter() - t) * 1e3 / n

    FILL = 160
    res = {(a, f): peak(a, f) for f in (FILL, 0) for a in (False, True)}
    for (a, f), (gr, ms) in res.items():
        print(f"{'step-ahead' if a else 'sync      '} fill {f:3d}: {ms:.2f} ms/step, "
              f"peak growth {gr / 2**20:.1f} MiB (one layer's K+V {layer_kv / 2**20:.0f} MiB)")
    assert g.cap == cap0                    # no growth inside the measured window
    growth, sync = res[(True, FILL)][0], res[(False, FILL)][0]
    assert growth < sync + layer_kv, (growth, sync, layer_kv)
