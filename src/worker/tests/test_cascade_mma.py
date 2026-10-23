"""QWEN_CASCADE_KERNEL: cascade_mma.py (prefix attention on simdgroup MMA over
an fp16 prefix) and cascade.PrefixGroup's kernel mode, plus the worker's side
(knob, charge, formation).

CPU by default (conftest): cascade_mma then computes its outputs with mx ops
(`reference`), so the group's kernel-mode bookkeeping (the aliased fp16
prefix, release, steps, the scheduler's charge and formation) runs here; the
Metal kernel itself only with `CASCADE_MMA_GPU=1` on a Metal device (the
GPU lease: `CASCADE_MMA_GPU=1 .venv/bin/python -m pytest -q
tests/test_cascade_mma.py`): the kernel against an fp32 reference at the 9B
layout (G = 4, D = 256, P 4K-32K, 3-16 rows, NaN past P), partial blocks /
chunks on small shapes, and the full grouped attention (kernel prefix +
suffix + merge) against per-row fp32 SDPA, next to today's fp32 cascade.
Knob off: the existing tests (test_cascade*.py) run the default path; here
only that the knob is off by default and that nothing of the kernel runs."""
import logging
import math
import os
import types

import mlx.core as mx
import pytest

import cascade
import cascade_mma
import worker
from conftest import build_lm, prompts, request, run, scheduler
from test_cascade import _cache, per_row, queries, rms_rel
from test_cascade_sched import (REAL_WINS, FakeRow, Logits, Spy, any_three, chats,
                                check_accounting, one_per_tick, prefix_rows)

F16, F32 = mx.float16, mx.float32
GPU = os.environ.get("CASCADE_MMA_GPU") == "1"
gpu_only = pytest.mark.skipif(not (GPU and mx.metal.is_available()),
                              reason="CASCADE_MMA_GPU=1 and Metal only")
GIB = 1 << 30


@pytest.fixture
def gpu():
    prev = mx.default_device()
    mx.set_default_device(mx.gpu)
    cascade.clear_traces()
    yield
    mx.set_default_device(prev)
    cascade.clear_traces()


@pytest.fixture(autouse=True)
def traces():
    cascade.clear_traces()
    yield
    cascade.clear_traces()


class Rows:
    """B rows sharing a P-token prefix (test_cascade.Rows, whose `group`
    builds in kernel mode by default): rows[b][layer] a KVCache whose buffers
    are P + S_b rounded up to 256 slots, as a prefill leaves them."""

    def __init__(self, P, lens, Hkv, D, n_layers=1, dtype=F16, seed=0, garbage=None):
        mx.random.seed(seed)
        self.P, self.lens, self.Hkv, self.D = P, list(lens), Hkv, D
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

    def group(self, kernel=True, slack=cascade.ALIAS_SLACK, step=16, chunk=16):
        return cascade.PrefixGroup.build(self.rows[0], self.rows, self.P, self.layers,
                                         chunk=chunk, step=step, kernel=kernel, slack=slack)


LAYOUTS = [(4, 2, 64), (16, 4, 256)]       # the tiny test model, the 9B


# ---------------------------------------------------------------- group, CPU


@pytest.mark.parametrize("Hq,Hkv,D", LAYOUTS)
@pytest.mark.parametrize("P", [40, 300])
def test_kernel_group_matches_rows(Hq, Hkv, D, P):
    """A kernel-mode group's attention (reference prefix on the CPU + the
    suffix merge) vs per-row fp32 SDPA: within the fp32 cascade's bound, and
    within fp rounding of it; ragged suffixes, an empty one."""
    d = Rows(P, [9, 0, 17, 4], Hkv, D, seed=P)
    g = d.group()
    assert g.kernel and g.aliased and g.pmask is None and g.c == 0 and g.Ppad == P
    g32 = d.group(kernel=False)
    i = d.layers[0]
    caches = [r[i] for r in d.rows]
    for qstd in (1.0, 4.0):
        q = queries(4, Hq, D, qstd)
        out = cascade.attention(q, g, i, D ** -0.5)
        ref = per_row(q, caches, D ** -0.5, fp32=True)
        assert out.shape == (4, Hq, 1, D) and out.dtype == F16
        assert rms_rel(out, ref) <= 3e-4
        assert rms_rel(out, cascade.attention(q, g32, i, D ** -0.5)) <= 3e-4


def test_kernel_group_aliases_the_donor_buffer():
    """The prefix is the donor row's own KVCache buffer (no copy: the build
    allocates only the stacks), the whole buffer is counted in nbytes, and it
    survives the donor's KVCache being dropped (the scheduler replaces it by
    a _GroupRef)."""
    d = Rows(2000, [5, 7, 3], 4, 256, seed=1)
    donor = d.rows[0][d.layers[0]]
    cap = donor.keys.shape[2]
    assert cap == 2048                                  # 2005 rounded up to 256
    mx.eval([(c.keys, c.values) for r in d.rows for c in r.values()])
    before = mx.get_active_memory()
    g = d.group()
    grew = mx.get_active_memory() - before
    stacks = sum(g.sk[i].nbytes + g.sv[i].nbytes for i in d.layers)
    assert g.aliased and g.kc[d.layers[0]].shape == (1, 4, cap, 256)
    assert grew <= stacks + (1 << 20), (grew, stacks)   # no prefix copy (it would be 2 x 8 MiB)
    assert g.nbytes == 2 * donor.keys.nbytes + stacks
    q = queries(3, 16, 256)
    want = cascade.attention(q, g, d.layers[0], 1 / 16)
    mx.eval(want)
    caches = [r[d.layers[0]] for r in d.rows]
    ref = per_row(q, caches, 1 / 16, fp32=True)
    for r in d.rows:
        r.clear()                                       # every row's KVCache gone
    del donor, caches
    got = cascade.attention(q, g, d.layers[0], 1 / 16)
    assert mx.array_equal(got, want).item() and rms_rel(got, ref) <= 3e-4


def test_kernel_group_copies_a_long_donor_tail():
    """More than `slack` slots past P: a fresh (1, Hkv, P + 1, D) copy of
    [0, P) (+ a zero slot), not the donor's buffer; same outputs."""
    d = Rows(40, [100, 3, 5], 2, 64, seed=2)          # donor: 140 -> 256 slots, 216 past P
    g, ga = d.group(slack=200), d.group()
    assert not g.aliased and ga.aliased
    k = g.kc[d.layers[0]]
    assert k.shape == (1, 2, 41, 64) and mx.all(k[..., 40:, :] == 0).item()
    assert mx.array_equal(k[..., :40, :], d.rows[0][d.layers[0]].keys[..., :40, :]).item()
    q = queries(3, 4, 64)
    assert mx.array_equal(cascade.attention(q, g, d.layers[0], 0.125),
                          cascade.attention(q, ga, d.layers[0], 0.125)).item()


@pytest.mark.parametrize("dtype", [F32, mx.bfloat16])
def test_kernel_mode_is_fp16_only(dtype):
    d = Rows(40, [3, 4, 5], 2, 64, dtype=dtype)
    with pytest.raises(ValueError, match="fp16 KV"):
        d.group()
    assert not d.group(kernel=False).kernel


def test_kernel_release_round_trip_and_steps_bit_exact():
    """build -> steps (the stacks grow) -> release gives every row its cache
    bitwise, as in fp32 mode; the released prefix is the donor's fp16 values;
    garbage past the donor's offset (NaN) never reaches an output; a row
    added and the stacks re-taken keep the outputs close to per-row."""
    d = Rows(40, [6, 0, 17], 2, 64, n_layers=2, seed=7, garbage=math.nan)
    g = d.group(step=16)
    mx.random.seed(70)
    for s in range(19):
        g.begin_step()
        for i in d.layers:
            k = (mx.random.normal((3, 2, 1, 64)) * 3).astype(F16)
            v = mx.random.normal((3, 2, 1, 64)).astype(F16)
            g.append(i, k, v)
            out = cascade.attention(queries(3, 4, 64), g, i, 0.125)
            assert mx.all(mx.isfinite(out)).item()
            for b, row in enumerate(d.rows):
                row[i].update_and_fetch(k[b:b + 1], v[b:b + 1])
        g.end_step()
    assert g.lens == (25, 19, 36)
    for j, row in enumerate(d.rows):
        back = g.release(j)
        for i in d.layers:
            n = row[i].offset
            assert back[i].offset == n and back[i].keys.dtype == F16
            assert mx.array_equal(back[i].keys[..., :n, :], row[i].keys[..., :n, :]).item()
            assert mx.array_equal(back[i].values[..., :n, :], row[i].values[..., :n, :]).item()
            assert mx.all(back[i].keys[..., n:, :] == 0).item()
    assert all(g.prefix_equal(r) for r in d.rows)
    extra = Rows(40, [3], 2, 64, n_layers=2, seed=7)
    assert not g.prefix_equal(extra.rows[0])            # other prefix bits
    g.add(d.rows[1])                                    # a row's copy (same prefix)
    g.take([3, 0, 2])
    i = d.layers[1]
    q = queries(3, 4, 64)
    out = cascade.attention(q, g, i, 0.125)
    ref = per_row(q, [d.rows[1][i], d.rows[0][i], d.rows[2][i]], 0.125, fp32=True)
    assert rms_rel(out, ref) <= 3e-4


def test_reference_is_the_fp32_prefix_math():
    """cascade_mma.reference's (o_p, lse_p) are the fp32 path's prefix
    statistics: o_p merged with an empty suffix is the prefix-only
    attention; lse_p is the per-row logsumexp of the prefix scores."""
    mx.random.seed(3)
    q = queries(5, 4, 64, 2.0)
    k = mx.random.normal((1, 2, 64, 64)).astype(F16)
    v = mx.random.normal((1, 2, 64, 64)).astype(F16)
    o, lse = cascade_mma.reference(q, k, v, 37, 0.125)
    assert o.shape == (2, 10, 64) and lse.shape == (2, 10) and o.dtype == F32
    kk, vv = (mx.broadcast_to(x[..., :37, :].astype(F32), (5, 2, 37, 64)) for x in (k, v))
    want = mx.fast.scaled_dot_product_attention(q.astype(F32), kk, vv, scale=0.125)
    got = o.reshape(2, 5, 2, 64).transpose(1, 0, 2, 3).reshape(5, 4, 1, 64)
    assert mx.allclose(got, want, atol=1e-5).item()
    s = (q.astype(F32) * 0.125).reshape(5, 2, 2, 64) @ k[0, :, :37, :].astype(F32).swapaxes(-1, -2)[None]
    assert mx.allclose(lse.reshape(2, 5, 2).transpose(1, 0, 2), mx.logsumexp(s, axis=-1),
                       atol=1e-5).item()
    with pytest.raises(ValueError, match="fp16 only"):
        cascade_mma.prefix_attention(q.astype(F32), k, v, 37, 0.125)
    with pytest.raises(ValueError, match="P 65 of 64"):
        cascade_mma.prefix_attention(q, k, v, 65, 0.125)


def test_chunk_and_partial_bytes():
    """pick_chunk aims at TARGET_TG threadgroups between MIN_CHUNK and
    MAX_CHUNK; partial_bytes bounds the partials of any default chunk."""
    assert cascade_mma.pick_chunk(32768, 4, 2) == 1024          # 32 x 4 x 2 = 256
    assert cascade_mma.pick_chunk(8192, 4, 2) == 256            # 32 x 4 x 2
    assert cascade_mma.pick_chunk(8192, 4, 1) == 256            # 32 x 4 x 1 = 128 < 256: floor
    assert cascade_mma.pick_chunk(200000, 4, 2) == 2048
    for P in (4096, 8192, 33000):
        for rows in (3, 16):
            m0 = rows * 4
            c = cascade_mma.pick_chunk(P, 4, -(-m0 // 32))
            assert c % 16 == 0
            used = (cascade_mma.items(P, c) * 4 * m0 * (256 + 2) + 4 * m0 * (256 + 1)) * 4
            assert used <= cascade_mma.partial_bytes(P, rows, 16, 256)


# ---------------------------------------------------------------- worker, CPU


@pytest.fixture(scope="module")
def dense16():
    """The tiny dense model in fp16 (kernel mode is fp16 only)."""
    return build_lm("dense", None, F16)


@pytest.fixture
def casc16(monkeypatch):
    """test_cascade_sched's `casc` setup (8-token slabs, 16-token prefixes,
    any 3 rows win) with QWEN_CASCADE_KERNEL on."""
    for k, v in dict(PROMPT_CACHE=True, PROMPT_CACHE_REPLY=False, PREFILL_STEP=8, PREFILL_MIN_STEP=4,
                     PROMPT_CACHE_STRIDE=16, PROMPT_CACHE_CKPTS=12, PROMPT_CACHE_KEEP_TOKENS=8,
                     PROMPT_CACHE_KEEP_WAIT_S=float("inf"), CASCADE=True, CASCADE_MIN_ROWS=3,
                     CASCADE_MIN_PREFIX=16, CASCADE_CHUNK=16, CASCADE_STEP=16,
                     CASCADE_CHECK=True, CASCADE_KERNEL=True).items():
        monkeypatch.setattr(worker, k, v)
    monkeypatch.setattr(worker._Scheduler, "_group_wins", any_three)


def test_knob_default_off(dense16, monkeypatch):
    """QWEN_CASCADE_KERNEL is off unless "1"; off, an fp16 scheduler keeps
    the fp32 prefix and the load log says what it said before."""
    assert worker._cascade_kernel_knob({}) is False
    assert worker._cascade_kernel_knob({"QWEN_CASCADE_KERNEL": "1"}) is True
    for off in ("0", "", "true", "on", "2"):
        assert worker._cascade_kernel_knob({"QWEN_CASCADE_KERNEL": off}) is False
    if "QWEN_CASCADE_KERNEL" not in os.environ:
        assert worker.CASCADE_KERNEL is False
    for k, v in dict(CASCADE=True, CASCADE_MIN_ROWS=8, CASCADE_MIN_PREFIX=4096,
                     CASCADE_CHECK=False, PROMPT_CACHE=True, CASCADE_KERNEL=False).items():
        monkeypatch.setattr(worker, k, v)
    s = scheduler(dense16, max_batch=16)
    assert not s.cascade_kernel and s.cascade_state() == "on: >= 8 rows sharing >= 4096 prompt tokens"
    monkeypatch.setattr(worker, "CASCADE_KERNEL", True)
    s = scheduler(dense16, max_batch=16)
    assert s.cascade_kernel and s.cascade_state().endswith("MMA kernel over the fp16 prefix "
                                                           "(QWEN_CASCADE_KERNEL)")


def test_knob_ignored_without_fp16(dense, monkeypatch):
    """An fp32 (or bf16) model: the knob is ignored, said once in the load log."""
    for k, v in dict(CASCADE=True, PROMPT_CACHE=True, CASCADE_KERNEL=True).items():
        monkeypatch.setattr(worker, k, v)
    s = scheduler(dense, max_batch=16)
    assert s.cascade and not s.cascade_kernel
    assert "QWEN_CASCADE_KERNEL=1 ignored" in s.cascade_state()


def nine_b(lm, monkeypatch, kernel):
    """test_cascade_sched.nine_b with head_dim (the kernel's partials) and the
    knob: 9B byte layout (32 KiB fp16 KV per token over 8 FA layers, 16 q
    heads, head_dim 256), stack step 256, 64 MiB slack, the real rule."""
    s = scheduler(lm, max_batch=16)
    s.kv_fp16 = 32 << 10
    s.fa_layers = list(range(3, 32, 4))
    s.lm = types.SimpleNamespace(args=types.SimpleNamespace(num_attention_heads=16, head_dim=256))
    s.cascade_kernel = kernel
    for k, v in dict(CASCADE_CHUNK=1024, CASCADE_STEP=256, CASCADE_SLACK=64 << 20,
                     CASCADE_ALIAS_SLACK=1024).items():
        monkeypatch.setattr(worker, k, v)
    monkeypatch.setattr(worker._Scheduler, "_group_wins", REAL_WINS)
    return s


def test_kernel_charge_bytes(dense16, monkeypatch):
    """_group_bytes in kernel mode, 9B layout, term by term: the fp16 prefix
    (P + ALIAS_SLACK tokens), the fp32 stack (+5 %), the suffix scores + the
    kernel's partials per layer + slack + one layer's stack moves, never
    below the 2-row dissolve floor. Then what it changes: the charge is
    smaller than the fp32 prefix's at every size, and a group wins (_group_wins)
    at fewer rows."""
    s = nine_b(dense16, monkeypatch, True)
    kv, member = 32 << 10, s._member_bytes()

    def rows(P, n, S, reserve=4096):
        return [FakeRow(P + S, P + S + reserve, s._bytes(P + S + reserve)) for _ in range(n)]

    P, st, n = 8192, 4352, 16
    scores = 8 * (8 * 16 * n * st + cascade_mma.partial_bytes(P, n, 16, 256)) + (64 << 20)
    moves = 2 * n * st * 2 * kv // 8
    floor = 2 * int((P + st) * kv * 1.05)
    want = (P + 1024) * kv + int(n * st * 2 * kv * 1.05) + max(scores + moves, floor)
    assert s._group_bytes(P, rows(P, n, 150)) == want
    s32 = nine_b(dense16, monkeypatch, False)
    for P in (4096, 8192, 32768):
        for k in (3, 8, 16):
            assert s._group_bytes(P, rows(P, k, 150)) < s32._group_bytes(P, rows(P, k, 150))
    # the rule at the real lengths (~150-token suffixes): kernel mode wins from
    # 4 rows at every prefix (fp32: 5; 4 only from 32K), never at 2..3
    for P in (4096, 8192, 32768):
        for k in (4, 5, 8, 16):
            assert s._group_wins(P, rows(P, k, 150)), (P, k)
        for k in (2, 3):                    # the 2-row floor: never (3 x 32K: 3.20 vs 3.16 GiB)
            assert not s._group_wins(P, rows(P, k, 150)), (P, k)
        assert s32._group_wins(P, rows(P, 4, 150)) == (P == 32768), P
    # a 16 x 32K group: 1 GiB less than the fp32 prefix's charge
    rs = rows(32768, 16, 150)
    assert 0.9 * GIB <= s32._group_bytes(32768, rs) - s._group_bytes(32768, rs) <= 1.1 * GIB


def _grouped_logits(lm, kernel, P=32):
    """test_cascade_sched.test_grouped_logits' run: 6 rows sharing a 36-token
    prefix, rows 0..2 grouped over [0, P) (fp32 or kernel mode), 7
    teacher-forced steps -> (per-step logits, the group)."""
    shared = prompts([36], 70)[0]
    tails = prompts([2, 5, 3, 6, 1, 4], 71)
    feed = prompts([7] * 6, 72)
    layers = lm.model.layers
    fa = [i for i, x in enumerate(layers) if not x.is_linear]
    dt = lm.model.embed_tokens(mx.array([[0]])).dtype
    base, caches = prefix_rows(lm, shared, tails)
    gdn = worker._gdn_stack(lm, caches, dt)
    pg = None
    if kernel is not None:
        pg = cascade.PrefixGroup.build(caches[0], caches[:3], P, fa, chunk=16, step=16,
                                       kernel=kernel)
    pos = [36 + len(t) for t in tails]
    out = []
    for s in range(7):
        plan = None
        if pg is not None:
            plan = worker._StepPlan([3, 4, 5], [(pg, mx.array([0, 1, 2]))],
                                    mx.array([3, 4, 5, 0, 1, 2]), [])
            pg.begin_step()
        cache = [gdn[i] if i in gdn else worker.RowsKV([c[i] for c in caches], i, plan)
                 for i in range(len(layers))]
        lg = lm(mx.array([f[s] for f in feed])[:, None], cache=cache,
                position_ids=mx.array([[p] for p in pos], dtype=mx.int32)).logits[:, -1, :]
        if pg is not None:
            pg.end_step()
        mx.eval(lg)
        out.append(lg.astype(F32))
        pos = [p + 1 for p in pos]
    return out, pg


@pytest.mark.parametrize("model", ["dense16", "dense_fused"])
def test_kernel_grouped_logits(model, request):
    """Through the model (fp16; dense_fused = 8-bit + GDN fuse): kernel-mode
    grouped rows stay as close to the per-row path as fp32-mode grouped rows
    (teacher-forced, top-1 equal every step); the ungrouped rows of the batch
    are bit-identical."""
    lm = request.getfixturevalue(model)
    ref, _ = _grouped_logits(lm, None)
    f32, _ = _grouped_logits(lm, False)
    ker, pg = _grouped_logits(lm, True)
    assert pg.kernel
    d32 = dk = 0.0
    for a, b, c in zip(ref, f32, ker):
        assert mx.array_equal(a[3:], c[3:]).item()
        d32 = max(d32, mx.abs(a[:3] - b[:3]).max().item())
        dk = max(dk, mx.abs(a[:3] - c[:3]).max().item())
        assert mx.array_equal(mx.argmax(a[:3], axis=-1), mx.argmax(c[:3], axis=-1)).item()
    assert dk <= 3e-2 and dk <= 2 * d32 + 1e-3, (dk, d32)


@pytest.mark.parametrize("check", [True, False], ids=["check", "nocheck"])
def test_kernel_wave(dense16, casc16, monkeypatch, caplog, check):
    """6 requests on one system prompt, one per tick, fp16 model, kernel mode:
    the group forms (kernel, its prefix the donor's aliased buffer: the one
    with the fewest slots past P), the rest join, the accounting holds every
    tick (nbytes <= the charge), every row's logits stay as close to the
    batch without groups as with today's fp32 groups, nothing is left at
    the end."""
    monkeypatch.setattr(worker, "CASCADE_CHECK", check)
    caplog.set_level(logging.INFO, logger="qwen.worker")
    ids = chats(6)
    lg = Logits(monkeypatch)
    outs, built = {}, []
    build = cascade.PrefixGroup.build

    def spy_build(*a, **k):
        g = build(*a, **k)
        built.append(g)
        return g
    monkeypatch.setattr(cascade.PrefixGroup, "build", spy_build)
    for name, on, kern in (("plain", False, False), ("fp32", True, False), ("kernel", True, True)):
        monkeypatch.setattr(worker, "CASCADE", on)
        monkeypatch.setattr(worker, "CASCADE_KERNEL", kern)
        lg.start()
        spy = Spy(monkeypatch)
        sched = scheduler(dense16, max_batch=8)
        assert sched.cascade_kernel == kern
        reqs = [request(p, 20, f"r{i}") for i, p in enumerate(ids)]
        outs[name] = run(sched, reqs, arrive=one_per_tick(reqs), on_tick=spy.tick)
        assert sched.groups == [] and sched.reserved == 0 and sched.n_grouped == 0
        if on:
            (P, members), = spy.builds
            assert P == 151 and len(members) == 3 and spy.calls > 0
            assert [rid for _, rid in spy.adds] == ["r3", "r4", "r5"]
    g32, gk = built
    assert not g32.kernel and gk.kernel and gk.aliased
    d32, dk = lg.max_diff(0, 1), lg.max_diff(0, 2)
    assert 0 < dk <= max(2 * d32, 1e-3), (dk, d32)
    assert any("prefix fp16 aliased + stacks fp32" in m for m in caplog.messages)


def test_kernel_wave_picks_the_tightest_donor(dense16, casc16, monkeypatch):
    """Formation in kernel mode aliases the candidate whose buffers have the
    fewest slots past P (a cold producer's slabs and a hit's restore tail
    leave different spare capacity); the accounting holds every tick."""
    seen = []
    build = cascade.PrefixGroup.build

    def spy_build(source, members, P, layers, **k):
        seen.append((source[layers[0]].keys.shape[2],
                     [m[layers[0]].keys.shape[2] for m in members]))
        return build(source, members, P, layers, **k)
    monkeypatch.setattr(cascade.PrefixGroup, "build", spy_build)
    sched = scheduler(dense16, max_batch=8)
    reqs = [request(p, 6, f"r{i}") for i, p in enumerate(chats(4))]
    run(sched, reqs, arrive=one_per_tick(reqs), on_tick=lambda t, s: check_accounting(s))
    (src, caps), = seen
    assert len(caps) == 3 and src == min(caps), (src, caps)


def test_knob_off_never_runs_the_kernel(dense16, casc16, monkeypatch):
    """QWEN_CASCADE_KERNEL off on an fp16 model with cascade on: groups form
    in fp32 mode and cascade_mma never runs."""
    monkeypatch.setattr(worker, "CASCADE_KERNEL", False)
    monkeypatch.setattr(cascade_mma, "prefix_attention", lambda *a, **k: pytest.fail("kernel ran"))
    spy = Spy(monkeypatch)
    sched = scheduler(dense16, max_batch=8)
    assert not sched.cascade_kernel
    reqs = [request(p, 8, f"r{i}") for i, p in enumerate(chats(4))]
    run(sched, reqs, arrive=one_per_tick(reqs), on_tick=spy.tick)
    assert spy.builds and spy.calls > 0


# ---------------------------------------------------------------- Metal


def _prefix_data(B, Hq, Hkv, D, P, cap, qstd=2.0, seed=0):
    mx.random.seed(seed)
    q = (mx.random.normal((B, Hq, 1, D)) * qstd).astype(F16)
    k = mx.random.normal((1, Hkv, cap, D)).astype(F16)
    v = mx.random.normal((1, Hkv, cap, D)).astype(F16)
    if cap > P:                                  # what lies past P is never read
        k[..., P:, :] = mx.nan
        v[..., P:, :] = mx.nan
    mx.eval(q, k, v)
    return q, k, v


def _errs(q, k, v, P, scale, **kw):
    o, lse = cascade_mma.prefix_attention(q, k, v, P, scale, **kw)
    ro, rl = cascade_mma.reference(q, k, v, P, scale)
    mx.eval(o, lse, ro, rl)
    assert mx.all(mx.isfinite(o)).item() and mx.all(mx.isfinite(lse)).item()
    return mx.abs(o - ro).max().item(), rms_rel(o, ro), mx.abs(lse - rl).max().item()


@gpu_only
@pytest.mark.parametrize("P", [4096, 8192 + 37, 32768])
@pytest.mark.parametrize("B", [3, 8, 16])
@pytest.mark.parametrize("qstd", [1.0, 4.0])
def test_gpu_kernel_vs_reference_9b(gpu, P, B, qstd):
    """9B layout (16 q / 4 kv heads, D 256): the kernel's fp32 (o_p, lse_p)
    against the fp32 mx reference, NaN past P in the buffers."""
    q, k, v = _prefix_data(B, 16, 4, 256, P, P + 200, qstd, seed=P + B)
    e_max, e_rms, e_lse = _errs(q, k, v, P, 1 / 16)
    print(f"\nP={P} B={B} qstd={qstd}: max {e_max:.2e} rms {e_rms:.2e} lse {e_lse:.2e}", end="")
    assert e_rms <= 2e-5 and e_max <= 1e-4 and e_lse <= 1e-4, (e_max, e_rms, e_lse)


@gpu_only
@pytest.mark.parametrize("Hq,Hkv,D", [(4, 2, 64), (16, 4, 256), (8, 2, 128)])
@pytest.mark.parametrize("P,chunk", [(1, None), (5, None), (37, 16), (100, 32), (1000, 256),
                                     (4097, 512), (300, None)])
@pytest.mark.parametrize("B", [1, 3, 5, 16])
def test_gpu_kernel_partial_blocks(gpu, Hq, Hkv, D, P, chunk, B):
    """Short / odd prefixes, partial key blocks and chunks, partial query
    blocks (M0 = B * rep not a multiple of BQ), other layouts."""
    q, k, v = _prefix_data(B, Hq, Hkv, D, P, P + 3, seed=P * 7 + B)
    e_max, e_rms, e_lse = _errs(q, k, v, P, D ** -0.5, chunk=chunk)
    assert e_rms <= 2e-5 and e_max <= 1e-4 and e_lse <= 1e-4, (e_max, e_rms, e_lse)


@gpu_only
@pytest.mark.parametrize("P,S", [(4096, 64), (8192 + 37, 300), (32768, 128)])
@pytest.mark.parametrize("B", [3, 16])
def test_gpu_grouped_attention_9b(gpu, P, S, B):
    """The whole grouped attention in kernel mode (kernel prefix + fp32 suffix
    stack + merge), 9B layout, ragged suffixes (one empty), a decode step
    inside: rms rel vs per-row fp32 SDPA <= 2.3e-4 (B.7's bound) and within
    5 % (+ 1e-6) of today's fp32 cascade on the same data."""
    lens = [S - (b % 8) * (S // 8) for b in range(B)]
    lens[1] = 0
    d = Rows(P, lens, 4, 256, seed=P + B)
    g, g32 = d.group(step=256), d.group(kernel=False, step=256, chunk=1024)
    assert g.aliased
    i = d.layers[0]
    caches = [r[i] for r in d.rows]
    for qstd in (1.0, 4.0):
        q = queries(B, 16, 256, qstd)
        out, out32 = cascade.attention(q, g, i, 1 / 16), cascade.attention(q, g32, i, 1 / 16)
        ref = per_row(q, caches, 1 / 16, fp32=True)
        e, e32 = rms_rel(out, ref), rms_rel(out32, ref)
        print(f"\nP={P} S={S} B={B} qstd={qstd}: kernel {e:.3e} fp32 cascade {e32:.3e}", end="")
        assert e <= 2.3e-4 and e <= 1.05 * e32 + 1e-6, (e, e32)
    mx.random.seed(5)
    k, v = (mx.random.normal((B, 4, 1, 256)).astype(F16) for _ in range(2))
    for x in (g, g32):
        x.begin_step()
        x.append(i, k, v)
    q = queries(B, 16, 256, 2.0)
    out, out32 = cascade.attention(q, g, i, 1 / 16), cascade.attention(q, g32, i, 1 / 16)
    for x in (g, g32):
        x.end_step()
    for b, c in enumerate(caches):
        c.update_and_fetch(k[b:b + 1], v[b:b + 1])
    ref = per_row(q, caches, 1 / 16, fp32=True)
    assert rms_rel(out, ref) <= 2.3e-4 and rms_rel(out, ref) <= 1.05 * rms_rel(out32, ref) + 1e-6
    back = g.release(2)[i]
    n = caches[2].offset
    assert mx.array_equal(back.keys[..., :n, :], caches[2].keys[..., :n, :]).item()
