"""Batch-invariant mode (QWEN_BATCH_INVARIANT, docs/perf/2026-10-24/design/
batch_invariant.md), CPU part: mrqmm's dispatch and K split with the mode on /
off (no kernel runs here), its install / coverage report, cascade and DFlash
off, the decode teacher-forcing hook (_Req.golden), and the scheduler: a row
decoded in a batch of 4 with mixed lengths (the batch shrinking to B = 1 = the
library call) gives the same tokens and log-probabilities, bit for bit, as
alone. On the CPU quantized_matmul computes each row alone, so this checks
everything around the matmuls (attention per row, stacked GDN, norms, the
sampler, logprobs); the kernel itself and the same scheduler run on the GPU
are tests/test_batch_invariant_metal.py."""
import mlx.core as mx
import mlx.nn as nn
import pytest

import mrqmm
import worker
from conftest import EOS, V, engine, prompts, request, scheduler
from test_logprobs import drive, gen_req


@pytest.fixture
def inv(monkeypatch):
    """The mode on (worker + mrqmm), restored after."""
    monkeypatch.setattr(worker, "BATCH_INVARIANT", True)
    mrqmm.set_invariant(True)
    yield
    mrqmm.set_invariant(False)


@pytest.fixture
def pcache(monkeypatch):
    monkeypatch.setattr(worker, "PROMPT_CACHE", True)
    monkeypatch.setattr(worker, "PREFILL_STEP", 8)
    monkeypatch.setattr(worker, "PREFILL_MIN_STEP", 4)


# --- mrqmm dispatch ------------------------------------------------------------
# The 9B's K / N (hidden 4096, MLP 12288; q 8192, k/v 1024, o / GDN out 4096,
# GDN qkv 8192 / z 4096 / b, a 32, fused GDN in-projection 12352, lm_head 248320)
K9 = [4096, 12288]
N9 = [32, 1024, 4096, 8192, 12288, 12352, 248320]


@pytest.mark.parametrize("K", K9 + [128, 192, 320])
@pytest.mark.parametrize("N", N9 + [8, 776])
def test_k_split_fixed_per_k(K, N, inv):
    """On: one K split for both row buckets (the sum order of an output
    depends on KS and K only), and KS divides K / 64."""
    a, b = mrqmm._config(K, N, 1), mrqmm._config(K, N, 2)
    assert a[1] == b[1] and (K // 64) % a[1] == 0
    if K in K9:
        assert a[1] == mrqmm.INVARIANT_KS == 8
    # 3..8 rows: the default config is this one (the mode changes the bits
    # only at B <= 2, qmv before, and at 9..16 rows, KS 4 before)
    mrqmm.set_invariant(False)
    assert mrqmm._config(K, N, 1) == a
    mrqmm.set_invariant(True)


def test_k_split_default_unchanged():
    """Off: the measured speed configs, 8-row bucket 2 x 8, 16-row 4 x 4."""
    assert not mrqmm.INVARIANT
    assert mrqmm._config(4096, 12288, 1) == (2, 8)
    assert mrqmm._config(4096, 12288, 2) == (4, 4)
    assert mrqmm._config(12288, 4096, 2) == (4, 4)
    assert mrqmm._config(4096, 1024, 2) == (2, 4)


def x16(rows, K=4096, L=1, dtype=mx.float16):
    return mx.zeros((rows, L, K), dtype=dtype)


@pytest.mark.parametrize("rows", range(1, 18))
@pytest.mark.parametrize("n", [32, 512, 4096])
def test_use(rows, n, monkeypatch):
    """Which calls take the kernel: off = 3..16 decode rows of a layer with
    N >= MIN_N; on = 1..16 rows, any N. Prefill-shaped (1, T, K) and bf16
    inputs never (both modes)."""
    for on in (False, True):
        monkeypatch.setattr(mrqmm, "INVARIANT", on)
        want = (1 <= rows <= 16) if on else (3 <= rows <= 16 and n >= mrqmm.MIN_N)
        assert mrqmm._use(x16(rows), n) == want, (on, rows, n)
        # the fused GDN in-projection (n vetted at install: None)
        assert mrqmm._use(x16(rows)) == ((1 <= rows <= 16) if on else (3 <= rows <= 16))
        assert not mrqmm._use(x16(1, L=rows + 1), n)
        assert not mrqmm._use(x16(rows, dtype=mx.bfloat16), n)


def lin(K, N, bits=8, dtype=mx.float16):
    m = nn.QuantizedLinear(K, N, bias=False, group_size=64, bits=bits)
    m.scales, m.biases = m.scales.astype(dtype), m.biases.astype(dtype)
    return m


class Holder(nn.Module):
    def __init__(self, **mods):
        super().__init__()
        for k, v in mods.items():
            setattr(self, k, v)


def test_install_and_coverage(monkeypatch):
    """On, install swaps every 8-bit fp16 module whatever its N (b/a N=32, the
    35B router N=256); uncovered() names what stays outside: other bits,
    N % 8, bf16 scales, unquantized, MoE experts, a tied head. Off, the
    narrow layers stay on quantized_matmul (MIN_N), and a module swapped in
    the mode keeps the speed rules once it is off (the N check is per call)."""
    from mlx_vlm.models.switch_layers import QuantizedSwitchLinear
    monkeypatch.setattr(mrqmm, "enabled", lambda: True)
    mk = lambda: Holder(q=lin(256, 1024), ba=lin(256, 32), router=lin(256, 256),  # noqa: E731
                        five=lin(256, 1024, bits=5), g1=lin(256, 4),
                        bf=lin(256, 1024, dtype=mx.bfloat16), dense=nn.Linear(256, 64),
                        experts=QuantizedSwitchLinear(256, 64, 4, bias=False, bits=4))
    off = mk()
    assert mrqmm.install(off, warmup=False) == 1
    assert type(off.q) is mrqmm.MRQuantizedLinear and type(off.ba) is nn.QuantizedLinear
    mrqmm.set_invariant(True)
    try:
        on = mk()
        assert mrqmm.install(on, warmup=False) == 3
        assert all(type(getattr(on, k)) is mrqmm.MRQuantizedLinear for k in ("q", "ba", "router"))
        miss = mrqmm.uncovered(on)
        assert miss == {"QuantizedLinear 5-bit g64": 1, "QuantizedLinear 8-bit g64 not fp16": 1,
                        "QuantizedLinear 8-bit g64 N % 8": 1,
                        "Linear (unquantized)": 1, "QuantizedSwitchLinear (MoE experts)": 1}, miss
    finally:
        mrqmm.set_invariant(False)
    assert not mrqmm._use(x16(4, K=256), on.ba.weight.shape[0])   # off: N=32 < MIN_N
    assert mrqmm._use(x16(4, K=256), on.q.weight.shape[0])
    tied = Holder(q=lin(256, 1024))
    tied.args = type("A", (), {"tie_word_embeddings": True})()
    assert mrqmm.uncovered(tied) == {"QuantizedLinear 8-bit g64": 1,
                                     "tied lm_head (embed_tokens.as_linear)": 1}


def test_uncovered_9b_layout(dense_fused, monkeypatch):
    """The production layout (8-bit fp16 everywhere, GDN fused): every module
    on the kernel; the fused GDN in-projections count as their one call
    (here fused without mrqmm, as on the CPU: one entry per layer; with it,
    nothing: tests/test_batch_invariant_metal.py). The 9B is this layout (all
    decode linears 8-bit g64, lm_head untied; checked from the checkpoint's
    tensors)."""
    monkeypatch.setattr(mrqmm, "enabled", lambda: True)
    mrqmm.set_invariant(True)
    try:
        assert mrqmm.install(dense_fused, warmup=False) > 0
        n = sum(1 for x in dense_fused.model.layers if x.is_linear)
        assert worker._invariant_uncovered(dense_fused) == {
            "fused GDN in-projection on quantized_matmul": n}
    finally:
        mrqmm.uninstall(dense_fused)
        mrqmm.set_invariant(False)


# --- scheduler switches -----------------------------------------------------------
def test_cascade_off(dense, pcache, monkeypatch):
    s = scheduler(dense)
    assert s.cascade                                  # the default would group
    monkeypatch.setattr(worker, "BATCH_INVARIANT", True)
    s = scheduler(dense)
    assert not s.cascade and "QWEN_BATCH_INVARIANT=1" in s.cascade_state()
    s.set_cascade(True)
    assert not s.cascade


def test_no_dflash(dense, monkeypatch):
    eng = engine(dense)
    eng.draft = object()
    s = scheduler(dense, eng=eng)
    req = gen_req(prompts([9], 3)[0], 4, "g")
    row = worker._Row(req, 13, 0)
    row.pen = worker._Penalties(req.sampling, req.ids, V)
    assert s._spec_ok(row)                             # nothing running: DFlash
    monkeypatch.setattr(worker, "BATCH_INVARIANT", True)
    assert not s._spec_ok(row)


# --- golden tokens (decode teacher forcing) --------------------------------------------
def test_golden_tokens(dense, pcache):
    """_Req.golden: the k-th generated token is golden[k] (the drawn one is
    dropped), its logprob is that token's under the forced history; past
    the list the draw goes on. Equal to teacher forcing the same tokens
    through the prompt (prefill vs decode kernels: a tolerance)."""
    P = prompts([13], 5)[0]
    free = drive(scheduler(dense), [gen_req(P, 10, "f", logprobs=True)])["f"][0]
    gold = [t for t in prompts([12], 6)[0] if t != EOS][:6]
    assert gold != free[:6]
    g = gen_req(P, 10, "g", logprobs=True, top_logprobs=2)
    g.golden = gold
    toks, lps, end = drive(scheduler(dense), [g])["g"]
    assert toks[:6] == gold and len(toks) == 10 and end == ("done", "length")
    t = request(P + toks + [EOS], 0, "tf", prompt_logprobs=True)
    drive(scheduler(dense), [t])
    first, sc = t.plp
    for i, (lp, top) in enumerate(lps):
        assert abs(sc[len(P) + i - first] - lp) <= 1e-4, i
        assert len(top) == 2


# --- a row in a batch vs the row alone ---------------------------------------------------
def mixed(golden=False):
    """4 rows, prompts 9..31 tokens, 5..14 new tokens: the batch shrinks
    4 -> 3 -> 2 -> 1 (the last row ends on the plain B = 1 call); two rows
    forced with golden tokens (one past its list), one sampled."""
    ps = prompts([31, 9, 22, 14], 71)
    reqs = [gen_req(ps[0], 14, "a", logprobs=True, top_logprobs=3),
            gen_req(ps[1], 5, "b", logprobs=True, top_logprobs=3),
            request(ps[2], 9, "c", temperature=0.8, top_k=40, seed=3, logprobs=True),
            gen_req(ps[3], 12, "d", logprobs=True, top_logprobs=1)]
    if golden:
        reqs[0].golden = [t for t in prompts([14], 72)[0] if t != EOS]
        reqs[3].golden = [t for t in prompts([6], 73)[0] if t != EOS]
    return reqs


def test_batch_row_matches_solo(dense_fused, pcache, inv):
    """The harness of the Metal test (test_batch_invariant_metal.py, ==
    there): with golden tokens the histories are the same, so the tokens are
    and the logprobs agree to fp16 noise. Not bit for bit on the CPU: its
    vectorized and compiled fp16 elementwise kernels round an element by its
    position in the array (the library's compiled sigmoid of the GDN beta
    differs between row 0 and row 1 of a stack), the GPU's do not."""
    both = drive(scheduler(dense_fused, max_batch=4), mixed(True))
    for r in mixed(True):
        solo = drive(scheduler(dense_fused, max_batch=1), [r])[r.id]
        got = both[r.id]
        assert solo[0] == got[0] and solo[2] == got[2], r.id
        if r.golden:
            assert solo[0][:len(r.golden)] == r.golden[:len(solo[0])]
        for (a, ta), (b, tb) in zip(solo[1], got[1]):
            assert abs(a - b) <= 1e-2 and len(ta) == len(tb), r.id
