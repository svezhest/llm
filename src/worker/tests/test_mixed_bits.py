"""Mixed-precision checkpoints (CPU): a tiny random qwen3_5 checkpoint written
to disk in the layout of Jundot's oQ4e 27B (affine g64, base 4 bits, per-module
overrides keyed by full module path in config.json "quantization"; the real
model: lm_head 6, embed 5, early z / b / a / down 5, b / a 8 or 6 in a few late
layers, the last full-attention layer's k / v 6), loaded through
worker._load_model (mlx-vlm's own per-module resolution) and cast to fp16 as
the engine does. Checked:
  - the fused GDN in-projection by width groups (_install_gdn_fuse): which
    projections share a matmul in each layer, row views of the concatenation;
  - decode logits bit-identical to the unfused model at B = 1 (a row of a
    1-row quantized matmul does not depend on N, on the CPU as in MLX's qmv;
    on the GPU also at B = 2, qmv_wide: gate_oq4e.sh), close at B > 1 (the
    CPU's multi-row path rounds by N);
  - with the kernel knobs (QWEN_MRQMM4 / 5 / 6, the Metal kernel replaced by
    quantized_matmul): which calls take mrqmm in a 5-row decode step and in a
    DFlash verify block of 8 tokens, the 6-bit lm_head and k / v only with
    QWEN_MRQMM6, the narrow b + a group never (N < MIN_N, as on the 27B);
  - a uniform-width model: one group of all four (the fusion as it was).
Metal: tests/test_mrqmm.py (6-bit kernel), scripts/perf/dflash2/gate_oq4e.sh."""
import json
import logging
import types

import mlx.core as mx
import mlx.nn as nn
import mlx_vlm.utils as mu
import pytest
from conftest import COMMON, build_lm
from mlx.utils import tree_flatten
from mlx_vlm.models.qwen3_5 import speculative_verifier
from mlx_vlm.speculative.ops import linear as vlinear
from test_mrqmm_bits import mr, raw  # noqa: F401  (fixtures)
from test_text_only import VISION

import worker

GDN = (0, 1, 2, 4, 5, 6)                  # 8 layers, full attention every 4th: 3 and 7
L = "language_model.model.layers"
# per-module widths (base 4), the oQ4e pattern on 8 layers
OVR = {"language_model.lm_head": 6, "language_model.model.embed_tokens": 5}
for i in (0, 1, 2):                       # "layers 0-18": z / b / a / down 5
    for k in ("in_proj_z", "in_proj_b", "in_proj_a"):
        OVR[f"{L}.{i}.linear_attn.{k}"] = 5
    OVR[f"{L}.{i}.mlp.down_proj"] = 5
OVR.update({f"{L}.4.linear_attn.in_proj_b": 5, f"{L}.4.linear_attn.in_proj_a": 5,
            f"{L}.5.linear_attn.in_proj_b": 8, f"{L}.5.linear_attn.in_proj_a": 8,    # L50 / L62
            f"{L}.6.linear_attn.in_proj_b": 5, f"{L}.6.linear_attn.in_proj_a": 6})   # L52
for i in GDN:
    OVR[f"{L}.{i}.linear_attn.out_proj"] = 5
for i in (3, 7):
    for k in ("q_proj", "o_proj", "k_proj", "v_proj"):
        OVR[f"{L}.{i}.self_attn.{k}"] = 6 if i == 7 and k in ("k_proj", "v_proj") else 5
# expected fusion groups per GDN layer: (members, bits); a projection in none stays alone
GROUPS = {0: [(("in_proj_z", "in_proj_b", "in_proj_a"), 5)],
          4: [(("in_proj_qkv", "in_proj_z"), 4), (("in_proj_b", "in_proj_a"), 5)],
          5: [(("in_proj_qkv", "in_proj_z"), 4), (("in_proj_b", "in_proj_a"), 8)],
          6: [(("in_proj_qkv", "in_proj_z"), 4)]}
GROUPS[1] = GROUPS[2] = GROUPS[0]


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    """config.json + model.safetensors as an MLX mixed-precision converter
    writes them (bf16 scales, base quantization + per-path overrides)."""
    d = tmp_path_factory.mktemp("tiny-oq4e")
    quant = {"group_size": 64, "bits": 4, "mode": "affine"}
    quant.update({k: {"group_size": 64, "bits": b, "mode": "affine"} for k, b in OVR.items()})
    config = {"model_type": "qwen3_5", "image_token_id": 500, "video_token_id": 501,
              "vision_start_token_id": 502, "tie_word_embeddings": False,
              "quantization": quant, "quantization_config": quant,
              "text_config": dict(COMMON, model_type="qwen3_5_text", intermediate_size=256,
                                  tie_word_embeddings=False),
              "vision_config": VISION}
    arch, _ = mu.get_model_and_args(config)
    mc = mu.update_module_configs(arch.ModelConfig.from_dict(config), arch, config, ["text", "vision"])
    mx.random.seed(0)
    model = arch.Model(mc)

    def pred(p, m):
        if not hasattr(m, "to_quantized") or m.weight.shape[-1] % 64:
            return False
        return quant.get("language_model." + p, True)
    nn.quantize(model.language_model, group_size=64, bits=4, class_predicate=pred)
    model.apply(lambda a: a.astype(mx.bfloat16) if a.dtype == mx.float32 else a)
    w = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(d / "model.safetensors"), w)
    (d / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "model.safetensors" for k in w}}))
    (d / "config.json").write_text(json.dumps(config))
    return str(d)


@pytest.fixture
def no_processor(monkeypatch):
    proc = types.SimpleNamespace(tokenizer=None)
    monkeypatch.setattr(mu, "load_processor", lambda *a, **k: proc)
    monkeypatch.setattr(mu, "load_image_processor", lambda *a, **k: None)
    return proc


def load(ckpt):
    """The engine's load of the language model: mlx-vlm (text only), the fp16
    cast, the rows attention. Nothing else installed."""
    model, _, _ = worker._load_model(ckpt, True)
    lm = model.language_model
    lm.apply(lambda a: a.astype(mx.float16) if a.dtype == mx.bfloat16 else a)
    mx.eval(lm.parameters())
    assert worker._install_rows_attention(lm)
    return lm


def decode(lm, B, seed=0):
    ids = mx.random.randint(0, 500, (B, 1), key=mx.random.key(seed))
    return lm(ids, cache=lm.make_cache()).logits.astype(mx.float32)


def groups(lm, i):
    return [(g.names, g.bits) for g in lm.model.layers[i].linear_attn.__dict__.get("_fused_in", ())]


def test_layout_resolves_per_module(ckpt, no_processor):
    """mlx-vlm gives every module the width config.json names for its full
    path, the base 4 bits elsewhere."""
    lm = load(ckpt)
    got = {f"language_model.{n}": m.bits for n, m in lm.named_modules()
           if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding))}
    assert {k: got[k] for k in OVR} == OVR
    assert {b for k, b in got.items() if k not in OVR} == {4}
    assert lm.lm_head.scales.dtype == mx.float16


def test_gdn_fuse_by_width(ckpt, no_processor, caplog):
    """Every GDN layer fuses its same-width projections (row views of one
    concatenation per group); a projection alone at its width keeps its own
    weights; the load log says how."""
    lm = load(ckpt)
    a0 = lm.model.layers[0].linear_attn
    qkv0 = a0.in_proj_qkv.weight
    with caplog.at_level(logging.INFO, logger="qwen.worker"):
        assert worker._install_gdn_fuse(lm) == len(GDN)
    assert {i: groups(lm, i) for i in GDN} == GROUPS
    assert a0.in_proj_qkv.weight is qkv0                         # alone: untouched
    for i in GDN:
        a = lm.model.layers[i].linear_attn
        for g in a.__dict__["_fused_in"]:
            assert g.mr is None and g.split == list(
                mx.cumsum(mx.array([a[n].weight.shape[0] for n in g.names[:-1]])).tolist())
            lo = 0
            for n in g.names:
                hi = lo + a[n].weight.shape[0]
                assert mx.array_equal(a[n].weight, g.w[lo:hi]).item()
                assert mx.array_equal(a[n].scales, g.s[lo:hi]).item()
                lo = hi
            assert lo == g.w.shape[0]
    msg = [r.getMessage() for r in caplog.records if "fused by width" in r.getMessage()]
    assert msg and "3 x [z+b+a 5-bit, qkv alone]" in msg[0] and "1 x [qkv+z 4-bit, b alone, a alone]" in msg[0]


@pytest.mark.parametrize("B", [1, 2, 5])
def test_grouped_fuse_is_bit_identical(ckpt, no_processor, B):
    """A decode step through the grouped fusion == the unfused library calls:
    bit for bit at one row, within the multi-row path's rounding above."""
    ref = load(ckpt)
    want = decode(ref, B, seed=B)
    lm = load(ckpt)
    assert worker._install_gdn_fuse(lm) == len(GDN)
    got = decode(lm, B, seed=B)
    if B == 1:
        assert mx.array_equal(got, want).item()
    else:
        assert mx.allclose(got, want, atol=2e-2, rtol=1e-2).item()


def _kernel_model(ckpt, mr, monkeypatch, six=True):
    for k in ("QWEN_MRQMM4", "QWEN_MRQMM5") + (("QWEN_MRQMM6",) if six else ()):
        monkeypatch.setenv(k, "1")
    monkeypatch.setattr(mr, "MIN_N", 16)          # b + a (N = 8 here, 96 on the 27B) stays narrow
    lm = load(ckpt)
    n = mr.install(lm, warmup=False)
    fused = worker._install_gdn_fuse(lm)
    return lm, n, fused


@pytest.mark.parametrize("six", [True, False])
def test_decode_rows_on_the_kernel(ckpt, no_processor, mr, monkeypatch, six):
    """5 decode rows with QWEN_MRQMM4 / 5 (/ 6): the wide fusion groups take
    the kernel at their width, b + a never; the 6-bit lm_head and last k / v
    only with QWEN_MRQMM6; logits close to the unfused library model."""
    ref = load(ckpt)
    want = decode(ref, 5)
    lm, n, fused = _kernel_model(ckpt, mr, monkeypatch, six)
    assert fused == len(GDN)
    on = {i: [(g.names, g.mr is mr) for g in lm.model.layers[i].linear_attn.__dict__["_fused_in"]] for i in GDN}
    assert on[0] == [(("in_proj_z", "in_proj_b", "in_proj_a"), True)]
    assert on[4] == [(("in_proj_qkv", "in_proj_z"), True), (("in_proj_b", "in_proj_a"), False)]
    assert on[5] == [(("in_proj_qkv", "in_proj_z"), True), (("in_proj_b", "in_proj_a"), False)]
    assert on[6] == [(("in_proj_qkv", "in_proj_z"), True)]
    got = decode(lm, 5)
    calls = [(c[1], c[2]) for c in mr.calls]
    assert {c[0][:2] for c in mr.calls} == {(5, 1)}
    nz = 256 + 4 + 4                          # z + b + a
    nqz = 2 * 128 + 256 + 256                 # qkv + z
    assert calls.count((nz, 5)) == 3 and calls.count((nqz, 4)) == 3
    assert calls.count((512, 6)) == int(six) and calls.count((128, 6)) == 2 * int(six)   # lm_head, k / v
    assert all(N >= 16 for N, _ in calls)
    assert {b for _, b in calls} == ({4, 5, 6} if six else {4, 5})
    assert mx.allclose(got, want, atol=2e-2, rtol=1e-2).item()
    # batch-invariant coverage: b + a groups (L4, L5) are one quantized_matmul each
    miss = worker._invariant_uncovered(lm)
    assert miss["fused GDN in-projection on quantized_matmul"] == 2


def test_verify_block_on_the_kernel(ckpt, no_processor, mr, monkeypatch):
    """A DFlash verify block of 8 tokens with QWEN_VERIFY_MRQMM: the kernel
    groups as in decode (one mrqmm each, qkv alone through its own module),
    b + a through the library; the 6-bit lm_head and the drafter's head (the
    target's) on the kernel."""
    monkeypatch.setattr(vlinear, "_use_target_verify_dense", lambda *a: False)
    v = speculative_verifier.Qwen3_5BatchInvariantForward
    monkeypatch.setattr(worker, "_VERIFY_LINEAR", v._linear)
    monkeypatch.setattr(worker, "_VERIFY_LINEARS", v._linears)
    monkeypatch.setattr(v, "_linear", worker._verify_linear)
    monkeypatch.setattr(v, "_linears", worker._verify_linears)
    monkeypatch.setattr(worker, "VERIFY_MRQMM", "1")
    monkeypatch.setattr(worker, "VERIFY_MR", True)
    monkeypatch.setattr(worker, "VERIFY_MR_HEAD", True)
    monkeypatch.setattr(worker, "DRAFT_BLOCK", 0)
    T = 8
    ids = mx.random.randint(0, 500, (1, T), key=mx.random.key(T))
    ref = load(ckpt)
    want = ref(ids, cache=ref.make_cache(), speculative_verify=True).logits.astype(mx.float32)
    lm, n, fused = _kernel_model(ckpt, mr, monkeypatch)
    draft = types.SimpleNamespace(prefer_requested_block_size=False)
    how = worker._setup_verify_mrqmm(lm, draft)
    assert how.startswith("verify blocks of 3..16 tokens") and f"({len(GDN)} GDN layers fused)" in how
    mr.calls.clear()
    got = lm(ids, cache=lm.make_cache(), speculative_verify=True).logits.astype(mx.float32)
    calls = [(c[1], c[2]) for c in mr.calls]
    assert {c[0][:2] for c in mr.calls} == {(T, 1)}
    assert calls.count((256 + 8, 5)) == 3 and calls.count((2 * 128 + 512, 4)) == 3
    assert calls.count((512, 6)) == 1 and calls.count((128, 6)) == 2
    assert all(N >= 16 for N, _ in calls)
    assert mx.allclose(got, want, atol=2e-2, rtol=1e-2).item()
    mr.calls.clear()
    worker._verify_head(lm.lm_head)(mx.zeros((1, 7, 128), mx.float16))
    assert mr.calls == [((7, 1, 128), 512, 6)]


def test_uniform_width_is_one_group(no_processor):
    """A uniform 4-bit model: one group of all four per GDN layer (the fusion
    as it always was), bit-identical to the unfused calls at one row."""
    ref = build_lm("dense", 4, mx.float16)
    want = decode(ref, 1)
    lm = build_lm("dense", 4, mx.float16)
    assert worker._install_gdn_fuse(lm) == sum(1 for x in lm.model.layers if x.is_linear)
    for x in lm.model.layers:
        if x.is_linear:
            (g,) = x.linear_attn.__dict__["_fused_in"]
            assert g.names == worker._GDN_IN and g.bits == 4 and g.mr is None and len(g.split) == 3
    assert mx.array_equal(decode(lm, 1), want).item()
