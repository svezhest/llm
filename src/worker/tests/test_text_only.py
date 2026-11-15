"""QWEN_TEXT_ONLY=1 (CPU): a VL checkpoint loaded without its vision tower
(engine._load_model through mlx_vlm.load, on a tiny random qwen3_5 checkpoint
written to disk with the real layout: `vision_tower.*` bf16, an 8-bit
language model), and image requests refused as on a text-only checkpoint."""
import gc
import json
import types

import mlx.core as mx
import mlx.nn as nn
import mlx_vlm.utils as mu
import pytest
from mlx.utils import tree_flatten
from mlx_vlm.models.qwen3_5 import qwen3_5

import worker
import worker_pb2 as pb
from conftest import COMMON

VISION = dict(model_type="qwen3_5", depth=2, hidden_size=256, intermediate_size=512,
              out_hidden_size=COMMON["hidden_size"], num_heads=4, patch_size=16,
              in_channels=3, temporal_patch_size=2, spatial_merge_size=2,
              num_position_embeddings=64)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    """config.json + model.safetensors(+ index) as mlx-vlm's converter writes
    them for the 27B: sanitized key names, a quantized LM, a bf16 tower."""
    d = tmp_path_factory.mktemp("tiny-vl")
    config = {"model_type": "qwen3_5", "image_token_id": 500, "video_token_id": 501,
              "vision_start_token_id": 502,
              "quantization": {"group_size": 64, "bits": 8},
              "text_config": dict(COMMON, model_type="qwen3_5_text", intermediate_size=256),
              "vision_config": VISION}
    arch, _ = mu.get_model_and_args(config)                # as load_model builds it
    mc = mu.update_module_configs(arch.ModelConfig.from_dict(config), arch, config,
                                  ["text", "vision"])
    mx.random.seed(0)
    model = arch.Model(mc)
    assert isinstance(model, qwen3_5.Model)
    nn.quantize(model.language_model, group_size=64, bits=8,
                class_predicate=lambda p, m: hasattr(m, "to_quantized")
                and m.weight.shape[-1] % 64 == 0)
    model.apply(lambda a: a.astype(mx.bfloat16) if a.dtype == mx.float32 else a)
    w = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(d / "model.safetensors"), w)
    (d / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "model.safetensors" for k in w}}))
    (d / "config.json").write_text(json.dumps(config))
    vis = {k: v for k, v in w.items() if k.startswith("vision_tower.")}
    return types.SimpleNamespace(path=str(d), weights=w, n_vis=len(vis),
                                 vis_bytes=sum(v.nbytes for v in vis.values()))


@pytest.fixture
def no_processor(monkeypatch):
    """The processor is not under test (and needs tokenizer files)."""
    proc = types.SimpleNamespace(tokenizer=None)
    monkeypatch.setattr(mu, "load_processor", lambda *a, **k: proc)
    monkeypatch.setattr(mu, "load_image_processor", lambda *a, **k: None)
    return proc


def _logits(model):
    ids = mx.array([[5, 9, 200, 31, 77, 3]])
    emb = model.get_input_embeddings(ids)
    return model.language_model(ids, inputs_embeds=emb.inputs_embeds).logits


def test_load_text_only_skips_tower(ckpt, no_processor):
    """Knob off: the full model, nothing skipped. On: vision_tower = None
    (the text-only checkpoint's state), the skipped tensors = the file's
    `vision_tower.*`, the language model bit-for-bit the same, same logits."""
    full, proc, skipped = worker._load_model(ckpt.path, False)
    assert skipped is None and full.vision_tower is not None and proc is no_processor
    assert len(tree_flatten(full.vision_tower.parameters())) == ckpt.n_vis
    text, proc, skipped = worker._load_model(ckpt.path, True)
    assert proc is no_processor
    assert text.vision_tower is None
    assert skipped == (ckpt.n_vis, ckpt.vis_bytes)
    a = dict(tree_flatten(full.parameters()))
    b = dict(tree_flatten(text.parameters()))
    assert set(a) - set(b) == {k for k in ckpt.weights if k.startswith("vision_tower.")}
    assert set(b) <= set(a)
    for k, v in b.items():
        assert v.dtype == a[k].dtype and mx.array_equal(v, a[k]).item(), k
    assert mx.array_equal(_logits(full), _logits(text)).item()


def test_text_only_tower_never_resident(ckpt, no_processor):
    """The tower's tensors are never read, not even for a moment: the load's
    PEAK is the language model's alone (an eager load then `vision_tower =
    None` frees the same bytes afterwards, so only the peak tells them
    apart; checked by hand: with load(lazy=False) this test fails)."""
    lm_bytes = sum(v.nbytes for k, v in ckpt.weights.items() if not k.startswith("vision_tower."))
    assert ckpt.vis_bytes > 2 * lm_bytes                   # the checks below mean something

    def load(text_only):
        gc.collect()
        mx.clear_cache()
        m0 = mx.get_active_memory()
        mx.reset_peak_memory()
        model, _, _ = worker._load_model(ckpt.path, text_only)
        return model, mx.get_active_memory() - m0, mx.get_peak_memory() - m0

    text, held, peak = load(True)
    assert held < lm_bytes + ckpt.vis_bytes // 4, (held, lm_bytes, ckpt.vis_bytes)
    assert peak < lm_bytes + ckpt.vis_bytes // 4, (peak, lm_bytes, ckpt.vis_bytes)
    full, held, peak = load(False)
    assert held >= lm_bytes + ckpt.vis_bytes * 3 // 4, (held, lm_bytes, ckpt.vis_bytes)
    del text, full


# --- requests --------------------------------------------------------------------------------
def _engine(monkeypatch, config, text_only):
    import mlx_vlm.prompt_utils
    monkeypatch.setattr(mlx_vlm.prompt_utils, "apply_chat_template", lambda *a, **k: "prompt")
    eng = worker.Engine("/tmp/qwen38-27b-mlx-5bit")
    eng.text_only = text_only
    eng.config, eng._template_vars, eng._developer = config, set(), True
    eng.vocab, eng._llg_tok, eng._not_ws = 512, None, None
    eng.grammar_for = lambda p: None
    eng.processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(
        encode=lambda s, add_special_tokens=True: [1, 2, 3]))
    return eng


def _req(images=()):
    return pb.Request(id="r", messages=[pb.Message(role="user", content="hi")],
                      params=pb.Params(), images=list(images))


@pytest.mark.parametrize("config, text_only", [({}, False), ({"vision_config": {}}, True)],
                         ids=["text-only checkpoint", "QWEN_TEXT_ONLY"])
def test_images_refused(monkeypatch, config, text_only):
    """Check and Generate refuse an image request as invalid_request (gateway:
    400) on a text-only checkpoint and on a VL one loaded with the knob; a
    text request renders as before."""
    eng = _engine(monkeypatch, config, text_only)
    err, internal = eng.check(_req(["file:///a.png"]))
    assert "image inputs are not supported" in err and internal is False
    assert ("QWEN_TEXT_ONLY=1" in err) == text_only
    evs = list(eng.generate(_req(["file:///a.png"])))
    assert evs[-1].error.kind == "invalid_request" and evs[-1].error.msg == err
    assert eng.check(_req()) == ("", False)
    assert eng._render(_req(), []) == ("prompt", None)


def test_knob_off_keeps_images(monkeypatch):
    """A VL checkpoint loaded normally (the default): image requests pass the
    render checks (their count is Generate's, not Check's)."""
    eng = _engine(monkeypatch, {"vision_config": {}}, False)
    assert worker.Engine("/tmp/m").text_only is False
    assert eng.check(_req(["file:///a.png"])) == ("", False)
