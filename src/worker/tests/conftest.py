"""CPU-only fixtures: tiny RANDOM qwen3_5 / qwen3_5_moe language models (no
weights, no GPU). The default device is the CPU before anything builds a
stream, and the Metal-only RoPE kernels are off."""
import os
import sys
import types

import mlx.core as mx

mx.set_default_device(mx.cpu)

import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from mlx_vlm.models import rope_utils  # noqa: E402

rope_utils._HAS_METAL = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import worker  # noqa: E402
import worker_pb2 as pb  # noqa: E402
from mlx_vlm.models.base import InputEmbeddingsFeatures  # noqa: E402
from mlx_vlm.models.qwen3_5 import language as q35  # noqa: E402
from mlx_vlm.models.qwen3_5.config import TextConfig  # noqa: E402
from mlx_vlm.models.qwen3_5_moe import language as q35moe  # noqa: E402
from mlx_vlm.models.qwen3_5_moe.config import TextConfig as MoeTextConfig  # noqa: E402

V = 512
EOS = 3
IM_START = 7
COMMON = dict(
    hidden_size=128, num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
    head_dim=64, linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=64,
    linear_value_head_dim=64, linear_conv_kernel_dim=4, rms_norm_eps=1e-6, vocab_size=V,
    max_position_embeddings=4096, full_attention_interval=4,
    rope_parameters={"type": "default", "mrope_section": [3, 3, 2], "rope_theta": 100000,
                     "partial_rotary_factor": 0.25},
)
VCFG = types.SimpleNamespace(
    vision_config=types.SimpleNamespace(spatial_merge_size=2),
    image_token_id=500, video_token_id=501, vision_start_token_id=502)


def build_lm(kind="dense", wbits=None, dtype=mx.float16):
    mx.random.seed(0)
    if kind == "dense":
        lm = q35.LanguageModel(TextConfig(model_type="qwen3_5_text", intermediate_size=256,
                                          **COMMON), VCFG)
    else:
        lm = q35moe.LanguageModel(MoeTextConfig(
            model_type="qwen3_5_moe_text", num_experts=8, num_experts_per_tok=2,
            shared_expert_intermediate_size=64, moe_intermediate_size=64, **COMMON), VCFG)
    if wbits:
        nn.quantize(lm, group_size=64, bits=wbits,
                    class_predicate=lambda p, m: hasattr(m, "to_quantized")
                    and getattr(m, "weight", mx.zeros((1, 64))).shape[-1] % 64 == 0)
    lm.apply(lambda a: a.astype(dtype) if a.dtype in (mx.float32, mx.bfloat16, mx.float16) else a)
    mx.eval(lm.parameters())
    assert worker._install_rows_attention(lm)
    return lm


IMAGE, VISION_START = VCFG.image_token_id, VCFG.vision_start_token_id


class TinyVLM:
    """What the scheduler uses of mlx-vlm's Model: qwen3_5.Model.get_input_embeddings
    of mlx-vlm 0.7.6 — the mRoPE position_ids / rope_deltas of the ids it is
    given (counted from 0; the image layout with pixel_values, any stand-in)
    are returned, not set on the model, for text too; the image embeddings
    are plain token embeddings (only positions matter here)."""

    def __init__(self, lm):
        self.language_model = lm

    def get_input_embeddings(self, input_ids, pixel_values=None, mask=None, **kwargs):
        lm = self.language_model
        grid = kwargs["image_grid_thw"] if pixel_values is not None else None
        position_ids, rope_deltas = lm.get_rope_index(input_ids, grid, None, mask)
        return InputEmbeddingsFeatures(inputs_embeds=lm.model.embed_tokens(input_ids),
                                       position_ids=position_ids, rope_deltas=rope_deltas)


def vision_ids(seed, image_side=8):
    """Text, <vision_start>, the merged image tokens of a 1 x side x side grid, text."""
    a, b = prompts([10, 30], seed)
    return a + [VISION_START] + [IMAGE] * (image_side * image_side // 4) + b


def vision_inputs(ids, side=8):
    """What Engine._vision_inputs hands the scheduler (numpy), for ids that
    already hold the image layout; the pixels are a stand-in."""
    return {"input_ids": np.array([ids]), "pixel_values": np.zeros((1, 4), np.float32),
            "image_grid_thw": np.array([[1, side, side]])}


class FakeTok:
    """Tokenizer stand-in for _Control (bad words) and ThinkingBudgetCriteria."""

    def __init__(self, table=None):
        self.table = table or {}

    def encode(self, s, add_special_tokens=False):
        return list(self.table.get(s, [ord(c) % V for c in s]))


def engine(lm, eos=(EOS,)):
    return types.SimpleNamespace(model=TinyVLM(lm), _pcache=worker._PrefixCache(),
                                 _im_start=IM_START, draft=None,
                                 stopping=lambda t: t in eos, vocab=V)


# share 1: a prefill runs straight through, so which rows share a step does not
# depend on wall-clock time (with s < 1 decode gets t*(1-s)/s after each slab)
def scheduler(lm, *, max_batch=4, budget=1 << 40, cap=1 << 50, share=1.0, kv_bits=None,
              kv_start=5000, eng=None, prefill_reserve=0, on_fault=None):
    return worker._Scheduler(eng or engine(lm), max_batch=max_batch, budget=budget, cap=cap,
                             prefill_reserve=prefill_reserve, share=share, on_fault=on_fault,
                             kv={"kv_bits": kv_bits, "kv_group_size": 64,
                                 "quantized_kv_start": kv_start})


def request(ids, max_tokens=12, rid="r", tok=None, guide=None, thinking=None, **params):
    """A _Req as Engine.generate builds it; params = pb.Params fields."""
    p = pb.Params(**params)
    sampling = dict(worker.SAMPLING_KWARGS, presence_penalty=0.0)
    for f in ("temperature", "top_p", "top_k", "min_p", "presence_penalty",
              "frequency_penalty", "repetition_penalty", "seed"):
        if p.HasField(f):
            sampling[f] = getattr(p, f)
    ctl = worker._Control(p, [EOS], tok or FakeTok(), V)
    return worker._Req(id=rid, p=p, ids=list(ids), max_tokens=max_tokens, sampling=sampling,
                       ctl=ctl, guide=guide, thinking=thinking)


def run(sched, reqs, arrive=None, max_ticks=5000, on_tick=None):
    """Tick the scheduler on this thread until every request ended. arrive:
    {tick: [reqs]} submitted at that tick (others at tick 0). -> {id: drain()}"""
    arrive = dict(arrive or {})
    later = {id(r) for rs in arrive.values() for r in rs}
    for r in reqs:
        if id(r) not in later:
            sched.submit(r)
    ends, outs = {}, {r.id: ([], [], None) for r in reqs}
    for t in range(max_ticks):
        for r in arrive.pop(t, []):
            sched.submit(r)
        if on_tick is not None:
            on_tick(t, sched)
        if not (sched.rows or sched.task or sched.waiting) and not arrive:
            break
        sched.tick()
        for r in reqs:
            while not r.out.empty():
                ev = r.out.get()
                toks, shown, _ = outs[r.id]
                if ev[0] == "tok":
                    toks.append(ev[1])
                    shown.append(ev[2])
                else:
                    outs[r.id] = (toks, shown, ev)
                    ends[r.id] = t
    else:
        raise AssertionError("scheduler did not finish")
    return outs


# fp32: a batched step then differs from a lone one by ~1e-6 (CPU matmuls
# with M rows vs 1 accumulate differently; in fp16 that is ~3e-3 and greedy
# near-ties flip), so token streams must match exactly.
@pytest.fixture(scope="session")
def dense():
    return build_lm("dense", None, mx.float32)


@pytest.fixture(scope="session")
def moe():
    return build_lm("moe", None, mx.float32)


@pytest.fixture(scope="session")
def dense_fused():
    """The production layout on the CPU: 8-bit weights with fp16 scales /
    biases (as after the QWEN_FP16 cast) and QWEN_GDN_FUSE installed (the GDN
    in-projections are row views of one concatenated weight; a decode step
    runs them as one quantized_matmul). mrqmm stays off (Metal only). CPU
    quantized_matmul computes each row alone, so a batched step matches a lone
    one bit for bit here too."""
    lm = build_lm("dense", 8, mx.float16)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("QWEN_MRQMM", "0")
        assert worker._install_gdn_fuse(lm) == sum(1 for x in lm.model.layers if x.is_linear)
    return lm


def prompts(n_list, seed=1):
    mx.random.seed(seed)
    out = []
    for n in n_list:
        ids = [int(x) for x in mx.random.randint(10, 400, (n,)).tolist()]
        out.append(ids)
    return out


# --- prompt cache: the bitwise "hit == cold" check (tests/test_segments.py) ----
def view(caches):
    """Every array cut to its cache's offset (restore pre-sizes a zeros tail)."""
    out = []
    for c in caches:
        if hasattr(c, "cache"):
            out += [a for a in c.cache if a is not None]
            continue
        for x in (c.keys, c.values):
            out += [y[..., :c.offset, :] for y in (x if isinstance(x, (tuple, list)) else [x])]
    return out


def same(a, b):
    """Two cache lists hold the same arrays, bit for bit."""
    x, y = view(a), view(b)
    return len(x) == len(y) and all(u.shape == v.shape and u.dtype == v.dtype
                                    and mx.array_equal(u, v).item() for u, v in zip(x, y))


def positions(e):
    """Every position best() can return for prompt-cache entry e."""
    tops = [*e.ckpts] + ([len(e.ids)] if e.end is not None else [])
    return sorted(p for p in tops if p >= e.low)


class Cold:
    """The entry a fresh worker stores for ids[:p] (= prompt ids[:p] + [x]),
    restored at its end: the cold prefill's caches at p. Memoized."""

    def __init__(self, lm, **kw):
        self.lm, self.kw, self.memo = lm, kw, {}

    def __call__(self, ids):
        key = tuple(ids)
        if key not in self.memo:
            sched = scheduler(self.lm, max_batch=1, **self.kw)
            run(sched, [request(list(ids) + [1], 1, "cold")])
            (e,) = sched.eng._pcache.entries
            assert e.ids == list(ids)
            self.memo[key] = sched.eng._pcache.restore(e, len(ids))
        return self.memo[key]


def check_cold(pc, ref):
    """Restore of every entry at every resumable position == the cold
    prefill of its ids up to there -> number of positions checked."""
    n = 0
    for e in pc.entries:
        for p in positions(e):
            assert same(pc.restore(e, p), ref(e.ids[:p])), (len(e.ids), p)
            n += 1
    return n
