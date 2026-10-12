"""CPU-only check of batching_proto.py on tiny RANDOM qwen3_5 / qwen3_5_moe
language models (no weights, no GPU): a batched decode step (per-row KV +
stacked GDN state, B rows) must give the same logits per row as B=1 decode
of that row alone, including rows joining and leaving mid-way, and with a
quantized KV cache.

Run (no GPU lease needed — the default device is forced to CPU and the
Metal-only fast paths are switched off):
  src/worker/.venv/bin/python <scratchpad>/design/test_batching_proto.py
"""
import sys
import types

import mlx.core as mx

mx.set_default_device(mx.cpu)

import mlx.nn as nn  # noqa: E402
from mlx_vlm.models import rope_utils  # noqa: E402

rope_utils._HAS_METAL = False          # fused RoPE kernels are Metal-only

from mlx_vlm.models.cache import KVCache, QuantizedKVCache  # noqa: E402
from mlx_vlm.models.qwen3_5 import language as q35  # noqa: E402
from mlx_vlm.models.qwen3_5.config import TextConfig  # noqa: E402
from mlx_vlm.models.qwen3_5_moe import language as q35moe  # noqa: E402
from mlx_vlm.models.qwen3_5_moe.config import TextConfig as MoeTextConfig  # noqa: E402

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import batching_proto as bp  # noqa: E402

COMMON = dict(
    hidden_size=128, num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
    head_dim=64, linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=64,
    linear_value_head_dim=64, linear_conv_kernel_dim=4, rms_norm_eps=1e-6, vocab_size=512,
    max_position_embeddings=4096, full_attention_interval=4,
    rope_parameters={"type": "default", "mrope_section": [3, 3, 2], "rope_theta": 100000,
                     "partial_rotary_factor": 0.25},
)
VCFG = types.SimpleNamespace(
    vision_config=types.SimpleNamespace(spatial_merge_size=2),
    image_token_id=500, video_token_id=501, vision_start_token_id=502)


def build(kind, quant_bits=None, dtype=mx.float32):
    mx.random.seed(0)
    if kind == "dense":
        args = TextConfig(model_type="qwen3_5_text", intermediate_size=256, **COMMON)
        lm = q35.LanguageModel(args, VCFG)
    else:
        args = MoeTextConfig(model_type="qwen3_5_moe_text", num_experts=8, num_experts_per_tok=2,
                             shared_expert_intermediate_size=64, moe_intermediate_size=64,
                             **COMMON)
        lm = q35moe.LanguageModel(args, VCFG)
    # random init is ~N(0, 1/sqrt(d)); make A_log sane (library inits it log U(0,16))
    if quant_bits:
        nn.quantize(lm, group_size=64, bits=quant_bits,
                    class_predicate=lambda p, m: hasattr(m, "to_quantized")
                    and getattr(m, "weight", mx.zeros((1, 64))).shape[-1] % 64 == 0)
    lm.apply(lambda a: a.astype(dtype) if a.dtype in (mx.float32, mx.bfloat16, mx.float16) else a)
    mx.eval(lm.parameters())
    return lm


def copy_caches(caches):
    out = []
    for c in caches:
        if hasattr(c, "cache"):
            n = type(c)(size=len(c.cache))
            n.cache = list(c.cache)
        elif isinstance(c.keys, (tuple, list)):
            n = type(c)(group_size=c.group_size, bits=c.bits)
            n.keys = type(c.keys)(k[..., :c.offset, :] for k in c.keys)
            n.values = type(c.values)(v[..., :c.offset, :] for v in c.values)
            n.offset = c.offset
        else:
            n = type(c)()
            n.keys, n.values = c.keys[..., :c.offset, :], c.values[..., :c.offset, :]
            n.offset = c.offset
        out.append(n)
    return out


def prefill(lm, ids, kv_bits):
    lm._position_ids = None          # what get_input_embeddings does per request
    lm._rope_deltas = None
    cache = lm.make_cache()
    lm(mx.array([ids[:-1]]), cache=cache, skip_logits=True)
    if kv_bits:
        for i, c in enumerate(cache):
            if isinstance(c, KVCache):
                cache[i] = c.to_quantized(group_size=64, bits=kv_bits)
    mx.eval([c.state for c in cache])
    return cache


def run(kind, quant_bits=None, kv_bits=None, dtype=mx.float32, steps=7):
    lm = build(kind, quant_bits, dtype)
    bp.install_rows_attention()
    rng_ids = [[int(x) for x in mx.random.randint(0, 400, (n,)).tolist()] for n in (7, 13, 21)]
    pre = [prefill(lm, ids, kv_bits) for ids in rng_ids]

    # reference: each row alone, B=1, explicit positions; greedy
    ref_logits, ref_toks = [], []
    for r, ids in enumerate(rng_ids):
        rc = copy_caches(pre[r])
        y, pos, ls, ts = ids[-1], len(ids) - 1, [], []
        for _ in range(steps):
            lg = lm(mx.array([[y]]), cache=rc, position_ids=mx.array([[pos]], dtype=mx.int32)).logits[:, -1]
            mx.eval(lg)
            ls.append(lg[0])
            ts.append(y)
            y, pos = int(mx.argmax(lg[0]).item()), pos + 1
        ref_logits.append(ls)
        ref_toks.append(ts)

    # explicit positions == library-computed positions at B=1 (row 0)
    rc = copy_caches(pre[0])
    lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int32)
    lm._position_ids = None
    lg_auto = lm(mx.array([[rng_ids[0][-1]]]), cache=rc).logits[:, -1]
    same_pos = bool(mx.array_equal(lg_auto[0], ref_logits[0][0]).item())

    # batched: rows 0,1 for 2 steps; row 2 joins; 2 steps; row 0 leaves; 2 steps
    rows = [0, 1]
    caches = {r: copy_caches(pre[r]) for r in range(3)}
    step_of = {0: 0, 1: 0, 2: 0}
    gdn = bp.stack_gdn(lm, [caches[r] for r in rows], dtype)
    worst = 0.0
    agree = total = 0
    for phase in range(3):
        if phase == 1:
            gdn = bp.concat_gdn(gdn, bp.stack_gdn(lm, [caches[2]], dtype))
            rows = rows + [2]
        if phase == 2:
            keep = [i for i, r in enumerate(rows) if r != 0]
            gdn = bp.take_gdn(gdn, keep)
            rows = [rows[i] for i in keep]
        for _ in range(2):
            toks = mx.array([ref_toks[r][step_of[r]] for r in rows], dtype=mx.int32)
            pos = mx.array([[len(rng_ids[r]) - 1 + step_of[r]] for r in rows], dtype=mx.int32)
            lg = bp.batch_logits(lm, toks, [caches[r] for r in rows], gdn, pos)
            mx.eval(lg)
            for i, r in enumerate(rows):
                ref = ref_logits[r][step_of[r]]
                d = float(mx.abs(lg[i].astype(mx.float32) - ref.astype(mx.float32)).max().item())
                worst = max(worst, d)
                agree += int(mx.argmax(lg[i]).item() == mx.argmax(ref).item())
                total += 1
                step_of[r] += 1
    # back to B=1 for row 1: hand the state back, decode with the library path
    i1 = rows.index(1)
    bp.split_gdn(gdn, i1, caches[1])
    t = step_of[1]
    lg = lm(mx.array([[ref_toks[1][t]]]), cache=caches[1],
            position_ids=mx.array([[len(rng_ids[1]) - 1 + t]], dtype=mx.int32)).logits[:, -1]
    back = float(mx.abs(lg[0].astype(mx.float32) - ref_logits[1][t].astype(mx.float32)).max().item())
    print(f"{kind:5s} wbits={quant_bits} kvbits={kv_bits} dtype={dtype}: explicit==auto positions "
          f"{same_pos}; batched vs B=1 max|dlogit| {worst:.3g}, top-1 {agree}/{total}; "
          f"after split back to B=1 {back:.3g}")
    return same_pos and agree == total and worst < 1e-2 and back < 1e-2


if __name__ == "__main__":
    ok = True
    for kind in ("dense", "moe"):
        for qb, kvb in ((None, None), (8, None), (8, 8), (4, 4)):
            ok &= run(kind, qb, kvb)
        ok &= run(kind, 8, None, dtype=mx.float16)
    print("OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)
