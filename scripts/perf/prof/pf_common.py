"""Shared helpers for the prefill profiling scripts: load the model the way
worker.py Engine.load does (fp16 cast, fast-path overrides, rows attention,
GDN fuse, cache limit), synthetic caches at a given offset, one worker slab.

Run from src/worker (imports worker.py)."""
import os, sys, time, statistics
sys.path.insert(0, os.getcwd())
import mlx.core as mx
import worker as W


def load(path=None):
    from mlx_vlm import load as vload
    path = path or W.DEFAULT_MODEL
    model, proc = vload(path)
    lm = model.language_model
    if W.FP16:
        lm.apply(lambda a: a.astype(mx.float16) if a.dtype == mx.bfloat16 else a)
        mx.eval(lm.parameters())
    if W._WIRED_GB > 0:
        cap = int(mx.device_info()["max_recommended_working_set_size"])
        mx.set_wired_limit(min(int(W._WIRED_GB * 1024**3), cap))
    mx.set_cache_limit(int(W.CACHE_LIMIT_GB * 1024**3))
    print("fast paths:", W._install_fast_paths(), flush=True)
    W._install_rows_attention(lm)
    W._install_gdn_fuse(lm)
    return model, proc


def synth_cache(lm, offset, seed=0):
    """A prompt cache as if `offset` tokens had been prefilled (random KV,
    random small recurrent states). KV capacity == offset (as after a real
    prefill whose slab ends were multiples of 256)."""
    from mlx_vlm.models.cache import make_prompt_cache
    cache = make_prompt_cache(lm)
    if offset == 0:
        return cache
    mx.random.seed(seed)
    a = lm.args
    for c in cache:
        if hasattr(c, "cache"):
            conv_dim = a.linear_key_head_dim * a.linear_num_key_heads * 2 + \
                a.linear_value_head_dim * a.linear_num_value_heads
            c.cache[0] = (0.1 * mx.random.normal((1, a.linear_conv_kernel_dim - 1, conv_dim))).astype(mx.float16)
            c.cache[1] = 0.01 * mx.random.normal((1, a.linear_num_value_heads, a.linear_value_head_dim,
                                                  a.linear_key_head_dim))
        else:
            shp = (1, a.num_key_value_heads, offset, a.head_dim)
            c.keys = mx.random.normal(shp).astype(mx.float16)
            c.values = mx.random.normal(shp).astype(mx.float16)
            c.offset = offset
        mx.eval([x for x in W._cache_arrays([c])])
    return cache


def fresh(base):
    from mlx_vlm.models.cache import ArraysCache, KVCache
    if all(getattr(c, "keys", 1) is None or (hasattr(c, "cache") and c.cache[0] is None) for c in base):
        return [type(c)(size=len(c.cache)) if hasattr(c, "cache") else type(c)() for c in base]
    return W._copy_cache(base)


def slab(lm, ids, emb, cache, quantize=None):
    """One worker prefill slab (worker.py _Scheduler._prefill body)."""
    lm(ids, inputs_embeds=emb, cache=cache, skip_logits=True)
    if quantize is not None:
        quantize(cache)
    mx.eval([c.state for c in cache])


def med(xs):
    return statistics.median(xs) if xs else float("nan")
