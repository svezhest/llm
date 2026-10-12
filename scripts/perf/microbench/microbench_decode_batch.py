#!/usr/bin/env python3
"""Batched-decode ceiling: ms per decode step for B = 1, 2, 4, 8, 16 rows at a
given context, on the REAL model loaded exactly as src/worker/worker.py loads
it (mlx_vlm.load, non-strict for text-only checkpoints, QWEN_FP16 cast, wired
/ cache limits, generation_stream). NEEDS THE GPU LEASE.

Modes (MB_MODES, comma list):
  lib1   B=1 only: the worker's current decode call, lm(y, cache, position_ids)
         on plain caches  -> the baseline (today's tok/s).
  rows   the proposed design: per-row KV caches (each row its own KVCache /
         QuantizedKVCache, attention + cache update per row) + GDN state of all
         rows stacked in one ArraysCache per linear layer; projections, MLP/MoE,
         GDN and lm_head run once for B rows (batching_proto.py).
  lib    mlx-vlm's own batch cache: BatchKVCache (left-padded, ragged Metal SDPA
         kernel when pads differ) + stacked GDN. fp16 KV only (the library's
         BatchQuantizedKVCache ignores left padding at decode, see batching.md).

Each row is a copy of ONE prefilled prompt of MB_CTX tokens (random ids), so
all rows have the same length unless MB_RAGGED=1 (row r is cut by r*37 tokens,
less when the context is short).
Greedy argmax per step, step-ahead pipelined exactly like Engine._run
(queue step k+1, then read token k). MB_SYNC=1 measures the non-pipelined loop
(what a batch with a grammar / bad-words row would run).

WARNING: MB_RAGGED=1 with mode `lib` sends rows of different lengths through
mlx-vlm 0.6.2's ragged decode kernel, which requests a 1024-thread threadgroup
(vlm#1799, see versions.md 3.4) that M1 may reject once rows pass 1024 KV
tokens — possibly a hard Metal abort, not a Python exception. Run that case
last and on its own (MB_MODES=lib MB_RAGGED=1).

Memory: each row costs ctx * kv_bytes_per_token + GDN state; configs whose
estimate exceeds MB_MEM_GB (default 6) are skipped. Run with nothing else on
the GPU. Peak memory is printed per config.

Usage (from the repo root):
  cd src/worker
  QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 \
    uv run python <scratchpad>/design/microbench_decode_batch.py
  # 27B: QWEN_MODEL=../../weights/qwen38-27b-mlx-5bit QWEN_KV_BITS=8 QWEN_WIRED_LIMIT_GB=27 MB_MEM_GB=5
  # 35B: QWEN_MODEL=../../weights/qwen36-35b-a3b-mlx-ud3 QWEN_KV_BITS=4 QWEN_WIRED_LIMIT_GB=27
  # code-path smoke test on the CPU with a tiny random model (no weights, no GPU):
  .venv/bin/python <scratchpad>/design/microbench_decode_batch.py --smoke

Env: MB_BATCHES (1,2,4,8,16)  MB_CTX (1024,8192)  MB_STEPS (48)  MB_WARMUP (6)
     MB_MODES (lib1,rows,lib)  MB_MEM_GB (6)  MB_RAGGED (0)  MB_SYNC (0)
     MB_ROW_SAMPLER (1: argmax per row as the worker would; 0: one batched argmax)
"""
import gc
import json
import os
import sys
import time

SMOKE = "--smoke" in sys.argv
import mlx.core as mx  # noqa: E402

if SMOKE:
    mx.set_default_device(mx.cpu)
    from mlx_vlm.models import rope_utils
    rope_utils._HAS_METAL = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import batching_proto as bp  # noqa: E402

REPO = "/Users/user/Projects/llm"
MODEL = os.environ.get("QWEN_MODEL", os.path.join(REPO, "weights", "ornith15-9b-mlx-8bit"))
FP16 = os.environ.get("QWEN_FP16", "1") == "1"
KV_BITS = int(os.environ.get("QWEN_KV_BITS", "0"))
KV_START = 5000                                     # worker KV_KWARGS["quantized_kv_start"]
PREFILL_STEP = int(os.environ.get("QWEN_PREFILL_STEP", "2048"))
PREFILL_MIN_STEP = int(os.environ.get("QWEN_PREFILL_MIN_STEP", "256"))
PREFILL_BUDGET = int(os.environ.get("QWEN_PREFILL_BUDGET", str(2048 * 32768)))
WIRED_GB = float(os.environ.get("QWEN_WIRED_LIMIT_GB", "0") or "0")
CACHE_LIMIT_GB = float(os.environ.get("QWEN_CACHE_LIMIT_GB", "3"))


def _env_list(name, default):
    return [int(x) for x in os.environ.get(name, default).split(",") if x.strip()]


BATCHES = _env_list("MB_BATCHES", "1,2,4" if SMOKE else "1,2,4,8,16")
CTXS = _env_list("MB_CTX", "48,96" if SMOKE else "1024,8192")
STEPS = int(os.environ.get("MB_STEPS", "4" if SMOKE else "48"))
WARMUP = int(os.environ.get("MB_WARMUP", "2" if SMOKE else "6"))
MODES = [m for m in os.environ.get("MB_MODES", "lib1,rows,lib").split(",") if m]
MEM_GB = float(os.environ.get("MB_MEM_GB", "6"))
RAGGED = os.environ.get("MB_RAGGED", "0") == "1"
SYNC = os.environ.get("MB_SYNC", "0") == "1"
ROW_SAMPLER = os.environ.get("MB_ROW_SAMPLER", "1") == "1"


# ---------------------------------------------------------------- loading (= worker.py Engine.load)
def load_like_worker(path):
    from mlx_vlm import load
    from mlx_vlm.utils import load_config
    config = load_config(path)
    if "vision_config" not in config:
        import mlx.nn as nn
        _orig = nn.Module.load_weights
        nn.Module.load_weights = lambda m, w, strict=True: _orig(m, w, strict=False)
        try:
            model, processor = load(path)
        finally:
            nn.Module.load_weights = _orig
    else:
        model, processor = load(path)
    if FP16:
        lm = model.language_model
        lm.apply(lambda a: a.astype(mx.float16) if a.dtype == mx.bfloat16 else a)
        mx.eval(lm.parameters())
    if WIRED_GB > 0:
        want = int(WIRED_GB * 1024**3)
        cap = int(mx.device_info()["max_recommended_working_set_size"])
        mx.set_wired_limit(min(want, cap))
    mx.set_cache_limit(int(CACHE_LIMIT_GB * 1024**3))
    return model


def tiny_model():
    import test_batching_proto as t
    lm = t.build("dense", 8, mx.float16)
    return type("M", (), {"language_model": lm})()


# ---------------------------------------------------------------- sizes
def kv_bytes_per_token(lm, bits):
    a = lm.args
    n_fa = sum(1 for layer in lm.model.layers if not layer.is_linear)
    elems = n_fa * a.num_key_value_heads * a.head_dim * 2
    return elems * (2 if not bits else bits / 8 + 4 / 64)


def gdn_bytes_per_row(lm):
    total = 0
    for layer in lm.model.layers:
        if layer.is_linear:
            la = layer.linear_attn
            total += la.num_v_heads * la.head_v_dim * la.head_k_dim * 4
            total += (la.conv_kernel_size - 1) * la.conv_dim * 2
    return total


# ---------------------------------------------------------------- prefill (= Engine._run slab loop)
def _step_size(pos):
    return max(PREFILL_MIN_STEP, min(PREFILL_STEP, PREFILL_BUDGET // max(pos, 1)))


def quantize(cache):
    if not KV_BITS:
        return
    for i, c in enumerate(cache):
        if hasattr(c, "to_quantized") and c.offset >= KV_START:
            cache[i] = c.to_quantized(group_size=64, bits=KV_BITS)


def prefill_row(model, ids):
    lm = model.language_model
    cache = lm.make_cache()
    inp = mx.array([ids])
    emb = model.get_input_embeddings(inp, None).inputs_embeds if not SMOKE \
        else lm.model.embed_tokens(inp)
    pos = 0
    while inp.shape[1] > 1:
        n = min(_step_size(pos), inp.shape[1] - 1)
        lm(inp[:, :n], inputs_embeds=emb[:, :n], cache=cache, skip_logits=True)
        quantize(cache)
        mx.eval([c.state for c in cache])
        pos += n
        inp, emb = inp[:, n:], emb[:, n:]
        mx.clear_cache()
    return cache, ids[-1]


def clone_row(cache, cut=0):
    """Independent buffers (rows must not share storage: a slice assignment on
    a shared buffer would copy the whole thing)."""
    out = []
    for c in cache:
        if hasattr(c, "cache"):
            n = type(c)(size=len(c.cache))
            n.cache = [None if a is None else a + 0 for a in c.cache]
        elif isinstance(c.keys, (tuple, list)):
            end = c.offset - cut
            n = type(c)(group_size=c.group_size, bits=c.bits)
            n.keys = tuple(k[..., :end, :] + 0 for k in c.keys)
            n.values = tuple(v[..., :end, :] + 0 for v in c.values)
            n.offset = end
        else:
            end = c.offset - cut
            n = type(c)()
            n.keys, n.values = c.keys[..., :end, :] + 0, c.values[..., :end, :] + 0
            n.offset = end
        out.append(n)
    mx.eval([c.state for c in out])
    return out


# ---------------------------------------------------------------- decode steps
def pick(logits):
    if ROW_SAMPLER and logits.shape[0] > 1:
        return mx.concatenate([
            mx.argmax(logits[i:i + 1] - mx.logsumexp(logits[i:i + 1], axis=-1, keepdims=True), axis=-1)
            for i in range(logits.shape[0])])
    return mx.argmax(logits - mx.logsumexp(logits, axis=-1, keepdims=True), axis=-1)


def make_stepper(mode, lm, rows, first_tok, start_pos):
    """-> step(tokens (B,), k) -> next tokens (B,) (lazy)."""
    B = len(rows)
    dtype = mx.float16 if FP16 else mx.bfloat16
    if mode == "lib1":
        cache = rows[0]

        def step(y, k):
            pos = mx.array([[start_pos[0] + k]], dtype=mx.int32)
            out = lm(y[:, None], cache=cache, position_ids=pos)
            quantize(cache)
            return pick(out.logits[:, -1, :])
        return step
    if mode == "rows":
        gdn = bp.stack_gdn(lm, rows, dtype)

        def step(y, k):
            pos = mx.array([[p + k] for p in start_pos], dtype=mx.int32)
            logits = bp.batch_logits(lm, y, rows, gdn, pos)
            for rc in rows:
                quantize(rc)
            return pick(logits)
        return step
    if mode == "lib":
        from mlx_vlm.models.cache import BatchKVCache
        gdn = bp.stack_gdn(lm, rows, dtype)
        cache = []
        for i, layer in enumerate(lm.model.layers):
            cache.append(gdn[i] if layer.is_linear else BatchKVCache.merge([rc[i] for rc in rows]))
        for rc in rows:                      # merged copies live in `cache` now
            rc.clear()

        def step(y, k):
            pos = mx.array([[p + k] for p in start_pos], dtype=mx.int32)
            out = lm(y[:, None], cache=cache, position_ids=pos)
            return pick(out.logits[:, -1, :])
        return step
    raise ValueError(mode)


def bench(mode, lm, rows, first_tok, start_pos):
    from mlx_vlm.generate.common import generation_stream
    B = len(rows)
    step = make_stepper(mode, lm, rows, first_tok, start_pos)
    y = mx.array([first_tok] * B, dtype=mx.int32)
    mx.reset_peak_memory()
    with mx.stream(generation_stream):
        nxt = step(y, 0)
    mx.async_eval(nxt)
    t0 = None
    for k in range(1, WARMUP + STEPS + 1):
        if k == WARMUP + 1:
            mx.eval(nxt)
            t0 = time.perf_counter()
        cur = nxt
        if SYNC:
            cur.tolist()
            with mx.stream(generation_stream):
                nxt = step(cur, k)
            mx.async_eval(nxt)
        else:
            with mx.stream(generation_stream):
                nxt = step(cur, k)
            mx.async_eval(nxt)
            cur.tolist()                      # the worker's y.item(): token k-1 to the host
    mx.eval(nxt)
    dt = (time.perf_counter() - t0) / STEPS
    return dt, mx.get_peak_memory() / 1e9


def main():
    from mlx_vlm.generate.common import generation_stream, wired_limit
    t = time.time()
    model = tiny_model() if SMOKE else load_like_worker(MODEL)
    lm = model.language_model
    bp.install_rows_attention()
    name = "tiny" if SMOKE else os.path.basename(os.path.normpath(MODEL))
    kvb, gdnb = kv_bytes_per_token(lm, KV_BITS), gdn_bytes_per_row(lm)
    print(f"model {name} loaded in {time.time() - t:.1f}s, active {mx.get_active_memory() / 1e9:.2f} GB; "
          f"KV {kvb / 1024:.1f} KiB/token (kv_bits={KV_BITS}), GDN {gdnb / 1e6:.1f} MB/row; "
          f"modes={MODES} sync={SYNC} ragged={RAGGED} row_sampler={ROW_SAMPLER}")
    vocab = lm.args.vocab_size
    results = []
    import contextlib
    wl = contextlib.nullcontext() if SMOKE else wired_limit(model, [generation_stream])
    with wl:
        for ctx in CTXS:
            lo, hi = (0, 400) if SMOKE else (1000, min(vocab, 150000))
            ids = [int(x) for x in mx.random.randint(lo, hi, (ctx,)).tolist()]
            with mx.stream(generation_stream):
                base, last = prefill_row(model, ids)
            base1 = None
            for B in BATCHES:
                est = B * (ctx * kvb + gdnb) / 1e9
                if est > MEM_GB:
                    print(f"ctx {ctx} B {B}: skipped (~{est:.1f} GB > MB_MEM_GB={MEM_GB})")
                    continue
                for mode in MODES:
                    if mode == "lib1" and B != 1:
                        continue
                    if mode in ("rows", "lib") and B == 1 and "lib1" in MODES:
                        continue                     # B=1 runs the lib1 path in the design
                    if mode == "lib" and KV_BITS and ctx >= KV_START:
                        continue                     # quantized batch cache is broken upstream
                    gap = max(1, min(37, ctx // (2 * B)))
                    cuts = [r * gap if RAGGED else 0 for r in range(B)]
                    rows = [clone_row(base, cut) for cut in cuts]
                    start = [len(ids) - 1 - c for c in cuts]
                    try:
                        dt, peak = bench(mode, lm, rows, last, start)
                    except Exception as e:  # noqa: BLE001
                        print(f"ctx {ctx} B {B} {mode}: FAILED {type(e).__name__}: {e}")
                        continue
                    finally:
                        del rows
                        gc.collect()
                        mx.clear_cache()
                    if B == 1:
                        base1 = dt
                    r = {"model": name, "ctx": ctx, "B": B, "mode": mode, "ms_step": dt * 1e3,
                         "agg_tok_s": B / dt, "row_tok_s": 1 / dt,
                         "speedup": (base1 / dt * B) if base1 else None, "peak_gb": peak,
                         "kv_bits": KV_BITS, "sync": SYNC, "ragged": RAGGED}
                    results.append(r)
                    print(f"ctx {ctx:6d} B {B:2d} {mode:4s}: {dt * 1e3:7.1f} ms/step  "
                          f"{B / dt:7.1f} tok/s aggregate  {1 / dt:6.1f} tok/s per row  "
                          f"x{(base1 / dt * B) if base1 else float('nan'):.2f} vs B=1  peak {peak:.2f} GB")
            del base
            gc.collect()
            mx.clear_cache()
    if SMOKE:
        print("smoke run: code paths OK (CPU timings are meaningless), nothing written")
        return
    out = os.path.join(HERE, "..", "results", f"microbench_decode_{name}_{int(time.time())}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    main()
