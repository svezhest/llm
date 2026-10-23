#!/usr/bin/env python3
"""Cascade attention microbench: the MMA kernel over an fp16 prefix
(QWEN_CASCADE_KERNEL, src/worker/cascade_mma.py) vs today's fp32 cascade vs
per-row attention, synthetic 9B shapes (Hq 16, Hkv 4, D 256, fp16 KV, 8
full-attention layers per step). NEEDS THE GPU LEASE (nothing else on the GPU;
no model is loaded). One process, variants interleaved per round (order
reversed every other round), medians over rounds -- the machine drifts
+-20-30 % over minutes (CLAUDE.md "Measuring performance").

Per config (P, B; ragged suffixes S_j = 150 .. ~500, one decode step = 8
layers of attention, layers independent, one eval per step):
  fp32       today's cascade.attention (fp32 chunked prefix + fp32 suffix stack)
  mma        kernel-mode cascade.attention (cascade_mma prefix + the same suffix merge)
  mma_pre    cascade_mma.prefix_attention alone (the prefix half), default chunk
  mma_pre@C  ... with chunk C (CMB_CHUNKS, the chunk sweep)
  rows       per-row MLX SDPA over each row's own P + S_j keys (today's per-row
             decode kernel, without the update; B x 8 calls)
  rattn      src/worker/rattn.py over the same rows (one launch per <= 8 rows)
  fp32_step / mma_step   the group variants inside the step protocol
             (begin_step + append + attention + end_step: + the stack writes)
The groups hold 8 distinct layers (fp32: 64 KiB/token of prefix; kernel: the
donor's fp16 buffers, aliased); the per-row variants reuse one KVCache per
row for all 8 layers (attention only, no append: no copies; 128 MiB of K+V per
row-layer at 32K, far past the 48 MB SLC, so the DRAM traffic is the real one).

Also per config: the prefix's ideal cost (FLOPs at 8 TFLOPS, bytes at 400
GB/s), achieved TFLOPS of mma_pre, peak memory above the start per variant,
and numerics (rms rel of mma and fp32 vs a per-row fp32 SDPA reference).

  cd src/worker && PYTHONPATH=. .venv/bin/python ../../scripts/perf/cascade_mma/bench_attn.py
Env: CMB_CFG ("8192x8,8192x16,32768x8,32768x16")  CMB_STEPS (20)  CMB_ROUNDS (5)
     CMB_CHUNKS ("256,512,1024,2048")  CMB_VARS (all of the above)
     CMB_ROWS_MAX_GB (6: skip rows / rattn when the rows' caches exceed it)
     PERF_OUT (/tmp/perf): writes cascade_mma_bench.json there
"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.getcwd())
import mlx.core as mx  # noqa: E402
from mlx_vlm.models.base import scaled_dot_product_attention  # noqa: E402
from mlx_vlm.models.cache import KVCache  # noqa: E402

import cascade  # noqa: E402
import cascade_mma  # noqa: E402
import rattn  # noqa: E402

HQ, HKV, D, LAYERS = 16, 4, 256, 8
SCALE = D ** -0.5
STEPS = int(os.environ.get("CMB_STEPS", "20"))
ROUNDS = int(os.environ.get("CMB_ROUNDS", "5"))
CFG = os.environ.get("CMB_CFG", "8192x8,8192x16,32768x8,32768x16")
CHUNKS = [int(c) for c in os.environ.get("CMB_CHUNKS", "256,512,1024,2048").split(",") if c]
VARS = os.environ.get("CMB_VARS", "fp32,mma,mma_pre,rows,rattn,fp32_step,mma_step").split(",")
ROWS_MAX = float(os.environ.get("CMB_ROWS_MAX_GB", "6")) * 1e9
OUT = os.path.join(os.environ.get("PERF_OUT", "/tmp/perf"), "cascade_mma_bench.json")
F16, F32 = mx.float16, mx.float32


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def kvcache(k, v, spare=0):
    c = KVCache()
    c.update_and_fetch(k, v)
    if spare:
        c.prefix_cache_reserve(c.offset + spare)
    mx.eval(c.keys, c.values)
    return c


def build(P, lens):
    """8 layers of rows sharing [0, P). Per layer ONE donor buffer (the prefix
    + max(lens) suffix tokens, as a prompt-cache hit leaves it); every member
    is a KVCache over that same buffer with its own offset P + S_b (the group
    copies only [P, P + S_b) of a member, so no per-row prefix copies exist);
    the groups (fp32, kernel: the donor's buffers aliased) and one full cache
    per row for the per-row variants (layer 0's data, shared by the 8 per-row
    'layers')."""
    layers = [4 * i + 3 for i in range(LAYERS)]
    B, smax = len(lens), max(lens)
    mx.random.seed(P + B)
    members = [{} for _ in lens]
    full = None
    for li, i in enumerate(layers):
        k = (mx.random.normal((1, HKV, P + smax, D)) * 0.5).astype(F16)
        v = (mx.random.normal((1, HKV, P + smax, D)) * 0.5).astype(F16)
        donor = kvcache(k, v)
        for b, S in enumerate(lens):
            c = KVCache()
            c.keys, c.values, c.offset = donor.keys, donor.values, P + S
            members[b][i] = c
        if li == 0:
            full = [kvcache(k[..., :P + S, :], v[..., :P + S, :], 4096) for S in lens]
        del k, v
    src = {i: members[lens.index(smax)][i] for i in layers}
    g32 = cascade.PrefixGroup.build(src, members, P, layers, chunk=cascade.CHUNK, step=cascade.STEP)
    gk = cascade.PrefixGroup.build(src, members, P, layers, step=cascade.STEP, kernel=True)
    del src, members
    mx.eval(g32.kc, gk.kc)
    return layers, g32, gk, full


def step_fn(variant, layers, g32, gk, full, q, kn, vn, chunk=None):
    def attn_only(g):
        return [cascade.attention(q, g, i, SCALE) for i in layers]

    def in_step(g):
        g.begin_step()
        outs = []
        for i in layers:
            g.append(i, kn, vn)
            outs.append(cascade.attention(q, g, i, SCALE))
        g.end_step()
        return outs

    if variant == "fp32":
        return attn_only(g32)
    if variant == "mma":
        return attn_only(gk)
    if variant == "fp32_step":
        return in_step(g32)
    if variant == "mma_step":
        return in_step(gk)
    if variant == "mma_pre":
        return [cascade_mma.prefix_attention(q, gk.kc[i], gk.vc[i], gk.P, SCALE, chunk=chunk)
                for i in layers]
    kv = [(c.keys[..., :c.offset, :], c.values[..., :c.offset, :]) for c in full]
    if variant == "rows":
        return [mx.concatenate([scaled_dot_product_attention(q[b:b + 1], k, v, cache=full[b], scale=SCALE,
                                                             mask=None) for b, (k, v) in enumerate(kv)], axis=0)
                for _ in layers]
    if variant == "rattn":
        return [rattn.attention(q, [c.keys for c in full], [c.values for c in full],
                                [c.offset for c in full], SCALE) for _ in layers]
    raise ValueError(variant)


def bench(fn):
    for _ in range(2):
        mx.eval(fn())
    mx.synchronize()
    mx.reset_peak_memory()
    a0 = mx.get_active_memory()
    ts = []
    for _ in range(STEPS):
        t = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t)
    return 1e3 * statistics.median(ts), (mx.get_peak_memory() - a0) / 2**20


def rms_rel(x, ref):
    d = x.astype(F32) - ref
    return (mx.sqrt(mx.mean(d * d)) / mx.sqrt(mx.mean(ref * ref))).item()


def numerics(layers, g32, gk, full, B):
    """One layer, flat and peaked queries: kernel-mode / fp32 cascade / per-row
    fp16 SDPA vs per-row fp32 SDPA over the same keys."""
    res = {}
    i = layers[0]
    for regime, mul in (("flat", 1.0), ("peaky", 4.0)):
        mx.random.seed(7)
        q = (mx.random.normal((B, HQ, 1, D)) * mul).astype(F16)
        ref = mx.concatenate([mx.fast.scaled_dot_product_attention(
            q[b:b + 1].astype(F32), c.keys[..., :c.offset, :].astype(F32),
            c.values[..., :c.offset, :].astype(F32), scale=SCALE) for b, c in enumerate(full)], axis=0)
        row16 = mx.concatenate([scaled_dot_product_attention(
            q[b:b + 1], c.keys[..., :c.offset, :], c.values[..., :c.offset, :], cache=c, scale=SCALE,
            mask=None) for b, c in enumerate(full)], axis=0)
        for name, o in (("mma", cascade.attention(q, gk, i, SCALE)),
                        ("fp32", cascade.attention(q, g32, i, SCALE)), ("rows_fp16", row16)):
            res[f"{regime}_{name}"] = rms_rel(o, ref)
    return res


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    log(f"mlx {mx.__version__}, {mx.device_info().get('architecture')}, steps {STEPS}, rounds {ROUNDS}, "
        f"chunks {CHUNKS}")
    out = {"steps": STEPS, "rounds": ROUNDS, "configs": {}}
    for spec in CFG.split(","):
        P, B = (int(x) for x in spec.split("x"))
        lens = [150 + (350 * b) // max(1, B - 1) for b in range(B)]
        mx.clear_cache()
        layers, g32, gk, full = build(P, lens)
        rows_gb = sum(c.keys.nbytes + c.values.nbytes for c in full) / 1e9
        flops = LAYERS * 4 * B * HQ * D * P
        pbytes = LAYERS * 2 * HKV * P * D * 2
        log(f"== P={P} B={B}: suffixes {lens[0]}..{lens[-1]}; prefix per step {flops / 1e9:.1f} GFLOP "
            f"({flops / 8e12 * 1e3:.1f} ms at 8 TF), {pbytes / 1e9:.2f} GB fp16 ({pbytes / 400e9 * 1e3:.1f} ms "
            f"at 400 GB/s); groups fp32 {g32.nbytes / 1e9:.2f} GB, kernel {gk.nbytes / 1e9:.2f} GB "
            f"(aliased {gk.aliased}); rows {rows_gb:.2f} GB; chunk {cascade_mma.pick_chunk(P, HKV, -(-B * 4 // 32))}")
        num = numerics(layers, g32, gk, full, B)
        log("   numerics (rms rel vs fp32 per-row): " + ", ".join(f"{k} {v:.3e}" for k, v in num.items()))
        mx.random.seed(1)
        q = mx.random.normal((B, HQ, 1, D)).astype(F16)
        kn = (mx.random.normal((B, HKV, 1, D)) * 0.5).astype(F16)
        vn = (mx.random.normal((B, HKV, 1, D)) * 0.5).astype(F16)
        mx.eval(q, kn, vn)
        vlist = []
        for v in VARS:
            if v in ("rows", "rattn") and rows_gb > ROWS_MAX:
                log(f"   {v} skipped: rows {rows_gb:.1f} GB > CMB_ROWS_MAX_GB")
                continue
            vlist.append((v, None))
            if v == "mma_pre":
                vlist += [(v, c) for c in CHUNKS]
        res = {}
        for r in range(ROUNDS):
            for v, c in (vlist if r % 2 == 0 else vlist[::-1]):
                key = v if c is None else f"{v}@{c}"
                ms, peak = bench(lambda: step_fn(v, layers, g32, gk, full, q, kn, vn, c))
                res.setdefault(key, []).append((ms, peak))
                log(f"   round {r} {key:12s} {ms:7.2f} ms/step  peak +{peak:.0f} MiB")
        summ = {k: dict(ms=statistics.median(x[0] for x in v), ms_all=[round(x[0], 2) for x in v],
                        peak_mib=max(x[1] for x in v)) for k, v in res.items()}
        for k, v in summ.items():
            rel = "".join(f" x{summ[b]['ms'] / v['ms']:.2f} vs {b}" for b in ("fp32", "rows")
                          if b in summ and b != k)
            tf = f" ({flops / v['ms'] / 1e9:.2f} TFLOPS)" if k.startswith("mma_pre") else ""
            log(f"   P={P} B={B} {k:12s} {v['ms']:7.2f} ms/step{tf}{rel}  peak +{v['peak_mib']:.0f} MiB")
        out["configs"][spec] = dict(P=P, B=B, lens=lens, gflop=flops / 1e9, prefix_gb=pbytes / 1e9,
                                    group_gb={"fp32": g32.nbytes / 1e9, "mma": gk.nbytes / 1e9},
                                    rows_gb=rows_gb, numerics=num, timing=summ)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=1)
        del layers, g32, gk, full, q, kn, vn
        cascade.clear_traces()
        mx.clear_cache()
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
