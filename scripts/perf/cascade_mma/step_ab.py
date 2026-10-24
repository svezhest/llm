#!/usr/bin/env python3
"""Grouped decode step on the real 9B, in one process: today's fp32 cascade vs
the MMA kernel over the fp16 prefix (QWEN_CASCADE_KERNEL), plus the
decomposition behind cascade_gpu.md's open item (32K x 16: 92 ms per step vs
~62 projected). NEEDS THE GPU LEASE (stop every worker first: this loads its
own model copy; nothing else on the GPU).

Rows as in scripts/perf/prof/cascade_step_probe.py: one shared-prefix prefill
up to the deepest common slab end P, per-row resume (a copy, as a prompt-cache
hit) + the row's own suffix, the group built as _Scheduler._group_build does
(kernel mode: the first row's buffers aliased) and decoded greedily on its own
tokens through the scheduler's _plan / _grouped_forward, step-ahead as
_Scheduler._iterate (the next step queued before the previous tokens are read).
Variants, each with its own rows / group / GDN stack, alternating in blocks of
SA_BLOCK steps (2 warm-up steps dropped per block, order reversed every other
round), medians over SA_ROUNDS blocks:

  fp32       the worker's grouped step today
  mma        the same with the kernel group
  stub       the kernel group's step with cascade.attention replaced by a stub
             (returns the queries): everything of the step but the grouped
             attention -> "rest of the step" at this batch and context
  attn_fp32 / attn_mma   only the 8 FA layers' cascade.attention on the live
             fp32 / kernel group (outside a step, random queries, one eval):
             the attention in isolation at the group's real shapes
  control    today's per-row batch (no group), where its KV fits SA_CONTROL_MAX_GB

Decomposition printed per config: in-model attention = step - stub, vs the
isolated attention; a gap is contention with the rest of the step (SLC /
bandwidth / scheduling), not the attention kernel. Also per variant the MLX
peak above the block start and the group's bytes.

  cd src/worker && QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 \\
      PYTHONPATH=. .venv/bin/python ../../scripts/perf/cascade_mma/step_ab.py
Env: SA_PREFIX (8000,32000)  SA_B (8,16)  SA_ROUNDS (4)  SA_BLOCK (48)
     SA_VARIANTS (fp32,mma,stub,attn_fp32,attn_mma,control)  SA_CONTROL_MAX_GB (5)
     PERF_OUT (/tmp/perf): cascade_mma_step_ab.json
"""
import gc
import importlib.util
import json
import os
import statistics as st
import time
import types

import mlx.core as mx

import cascade
import worker

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "bench_concurrency", os.path.join(HERE, "..", "..", "bench_concurrency.py"))
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)
OUT = os.path.join(os.environ.get("PERF_OUT", "/tmp/perf"), "cascade_mma_step_ab.json")
STUB = {"on": False}
_ATTN = cascade.attention


def _attention(queries, group, layer, scale):
    if STUB["on"]:
        return queries
    return _ATTN(queries, group, layer, scale)


cascade.attention = _attention


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Row:
    """A batch row stand-in (pos, group, cache); weak-referenceable, as _GroupRef needs."""

    def __init__(self, pos, group, cache):
        self.pos, self.group, self.cache = pos, group, cache


def main():
    from mlx_vlm.models.cache import make_prompt_cache
    from mlx_vlm.prompt_utils import apply_chat_template
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    eng = worker.Engine(worker.DEFAULT_MODEL)
    eng.load()
    lm = eng.model.language_model
    tok = eng.processor.tokenizer
    fa = eng.sched.fa_layers
    n_layers = len(lm.model.layers)
    dt = lm.model.embed_tokens(mx.array([[0]])).dtype
    batches = [int(b) for b in os.environ.get("SA_B", "8,16").split(",")]
    rounds = int(os.environ.get("SA_ROUNDS", "4"))
    blk = int(os.environ.get("SA_BLOCK", "48"))
    want = os.environ.get("SA_VARIANTS", "fp32,mma,stub,attn_fp32,attn_mma,control").split(",")
    ctl_max = float(os.environ.get("SA_CONTROL_MAX_GB", "5")) * 1e9
    scale = lm.model.layers[fa[0]].self_attn.scale
    S = worker._Scheduler
    results = {"block": blk, "rounds": rounds, "configs": []}
    log(f"loaded; FA layers {fa}; base active {mx.get_active_memory() / 1e9:.2f} GB")

    def render(m):
        return tok.encode(apply_chat_template(eng.processor, eng.config, m,
                                              add_generation_prompt=True, enable_thinking=False))

    def prefill(cache, ids, start, cuts):
        x, pos = mx.array([ids[start:-1]]), start
        lm._position_ids = None
        lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int32) if start else None
        while x.shape[1]:
            n = worker._slab(pos, x.shape[1], cuts)
            lm(x[:, :n], cache=cache, skip_logits=True)
            mx.eval([c.state for c in cache])
            x, pos = x[:, n:], pos + n
            mx.clear_cache()
        return cache

    for target in [int(p) for p in os.environ.get("SA_PREFIX", "8000,32000").split(",")]:
        nb = max(batches)
        tag = f"sa-{target}"
        n0 = len(render(bench.make_messages(0, tag, target * 3, True)))
        n1 = len(render(bench.make_messages(0, tag, 0, True)))
        n_chars = int((target - n1) * target * 3 / max(1, n0 - n1))
        ids = [render(bench.make_messages(i, tag, n_chars, True)) for i in range(nb)]
        bnds = [set(worker._bounds(x, eng._im_start)) for x in ids]
        lcp = min(worker._lcp(ids[0], x) for x in ids[1:])
        P = max(p for p in set.intersection(*bnds) if p <= lcp)
        t = time.time()
        base = prefill(make_prompt_cache(lm), ids[0][:P + 1], 0, bnds[0])
        log(f"target {target}: P {P}, prompts {min(map(len, ids))}..{max(map(len, ids))}; "
            f"prefix prefilled in {time.time() - t:.0f} s")

        def row_cache(r):
            return prefill(worker._copy_cache(base), ids[r], P, bnds[r])

        for B in batches:
            def grouped(kernel):
                rows, pg = [], None
                for r in range(B):
                    rows.append(Row(pos=len(ids[r]) - 1, group=None, cache=row_cache(r)))
                    if r == 2:
                        pg = cascade.PrefixGroup.build(rows[0].cache, [x.cache for x in rows[:3]], P, fa,
                                                       chunk=worker.CASCADE_CHUNK, step=worker.CASCADE_STEP,
                                                       kernel=kernel)
                        done = rows[:3]
                    elif r > 2:
                        pg.add(rows[-1].cache)
                        mx.eval(pg.sk, pg.sv)
                        done = rows[-1:]
                    else:
                        continue
                    for x in done:
                        for i in fa:
                            x.cache[i] = None
                    mx.clear_cache()
                g = worker._PrefixGroup(pg, ids[0][:P], list(rows))
                for x in rows:
                    for i in fa:
                        x.cache[i] = worker._GroupRef(x)
                    x.group = g
                gdn = worker._gdn_stack(lm, [x.cache for x in rows], dt)
                mx.eval([a for b in gdn.values() for a in b.cache])
                fake = types.SimpleNamespace(groups=[g], lm=lm)

                def step(y):
                    pos = mx.array([[x.pos] for x in rows], dtype=mx.int32)
                    plan = S._plan(fake, rows)
                    cache = [gdn[i] if i in gdn else worker.RowsKV([x.cache[i] for x in rows], i, plan)
                             for i in range(n_layers)]
                    out = S._grouped_forward(fake, plan, y[:, None], cache, pos)
                    for x in rows:
                        x.pos += 1
                    return mx.argmax(out.logits[:, -1, :], axis=-1).astype(mx.int32)
                return step, (rows, g, gdn, pg)

            def control():
                caches = [row_cache(r) for r in range(B)]
                gdn = worker._gdn_stack(lm, caches, dt)
                pos0 = [len(x) - 1 for x in ids[:B]]
                st_ = {"k": 0}

                def step(y):
                    pos = mx.array([[p + st_["k"]] for p in pos0], dtype=mx.int32)
                    cache = [gdn[i] if i in gdn else worker.RowsKV([c[i] for c in caches], i, None)
                             for i in range(n_layers)]
                    out = lm(y[:, None], cache=cache, position_ids=pos)
                    st_["k"] += 1
                    return mx.argmax(out.logits[:, -1, :], axis=-1).astype(mx.int32)
                return step, (caches, gdn)

            live = {}
            if "fp32" in want or "attn_fp32" in want:
                live["fp32"] = grouped(False)
            if "mma" in want or "attn_mma" in want:
                live["mma"] = grouped(True)
            if "stub" in want:
                live["stub"] = grouped(True)
            need = B * (max(map(len, ids[:B])) + 2 * rounds * blk) * eng.sched.kv_fp16
            if "control" in want:
                if need <= ctl_max:
                    live["control"] = control()
                else:
                    log(f"P={P} B={B}: control skipped ({need / 1e9:.1f} GB of per-row KV > "
                        f"SA_CONTROL_MAX_GB)")
            gbytes = {k: v[1][3].nbytes / 1e9 for k, v in live.items() if k != "control"}
            mx.clear_cache()
            log(f"P={P} B={B}: built {sorted(live)}; group GB {gbytes}; active "
                f"{mx.get_active_memory() / 1e9:.2f} GB")
            ys = {k: mx.array([x[-1] for x in ids[:B]], dtype=mx.int32) for k in live}
            order = [v for v in ("fp32", "mma", "stub", "attn_fp32", "attn_mma", "control")
                     if v in want and (v in live or v[5:] in live)]
            res = {v: [] for v in order}
            peaks = {v: [] for v in order}
            mx.random.seed(3)
            qa = mx.random.normal((B, lm.args.num_attention_heads, 1, lm.args.head_dim)).astype(dt)
            for r in range(rounds):
                for v in (order if r % 2 == 0 else order[::-1]):
                    if v.startswith("attn_"):
                        pg = live[v[5:]][1][3]

                        def fn():
                            return [cascade.attention(qa, pg, i, scale) for i in fa]
                        for _ in range(2):
                            mx.eval(fn())
                        mx.synchronize()
                        mx.reset_peak_memory()
                        base_mem = mx.get_active_memory()
                        t0 = time.perf_counter()
                        for _ in range(blk - 2):
                            mx.eval(fn())
                        mx.synchronize()
                    else:
                        step = live[v][0]
                        STUB["on"] = v == "stub"
                        y, prev = ys[v], None
                        for s in range(blk):
                            if s == 2:
                                mx.synchronize()
                                if prev is not None:
                                    prev.tolist()
                                mx.reset_peak_memory()
                                base_mem = mx.get_active_memory()
                                t0 = time.perf_counter()
                            nxt = step(y)
                            mx.async_eval(nxt)
                            if prev is not None:
                                prev.tolist()
                            prev, y = nxt, nxt
                        prev.tolist()
                        mx.synchronize()
                        STUB["on"] = False
                        ys[v] = y
                    res[v].append(1000 * (time.perf_counter() - t0) / (blk - 2))
                    peaks[v].append((mx.get_peak_memory() - base_mem) / 2**20)
                    log(f"  round {r} P={P} B={B} {v:9s} {res[v][-1]:7.2f} ms  peak+ {peaks[v][-1]:.0f} MiB")
            med = {v: st.median(x) for v, x in res.items()}
            line = ", ".join(f"{v} {m:.1f}" for v, m in med.items())
            log(f"P={P} B={B}: medians ms/step: {line}")
            dec = {}
            if "stub" in med:
                for k in ("fp32", "mma"):
                    if k in med:
                        dec[k] = {"in_model": med[k] - med["stub"], "isolated": med.get("attn_" + k)}
                        log(f"P={P} B={B}: {k} attention in the model {med[k] - med['stub']:.1f} ms "
                            f"(step {med[k]:.1f} - rest {med['stub']:.1f}) vs isolated "
                            f"{med.get('attn_' + k, float('nan')):.1f} ms")
            if "fp32" in med and "mma" in med:
                log(f"P={P} B={B}: mma step x{med['fp32'] / med['mma']:.3f} vs fp32")
            results["configs"].append(dict(P=P, B=B, ms=res, median=med, peak_mib=peaks,
                                           group_gb=gbytes, decomposition=dec))
            with open(OUT, "w") as f:
                json.dump(results, f, indent=1)
            del live, ys
            gc.collect()           # row <-> group cycles of the stand-in rows
            cascade.clear_traces()
            mx.clear_cache()
        del base
        mx.clear_cache()
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
