#!/usr/bin/env python3
"""Grouped decode step cost in-process: step-ahead (as _Scheduler._iterate:
the next forward queued before the previous tokens are read) vs synchronous,
and vs today's per-row path (control) where that fits. NEEDS THE GPU LEASE
(stop the :50051 worker: this loads its own model copy).

Rows are built as in agree_cascade.py (one shared-prefix prefill, per-row
resume + suffix prefill, PrefixGroup.build at the deepest common slab end) and
decoded greedily on their own tokens through the scheduler's _plan /
_grouped_forward. Modes alternate in blocks of SP_BLOCK steps (2 warm-up steps
dropped per block); per block: ms/step and MLX peak above the block's start
(a per-step stack copy under step-ahead -- lost donation, review cascade_b1
finding 5 -- shows as a higher peak and a slower step than sync).

  cd src/worker && QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 \\
      PYTHONPATH=. .venv/bin/python ../../scripts/perf/prof/cascade_step_probe.py
Env: SP_PREFIX (8000,32000)  SP_B (16)  SP_BLOCKS (4 per mode)  SP_BLOCK (64)
     SP_CONTROL_MAX_GB (12)
"""
import gc
import importlib.util
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


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Row:
    """A batch row stand-in (pos, group, cache); weak-referenceable, as _GroupRef needs."""

    def __init__(self, pos, group, cache):
        self.pos, self.group, self.cache = pos, group, cache


def main():
    from mlx_vlm.models.cache import make_prompt_cache
    from mlx_vlm.prompt_utils import apply_chat_template
    eng = worker.Engine(worker.DEFAULT_MODEL)
    eng.load()
    lm = eng.model.language_model
    tok = eng.processor.tokenizer
    fa = eng.sched.fa_layers
    n_layers = len(lm.model.layers)
    dt = lm.model.embed_tokens(mx.array([[0]])).dtype
    B = int(os.environ.get("SP_B", "16"))
    nblk = int(os.environ.get("SP_BLOCKS", "4"))
    blk = int(os.environ.get("SP_BLOCK", "64"))
    ctl_max = float(os.environ.get("SP_CONTROL_MAX_GB", "12")) * 1e9
    S = worker._Scheduler

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

    for target in [int(p) for p in os.environ.get("SP_PREFIX", "8000,32000").split(",")]:
        tag = f"sp-{target}"
        n0 = len(render(bench.make_messages(0, tag, target * 3, True)))
        n1 = len(render(bench.make_messages(0, tag, 0, True)))
        n_chars = int((target - n1) * target * 3 / max(1, n0 - n1))
        ids = [render(bench.make_messages(i, tag, n_chars, True)) for i in range(B)]
        bnds = [set(worker._bounds(x, eng._im_start)) for x in ids]
        lcp = min(worker._lcp(ids[0], x) for x in ids[1:])
        P = max(p for p in set.intersection(*bnds) if p <= lcp)
        base = prefill(make_prompt_cache(lm), ids[0][:P + 1], 0, bnds[0])

        def row_cache(r):
            return prefill(worker._copy_cache(base), ids[r], P, bnds[r])

        def grouped():
            rows, pg = [], None
            for r in range(B):
                rows.append(Row(pos=len(ids[r]) - 1, group=None, cache=row_cache(r)))
                if r == 2:
                    pg = cascade.PrefixGroup.build(rows[0].cache, [x.cache for x in rows[:3]], P, fa,
                                                   chunk=worker.CASCADE_CHUNK, step=worker.CASCADE_STEP)
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
            return step, (rows, g, gdn)

        def control():
            caches = [row_cache(r) for r in range(B)]
            gdn = worker._gdn_stack(lm, caches, dt)
            pos0 = [len(x) - 1 for x in ids]
            st_ = {"k": 0}

            def step(y):
                pos = mx.array([[p + st_["k"]] for p in pos0], dtype=mx.int32)
                cache = [gdn[i] if i in gdn else worker.RowsKV([c[i] for c in caches], i, None)
                         for i in range(n_layers)]
                out = lm(y[:, None], cache=cache, position_ids=pos)
                st_["k"] += 1
                return mx.argmax(out.logits[:, -1, :], axis=-1).astype(mx.int32)
            return step, (caches, gdn)

        variants = [("grouped", grouped)]
        need = B * (max(map(len, ids)) + 2 * nblk * blk) * eng.sched.kv_fp16
        if need <= ctl_max:
            variants.append(("control", control))
        else:
            log(f"P={P} B={B}: control skipped ({need / 1e9:.1f} GB of per-row KV)")
        for name, make in variants:
            step, keep = make()
            mx.clear_cache()
            y = mx.array([x[-1] for x in ids], dtype=mx.int32)
            res = {"ahead": [], "sync": []}
            peaks = {"ahead": [], "sync": []}
            for b in range(2 * nblk):
                mode = ("ahead", "sync")[b % 2]
                prev = None
                for s in range(blk):
                    if s == 2:
                        mx.synchronize()
                        if prev is not None:
                            prev.tolist()
                        mx.reset_peak_memory()
                        base_mem = mx.get_active_memory()
                        t0 = time.perf_counter()
                    nxt = step(y)
                    if mode == "ahead":
                        mx.async_eval(nxt)
                        if prev is not None:
                            prev.tolist()          # the previous step's tokens, read after queuing
                    else:
                        mx.eval(nxt)
                        nxt.tolist()
                    prev, y = nxt, nxt
                prev.tolist()
                mx.synchronize()
                dt_ = (time.perf_counter() - t0) / (blk - 2)
                res[mode].append(1000 * dt_)
                peaks[mode].append((mx.get_peak_memory() - base_mem) / 2**20)
            log(f"P={P} B={B} {name:8s}: ahead {st.median(res['ahead']):.1f} ms/step "
                f"{[round(v, 1) for v in res['ahead']]} peak+ {[round(v) for v in peaks['ahead']]} MiB | "
                f"sync {st.median(res['sync']):.1f} {[round(v, 1) for v in res['sync']]} "
                f"peak+ {[round(v) for v in peaks['sync']]} MiB")
            del step, keep, y, prev, nxt
            gc.collect()           # row <-> group cycles of the stand-in rows
            mx.clear_cache()
        del base
        mx.clear_cache()


if __name__ == "__main__":
    main()
