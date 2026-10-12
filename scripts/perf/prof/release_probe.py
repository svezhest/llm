#!/usr/bin/env python3
"""Metal check (review cascade_b2b3): a cache given back by PrefixGroup.release
-- capacity P + S + step, not a multiple of 256 -- decodes with the same bits
as the row's own cache. NEEDS THE GPU LEASE (loads its own model copy).

Three rows share an ~8K system prefix (bench_concurrency's --prefix shared),
resumed from one prefix prefill as agree_cascade.py does. A group is built over
them (scheduler parameters) and every row is released at once, so the released
KV must equal the row's own [0, P + S) bit for bit; then both copies decode
greedily, fed the same tokens, for RP_STEPS (300) steps -- across several
capacity growths of both layouts -- (1) each row alone (the B = 1 library call)
and (2) the three rows as one RowsKV batch without a plan (the per-row path a
released row takes at B >= 3). Every step's logits must be bitwise equal.

  cd src/worker && QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 \\
      PYTHONPATH=. .venv/bin/python ../../scripts/perf/prof/release_probe.py
"""
import importlib.util
import os
import time

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


def main():
    from mlx_vlm.models.cache import make_prompt_cache
    from mlx_vlm.prompt_utils import apply_chat_template
    eng = worker.Engine(worker.DEFAULT_MODEL)
    eng.load()
    lm = eng.model.language_model
    tok = eng.processor.tokenizer
    fa = eng.sched.fa_layers
    steps = int(os.environ.get("RP_STEPS", "300"))
    dt = lm.model.embed_tokens(mx.array([[0]])).dtype
    n_layers = len(lm.model.layers)

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
        return cache

    ids = [render(bench.make_messages(i, "rp-8000", 22000, True)) for i in range(3)]
    bnds = [set(worker._bounds(x, eng._im_start)) for x in ids]
    lcp = min(worker._lcp(ids[0], x) for x in ids[1:])
    P = max(p for p in set.intersection(*bnds) if p <= lcp)
    base = prefill(make_prompt_cache(lm), ids[0][:P + 1], 0, bnds[0])
    own = [prefill(worker._copy_cache(base), ids[r], P, bnds[r]) for r in range(3)]
    pg = cascade.PrefixGroup.build(own[0], own, P, fa, chunk=worker.CASCADE_CHUNK,
                                   step=worker.CASCADE_STEP)
    rel = []
    for j in range(3):
        caches = pg.release(j)
        c = worker._copy_cache(own[j])          # same GDN state objects -> fresh lists
        for i, kv in caches.items():
            c[i] = kv
        rel.append(c)
    del pg
    ok = True
    for j in range(3):
        o, r = own[j][fa[0]], rel[j][fa[0]]
        same = all(mx.array_equal(own[j][i].keys[..., :own[j][i].offset, :],
                                  rel[j][i].keys[..., :rel[j][i].offset, :]).item()
                   and mx.array_equal(own[j][i].values[..., :own[j][i].offset, :],
                                      rel[j][i].values[..., :rel[j][i].offset, :]).item() for i in fa)
        log(f"row {j}: P={P} offset {o.offset}, capacity own {o.keys.shape[2]} vs released "
            f"{r.keys.shape[2]} (mod 256 = {r.keys.shape[2] % 256}); KV bitwise equal: {same}")
        ok &= same

    def fork(c):                                # two independent copies of a row's caches
        return worker._copy_cache(c)

    # (1) each row alone, B = 1
    for j in range(3):
        a, b = fork(own[j]), fork(rel[j])
        y, pos = ids[j][-1], len(ids[j]) - 1
        bad = 0
        for s in range(steps):
            pp = mx.array([[pos]], dtype=mx.int32)
            la = lm(mx.array([[y]]), cache=a, position_ids=pp).logits[0, -1]
            lb = lm(mx.array([[y]]), cache=b, position_ids=pp).logits[0, -1]
            eq = mx.array_equal(la, lb).item()
            bad += not eq
            y = mx.argmax(la).item()
            pos += 1
        log(f"B=1 row {j}: {steps} steps, logits bitwise equal in {steps - bad}/{steps}; "
            f"capacity at the end own {a[fa[0]].keys.shape[2]} vs released {b[fa[0]].keys.shape[2]}")
        ok &= bad == 0
    # (2) the three rows as one per-row batch (RowsKV, no plan)
    A, Bc = [fork(c) for c in own], [fork(c) for c in rel]
    ga, gb = worker._gdn_stack(lm, A, dt), worker._gdn_stack(lm, Bc, dt)
    y = [x[-1] for x in ids]
    pos = [len(x) - 1 for x in ids]
    bad = 0
    for s in range(steps):
        pp = mx.array([[p] for p in pos], dtype=mx.int32)
        ca = [ga[i] if i in ga else worker.RowsKV([c[i] for c in A], i, None) for i in range(n_layers)]
        cb = [gb[i] if i in gb else worker.RowsKV([c[i] for c in Bc], i, None) for i in range(n_layers)]
        la = lm(mx.array(y)[:, None], cache=ca, position_ids=pp).logits[:, -1, :]
        lb = lm(mx.array(y)[:, None], cache=cb, position_ids=pp).logits[:, -1, :]
        bad += not mx.array_equal(la, lb).item()
        y = mx.argmax(la, axis=-1).tolist()
        pos = [p + 1 for p in pos]
    log(f"B=3 per-row batch: {steps} steps, logits bitwise equal in {steps - bad}/{steps}")
    ok &= bad == 0
    log("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
