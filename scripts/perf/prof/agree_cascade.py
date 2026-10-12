#!/usr/bin/env python3
"""Cascade exactness gate (plan B.7): teacher-forced agreement of GROUPED batch
rows (cascade attention) vs solo decode, with today's per-row batch path as the
reference class. NEEDS THE GPU LEASE (nothing else on the GPU; stop the :50051
worker first: this loads its own copy of the model).

Prompts: one shared system prompt of ~P tokens (bench_concurrency.py's
`--prefix shared` filler, in the system message) + B distinct user questions.
The shared prefix is prefilled ONCE up to its deepest common slab end Pc (the
worker's `_bounds` cuts: every row's KV [0, Pc) then holds the same bits, the
prompt cache's hit == cold invariant) and each row resumes from a copy of it
(`worker._copy_cache`, as a prompt-cache restore). Three copies per prompt:

  solo     the B=1 library call (as _Scheduler._step at B=1), greedy, 128 steps:
           its tokens are the teacher stream; its log-probs are kept (numpy fp32)
  grouped  one batch of B rows through the scheduler's OWN code path: the group
           is a cascade.PrefixGroup built as _Scheduler._group_build does (source =
           the first row's cache, chunk/step = worker.CASCADE_CHUNK/STEP) and grown
           with PrefixGroup.add as _group_add does; the rows' FA caches are
           worker._GroupRef; each step calls the unbound _Scheduler._plan and
           _Scheduler._grouped_forward on a stand-in `self` (groups, lm) with
           worker.RowsKV(caches, layer, plan) caches and the stacked GDN state
           (worker._gdn_stack) -- i.e. _rows_attention -> _cascade_rows ->
           cascade.attention, exactly the decode step of the worker.
  control  the same batch with no plan: today's per-row RowsKV path (mrqmm etc.).

Every step each row is fed its solo-greedy token. Reports top-1 agreement and
KL mean / p99 / max for grouped-vs-solo, control-vs-solo, grouped-vs-control.
Optionally (AGC_RMS_STEP, default 64; -1 = off) one step's per-FA-layer
attention outputs of the grouped rows vs an fp32 reference (per-row
mx.fast.scaled_dot_product_attention on fp32 q/K/V), next to the per-row fp16
kernel's own error on the same q/K/V.

  cd src/worker && QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 \\
      PYTHONPATH=. .venv/bin/python ../../scripts/perf/prof/agree_cascade.py
Env: AGC_PREFIX (8000,32000 target prompt tokens)  AGC_BATCHES (8,16)
     AGC_STEPS (128)  AGC_CONTROL_MAX_GB (12: skip the control batch when its
     per-row caches would exceed this)  AGC_OUT (json path)  AGC_RMS_STEP (64)
     AGC_SEED (0: another filler tag and question set, questions 16*seed..)
"""
import gc
import importlib.util
import json
import math
import os
import sys
import time
import types

import mlx.core as mx
import numpy as np

import cascade
import worker

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "bench_concurrency", os.path.join(HERE, "..", "..", "bench_concurrency.py"))
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def stats(kls, agree, total):
    k = sorted(kls)
    return {"top1": 100.0 * agree / total, "agree": agree, "n": total,
            "kl_mean": sum(k) / len(k), "kl_p99": k[int(0.99 * (len(k) - 1))], "kl_max": k[-1]}


def fmt(s):
    return (f"top-1 {s['top1']:.2f}% ({s['agree']}/{s['n']}), KL mean {s['kl_mean']:.2e} "
            f"p99 {s['kl_p99']:.2e} max {s['kl_max']:.2e}")


def logsm(x):
    x = x.astype(mx.float32)
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def kl_top1(lp, lq):
    """(B, V) log-probs p (reference), q -> KL(p||q) per row, top-1 equal per row."""
    kl = (mx.exp(lp) * (lp - lq)).sum(axis=-1)
    eq = mx.argmax(lp, axis=-1) == mx.argmax(lq, axis=-1)
    mx.eval(kl, eq)
    return kl.tolist(), eq.tolist()


# ------------------------------------------------------------ per-layer rms

RMS = {"on": False, "rows": []}
_ORIG_CASCADE_ROWS = worker._cascade_rows


def _rms_rel(x, ref):
    x, ref = x.astype(mx.float32), ref.astype(mx.float32)
    return math.sqrt(((x - ref) ** 2).mean().item() / max((ref ** 2).mean().item(), 1e-30))


def _capture_cascade_rows(attn, cache, queries, keys, values):
    out = _ORIG_CASCADE_ROWS(attn, cache, queries, keys, values)
    if not RMS["on"]:
        return out
    layer = cache.layer
    for pg, idx in cache.plan.groups:
        mine = out[idx]                                   # (n, Hq, 1, D) member order
        q = queries[idx]
        Hkv, D, P = pg.Hkv, pg.D, pg.P
        pk = pg.kc[layer].reshape(1, Hkv, pg.Ppad, D)[..., :P, :]
        pv = pg.vc[layer].reshape(1, Hkv, pg.Ppad, D)[..., :P, :]
        e_c, e_r = [], []
        for j, S in enumerate(pg.lens):                   # inside the step: slot S is this token
            k32 = mx.concatenate([pk, pg.sk[layer][j:j + 1, :, :S + 1, :]], axis=2)
            v32 = mx.concatenate([pv, pg.sv[layer][j:j + 1, :, :S + 1, :]], axis=2)
            qj = q[j:j + 1]
            ref = mx.fast.scaled_dot_product_attention(qj.astype(mx.float32), k32, v32,
                                                       scale=attn.scale)
            k16, v16 = k32.astype(pg.kv_dtype), v32.astype(pg.kv_dtype)
            row16 = worker._Q35.scaled_dot_product_attention(qj, k16, v16, cache=None,
                                                             scale=attn.scale, mask=None)
            mx.eval(ref, row16, mine[j:j + 1])
            e_c.append(_rms_rel(mine[j:j + 1], ref))
            e_r.append(_rms_rel(row16, ref))
        RMS["rows"].append({"layer": layer, "cascade": e_c, "per_row_fp16": e_r})
    return out


worker._cascade_rows = _capture_cascade_rows


# ------------------------------------------------------------ main

class Row:
    """A batch row stand-in (pos, group, cache); weak-referenceable, as _GroupRef needs."""

    def __init__(self, pos, group, cache):
        self.pos, self.group, self.cache = pos, group, cache


def main():
    eng = worker.Engine(worker.DEFAULT_MODEL)
    eng.load()
    lm = eng.model.language_model
    from mlx_vlm.prompt_utils import apply_chat_template
    tok = eng.processor.tokenizer
    steps = int(os.environ.get("AGC_STEPS", "128"))
    batches = [int(b) for b in os.environ.get("AGC_BATCHES", "8,16").split(",")]
    targets = [int(p) for p in os.environ.get("AGC_PREFIX", "8000,32000").split(",")]
    ctl_max = float(os.environ.get("AGC_CONTROL_MAX_GB", "12")) * 1e9
    rms_step = int(os.environ.get("AGC_RMS_STEP", "64"))
    out_path = os.environ.get("AGC_OUT")
    dt = lm.model.embed_tokens(mx.array([[0]])).dtype
    n_layers = len(lm.model.layers)
    fa = eng.sched.fa_layers
    nb = max(batches)
    sched_cls = worker._Scheduler
    results = {"seed": int(os.environ.get("AGC_SEED", "0")), "steps": steps, "chunk": worker.CASCADE_CHUNK, "step": worker.CASCADE_STEP,
               "mrqmm": os.environ.get("QWEN_MRQMM", "1"), "runs": []}
    log(f"loaded; FA layers {fa}, base active {mx.get_active_memory() / 1e9:.2f} GB")

    def render(msgs):
        return tok.encode(apply_chat_template(eng.processor, eng.config, msgs,
                                              add_generation_prompt=True, enable_thinking=False))

    def prefill(cache, ids, start, cuts):
        """ids[start:-1] into `cache` (holding [0, start)) with the worker's slab cuts."""
        x, pos = mx.array([ids[start:-1]]), start
        lm._position_ids = None
        lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int32) if start else None
        while x.shape[1]:
            n = worker._slab(pos, x.shape[1], cuts)
            lm(x[:, :n], cache=cache, skip_logits=True)
            mx.eval([c.state for c in cache])
            x, pos = x[:, n:], pos + n
            if worker.CLEAR_CACHE:
                mx.clear_cache()
        return cache

    for target in targets:
        # prompts as the bench builds them: chars per token calibrated on the tokenizer
        seed = int(os.environ.get("AGC_SEED", "0"))      # another filler + question set
        tag = f"agc-{target}" if not seed else f"agc{seed}-{target}"
        qoff = 16 * seed
        probe = target * 3
        n0 = len(render(bench.make_messages(qoff, tag, probe, True)))
        n1 = len(render(bench.make_messages(qoff, tag, 0, True)))
        n_chars = int((target - n1) * probe / max(1, n0 - n1))
        ids = [render(bench.make_messages(qoff + i, tag, n_chars, True)) for i in range(nb)]
        lcp = min(worker._lcp(ids[0], x) for x in ids[1:])
        bnds = [set(worker._bounds(x, eng._im_start)) for x in ids]
        common = sorted(p for p in set.intersection(*bnds) if p <= lcp)
        P = common[-1]
        log(f"prefix target {target}: prompts {min(map(len, ids))}..{max(map(len, ids))} tokens, "
            f"LCP {lcp}, group P {P} (common slab ends {common[-4:]}), "
            f"suffixes {min(len(x) - 1 - P for x in ids)}..{max(len(x) - 1 - P for x in ids)}")
        t = time.time()
        base = prefill(make_cache(lm), ids[0][:P + 1], 0, set(worker._bounds(ids[0], eng._im_start)))
        log(f"shared prefix prefilled in {time.time() - t:.1f} s ({P} tokens)")

        def row_cache(r):
            c = worker._copy_cache(base)
            return prefill(c, ids[r], P, bnds[r])

        # cold-vs-resumed sanity on row 0 (KV and next-token logits bitwise): 8K only
        if target <= 10000:
            cold = prefill(make_cache(lm), ids[0], 0, bnds[0])
            warm = row_cache(0)
            same = all(mx.array_equal(cold[i].keys[..., :cold[i].offset, :],
                                      warm[i].keys[..., :warm[i].offset, :]).item() for i in fa)
            log(f"resumed copy == cold prefill (FA keys, all layers): {same}")
            del cold, warm

        # ---- solo: teacher streams + log-probs
        t = time.time()
        solo_tok, solo_lp = [], []
        for r in range(nb):
            c = row_cache(r)
            y, pos = ids[r][-1], len(ids[r]) - 1
            toks, rows_lp = [], []
            for s in range(steps):
                lg = lm(mx.array([[y]]), cache=c,
                        position_ids=mx.array([[pos]], dtype=mx.int32)).logits[0, -1]
                lp = logsm(lg)
                y = mx.argmax(lp).item()
                rows_lp.append(np.array(lp))
                toks.append(y)
                pos += 1
            solo_tok.append(toks)
            solo_lp.append(np.stack(rows_lp))
            del c, rows_lp
            mx.clear_cache()
        log(f"solo streams: {nb} rows x {steps} steps in {time.time() - t:.1f} s; "
            f"row 0 starts {tok.decode(solo_tok[0][:24])!r}")

        def one_batch(B):
            run = {"target": target, "P": P, "B": B, "prompt_tokens": [len(x) for x in ids[:B]]}
            # ---- grouped batch: the scheduler's code path
            rows = []
            pg = None
            mx.reset_peak_memory()
            for r in range(B):
                c = row_cache(r)
                rows.append(Row(pos=len(ids[r]) - 1, group=None, cache=c))
                if r < 2:
                    continue
                if r == 2:      # >= 3 members: build from the first row's cache (_group_build)
                    for x in rows[1:3]:
                        if not _same_prefix(rows[0].cache, x.cache, fa, P):
                            raise SystemExit("prefix bits differ between rows")
                    pg = cascade.PrefixGroup.build(rows[0].cache, [x.cache for x in rows[:3]], P,
                                                   fa, chunk=worker.CASCADE_CHUNK,
                                                   step=worker.CASCADE_STEP)
                    done = rows[:3]
                else:           # _group_add (lazy there; evaluated here to free the row's KV)
                    if not pg.prefix_equal(c):
                        raise SystemExit("prefix bits differ between rows")
                    pg.add(c)
                    mx.eval(pg.sk, pg.sv)
                    done = rows[r:]
                for x in done:  # the rows' FA buffers go (the scheduler swaps in _GroupRefs)
                    for i in fa:
                        x.cache[i] = None
                mx.clear_cache()
            g = worker._PrefixGroup(pg, ids[0][:P], list(rows))
            for x in rows:
                for i in fa:
                    x.cache[i] = worker._GroupRef(x)
                x.group = g
            gdn_g = worker._gdn_stack(lm, [x.cache for x in rows], dt)
            mx.eval([a for b in gdn_g.values() for a in b.cache])
            mx.clear_cache()
            fake = types.SimpleNamespace(groups=[g], lm=lm)
            log(f"P={P} B={B}: group built: n={pg.n} lens {min(pg.lens)}..{max(pg.lens)} cap {pg.cap} "
                f"c={pg.c} Ppad={pg.Ppad} pmask={'yes' if pg.pmask is not None else 'no'}, "
                f"{pg.nbytes / 1e9:.2f} GB; active {mx.get_active_memory() / 1e9:.2f} GB")

            # ---- control batch: per-row caches, no plan
            ctl_bytes = B * (max(map(len, ids[:B])) + steps) * 2 * eng.sched.kv_fp16 // 2
            ctl = gdn_c = None
            if ctl_bytes <= ctl_max:
                ctl = [row_cache(r) for r in range(B)]
                gdn_c = worker._gdn_stack(lm, ctl, dt)
                mx.eval([a for b in gdn_c.values() for a in b.cache])
                mx.clear_cache()
            else:
                log(f"control batch skipped: {ctl_bytes / 1e9:.1f} GB of per-row KV > "
                    f"AGC_CONTROL_MAX_GB (16 x 32K does not fit per-row in the worker either)")

            acc = {"grouped_vs_solo": ([], 0), "control_vs_solo": ([], 0), "grouped_vs_control": ([], 0)}
            per_q = {k: [[[], 0] for _ in range(4)] for k in acc}
            y = [ids[r][-1] for r in range(B)]
            t_g = t_c = 0.0
            for s in range(steps):
                pos = mx.array([[x.pos] for x in rows], dtype=mx.int32)
                RMS["on"] = s == rms_step
                t0 = time.time()
                plan = sched_cls._plan(fake, rows)
                cache = [gdn_g[i] if i in gdn_g else worker.RowsKV([x.cache[i] for x in rows], i, plan)
                         for i in range(n_layers)]
                out = sched_cls._grouped_forward(fake, plan, mx.array(y, dtype=mx.int32)[:, None],
                                                 cache, pos)
                lg_g = logsm(out.logits[:, -1, :])
                mx.eval(lg_g)
                t_g += time.time() - t0
                RMS["on"] = False
                for x in rows:
                    x.pos += 1
                lg_c = None
                if ctl is not None:
                    t0 = time.time()
                    cc = [gdn_c[i] if i in gdn_c else worker.RowsKV([c[i] for c in ctl])
                          for i in range(n_layers)]
                    lg_c = logsm(lm(mx.array(y, dtype=mx.int32)[:, None], cache=cc,
                                    position_ids=pos).logits[:, -1, :])
                    mx.eval(lg_c)
                    t_c += time.time() - t0
                lp_s = mx.array(np.stack([solo_lp[r][s] for r in range(B)]))
                pairs = [("grouped_vs_solo", lp_s, lg_g)]
                if lg_c is not None:
                    pairs += [("control_vs_solo", lp_s, lg_c), ("grouped_vs_control", lg_c, lg_g)]
                for name, a, b in pairs:
                    kl, eq = kl_top1(a, b)
                    kls, ag = acc[name]
                    acc[name] = (kls + kl, ag + sum(eq))
                    q = per_q[name][min(3, 4 * s // steps)]
                    q[0] += kl
                    q[1] += sum(eq)
                y = [solo_tok[r][s] for r in range(B)]
                del lp_s, lg_g, lg_c, out
            run["peak_gb"] = mx.get_peak_memory() / 1e9
            run["ms_per_step_grouped"] = 1000 * t_g / steps
            run["ms_per_step_control"] = 1000 * t_c / steps if ctl is not None else None
            for name, (kls, ag) in acc.items():
                if kls:
                    run[name] = stats(kls, ag, len(kls))
                    run[name + "_by_quarter"] = [stats(k, a, len(k)) for k, a in per_q[name]]
                    log(f"P={P} B={B:2d} {name:18s}: {fmt(run[name])}")
            if RMS["rows"]:
                lay = []
                for e in RMS["rows"]:
                    c_, r_ = np.array(e["cascade"]), np.array(e["per_row_fp16"])
                    lay.append({"layer": e["layer"], "cascade_mean": float(c_.mean()),
                                "cascade_max": float(c_.max()), "per_row_mean": float(r_.mean()),
                                "per_row_max": float(r_.max()),
                                "rows_cascade_le_per_row": int((c_ <= r_).sum()), "rows": len(c_)})
                run["rms_step"] = rms_step
                run["rms_layers"] = lay
                allc = np.concatenate([e["cascade"] for e in RMS["rows"]])
                allr = np.concatenate([e["per_row_fp16"] for e in RMS["rows"]])
                log(f"P={P} B={B:2d} per-layer attention rms rel vs fp32 at step {rms_step}: cascade "
                    f"mean {allc.mean():.2e} max {allc.max():.2e}; per-row fp16 mean {allr.mean():.2e} "
                    f"max {allr.max():.2e}; cascade <= per-row in {(allc <= allr).sum()}/{len(allc)} "
                    f"(row, layer) pairs")
                RMS["rows"] = []
            log(f"P={P} B={B:2d}: {run['ms_per_step_grouped']:.0f} ms/step grouped (sync, incl. "
                f"KL), control {run['ms_per_step_control'] or 0:.0f}; peak {run['peak_gb']:.2f} GB")
            mx.clear_cache()
            return run

        for B in batches:
            results["runs"].append(one_batch(B))
            cascade.clear_traces()
            gc.collect()           # row <-> group cycles of the stand-in rows
            if out_path:
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=1)
            mx.clear_cache()
        del base, solo_lp
        mx.clear_cache()
    if out_path:
        with open(out_path, "w") as f:
            json.dump(results, f, indent=1)


def _same_prefix(a, b, fa, P):
    """Rows a and b hold the same KV [0, P) bit for bit (_Scheduler._same_prefix)."""
    return all(mx.array_equal(a[i].keys[..., :P, :], b[i].keys[..., :P, :]).item()
               and mx.array_equal(a[i].values[..., :P, :], b[i].values[..., :P, :]).item()
               for i in fa)


def make_cache(lm):
    from mlx_vlm.models.cache import make_prompt_cache
    return make_prompt_cache(lm)


if __name__ == "__main__":
    main()
