#!/usr/bin/env python3
"""Reply-KV gate 1, teacher-forced: hit through the reply KV vs a cold prefill, in-process
(agree.py style; NEEDS THE GPU LEASE, no other model process). Loads like the worker
(Engine.load: fp16, pfqmm, GDN fuse, mrqmm; its scheduler thread stays idle).

For each conversation (a gate-1 hit-run JSON from reply_kv_drive.py chat) it rebuilds what the
REPLY=1, B=1 worker computed:
  chain  one cache carried across the turns, as the prompt cache does: turn t is resumed at the
         hit run's own cached_tokens (the chain's end = previous prompt + its fed reply, or a
         prefill-slab checkpoint of the chain kept here), the rest of the prompt prefilled in the
         worker's slabs (_slab over _bounds cuts, resumed from that point), then decoded at B=1
         exactly as _Scheduler._step (lm(y, cache, position_ids=[[pos]])) feeding the hit run's
         own generated ids (+ the EOS a stop row's step-ahead fed; a length row fed every
         generated token, the worker log's "+N generated").
  cold   a fresh cache per turn, the whole prompt prefilled in the worker's slabs, then the same
         teacher-forced decode.
Per generated position: top-1 agreement (argmax chain vs argmax cold), KL(cold || chain) on fp32
log-softmax, and the emulation's fidelity (chain argmax == the hit run's token).

  cd src/worker && QWEN_MODEL=/Users/user/Projects/llm/weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 \
      PYTHONPATH=. .venv/bin/python ../../docs/perf/2026-10-17/raw/reply_kv_tf.py HIT.json [HIT2.json ...] --out OUT.json
"""
import json
import sys

import mlx.core as mx

import worker

FLOOR = False


def main():
    args = sys.argv[1:]
    out = args[args.index("--out") + 1] if "--out" in args else None
    global FLOOR
    FLOOR = "--floor" in args
    files = [a for a in args if a.endswith(".json") and a != out]
    eng = worker.Engine(worker.DEFAULT_MODEL)
    eng.load()
    lm = eng.model.language_model
    from mlx_vlm.models.cache import make_prompt_cache
    from mlx_vlm.prompt_utils import apply_chat_template
    tok = eng.processor.tokenizer
    im_start, im_end = eng._im_start, tok.convert_tokens_to_ids("<|im_end|>")

    def render(msgs):
        return tok.encode(apply_chat_template(eng.processor, eng.config, msgs,
                                              add_generation_prompt=True, enable_thinking=False))

    def prefill(cache, ids, start, snaps, extra=()):
        """ids[start:-1] into cache in the worker's slabs (+ `extra` cuts); snapshots at every cut (the chain)."""
        cuts = set(worker._bounds(ids, im_start)) | set(extra)
        if start:
            lm._rope_deltas, lm._position_ids = mx.zeros((1, 1), dtype=mx.int32), None
        else:
            lm._rope_deltas, lm._position_ids = None, None
        pos = start
        while pos < len(ids) - 1:
            n = worker._slab(pos, len(ids) - 1 - pos, cuts)
            lm(mx.array([ids[pos:pos + n]]), cache=cache, skip_logits=True)
            mx.eval([c.state for c in cache])
            pos += n
            if snaps is not None and pos in cuts:
                snaps[pos] = worker._copy_cache(cache)
            mx.clear_cache()
        return pos

    def decode(cache, pos, inputs):
        """B=1 decode steps; -> fp32 log-softmax rows (one per input)."""
        rows = []
        for t in inputs:
            lg = lm(mx.array([[t]]), cache=cache,
                    position_ids=mx.array([[pos]], dtype=mx.int32)).logits[0, -1].astype(mx.float32)
            lp = lg - mx.logsumexp(lg)
            mx.eval(lp)
            rows.append(lp)
            pos += 1
        return rows, pos

    report = []
    for f in files:
        recs = json.load(open(f))
        ids = [render(r["messages"]) for r in recs]
        nxt = render(recs[-1]["messages"] + [{"role": "assistant", "content": recs[-1]["content"]},
                                             {"role": "user", "content": "ok"}])
        chain_ids, chain, snaps = None, None, {}
        seams = []          # where the chain switched between prefill slabs and decode steps
        for t, r in enumerate(recs):
            P = ids[t]
            assert len(P) == r["prompt_tokens"], (f, t, len(P), r["prompt_tokens"])
            c = r["completion_tokens"]
            after = ids[t + 1] if t + 1 < len(recs) else nxt
            assert after[:len(P)] == P
            gen = after[len(P):len(P) + c]
            assert after[len(P) + c] == im_end, (f, t)
            assert gen == tok.encode(r["content"], add_special_tokens=False), (f, t)
            stop = r["finish"] == "stop"
            targets = gen + ([im_end] if stop else [])          # what the model chose at each step
            inputs = [P[-1]] + targets[:-1]                      # compared steps
            # --- cold
            cold = make_prompt_cache(lm)
            pos = prefill(cold, P, 0, None)
            lc, _ = decode(cold, pos, inputs)
            del cold
            # --- chain (hit)
            hit = r["cached_tokens"] or 0
            if t == 0 or hit == 0:
                # turn 1: a miss, or a resume at a prefill checkpoint of an earlier conversation
                # (a cut of this prompt: == cold by the _bounds invariant)
                assert hit == 0 or hit in set(worker._bounds(P, im_start)), (f, t, hit)
                chain, snaps = make_prompt_cache(lm), {}
                pos = prefill(chain, P, 0, snaps)
                mode = "cold-start" if hit == 0 else f"checkpoint {hit} (cold-equivalent)"
            elif hit == len(chain_ids):
                assert P[:hit] == chain_ids
                pos = prefill(chain, P, hit, snaps)
                mode = "reply end"
            else:
                assert hit in snaps and P[:hit] == chain_ids[:hit], (f, t, hit, sorted(snaps))
                chain = worker._copy_cache(snaps[hit])
                snaps = {p: s for p, s in snaps.items() if p <= hit}
                pos = prefill(chain, P, hit, snaps)
                mode = f"checkpoint {hit} (slab end; reply not kept)"
            seams += ([hit] if t and hit else []) + [len(P) - 1]
            lh, pos = decode(chain, pos, inputs)
            # --- floor: cold prefill cut at the chain's seams too (the reply region prefilled as
            # slabs, not decoded): cold vs alt = slab shapes alone, alt vs hit = decode vs prefill
            la = None
            if FLOOR and t:
                alt = make_prompt_cache(lm)
                apos = prefill(alt, P, 0, None, [x for x in seams[:-1] if 0 < x < len(P) - 1])
                la, _ = decode(alt, apos, inputs)
                del alt
            # the worker fed one token more after the last compared step: the EOS of a stop row
            # (step-ahead) / the last generated token of a length row ("+N generated")
            extra = targets[-1]
            _, pos = decode(chain, pos, [extra])
            chain_ids = P + targets
            assert len(chain_ids) == pos
            # --- compare
            S = mx.stack(lc)
            H = mx.stack(lh)
            kl = (mx.exp(S) * (S - H)).sum(axis=-1)
            ac, ah = mx.argmax(S, axis=-1), mx.argmax(H, axis=-1)
            top2 = mx.sort(S, axis=-1)[:, -2:]
            mx.eval(kl, ac, ah, top2)
            kl, ac, ah = kl.tolist(), ac.tolist(), ah.tolist()
            margin = (top2[:, 1] - top2[:, 0]).tolist()
            n = len(targets)
            agree = sum(a == b for a, b in zip(ac, ah))
            fid = sum(a == b for a, b in zip(ah, targets))
            fidc = sum(a == b for a, b in zip(ac, targets))
            if la is not None:
                A = mx.stack(la)
                kl_ca = (mx.exp(S) * (S - A)).sum(axis=-1)
                kl_ah = (mx.exp(A) * (A - H)).sum(axis=-1)
                aa = mx.argmax(A, axis=-1)
                mx.eval(kl_ca, kl_ah, aa)
                floor = {"kl_cold_alt": sum(kl_ca.tolist()) / n, "kl_alt_hit": sum(kl_ah.tolist()) / n,
                         "top1_cold_alt": sum(a == b for a, b in zip(ac, aa.tolist())),
                         "top1_alt_hit": sum(a == b for a, b in zip(aa.tolist(), ah))}
            else:
                floor = None
            flips = [{"step": k, "cold": ac[k], "hit": ah[k], "cold_margin": margin[k], "kl": kl[k],
                      "cold_tok": tok.decode([ac[k]]), "hit_tok": tok.decode([ah[k]])}
                     for k in range(n) if ac[k] != ah[k]]
            row = {"file": f.rsplit("/", 1)[-1], "turn": t + 1, "prompt": len(P), "resume": hit, "mode": mode,
                   "finish": r["finish"], "steps": n, "top1": agree, "kl_mean": sum(kl) / n,
                   "kl_max": max(kl), "kl_p99": sorted(kl)[int(0.99 * (n - 1))],
                   "hit_argmax_eq_hit_run": fid, "cold_argmax_eq_hit_run": fidc, "flips": flips, "floor": floor}
            report.append(row)
            print(f"{row['file']} turn{t + 1}: prompt {len(P)} resume {hit} ({mode}) {r['finish']} "
                  f"steps {n}: top-1 {agree}/{n}, KL mean {row['kl_mean']:.2e} max {row['kl_max']:.2e}; "
                  f"hit argmax == hit run {fid}/{n}, cold argmax == hit run {fidc}/{n}"
                  + (f"; flips {[(x['step'], x['cold_tok'], x['hit_tok'], round(x['cold_margin'], 4)) for x in flips]}"
                     if flips else "")
                  + (f"; floor KL cold||alt {floor['kl_cold_alt']:.2e} ({floor['top1_cold_alt']}/{n}), "
                     f"alt||hit {floor['kl_alt_hit']:.2e} ({floor['top1_alt_hit']}/{n})" if floor else ""), flush=True)
            mx.clear_cache()
    turns = [x for x in report if x["turn"] > 1]          # turn 1: computed cold on both sides
    tot = sum(x["steps"] for x in turns)
    ag = sum(x["top1"] for x in turns)
    klm = sum(x["kl_mean"] * x["steps"] for x in turns) / tot
    print(f"ALL resumed turns: {len(turns)} turns, {tot} positions: top-1 {ag}/{tot} = {100 * ag / tot:.2f}%, "
          f"KL mean {klm:.2e}, max {max(x["kl_max"] for x in turns):.2e}, peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
    if out:
        json.dump({"turns": report, "top1_pct": 100 * ag / tot, "kl_mean": klm, "positions": tot}, open(out, "w"), indent=1)
    import os
    os._exit(0)          # the Engine's scheduler thread never returns


if __name__ == "__main__":
    main()
