#!/usr/bin/env python3
"""Teacher-forced agreement of batched vs solo decode on the real model.
NEEDS THE GPU LEASE. Loads like the worker (Engine.load; its scheduler thread
stays idle). For B in AG_BATCHES: B prompts, each prefilled twice (a solo copy
and a batch copy); each step every row is fed its own solo-greedy token, the
solo copies decode one by one (the B=1 library call), the batch copies in one
forward (RowsKV + stacked recurrent state, as _Scheduler._step). Reports per
position top-1 agreement and KL(solo || batched).

  cd src/worker && QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 \
      PYTHONPATH=. uv run python ../../scripts/perf/prof/agree.py
Env: AG_BATCHES (2,4,8,16)  AG_STEPS (128)
"""
import functools
import os

import mlx.core as mx

import worker

TASKS = [l.split('"')[1] for l in open(os.path.join(os.path.dirname(__file__), "wtest.py"))
         if l.startswith('    "') and l.rstrip().endswith('",')]


def main():
    eng = worker.Engine(worker.DEFAULT_MODEL)
    eng.load()
    lm = eng.model.language_model
    from mlx_vlm.generate.common import maybe_quantize_kv_cache
    from mlx_vlm.models.cache import make_prompt_cache
    from mlx_vlm.prompt_utils import apply_chat_template
    quant = functools.partial(maybe_quantize_kv_cache, **{
        "quantized_kv_start": worker.KV_KWARGS["quantized_kv_start"],
        "kv_group_size": 64, "kv_bits": worker.KV_KWARGS["kv_bits"]})
    tok = eng.processor.tokenizer
    steps = int(os.environ.get("AG_STEPS", "128"))
    dt = lm.model.embed_tokens(mx.array([[0]])).dtype

    def prefill(ids):
        cache, x, pos = make_prompt_cache(lm), mx.array([ids[:-1]]), 0
        lm._position_ids, lm._rope_deltas = None, None
        while x.shape[1]:
            n = worker._slab(pos, x.shape[1], set())
            lm(x[:, :n], cache=cache, skip_logits=True)
            quant(cache)
            mx.eval([c.state for c in cache])
            x, pos = x[:, n:], pos + n
        return cache

    for B in [int(b) for b in os.environ.get("AG_BATCHES", "2,4,8,16").split(",")]:
        ids = [tok.encode(apply_chat_template(eng.processor, eng.config,
                                              [{"role": "user", "content": TASKS[i % len(TASKS)]}],
                                              add_generation_prompt=True, enable_thinking=False))
               for i in range(B)]
        solo = [prefill(x) for x in ids]
        batch = [prefill(x) for x in ids]
        gdn = worker._gdn_stack(lm, batch, dt)
        y = [x[-1] for x in ids]
        pos = [len(x) - 1 for x in ids]
        agree = total = 0
        kls = []
        for _ in range(steps):
            ls = []
            for r in range(B):
                lg = lm(mx.array([[y[r]]]), cache=solo[r],
                        position_ids=mx.array([[pos[r]]], dtype=mx.int32)).logits[0, -1]
                quant(solo[r])
                ls.append(lg)
            cache = [gdn[i] if i in gdn else worker.RowsKV([c[i] for c in batch])
                     for i in range(len(lm.model.layers))]
            bl = lm(mx.array(y)[:, None], cache=cache,
                    position_ids=mx.array([[p] for p in pos], dtype=mx.int32)).logits[:, -1, :]
            for c in batch:
                quant(c)
            sl = mx.stack(ls).astype(mx.float32)
            bl = bl.astype(mx.float32)
            ps = sl - mx.logsumexp(sl, axis=-1, keepdims=True)
            pb_ = bl - mx.logsumexp(bl, axis=-1, keepdims=True)
            kl = (mx.exp(ps) * (ps - pb_)).sum(axis=-1)
            a_s, a_b = mx.argmax(sl, axis=-1), mx.argmax(bl, axis=-1)
            mx.eval(kl, a_s, a_b)
            agree += int((a_s == a_b).sum().item())
            total += B
            kls += kl.tolist()
            y = a_s.tolist()                   # teacher forcing: the solo greedy stream
            pos = [p + 1 for p in pos]
        kls.sort()
        print(f"B={B:2d}: top-1 agreement {100 * agree / total:.2f}% ({agree}/{total}), "
              f"KL mean {sum(kls) / len(kls):.2e} p99 {kls[int(0.99 * (len(kls) - 1))]:.2e} "
              f"max {kls[-1]:.2e}, peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
        del solo, batch, gdn
        mx.clear_cache()


if __name__ == "__main__":
    main()
