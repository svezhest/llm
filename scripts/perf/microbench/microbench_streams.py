#!/usr/bin/env python3
"""Does Metal overlap a compute-bound prefill with a bandwidth-bound batched
decode when they are queued on two MLX GPU streams? (batching.md V2 "piggyback")

Loads the 9B like the worker. Decode = the design's batched step (`rows` mode of
microbench_decode_batch: per-row KV caches + stacked GDN, B rows cloned from one
prefilled ~1K-token row), pipelined like Engine._run (queue step k+1, then
read token k). Prefill = the worker's slab loop over a fresh PF_LEN-token prompt
(random ids) into its own cache, slab PF_SLAB.

Phases, repeated PF_REPS times (A/B alternation inside one process):
  P   prefill alone on stream A (async_eval per slab, eval at the end)
  D   K decode steps alone on stream B (K chosen so K * step ~ prefill time)
  PD  both, one scheduler thread (as the worker would): all prefill slabs are
      queued on stream A (async_eval), then the K decode steps run pipelined on
      stream B; a waiter thread blocks on the last slab and records when it
      finished.
      serialized  -> wall ~ P + D, decode steps inside the prefill window ~ P/K slower
      overlapped  -> wall ~ max(P, D)
  SER the worker's current interleave on ONE stream: slab, decode step, slab, ...
      (single thread), for reference.

NEEDS THE GPU LEASE.
  cd src/worker && QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 \
     <venv>/bin/python <scratchpad>/design/microbench_streams.py
Env: PF_LEN (4096)  PF_SLABS (512,2048)  PF_B (4)  PF_CTX (1024)  PF_REPS (3)
"""
import os
import statistics
import sys
import threading
import time

import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import batching_proto as bp  # noqa: E402
import microbench_decode_batch as mbd  # noqa: E402

PF_LEN = int(os.environ.get("PF_LEN", "4096"))
SLABS = [int(x) for x in os.environ.get("PF_SLABS", "512,2048").split(",")]
B = int(os.environ.get("PF_B", "4"))
CTX = int(os.environ.get("PF_CTX", "1024"))
REPS = int(os.environ.get("PF_REPS", "3"))


def prefill_graph(model, ids, slab, stream):
    """Queue every slab of the prompt on `stream` (async); returns the arrays
    to wait on after the last slab."""
    lm = model.language_model
    cache = lm.make_cache()
    with mx.stream(stream):
        inp = mx.array([ids])
        emb = model.get_input_embeddings(inp, None).inputs_embeds
        while inp.shape[1] > 1:
            n = min(slab, inp.shape[1] - 1)
            lm(inp[:, :n], inputs_embeds=emb[:, :n], cache=cache, skip_logits=True)
            st = [c.state for c in cache]
            mx.async_eval(st)
            inp, emb = inp[:, n:], emb[:, n:]
    return cache, st


def run_prefill(model, ids, slab, stream):
    t = time.perf_counter()
    cache, st = prefill_graph(model, ids, slab, stream)
    mx.eval(st)
    return time.perf_counter() - t


def run_decode(step, y0, k_steps, stream, k_base):
    """Pipelined decode on `stream`; returns (total s, per-step times, last y)."""
    ta = time.perf_counter()
    with mx.stream(stream):
        nxt = step(y0, k_base)
    tb = time.perf_counter()
    mx.async_eval(nxt)
    FIRST[:] = [ta, tb, time.perf_counter()]
    times = []
    t0 = time.perf_counter()
    last = t0
    for k in range(1, k_steps + 1):
        cur = nxt
        with mx.stream(stream):
            nxt = step(cur, k_base + k)
        mx.async_eval(nxt)
        cur.tolist()
        now = time.perf_counter()
        times.append(now - last)
        STAMPS.append(now)
        last = now
    mx.eval(nxt)
    return time.perf_counter() - t0, times, nxt


STAMPS = []
FIRST = []


def main():
    model = mbd.load_like_worker(mbd.MODEL)
    lm = model.language_model
    bp.install_rows_attention()
    sA = mx.new_stream(mx.gpu)
    sB = mx.new_stream(mx.gpu)
    import random
    random.seed(1)
    vocab = lm.args.vocab_size
    ctx_ids = [random.randrange(1000, min(vocab, 150000)) for _ in range(CTX)]
    pf_ids = [random.randrange(1000, min(vocab, 150000)) for _ in range(PF_LEN)]
    with mx.stream(sB):
        base, last = mbd.prefill_row(model, ctx_ids)
    rows = [mbd.clone_row(base) for _ in range(B)]
    start = [CTX - 1] * B
    step = mbd.make_stepper("rows", lm, rows, last, start)
    y0 = mx.array([last] * B, dtype=mx.int32)
    kpos = [0]

    def dec(k_steps, stream):
        r = run_decode(step, y0, k_steps, stream, kpos[0])
        kpos[0] += k_steps + 1
        return r

    # warm-up both paths (JIT, buffer pool)
    run_prefill(model, pf_ids[:1024], 512, sA)
    dec(8, sB)
    print(f"9B, decode B={B} at ctx {CTX} (rows mode), prefill {PF_LEN} tokens; "
          f"mlx {mx.__version__}", flush=True)
    for slab in SLABS:
        res = {"P": [], "D_step": [], "PD_wall": [], "PD_pf": [], "PD_step_in": [], "PD_n_in": [], "PD_max_step": [], "SER": [],
               "SER_dstep": []}
        tp = run_prefill(model, pf_ids, slab, sA)
        _, ts, _ = dec(8, sB)
        K = max(4, int(tp / statistics.median(ts)))
        for rep in range(REPS):
            # P alone
            mx.clear_cache()
            res["P"].append(run_prefill(model, pf_ids, slab, sA))
            # D alone
            tot, ts, _ = dec(K, sB)
            res["D_step"].append(statistics.median(ts))
            # PD: prefill queued from a helper thread on stream A, decode in the main thread on B
            mx.clear_cache()
            done = {}

            # MLX streams are per thread: queue every slab from THIS thread
            # (async_eval on stream A), a waiter thread only blocks on the result.
            t0 = time.perf_counter()
            pcache, st = prefill_graph(model, pf_ids, slab, sA)
            done["t0"] = t0

            def pf():
                mx.eval(st)
                done["t"] = time.perf_counter()
            th = threading.Thread(target=pf)
            th.start()
            STAMPS.clear()
            tot, ts, _ = dec(K, sB)
            t_dec = time.perf_counter()
            th.join()
            wall = max(t_dec, done["t"]) - t0
            res["PD_wall"].append(wall)
            res["PD_pf"].append(done["t"] - done["t0"])
            # decode steps that completed while the prefill was still running
            inside = [x for x in STAMPS if x <= done["t"]]
            print(f"    [PD slab {slab}] prefill done at {done['t'] - t0:.2f} s, decode done at "
                  f"{t_dec - t0:.2f} s, {len(inside)}/{K} steps before the prefill finished; first step: build "
                  f"{FIRST[1] - FIRST[0]:.3f} s, async_eval {FIRST[2] - FIRST[1]:.3f} s, first token at "
                  f"{(STAMPS[0] - t0) if STAMPS else float('nan'):.2f} s", flush=True)
            # effective decode step time while the prefill ran: window / steps done in it
            res["PD_step_in"].append((done["t"] - t0) / len(inside) if inside else float("nan"))
            res["PD_n_in"].append(len(inside) / 1e3)
            res["PD_max_step"].append(max(ts))
            # SER: the worker's single-stream interleave: one slab, one decode step, ...
            mx.clear_cache()
            lmc = lm.make_cache()
            t = time.perf_counter()
            with mx.stream(sB):
                inp = mx.array([pf_ids])
                emb = model.get_input_embeddings(inp, None).inputs_embeds
            y = y0
            dsteps = []
            n_slabs = 0
            while inp.shape[1] > 1:
                n = min(slab, inp.shape[1] - 1)
                with mx.stream(sB):
                    lm(inp[:, :n], inputs_embeds=emb[:, :n], cache=lmc, skip_logits=True)
                mx.eval([c.state for c in lmc])
                inp, emb = inp[:, n:], emb[:, n:]
                n_slabs += 1
                td = time.perf_counter()
                with mx.stream(sB):
                    y = step(y, kpos[0])
                kpos[0] += 1
                y.tolist()
                dsteps.append(time.perf_counter() - td)
            res["SER"].append(time.perf_counter() - t)
            res["SER_dstep"].append(statistics.median(dsteps))
            del lmc
        med = {k: statistics.median(v) for k, v in res.items()}
        p, d = med["P"], med["D_step"] * K
        print(f"\nslab {slab}: K={K} decode steps per prefill window, reps={REPS}")
        for k, v in res.items():
            print(f"  {k:11s} " + "  ".join(f"{x * 1e3:8.1f}" for x in v) + f"   median {med[k] * 1e3:8.1f} ms")
        print(f"  prefill alone {p:.2f} s; K decode steps alone {d:.2f} s "
              f"({med['D_step'] * 1e3:.1f} ms/step); sum {p + d:.2f} s, max {max(p, d):.2f} s")
        print(f"  concurrent wall {med['PD_wall']:.2f} s  -> overlap = (sum - wall) / min = "
              f"{(p + d - med['PD_wall']) / min(p, d):.2f} (0 = serialized, 1 = perfect)")
        print(f"  prefill under decode {med['PD_pf']:.2f} s (x{med['PD_pf'] / p:.2f}); decode steps done "
              f"during it {med['PD_n_in'] * 1e3:.0f} of {K} -> effective {med['PD_step_in'] * 1e3:.1f} ms/step "
              f"(x{med['PD_step_in'] / med['D_step']:.2f} vs alone); longest step {med['PD_max_step'] * 1e3:.0f} ms")
        print(f"  single-stream interleave (slab, step, ...): {med['SER']:.2f} s for the prefill + "
              f"{PF_LEN // slab} steps; decode step there {med['SER_dstep'] * 1e3:.1f} ms", flush=True)


if __name__ == "__main__":
    main()
