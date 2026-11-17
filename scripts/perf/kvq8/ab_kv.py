"""fp16 KV vs quantized KV on the 27B single stream, interleaved in ONE process
(gates (ii) and (iii) of docs/perf/2026-11-15/design/kv_q8.md): prefill time,
plain decode and DFlash tok/s at a few contexts.

Real Engine + _Scheduler, loaded once (start it with QWEN_KV_BITS=8 and
QWEN_KVQ_PREFILL_FUSED=1, so the prefill reserve has room for the dequantized
layer). A variant is switched between requests, never during one (`use`):
the scheduler's KV settings (self.kv's kv_bits, self.quantize, self.kv_q),
the frozen slab policy (sizing._FIXED_SLAB) and QWEN_KVQ_PREFILL_FUSED, and
the prompt cache — every KV format has its own _PrefixCache (an fp16 entry is
never read by a q8 run). Variants: fp16 / q8 / q8f (q8 + the dequantized
fused prefill; same decode as q8); a "v" suffix on a quantized variant (q8v,
q8fv): its decode steps and DFlash verify blocks through the one-pass q8
kernel (QWEN_VATT_Q8; the worker loaded with it), without: the library's
quantized SDPA. Variants that differ only by "v" share their prefix cache and
their prefill (it is the same). fp16 runs whatever QWEN_VERIFY_ATT /
QWEN_DECODE_ATT the worker was loaded with.

Per context (--contexts; a system document of that many tokens, the repo's
docs, + a coding task, thinking off unless --thinking):
  prefill  --prefill-rounds rounds; each round runs every variant once
           (order alternating per round), prompt_cache "off" (cold, stored:
           the entry the decode runs resume from), max_tokens 1: TTFT =
           prefill time, its peak.
  decode   --rounds rounds; each runs every variant (order alternating) as
           plain decode and as DFlash (drafter on), prompt_cache "read"
           (resumes at the prompt's end, stores nothing), --max-tokens
           tokens: tok/s, DFlash rounds / accepted per round / ms per round,
           the peak.
Medians per (context, variant, mode) at the end, and the ratios q8 / fp16.

Memory: every variant's entry is held at once, and a "read" run copies its
entry into the row (MOVE never takes an entry for a read-only request). 27B
4-bit (weights + drafter ~16 GiB): 8K and 32K fit all three; at 80K fp16's
entry + its copy are 10 GiB, so run 80K with --variants q8 q8f (fp16 at 80K:
long_ctx.py, gate (iv)).

  cd src/worker && QWEN_MODEL=../../weights/qwen38-27b-mlx-4bit QWEN_KV_BITS=8 \\
    QWEN_KVQ_PREFILL_FUSED=1 QWEN_MRQMM4=1 QWEN_DRAFT=../../weights/qwen38-27b-dflash2 QWEN_PROMPT_CACHE=1 \\
    QWEN_PROMPT_CACHE_MOVE=1 QWEN_PROMPT_CACHE_GB=12 QWEN_MAX_BATCH=1 QWEN_WIRED_LIMIT_GB=27 \\
    QWEN_CACHE_LIMIT_GB=1 QWEN_SDPA_FUSED_MIN_KV=8192 QWEN_PREFILL_BUDGET=16777216 \\
    QWEN_PRESENCE_PENALTY=0 QWEN_MAX_CONTEXT=98304 PYTHONPATH=. .venv/bin/python \\
    ../../scripts/perf/kvq8/ab_kv.py --contexts 8192 32768 --json $PERF_OUT/kvq8/ab_8k_32k.json
  ... ab_kv.py --contexts 81920 --variants q8 q8f --prefill-rounds 1 --json $PERF_OUT/kvq8/ab_80k.json
"""
import argparse
import functools
import glob
import json
import logging
import os
import statistics
import sys
import time

sys.path.insert(0, os.getcwd())
import worker as W  # noqa: E402
import worker_pb2 as pb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "prof"))
from doc_notes import NOTES  # noqa: E402
from wtest import TASKS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--contexts", type=int, nargs="+", default=[8192, 32768])
ap.add_argument("--variants", nargs="+", default=["fp16", "q8", "q8f"])
ap.add_argument("--prefill-rounds", type=int, default=2)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--modes", nargs="+", default=["plain", "dflash"])
ap.add_argument("--thinking", action="store_true")
ap.add_argument("--decode-mode", default="read",
                help="prompt_cache mode of the decode runs: read (a copy of the entry, nothing stored) or '' "
                     "(with QWEN_PROMPT_CACHE_MOVE the entry moves into the row and comes back with the reply: "
                     "one KV copy, for contexts where two do not fit)")
ap.add_argument("--json", default="")
a = ap.parse_args()
assert os.environ.get("QWEN_PROMPT_CACHE") == "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
import mlx.core as mx  # noqa: E402
from mlx_vlm.generate.common import maybe_quantize_kv_cache  # noqa: E402

eng = W.Engine(os.environ["QWEN_MODEL"])
eng.load()
sched, lm = eng.sched, eng.model.language_model
DRAFT = eng.draft
if DRAFT is None and "dflash" in a.modes:
    print("no drafter (QWEN_DRAFT): DFlash runs skipped", flush=True)
    a.modes = [m for m in a.modes if m != "dflash"]
tok = getattr(eng.processor, "tokenizer", eng.processor)
START = sched.kv["quantized_kv_start"]
VQ8 = (W._VATT_Q8, W._VATT_Q8_DEC)               # as the load's self-test left them


def fmt(v):
    """The KV format of a variant (its prefix cache): the "v" suffix only changes the attention."""
    return v[:-1] if v.endswith("v") and v != "fp16" else v


def parse(v):
    if v == "fp16":
        return None, False
    v = fmt(v)
    assert v[0] == "q", v
    return int(v[1:].rstrip("f")), v.endswith("f")


FMTS = list(dict.fromkeys(fmt(v) for v in a.variants))
CACHES = {f: W._PrefixCache() for f in FMTS}
if any(v != fmt(v) for v in a.variants):
    assert VQ8[0] and VQ8[1], "a 'v' variant needs the worker loaded with QWEN_VATT_Q8=1 (self-test passed)"


def use(v):
    """Switch the KV variant between requests (the scheduler is idle)."""
    assert sched.task is None and not sched.rows and not sched.waiting
    bits, fused = parse(v)
    for kv in {id(sched.kv): sched.kv, id(W.KV_KWARGS): W.KV_KWARGS}.values():
        kv["kv_bits"] = bits
    sched.quantize = functools.partial(maybe_quantize_kv_cache, quantized_kv_start=START,
                                       kv_group_size=sched.kv["kv_group_size"], kv_bits=bits)
    sched.kv_q = W._kv_token_bytes(lm, bits)
    W.KVQ_PREFILL_FUSED = fused
    W._FIXED_SLAB = bool((bits is None or fused) and W.SDPA_FUSED and W.FP16)
    vq = v != fmt(v)
    W._VATT_Q8, W._VATT_Q8_DEC = VQ8[0] and vq, VQ8[1] and vq
    eng._pcache = CACHES[fmt(v)]


def doc(n):
    text = ""
    for f in list(NOTES) + sorted(p for p in
            glob.glob(os.path.join(ROOT, "docs", "**", "*.md"), recursive=True)
            if os.path.relpath(p, ROOT) not in NOTES):
        text += open(os.path.join(ROOT, f), encoding="utf-8").read() + "\n\n"
        if len(text) > 6 * n:
            break
    ids = tok.encode(text)
    assert len(ids) >= n, f"only {len(ids)} tokens of documents"
    return ("You are a coding assistant. Project notes follow; use them only if relevant.\n\n"
            + tok.decode(ids[:n]))


def run(messages, rid, mode, max_tokens):
    if DRAFT is not None:
        DRAFT.accept_lens, DRAFT.draft_lens = [], []
    req = pb.Request(id=rid, messages=[pb.Message(role=r, content=c) for r, c in messages],
                     params=pb.Params(temperature=0.0, presence_penalty=0.0, max_tokens=max_tokens,
                                      prompt_cache=mode, chat_template_kwargs_json=json.dumps(
                                          {"enable_thinking": a.thinking})))
    formatted, _ = eng._render(req, [])
    pids = tok.encode(formatted)
    p = req.params
    sampling = dict(W.SAMPLING_KWARGS)
    for f in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "seed"):
        if p.HasField(f):
            sampling[f] = getattr(p, f)
    ctl = W._Control(p, tok.stopping_criteria.eos_token_ids, tok, eng.vocab)
    r = W._Req(id=rid, p=p, ids=pids, max_tokens=eng._limit(p, len(pids)), sampling=sampling, ctl=ctl,
               formatted=formatted)
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    eng.sched.submit(r)
    toks, times, end = [], [], None
    while True:
        ev = r.out.get()
        if ev[0] != "tok":
            end = ev
            break
        toks.append(ev[1])
        times.append(time.perf_counter())
    peak = mx.get_peak_memory()
    time.sleep(0.2)                              # the row's store runs right after its last event
    while sched.task is not None or sched.rows:
        time.sleep(0.05)
    acc = [int(x) for x in DRAFT.accept_lens] if DRAFT is not None and eng.draft is not None else []
    dt = times[-1] - times[0] if len(toks) > 2 else 0.0
    return dict(id=rid, prompt=len(pids), reused=r.reused, n=len(toks), end=list(end)[:2], tokens=toks,
                ttft=round(times[0] - t0, 2) if times else None,
                tps=round((len(toks) - 1) / dt, 2) if dt else None, rounds=len(acc),
                accepted=round(sum(acc) / len(acc), 2) if acc else None,
                round_ms=round(1e3 * dt / len(acc), 1) if acc and dt else None,
                peak_gib=round(peak / 2**30, 2), active_gib=round(mx.get_active_memory() / 2**30, 2),
                cache_gib=round(sum(c.bytes for c in CACHES.values()) / 2**30, 2))


def show(o):
    print(f"  {o['id']:24s} prompt {o['prompt']:6d} reused {o['reused']:6d} ttft {o['ttft']}s tps {o['tps']} "
          f"n {o['n']} rounds {o['rounds']} acc/round {o['accepted']} round {o['round_ms']} ms "
          f"peak {o['peak_gib']} GiB active {o['active_gib']} cache {o['cache_gib']} GiB end {o['end']}",
          flush=True)


print(f"loaded; drafter {'on' if DRAFT is not None else 'OFF'}, move: {sched.move_state()}, budget "
      f"{sched.budget / 2**30:.2f} GiB, prefill reserve {sched.prefill_reserve / 2**30:.2f} GiB; active "
      f"{mx.get_active_memory() / 2**30:.2f} GiB", flush=True)
res = []
for ctx in a.contexts:
    msgs = [("system", doc(ctx)), ("user", TASKS[0])]
    for f in FMTS:
        CACHES[f] = W._PrefixCache()             # the previous context's entries go
    mx.clear_cache()
    for k in range(a.prefill_rounds):
        order = FMTS if k % 2 == 0 else FMTS[::-1]
        for v in order:
            use(v)
            o = run(msgs, f"{ctx}-{v}-prefill{k + 1}", "off", 1)
            o.update(ctx=ctx, variant=v, mode="prefill", round=k + 1)
            show(o)
            res.append(o)
    for k in range(a.rounds):
        r = k % len(a.variants)
        order = a.variants[r:] + a.variants[:r]   # rotate: no variant always first
        for v in order:
            use(v)
            for m in (a.modes if k % 2 == 0 else a.modes[::-1]):
                eng.draft = DRAFT if m == "dflash" else None
                o = run(msgs, f"{ctx}-{v}-{m}{k + 1}", a.decode_mode, a.max_tokens)
                o.update(ctx=ctx, variant=v, mode=m, round=k + 1)
                eng.draft = DRAFT
                show(o)
                res.append(o)

print("\nidentity (tokens == the variant's first plain run):", flush=True)
for ctx in a.contexts:
    for v in a.variants:
        os_ = [o for o in res if o["ctx"] == ctx and o["variant"] == v and o["mode"] != "prefill"]
        ref = next((o["tokens"] for o in os_ if o["mode"] == "plain"), None)
        for o in os_:
            o["same_as_plain"] = ref is not None and o["tokens"] == ref
        print(f"{ctx:6d} {v:5s} " + " ".join(f"{o['mode']}{o['round']}={'=' if o['same_as_plain'] else 'x'}"
                                             for o in os_), flush=True)
    toks = {v: next((o["tokens"] for o in res if o["ctx"] == ctx and o["variant"] == v and o["mode"] == "plain"),
                    None) for v in a.variants}
    print(f"{ctx:6d} plain answers equal across variants: "
          + ", ".join(f"{v}={'=' if toks[v] == toks[a.variants[0]] else 'x'}" for v in a.variants), flush=True)
for o in res:
    o.pop("tokens", None)
print("\nmedians (prefill: TTFT s; plain / dflash: tok/s, DFlash ms per round)", flush=True)
summary = {}
for ctx in a.contexts:
    for m in ["prefill"] + a.modes:
        for v in a.variants:
            os_ = [o for o in res if o["ctx"] == ctx and o["variant"] == v and o["mode"] == m]
            key = "ttft" if m == "prefill" else "tps"
            xs = [o[key] for o in os_ if o[key] is not None]
            if not xs:
                continue
            s = dict(median=statistics.median(xs), n=len(xs), peak_gib=max(o["peak_gib"] for o in os_))
            if m == "dflash":
                rm = [o["round_ms"] for o in os_ if o["round_ms"]]
                s["round_ms"] = statistics.median(rm) if rm else None
                s["accepted"] = statistics.median([o["accepted"] for o in os_ if o["accepted"]] or [0])
            summary[f"{ctx}/{m}/{v}"] = s
            ref = summary.get(f"{ctx}/{m}/{a.variants[0]}")
            ratio = f"  x{s['median'] / ref['median']:.3f} vs {a.variants[0]}" if ref and v != a.variants[0] else ""
            print(f"{ctx:6d} {m:8s} {v:5s} {key} {s['median']:.2f} (n {s['n']}) peak {s['peak_gib']} GiB"
                  + (f" round {s['round_ms']} ms acc {s['accepted']}" if m == "dflash" else "") + ratio,
                  flush=True)
if a.json:
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    json.dump(dict(args=vars(a), env={k: v for k, v in os.environ.items() if k.startswith("QWEN_")},
                   runs=res, summary=summary), open(a.json, "w"), indent=1)
sys.stdout.flush()
os._exit(0)
