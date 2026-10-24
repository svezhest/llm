#!/usr/bin/env python3
"""GPU A/B of the prefill lane (QWEN_PREFILL_LANE), in ONE worker process.
NEEDS THE GPU LEASE (loads its own 9B copy; ~10 GB + KV, peak ~16-18 GB).

Each round: a long COLD prompt (LANE_LONG tokens, default 40K: ~100 s of
prefill) starts; LANE_SHORT (6) short next-turn requests arrive while it
prefills (every LANE_GAP_S = 4 s from LANE_START_S = 5 s). Each short request
continues a conversation whose first turn was answered at warm-up (prompt
4.5-12K tokens), so it is a prompt-cache hit after the reply (reply KV) +
~150-300 new tokens: one slab. Rounds alternate lane on / off
(worker.PREFILL_LANE toggled between rounds; everything else equal), default
on,off,off,on,on,off (LANE_ROUNDS). Before every round the prompt cache is
reset to the warm-up snapshot: every round's long prompt is cold and every
short one hits the same entry.

Per round: client-side TTFT / wall of each short request and of the long one
(eng.generate: the real handler path) and a checksum of the long prompt's
prefill caches at its prompt-cache store (KV + recurrent state): it must be
bitwise equal in every round (the lane must not change the long prefill's
slabs or positions on Metal). Greedy answers of the short requests are
compared across modes (informational: B >= 3 decode is not batch-invariant).

PASS: (1) checksum identical in every round; (2) lane on: short TTFT p50 <= 4 s,
max <= 15 s; lane off: short TTFT p50 >= 20 s (they wait for the long prefill);
(3) long TTFT on / off (median per mode) <= 1.35 (the decode share the short
rows take while it prefills); (4) no error, every request ends with stop/length.

  cd src/worker && setopt NO_BG_NICE; mkdir -p /tmp/perf; \\
  QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 QWEN_PROMPT_CACHE=1 \\
  QWEN_PROMPT_CACHE_GB=6 QWEN_BATCH_RESERVE_TOKENS=1024 QWEN_PRESENCE_PENALTY=0 \\
  QWEN_PREFILL_LANE=2048 PERF_OUT=/tmp/perf PYTHONPATH=. \\
  .venv/bin/python ../../scripts/perf/lane/lane_ab.py 2>&1 | tee /tmp/perf/lane_ab.log
"""
import hashlib
import json
import os
import random
import statistics as st
import threading
import time

import mlx.core as mx
import numpy as np

import worker
import worker_pb2 as pb

OUT = os.environ.get("PERF_OUT", "/tmp/perf")
LONG = int(os.environ.get("LANE_LONG", "40000"))
NSHORT = int(os.environ.get("LANE_SHORT", "6"))
START_S = float(os.environ.get("LANE_START_S", "5"))
GAP_S = float(os.environ.get("LANE_GAP_S", "4"))
ROUNDS = os.environ.get("LANE_ROUNDS", "on,off,off,on,on,off").split(",")
LANE_ON = int(os.environ.get("QWEN_PREFILL_LANE") or "2048")     # tokens, the "on" rounds' value
WORDS = ("name id status amount date user playlist song album artist email phone count total "
         "created updated access token page limit offset title description value price").split()
OFF = json.dumps({"enable_thinking": False})


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def filler(seed, n_tok, tok):
    """~n_tok tokens of AppWorld-like JSON output, distinct per seed."""
    rnd = random.Random(seed)
    lines = [f"Output {seed}:", "```"]
    while len(tok.encode("\n".join(lines))) < n_tok:
        lines += [f'{{"{rnd.choice(WORDS)}_{seed}": "{rnd.choice(WORDS)} {rnd.randrange(10**6)}", '
                  f'"{rnd.choice(WORDS)}": {rnd.randrange(10**4)}}},' for _ in range(16)]
    return "\n".join(lines) + "\n```\n"


def req(rid, messages, max_tokens):
    return pb.Request(id=rid, messages=[pb.Message(role=m["role"], content=m["content"]) for m in messages],
                      params=pb.Params(max_tokens=max_tokens, temperature=0.0, presence_penalty=0.0,
                                       chat_template_kwargs_json=OFF))


def call(eng, r, out):
    t0, first, text = time.perf_counter(), None, []
    for ev in eng.generate(r):
        k = ev.WhichOneof("kind")
        if k == "token":
            first = first or time.perf_counter()
            text.append(ev.token.text)
        elif k == "done":
            u = ev.done.usage
            # worker-side TTFT: arrival -> its first token (any token; the
            # client's first Token event waits for the first text the
            # detokenizer flushes: a token starting with a space)
            out.update(finish=ev.done.finish, cached=u.cached_tokens, prompt=u.prompt_tokens,
                       queue_ms=u.queue_ms, w_ttft_s=round((u.queue_ms + u.ttft_ms) / 1e3, 3))
        elif k == "error":
            out.update(error=f"{ev.error.kind}: {ev.error.msg}")
    t1 = time.perf_counter()
    out.update(ttft_s=round((first or t1) - t0, 3), wall_s=round(t1 - t0, 3), text="".join(text))


def cache_sum(cache):
    """sha1 of a row's caches: every full-attention layer's KV up to its
    offset, every recurrent state (the bits a prefill leaves)."""
    h = hashlib.sha1()
    for c in cache:
        if hasattr(c, "cache"):
            xs = [a for a in c.cache if a is not None]
        else:
            xs = [y[..., :c.offset, :] for x in (c.keys, c.values)
                  for y in (x if isinstance(x, (tuple, list)) else (x,))]
        for x in xs:
            h.update(np.array(x.astype(mx.float32)).tobytes())
    return h.hexdigest()[:16]


def hook_store(pc, sums):
    """cache_sum of every prefix store (not reply stores), by its length: the
    long prompt's prefill caches as its prefill left them. Runs on the
    scheduler thread, in the prefill tick (a host wait there is fine)."""
    orig = pc.store

    def store(ids, cache, *a, **k):
        if k.get("reply") is None:
            sums[len(ids)] = cache_sum(cache)
        return orig(ids, cache, *a, **k)
    pc.store = store


def main():
    os.makedirs(OUT, exist_ok=True)
    eng = worker.Engine(worker.DEFAULT_MODEL)
    eng.load()
    tok = eng.processor.tokenizer
    pc = eng._pcache
    sums = {}
    hook_store(pc, sums)
    sysmsg = "You are an agent that writes Python against the apis object. " + filler(1, 1200, tok)
    convs = [[{"role": "system", "content": sysmsg},
              {"role": "user", "content": f"Task {i}: " + filler(100 + i, 3000 + 1500 * i, tok)}]
             for i in range(NSHORT)]
    log(f"warm-up: {NSHORT} conversations")
    replies = []
    for i, m in enumerate(convs):
        o = {}
        call(eng, req(f"warm-{i}", m, 48), o)
        replies.append(o["text"])
        log(f"  warm-{i}: prompt {o.get('prompt')} ttft {o['ttft_s']} s finish {o.get('finish')}")
    nxt = [m + [{"role": "assistant", "content": replies[i]},
                {"role": "user", "content": filler(200 + i, 150 + 30 * i, tok)}]
           for i, m in enumerate(convs)]
    snap = list(pc.entries)
    long_msgs = [{"role": "system", "content": "You summarize logs."},
                 {"role": "user", "content": filler(999, LONG, tok)}]
    long_ids = tok.encode(eng._render(req("x", long_msgs, 1), [])[0])
    log(f"long prompt: {len(long_ids)} tokens; rounds {ROUNDS}")
    results = []
    for k, mode in enumerate(ROUNDS):
        pc.entries = list(snap)
        pc._recount()
        worker.PREFILL_LANE = LANE_ON if mode == "on" else 0
        time.sleep(2)
        outs = {"long": {}}
        ths = [threading.Thread(target=call, args=(eng, req(f"r{k}-long", long_msgs, 32), outs["long"]))]
        ths[0].start()
        time.sleep(START_S)
        for i in range(NSHORT):
            outs[i] = {}
            t = threading.Thread(target=call, args=(eng, req(f"r{k}-s{i}", nxt[i], 48), outs[i]))
            t.start()
            ths.append(t)
            time.sleep(GAP_S)
        for t in ths:
            t.join()
        sk = sums.pop(len(long_ids) - 1, None)
        sh = [outs[i] for i in range(NSHORT)]
        row = {"round": k, "mode": mode, "long_ttft": outs["long"]["ttft_s"],
               "long_wall": outs["long"]["wall_s"], "short_ttft": [o["ttft_s"] for o in sh],
               "short_wall": [o["wall_s"] for o in sh], "short_w_ttft": [o.get("w_ttft_s") for o in sh], "short_cached": [o.get("cached") for o in sh],
               "short_prompt": [o.get("prompt") for o in sh],
               "errors": [o.get("error") for o in [outs["long"]] + sh if o.get("error")],
               "finish": [o.get("finish") for o in [outs["long"]] + sh], "entry": sk,
               "texts": [hashlib.sha1(o["text"].encode()).hexdigest()[:10] for o in sh]}
        results.append(row)
        log(f"round {k} lane={mode}: long ttft {row['long_ttft']} s wall {row['long_wall']} s; short "
            f"ttft p50 {st.median(row['short_ttft']):.2f} max {max(row['short_ttft']):.2f} s; "
            f"cached {row['short_cached']} of {row['short_prompt']}; entry {sk}; errors {row['errors']}")
    on = [r for r in results if r["mode"] == "on"]
    off = [r for r in results if r["mode"] == "off"]
    s_on = [x for r in on for x in r["short_ttft"]]
    s_off = [x for r in off for x in r["short_ttft"]]
    ratio = st.median(r["long_ttft"] for r in on) / st.median(r["long_ttft"] for r in off)
    checks = {
        "entry_identical": len({r["entry"] for r in results}) == 1 and results[0]["entry"] is not None,
        "short_on_p50<=4s": st.median(s_on) <= 4.0, "short_on_max<=15s": max(s_on) <= 15.0,
        "short_off_p50>=20s": st.median(s_off) >= 20.0, "long_on/off<=1.35": ratio <= 1.35,
        "no_errors": all(not r["errors"] for r in results)
        and all(f in ("stop", "length") for r in results for f in r["finish"]),
    }
    same = sum(a == b for r1 in on for r2 in off for a, b in zip(r1["texts"], r2["texts"]))
    w_on = [x for r in on for x in r["short_w_ttft"]]
    w_off = [x for r in off for x in r["short_w_ttft"]]
    summary = {"short_ttft_on": sorted(s_on), "short_ttft_off": sorted(s_off),
               # informational: worker-side (arrival -> first token, text or not)
               "short_w_ttft_on_p50": st.median(w_on), "short_w_ttft_on_max": max(w_on),
               "short_w_ttft_off_p50": st.median(w_off),
               "long_ttft_on": [r["long_ttft"] for r in on], "long_ttft_off": [r["long_ttft"] for r in off],
               "long_ttft_ratio_on_off": round(ratio, 3),
               "short_text_equal_pairs": f"{same}/{len(on) * len(off) * NSHORT}",
               "checks": checks, "PASS": all(checks.values())}
    log(json.dumps(summary, indent=1))
    with open(os.path.join(OUT, "lane_ab.json"), "w") as f:
        json.dump({"rounds": results, "summary": summary, "long_tokens": len(long_ids)}, f, indent=1)
    log("RESULT:", "PASS" if summary["PASS"] else "FAIL", "->", os.path.join(OUT, "lane_ab.json"))
    os._exit(0)          # the scheduler thread never returns (see _Scheduler._run)


if __name__ == "__main__":
    main()
