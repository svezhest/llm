#!/usr/bin/env python3
"""V4 (segments, Part A): long-context store bound on the real model, through the gateway (:8090).

  pcache3_v4.py OUT.json WORKER_LOG [chars_120k] [chars_32k]
Sequential, streamed, temperature 0, thinking off, ignore_eos, max_tokens 32:
  L1  ~120K-token prompt (cold)        -> store or "not kept: over the batch's headroom"
  L2  the same prompt again             -> HIT if L1 was kept
  M1  ~32K-token prompt (other filler)  -> admitted next to the resident 120K entry
Health (entries/bytes/hits/misses/evictions) after each. No calibration request (the bench's
would itself prefill 120K): chars per token from the phase-1 32K wave (82770 chars -> 33320 tok).
Run with src/worker/.venv/bin/python, PYTHONPATH=src/worker."""
import json, os, sys, time
sys.path.insert(0, "/Users/user/Projects/llm/scripts")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_concurrency import Client, make_messages, parse_worker_log
import pcache2_v1 as v1

out, wlog = sys.argv[1], sys.argv[2]
c120 = int(sys.argv[3]) if len(sys.argv) > 3 else 298000
c32 = int(sys.argv[4]) if len(sys.argv) > 4 else 80000
cl = Client("http://127.0.0.1:8090", 1800)
tag = f"v4-{int(time.time()) % 100000}"


def body(msgs, rid):
    return {"model": v1.MODEL, "messages": msgs, "stream": True, "max_tokens": 32, "temperature": 0,
            "presence_penalty": 0, "request_id": rid, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}, "ignore_eos": True}


L = make_messages(1, f"{tag}-L", c120, shared=True)
M = make_messages(2, f"{tag}-M", c32, shared=True)
res = {"tag": tag, "chars": [c120, c32], "steps": []}
for key, msgs in (("L1", L), ("L2", L), ("M1", M)):
    rid = f"bench-{tag}-{key}"
    t = time.time()
    try:
        r = cl.chat(body(msgs, rid))
    except Exception as e:  # noqa: BLE001
        r = {"error": str(e)}
    r["request_id"], r["key"], r["t_wall"] = rid, key, t
    time.sleep(0.5)
    r.update(parse_worker_log(wlog, {f"chatcmpl-{rid}"}).get(f"chatcmpl-{rid}", {}))
    r["health"] = v1.health()
    res["steps"].append(r)
    print(f"{key}: prompt={r.get('prompt_tokens')} ttft={r.get('ttft_s', float('nan')):.2f}s "
          f"tok={r.get('completion_tokens')} hit={r.get('hit')} reused={r.get('reused')} "
          f"peak={r.get('peak_gb')} store_mb={r.get('store_mb')} health={r['health']} err={r.get('error')}",
          flush=True)
    json.dump(res, open(out, "w"), indent=1)
    if r.get("error"):
        break
