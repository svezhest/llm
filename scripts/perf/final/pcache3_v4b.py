#!/usr/bin/env python3
"""V4b: on-demand eviction of a resident ~120K entry. One cold ~120K prompt (stored over the
soft cap), then N concurrent distinct ~32K prompts with max_tokens 4096 (reservation ~1.3 GiB
per row): the 9th admission or so only fits once the 120K entry goes. The clients disconnect
after HOLD s (gateway -> Worker.Cancel). Args: OUT.json WORKER_LOG [N=10] [HOLD=150]"""
import json, os, sys, threading, time
sys.path.insert(0, "/Users/user/Projects/llm/scripts")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_concurrency import Client, make_messages
import pcache2_v1 as v1
out, wlog = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 10
HOLD = float(sys.argv[4]) if len(sys.argv) > 4 else 150
tag = f"v4b-{int(time.time()) % 100000}"
cl = Client("http://127.0.0.1:8090", 1800)
B = lambda msgs, rid, mt: {"model": v1.MODEL, "messages": msgs, "stream": True, "max_tokens": mt, "temperature": 0,
                           "presence_penalty": 0, "request_id": rid, "stream_options": {"include_usage": True},
                           "chat_template_kwargs": {"enable_thinking": False}, "ignore_eos": True}
res = {"tag": tag}
r = cl.chat(B(make_messages(1, f"{tag}-L", 298000, shared=True), f"bench-{tag}-L", 32))
res["L"] = r; res["health_after_L"] = v1.health()
print(f"L: prompt={r['prompt_tokens']} ttft={r['ttft_s']:.1f}s health={res['health_after_L']}", flush=True)
recs = {}
def go(i):
    try:
        recs[i] = cl.chat(B(make_messages(10 + i, f"{tag}-M{i}", 80000), f"bench-{tag}-M{i}", 4096))
    except Exception as e:  # noqa: BLE001
        recs[i] = {"error": str(e)}
ths = [threading.Thread(target=go, args=(i,), daemon=True) for i in range(N)]
t0 = time.time()
for t in ths:
    t.start()
time.sleep(HOLD)
res["health_at_hold"] = v1.health()
print(f"after {HOLD:.0f}s: health={res['health_at_hold']} finished={len(recs)}", flush=True)
json.dump(res, open(out, "w"), indent=1, default=str)
os._exit(0)          # drop the open streams: the gateway cancels them on the worker
