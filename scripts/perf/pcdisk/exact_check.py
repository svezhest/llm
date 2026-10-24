#!/usr/bin/env python3
"""Real-model check of the disk tier (QWEN_PROMPT_CACHE_DISK): a hit through
the disk == cold, bit for bit (stdlib only; through the gateway).

Worker: QWEN_MAX_BATCH=1, QWEN_PROMPT_CACHE_REPLY=0 (hit == cold holds
bitwise there), a QWEN_PROMPT_CACHE_GB small enough that every store keeps
only the newest entry (0.3: one 6K+ AppWorld prompt is more), the disk tier
on (run_exact.sh starts it so). For each pair (A, B) of recorded AppWorld calls
from different tasks:
  prime   A, max_tokens 1                      -> A's entry in RAM
  evict   B, max_tokens 1                      -> B's entry evicts A's to disk
  tf_hit  A, prompt_logprobs, max_tokens 0, prompt_cache "read": resumes from A's
          entry loaded back from disk (cached > 0, a disk load in the worker log)
  tf_cold the same, prompt_cache false: every position scored
          -> the scored log-probs after the resume point are equal (==)
  evict again (B), then
  gen_hit A + "Continue." as a new user message, 64 greedy tokens (disk hit)
  gen_cold the same, prompt_cache false      -> identical content
Usage: exact_check.py [--url http://127.0.0.1:8090/v1] [--n 3] [--worker-log W] TASKS_DIR
PERF_OUT (default /tmp/perf) gets pcdisk_exact.json; exit 1 on any mismatch."""
import argparse
import glob
import json
import os
import sys
import time
import urllib.request


def post(url, body):
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    return out, time.time() - t


def disk_loads(log):
    if not log or not os.path.exists(log):
        return 0
    return sum(1 for line in open(log, errors="replace") if "prompt cache disk: entry of" in line
               and "loaded" in line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks")
    ap.add_argument("--url", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--model", default="ornith15-9b-mlx-8bit")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--worker-log", default=None)
    ap.add_argument("--settle", type=float, default=3.0, help="s for the writer after an eviction")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.tasks, "*/logs/lm_calls.jsonl")))
    # the last call of each task: the longest contexts (6K-30K tokens)
    recs = [json.loads(open(f).readlines()[-1]) for f in files]
    base = dict(model=a.model, temperature=0, seed=100, chat_template_kwargs={"enable_thinking": False})
    rows, bad = [], 0
    for i in range(a.n):
        A, B = recs[2 * i]["input"]["messages"], recs[2 * i + 1]["input"]["messages"]
        post(a.url, dict(base, messages=A, max_tokens=1))
        post(a.url, dict(base, messages=B, max_tokens=1))
        time.sleep(a.settle)
        n0 = disk_loads(a.worker_log)
        tf = dict(base, messages=A, max_tokens=0, prompt_logprobs=True)
        hit, t_hit = post(a.url, dict(tf, prompt_cache="read"))
        n1 = disk_loads(a.worker_log)
        cold, t_cold = post(a.url, dict(tf, prompt_cache=False))
        ph, pc = hit["prompt_logprobs"], cold["prompt_logprobs"]
        first = next((k for k, x in enumerate(ph) if x is not None), None)
        same = first is not None and len(ph) == len(pc) and all(x == y for x, y in zip(ph[first:], pc[first:]))
        cached = hit["usage"].get("prompt_tokens_details", {}).get("cached_tokens")
        post(a.url, dict(base, messages=B, max_tokens=1))
        time.sleep(a.settle)
        n2 = disk_loads(a.worker_log)
        A2 = A + [{"role": "user", "content": "Continue."}]
        gh, t_gh = post(a.url, dict(base, messages=A2, max_tokens=64))
        n3 = disk_loads(a.worker_log)
        gc, t_gc = post(a.url, dict(base, messages=A2, max_tokens=64, prompt_cache=False))
        g_same = gh["choices"][0]["message"] == gc["choices"][0]["message"]
        g_cached = gh["usage"].get("prompt_tokens_details", {}).get("cached_tokens")
        row = dict(pair=i, prompt_tokens=cold["usage"]["prompt_tokens"], tf_cached=cached,
                   tf_first_scored=first, tf_equal=same, tf_disk_load=n1 - n0, gen_cached=g_cached,
                   gen_equal=g_same, gen_disk_load=n3 - n2,
                   t=dict(tf_hit=t_hit, tf_cold=t_cold, gen_hit=t_gh, gen_cold=t_gc))
        ok = same and g_same and (cached or 0) > 0 and (g_cached or 0) > 0
        if a.worker_log:
            ok = ok and row["tf_disk_load"] == 1 and row["gen_disk_load"] == 1
        bad += not ok
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = os.environ.get("PERF_OUT", "/tmp/perf")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "pcdisk_exact.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(f"{len(rows)} pairs, {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
