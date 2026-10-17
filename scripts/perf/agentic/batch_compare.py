#!/usr/bin/env python3
"""S1 (docs/perf/2026-10-17/replay_batch1.md): replay runs at different QWEN_MAX_BATCH side by side,
with the gateway's wait split out and the run-to-run identity of the replies.

  python3 scripts/perf/agentic/batch_compare.py RUN.json ... [--probes DIR] [--ref NAME]

Per run (window = the replay's --duration-min): requests/min, client latency, the wait in front
of the worker (client latency - the worker's arrival -> done: the gateway's in-flight semaphore,
= max_batch, plus HTTP/gRPC), the worker's queue / ttft / decode, the scheduler split
(sched_timer), mean decode rows per step. Identity: per (task, call index) present in both runs,
the reply_sha (content + tool calls) equal or not; a task's trajectory diverges at its first
different reply (own replies: every later prompt differs too).
"""
import argparse, json, os, re, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay import pct  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--probes", default=None, help="dir with replay_b1_probe_<name>_{cool,warm}.txt")
ap.add_argument("--ref", default=None)
a = ap.parse_args()

R = {}
for p in a.runs:
    name = os.path.basename(p).replace("replay_b1_", "").replace("replay_", "").replace(".json", "")
    R[name] = json.load(open(p))
names = list(R)


def window(r):
    dl = r.get("deadline") or r["t_end"]
    return [x for x in r["requests"] if x["t_end"] <= dl]


def m3(xs):
    return f"{st.mean(xs):.1f} / {pct(xs, 50):.1f} / {pct(xs, 90):.1f}" if xs else "-"


rows = []
for n in names:
    r = R[n]
    S = r["summary"]
    W = S.get("worker", {})
    sc = W.get("sched") or {}
    recs = r["requests"]
    j = [x for x in recs if "w_lat" in x]
    gw = [x["latency_s"] - x["w_lat"] for x in j]
    probe = {}
    if a.probes:
        for k in ("cool", "warm"):
            f = os.path.join(a.probes, f"replay_b1_probe_{n}_{k}.txt")
            if os.path.exists(f):
                t = open(f).read()
                mm = re.search(r"\| 0 \| 1 \| 1 \| (\d+) \| [\d.]+ \| [\d.]+ \| ([\d.]+) \| [\d.]+ \| [\d.]+ \| [\d.]+ \| ([\d.]+) \|", t)
                if mm:
                    probe[k] = (float(mm[2]), float(mm[3]))
    rows.append((n, {
        "requests/min (window)": f"{S['req_per_min']:.1f}",
        "requests in window / total / errors": f"{S['requests_in_window']} / {S['requests']} / {S['errors']}",
        "tasks started": f"{S['tasks_started']}",
        "latency mean / p50 / p90 s": m3([x["latency_s"] for x in recs]),
        "wait before the worker (gateway) mean / p50 / p90 s": m3(gw),
        "worker queue mean / p50 / p90 s": m3([x["w_queue"] / 1000 for x in j]),
        "worker ttft mean / p50 / p90 s": m3([x["w_ttft"] / 1000 for x in j]),
        "worker decode mean / p50 / p90 s": m3([x["w_dec_s"] for x in j]),
        "prefill / decode tok/s (wall)": f"{S['prefill_tps_wall']:.0f} / {S['decode_tps_wall']:.1f}",
        "cache hit rate / mean reused": f"{100 * S['hit_rate']:.1f} % / {S['reused_mean'] / 1000:.1f}K",
        "scheduler prefill / decode / idle %": (f"{100 * sc['advance']:.0f} / {100 * sc['iterate']:.0f} / {100 * sc['idle']:.0f}"
                                                if sc else "-"),
        "ms per slab / per decode step": f"{sc['ms_per_slab']:.0f} / {sc['ms_per_step']:.0f}" if sc else "-",
        "mean decode rows per step": (f"{sum(v[0] * int(k) for k, v in sc['ms_per_step_by_rows'].items()) / max(sum(v[0] for v in sc['ms_per_step_by_rows'].values()), 1):.2f}"
                                      if sc.get("ms_per_step_by_rows") and isinstance(next(iter(sc["ms_per_step_by_rows"].values())), list) else f"{W.get('mean_rows', 0):.2f}"),
        "probe 8K TTFT s / decode tok/s, cool -> warm": " -> ".join(f"{probe[k][0]:.1f} / {probe[k][1]:.1f}" for k in ("cool", "warm") if k in probe) or "-",
    }))
keys = list(rows[0][1])
print("| | " + " | ".join(names) + " |")
print("|---|" + "---|" * len(names))
for k in keys:
    print(f"| {k} | " + " | ".join(d[k] for _, d in rows) + " |")

# identity
ref = a.ref or names[0]


def bykey(r):
    return {(x["task"], x["call"]): x for x in r["requests"] if x.get("reply_sha")}


print(f"\nreply identity vs other runs, per (task, call) present in both (own replies: a task diverges at "
      f"its first different reply):\n")
print("| pair | matched calls | same reply | same prompt length | tasks matched | tasks identical over all matched calls | first divergence (call index) median |")
print("|---|---|---|---|---|---|---|")
pairs = [(x, y) for i, x in enumerate(names) for y in names[i + 1:]]
for x, y in pairs:
    A, B = bykey(R[x]), bykey(R[y])
    ks = sorted(set(A) & set(B))
    if not ks:
        continue
    same = sum(1 for k in ks if A[k]["reply_sha"] == B[k]["reply_sha"])
    samep = sum(1 for k in ks if A[k]["prompt_tokens"] == B[k]["prompt_tokens"])
    tasks = sorted({k[0] for k in ks})
    ident, firsts = 0, []
    for t in tasks:
        cs = sorted(k[1] for k in ks if k[0] == t)
        d = next((c for c in cs if A[(t, c)]["reply_sha"] != B[(t, c)]["reply_sha"]), None)
        if d is None:
            ident += 1
        else:
            firsts.append(d)
    print(f"| {x} vs {y} | {len(ks)} | {same} ({100 * same / len(ks):.0f} %) | {samep} | {len(tasks)} | {ident} | "
          f"{st.median(firsts) if firsts else '-'} |")
