#!/usr/bin/env python3
"""Gate 3 (step cost): per-run summary of reply_kv_drive.py convs runs joined with their worker logs.
  reply_kv_g3_summary.py NAME=RUN.json:WORKER.log:TAG ...
Per run: wall s, aggregate completion tok/s, worker per-request decode tok/s (tps=), worker ttft_ms
(prefill start -> first token), queue_ms, client latency; split turn 1 / turns >= 2; prefilled tokens
(prompt - cached) per request; reply kept count."""
import json, re, statistics as st, sys

RX = re.compile(r"id=chatcmpl-(rk-[\w-]+?)-[0-9a-f]{12} finish=(\w+) gen_tokens=(\d+) ttft_ms=(\d+) tps=([\d.]+).*?queue_ms=(\d+) batch=([\d.]+)")
def med(xs): return st.median(xs) if xs else float("nan")
def mean(xs): return st.mean(xs) if xs else float("nan")
print("| run | wall s | agg tok/s | decode tok/s/req mean (t1 / t>=2) | ttft ms mean t1 / t>=2 | queue ms mean | client lat s mean t>=2 | prefilled tok/req t>=2 | mean batch | reply kept |")
print("|---|---|---|---|---|---|---|---|---|---|")
out = {}
for arg in sys.argv[1:]:
    name, rest = arg.split("=", 1)
    run, log, tag = rest.split(":")
    R = json.load(open(run))
    w = {}
    kept = 0
    for ln in open(log, errors="replace"):
        if f"rk-{tag}-" not in ln:
            continue
        m = RX.search(ln)
        if m:
            w[m.group(1)] = dict(finish=m.group(2), gen=int(m.group(3)), ttft=int(m.group(4)), tps=float(m.group(5)),
                                 queue=int(m.group(6)), batch=float(m.group(7)))
        if "reply kept" in ln:
            kept += 1
    rs = R["requests"]
    for r in rs:
        r["w"] = w.get(r["rid"])
    assert all(r["w"] for r in rs), "unjoined"
    wall = R["t_end"] - R["t_start"]
    t1 = [r for r in rs if r["turn"] == 1]
    tn = [r for r in rs if r["turn"] >= 2]
    S = {"wall": wall, "agg_tps": sum(r["completion_tokens"] for r in rs) / wall,
         "tps_t1": mean([r["w"]["tps"] for r in t1]), "tps_tn": mean([r["w"]["tps"] for r in tn]),
         "ttft_t1": mean([r["w"]["ttft"] for r in t1]), "ttft_tn": mean([r["w"]["ttft"] for r in tn]),
         "ttft_tn_p50": med([r["w"]["ttft"] for r in tn]),
         "queue": mean([r["w"]["queue"] for r in rs]), "lat_tn": mean([r["elapsed"] for r in tn]),
         "prefill_tn": mean([r["prompt_tokens"] - r["cached_tokens"] for r in tn]),
         "batch": mean([r["w"]["batch"] for r in rs]), "kept": kept, "n": len(rs)}
    out[name] = S
    print(f"| {name} | {wall:.1f} | {S['agg_tps']:.1f} | {mean([r['w']['tps'] for r in rs]):.1f} ({S['tps_t1']:.1f} / {S['tps_tn']:.1f}) | "
          f"{S['ttft_t1']:.0f} / {S['ttft_tn']:.0f} (p50 {S['ttft_tn_p50']:.0f}) | {S['queue']:.0f} | {S['lat_tn']:.2f} | "
          f"{S['prefill_tn']:.0f} | {S['batch']:.2f} | {kept}/{len(rs)} |")
if len(sys.argv) > 2 and "--json" not in sys.argv:
    pass
