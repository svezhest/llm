import os
import json, sys, statistics as st
def pct(xs, p):
    xs = sorted(xs)
    if not xs: return float('nan')
    k = (len(xs) - 1) * p / 100; f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)
def f(x, n=1): return "-" if x is None or x != x else f"{x:.{n}f}"
import glob, re
RX = re.compile(r"id=chatcmpl-(\S+?)(?:-[0-9a-f]{12})? finish=\S+ gen_tokens=\d+ ttft_ms=(\d+) tps=([\d.]+) cached=\d+ peak=([\d.]+)GB.*?queue_ms=(\d+) batch=([\d.]+)")
W = {}
for lp in glob.glob(os.environ.get("PERF_OUT", "/tmp/perf") + "/results/worker_final*.log"):
    for line in open(lp, errors="replace"):
        m = RX.search(line)
        if m: W[m.group(1)] = dict(w_ttft_ms=int(m.group(2)), w_tps=float(m.group(3)), peak_gb=float(m.group(4)), queue_ms=int(m.group(5)), batch_mean=float(m.group(6)))
for path in sys.argv[1:]:
    d = json.load(open(path)); L = d["levels"]
    for l in L:
        for r in l["requests_detail"]:
            if r.get("request_id") in W:
                r.update(W[r["request_id"]])
        pk = [r["peak_gb"] for r in l["requests_detail"] if "peak_gb" in r]
        if pk: l["peak_gb_max"] = max(pk)
    print(f"\n### {path.split('/')[-1]}  (run_tag {d['run_tag']}, max_tokens {d['config']['max_tokens']}, prompt-tokens {d['config']['prompt_tokens']})\n")
    print("| rep | conc | reqs | prompt tok | wall s | agg tok/s | probe tok/s | agg/probe | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB | errors |")
    print("|" + "---|" * 17)
    for l in L:
        R = l["requests_detail"]
        bm = [r["batch_mean"] for r in R if "batch_mean" in r]; q = [r["queue_ms"] for r in R if "queue_ms" in r]
        pr = l.get("probe_decode_tps")
        print(f"| {l['rep']} | {l['concurrency']} | {l['requests']} | {l['mean_prompt_tokens'] or 0:.0f} | {l['wall_s']:.1f} | {l['agg_tps']:.1f} | {f(pr)} | "
              f"{f(l['agg_tps']/pr if pr else None, 2)} | {l['ttft_p50_s']:.2f} | {l['ttft_p95_s']:.2f} | {l['lat_p50_s']:.2f} | {l['lat_p95_s']:.2f} | "
              f"{l['decode_tps_mean']:.1f} | {f(st.mean(bm) if bm else None, 2)} | {f(st.mean(q) if q else None, 0)} | {f(l['peak_gb_max'], 2)} | {len(l.get('errors') or [])} |")
    conc = sorted({l["concurrency"] for l in L})
    print("\nPooled over repetitions (percentiles over all requests of the level):\n")
    print("| conc | agg tok/s mean (min-max) | agg/probe | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB |")
    print("|" + "---|" * 11)
    for n in conc:
        ls = [l for l in L if l["concurrency"] == n]; R = [r for l in ls for r in l["requests_detail"]]
        ag = [l["agg_tps"] for l in ls]; rat = [l["agg_tps"] / l["probe_decode_tps"] for l in ls if l.get("probe_decode_tps")]
        tt = [r["ttft_s"] for r in R]; la = [r["latency_s"] for r in R]; de = [r["decode_tps"] for r in R]
        bm = [r["batch_mean"] for r in R if "batch_mean" in r]; q = [r["queue_ms"] for r in R if "queue_ms" in r]
        pk = [l["peak_gb_max"] for l in ls if l["peak_gb_max"]]
        print(f"| {n} | {st.mean(ag):.1f} ({min(ag):.1f}-{max(ag):.1f}) | {f(st.mean(rat) if rat else None, 2)} | {pct(tt,50):.2f} | {pct(tt,95):.2f} | "
              f"{pct(la,50):.2f} | {pct(la,95):.2f} | {st.mean(de):.1f} | {f(st.mean(bm) if bm else None, 2)} | {f(st.mean(q) if q else None, 0)} | {f(max(pk) if pk else None, 2)} |")
    if d['config']['prompt_tokens'] > 2000:
        for l in L:
            print("\nprefill requests:", [(r.get("prompt_tokens"), round(r["ttft_s"], 2), r.get("w_ttft_ms"), round(r["decode_tps"], 1), r.get("peak_gb")) for r in l["requests_detail"]])
