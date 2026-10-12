import os
import json, sys, statistics as st, importlib.util
spec = importlib.util.spec_from_file_location("t", sys.argv[0].replace("pooled.py", "tables.py"))
R = os.environ.get("PERF_OUT", "/tmp/perf") + "/results/"
import re, glob
RX = re.compile(r"id=chatcmpl-(\S+?)(?:-[0-9a-f]{12})? finish=\S+ gen_tokens=\d+ ttft_ms=(\d+) tps=([\d.]+) cached=\d+ peak=([\d.]+)GB.*?queue_ms=(\d+) batch=([\d.]+)")
W = {}
for lp in glob.glob(R + "worker_final*.log"):
    for line in open(lp, errors="replace"):
        m = RX.search(line)
        if m: W[m.group(1)] = dict(peak_gb=float(m.group(4)), queue_ms=int(m.group(5)), batch_mean=float(m.group(6)))
def pct(xs, p):
    xs = sorted(xs); k = (len(xs) - 1) * p / 100; f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)
sel = json.loads(sys.argv[1])  # {"file": [[rep, conc], ...] or "all"}
title = sys.argv[2]
levels = []
for fn, which in sel.items():
    d = json.load(open(R + fn))
    for l in d["levels"]:
        if which == "all" or [l["rep"], l["concurrency"]] in which:
            levels.append(l)
print(f"\n{title}\n")
print("| conc | runs | agg tok/s (runs) | TTFT p50 s | TTFT p95 s | lat p50 s | lat p95 s | decode tok/s/req | batch_mean | queue_ms mean | peak GB |")
print("|" + "---|" * 11)
for n in sorted({l["concurrency"] for l in levels}):
    ls = [l for l in levels if l["concurrency"] == n]; rs = [r for l in ls for r in l["requests_detail"]]
    for r in rs: r.update(W.get(r["request_id"], {}))
    ag = [l["agg_tps"] for l in ls]
    tt = [r["ttft_s"] for r in rs]; la = [r["latency_s"] for r in rs]
    bm = [r["batch_mean"] for r in rs if "batch_mean" in r]; q = [r["queue_ms"] for r in rs if "queue_ms" in r]; pk = [r["peak_gb"] for r in rs if "peak_gb" in r]
    f = lambda xs, fmt: (fmt % xs) if xs is not None else "-"
    print(f"| {n} | {len(ls)} | {st.median(ag):.1f} ({', '.join('%.1f' % a for a in ag)}) | {pct(tt,50):.2f} | {pct(tt,95):.2f} | {pct(la,50):.2f} | {pct(la,95):.2f} | "
          f"{st.mean(r['decode_tps'] for r in rs):.1f} | {f(st.mean(bm) if bm else None, '%.2f')} | {f(st.mean(q) if q else None, '%.0f')} | {f(max(pk) if pk else None, '%.2f')} |")
