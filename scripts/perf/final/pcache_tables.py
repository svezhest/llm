import json, sys, statistics as st, os
sys.path.insert(0, "/Users/user/Projects/llm/scripts")
from bench_concurrency import pct
S = os.environ.get("PCACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../docs/perf/2026-10-12/raw"))
def f(x, n=1): return "-" if x is None or x != x else f"{x:.{n}f}"
for name in sys.argv[1:]:
    p = f"{S}/pcache_{name}.json"
    if not os.path.exists(p): print(f"\n### {name}: missing"); continue
    d = json.load(open(p))
    for lv in d["levels"]:
        R = sorted(lv["requests_detail"], key=lambda r: r["t_start"])
        t0 = min(r["t_start"] for r in R)
        print(f"\n### {name} (prefix {d['config']['prefix']}, c={lv['concurrency']}, filler {d['config']['filler_chars']} chars)\n")
        print(f"wall {lv['wall_s']:.1f}s, agg {lv['agg_tps']:.1f} tok/s, TTFT p50 {lv['ttft_p50_s']:.2f} / p95 {lv['ttft_p95_s']:.2f} s, "
              f"decode/req mean {lv['decode_tps_mean']:.1f}, probe {f(lv['probe_decode_tps'])} tok/s (ttft {f(lv['probe_ttft_s'],2)} s), "
              f"peak max {f(lv['peak_gb_max'],2)} GB, errors {len(lv['errors'])}, hit rate {f(lv.get('pcache_hit_rate'),2)}, reused mean {f(lv.get('pcache_reused_mean'),0)}\n")
        print("| # (worker admission order) | start s | prompt | TTFT s | w queue s | w ttft s | hit | reused | prefilled | decode tok/s | w tps | batch | peak GB | store MB | entries / total MB |")
        print("|" + "---|" * 15)
        for i, r in enumerate(sorted(R, key=lambda r: r["t_start"] + r.get("queue_ms", 0) / 1000)):
            hit = {True: "HIT", False: "miss"}.get(r.get("hit"), "-")
            ent = f"{r['store_entries']} / {r['store_total_mb']:.0f}" if "store_entries" in r else "-"
            print(f"| {i+1} | {r['t_start']-t0:.2f} | {r['prompt_tokens']} | {r['ttft_s']:.2f} | {f(r.get('queue_ms',float('nan'))/1000,2)} | "
                  f"{f(r.get('w_ttft_ms',float('nan'))/1000,2)} | {hit} | {r.get('reused','-')} | {r.get('prefilled', r['prompt_tokens'] if r.get('hit') is False else '-')} | "
                  f"{r['decode_tps']:.1f} | {f(r.get('w_tps'))} | {f(r.get('batch_mean'),1)} | {f(r.get('peak_gb'),2)} | {f(r.get('store_mb'),0)} | {ent} |")
