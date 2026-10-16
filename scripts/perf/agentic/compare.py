#!/usr/bin/env python3
"""Key numbers of several replay runs side by side (Markdown table).

    python3 scripts/perf/agentic/compare.py docs/perf/2026-10-16/raw/replay_r0.json ...
"""
import json
import os
import sys


def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def f(v, fmt):
    return fmt.format(v) if v is not None else "-"


rows = [
    ("requests/min (window)", ("req_per_min",), "{:.1f}"),
    ("latency p50 / p90 s", None, None),
    ("queue mean / p90 s", None, None),
    ("ttft mean / p90 s", None, None),
    ("decode mean / p90 s", None, None),
    ("hit rate %", ("hit_rate",), "{:.1%}"),
    ("mean reused tok", ("reused_mean",), "{:.0f}"),
    ("prefill tok/s wall", ("prefill_tps_wall",), "{:.0f}"),
    ("decode tok/s wall", ("decode_tps_wall",), "{:.1f}"),
    ("misses", ("worker", "misses"), "{}"),
    ("evictions to admit", ("worker", "events", "evict_admit"), "{}"),
    ("sched: prefill / decode / idle %", None, None),
    ("ms per slab / per decode step", None, None),
    ("step ms at 4 / 6 / 8 rows", None, None),
    ("mean rows (ticks)", ("worker", "mean_rows"), "{:.2f}"),
    ("store MB / checkpoints", None, None),
    ("pcache / active GiB max, peak GB", None, None),
    ("swap GB max, worker CMPRS GB max", None, None),
    ("pageins/s max, decompr/s max (1-min means)", None, None),
]
runs = [json.load(open(p)) for p in sys.argv[1:]]
names = [os.path.basename(p).replace("replay_", "").replace(".json", "") for p in sys.argv[1:]]
print("| | " + " | ".join(names) + " |")
print("|---|" + "---|" * len(names))
for label, path, fmt in rows:
    cells = []
    for r in runs:
        S = r["summary"]
        if path:
            v = g(S, *path)
            cells.append(f(v * (1 if "%" not in fmt else 1), fmt) if v is not None else "-")
            continue
        if label.startswith("latency"):
            cells.append(f"{S['latency']['p50']:.1f} / {S['latency']['p90']:.1f}")
        elif label.startswith(("queue", "ttft", "decode mean")):
            k = {"queue": "queue_s", "ttft": "ttft_s", "deco": "decode_s"}[label[:5].strip()[:5] if label[:5] != "decod" else "deco"]
            x = S.get(k)
            cells.append(f"{x['mean']:.1f} / {x['p90']:.1f}" if x else "-")
        elif label.startswith("sched:"):
            x = g(S, "worker", "sched")
            cells.append(f"{100 * x['advance']:.0f} / {100 * x['iterate']:.0f} / {100 * x['idle']:.0f}" if x else "-")
        elif label.startswith("ms per slab"):
            x = g(S, "worker", "sched")
            cells.append(f"{x['ms_per_slab']:.0f} / {x['ms_per_step']:.0f}" if x else "-")
        elif label.startswith("step ms at"):
            x = g(S, "worker", "sched", "ms_per_step_by_rows") or {}
            cells.append(" / ".join(f"{x[k][1]:.0f}" if k in x else "-" for k in ("4", "6", "8")))
        elif label.startswith("store"):
            x = g(S, "worker", "store")
            cells.append(f"{x['mb_mean']:.0f} / {x['ckpts_mean']:.1f}" if x else "-")
        elif label.startswith("pcache"):
            w = S.get("worker", {})
            cells.append(f"{w.get('pcache_max', 0):.1f} / {w.get('active_max', 0):.1f}, {w.get('peak_max', 0):.1f}")
        elif label.startswith("swap"):
            m = S.get("mem_max")
            cells.append(f"{m['swap_gb']:.1f}, {m['cmprs_gb']:.2f}" if m else "-")
        elif label.startswith("pageins"):
            m = S.get("mem_max")
            cells.append(f"{m['pageins_ps']:.0f}, {m['decomp_ps']:.0f}" if m else "-")
    print(f"| {label} | " + " | ".join(cells) + " |")
