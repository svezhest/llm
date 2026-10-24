#!/usr/bin/env python3
"""The disk tier's (QWEN_PROMPT_CACHE_DISK) side of a worker log (stdlib only):
per request MISS / RAM hit / disk hit (an entry loaded for it), TTFT and
queue by class, MISS TTFT by prompt length; the writer's and loader's work
(entries, MB, seconds, skipped and failed writes, load ms, dropped loads).
Joins worker_log.parse (scripts/perf/agentic) by request id.

    python3 scripts/perf/pcdisk/disk_log.py WORKER_LOG [--since HH:MM:SS] [--json out.json]
"""
import argparse
import json
import os
import re
import statistics as st
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agentic"))
import worker_log  # noqa: E402

RX_LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) (\w+)\s+(\S+) (.*)$")
RX_WROTE = re.compile(r"prompt cache disk: wrote an entry of (\d+) tokens \(([\d.]+) MB new, (\d+) "
                      r"blobs shared\) in ([\d.]+) s; (\d+) entries, ([\d.]+) GiB on disk")
RX_LOADED = re.compile(r"id=(\S+) prompt cache disk: entry of (\d+) tokens loaded \(([\d.]+) MB read "
                       r"in (\d+) ms")
RX_LOADING = re.compile(r"id=(\S+) prompt cache disk: loading an entry of (\d+) tokens \(([\d.]+) MB "
                        r"to read\) for a (\d+)-token hit \((\d+) in RAM\)")


def q(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))] if s else float("nan")


def disk_events(path, since, until=None):
    E = {"writes": [], "loads": {}, "loading": {}, "not_written": 0, "write_failed": 0,
         "load_failed": 0, "dropped": 0, "not_kept": 0, "on_disk_max": 0, "entries_max": 0}
    for line in open(path, errors="replace"):
        m = RX_LINE.match(line)
        if not m:
            continue
        ts = worker_log.T(m[1])
        if ts < since or (until is not None and ts > until):
            continue
        msg = m[4]
        if "prompt cache disk" not in msg:
            continue
        if x := RX_WROTE.search(msg):
            E["writes"].append({"tokens": int(x[1]), "mb": float(x[2]), "shared": int(x[3]),
                                "s": float(x[4])})
            E["entries_max"] = max(E["entries_max"], int(x[5]))
            E["on_disk_max"] = max(E["on_disk_max"], float(x[6]))
        elif x := RX_LOADED.search(msg):
            E["loads"][x[1]] = {"tokens": int(x[2]), "mb": float(x[3]), "ms": int(x[4])}
        elif x := RX_LOADING.search(msg):
            E["loading"][x[1]] = {"tokens": int(x[2]), "depth": int(x[4]), "ram": int(x[5])}
        elif "not written" in msg:
            E["not_written"] += 1
        elif "writing an entry" in msg and "failed" in msg:
            E["write_failed"] += 1
        elif "loading an entry" in msg and "failed" in msg:
            E["load_failed"] += 1
        elif "dropped: its request no longer waits" in msg:
            E["dropped"] += 1
        elif "not kept" in msg:
            E["not_kept"] += 1
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--since", default=None, help="HH:MM:SS or 'YYYY-MM-DD HH:MM:SS' (default: all)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    since = datetime.min
    if a.since:
        first = next(worker_log.T(m[1]) for m in map(RX_LINE.match, open(a.log, errors="replace")) if m)
        s = a.since if " " in a.since else f"{first:%Y-%m-%d} {a.since}"
        since = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    P = worker_log.parse(a.log, since)
    E = disk_events(a.log, since)
    done = P["requests"]
    if not done:
        sys.exit("no finished requests")
    lost = 0
    for r in done:
        # a disk hit = an entry loaded for it AND a resume (cached > 0); loaded but resumed from
        # nothing (the entry left RAM again before its turn) is a MISS, counted in `lost`
        loaded = r["id"] in E["loads"]
        if loaded and r["cached"] > 0:
            r["cls"] = "disk"
        else:
            r["cls"] = "ram" if r.get("hit") and r["cached"] > 0 else "miss"
            lost += loaded
        r["w_ttft"] = (r["queue"] + r["ttft"]) / 1000
    out = {"requests": len(done), "loaded_unused": lost}
    print(f"requests {len(done)} (loaded for but resumed from nothing, in miss: {lost})")
    for cls in ("miss", "ram", "disk"):
        xs = [r for r in done if r["cls"] == cls]
        if not xs:
            print(f"{cls}: 0")
            continue
        t = [r["ttft"] / 1000 for r in xs]
        w = [r["w_ttft"] for r in xs]
        pf = [r["prefill"] for r in xs]
        out[cls] = {"n": len(xs), "share": len(xs) / len(done), "ttft_p50": q(t, .5), "ttft_p90": q(t, .9),
                    "arrival_to_first_p50": q(w, .5), "arrival_to_first_p90": q(w, .9),
                    "prefilled_mean": st.mean(pf), "prompt_mean": st.mean(r["prompt"] for r in xs)}
        o = out[cls]
        print(f"{cls}: n={o['n']} ({100 * o['share']:.1f} %), ttft p50/p90 {o['ttft_p50']:.1f}/{o['ttft_p90']:.1f} s,"
              f" arrival->first token p50/p90 {o['arrival_to_first_p50']:.1f}/{o['arrival_to_first_p90']:.1f} s,"
              f" prompt {o['prompt_mean']:.0f}, prefilled {o['prefilled_mean']:.0f} tok/request")
    big = [r for r in done if r["cls"] == "miss" and r["prompt"] >= 10000]
    if big:
        print(f"MISS >= 10K tokens: n={len(big)} ttft p50 {q([r['ttft'] / 1000 for r in big], .5):.1f} s, "
              f"max {max(r['ttft'] for r in big) / 1000:.1f} s at {max(big, key=lambda r: r['ttft'])['prompt']} tokens")
    W = E["writes"]
    out["disk"] = {"writes": len(W), "write_mb": sum(w["mb"] for w in W), "write_s": sum(w["s"] for w in W),
                   "write_s_p50": q([w["s"] for w in W], .5) if W else 0,
                   "loads": len(E["loads"]), "load_mb": sum(x["mb"] for x in E["loads"].values()),
                   "load_ms_p50": q([x["ms"] for x in E["loads"].values()], .5) if E["loads"] else 0,
                   "load_ms_max": max((x["ms"] for x in E["loads"].values()), default=0),
                   **{k: E[k] for k in ("not_written", "write_failed", "load_failed", "dropped", "not_kept",
                                        "on_disk_max", "entries_max")}}
    d = out["disk"]
    print(f"disk: {d['writes']} writes, {d['write_mb']:.0f} MB in {d['write_s']:.1f} s (p50 {d['write_s_p50']:.2f} s"
          f" per entry), {d['not_written']} not written, {d['write_failed']} failed; {d['loads']} loads, "
          f"{d['load_mb']:.0f} MB, p50 {d['load_ms_p50']:.0f} ms (max {d['load_ms_max']}), {d['load_failed']} failed, "
          f"{d['dropped']} dropped, {d['not_kept']} not kept; max {d['entries_max']} entries / "
          f"{d['on_disk_max']:.2f} GiB on disk")
    out["peak_max"] = max(r["peak"] for r in done)
    print(f"peak max {out['peak_max']:.2f} GB")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
