#!/usr/bin/env python3
"""Aggregate benchmark sessions (session.json files under --root, or the sessions an
index-*.json lists) into one table: per (system, class, ctx, mode, max_tokens) the
median over SESSIONS of each session's median decode tok/s, with min-max over sessions,
the number of sessions and requests; TTFT / prefill tok/s of cold requests (cached 0);
turn-2 TTFT; peak footprint; the session probes. Sessions marked off-probe in an index
(> tol from the matrix median) are left out unless --keep-off-probe; --exclude SESSION
leaves out one more session by name (repeatable).

  python3 scripts/perf/bench27/aggregate.py --root $PERF_OUT/bench27 --json agg.json --md agg.md
  python3 scripts/perf/bench27/aggregate.py --index $PERF_OUT/bench9/index-A.json \
      --index $PERF_OUT/bench9/index-B.json --json agg.json     # the sessions of several matrices
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics


def med(v):
    return round(statistics.median(v), 2) if v else None


def load_sessions(root, index, keep_off, exclude=()):
    """index: one index path or a list of them (their sessions in order, each once); none = every
    session.json under root."""
    indexes = [index] if isinstance(index, str) and index else list(index or [])
    if indexes:
        paths = []
        for ix in indexes:
            idx = json.load(open(ix))
            for s in idx["sessions"]:
                p = os.path.join(os.path.dirname(ix), s["session"], "session.json")
                if (keep_off or not s.get("off_probe")) and s["session"] not in exclude and p not in paths:
                    paths.append(p)
    else:
        paths = [p for p in sorted(glob.glob(os.path.join(root, "*", "session.json")))
                 if os.path.basename(os.path.dirname(p)) not in exclude]
    out = []
    for p in paths:
        m = json.load(open(p))
        if m.get("results"):
            out.append(m)
    return out


def aggregate(sessions, label_by="system"):
    cells, extra = {}, {}
    for m in sessions:
        sysname = m[label_by] + (f"[{','.join(f'{k}={v}' for k, v in sorted(m['set'].items()))}]"
                                 if label_by == "system" and m.get("set") else "")
        per = {}
        for r in m["results"]:
            if r.get("kind") == "cell" and r.get("decode_tps"):
                k = (sysname, r["cls"], r["ctx"], r["mode"], r["max_tokens"], bool(r.get("thinking")))
                d = per.setdefault(k, {"tps": [], "ttft": [], "pre": [], "n": []})
                d["tps"].append(r["decode_tps"])
                d["n"].append(r["completion_tokens"])
                if not r.get("cached_tokens"):
                    d["ttft"].append(r["ttft_s"])
                    if r.get("prefill_tps"):
                        d["pre"].append(r["prefill_tps"])
            if r.get("kind") == "turn2" and r.get("ttft_s") is not None:
                e = extra.setdefault((sysname, r["prompt"]), [])
                e.append({"ttft_s": r["ttft_s"], "cached": r.get("cached_tokens"), "prompt": r.get("prompt_tokens")})
        for k, d in per.items():
            c = cells.setdefault(k, {"session_medians": [], "requests": 0, "ttft": [], "prefill": [],
                                     "probes": [], "sessions": [], "peak_gib": []})
            c["session_medians"].append(statistics.median(d["tps"]))
            c["requests"] += len(d["tps"])
            c["ttft"] += d["ttft"]
            c["prefill"] += d["pre"]
            c["probes"].append((m.get("probe") or {}).get("sum_gbs"))
            c["sessions"].append(m["session"])
            c["peak_gib"] += [x.get("peak_gib") for x in m.get("memory") or [] if x.get("peak_gib")]
    rows = []
    for k, c in sorted(cells.items()):
        sm = c["session_medians"]
        rows.append({"system": k[0], "cls": k[1], "ctx": k[2], "mode": k[3], "max_tokens": k[4], "thinking": k[5],
                     "decode_tps": med(sm), "min": round(min(sm), 2), "max": round(max(sm), 2),
                     "sessions": len(sm), "requests": c["requests"], "ttft_cold_s": med(c["ttft"]),
                     "prefill_tps": med(c["prefill"]), "peak_gib": max(c["peak_gib"]) if c["peak_gib"] else None,
                     "probes": c["probes"]})
    turn2 = [{"system": k[0], "prompt": k[1], "ttft_s": med([x["ttft_s"] for x in v]),
              "cached": [x["cached"] for x in v], "prompt_tokens": v[0]["prompt"]} for k, v in sorted(extra.items())]
    return {"rows": rows, "turn2": turn2,
            "probes": {m["session"]: (m.get("probe") or {}).get("sum_gbs") for m in sessions}}


def identity(sessions, a, b):
    """Greedy answers of system a vs system b, per (prompt, max_tokens): equal or the char
    where they first differ (every session's text of a against b's first)."""
    texts = {}
    for m in sessions:
        for r in m["results"]:
            if r.get("kind") == "cell" and r["mode"] == "greedy" and r["system"] in (a, b):
                texts.setdefault((r["system"], r["prompt"], r["max_tokens"]), []).append(r["text"])
    out = []
    for (s, pid, mt), ta in sorted(texts.items()):
        if s != a or (b, pid, mt) not in texts:
            continue
        tb = texts[(b, pid, mt)][0]
        for t in ta:
            first = next((i for i, (x, y) in enumerate(zip(t, tb)) if x != y), None)
            if first is None and len(t) != len(tb):
                first = min(len(t), len(tb))
            out.append({"prompt": pid, "max_tokens": mt, "equal": first is None, "first_diff_char": first})
    return {"a": a, "b": b, "pairs": len(out), "equal": sum(o["equal"] for o in out), "detail": out}


def markdown(agg):
    lines = ["| system | class | ctx | mode | gen | decode tok/s (median of sessions) | min–max | sessions / requests "
             "| cold TTFT s | prefill tok/s | peak GiB |", "|" + "---|" * 11]
    for r in agg["rows"]:
        lines.append(f"| {r['system']} | {r['cls']} | {r['ctx'] // 1024}K | {r['mode']}{' +think' if r['thinking'] else ''}"
                     f" | {r['max_tokens']} | {r['decode_tps']} | {r['min']}–{r['max']} | {r['sessions']} / "
                     f"{r['requests']} | {r['ttft_cold_s']} | {r['prefill_tps']} | {r['peak_gib']} |")
    if agg["turn2"]:
        lines += ["", "| system | prompt | turn-2 TTFT s | cached tokens | prompt tokens |", "|---|---|---|---|---|"]
        for t in agg["turn2"]:
            lines.append(f"| {t['system']} | {t['prompt']} | {t['ttft_s']} | {t['cached']} | {t['prompt_tokens']} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.environ.get("PERF_OUT", "/tmp/perf"), "bench27"))
    ap.add_argument("--index", action="append", default=[],
                    help="index-*.json (repeatable); none: every session under --root")
    ap.add_argument("--keep-off-probe", action="store_true")
    ap.add_argument("--exclude", action="append", default=[], metavar="SESSION",
                    help="leave out this session (its directory name); repeatable")
    ap.add_argument("--json", default="")
    ap.add_argument("--md", default="")
    ap.add_argument("--identity", action="append", default=[], help="A,B: greedy answers of A vs B")
    a = ap.parse_args()
    sessions = load_sessions(a.root, a.index, a.keep_off_probe, a.exclude)
    agg = aggregate(sessions)
    agg["identity"] = [identity(sessions, *x.split(",")) for x in a.identity]
    for i in agg["identity"]:
        print(f"identity {i['a']} vs {i['b']}: {i['equal']}/{i['pairs']} equal")
    md = markdown(agg)
    print(md)
    if a.json:
        json.dump(agg, open(a.json, "w"), indent=1)
    if a.md:
        open(a.md, "w").write(md)


if __name__ == "__main__":
    main()
