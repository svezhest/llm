#!/usr/bin/env python3
"""ABAB orchestration of benchmark sessions: --repeat passes over the entries in the
given order (ours -> llama.cpp -> stock -> ours -> ...), each a fresh session.py session
(own cooldown + bandwidth probe). Then the drift check: every session's probe vs the
median probe of the whole matrix; sessions more than --probe-tol off are re-run once at
the end (--rerun-off-probe) and marked in index.json.

Entry = SYSTEM:PLAN[:K=V[,K=V]] (K=V go to --set; env for ours as env.K=V).

  python3 scripts/perf/bench27/run_matrix.py --out $PERF_OUT/bench27 --repeat 3 --lease bench \\
      ours-dflash:headline llama-q4_0:headline mlx-vlm:headline ours-plain:headline
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import session as S  # noqa: E402


def parse(entry):
    parts = entry.split(":")
    system, plan = parts[0], parts[1] if len(parts) > 1 else "headline"
    sets, env = {}, {}
    for kv in (parts[2].split(",") if len(parts) > 2 and parts[2] else []):
        k, _, v = kv.partition("=")
        (env if k.startswith("env.") else sets)[k[4:] if k.startswith("env.") else k] = v
    return system, plan, sets, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entries", nargs="+")
    ap.add_argument("--out", default=os.path.join(os.environ.get("PERF_OUT", "/tmp/perf"), "bench27"))
    ap.add_argument("--code", default=S.CODE_DEFAULT)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--cooldown", type=int, default=90)
    ap.add_argument("--probe-s", type=int, default=30)
    ap.add_argument("--probe-tol", type=float, default=0.10)
    ap.add_argument("--rerun-off-probe", action="store_true")
    ap.add_argument("--lease", default="", help="lease name; acquired per session, released between them")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true", help="session.py --force (skip the alive-process check)")
    a = ap.parse_args()
    for e in a.entries:
        if parse(e)[0] not in S.SYSTEMS:
            raise SystemExit(f"unknown system in {e!r}")
    os.makedirs(a.out, exist_ok=True)
    idx_path = os.path.join(a.out, f"index-{time.strftime('%Y%m%d-%H%M%S')}{'-' + a.tag if a.tag else ''}.json")
    index = {"entries": a.entries, "repeat": a.repeat, "code": os.path.abspath(a.code), "sessions": []}

    def one(entry, rep, note=""):
        system, plan, sets, env = parse(entry)
        m = S.run_session(system, plan, a.out, a.code, env, sets, a.port, a.cooldown, a.probe_s, a.lease,
                          a.force, tag=f"r{rep}{note}")
        index["sessions"].append({"entry": entry, "rep": rep, "session": m["session"], "status": m["status"],
                                  "probe": (m.get("probe") or {}).get("sum_gbs"), "summary": m.get("summary"),
                                  "note": note})
        with open(idx_path, "w") as f:
            json.dump(index, f, ensure_ascii=False, indent=1)

    for rep in range(a.repeat):
        for e in a.entries:
            one(e, rep)
    probes = [s["probe"] for s in index["sessions"] if s["probe"]]
    if probes:
        med = statistics.median(probes)
        index["probe_median"] = med
        off = [s for s in index["sessions"] if s["probe"] and abs(s["probe"] / med - 1) > a.probe_tol]
        for s in off:
            s["off_probe"] = True
        print(f"probe median {med} GB/s; {len(off)} session(s) more than {a.probe_tol:.0%} off", flush=True)
        if a.rerun_off_probe:
            for s in list(off):
                one(s["entry"], s["rep"], "-rerun")
    with open(idx_path, "w") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    print("wrote", idx_path)


if __name__ == "__main__":
    main()
