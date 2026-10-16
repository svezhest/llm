#!/usr/bin/env python3
"""Memory-pressure sampler for agentic runs (stdlib only, macOS).

Every --interval s (10) appends one JSON line to --out:
  swap used/total (sysctl vm.swapusage), the worker's MEM / CMPRS / PAGEINS (top -l1 -pid;
  CMPRS = its bytes held by the compressor: the swap indicator), and vm_stat deltas per
  second (pageins, swapins/outs, decompressions/compressions) + free / compressor pages.
The worker pid is looked up every sample (`pgrep -f '.venv/bin/python3 worker.py'`), so a
worker restart is followed. Stops after --duration-min (0 = until killed).

    python3 scripts/perf/agentic/memsampler.py --out /tmp/perf/mem_r0.jsonl --duration-min 20
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time

UNITS = {"B": 1, "K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}


def sh(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except Exception:  # noqa: BLE001
        return ""


def size(s: str) -> float | None:
    m = re.match(r"([\d.]+)([BKMGT])?\+?-?$", s)
    if not m:
        return None
    return float(m[1]) * UNITS[m[2] or "B"]


def swap() -> dict:
    m = re.search(r"total = ([\d.]+)M\s+used = ([\d.]+)M", sh("sysctl", "vm.swapusage"))
    return {"swap_total_mb": float(m[1]), "swap_used_mb": float(m[2])} if m else {}


def worker_pid(pattern: str) -> int | None:
    out = sh("pgrep", "-f", pattern).split()
    return int(out[0]) if out else None


def top_pid(pid: int) -> dict:
    lines = [ln.split() for ln in sh("top", "-l1", "-pid", str(pid), "-stats", "pid,mem,cmprs,pageins").splitlines()]
    for f in lines:
        if f and f[0] == str(pid) and len(f) >= 4:
            return {"w_mem_b": size(f[1]), "w_cmprs_b": size(f[2]), "w_pageins": int(re.sub(r"\D", "", f[3]) or 0)}
    return {}


VM_KEYS = {"Pages free": "free", "Pages occupied by compressor": "comp_occupied",
           "Pages stored in compressor": "comp_stored", "Pageins": "pageins", "Pageouts": "pageouts",
           "Swapins": "swapins", "Swapouts": "swapouts", "Decompressions": "decompressions",
           "Compressions": "compressions"}


def vmstat() -> dict:
    out, d = sh("vm_stat"), {}
    page = int(m[1]) if (m := re.search(r"page size of (\d+) bytes", out)) else 16384
    for ln in out.splitlines():
        k, _, v = ln.partition(":")
        k = k.strip().strip('"')
        if k in VM_KEYS:
            d[VM_KEYS[k]] = int(v.strip().rstrip("."))
    d["page"] = page
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=10)
    ap.add_argument("--duration-min", type=float, default=0)
    ap.add_argument("--pattern", default=r"\.venv/bin/python3 worker\.py")
    a = ap.parse_args()
    t_end = time.time() + a.duration_min * 60 if a.duration_min > 0 else float("inf")
    prev, t_prev = vmstat(), time.time()
    while time.time() < t_end:
        time.sleep(a.interval)
        now, vm = time.time(), vmstat()
        dt = now - t_prev
        rec = {"t": now, "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))}
        rec.update(swap())
        pid = worker_pid(a.pattern)
        rec["worker_pid"] = pid
        if pid:
            rec.update(top_pid(pid))
        # the biggest memory users (other agents' processes, the browser...) explain outside pressure
        tops = [ln.split(None, 3) for ln in sh("top", "-l1", "-o", "mem", "-n", "6", "-stats", "pid,mem,cmprs,command").splitlines()]
        rec["top_mem"] = [[int(f[0]), f[1], f[2], f[3].strip()] for f in tops if len(f) == 4 and f[0].isdigit()]
        pg = vm["page"]
        rec["free_mb"] = vm.get("free", 0) * pg / 2**20
        rec["compressor_mb"] = vm.get("comp_occupied", 0) * pg / 2**20
        rec["compressed_mb"] = vm.get("comp_stored", 0) * pg / 2**20
        for k in ("pageins", "pageouts", "swapins", "swapouts", "decompressions", "compressions"):
            rec[k + "_ps"] = (vm.get(k, 0) - prev.get(k, 0)) / dt
        prev, t_prev = vm, now
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
