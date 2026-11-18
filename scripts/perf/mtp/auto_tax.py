#!/usr/bin/env python3
"""Where `auto` loses time against the MTP head alone (variant mtp1+q) in one ab_mtp --trace run.

Trace row: (arm, rows in the verify block, drafts kept, t0, ms, fed context, shadow drafts, tokens out,
q1, depth). Every round of `auto` is charged ms - tokens * c, c = the head-alone run's ms per token in
the same cell (its decode wall over its tokens); the charges sum to auto's extra wall time. Categories:
the DFlash rounds a request starts with, later DFlash rounds, head rounds with a shadow draft of DFlash,
head rounds without one (one draft, as the head-alone run; two to four drafts: the head's depth rule),
plain steps (a row of one: a burst), lookup rounds.

  python3 scripts/perf/mtp/auto_tax.py docs/perf/2026-11-18/raw/dfl/93c_ab9_trace.json [cell,cell,...]
"""
import json
import sys

d = json.load(open(sys.argv[1]))
want = sys.argv[2].split(",") if len(sys.argv) > 2 else None
by = {}
for o in d["speed"]:
    by.setdefault((o["key"], o["variant"]), []).append(o)
for key in sorted({k for k, _ in by}):
    if want and key not in want:
        continue
    A, H = by.get((key, "auto"), []), by.get((key, "mtp1+q"), [])
    if not A or not H or any(o.get("trace") is None for o in A):
        continue
    wall = lambda o: 1000.0 * (o["times"][-1] - o["times"][0])
    wa, wh = sum(map(wall, A)), sum(map(wall, H))
    c = wh / sum(len(o["times"]) - 1 for o in H)          # ms per token of the head alone
    cats = {k: [0.0, 0, 0] for k in ("DFlash, request start", "DFlash, later", "head + shadow draft",
                                     "head, 1 draft", "head, 2-4 drafts", "plain steps", "lookup")}
    for o in A:
        start = True
        for r in o["trace"]:
            arm, rows, ms, sh, new = r[0], r[1], r[4], r[6], r[7]
            if arm == "dflash" and rows > 1:
                k = "DFlash, request start" if start else "DFlash, later"
            else:
                start = False
                k = ("lookup" if arm == "lookup" else "plain steps" if rows == 1
                     else "head + shadow draft" if sh else "head, 1 draft" if rows == 2 else "head, 2-4 drafts")
            v = cats[k]
            v[0] += ms - len(new) * c
            v[1] += 1
            v[2] += len(new)
    tot = sum(v[0] for v in cats.values())
    print(f"{key}: {len(A)} / {len(H)} requests; decode wall auto {wa:.0f} ms, head alone {wh:.0f} ms: "
          f"{wa - wh:+.0f} ms ({100 * (wa / wh - 1):+.1f} %); head alone {c:.2f} ms per token; "
          f"charges sum {tot:+.0f} ms")
    for k, (ms, n, t) in cats.items():
        if n:
            print(f"   {k:22s} {ms:+7.0f} ms  {100 * ms / wh:+5.1f} %   rounds {n:4d}  tokens {t:4d}")
