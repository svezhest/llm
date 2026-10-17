#!/usr/bin/env python3
"""Checks a reply_kv_drive.py convs run against its worker log.
  reply_kv_check.py RUN.json WORKER.log TAG
- errors: Traceback / 'batch step failed' / worker_fault / ERROR lines in the log; failed requests
- per request: the worker's reply line (kept / not kept / none), and whether the NEXT turn's
  cached_tokens == this prompt + its fed reply (kept rows: the "N tokens" of the reply line)
- passes: reply-kept lines grouped by _drop_done pass (consecutive lines, same ms)
- foreign Generates in the run's window (requests not from this run)
- cascade: `batch:` tick lines with groups > 0, cascade log lines
"""
import collections, json, re, sys
from datetime import datetime

run, logf, tag = sys.argv[1:4]
R = json.load(open(run))
reqs = R["requests"] if isinstance(R, dict) else R
t0, t1 = (R["t_start"], R["t_end"]) if isinstance(R, dict) else (reqs[0]["t_start"], reqs[-1]["t_end"])
RX = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) (\w+)\s+(\S+) (.*)")
lines = []
for ln in open(logf, errors="replace"):
    m = RX.match(ln)
    if m:
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f").timestamp()
        if t0 - 2 <= ts <= t1 + 5:
            lines.append((ts, m.group(2), m.group(4)))
    elif lines and (ln.startswith("Traceback") or ln.startswith("  File")):
        lines.append((lines[-1][0], "TRACE", ln.strip()))
bad = [l for l in lines if l[1] in ("ERROR", "TRACE") or "batch step failed" in l[2] or "worker_fault" in l[2]]
print(f"log lines in window {len(lines)}; errors/tracebacks/step failures/worker_fault: {len(bad)}")
for l in bad[:10]:
    print("  ", l)
fails = [r for r in reqs if not r.get("ok")]
print(f"requests {len(reqs)}, failed {len(fails)}")
gens = [l for l in lines if l[2].startswith("Generate id=")]
foreign = [l for l in gens if f"rk-{tag}-" not in l[2]]
print(f"Generates in window {len(gens)}, foreign (not rk-{tag}-*) {len(foreign)}")
# reply lines
rid = lambda s: (re.search(r"id=chatcmpl-(rk-[\w-]+?)-[0-9a-f]{12}\b", s) or [None, None])[1]
reply = {}
for ts, lv, msg in lines:
    if "prompt cache reply" in msg:
        k = rid(msg)
        m = re.search(r"reply kept: (\d+) tokens \(\+(\d+) generated\)", msg)
        reply[k] = (ts, ("kept", int(m.group(1)), int(m.group(2))) if m else ("not kept", msg.split("reply not kept:")[-1].strip()[:60]))
bym = {(r["conv"], r["turn"]): r for r in reqs if "conv" in r}
kept = sum(1 for v in reply.values() if v[1][0] == "kept")
print(f"reply lines: kept {kept}, not kept {sum(1 for v in reply.values() if v[1][0] != 'kept')}, "
      f"no line {sum(1 for r in reqs if r['rid'] not in reply)}")
why = collections.Counter(v[1][1] for v in reply.values() if v[1][0] != "kept")
for w, n in why.items():
    print(f"   not kept x{n}: {w}")
ok_cov = bad_cov = 0
rows = []
for (c, t), r in sorted(bym.items()):
    nx = bym.get((c, t + 1))
    v = reply.get(r["rid"])
    st = "none" if v is None else v[1][0]
    exp = v[1][1] if v and v[1][0] == "kept" else None
    if nx is not None and exp is not None:
        good = nx["cached_tokens"] == exp and exp == r["prompt_tokens"] + v[1][2]
        ok_cov += good; bad_cov += not good
        rows.append((r["rid"], st, exp, nx["cached_tokens"], good))
    elif nx is not None:
        rows.append((r["rid"], st, None, nx["cached_tokens"], None))
print(f"next turn resumes at the reply end (cached == prompt + fed reply == entry length): {ok_cov} ok, {bad_cov} not")
for x in rows:
    if x[4] is not True:
        print("   ", x)
# passes
kts = sorted(v[0] for v in reply.values() if v[1][0] == "kept")
passes = []
for ts in kts:
    if passes and ts - passes[-1][-1] < 0.005:
        passes[-1].append(ts)
    else:
        passes.append([ts])
print(f"reply-kept passes (lines < 5 ms apart): sizes {[len(p) for p in passes]}")
groups = [l for l in lines if l[2].startswith("batch:") and re.search(r"groups=[1-9]", l[2])]
casc = [l for l in lines if "cascade" in l[2]]
print(f"batch ticks with groups>0: {len(groups)}; cascade log lines: {len(casc)}")
for l in (groups[:2] + casc[:6]):
    print("   ", datetime.fromtimestamp(l[0]).strftime("%H:%M:%S.%f")[:-3], l[2][:230])
