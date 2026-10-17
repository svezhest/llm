#!/usr/bin/env python3
"""Slab histogram of a replay run from sched_timer's `sched_slab:` lines (S2a,
docs/perf/2026-10-17/slabs.md), and what merging cuts would save (S2 estimate).

  python3 scripts/perf/agentic/slab_hist.py WORKER_LOG [--prefix chatcmpl-replay-b16a-] \
      [--replay replay.json] [--cost slabs_cost.json] [--json out.json]

Per request (one id): its slabs in order (pos, n tokens, cut type m/s/end, t_ms), hit or miss
(reused > 0), the first slab's plan (every slab end past the resume point) and the message
starts the chain policy keeps (starts[1] = end of the first message block, the last 3).

Policies re-cut each request's prefill region [reused, prompt-1) with the worker's rule
(a cut at i when i - last >= 2048, or i is an allowed message start and i - last >= 256):
  today   every message start allowed (the worker; the logged slabs, as a check)
  tail    today, minus a message start within --tail tokens of the prompt end (the
          generation-prompt tail becomes part of the slab before it)
  policy  only the message starts the chain policy keeps (starts[1], last 3) allowed
  head    today, minus a message start within --tail tokens past the resume point (a hit
          resumed at the reply end: the tool message's start 2 tokens later)
  min     head + tail (no message-cut slab shorter than --tail tokens)
  both    policy + tail;  policy+min  policy + head + tail
Cost of a slab of n tokens at offset p: --cost (slab_cost.py: medians per (offset, N),
linear in N between the measured N, linear in the offset between the measured offsets,
clamped outside), else the run's own median t_ms per N bucket.
"""
import argparse, bisect, collections, json, re, statistics as st, sys

ap = argparse.ArgumentParser()
ap.add_argument("logs", nargs="+")
ap.add_argument("--prefix", default="chatcmpl-replay-")
ap.add_argument("--replay", action="append", default=[], help="replay JSON (window, busy share)")
ap.add_argument("--cost", default=None)
ap.add_argument("--tail", type=int, default=16)
ap.add_argument("--tile", type=int, default=64, help="--cost: tile rows (0 = linear in N)")
ap.add_argument("--json", default=None)
a = ap.parse_args()

RX = re.compile(r"^(\S+ \S+) .*sched_slab: (.*)$")
reqs = collections.OrderedDict()
for path in a.logs:
    for line in open(path, errors="replace"):
        m = RX.match(line)
        if not m:
            continue
        kv = dict(x.split("=", 1) for x in m.group(2).split() if "=" in x)
        rid = kv.get("id", "")
        if not rid.startswith(a.prefix):
            continue
        r = reqs.setdefault(rid, {"id": rid, "slabs": []})
        s = dict(pos=int(kv["pos"]), n=int(kv["n"]), cut=kv["cut"], t=float(kv["t_ms"]), rows=int(kv["rows"]),
                 first=kv["first"] == "1", last=kv["last"] == "1", store=float(kv["store_ms"]),
                 restore=float(kv["restore_ms"]), ckpt=int(kv["ckpt"]))
        r["slabs"].append(s)
        r["prompt"], r["reused"] = int(kv["prompt"]), int(kv["reused"])
        if "plan" in kv:
            r["plan"] = [(int(x[:-1]), x[-1]) for x in kv["plan"].split(",") if x]
            r["marks"] = [int(x) for x in kv.get("marks", "").split(",") if x]
            r["prev"] = int(kv.get("prev", kv["pos"]))
reqs = [r for r in reqs.values() if r["slabs"] and r["slabs"][0]["first"] and r["slabs"][-1]["last"]]
if not reqs:
    sys.exit("no complete requests")
hits = [r for r in reqs if r["reused"] > 0]
miss = [r for r in reqs if r["reused"] == 0]
allslabs = [s for r in reqs for s in r["slabs"]]
pr = print
pr(f"requests with all slabs logged: {len(reqs)} (hits {len(hits)}, misses {len(miss)}); slabs {len(allslabs)} "
   f"= {len(allslabs) / len(reqs):.2f}/request (hits {sum(len(r['slabs']) for r in hits) / max(len(hits), 1):.2f}, "
   f"misses {sum(len(r['slabs']) for r in miss) / max(len(miss), 1):.2f})")
tot_t = sum(s["t"] for s in allslabs)
pr(f"sum of slab time {tot_t / 1e3:.1f} s; tokens {sum(s['n'] for s in allslabs)}")

pr("\nslabs per request:\n\n| slabs | hits | misses |\n|---|---|---|")
c_h = collections.Counter(len(r["slabs"]) for r in hits)
c_m = collections.Counter(len(r["slabs"]) for r in miss)
for k in sorted(set(c_h) | set(c_m)):
    pr(f"| {k} | {c_h.get(k, 0)} | {c_m.get(k, 0)} |")

BK = [(1, 8), (9, 32), (33, 128), (129, 255), (256, 511), (512, 1023), (1024, 2047), (2048, 2048)]
pr("\ntokens per slab (all requests; t = the whole _advance, incl. the decode step queued ahead of it):\n")
pr("| tokens | slabs | % of slabs | of which in hits | ends: m / s / end | median t ms | ms per token | % of slab time |")
pr("|---|---|---|---|---|---|---|---|")
hist = []
for lo, hi in BK:
    xs = [s for s in allslabs if lo <= s["n"] <= hi]
    if not xs:
        continue
    nh = sum(1 for r in hits for s in r["slabs"] if lo <= s["n"] <= hi)
    cc = collections.Counter(s["cut"] for s in xs)
    mt = st.median(s["t"] for s in xs)
    tok = sum(s["n"] for s in xs)
    tt = sum(s["t"] for s in xs)
    hist.append(dict(lo=lo, hi=hi, slabs=len(xs), hits=nh, m=cc["msg"], s=cc["step"], end=cc["end"], t_med=mt,
                     ms_tok=tt / tok, t_share=tt / tot_t))
    pr(f"| {lo}–{hi} | {len(xs)} | {100 * len(xs) / len(allslabs):.0f} | {nh} | {cc['msg']} / {cc['step']} / {cc['end']} | "
       f"{mt:.0f} | {tt / tok:.2f} | {100 * tt / tot_t:.1f} |")

pr("\nhit requests, slab shapes (tokens, binned: tail ≤ 16, short ≤ 128, else the rounded size):")
def b(n):
    return "t" if n <= 16 else "s" if n <= 128 else ("%dK" % round(n / 1024) if n >= 1536 else str(int(round(n / 100.0) * 100)))
shapes = collections.Counter(" + ".join(b(s["n"]) for s in r["slabs"]) for r in hits)
for k, v in shapes.most_common(15):
    pr(f"  {v:4d}  {k}")
two_tail = [r for r in hits if len(r["slabs"]) >= 2 and r["slabs"][-1]["n"] <= a.tail]
pr(f"hit requests whose last slab is ≤ {a.tail} tokens: {len(two_tail)}/{len(hits)}; their last slab: "
   f"median {st.median(r['slabs'][-1]['n'] for r in two_tail) if two_tail else 0} tokens, "
   f"{st.median(r['slabs'][-1]['t'] for r in two_tail) if two_tail else 0:.0f} ms")
sing = [r for r in hits if len(r["slabs"]) == 1]
pr(f"hit requests in one slab: {len(sing)}")
pos_split = collections.Counter()
for r in reqs:
    for i, s in enumerate(r["slabs"]):
        pos_split["only" if len(r["slabs"]) == 1 else "first" if i == 0 else "last" if i == len(r["slabs"]) - 1 else "middle"] += 1
pr("slab position in its request: " + ", ".join(f"{k} {v}" for k, v in pos_split.items()))

# ---------------------------------------------------------------- cost model
if a.cost:
    C = json.load(open(a.cost))
    offs, ns = sorted(C["offsets"]), sorted(C["ns"])
    med = {}
    for o in offs:
        for n in ns:
            med[(o, n)] = st.median(x["s"] * 1e3 for x in C["rows"] if x["off"] == o and x["n"] == n)

    # pfqmm runs 64-row tiles: a slab of n >= 64 rows costs ceil(n / 64) tiles (slab_cost.py: per-tile
    # time flat over N = 128..2048 within ±3 %); below 64 rows MLX's own kernels (measured N, interpolated)
    tile = {o: st.median(med[(o, n)] / -(-n // a.tile) for n in ns if n >= 2 * a.tile) for o in offs}

    def cost_at(o, n):
        if a.tile and n >= a.tile:
            return tile[o] * -(-n // a.tile)
        if n <= ns[0]:
            return med[(o, ns[0])] * 1.0  # below the smallest N: the floor
        if n >= ns[-1]:
            return med[(o, ns[-1])] * n / ns[-1]
        k = bisect.bisect_left(ns, n)
        n0, n1 = ns[k - 1], ns[k]
        return med[(o, n0)] + (med[(o, n1)] - med[(o, n0)]) * (n - n0) / (n1 - n0)

    def cost(p, n):
        if p <= offs[0]:
            return cost_at(offs[0], n)
        if p >= offs[-1]:
            return cost_at(offs[-1], n)
        k = bisect.bisect_left(offs, p)
        o0, o1 = offs[k - 1], offs[k]
        return cost_at(o0, n) + (cost_at(o1, n) - cost_at(o0, n)) * (p - o0) / (o1 - o0)
    model = (f"slab_cost.py (isolated, synced), {a.tile}-row tiles: " + ", ".join(f"{tile[o]:.0f} ms/tile at {o}" for o in offs)
             if a.tile else "slab_cost.py (isolated, synced), linear in N")
else:
    by = collections.defaultdict(list)
    for s in allslabs:
        by[min(BK, key=lambda bk: 0 if bk[0] <= s["n"] <= bk[1] else 1)].append(s["t"] / max(s["n"], 1))

    def cost(p, n):
        return n * st.median(by[min(BK, key=lambda bk: 0 if bk[0] <= n <= bk[1] else 1)] or [0])
    model = "in-run medians"


def recut(r, allow, tail, head=0):
    """Slab ends of r's prefill region under a policy: worker rule over the logged plan."""
    lo, end = r["reused"], r["prompt"] - 1
    plan = r.get("plan")
    if plan is None:
        return None
    # "all": the message starts today's rule cut at (the plan's m ends; dropping only the tail
    # cut cannot make a start that rule skipped eligible); "marks": the chain policy's message
    # starts (starts[1], the last 3) past the resume point, under the same 256 rule
    if allow == "all":
        cand = [c for c, k in plan if k == "m"]
    else:
        cand = [c for c in r.get("marks", []) if lo < c < end]
    cand = sorted(c for c in set(cand) if not (tail and end - c < tail) and not (head and c - lo < head))
    out, last = [], r.get("prev", lo)   # the worker's chain: the last cut at or below the resume point
    pts, j = cand, 0
    while True:
        while j < len(pts) and pts[j] - last < 256:
            j += 1
        nxt_msg = pts[j] if j < len(pts) else None
        nxt = last + 2048 if nxt_msg is None else min(last + 2048, nxt_msg)
        if nxt >= end:
            break
        out.append(nxt)
        last = nxt
        if nxt == nxt_msg:
            j += 1
    return out


def slabs_of(r, cuts):
    pts = [r["reused"]] + cuts + [r["prompt"] - 1]
    return [(pts[i], pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]


pols = {"today": ("all", 0, 0), "tail": ("all", a.tail, 0), "head": ("all", 0, a.tail), "min": ("all", a.tail, a.tail),
        "policy": ("marks", 0, 0), "both": ("marks", a.tail, 0), "policy+min": ("marks", a.tail, a.tail)}
pr(f"\npolicies (cost model: {model}); the 'today' re-cut is checked against the logged slabs:\n")
pr("| policy | requests | slabs | slabs/request (hits / misses) | modelled prefill s | saved s vs today | saved % of the window's busy time |")
pr("|---|---|---|---|---|---|---|")
win = busy = None
if a.replay:
    win = 0.0
    for p in a.replay:
        R = json.load(open(p))
        win += (R.get("deadline") or R["t_end"]) - R["t_start"]
    busy = win * 0.96                         # sched_timer: idle 3-6 %
check = sum(1 for r in reqs if r.get("plan") is not None and
            [n for _, n in slabs_of(r, recut(r, "all", 0))] == [s["n"] for s in r["slabs"]])
withplan = [r for r in reqs if r.get("plan") is not None]
res = {}
base = None
for name, (allow, tail, head) in pols.items():
    tot, ns_, nh, nm = 0.0, 0, 0, 0
    for r in withplan:
        sl = slabs_of(r, recut(r, allow, tail, head))
        tot += sum(cost(p, n) for p, n in sl)
        ns_ += len(sl)
        if r["reused"] > 0:
            nh += len(sl)
        else:
            nm += len(sl)
    if base is None:
        base = tot
    nH = sum(1 for r in withplan if r["reused"] > 0)
    nM = len(withplan) - nH
    res[name] = dict(slabs=ns_, cost_s=tot / 1e3, saved_s=(base - tot) / 1e3)
    pr(f"| {name} | {len(withplan)} | {ns_} | {ns_ / len(withplan):.2f} ({nh / max(nH, 1):.2f} / {nm / max(nM, 1):.2f}) | "
       f"{tot / 1e3:.1f} | {(base - tot) / 1e3:.1f} | "
       f"{100 * (base - tot) / 1e3 / busy if busy else float('nan'):.1f} |")
pr(f"\n'today' re-cut == logged slab sizes for {check}/{len(withplan)} requests; window {win or 0:.0f} s")
if a.json:
    json.dump(dict(hist=hist, policies=res, shapes=shapes.most_common(30),
                   reqs=[{k: v for k, v in r.items()} for r in reqs]), open(a.json, "w"), indent=1)
