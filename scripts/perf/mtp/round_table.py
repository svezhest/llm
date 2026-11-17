"""The round profile of an `ab_mtp.py --prof N` run (QWEN_SPEC_PROF,
src/worker/qwen_worker/rprof.py): per mode x variant x record kind, pooled over
the prompts and rounds of the run (--by-prompt: per prompt too), ms per record
(a burst of plain steps and the plain loop: per step).

  kind     mtpK: the head's rounds of K drafts; mtp0: its plain bursts; dflash; plain:
           the plain arm's bursts; loop: the plain loop's step (the reference; its `pre`
           is the next step's build + encode, queued while the step before runs)
  n tok    records; tokens per record
  wall     p50 / mean
  pre build wait other post betw   means (rprof: host before the first queued work, host
           from there to the sync, the sync, other reads, host after the sync, the tick)
  gpu      the first queued work to the end of the sync (mean)
  exposed  wall - gpu (mean / p50 / p90): none of the record's work queued or waited for
  cpu      the thread's CPU time over the host phases' wall time (< 1: descheduled, the GIL)
  tail     QWEN_SPEC_PROF=2: GPU work left after the walk (p50 of the probe rounds)
  xP       wall and exposed in plain-loop steps (the loop's wall p50, same mode and prompt set)
then the laps of each kind (mean ms per record).

  python3 scripts/perf/mtp/round_table.py <ab_mtp json> [--by-prompt]
"""
import argparse
import json
import sys

SPANS = ("pre", "build", "wait", "other", "post", "between")


def stats(xs):
    xs = sorted(xs)
    if not xs:
        return dict(n=0, mean=0.0, p50=0.0, p90=0.0)
    n = len(xs)
    return dict(n=n, mean=sum(xs) / n, p50=xs[min(n - 1, int(0.5 * n))], p90=xs[min(n - 1, int(0.9 * n))])


def order(k):
    return (0, 0) if k == "loop" else (1, int(k[3:])) if k.startswith("mtp") and k[3:].isdigit() else (2, k)


def table(rows, title, out):
    """rows: ab_mtp speed rows (with "prof") of one group."""
    raw = {}
    for o in rows:
        for key, steps, probe, f, laps in (o.get("prof") or {}).get("raw", []):
            raw.setdefault(key, []).append((probe, f, laps))
    if not raw:
        return
    loop = stats([f["wall"] for p, f, _ in raw.get("loop", []) if not p])["p50"]
    out.append(f"\n== {title} ==")
    out.append(f"{'kind':7s} {'n':>4s} {'tok':>5s} {'wall p50/mean':>14s} " + " ".join(f"{s[:5]:>6s}" for s in SPANS)
               + f" {'gpu':>6s} {'exposed mean/p50/p90':>21s} {'cpu':>5s} {'tail':>5s} {'xP wall/exp':>12s}")
    lapnames = {}
    for key in sorted(raw, key=order):
        rs = raw[key]
        main = [(f, laps) for p, f, laps in rs if not p]
        if not main:
            continue
        st = {k: stats([f[k] for f, _ in main]) for k in main[0][0]}
        host = sum(st[k]["mean"] for k in ("pre", "build", "post", "between"))
        cpu = sum(st["cpu_" + k]["mean"] for k in ("pre", "build", "post", "between"))
        tail = stats([laps.get("tail", 0.0) for p, f, laps in rs if p])
        ex = st["exposed"]
        xp = (f"{st['wall']['mean'] / loop:5.2f}/{ex['mean'] / loop:5.2f}" if loop else "")
        out.append(f"{key:7s} {len(main):4d} {st['tokens']['mean']:5.2f} "
                   f"{st['wall']['p50']:6.1f}/{st['wall']['mean']:6.1f}  "
                   + " ".join(f"{st[s]['mean']:6.2f}" for s in SPANS)
                   + f" {st['gpu']['mean']:6.1f} {ex['mean']:6.2f}/{ex['p50']:6.2f}/{ex['p90']:6.2f}"
                   + f" {cpu / host if host else 0:5.2f} {tail['p50'] if tail['n'] else float('nan'):5.2f} {xp:>12s}")
        names = sorted({n for _, laps in main for n in laps})
        lapnames[key] = {n: sum(laps.get(n, 0.0) for _, laps in main) / len(main) for n in names}
    out.append("laps (mean ms per record):")
    for key in sorted(lapnames, key=order):
        out.append(f"  {key:7s} " + " ".join(f"{n} {v:.2f}" for n, v in
                                              sorted(lapnames[key].items(), key=lambda x: -x[1]) if v >= 0.005))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--by-prompt", action="store_true")
    a = ap.parse_args()
    res = json.load(open(a.json))
    rows = [o for o in res.get("speed", []) if o.get("prof")]
    if not rows:
        sys.exit("no profiled requests (ab_mtp.py --prof 1|2)")
    out = []
    modes = sorted({"sampling" if o["key"].endswith("@s") else "greedy" for o in rows})
    for mode in modes:
        mrows = [o for o in rows if o["key"].endswith("@s") == (mode == "sampling")]
        # the plain loop's rows join every variant's table (the xP reference)
        ref = [o for o in mrows if o["variant"] == "plain"]
        groups = [("all prompts", mrows)]
        if a.by_prompt:
            groups += [(k, [o for o in mrows if o["key"] == k]) for k in sorted({o["key"] for o in mrows})]
        for gname, grows in groups:
            gref = [o for o in ref if o in grows]
            for v in sorted({o["variant"] for o in grows}, key=lambda v: (v != "plain", v)):
                vr = [o for o in grows if o["variant"] == v]
                table(vr + ([] if v == "plain" else gref), f"{mode} / {v} / {gname}", out)
    print("\n".join(out))


if __name__ == "__main__":
    main()
