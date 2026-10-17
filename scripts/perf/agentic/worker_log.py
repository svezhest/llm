#!/usr/bin/env python3
"""Worker-log analysis for agentic runs (extends docs/perf/2026-10-16/analyze_worker_log.py).

Per request (joined by the worker id `chatcmpl-<request_id>-<12 hex>`): Generate arrival,
prompt tokens, prompt-cache HIT/MISS (reused, prefilled), store (MB, checkpoints, entries,
total MB), finish line (gen tokens, ttft_ms, tps, cached, queue_ms, batch, peak, active).
Per 10 s scheduler tick: rows, waiting, prefilling y/n, steps/s, mean rows, reserved, pcache,
active. Events: evictions to admit / for growth, entries not kept, hits given up, rows cut
by memory, refused Generates.

    python3 scripts/perf/agentic/worker_log.py logs/worker.log "2026-10-16 01:39:21" [until] [--prefix chatcmpl-replay-abc]
"""
from __future__ import annotations

import collections
import re
import statistics as st
import sys
from datetime import datetime

RX_LINE = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) (\w+)\s+qwen\.worker (.*)")
RX_STORE = re.compile(r"prompt cache store: (\d+) tokens, (\d+) checkpoints, ([\d.]+) MB \((\d+) entries, ([\d.]+) MB total\)")


def T(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S,%f")


def q(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))] if s else float("nan")


def parse(path: str, since: datetime, until: datetime | None = None, prefix: str | None = None) -> dict:
    """Requests, ticks and events of the window [since, until]; only ids starting with
    `prefix` when given (ticks and events are not filtered)."""
    P_sched = []
    R, B, E = {}, [], {"evict_admit": 0, "evict_admit_entries": 0, "evict_growth": 0, "not_kept": 0,
                      "gave_up": 0, "cut_length": 0, "refused": 0, "over_cap_warn": 0,
                      # QWEN_PROMPT_CACHE_REPLY: extensions kept / over the headroom / the
                      # prompt entry gone / skipped for reasoning
                      "reply_kept": 0, "reply_not_kept": 0, "reply_entry_gone": 0, "reply_reasoning": 0}
    for line in open(path, errors="replace"):
        m = RX_LINE.match(line)
        if not m:
            continue
        ts = T(m[1])
        if ts < since or (until is not None and ts > until):
            continue
        msg = m[3]
        if msg.startswith("batch: rows="):
            d = dict(kv.split("=", 1) for kv in msg[len("batch: "):].split() if "=" in kv)
            B.append({"t": ts, "rows": int(d["rows"]), "waiting": int(d["waiting"]),
                      "pf": d["prefilling"] != "-", "sps": float(d["steps/s"]), "mean_rows": float(d["mean_rows"]),
                      "reserved": float(d["reserved"].split("/")[0]), "pcache": float(d["pcache"].rstrip("GiB")),
                      "active": float(d["active"].rstrip("GiB"))})
            continue
        if msg.startswith("sched_time:"):
            d = dict(kv.split("=", 1) for kv in msg[len("sched_time: "):].split() if "=" in kv)
            st_ = {k: float(d[k]) for k in ("wall", "advance", "iterate", "admit", "idle")}
            st_.update(slabs=int(d["slabs"]), steps=int(d["steps"]), t=ts, by_rows={})
            for x in filter(None, d.get("by_rows", "").split(",")):
                k, v = x.split(":")
                n, sec = v.split("/")
                st_["by_rows"][int(k)] = (int(n), float(sec))
            P_sched.append(st_)
            continue
        if msg.startswith("prompt cache reply not kept: over the batch's headroom"):
            E["reply_not_kept"] += 1                  # (no id: logged by the store itself)
            continue
        if msg.startswith("prompt cache:"):
            if "evicted to admit" in msg:
                E["evict_admit"] += 1
                E["evict_admit_entries"] += int(re.search(r"(\d+) entries", msg)[1])
            elif "evicted for the growth" in msg:
                E["evict_growth"] += 1
            elif "not kept" in msg:
                E["not_kept"] += 1
            elif "gives up" in msg:
                E["gave_up"] += 1
            elif "newest entry alone" in msg:
                E["over_cap_warn"] += 1
            continue
        if msg.startswith("Generate id="):
            rid = msg.split()[1][3:]
            if "refused" in msg:
                E["refused"] += 1
                continue
            if prefix and not rid.startswith(prefix):
                continue
            R.setdefault(rid, {"id": rid})["t0"] = ts
            continue
        mm = re.match(r"id=(\S+) (.*)", msg)
        if not mm:
            continue
        rid, rest = mm[1], mm[2]
        if "ended at" in rest and "headroom" in rest:
            E["cut_length"] += 1
        if rest.startswith("prompt cache reply kept:"):
            E["reply_kept"] += 1
        elif rest.startswith("prompt cache reply not kept: its prompt entry is gone"):
            E["reply_entry_gone"] += 1
        elif rest.startswith("prompt cache reply not kept: reasoning"):
            E["reply_reasoning"] += 1
        if prefix and not rid.startswith(prefix):
            continue
        r = R.setdefault(rid, {"id": rid})
        if rest.startswith("prompt_tokens="):
            r["prompt"] = int(re.match(r"prompt_tokens=(\d+)", rest)[1])
        elif "prompt cache HIT" in rest:
            x = re.search(r"HIT: (\d+)/(\d+).*prefilling (\d+)", rest)
            r.update(reused=int(x[1]), prefill=int(x[3]), hit=True, t_hit=ts)
        elif "prompt cache MISS" in rest:
            r.update(hit=False, reused=0, t_hit=ts)
        elif x := RX_STORE.search(rest):
            r.update(t_store=ts, store_tokens=int(x[1]), store_ckpts=int(x[2]), store_mb=float(x[3]),
                     store_entries=int(x[4]), store_total_mb=float(x[5]))
        elif "finish=" in rest:
            d = dict(kv.split("=", 1) for kv in rest.split() if "=" in kv)
            r.update(t1=ts, gen=int(d["gen_tokens"]), ttft=int(d["ttft_ms"]), tps=float(d["tps"]),
                     cached=int(d["cached"]), queue=int(d.get("queue_ms", 0)), batch=float(d.get("batch", 1)),
                     peak=float(d["peak"].rstrip("GB")), active=float(d["active"].rstrip("GB")), finish=d["finish"])
    done = [r for r in R.values() if "t1" in r and "t0" in r and "prompt" in r]
    for r in done:
        r.setdefault("prefill", r["prompt"] - r["cached"])
        r["lat"] = (r["t1"] - r["t0"]).total_seconds()
        r["dec_s"] = r["gen"] / r["tps"] if r["tps"] > 0 else 0.0
    return {"requests": done, "ticks": B, "events": E, "sched": P_sched}


def summarize(P: dict, out=sys.stdout) -> dict:
    """Prints the analysis; returns the key numbers."""
    done, B, E = P["requests"], P["ticks"], P["events"]
    pr = lambda *a: print(*a, file=out)  # noqa: E731
    res = {"events": E}
    if not done:
        pr("no finished requests in the window")
        return res

    def summ(name, xs, fmt="{:.1f}"):
        if not xs:
            pr(f"{name}: -")
            return
        pr(f"{name}: n={len(xs)} mean={fmt.format(st.mean(xs))} p50={fmt.format(q(xs, .5))} "
           f"p90={fmt.format(q(xs, .9))} max={fmt.format(max(xs))}")
        res[name.split()[0] + "_" + "_".join(name.split()[1:2])] = {"mean": st.mean(xs), "p50": q(xs, .5), "p90": q(xs, .9)}

    t_first = min(r["t0"] for r in done)
    t_last = max(r["t1"] for r in done)
    wall = (t_last - t_first).total_seconds()
    Ptok = sum(r["prompt"] for r in done)
    C = sum(r["cached"] for r in done)
    G = sum(r["gen"] for r in done)
    PF = sum(r["prefill"] for r in done)
    hits = sum(1 for r in done if r.get("hit"))
    misses = sum(1 for r in done if r.get("hit") is False)
    pr(f"window {t_first:%H:%M:%S}–{t_last:%H:%M:%S} = {wall / 60:.1f} min, requests={len(done)} "
       f"({len(done) / (wall / 60):.1f}/min)")
    pr(f"prompt tokens {Ptok} (cached {C} = {100 * C / Ptok:.1f}%), prefilled {PF}, generated {G}")
    pr(f"aggregate: prefill {PF / wall:.0f} tok/s wall, decode {G / wall:.1f} tok/s wall")
    pr(f"cache: hits {hits}, misses {misses}; events {E}")
    pr(f"finish reasons: { {k: sum(1 for r in done if r['finish'] == k) for k in set(r['finish'] for r in done)} }"
       f"  peak max {max(r['peak'] for r in done):.2f} GB")
    res.update(wall_s=wall, requests=len(done), req_per_min=len(done) / (wall / 60), prefill_tps=PF / wall,
               decode_tps=G / wall, hits=hits, misses=misses, cached_frac=C / Ptok, peak_max=max(r["peak"] for r in done))
    summ("prompt tokens", [r["prompt"] for r in done], "{:.0f}")
    summ("prefilled tokens/request", [r["prefill"] for r in done], "{:.0f}")
    summ("gen tokens/request", [r["gen"] for r in done], "{:.0f}")
    summ("latency s (Generate→finish)", [r["lat"] for r in done])
    summ("queue s", [r["queue"] / 1000 for r in done])
    summ("ttft s", [r["ttft"] / 1000 for r in done])
    summ("decode s (gen/tps)", [r["dec_s"] for r in done])
    summ("decode tok/s per request", [r["tps"] for r in done])
    tot = sum(r["lat"] for r in done)
    shares = {k: sum(f(r) for r in done) / tot for k, f in
              (("queue", lambda r: r["queue"] / 1000), ("ttft", lambda r: r["ttft"] / 1000), ("decode", lambda r: r["dec_s"]))}
    res["latency_share"] = shares
    pr("latency share: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in shares.items()))
    big = [r for r in done if r["prefill"] >= 1500 and r["ttft"] > 0]
    if big:
        res["prefill_rate_big"] = st.mean(r["prefill"] / (r["ttft"] / 1000) for r in big)
        pr(f"prefill rate on requests with >=1500 new tokens: {res['prefill_rate_big']:.0f} tok/s (prefill/ttft, n={len(big)})")
    stores = [r for r in done if "store_mb" in r]
    if stores:
        res["store"] = {"n": len(stores), "mb_mean": st.mean(r["store_mb"] for r in stores),
                        "tokens_mean": st.mean(r["store_tokens"] for r in stores),
                        "ckpts_mean": st.mean(r["store_ckpts"] for r in stores),
                        "entries_max": max(r["store_entries"] for r in stores),
                        "total_mb_max": max(r["store_total_mb"] for r in stores)}
        s = res["store"]
        pr(f"stores: n={s['n']} mean {s['mb_mean']:.0f} MB for {s['tokens_mean']:.0f} tokens, {s['ckpts_mean']:.1f} "
           f"checkpoints; entries max {s['entries_max']}, total max {s['total_mb_max']:.0f} MB")
    pr("\ndecode tok/s per request by prompt length:")
    for lo, hi in [(0, 5000), (5000, 10000), (10000, 15000), (15000, 20000), (20000, 30000), (30000, 10**6)]:
        xs = [r["tps"] for r in done if lo <= r["prompt"] < hi]
        if xs:
            pr(f"  {lo // 1000:>2}–{hi // 1000 if hi < 10**6 else '∞'}K: n={len(xs)} mean={st.mean(xs):.1f} p50={q(xs, .5):.1f}")
    # GPU time split: prefill is serialized; [HIT/MISS line, store line] of a request brackets
    # its prefill (one slab for most hit suffixes, plus decode steps between slabs otherwise)
    iv = sorted((r["t_hit"], r["t_store"]) for r in done if "t_hit" in r and "t_store" in r)
    busy, cur = 0.0, None
    for a_, b_ in iv:
        if cur and a_ <= cur[1]:
            cur[1] = max(cur[1], b_)
        else:
            if cur:
                busy += (cur[1] - cur[0]).total_seconds()
            cur = [a_, b_]
    if cur:
        busy += (cur[1] - cur[0]).total_seconds()
    pf_tok = sum(r["prefill"] for r in done if "t_hit" in r and "t_store" in r)
    res["prefill_busy_frac"] = busy / wall if wall else 0
    res["prefill_rate_in_busy"] = pf_tok / busy if busy else 0
    pr(f"prefill busy (union of HIT/MISS→store intervals): {busy:.0f} s = {100 * busy / wall:.0f}% of the window, "
       f"{res['prefill_rate_in_busy']:.0f} tok/s inside it")
    SC = P.get("sched") or []
    if SC:
        W_ = sum(x["wall"] for x in SC)
        tot = {k: sum(x[k] for x in SC) for k in ("advance", "iterate", "admit", "idle", "slabs", "steps")}
        res["sched"] = {k: (v / W_ if k in ("advance", "iterate", "admit", "idle") else v) for k, v in tot.items()}
        res["sched"]["wall"] = W_
        res["sched"]["ms_per_slab"] = 1000 * tot["advance"] / tot["slabs"] if tot["slabs"] else 0
        res["sched"]["ms_per_step"] = 1000 * tot["iterate"] / tot["steps"] if tot["steps"] else 0
        pr(f"scheduler thread time (sched_timer, {W_:.0f} s): prefill (_advance) {100 * tot['advance'] / W_:.0f}%, "
           f"decode (_iterate) {100 * tot['iterate'] / W_:.0f}%, admit {100 * tot['admit'] / W_:.1f}%, "
           f"idle {100 * tot['idle'] / W_:.0f}%; {tot['slabs']} slabs ({res['sched']['ms_per_slab']:.0f} ms each), "
           f"{tot['steps']} steps ({res['sched']['ms_per_step']:.0f} ms each)")
        br = collections.defaultdict(lambda: [0, 0.0])
        for x in SC:
            for k, (n, sec) in x["by_rows"].items():
                br[k][0] += n
                br[k][1] += sec
        res["sched"]["ms_per_step_by_rows"] = {k: (v[0], 1000 * v[1] / v[0]) for k, v in sorted(br.items()) if v[0]}
        pr("  decode step ms by rows: " + ", ".join(f"{k}: {ms:.0f} ms (n={n})"
                                                    for k, (n, ms) in res["sched"]["ms_per_step_by_rows"].items()))
    if B:
        steps = sum(b["sps"] * 10 for b in B)
        rsteps = sum(b["sps"] * 10 * b["mean_rows"] for b in B)
        span = len(B) * 10
        res["decode_steps"] = steps
        res["step_ms_outside_prefill"] = 1000 * max(span - busy, 0) / steps if steps else 0
        res["rows_per_step"] = rsteps / steps if steps else 0
        pr(f"decode steps {steps:.0f} (mean {res['rows_per_step']:.2f} rows/step) in {span} s of ticks: "
           f"{res['step_ms_outside_prefill']:.0f} ms/step outside prefill (upper bound: idle counted)")
        pr("\nscheduler (10 s ticks): steps/s by rows, with/without a prefill in progress:")
        res["steps_by_rows"] = {}
        for rows in range(0, 17):
            for pf in (False, True):
                xs = [b["sps"] for b in B if b["rows"] == rows and b["pf"] == pf]
                if xs:
                    res["steps_by_rows"][f"{rows}{'y' if pf else 'n'}"] = (len(xs), st.mean(xs))
                    pr(f"  rows={rows:>2} prefilling={'y' if pf else 'n'}: ticks={len(xs)} steps/s mean={st.mean(xs):.2f}")
        busy = [b for b in B if b["rows"] > 0]
        res.update(ticks_pf=sum(b["pf"] for b in B) / len(B), ticks_wait=sum(b["waiting"] > 0 for b in B) / len(B),
                   mean_rows=st.mean(b["rows"] for b in B), pcache_max=max(b["pcache"] for b in B),
                   active_max=max(b["active"] for b in B), reserved_max=max(b["reserved"] for b in B),
                   row_steps_per_s=st.mean(b["sps"] * b["mean_rows"] for b in busy) if busy else 0)
        pr(f"ticks with prefill in progress: {100 * res['ticks_pf']:.0f}%;  waiting>0: {100 * res['ticks_wait']:.0f}%;  "
           f"mean rows {res['mean_rows']:.2f}; row-steps/s {res['row_steps_per_s']:.1f}")
        pr(f"pcache GiB max {res['pcache_max']:.2f}, active GiB max {res['active_max']:.2f}, "
           f"reserved GiB max {res['reserved_max']:.2f}")
    return res


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--prefix")]
    pre = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--prefix=")), None)
    since = datetime.strptime(args[1], "%Y-%m-%d %H:%M:%S")
    until = datetime.strptime(args[2], "%Y-%m-%d %H:%M:%S") if len(args) > 2 else None
    summarize(parse(args[0], since, until, pre))
