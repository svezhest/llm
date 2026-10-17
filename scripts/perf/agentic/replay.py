#!/usr/bin/env python3
"""Reproducible agentic load: replays recorded AppWorld LM calls (stdlib only).

Source: <tasks-dir>/<task>/logs/lm_calls.jsonl (one line per call: {"id", "input": {"model",
"messages", ...}, "output": {... "usage": {"completion_tokens", "prompt_tokens", ...},
"timestamps": {"start", "end"}}}). Tasks are taken in sorted order.

N episodes run concurrently in a closed loop. Episode i starts on task i; it replays the task's
calls in order, sending the RECORDED input.messages (the prompt sequence is the recorded one
whatever the model answers now), non-streaming /v1/chat/completions, temperature 0, seed 100,
max_tokens = the recorded completion length with ignore_eos (the recorded decode load), the
recorded model. No presence_penalty is sent (as AppWorld: the worker default applies). Between
calls it sleeps the recorded gap (the environment's code execution; --gap). When the task's
calls are exhausted the episode takes the next task not yet started (a global round-robin over
the tasks); --rounds R stops after every task ran R times, --duration-min M stops starting new
requests after M minutes (in-flight ones finish).

--own-replies (needs --ignore-eos off): the assistant messages of the history are the model's
own answers of this replay, not the recorded ones (the user / tool turns stay recorded), so
every next prompt continues exactly what the model generated: the load that the worker's reply
KV (QWEN_PROMPT_CACHE_REPLY) serves. Recorded replies diverge from the model's own wherever a
greedy near-tie flips, and an ignore_eos reply holds EOS tokens its text does not.

Per request: wall latency, prompt / completion / cached tokens (usage), episode / task / call
index. With --worker-log the worker's lines are joined by request id (queue_ms, ttft_ms, tps,
hit / reused / prefilled, store MB, peak) and the worker-side analysis (worker_log.py) of the
run's window is printed; with --mem-log the memsampler.py series of the window is summarized.

    python3 scripts/perf/agentic/replay.py --episodes 8 --duration-min 15 \
        --worker-log /tmp/perf/worker_r0.log --mem-log /tmp/perf/mem_r0.jsonl --out /tmp/perf/replay_r0.json
    python3 scripts/perf/agentic/replay.py --summarize /tmp/perf/replay_r0.json   # re-print from a saved run
"""
from __future__ import annotations

import argparse
import glob
import http.client
import io
import json
import os
import statistics as st
import sys
import threading
import time
import urllib.parse
import uuid
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import worker_log  # noqa: E402

DEFAULT_TASKS = os.path.expanduser("~/Projects/appworld/experiments/outputs/ornith9b_react_dev/tasks")


def load_tasks(root: str) -> list[dict]:
    tasks = []
    for f in sorted(glob.glob(os.path.join(root, "*", "logs", "lm_calls.jsonl"))):
        calls, prev_end = [], None
        for line in open(f):
            d = json.loads(line)
            out = d["output"]
            ts = out.get("timestamps") or {}
            s = datetime.fromisoformat(ts["start"]) if ts.get("start") else None
            e = datetime.fromisoformat(ts["end"]) if ts.get("end") else None
            gap = (s - prev_end).total_seconds() if s and prev_end else 0.0
            prev_end = e
            u = out.get("usage") or {}
            calls.append({"model": d["input"]["model"], "messages": d["input"]["messages"],
                          "completion_tokens": int(u.get("completion_tokens") or 1),
                          "rec_prompt_tokens": u.get("prompt_tokens"), "gap_before": max(0.0, gap)})
        if calls:
            tasks.append({"task": f.split(os.sep)[-3], "calls": calls})
    return tasks


def pct(xs, p):
    s = sorted(xs)
    if not s:
        return float("nan")
    k = (len(s) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def mine(recorded, msg):
    """--own-replies: the recorded assistant message with our answer in it: its content and
    its tool calls (the recorded ones dropped when we made none)."""
    m = {k: v for k, v in recorded.items() if k != "tool_calls"}
    m["content"] = msg.get("content") or ""
    if msg.get("tool_calls"):
        m["tool_calls"] = msg["tool_calls"]
    return m


def post(host, port, body, timeout):
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(),
                     headers={"content-type": "application/json"})
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, data
    finally:
        conn.close()


def run(a) -> dict:
    tasks = load_tasks(a.tasks_dir)
    if not tasks:
        raise SystemExit(f"no lm_calls.jsonl under {a.tasks_dir}")
    u = urllib.parse.urlparse(a.url)
    host, port = u.hostname, u.port or 80
    tag = a.tag or uuid.uuid4().hex[:6]
    n_tasks_total = a.rounds * len(tasks) if a.rounds else 10**9
    cursor = {"next": 0}
    lock = threading.Lock()
    recs, errors = [], []
    t_start = time.time()
    deadline = t_start + a.duration_min * 60 if a.duration_min else float("inf")
    stop = threading.Event()

    def next_task():
        with lock:
            k = cursor["next"]
            if k >= n_tasks_total:
                return None, None
            cursor["next"] += 1
            return k, tasks[(k + a.task_offset) % len(tasks)]

    def episode(ep):
        seq = 0
        while not stop.is_set():
            k, task = next_task()
            if task is None:
                return
            own = {}                                  # --own-replies: message index -> our answer
            for ci, call in enumerate(task["calls"]):
                gap = call["gap_before"] if a.gap == "recorded" else float(a.gap)
                gap = min(gap, a.gap_cap)
                if ci and gap > 0:
                    time.sleep(gap)
                if time.time() >= deadline or stop.is_set():
                    return
                seq += 1
                rid = f"replay-{tag}-e{ep}-{seq}"
                msgs = call["messages"]
                if own:
                    msgs = [mine(m, own[i]) if i in own and m.get("role") == "assistant" else m
                            for i, m in enumerate(msgs)]
                body = {"model": a.model or call["model"], "messages": msgs, "stream": False,
                        "max_tokens": call["completion_tokens"], "temperature": 0, "seed": 100, "request_id": rid}
                if a.ignore_eos == "on":
                    body["ignore_eos"] = True
                for attempt in range(4):
                    t0 = time.time()
                    try:
                        status, data = post(host, port, body, a.timeout)
                    except Exception as e:  # noqa: BLE001
                        status, data = -1, str(e).encode()
                    t1 = time.time()
                    if status == 200:
                        break
                    with lock:
                        errors.append({"rid": rid, "status": status, "body": data[:300].decode(errors="replace"), "t": t1})
                    print(f"ERROR {rid} attempt {attempt}: {status} {data[:200]!r}", file=sys.stderr)
                    time.sleep(2)
                else:
                    stop.set()
                    return
                d = json.loads(data)
                if a.own_replies:                     # the next call's history holds it here
                    own[len(call["messages"])] = (d.get("choices") or [{}])[0].get("message") or {}
                us = d.get("usage") or {}
                rec = {"rid": rid, "resp_id": d.get("id"), "episode": ep, "task_seq": k, "task": task["task"],
                       "call": ci, "t_start": t0, "t_end": t1, "latency_s": t1 - t0,
                       "prompt_tokens": us.get("prompt_tokens"), "completion_tokens": us.get("completion_tokens"),
                       "cached_tokens": (us.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                       "rec_completion_tokens": call["completion_tokens"], "rec_prompt_tokens": call["rec_prompt_tokens"],
                       "finish_reason": (d.get("choices") or [{}])[0].get("finish_reason"), "attempts": attempt + 1}
                with lock:
                    recs.append(rec)
                if a.verbose:
                    print(f"{time.strftime('%H:%M:%S')} e{ep} {task['task']}#{ci} prompt={rec['prompt_tokens']} "
                          f"cached={rec['cached_tokens']} gen={rec['completion_tokens']} {rec['latency_s']:.1f}s",
                          file=sys.stderr)

    ths = [threading.Thread(target=episode, args=(i,), daemon=True) for i in range(a.episodes)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    t_end = time.time()
    res = {"config": {k: v for k, v in vars(a).items()}, "tag": tag, "t_start": t_start, "t_end": t_end,
           "deadline": deadline if a.duration_min else None, "tasks": [t["task"] for t in tasks],
           "tasks_started": cursor["next"], "requests": recs, "errors": errors}
    return res


def join_worker(res: dict, path: str):
    since = datetime.fromtimestamp(res["t_start"] - 1)
    until = datetime.fromtimestamp(res["t_end"] + 5)
    P = worker_log.parse(path, since.replace(microsecond=0), until, prefix=f"chatcmpl-replay-{res['tag']}-")
    by = {}
    for r in P["requests"]:
        key = r["id"].rsplit("-", 1)[0][len("chatcmpl-"):]   # strip the gateway's -<12 hex>
        by[key] = r
    for rec in res["requests"]:
        w = by.get(rec["rid"])
        if w:
            rec.update({f"w_{k}": (v.isoformat() if isinstance(v, datetime) else v) for k, v in w.items()
                        if k not in ("id",)})
    return P


def summarize(res: dict, worker_log_path: str | None, mem_log: str | None, out=sys.stdout) -> dict:
    pr = lambda *x: print(*x, file=out)  # noqa: E731
    recs = res["requests"]
    t0 = res["t_start"]
    dl = res.get("deadline") or res["t_end"]
    span = dl - t0
    inwin = [r for r in recs if r["t_end"] <= dl]
    S = {"tag": res["tag"], "episodes": res["config"]["episodes"], "requests": len(recs),
         "requests_in_window": len(inwin), "window_min": span / 60, "errors": len(res["errors"]),
         "tasks_started": res["tasks_started"]}
    if not recs:
        pr("no requests")
        return S
    S["req_per_min"] = len(inwin) / (span / 60)
    lat = [r["latency_s"] for r in recs]
    S["latency"] = {"mean": st.mean(lat), "p50": pct(lat, 50), "p90": pct(lat, 90), "max": max(lat)}
    hits = [r for r in recs if (r["cached_tokens"] or 0) > 0]
    S["hit_rate"] = len(hits) / len(recs)
    S["reused_mean"] = st.mean(r["cached_tokens"] or 0 for r in recs)
    S["prompt_mean"] = st.mean(r["prompt_tokens"] or 0 for r in recs)
    pf = sum((r["prompt_tokens"] or 0) - (r["cached_tokens"] or 0) for r in inwin)
    gen = sum(r["completion_tokens"] or 0 for r in inwin)
    S["prefill_tps_wall"] = pf / span
    S["decode_tps_wall"] = gen / span
    S["gen_mismatch"] = sum(1 for r in recs if r["completion_tokens"] != r["rec_completion_tokens"])
    pr(f"replay {res['tag']}: {S['episodes']} episodes, window {span / 60:.1f} min, {len(inwin)} requests done in it "
       f"({S['req_per_min']:.1f}/min), {len(recs)} total, errors {S['errors']}, tasks started {S['tasks_started']}")
    pr(f"client latency s: mean {S['latency']['mean']:.1f} p50 {S['latency']['p50']:.1f} p90 {S['latency']['p90']:.1f} "
       f"max {S['latency']['max']:.1f}")
    pr(f"prompt mean {S['prompt_mean']:.0f} tok; cache hit rate {100 * S['hit_rate']:.1f}% (cached_tokens > 0), "
       f"mean reused {S['reused_mean']:.0f} tok")
    pr(f"aggregate (window): prefill {S['prefill_tps_wall']:.0f} tok/s, decode {S['decode_tps_wall']:.1f} tok/s; "
       f"completion != recorded: {S['gen_mismatch']}")
    j = [r for r in recs if "w_queue" in r]
    if j:
        for k, f in (("queue_s", lambda r: r["w_queue"] / 1000), ("ttft_s", lambda r: r["w_ttft"] / 1000),
                     ("decode_s", lambda r: r["w_dec_s"]), ("worker_lat_s", lambda r: r["w_lat"])):
            xs = [f(r) for r in j]
            S[k] = {"mean": st.mean(xs), "p50": pct(xs, 50), "p90": pct(xs, 90)}
        pr("split (worker, joined n=%d): " % len(j) + "; ".join(
            f"{k} mean {S[k]['mean']:.1f} p50 {S[k]['p50']:.1f} p90 {S[k]['p90']:.1f}"
            for k in ("queue_s", "ttft_s", "decode_s", "worker_lat_s")))
    if worker_log_path:
        pr("\n--- worker log (this run's window, ticks unfiltered) ---")
        P = worker_log.parse(worker_log_path, datetime.fromtimestamp(t0 - 1).replace(microsecond=0),
                             datetime.fromtimestamp(res["t_end"] + 5), prefix=f"chatcmpl-replay-{res['tag']}-")
        S["worker"] = worker_log.summarize(P, out)
    else:
        P = None
    if mem_log and os.path.exists(mem_log):
        M = [json.loads(x) for x in open(mem_log) if x.strip()]
        M = [m for m in M if t0 - 15 <= m["t"] <= res["t_end"] + 15]
        if M:
            pr("\n--- per minute: memory (memsampler: last sample; rates = mean over the minute) + scheduler "
               "(sched_timer: prefill share of the scheduler thread, decode ms/step, mean rows; worker ticks: pcache) ---")
            pr("| min | swap used GB | worker MEM GB | worker CMPRS GB | free MB | compressor GB | pageins/s | swapins/s | "
               "decompressions/s | prefill % | step ms | rows/step | pcache GiB |")
            pr("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            rows = []
            for mnt in range(int((M[-1]["t"] - t0) // 60) + 1):
                ms = [m for m in M if mnt * 60 <= m["t"] - t0 < (mnt + 1) * 60]
                if not ms:
                    continue
                last = ms[-1]
                row = {"min": mnt, "swap_gb": last.get("swap_used_mb", 0) / 1024, "mem_gb": (last.get("w_mem_b") or 0) / 2**30,
                       "cmprs_gb": (last.get("w_cmprs_b") or 0) / 2**30, "free_mb": last.get("free_mb", 0),
                       "compressor_gb": last.get("compressor_mb", 0) / 1024,
                       "pageins_ps": st.mean(m["pageins_ps"] for m in ms), "swapins_ps": st.mean(m["swapins_ps"] for m in ms),
                       "decomp_ps": st.mean(m["decompressions_ps"] for m in ms)}
                sc = [x for x in (P or {}).get("sched", []) if mnt * 60 <= x["t"].timestamp() - t0 < (mnt + 1) * 60]
                tk = [x for x in (P or {}).get("ticks", []) if mnt * 60 <= x["t"].timestamp() - t0 < (mnt + 1) * 60]
                w = sum(x["wall"] for x in sc)
                stp = sum(x["steps"] for x in sc)
                rs = sum(n * k for x in sc for k, (n, _) in x["by_rows"].items())
                row.update(prefill_frac=sum(x["advance"] for x in sc) / w if w else None,
                           step_ms=1000 * sum(x["iterate"] for x in sc) / stp if stp else None,
                           rows_per_step=rs / stp if stp else None, pcache=max((x["pcache"] for x in tk), default=None))
                rows.append(row)
                f_ = lambda v, f: f.format(v) if v is not None else "-"  # noqa: E731
                pr(f"| {mnt} | {row['swap_gb']:.2f} | {row['mem_gb']:.1f} | {row['cmprs_gb']:.2f} | {row['free_mb']:.0f} | "
                   f"{row['compressor_gb']:.1f} | {row['pageins_ps']:.0f} | {row['swapins_ps']:.0f} | {row['decomp_ps']:.0f} | "
                   f"{f_(row['prefill_frac'] and 100 * row['prefill_frac'], '{:.0f}')} | {f_(row['step_ms'], '{:.0f}')} | "
                   f"{f_(row['rows_per_step'], '{:.1f}')} | {f_(row['pcache'], '{:.1f}')} |")
            S["mem"] = rows
            S["mem_max"] = {"swap_gb": max(r["swap_gb"] for r in rows), "cmprs_gb": max(r["cmprs_gb"] for r in rows),
                            "mem_gb": max(r["mem_gb"] for r in rows), "pageins_ps": max(r["pageins_ps"] for r in rows),
                            "decomp_ps": max(r["decomp_ps"] for r in rows)}
    return S


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--tasks-dir", default=DEFAULT_TASKS)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--duration-min", type=float, default=0, help="stop starting requests after M minutes")
    ap.add_argument("--rounds", type=int, default=0, help="stop after every task ran R times (0 = no limit)")
    ap.add_argument("--task-offset", type=int, default=0, help="episode i starts on task (i + offset) %% n")
    ap.add_argument("--gap", default="recorded", help="'recorded' (the env's time between calls) or seconds")
    ap.add_argument("--gap-cap", type=float, default=10.0)
    ap.add_argument("--model", default=None, help="override the recorded model name")
    ap.add_argument("--ignore-eos", choices=["on", "off"], default="on")
    ap.add_argument("--own-replies", action="store_true",
                    help="history holds the model's own answers (reply KV load); needs --ignore-eos off")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--worker-log", default=None)
    ap.add_argument("--mem-log", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summarize", default=None, help="re-print the summary of a saved run JSON")
    ap.add_argument("--window-min", type=float, default=0, help="--summarize: only the first M minutes")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        res = json.load(open(a.summarize))
        if a.window_min:                              # truncate a saved run to its first M minutes
            dl = res["t_start"] + a.window_min * 60
            res["requests"] = [r for r in res["requests"] if r["t_end"] <= dl]
            res["deadline"] = res["t_end"] = dl
        wl = a.worker_log or res["config"].get("worker_log")
        ml = a.mem_log or res["config"].get("mem_log")
        res["summary"] = summarize(res, wl, ml)
        return 0
    if a.own_replies and a.ignore_eos == "on":
        ap.error("--own-replies needs --ignore-eos off (an ignored EOS is fed but not in the text)")
    if not a.duration_min and not a.rounds:
        a.rounds = 1
    res = run(a)
    if a.worker_log:
        time.sleep(1)
        join_worker(res, a.worker_log)
    buf = io.StringIO()
    res["summary"] = summarize(res, a.worker_log, a.mem_log, buf)
    res["summary_text"] = buf.getvalue()
    print(buf.getvalue())
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)
        print(f"wrote {a.out}", file=sys.stderr)
    return 1 if res["errors"] and len(res["errors"]) > 3 else 0


if __name__ == "__main__":
    sys.exit(main())
