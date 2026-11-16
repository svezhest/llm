#!/usr/bin/env python3
"""Concurrency benchmark for the OpenAI-compatible gateway (stdlib only).

Closed loop per level: N client threads pull requests from a shared list until it
is empty. The ~--prompt-tokens filler context sits in the system message, the user
message holds only the question. --prefix distinct (default): every request has its
own filler (distinct task + distinct context) so a prefix cache cannot help;
--prefix shared: every request of a level shares one filler (one system prompt, the
worker's prompt cache can reuse it), only the question differs. The probe always
gets its own filler. Streaming, temperature 0, presence 0,
thinking off by default, ignore_eos by default so every answer is max_tokens long.

Per request: TTFT (send -> first content/reasoning token; includes queue wait),
completion tokens (usage of the final chunk), decode tok/s = tokens / (end - first
token), latency. Per level: wall, aggregate tok/s = sum(tokens) / wall, mean/p95
TTFT and latency, mean decode tok/s. --worker-log joins the worker's per-request
log line (peak memory, worker-side ttft/tps) by request id, and the prompt-cache
HIT/MISS/store lines (QWEN_PROMPT_CACHE=1) as hit / reused / store_* per request.

    uv run python scripts/bench_concurrency.py --url http://127.0.0.1:8080 --concurrency 1,2,4,8 --out bench.json

The default --url is port 8090, a measurement gateway (GATEWAY_ADDR=127.0.0.1:8090) kept
apart from the serving one; the gateway's own default is 8080, so pass --url for that.
"""
from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import re
import statistics
import sys
import threading
import time
import urllib.parse
import uuid

TASKS = [
    "Explain how a hash map handles collisions with open addressing versus separate chaining, and when each is the better choice.",
    "Write a Python function that parses an ISO-8601 duration string like 'P3DT4H5M' into seconds, then explain the edge cases.",
    "Describe how TCP congestion control works (slow start, congestion avoidance, fast retransmit) and how BBR differs from CUBIC.",
    "Implement an LRU cache in Rust with O(1) get and put, and walk through the ownership issues you run into.",
    "Explain the difference between optimistic and pessimistic locking in a relational database, with SQL examples for both.",
    "Write a Go worker pool that processes jobs from a channel with a bounded number of goroutines and graceful shutdown.",
    "Explain how garbage collection works in the JVM's G1 collector: regions, remembered sets, mixed collections and pause goals.",
    "Write a SQL query that finds, for each customer, the longest streak of consecutive days with at least one order, and explain it.",
    "Describe how a B-tree index is laid out on disk and why databases prefer it to a binary search tree for range queries.",
    "Implement binary search over a rotated sorted array in TypeScript and prove that your loop invariant holds.",
    "Explain how React's reconciliation and keys work, and what goes wrong when list items use the array index as key.",
    "Write a bash script that rotates log files older than seven days, compresses them, and keeps only the last ten archives.",
    "Explain the CAP theorem and PACELC with concrete examples of systems that choose each trade-off.",
    "Write a C function that reverses a singly linked list in place, then a recursive version, and compare their stack usage.",
    "Explain how HTTPS establishes a session: the TLS 1.3 handshake, key exchange, certificate validation and session resumption.",
    "Implement a token-bucket rate limiter in Python that is safe to use from multiple threads, and explain the timing math.",
    "Explain what a memory-mapped file is, how page faults load it lazily, and when mmap beats read() for large files.",
    "Write a Kotlin coroutine that fetches three URLs concurrently with a timeout and combines the results, and explain cancellation.",
    "Explain how Git stores objects (blobs, trees, commits), what a packfile is, and how a rebase rewrites history.",
    "Describe how a transformer decoder generates text token by token, what the KV cache stores, and why decode is memory-bound.",
    "Write a Python asyncio program that downloads files with bounded concurrency and retries with exponential backoff.",
    "Explain how consistent hashing works and how virtual nodes fix the load imbalance of the naive ring.",
    "Implement Dijkstra's algorithm in C++ with a binary heap and explain why it fails with negative edge weights.",
    "Explain the differences between processes, threads and green threads, and how the OS scheduler treats each.",
]

WORDS = ("alpha beta gamma delta epsilon zeta theta kappa lambda sigma omega "
         "buffer socket thread mutex queue cache index shard replica leader follower "
         "latency throughput window stream frame packet header payload checksum "
         "commit rollback snapshot segment cursor offset partition broker consumer").split()


def filler(seed: str, n_chars: int) -> str:
    """Distinct pseudo-log context of ~n_chars characters, deterministic per seed."""
    import hashlib
    lines, total, i = [], 0, 0
    while total < n_chars:
        h = hashlib.sha256(f"{seed}/{i}".encode()).digest()
        ws = " ".join(WORDS[b % len(WORDS)] for b in h[:8])
        line = f"[{seed}:{i:03d}] svc-{h[8] % 97:02d} {ws} id={h[9:13].hex()} ms={h[13] * 7 % 1000}"
        lines.append(line)
        total += len(line) + 1
        i += 1
    return "\n".join(lines)


SYSTEM = "You are a senior software engineer. Answer thoroughly with code where useful."


def make_messages(idx: int, tag: str, n_chars: int, shared: bool = False) -> list[dict]:
    """The filler goes in the system message (its end is an <|im_start|> cut where the
    worker checkpoints the prompt cache); shared: one filler per `tag`, else per request."""
    task = TASKS[idx % len(TASKS)]
    ctx = filler(f"{tag}-shared" if shared else f"{tag}-{idx}", n_chars)
    sys_msg = SYSTEM
    if n_chars > 0:
        sys_msg += f"\n\nBackground service log for context (unrelated to the question, ignore it):\n{ctx}"
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": f"Question #{idx}: {task} Be detailed and thorough."},
    ]


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class Client:
    def __init__(self, url: str, timeout: float):
        u = urllib.parse.urlparse(url)
        self.host, self.port = u.hostname, u.port or 80
        self.timeout = timeout

    def chat(self, body: dict) -> dict:
        """One streamed request. Returns a record; raises RuntimeError with the body on HTTP errors."""
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        data = json.dumps(body).encode()
        t0 = time.perf_counter()
        conn.request("POST", "/v1/chat/completions", body=data,
                     headers={"content-type": "application/json", "accept": "text/event-stream"})
        resp = conn.getresponse()
        if resp.status != 200:
            err = resp.read().decode(errors="replace")
            conn.close()
            raise RuntimeError(f"HTTP {resp.status}: {err}")
        t_first = None
        chunks = 0
        usage = None
        finish = None
        resp_id = None
        error = None
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            if "error" in ev:
                error = ev["error"]
                continue
            resp_id = ev.get("id", resp_id)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning") or d.get("reasoning_content") or d.get("tool_calls"):
                    chunks += 1
                    if t_first is None:
                        t_first = time.perf_counter()
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
        t_end = time.perf_counter()
        conn.close()
        if error is not None:
            raise RuntimeError(f"stream error frame: {json.dumps(error)}")
        toks = usage["completion_tokens"] if usage else chunks
        ttft = (t_first - t0) if t_first else float("nan")
        dec = (t_end - t_first) if t_first else float("nan")
        return {
            "id": resp_id, "t_start": t0, "t_end": t_end, "ttft_s": ttft, "latency_s": t_end - t0,
            "completion_tokens": toks, "content_chunks": chunks, "usage_present": usage is not None,
            "prompt_tokens": usage["prompt_tokens"] if usage else None, "finish_reason": finish,
            "decode_tps": toks / dec if dec and dec > 0 else float("nan"),
        }


def parse_worker_log(path: str, ids: set[str]) -> dict[str, dict]:
    rx = re.compile(r"id=(\S+) finish=(\S+) gen_tokens=(\d+) ttft_ms=(\d+) tps=([\d.]+) cached=(\d+) "
                    r"peak=([\d.]+)GB active=([\d.]+)GB pool=([\d.]+)GB"
                    r"(?: queue_ms=(\d+) batch=([\d.]+))?")
    # prompt cache (worker.py _Scheduler._prompt_cache)
    rx_hit = re.compile(r"id=(\S+) prompt cache HIT: (\d+)/(\d+) tokens reused \(common prefix (\d+)\), "
                        r"prefilling (\d+)")
    rx_miss = re.compile(r"id=(\S+) prompt cache MISS \(common prefix (\d+), (\d+) entries\)")
    rx_store = re.compile(r"id=(\S+) prompt cache store: (\d+) tokens, (\d+) checkpoints, ([\d.]+) MB "
                          r"\((\d+) entries, ([\d.]+) MB total\)")

    def key(wid):
        # the gateway sends the worker chatcmpl-<request_id>-<12 hex> (a request_id
        # reused by concurrent requests must not cancel the other one)
        return wid if wid in ids else re.sub(r"-[0-9a-f]{12}$", "", wid)

    out = {}
    try:
        with open(path, errors="replace") as f:
            for line in f:
                if "prompt cache" in line:
                    if m := rx_hit.search(line):
                        k = key(m.group(1))
                        if k in ids:
                            out.setdefault(k, {}).update(
                                hit=True, reused=int(m.group(2)), pc_prompt_tokens=int(m.group(3)),
                                common_prefix=int(m.group(4)), prefilled=int(m.group(5)))
                    elif m := rx_miss.search(line):
                        k = key(m.group(1))
                        if k in ids:
                            out.setdefault(k, {}).update(
                                hit=False, reused=0, common_prefix=int(m.group(2)),
                                entries_before=int(m.group(3)))
                    elif m := rx_store.search(line):
                        k = key(m.group(1))
                        if k in ids:
                            out.setdefault(k, {}).update(
                                store_tokens=int(m.group(2)), store_ckpts=int(m.group(3)),
                                store_mb=float(m.group(4)), store_entries=int(m.group(5)),
                                store_total_mb=float(m.group(6)))
                    continue
                m = rx.search(line)
                wid = m and key(m.group(1))
                if m and wid in ids:
                    out.setdefault(wid, {}).update(
                        w_ttft_ms=int(m.group(4)), w_tps=float(m.group(5)), w_gen_tokens=int(m.group(3)), w_cached=int(m.group(6)),
                        peak_gb=float(m.group(7)), active_gb=float(m.group(8)), pool_gb=float(m.group(9)))
                    if m.group(10) is not None:          # batching worker
                        out[wid].update(queue_ms=int(m.group(10)), batch_mean=float(m.group(11)))
    except OSError as e:
        print(f"worker log unreadable: {e}", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8090",
                    help="gateway base URL (default: a measurement gateway on :8090; "
                         "the gateway's own default port is 8080)")
    ap.add_argument("--model", default=None, help="default: first id of /v1/models")
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--requests-per-level", type=int, default=0, help="default max(8, 2*concurrency)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompt-tokens", type=int, default=600, help="target prompt size (approximate)")
    ap.add_argument("--prefix", choices=["distinct", "shared"], default="distinct",
                    help="distinct: every request has its own filler context (no prefix reuse); shared: all "
                         "requests of a level share one filler in the system message (prompt-cache test)")
    ap.add_argument("--thinking", choices=["on", "off"], default="off")
    ap.add_argument("--ignore-eos", choices=["on", "off"], default="on",
                    help="on: every answer is exactly max_tokens (vLLM ignore_eos)")
    ap.add_argument("--repeat", type=int, default=1, help="run the whole sweep this many times")
    ap.add_argument("--alternate", action="store_true",
                    help="reverse the level order on odd repeats (spreads drift: heat, background GPU load)")
    ap.add_argument("--worker-log", default=None, help="worker log to join peak memory per request")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--cooldown", type=float, default=0,
                    help="seconds to idle before each level (an M1 Max decode drops ~34 -> ~23 tok/s "
                         "after minutes of sustained load; cool down to compare levels fairly)")
    ap.add_argument("--probe", type=int, default=0,
                    help="N>0: before each level send one single-stream request of N tokens and record "
                         "its decode tok/s (machine state reference: thermal throttling)")
    ap.add_argument("--out", default=None, help="JSON output path")
    a = ap.parse_args()

    cl = Client(a.url, a.timeout)
    if a.model is None:
        c = http.client.HTTPConnection(cl.host, cl.port, timeout=30)
        c.request("GET", "/v1/models")
        a.model = json.loads(c.getresponse().read())["data"][0]["id"]
        c.close()
    levels = [int(x) for x in a.concurrency.split(",") if x.strip()]
    run_tag = uuid.uuid4().hex[:6]

    def body(msgs, max_tokens, rid):
        b = {"model": a.model, "messages": msgs, "stream": True, "max_tokens": max_tokens,
             "temperature": 0, "presence_penalty": 0, "request_id": rid,
             "stream_options": {"include_usage": True},
             "chat_template_kwargs": {"enable_thinking": a.thinking == "on"}}
        if a.ignore_eos == "on":
            b["ignore_eos"] = True
        return b

    # Calibrate characters per prompt token with one short request (also a warmup).
    # The filler's line labels carry its seed, so the calibration seed has the shape
    # of the real ones ("<tag>-<rep>-<n>-shared" / "<tag>-<rep>-<n>-<seq>"), else the
    # tokens per line differ (a "cal<tag>-9999" seed gave 8154 tokens for 8000).
    probe_chars = a.prompt_tokens * 3
    r = cl.chat(body(make_messages(99, f"{run_tag}-c-0", probe_chars, a.prefix == "shared"),
                     1 if a.no_warmup else 16, f"bench-{run_tag}-cal"))
    if not r["prompt_tokens"]:
        print("no usage in the stream: cannot calibrate", file=sys.stderr)
        return 1
    m0 = make_messages(9999, "x", 1)                 # 1 char: the filler header without the filler
    overhead = (len(m0[0]["content"]) + len(m0[1]["content"])) // 4 + 40  # rough template + header + task
    cpt = probe_chars / max(1, r["prompt_tokens"] - overhead)
    n_chars = max(0, int((a.prompt_tokens - overhead) * cpt))
    print(f"model={a.model} calibration: {r['prompt_tokens']} prompt tokens for {probe_chars} filler chars "
          f"-> {cpt:.2f} chars/token, filler {n_chars} chars", file=sys.stderr)

    results = {"config": {k: v for k, v in vars(a).items()}, "run_tag": run_tag, "levels": []}
    results["config"]["filler_chars"] = n_chars
    stop = threading.Event()
    seq = 0

    for rep in range(a.repeat):
        for n in (levels[::-1] if a.alternate and rep % 2 else levels):
            if stop.is_set():
                break
            total = a.requests_per_level or max(8, 2 * n)
            if a.cooldown > 0:
                print(f"cooldown {a.cooldown:.0f}s before c={n}", file=sys.stderr)
                time.sleep(a.cooldown)
            probe = None
            if a.probe > 0:
                seq += 1
                probe = cl.chat(body(make_messages(seq, f"{run_tag}-probe-{rep}-{n}", n_chars), a.probe,
                                     f"bench-{run_tag}-r{rep}-c{n}-probe"))
                print(f"probe before c={n}: decode {probe['decode_tps']:.1f} tok/s, "
                      f"ttft {probe['ttft_s']:.2f}s", file=sys.stderr)
            jobs = []
            for i in range(total):
                seq += 1
                rid = f"bench-{run_tag}-r{rep}-c{n}-{i}"
                jobs.append((rid, body(make_messages(seq, f"{run_tag}-{rep}-{n}", n_chars, a.prefix == "shared"),
                                       a.max_tokens, rid)))
            lock = threading.Lock()
            recs, errors = [], []

            def worker():
                while not stop.is_set():
                    with lock:
                        if not jobs:
                            return
                        rid, b = jobs.pop(0)
                    try:
                        rec = cl.chat(b)
                        rec["request_id"] = rid
                        with lock:
                            recs.append(rec)
                    except Exception as e:  # noqa: BLE001
                        with lock:
                            errors.append(f"{rid}: {e}")
                        print(f"ERROR {rid}: {e}", file=sys.stderr)
                        stop.set()

            t0 = time.perf_counter()
            ths = [threading.Thread(target=worker, daemon=True) for _ in range(n)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
            wall = time.perf_counter() - t0
            if a.worker_log:
                time.sleep(0.3)  # the worker logs right after the last token
                wl = parse_worker_log(a.worker_log, {f"chatcmpl-{r['request_id']}" for r in recs})
                for rec in recs:
                    rec.update(wl.get(f"chatcmpl-{rec['request_id']}", {}))
            ttft = [r["ttft_s"] for r in recs]
            lat = [r["latency_s"] for r in recs]
            toks = sum(r["completion_tokens"] for r in recs)
            peaks = [r["peak_gb"] for r in recs if "peak_gb" in r]
            pc = [r for r in recs if "hit" in r]
            lv = {
                "rep": rep, "concurrency": n, "requests": len(recs), "errors": errors, "wall_s": wall,
                "completion_tokens": toks, "agg_tps": toks / wall if wall else float("nan"),
                "mean_prompt_tokens": statistics.mean([r["prompt_tokens"] or 0 for r in recs]) if recs else None,
                "ttft_mean_s": statistics.mean(ttft) if ttft else None, "ttft_p50_s": pct(ttft, 50),
                "ttft_p95_s": pct(ttft, 95),
                "lat_mean_s": statistics.mean(lat) if lat else None, "lat_p50_s": pct(lat, 50),
                "lat_p95_s": pct(lat, 95),
                "decode_tps_mean": statistics.mean([r["decode_tps"] for r in recs]) if recs else None,
                "usage_present": all(r["usage_present"] for r in recs),
                "peak_gb_max": max(peaks) if peaks else None,
                "probe_decode_tps": probe["decode_tps"] if probe else None,
                "probe_ttft_s": probe["ttft_s"] if probe else None,
                "pcache_requests": len(pc),
                "pcache_hit_rate": sum(r["hit"] for r in pc) / len(pc) if pc else None,
                "pcache_reused_mean": statistics.mean([r["reused"] for r in pc]) if pc else None,
                "requests_detail": recs,
            }
            results["levels"].append(lv)
            print(f"rep {rep} c={n}: {len(recs)} req, wall {wall:.1f}s, agg {lv['agg_tps']:.1f} tok/s, "
                  f"ttft mean {lv['ttft_mean_s'] or 0:.2f}s, errors {len(errors)}"
                  + (f", pcache hits {sum(r['hit'] for r in pc)}/{len(pc)}" if pc else ""), file=sys.stderr)

    has_pc = any(lv["pcache_requests"] for lv in results["levels"])
    hdr = ("| rep | conc | reqs | prompt tok | wall s | agg tok/s | TTFT mean s | TTFT p95 s | "
           "lat mean s | lat p95 s | decode tok/s/req | probe tok/s | peak GB |"
           + (" pcache hit rate / mean reused tok |" if has_pc else ""))
    print("\n" + hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    for lv in results["levels"]:
        pk = f"{lv['peak_gb_max']:.2f}" if lv["peak_gb_max"] is not None else "-"
        pr = f"{lv['probe_decode_tps']:.1f}" if lv["probe_decode_tps"] is not None else "-"
        print(f"| {lv['rep']} | {lv['concurrency']} | {lv['requests']} | {lv['mean_prompt_tokens'] or 0:.0f} | "
              f"{lv['wall_s']:.1f} | {lv['agg_tps']:.1f} | {lv['ttft_mean_s'] or 0:.2f} | {lv['ttft_p95_s']:.2f} | "
              f"{lv['lat_mean_s'] or 0:.2f} | {lv['lat_p95_s']:.2f} | {lv['decode_tps_mean'] or 0:.1f} | {pr} | {pk} |"
              + ((f" {lv['pcache_hit_rate']:.2f} / {lv['pcache_reused_mean']:.0f} |" if lv["pcache_requests"]
                  else " - |") if has_pc else ""))
    if a.repeat > 1:
        print("\n| conc | reps | agg tok/s mean (min-max) | agg / probe | TTFT mean s | lat mean s | decode tok/s/req |")
        print("|---|---|---|---|---|---|---|")
        for n in levels:
            ls = [lv for lv in results["levels"] if lv["concurrency"] == n and lv["requests"]]
            if not ls:
                continue
            ag = [lv["agg_tps"] for lv in ls]
            rel = [lv["agg_tps"] / lv["probe_decode_tps"] for lv in ls if lv["probe_decode_tps"]]
            rs = f"{statistics.mean(rel):.2f}" if rel else "-"
            print(f"| {n} | {len(ls)} | {statistics.mean(ag):.1f} ({min(ag):.1f}-{max(ag):.1f}) | {rs} | "
                  f"{statistics.mean([lv['ttft_mean_s'] for lv in ls]):.2f} | "
                  f"{statistics.mean([lv['lat_mean_s'] for lv in ls]):.2f} | "
                  f"{statistics.mean([lv['decode_tps_mean'] for lv in ls]):.1f} |")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\nwrote {a.out}", file=sys.stderr)
    return 1 if stop.is_set() else 0


if __name__ == "__main__":
    sys.exit(main())
