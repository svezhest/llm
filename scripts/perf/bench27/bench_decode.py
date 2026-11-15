#!/usr/bin/env python3
"""One streaming OpenAI-compatible client for EVERY HTTP system of the 27B decode
benchmark (our gateway, llama-server, mlx_vlm.server). Stdlib only.

Per request (POST /v1/chat/completions, stream, stream_options.include_usage):
  ttft_s        send -> first chunk carrying text (content or reasoning)
  decode_tps    (n - k) / (t_last - t_first), n = usage.completion_tokens (the server's
                own count); t_first / t_last = arrival of the first / last text chunk; k =
                tokens in the first chunk (HF tokenizer when importable, else 1): a streaming
                detokenizer may send the first token's text only with a later one
                (decode_tps_raw = (n - 1) / span, for reference).
                A finish "stop" counts the EOS in n without a chunk: decode_tps_stop_adj
                uses n - 1 then. decode_tps_chunks = the same over chunk count (equal when
                a server sends one token per chunk -- `tokens_per_chunk` says).
  prefill_tps   (prompt_tokens - cached) / ttft_s (cached from usage
                .prompt_tokens_details.cached_tokens, or llama-server's timings.cache_n)
  text          the answer (identity checks across systems / modes), timings raw.
Modes: greedy (temperature 0) and sample (0.6 / top_p 0.95 / top_k 20, min_p 0, seed per
prompt+round, same seeds for every system). min_p 0 explicitly (llama-server's default is
0.05); penalties per system through --extra (session.py's systems: our worker's presence
default is 1.5 -> 0; mlx_vlm.server gets none, its defaults are off). Thinking off
(chat_template_kwargs + the top-level field mlx_vlm.server and our gateway read).

  python3 bench_decode.py --url http://127.0.0.1:8095 --plan smoke --system ours-plain --out x.json
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import http.client
import json
import os
import statistics
import sys
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prompts as P  # noqa: E402

PLANS = os.path.join(HERE, "plans.json")
MODES = {
    "greedy": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0},
    "sample": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
}
FOLLOW = {"code": "Now add type hints and a short docstring, and handle invalid input explicitly.",
          "en": "Now condense this into three short paragraphs.",
          "ru": "Теперь сократи это до трёх коротких абзацев."}


def seed_of(pid: str, tag: str) -> int:
    return int(hashlib.sha256(f"{pid}/{tag}".encode()).hexdigest()[:8], 16) % 2**31


def body(model, msgs, mode, max_tokens, seed, extra, ignore_eos=False, thinking=False, drop=()):
    b = {"model": model, "messages": msgs, "stream": True, "stream_options": {"include_usage": True},
         "max_tokens": max_tokens,
         "chat_template_kwargs": {"enable_thinking": thinking}, "enable_thinking": thinking, "seed": seed}
    b.update(MODES[mode])
    if ignore_eos:
        b["ignore_eos"] = True
    b.update(extra or {})
    for k in drop:
        b.pop(k, None)
    return b


def stream(url, payload, timeout=3600.0):
    """-> dict of the raw stream: chunk arrival times / sizes, usage, finish, text."""
    u = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    data = json.dumps(payload).encode()
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body=data, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    out = {"status": resp.status, "chunks": [], "usage": None, "finish": None, "text": "", "reasoning": "",
           "timings": None, "error": None, "t0": t0}
    if resp.status != 200:
        out["error"] = resp.read().decode(errors="replace")[:2000]
        conn.close()
        return out
    buf = b""
    while True:
        line = resp.readline()
        if not line:
            break
        t = time.perf_counter()
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        d = line[5:].strip()
        if d == b"[DONE]":
            break
        try:
            ev = json.loads(d)
        except json.JSONDecodeError:
            buf += d
            continue
        if ev.get("error"):
            out["error"] = json.dumps(ev["error"])[:2000]
        if ev.get("usage"):
            out["usage"] = ev["usage"]
        if ev.get("timings"):
            out["timings"] = ev["timings"]
        for ch in ev.get("choices") or []:
            delta = ch.get("delta") or {}
            c = delta.get("content") or ""
            r = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if c or r:
                out["chunks"].append((t - t0, len(c) + len(r)))
                if len(out["chunks"]) == 1:
                    out["first_text"] = r + c
                out["text"] += c
                out["reasoning"] += r
            if ch.get("finish_reason"):
                out["finish"] = ch["finish_reason"]
    conn.close()
    return out


_TOK = []


def first_chunk_tokens(text):
    """Tokens in the first text chunk (a server's detokenizer may hold a token's text back and
    send it with the next: then t_first is the 2nd+ token's arrival and (n - 1) overcounts).
    The HF tokenizer of the 27B when `tokenizers` imports (session.py under the worker venv),
    else None. Counted after the stream, never during it."""
    if not _TOK:
        try:
            import tokenizers
            _TOK.append(tokenizers.Tokenizer.from_file(
                os.path.join(P.weights_dir(), "qwen38-27b-mlx-4bit", "tokenizer.json")))
        except Exception:  # noqa: BLE001 -- optional
            _TOK.append(None)
    if _TOK[0] is None or not text:
        return None
    return len(_TOK[0].encode(text, add_special_tokens=False).ids)


def summarize(raw):
    ch = raw["chunks"]
    u = raw.get("usage") or {}
    tm = raw.get("timings") or {}
    n = u.get("completion_tokens")
    pt = u.get("prompt_tokens")
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is None and "cache_n" in tm:
        cached = tm.get("cache_n")
    r = {"status": raw["status"], "error": raw["error"], "finish": raw["finish"], "completion_tokens": n,
         "prompt_tokens": pt, "cached_tokens": cached, "n_chunks": len(ch), "timings": tm or None,
         "chars": len(raw["text"]) + len(raw["reasoning"]), "reasoning_chars": len(raw["reasoning"])}
    if not ch:
        return r
    t_first, t_last = ch[0][0], ch[-1][0]
    r.update(ttft_s=round(t_first, 4), t_last_s=round(t_last, 4), decode_s=round(t_last - t_first, 4))
    span = t_last - t_first
    if n and span > 0:
        r["decode_tps_raw"] = round((n - 1) / span, 3)
        k = first_chunk_tokens(raw.get("first_text", ""))
        r["first_chunk_tokens"] = k
        # tokens generated after t_first = n - k (k = 1 when a server sends one token per chunk)
        r["decode_tps"] = round((n - (k or 1)) / span, 3)
        r["decode_tps_stop_adj"] = (round((n - (k or 1) - 1) / span, 3) if raw["finish"] == "stop"
                                    else r["decode_tps"])
        r["tokens_per_chunk"] = round(n / len(ch), 3)
    if len(ch) > 1 and span > 0:
        r["decode_tps_chunks"] = round((len(ch) - 1) / span, 3)
    if pt:
        r["prefill_tps"] = round((pt - (cached or 0)) / t_first, 2) if t_first > 0 else None
    gaps = [b[0] - a[0] for a, b in zip(ch, ch[1:])]
    if gaps:
        gs = sorted(gaps)
        r["gap_ms"] = {"p50": round(1e3 * gs[len(gs) // 2], 2), "p90": round(1e3 * gs[int(len(gs) * .9)], 2),
                       "max": round(1e3 * gs[-1], 2)}
    # arrival times (ms from the first chunk) and sizes, for later re-analysis
    r["arrivals_ms"] = [round(1e3 * (t - t_first), 1) for t, _ in ch]
    r["chunk_chars"] = [c for _, c in ch]
    return r


def get_json(url, path, timeout=10.0):
    u = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    conn.request("GET", path)
    resp = conn.getresponse()
    d = resp.read()
    conn.close()
    return resp.status, d


def default_model(url):
    st, d = get_json(url, "/v1/models")
    if st != 200:
        raise SystemExit(f"/v1/models: {st} {d[:200]!r}")
    return json.loads(d)["data"][0]["id"]


def post_json(url, path, obj, timeout=120.0):
    u = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    conn.request("POST", path, body=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    d = resp.read()
    conn.close()
    return resp.status, d


def template_check(url, ids=("code-2k-0", "en-2k-0", "ru-2k-0", "code-8k-0", "ru-8k-0")):
    """llama-server: /apply-template + /tokenize of the manifest's prompts vs the manifest's
    count (HF chat template, the same tokenizer.json our worker uses)."""
    man = P.load_manifest()
    out = []
    for pr in man["prompts"]:
        if pr["id"] not in ids:
            continue
        msgs = P.messages(man, pr)
        st, d = post_json(url, "/apply-template", {"messages": msgs,
                                                   "chat_template_kwargs": {"enable_thinking": False}})
        if st != 200:
            out.append({"prompt": pr["id"], "error": f"apply-template {st}: {d[:300]!r}"})
            continue
        prompt = json.loads(d)["prompt"]
        st, d = post_json(url, "/tokenize", {"content": prompt, "add_special": False, "parse_special": True})
        n = len(json.loads(d)["tokens"]) if st == 200 else None
        out.append({"prompt": pr["id"], "llama_tokens": n, "manifest_tokens": pr["n_tokens"],
                    "equal": n == pr["n_tokens"], "rendered_tail": prompt[-80:]})
    return out


def load_plan(name):
    with open(PLANS) as f:
        plans = json.load(f)
    if name not in plans:
        raise SystemExit(f"unknown plan {name!r}; plans: {', '.join(k for k in plans if not k.startswith('_'))}")
    return plans[name]


def expand(plan, man):
    """-> [(round, cell, prompt)] in run order."""
    ids = [p["id"] for p in man["prompts"]]
    byid = {p["id"]: p for p in man["prompts"]}
    out = []
    for rnd in range(plan.get("rounds", 1)):
        for cell in plan["cells"]:
            for pid in [i for i in ids if fnmatch.fnmatch(i, cell["prompts"])]:
                for mode in cell.get("modes", ["greedy"]):
                    out.append((rnd, dict(cell, mode=mode), byid[pid]))
    return out


def run_plan(url, plan_name, system, model=None, extra=None, log=print, ignore_eos=False, warmup=True,
             only=None, drop=()):
    man = P.load_manifest()
    P.verify_all(man)
    plan = load_plan(plan_name)
    model = model or default_model(url)
    results = []
    if warmup:  # first request of a process: kernel compiles / graph warm-up, not recorded as a cell
        t = time.perf_counter()
        raw = stream(url, body(model, [{"role": "user", "content": "Say hello in five words."}], "greedy", 16,
                               1, extra, drop=drop))
        s = summarize(raw)
        log(f"warmup: {time.perf_counter() - t:.1f} s status {s['status']} {s.get('error') or ''}")
        results.append(dict(s, kind="warmup"))
    for rnd, cell, pr in expand(plan, man):
        if only and not fnmatch.fnmatch(pr["id"], only):
            continue
        tag = f"{rnd:02d}" if cell.get("cold_each_round") else "00"
        msgs = P.messages(man, pr, tag, check=(tag == "00"))
        mt = cell.get("max_tokens", 256)
        seed = seed_of(pr["id"], tag)
        think = bool(cell.get("thinking"))
        raw = stream(url, body(model, msgs, cell["mode"], mt, seed, extra, ignore_eos, think, drop))
        s = summarize(raw)
        rec = dict(kind="cell", system=system, prompt=pr["id"], cls=pr["cls"], ctx=pr["ctx"], thinking=think,
                   expected_prompt_tokens=pr["n_tokens"], mode=cell["mode"], max_tokens=mt, round=rnd, tag=tag,
                   seed=seed, **s, text=raw["text"], reasoning=raw["reasoning"])
        results.append(rec)
        pt_note = "" if s.get("prompt_tokens") in (None, pr["n_tokens"]) else f" [manifest {pr['n_tokens']}]"
        log(f"r{rnd} {pr['id']:11s} {cell['mode']:6s} mt {mt:4d} | prompt {s.get('prompt_tokens')}{pt_note} "
            f"(cached {s.get('cached_tokens')}) ttft {s.get('ttft_s')} s prefill {s.get('prefill_tps')} tok/s | "
            f"n {s.get('completion_tokens')} chunks {s.get('n_chunks')} decode {s.get('decode_tps')} tok/s "
            f"finish {s.get('finish')} {('ERROR ' + s['error']) if s.get('error') else ''}")
        if cell.get("turn2") and raw["text"] and not s.get("error"):
            msgs2 = msgs + [{"role": "assistant", "content": raw["text"]},
                            {"role": "user", "content": FOLLOW[pr["cls"]]}]
            raw2 = stream(url, body(model, msgs2, cell["mode"], cell.get("turn2_max_tokens", 64), seed, extra,
                                    ignore_eos, think, drop))
            s2 = summarize(raw2)
            results.append(dict(kind="turn2", system=system, prompt=pr["id"], cls=pr["cls"], ctx=pr["ctx"],
                                mode=cell["mode"], round=rnd, tag=tag, **s2, text=raw2["text"]))
            log(f"   turn2 {pr['id']:11s} prompt {s2.get('prompt_tokens')} (cached {s2.get('cached_tokens')}) "
                f"ttft {s2.get('ttft_s')} s decode {s2.get('decode_tps')} tok/s")
    return {"model": model, "plan": plan_name, "plan_def": plan, "manifest_commit": man["commit"],
            "results": results}


def table(results):
    """Median decode tok/s per (prompt class, ctx, mode, max_tokens) -- a quick look."""
    cells = {}
    for r in results:
        if r.get("kind") == "cell" and r.get("decode_tps"):
            cells.setdefault((r["cls"], r["ctx"], r["mode"], r["max_tokens"], r.get("thinking", False)),
                             []).append(r["decode_tps"])
    return [{"cls": k[0], "ctx": k[1], "mode": k[2], "max_tokens": k[3], "thinking": k[4], "n": len(v),
             "decode_tps_median": round(statistics.median(v), 2), "min": min(v), "max": max(v)}
            for k, v in sorted(cells.items())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8095")
    ap.add_argument("--plan", default="smoke")
    ap.add_argument("--system", default="unknown")
    ap.add_argument("--model", default="", help="model field (default: the first id of /v1/models)")
    ap.add_argument("--extra", default="", help="JSON merged into every request body")
    ap.add_argument("--drop", default="", help="comma-separated body keys to leave out")
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--only", default="", help="glob over prompt ids")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = run_plan(a.url, a.plan, a.system, a.model or None, json.loads(a.extra) if a.extra else None,
                   ignore_eos=a.ignore_eos, warmup=not a.no_warmup, only=a.only or None,
                   drop=[k for k in a.drop.split(",") if k])
    res["summary"] = table(res["results"])
    for row in res["summary"]:
        print(row)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
