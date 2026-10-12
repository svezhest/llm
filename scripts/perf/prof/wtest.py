#!/usr/bin/env python3
"""GPU test client for the batching worker: talks gRPC to the worker directly
(no gateway, no MLX here). Run with the worker venv and the pb2 modules:

  cd src/worker && PYTHONPATH=. .venv/bin/python ../../scripts/perf/prof/wtest.py <suite> [opts]

Suites:
  corpus  --out F        identity corpus, sequential (byte-identity reference / candidate)
  compare A B            byte-compare two corpus files
  conc    --n N --ref F  N distinct greedy prompts at once vs their sequential outputs (F from `seq`)
  seq     --n N --out F  the same N prompts one by one (reference for conc)
  cancel  --ref F        8 streams, 3 cancelled mid-way (2 Cancel RPC, 1 dropped stream)
  stops | mixedmax | think | grammar   feature checks inside a batch (PASS/FAIL)
  pcache                 8 conversations over one ~7.5K-token system prompt (QWEN_PROMPT_CACHE=1)
  memory  --n 8          8 rows with ~8K-token prompts: peak memory
"""
import argparse
import json
import sys
import threading
import time
import uuid

import grpc

import worker_pb2 as pb
import worker_pb2_grpc as pb_grpc

TASKS = [
    "Write a Python function that parses an ISO-8601 duration string like 'P3DT4H5M' into seconds, then explain the edge cases.",
    "Implement an LRU cache in Rust with O(1) get and put, and walk through the ownership issues you run into.",
    "Write a Go worker pool that processes jobs from a channel with a bounded number of goroutines and graceful shutdown.",
    "Write a SQL query that finds, for each customer, the longest streak of consecutive days with at least one order, and explain it.",
    "Implement binary search over a rotated sorted array in TypeScript and prove that your loop invariant holds.",
    "Write a bash script that rotates log files older than seven days, compresses them, and keeps only the last ten archives.",
    "Write a C function that reverses a singly linked list in place, then a recursive version, and compare their stack usage.",
    "Implement a token-bucket rate limiter in Python that is safe to use from multiple threads, and explain the timing math.",
    "Write a Kotlin coroutine that fetches three URLs concurrently with a timeout and combines the results, and explain cancellation.",
    "Write a Python asyncio program that downloads files with bounded concurrency and retries with exponential backoff.",
    "Implement Dijkstra's algorithm in C++ with a binary heap and explain why it fails with negative edge weights.",
    "Explain how a hash map handles collisions with open addressing versus separate chaining, and when each is the better choice.",
    "Describe how TCP congestion control works (slow start, congestion avoidance, fast retransmit) and how BBR differs from CUBIC.",
    "Explain the difference between optimistic and pessimistic locking in a relational database, with SQL examples for both.",
    "Explain how garbage collection works in the JVM's G1 collector: regions, remembered sets, mixed collections and pause goals.",
    "Describe how a B-tree index is laid out on disk and why databases prefer it to a binary search tree for range queries.",
]
OFF = json.dumps({"enable_thinking": False})
ON = json.dumps({"enable_thinking": True})


def stub(addr):
    ch = grpc.insecure_channel(addr, options=[("grpc.max_receive_message_length", 64 << 20)])
    return pb_grpc.WorkerStub(ch)


def msgs(text, system=None):
    out = [pb.Message(role="system", content=system)] if system else []
    return out + [pb.Message(role="user", content=text)]


def gen(st, messages, rid=None, on_event=None, **params):
    """-> dict(text, reasoning, finish, usage..., ttft_s, wall_s, error)."""
    rid = rid or uuid.uuid4().hex[:12]
    req = pb.Request(id=rid, messages=messages, params=pb.Params(**params))
    t0, first = time.time(), None
    res = {"id": rid, "text": "", "reasoning": "", "finish": None, "error": None, "calls": []}
    call = st.Generate(req)
    try:
        for ev in call:
            k = ev.WhichOneof("kind")
            if k == "token":
                first = first or time.time()
                res["reasoning" if ev.token.thinking else "text"] += ev.token.text
            elif k == "done":
                u = ev.done.usage
                res.update(finish=ev.done.finish, gen_tokens=u.gen_tokens, prompt_tokens=u.prompt_tokens,
                           reasoning_tokens=u.reasoning_tokens, tps=round(u.tps, 2), ttft_ms=u.ttft_ms,
                           peak_gb=round(u.peak_gb, 2), queue_ms=u.queue_ms,
                           batch_mean=round(u.batch_mean, 2),
                           calls=[(c.name, c.arguments) for c in ev.done.tool_calls])
            elif k == "error":
                res["error"] = (ev.error.kind, ev.error.msg)
            if on_event is not None and on_event(res, call) is False:
                call.cancel()
                res["finish"] = res["finish"] or "client-dropped"
                break
    except grpc.RpcError as e:
        if e.code() != grpc.StatusCode.CANCELLED:
            res["error"] = ("rpc", str(e.code()))
    res["ttft_s"] = round((first or time.time()) - t0, 3)
    res["wall_s"] = round(time.time() - t0, 3)
    return res


GREEDY = dict(temperature=0.0, presence_penalty=0.0, chat_template_kwargs_json=OFF)
SCHEMA = {"type": "json_schema", "json_schema": {"name": "x", "schema": {
    "type": "object", "properties": {"name": {"type": "string"}, "year": {"type": "integer"},
                                     "tags": {"type": "array", "items": {"type": "string"}}},
    "required": ["name", "year", "tags"]}}}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                          "days": {"type": "integer"}},
                         "required": ["city"]}}}]


def corpus_cases():
    """(case id, messages, params): what B=1 must reproduce byte for byte."""
    cases = [(f"code{i}", msgs(t), dict(GREEDY, max_tokens=256)) for i, t in enumerate(TASKS[:5])]
    cases += [
        ("seeded", msgs(TASKS[11]), dict(temperature=0.7, top_p=0.9, top_k=40, seed=42,
                                         presence_penalty=0.0, chat_template_kwargs_json=OFF, max_tokens=128)),
        ("penalties", msgs(TASKS[12]), dict(temperature=0.0, presence_penalty=1.5, frequency_penalty=0.2,
                                            repetition_penalty=1.1, chat_template_kwargs_json=OFF, max_tokens=128)),
        ("think-budget", msgs("What is 17*23? Answer briefly."), dict(temperature=0.0, presence_penalty=0.0,
                                                                     chat_template_kwargs_json=ON,
                                                                     thinking_budget=48, max_tokens=200)),
        ("think-on", msgs("Is 221 prime?"), dict(temperature=0.0, presence_penalty=0.0,
                                                 chat_template_kwargs_json=ON, max_tokens=300)),
        ("stop", msgs(TASKS[0]), dict(GREEDY, max_tokens=256, stop=["def ", "\n\n\n"])),
        ("min-ignore", msgs("Say hi."), dict(GREEDY, max_tokens=40, min_tokens=20, ignore_eos=True)),
        ("bad-words", msgs(TASKS[1]), dict(GREEDY, max_tokens=96, bad_words=["Rust", "cache"])),
        ("json", msgs("Describe the movie Alien as JSON."), dict(GREEDY, max_tokens=200,
                                                                 response_format_json=json.dumps(SCHEMA))),
        ("tools-auto", msgs("What's the weather in Paris for 3 days?"),
         dict(GREEDY, max_tokens=200, tools_json=json.dumps(TOOLS))),
        ("tools-required", msgs("Weather in Oslo?"),
         dict(GREEDY, max_tokens=200, tools_json=json.dumps(TOOLS), tool_choice="required")),
    ]
    return cases


STABLE = ("text", "reasoning", "finish", "error", "calls", "gen_tokens", "prompt_tokens", "reasoning_tokens")


def run_corpus(st, out):
    res = {}
    for cid, m, p in corpus_cases():
        r = gen(st, m, rid=f"corpus-{cid}", **p)
        res[cid] = r
        print(f"{cid:16s} finish={r['finish']} tokens={r.get('gen_tokens')} tps={r.get('tps')} "
              f"peak={r.get('peak_gb')} err={r['error']}", flush=True)
    json.dump(res, open(out, "w"), indent=1)


def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    bad = 0
    for k in A:
        x, y = A[k], B.get(k, {})
        diff = [f for f in STABLE if x.get(f) != y.get(f)]
        if diff:
            bad += 1
            t1, t2 = x.get("text", ""), y.get("text", "")
            i = next((j for j in range(min(len(t1), len(t2))) if t1[j] != t2[j]), min(len(t1), len(t2)))
            print(f"DIFF {k}: fields {diff}; text diverges at char {i}: {t1[i:i+40]!r} vs {t2[i:i+40]!r}")
        else:
            print(f"same {k}")
    print(f"{len(A) - bad}/{len(A)} byte-identical")
    return bad == 0


def prompts(n):
    return [(f"p{i}", msgs(TASKS[i % len(TASKS)] + ("" if i < len(TASKS) else f" (variant {i})")))
            for i in range(n)]


def run_seq(st, n, out, max_tokens):
    res = {pid: gen(st, m, rid=f"seq-{pid}", max_tokens=max_tokens, **GREEDY) for pid, m in prompts(n)}
    json.dump(res, open(out, "w"), indent=1)


def parallel(st, jobs):
    """jobs: [(key, messages, params, on_event)] all at once -> ({key: result}, wall)."""
    res, ths = {}, []
    t0 = time.time()
    for key, m, p, cb in jobs:
        th = threading.Thread(target=lambda k=key, m=m, p=p, cb=cb: res.__setitem__(
            k, gen(st, m, rid=f"{k}-{uuid.uuid4().hex[:6]}", on_event=cb, **p)))
        th.start()
        ths.append(th)
    for th in ths:
        th.join()
    return res, time.time() - t0


def agreement(res, ref):
    same, firsts = 0, []
    for k, r in res.items():
        a, b = ref[k]["text"], r["text"]
        if a == b:
            same += 1
        else:
            firsts.append(next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b))))
    return same, firsts


def report(res, wall, ref=None):
    toks = sum(r.get("gen_tokens", 0) for r in res.values())
    print(f"wall {wall:.1f}s, aggregate {toks / wall:.1f} tok/s, {len(res)} requests")
    for k, r in sorted(res.items()):
        print(f"  {k:6s} finish={r['finish']} tok={r.get('gen_tokens')} ttft={r['ttft_s']}s "
              f"tps={r.get('tps')} queue_ms={r.get('queue_ms')} batch={r.get('batch_mean')} "
              f"peak={r.get('peak_gb')} err={r['error']}")
    if ref:
        same, firsts = agreement(res, ref)
        print(f"identical to solo: {same}/{len(res)}; first divergence (chars) {sorted(firsts)}")


def run_conc(st, n, ref, max_tokens):
    ref = json.load(open(ref)) if ref else None
    jobs = [(pid, m, dict(GREEDY, max_tokens=max_tokens), None) for pid, m in prompts(n)]
    res, wall = parallel(st, jobs)
    report(res, wall, ref)


def run_cancel(st, ref, max_tokens):
    ref = json.load(open(ref))

    def cancel_after(n, how):
        def cb(res, call):
            if len(res["text"]) >= n:
                if how == "rpc" and "sent" not in res:
                    res["sent"] = True
                    st.Cancel(pb.CancelReq(id=res["id"]))
                elif how == "drop":
                    return False
        return cb
    plan = {"p1": cancel_after(80, "rpc"), "p4": cancel_after(300, "rpc"), "p6": cancel_after(150, "drop")}
    jobs = [(pid, m, dict(GREEDY, max_tokens=max_tokens), plan.get(pid)) for pid, m in prompts(8)]
    res, wall = parallel(st, jobs)
    report(res, wall)
    ok = all(res[k]["finish"] in ("cancelled", "client-dropped") for k in plan)
    rest = {k: v for k, v in res.items() if k not in plan}
    same, firsts = agreement(rest, ref)
    ok &= all(r["finish"] in ("length", "stop") for r in rest.values())
    print(f"cancelled rows ended: {[res[k]['finish'] for k in plan]}; others identical {same}/{len(rest)}")
    h = st.Health(pb.Empty())
    print(f"health ready={h.ready} max_batch={h.max_batch}")
    after = gen(st, msgs("Say OK."), max_tokens=4, **{k: v for k, v in GREEDY.items()})
    print(f"after: finish={after['finish']} peak={after.get('peak_gb')} (worker log: active= after idle)")
    print("PASS" if ok else "FAIL")


def run_stops(st):
    stops = {"p0": ["def "], "p1": ["impl"], "p2": ["func"], "p3": ["SELECT", "select"]}
    jobs = [(k, m, dict(GREEDY, max_tokens=300, stop=stops.get(k, [])), None) for k, m in prompts(6)]
    res, wall = parallel(st, jobs)
    report(res, wall)
    ok = True
    for k, s in stops.items():
        r = res[k]
        leak = any(x in r["text"] for x in s)
        print(f"  {k} stops={s} finish={r['finish']} leak={leak}")
        ok &= not leak
    print("PASS" if ok else "FAIL")


def run_mixedmax(st):
    want = {"p0": 1, "p1": 7, "p2": 64, "p3": 512, "p4": 7, "p5": 1}
    jobs = [(k, m, dict(GREEDY, max_tokens=want[k], ignore_eos=True), None) for k, m in prompts(6)]
    res, wall = parallel(st, jobs)
    report(res, wall)
    ok = all(res[k]["gen_tokens"] == n and res[k]["finish"] == "length" for k, n in want.items())
    print("PASS" if ok else "FAIL")


def run_think(st):
    jobs = []
    for i, (k, m) in enumerate(prompts(6)):
        p = dict(temperature=0.0, presence_penalty=0.0, max_tokens=400,
                 chat_template_kwargs_json=ON if i % 2 else OFF)
        if i == 5:
            p["thinking_budget"] = 32
        jobs.append((k, m, p, None))
    res, wall = parallel(st, jobs)
    report(res, wall)
    ok = True
    for i, k in enumerate(sorted(res)):
        r = res[k]
        on = i % 2 == 1
        print(f"  {k} thinking={'on' if on else 'off'} reasoning_tokens={r.get('reasoning_tokens')} "
              f"reasoning_chars={len(r['reasoning'])} answer_chars={len(r['text'])}")
        ok &= (r["reasoning"] == "") != on
        if i == 5:
            ok &= r.get("reasoning_tokens", 999) <= 32 + 2 and len(r["text"]) > 0
    print("PASS" if ok else "FAIL")


def run_grammar(st):
    jobs = [(f"j{i}", msgs(f"Describe the movie number {i} of the Alien franchise as JSON."),
             dict(GREEDY, max_tokens=500, response_format_json=json.dumps(SCHEMA)), None) for i in range(4)]
    jobs += [(k, m, dict(GREEDY, max_tokens=200), None) for k, m in prompts(4)]
    res, wall = parallel(st, jobs)
    report(res, wall)
    ok = True
    for i in range(4):
        try:
            d = json.loads(res[f"j{i}"]["text"])
            ok &= set(d) == {"name", "year", "tags"}
        except ValueError:
            ok = False
            print("  invalid JSON:", res[f"j{i}"]["text"][:200])
    print("PASS" if ok else "FAIL")


def filler(n_chars, seed="sys"):
    import hashlib
    words = "alpha beta gamma delta buffer socket thread mutex queue cache index shard".split()
    out, i = [], 0
    while sum(len(x) + 1 for x in out) < n_chars:
        h = hashlib.sha256(f"{seed}/{i}".encode()).digest()
        out.append(f"[{seed}:{i:04d}] " + " ".join(words[b % len(words)] for b in h[:8]))
        i += 1
    return "\n".join(out)


def run_pcache(st):
    system = "You are a helpful assistant. Reference log follows.\n" + filler(26000)
    jobs = [(f"c{i}", msgs(TASKS[i], system=system), dict(GREEDY, max_tokens=64), None) for i in range(8)]
    first = gen(st, *jobs[0][1:2], rid="pcache-warm", **jobs[0][2])
    print(f"cold: ttft={first['ttft_s']}s prompt_tokens={first.get('prompt_tokens')}")
    res, wall = parallel(st, jobs[1:])
    report(res, wall)
    print("(check the worker log for 'prompt cache HIT' lines)")


def run_memory(st, n, no_max=False):
    # no_max: no max_tokens and no ignore_eos (reservation capped by QWEN_BATCH_RESERVE_TOKENS)
    p = dict(GREEDY) if no_max else dict(GREEDY, max_tokens=256, ignore_eos=True)
    jobs = [(f"m{i}", msgs(TASKS[i % len(TASKS)], system=filler(28000, seed=f"m{i}")), p, None)
            for i in range(n)]
    res, wall = parallel(st, jobs)
    report(res, wall)
    print(f"max peak {max(r.get('peak_gb', 0) for r in res.values())} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--addr", default="127.0.0.1:50051")
    ap.add_argument("--out")
    ap.add_argument("--ref")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()
    if a.suite == "compare":
        sys.exit(0 if compare(*a.files) else 1)
    st = stub(a.addr)
    {"corpus": lambda: run_corpus(st, a.out),
     "seq": lambda: run_seq(st, a.n, a.out, a.max_tokens),
     "conc": lambda: run_conc(st, a.n, a.ref, a.max_tokens),
     "cancel": lambda: run_cancel(st, a.ref, a.max_tokens),
     "stops": lambda: run_stops(st),
     "mixedmax": lambda: run_mixedmax(st),
     "think": lambda: run_think(st),
     "grammar": lambda: run_grammar(st),
     "pcache": lambda: run_pcache(st),
     "memory": lambda: run_memory(st, a.n),
     "memory-nomax": lambda: run_memory(st, a.n, no_max=True)}[a.suite]()


if __name__ == "__main__":
    main()
