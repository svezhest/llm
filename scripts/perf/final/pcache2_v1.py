#!/usr/bin/env python3
"""V1: prompt-cache hit == cold on the real model, through the gateway (:8090).

  on  --out F            cache-ON worker (fresh): cold + repeat (hit) of 5 coding tasks and a
                         7.5K-system prompt, 2 siblings over that system prompt, two 3-turn chats
                         (turn N+1 = turn N + its answer, reasoning not sent back) -> F (with the
                         exact messages of every prompt)
  off --ref F --out G    cache-OFF worker: every distinct prompt of F once (same messages)
  cmp F G                byte-compare every ON result against the OFF run of the same prompt
Run with src/worker/.venv/bin/python, PYTHONPATH=src/worker (Health over gRPC)."""
import argparse, http.client, json, sys, time
import grpc
import worker_pb2_grpc as pb_grpc  # PYTHONPATH=src/worker


class w:  # the bits of the GPU test client (scratchpad gpu_tests/wtest.py) this uses
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

    @staticmethod
    def filler(n_chars, seed="sys"):
        import hashlib
        words = "alpha beta gamma delta buffer socket thread mutex queue cache index shard".split()
        out, i = [], 0
        while sum(len(x) + 1 for x in out) < n_chars:
            h = hashlib.sha256(f"{seed}/{i}".encode()).digest()
            out.append(f"[{seed}:{i:04d}] " + " ".join(words[b % len(words)] for b in h[:8]))
            i += 1
        return "\n".join(out)

    @staticmethod
    def stub(addr):
        ch = grpc.insecure_channel(addr, options=[("grpc.max_receive_message_length", 64 << 20)])
        return pb_grpc.WorkerStub(ch)

MODEL = "ornith15-9b-mlx-8bit"


def chat(messages, max_tokens, thinking, rid):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "presence_penalty": 0, "request_id": rid, "chat_template_kwargs": {"enable_thinking": thinking}}
    c = http.client.HTTPConnection("127.0.0.1", 8090, timeout=900)
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", body=json.dumps(body), headers={"content-type": "application/json"})
    r = c.getresponse()
    raw = r.read().decode()
    dt = time.time() - t0
    if r.status != 200:
        return {"error": f"HTTP {r.status}: {raw[:300]}", "wall_s": dt}
    d = json.loads(raw)
    ch = d["choices"][0]
    m = ch["message"]
    u = d.get("usage", {})
    return {"text": m.get("content") or "", "reasoning": m.get("reasoning_content") or m.get("reasoning") or "",
            "finish": ch.get("finish_reason"), "completion_tokens": u.get("completion_tokens"),
            "prompt_tokens": u.get("prompt_tokens"),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "usage": u, "wall_s": round(dt, 3), "error": None}


def health():
    import worker_pb2 as pb
    h = w.stub("127.0.0.1:50051").Health(pb.Empty())
    return {"entries": h.prompt_cache_entries, "bytes": h.prompt_cache_bytes, "hits": h.prompt_cache_hits,
            "misses": h.prompt_cache_misses, "evictions": h.prompt_cache_evictions, "max_batch": h.max_batch}


SYS7K = "You are a helpful assistant. Reference log follows.\n" + w.filler(26000)
CHATS = {
    "chatT": (True, 1024, "You are a careful assistant. Session notes follow.\n" + w.filler(9000, seed="chatT"),
              ["What is 17*23? Answer briefly.", "Now divide that by 7, rounded to two decimals.",
               "Write that last number in words."]),
    "chatN": (False, 128, "You are a concise assistant. Project notes follow.\n" + w.filler(9000, seed="chatN"),
              ["Name three classic sorting algorithms.", "Which of them are stable?",
               "Show a short Python implementation of one stable one."]),
}


def run(res, key, messages, max_tokens, thinking):
    rid = f"v1-{key}-{int(time.time()*1000) % 100000}"
    r = chat(messages, max_tokens, thinking, rid)
    r.update(key=key, rid=rid, messages=messages, max_tokens=max_tokens, thinking=thinking)
    res[key] = r
    print(f"{key:14s} prompt={r.get('prompt_tokens')} cached={r.get('cached_tokens')} "
          f"tok={r.get('completion_tokens')} finish={r.get('finish')} wall={r['wall_s']:.2f}s err={r['error']}",
          flush=True)
    return r


def on(out):
    res, hl = {}, {}
    hl["start"] = health()
    for i in range(5):
        m = [{"role": "user", "content": w.TASKS[i]}]
        run(res, f"code{i}.cold", m, 128, False)
        run(res, f"code{i}.hit", m, 128, False)
    sysm = lambda t: [{"role": "system", "content": SYS7K}, {"role": "user", "content": t}]
    run(res, "sysA.cold", sysm(w.TASKS[0]), 128, False)
    run(res, "sysA.hit", sysm(w.TASKS[0]), 128, False)
    run(res, "sysB.sib", sysm(w.TASKS[5]), 128, False)
    run(res, "sysC.sib", sysm(w.TASKS[7]), 128, False)
    hl["before_chats"] = health()
    for name, (think, mt, system, turns) in CHATS.items():
        msgs = [{"role": "system", "content": system}]
        for n, q in enumerate(turns, 1):
            msgs = msgs + [{"role": "user", "content": q}]
            r = run(res, f"{name}.{n}", msgs, mt, think)
            hl[f"{name}.{n}"] = health()
            msgs = msgs + [{"role": "assistant", "content": r.get("text", "")}]   # reasoning not sent back
    hl["end"] = health()
    json.dump({"results": res, "health": hl}, open(out, "w"), indent=1)
    print(json.dumps(hl, indent=1))


def off(ref, out):
    R = json.load(open(ref))["results"]
    res, seen = {}, set()
    for k, r in R.items():
        base = k.split(".")[0] if k.endswith((".cold", ".hit")) else k
        if base in seen:
            continue
        seen.add(base)
        run(res, base, r["messages"], r["max_tokens"], r["thinking"])
    json.dump({"results": res, "health": {"end": health()}}, open(out, "w"), indent=1)


FIELDS = ("text", "reasoning", "finish", "completion_tokens", "prompt_tokens", "error")


def cmp(a, b):
    A, B = json.load(open(a))["results"], json.load(open(b))["results"]
    bad = 0
    for k, r in A.items():
        base = k.split(".")[0] if k.endswith((".cold", ".hit")) else k
        base = base if base in B else k.split(".")[0]
        x = B[base]
        assert x["messages"] == r["messages"], k
        diff = [f for f in FIELDS if r.get(f) != x.get(f)]
        if diff:
            bad += 1
            for f in ("reasoning", "text"):
                t1, t2 = x.get(f, ""), r.get(f, "")
                if t1 != t2:
                    i = next((j for j in range(min(len(t1), len(t2))) if t1[j] != t2[j]), min(len(t1), len(t2)))
                    print(f"DIFF {k} vs cold-off {base}: {diff}; {f} diverges at char {i}: {t1[i:i+40]!r} (off) vs {t2[i:i+40]!r} (on)")
        else:
            print(f"same {k:12s} == off {base:8s} prompt={r['prompt_tokens']} cached={r['cached_tokens']} "
                  f"tok={r['completion_tokens']} finish={r['finish']} reasoning_chars={len(r['reasoning'])} text_chars={len(r['text'])}")
    print(f"{len(A) - bad}/{len(A)} identical (text+reasoning+finish+tokens) to the cache-off run")
    return bad == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode"); ap.add_argument("files", nargs="*")
    ap.add_argument("--out"); ap.add_argument("--ref")
    a = ap.parse_args()
    if a.mode == "health":
        print(json.dumps(health()))
    elif a.mode == "on":
        on(a.out)
    elif a.mode == "off":
        off(a.ref, a.out)
    else:
        sys.exit(0 if cmp(*a.files) else 1)
