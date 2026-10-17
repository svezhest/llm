#!/usr/bin/env python3
"""Reply-KV GPU gate driver (stdlib only), against our gateway on :8090.
  drive.py chat    CORPUS OUT TAG [KEY]                gate 1: one conversation, 6 turns, sequential, max_tokens 96;
                                                   the next turn's history holds the answer exactly as returned
  drive.py replay  IN.json OUT TAG                the requests of IN (exact messages), sequential, in order
  drive.py convs   CORPUS KEY OUT TAG TURNS MAXTOK [ignore_eos]
                                                   one closed-loop thread per conversation of CORPUS[KEY],
                                                   all started together, TURNS turns each
  drive.py compare A.json B.json                  answers (content, completion_tokens, finish) + messages
Every request: temperature 0, presence_penalty 0, non-streaming, enable_thinking false (template kwargs),
request_id rk-TAG-<conv>-<turn> (the worker log id is chatcmpl-<request_id>-<12 hex>).
"""
import json, sys, time, threading, urllib.request

URL = "http://127.0.0.1:8090/v1/chat/completions"
MODEL = "ornith15-9b-mlx-8bit"

def call(messages, tag, rid, maxtok, ignore_eos=False):
    body = {"model": MODEL, "messages": messages, "temperature": 0, "presence_penalty": 0,
            "max_tokens": maxtok, "stream": False, "request_id": rid,
            "chat_template_kwargs": {"enable_thinking": False}}
    if ignore_eos:
        body["ignore_eos"] = True
    t0 = time.time()
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    rec = {"tag": tag, "rid": rid, "messages": messages, "t_start": t0, "max_tokens": maxtok,
           "ignore_eos": ignore_eos}
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            j = json.loads(r.read())
        m = j["choices"][0]["message"]
        u = j.get("usage", {})
        rec.update(ok=True, id=j.get("id"), content=m.get("content"),
                   reasoning=m.get("reasoning_content") or m.get("reasoning"),
                   finish=j["choices"][0].get("finish_reason"),
                   prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                   cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens"))
    except urllib.error.HTTPError as e:
        rec.update(ok=False, error=f"HTTP {e.code}: {e.read()[:500]!r}")
    except Exception as e:  # noqa
        rec.update(ok=False, error=repr(e))
    rec["t_end"] = time.time()
    rec["elapsed"] = rec["t_end"] - t0
    print(f"{tag:10s} ok={rec['ok']} prompt={rec.get('prompt_tokens')} cached={rec.get('cached_tokens')} "
          f"compl={rec.get('completion_tokens')} {rec.get('finish')} {rec['elapsed']:.1f}s {rec.get('error','')}",
          flush=True)
    return rec

def chat(corpus, out, tag, key="g1"):
    C = json.load(open(corpus))
    msgs = [{"role": "system", "content": C["g1_system"]}]
    recs = []
    for t, u in enumerate(C[key + "_turns"], 1):
        msgs = msgs + [{"role": "user", "content": u}]
        r = call(msgs, f"turn{t}", f"rk-{tag}-0-{t}", 96); recs.append(r)
        if not r["ok"]:
            break
        msgs = msgs + [{"role": "assistant", "content": r["content"]}]
    json.dump(recs, open(out, "w"), indent=1, ensure_ascii=False)

def replay(pj, out, tag):
    recs = []
    for k, p in enumerate(json.load(open(pj))):
        recs.append(call(p["messages"], p["tag"], f"rk-{tag}-r-{k}", p.get("max_tokens", 96),
                         p.get("ignore_eos", False)))
    json.dump(recs, open(out, "w"), indent=1, ensure_ascii=False)

def convs(corpus, key, out, tag, turns, maxtok, ignore_eos):
    C = json.load(open(corpus))[key]
    recs, lock = [], threading.Lock()
    go = threading.Barrier(len(C))
    def run(i, c):
        msgs = [{"role": "system", "content": c["system"]}]
        go.wait()
        for t in range(turns):
            msgs = msgs + [{"role": "user", "content": c["turns"][t]}]
            r = call(msgs, f"c{i}t{t + 1}", f"rk-{tag}-{i}-{t + 1}", maxtok, ignore_eos)
            r["conv"], r["turn"] = i, t + 1
            with lock:
                recs.append(r)
            if not r["ok"]:
                return
            msgs = msgs + [{"role": "assistant", "content": r["content"]}]
    th = [threading.Thread(target=run, args=(i, c)) for i, c in enumerate(C)]
    t0 = time.time()
    [x.start() for x in th]; [x.join() for x in th]
    recs.sort(key=lambda r: (r["conv"], r["turn"]))
    json.dump({"t_start": t0, "t_end": time.time(), "requests": recs}, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"done {len(recs)} requests, ok {sum(r['ok'] for r in recs)}, {time.time() - t0:.0f} s")

def compare(a, b):
    A = json.load(open(a)); B = json.load(open(b))
    n = same = 0
    for x, y in zip(A, B):
        n += 1
        eq = (x.get("ok") and y.get("ok") and x["content"] == y["content"]
              and x["completion_tokens"] == y["completion_tokens"] and x["finish"] == y["finish"]
              and x["messages"] == y["messages"])
        same += bool(eq)
        print(f"{x['tag']:10s} {'SAME' if eq else 'DIFF'} prompt={y.get('prompt_tokens')} "
              f"cached A={x.get('cached_tokens')} B={y.get('cached_tokens')} compl A={x.get('completion_tokens')} "
              f"B={y.get('completion_tokens')} {x.get('finish')}/{y.get('finish')}")
    print(f"identical {same}/{n} (len A {len(A)}, B {len(B)})")

if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "chat": chat(*a)
    elif cmd == "replay": replay(*a)
    elif cmd == "convs": convs(a[0], a[1], a[2], a[3], int(a[4]), int(a[5]), len(a) > 6 and a[6] == "ignore_eos")
    elif cmd == "compare": compare(*a)
