#!/usr/bin/env python3
"""Chain/evict GPU gate driver (stdlib only).
  drive.py chat     CORPUS OUT          step 1/2 conversation script (sequential)
  drive.py pressure CORPUS OUT [INFL]   6 convs x 4 turns, round-robin, INFL (4) in flight
  drive.py replay   PRESSURE_JSON OUT   the pressure run's exact requests, sequential, in start order
  drive.py compare  A.json B.json       byte-compare answers (content, completion_tokens, finish)
"""
import json, sys, time, threading, urllib.request

URL = "http://127.0.0.1:8090/v1/chat/completions"
MODEL = "ornith15-9b-mlx-8bit"
MAXTOK = 80

def call(messages, tag):
    body = {"model": MODEL, "messages": messages, "temperature": 0, "presence_penalty": 0,
            "max_tokens": MAXTOK, "stream": False}
    t0 = time.time()
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    rec = {"tag": tag, "messages": messages, "t_start": t0}
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
    rec["elapsed"] = time.time() - t0
    print(f"{tag:14s} ok={rec['ok']} prompt={rec.get('prompt_tokens')} cached={rec.get('cached_tokens')} "
          f"compl={rec.get('completion_tokens')} {rec['elapsed']:.1f}s {rec.get('error','')}", flush=True)
    return rec

def chat(corpus, out):
    C = json.load(open(corpus))
    sysm = {"role": "system", "content": C["system"]}
    first = C["document"] + "\n\n" + C["q1"]
    msgs = [sysm, {"role": "user", "content": first}]
    recs, hist = [], {}
    for t in range(1, 8):
        if t > 1:
            msgs = msgs + [{"role": "user", "content": C["followups"][t - 2]}]
        hist[t] = list(msgs)
        r = call(msgs, f"turn{t}"); recs.append(r)
        if not r["ok"]:
            break
        msgs = msgs + [{"role": "assistant", "content": r["content"]}]
    if all(r["ok"] for r in recs):
        recs.append(call(hist[7], "regen7"))
        recs.append(call(hist[4], "regen4"))
        recs.append(call([sysm, {"role": "user", "content": C["document"] + "\n\n" + C["q1_sibling"]}],
                         "sibling"))
    json.dump(recs, open(out, "w"), indent=1, ensure_ascii=False)

def pressure(corpus, out, infl=4):
    C = json.load(open(corpus))
    convs = [{"msgs": [{"role": "system", "content": c["system"]}], "turns": c["turns"], "next": 0}
             for c in C["convs"]]
    lock = threading.Condition()
    busy = set(); recs = []; order = [0]
    def pick():
        # round-robin: the idle conversation with the fewest turns done, lowest index first
        cand = [i for i, c in enumerate(convs) if i not in busy and c["next"] < len(c["turns"])]
        if not cand:
            return None
        return min(cand, key=lambda i: (convs[i]["next"], (i - order[0]) % len(convs)))
    def worker():
        while True:
            with lock:
                while True:
                    left = any(c["next"] < len(c["turns"]) for i, c in enumerate(convs) if i not in busy)
                    if not left and not busy:
                        lock.notify_all(); return
                    i = pick()
                    if i is not None:
                        break
                    if not left:
                        return
                    lock.wait()
                busy.add(i); c = convs[i]; t = c["next"]
                msgs = c["msgs"] + [{"role": "user", "content": c["turns"][t]}]
                seq = len(recs); recs.append(None); order[0] = (i + 1) % len(convs)
            r = call(msgs, f"c{i}t{t}"); r["seq"] = seq; r["conv"] = i; r["turn"] = t
            with lock:
                recs[seq] = r
                c["msgs"] = msgs + [{"role": "assistant", "content": r.get("content") or ""}]
                c["next"] += 1; busy.discard(i); lock.notify_all()
    th = [threading.Thread(target=worker) for _ in range(infl)]
    [x.start() for x in th]; [x.join() for x in th]
    json.dump(recs, open(out, "w"), indent=1, ensure_ascii=False)

def replay(pj, out):
    P = json.load(open(pj))
    recs = []
    for p in P:
        r = call(p["messages"], p["tag"]); r["seq"] = p["seq"]; recs.append(r)
    json.dump(recs, open(out, "w"), indent=1, ensure_ascii=False)

def compare(a, b):
    A = json.load(open(a)); B = json.load(open(b))
    n = same = 0
    for x, y in zip(A, B):
        n += 1
        assert x["tag"] == y["tag"], (x["tag"], y["tag"])
        eq = (x.get("ok") and y.get("ok") and x["content"] == y["content"]
              and x["completion_tokens"] == y["completion_tokens"] and x["finish"] == y["finish"]
              and x["messages"] == y["messages"])
        same += bool(eq)
        print(f"{x['tag']:10s} {'SAME' if eq else 'DIFF'} prompt={y.get('prompt_tokens')} "
              f"cached A={x.get('cached_tokens')} B={y.get('cached_tokens')} compl={y.get('completion_tokens')}")
    print(f"identical {same}/{n} (len A {len(A)}, B {len(B)})")

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "chat": chat(sys.argv[2], sys.argv[3])
    elif cmd == "pressure": pressure(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 4)
    elif cmd == "replay": replay(sys.argv[2], sys.argv[3])
    elif cmd == "compare": compare(sys.argv[2], sys.argv[3])
