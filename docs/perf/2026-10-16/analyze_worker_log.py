import re, sys, statistics as st
from datetime import datetime
path, since = sys.argv[1], sys.argv[2]  # since = "2026-10-16 01:39:21"
T = lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S,%f")
t_since = datetime.strptime(since, "%Y-%m-%d %H:%M:%S")
R, B = {}, []
for line in open(path, errors="replace"):
    m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) INFO  qwen\.worker (.*)", line)
    if not m: continue
    ts = T(m[1])
    if ts < t_since: continue
    msg = m[2]
    if msg.startswith("Generate id="):
        rid = msg.split()[1][3:]; R.setdefault(rid, {})["t0"] = ts; continue
    if msg.startswith("batch: rows="):
        d = dict(kv.split("=", 1) for kv in msg[len("batch: "):].split() if "=" in kv)
        B.append((ts, int(d["rows"]), int(d["waiting"]), d["prefilling"] != "-", float(d["steps/s"]), float(d["mean_rows"]),
                  float(d["pcache"].rstrip("GiB")), float(d["active"].rstrip("GiB"))))
        continue
    mm = re.match(r"id=(\S+) (.*)", msg)
    if not mm: continue
    rid, rest = mm[1], mm[2]; r = R.setdefault(rid, {})
    if rest.startswith("prompt_tokens="): r["prompt"] = int(re.match(r"prompt_tokens=(\d+)", rest)[1])
    elif "prompt cache HIT" in rest:
        x = re.search(r"HIT: (\d+)/(\d+).*prefilling (\d+)", rest); r["reused"], r["prefill"] = int(x[1]), int(x[3]); r["hit"] = True; r["t_hit"] = ts
    elif "prompt cache MISS" in rest: r["hit"] = False; r["t_hit"] = ts
    elif rest.startswith("prompt cache store"): r["t_store"] = ts
    elif "finish=" in rest:
        d = dict(kv.split("=", 1) for kv in rest.split() if "=" in kv)
        r.update(t1=ts, gen=int(d["gen_tokens"]), ttft=int(d["ttft_ms"]), tps=float(d["tps"]), cached=int(d["cached"]),
                 queue=int(d["queue_ms"]), batch=float(d["batch"]), peak=float(d["peak"].rstrip("GB")), finish=d["finish"])
done = [r for r in R.values() if "t1" in r and "t0" in r and "prompt" in r]
for r in done:
    r.setdefault("prefill", r["prompt"] - r["cached"]); r["lat"] = (r["t1"] - r["t0"]).total_seconds()
    r["dec_s"] = r["gen"] / r["tps"] if r["tps"] > 0 else 0
q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]
def summ(name, xs, fmt="{:.1f}"):
    if not xs: print(f"{name}: -"); return
    print(f"{name}: n={len(xs)} mean={fmt.format(st.mean(xs))} p50={fmt.format(q(xs,.5))} p90={fmt.format(q(xs,.9))} max={fmt.format(max(xs))}")
t_first = min(r["t0"] for r in done); t_last = max(r["t1"] for r in done); wall = (t_last - t_first).total_seconds()
P = sum(r["prompt"] for r in done); C = sum(r["cached"] for r in done); G = sum(r["gen"] for r in done); PF = sum(r["prefill"] for r in done)
print(f"window {t_first:%H:%M:%S}–{t_last:%H:%M:%S} = {wall/60:.1f} min, requests={len(done)} ({len(done)/(wall/60):.1f}/min)")
print(f"prompt tokens {P} (cached {C} = {100*C/P:.1f}%), prefilled {PF}, generated {G}")
print(f"aggregate: prefill {PF/wall:.0f} tok/s wall, decode {G/wall:.1f} tok/s wall")
print(f"cache: hits {sum(1 for r in done if r.get('hit'))}, misses {sum(1 for r in done if r.get('hit') is False)}")
print(f"finish reasons: { {k: sum(1 for r in done if r['finish']==k) for k in set(r['finish'] for r in done)} }  peak max {max(r['peak'] for r in done):.2f} GB")
summ("prompt tokens", [r["prompt"] for r in done], "{:.0f}")
summ("prefilled tokens/request", [r["prefill"] for r in done], "{:.0f}")
summ("gen tokens/request", [r["gen"] for r in done], "{:.0f}")
summ("latency s (Generate→finish)", [r["lat"] for r in done])
summ("queue s", [r["queue"]/1000 for r in done])
summ("ttft s", [r["ttft"]/1000 for r in done])
summ("decode s (gen/tps)", [r["dec_s"] for r in done])
summ("decode tok/s per request", [r["tps"] for r in done])
tot = sum(r["lat"] for r in done)
print(f"latency share: queue {100*sum(r['queue']/1000 for r in done)/tot:.0f}%, ttft {100*sum(r['ttft']/1000 for r in done)/tot:.0f}%, decode {100*sum(r['dec_s'] for r in done)/tot:.0f}%  (ttft may include queue)")
big = [r for r in done if r["prefill"] >= 1500 and r["ttft"] > 0]
if big: print(f"prefill rate on requests with >=1500 new tokens: {st.mean(r['prefill']/(r['ttft']/1000) for r in big):.0f} tok/s (prefill/ttft, n={len(big)})")
print("\ndecode tok/s per request by prompt length:")
for lo, hi in [(0,5000),(5000,10000),(10000,15000),(15000,20000),(20000,30000),(30000,10**6)]:
    xs = [r["tps"] for r in done if lo <= r["prompt"] < hi]
    if xs: print(f"  {lo//1000:>2}–{hi//1000 if hi<10**6 else '∞'}K: n={len(xs)} mean={st.mean(xs):.1f} p50={q(xs,.5):.1f}")
print("\nscheduler (10 s ticks): steps/s by rows, with/without a prefill in progress:")
for rows in range(0, 17):
    for pf in (False, True):
        xs = [b[4] for b in B if b[1] == rows and b[3] == pf]
        if xs: print(f"  rows={rows:>2} prefilling={'y' if pf else 'n'}: ticks={len(xs)} steps/s mean={st.mean(xs):.2f}")
print(f"ticks with prefill in progress: {100*sum(b[3] for b in B)/len(B):.0f}%;  waiting>0: {100*sum(b[2]>0 for b in B)/len(B):.0f}%;  mean rows {st.mean(b[1] for b in B):.2f}")
print(f"pcache GiB max {max(b[6] for b in B):.2f}, active GiB max {max(b[7] for b in B):.2f}")
