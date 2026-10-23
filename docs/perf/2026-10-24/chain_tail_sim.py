"""QWEN_PROMPT_CACHE_CHAIN_TAIL: what the stride checkpoints in a chain entry's
last CHAIN_TAIL tokens cost (9B bytes) and buy (resume depth), through the real
scheduler + prompt cache (tiny random model, CPU, production knobs: slab 2048 /
min 256 / stride 2048 / cap 12 / marks 3, reply KV on).

As docs/perf/2026-10-16/chain_ckpts_sim.py, but the episode is driven by the
model's own replies (thinking off: the reply KV stores them, the next turn
resumes after the reply), and after every turn's store the cache is probed
with `peek` (no side effects) for prompts that would diverge from it:

  next     the real next turn (the run's own `reused`)
  regen    the same prompt again (regenerate the last answer)
  edit@f   the last message (the newest tool output / user message) edited at
           fraction f of its length (a chat UI's "edit": the history after it
           goes, the edited message is the last one)
  back2    the message two turns back edited at its middle (an agent framework
           rewriting / truncating an older tool output)
  sib@f    (SIM_USER0 > 1000) a new conversation over the same system prompt
           and first user document, diverging at fraction f of the document

Per probe: `replay` = tokens to prefill (len - resume), `lost` = d - resume
(tokens before the divergence point d that are prefilled again). 9B scale:
32 KiB of KV per token, 51.5 MB per GDN state (24 x 32x128x128 fp32 + conv).

Env: SIM_TAILS (4096,2048,0), SIM_USER0S (300,8000), SIM_DISTS (appworld,long):
appworld = tool outputs lognormal(5.93, 1.0)+30 capped at 6000 (median ~400,
as the 2026-10-16 trace: prefilled/request p50 379, p90 2762), long = uniform
2048..8192 (pasted documents / big tool outputs); SIM_TURNS (appworld 30,
long 10); SIM_SEEDS (1,2); SIM_MARKS (QWEN_PROMPT_CACHE_CHAIN_MARKS, 3);
SIM_JSON (path for the raw rows)."""
import json, os, random, statistics, sys
_W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src", "worker")
sys.path[:0] = [_W, os.path.join(_W, "tests")]
import conftest  # noqa (CPU device)
from conftest import EOS, IM_START, build_lm, prompts, request, run, scheduler
import mlx.core as mx
import worker

worker.PROMPT_CACHE = True
worker.PROMPT_CACHE_REPLY = True
worker.PROMPT_CACHE_CHAIN_MARKS = int(os.environ.get("SIM_MARKS", "3"))   # production: 3
lm = build_lm("dense", None, mx.float32)
A, NL, TH, NN, THC, IM_END = 11, 12, 13, 14, 15, 16
GEN = [IM_START, A, NL, TH, NN, THC, NN]      # assistant header, thinking off
KV_B, STATE_B = 32 * 1024, 51.5e6
EDIT_F = (0.25, 0.5, 0.75, 0.95)
SIB_F = (0.25, 0.5, 0.75, 0.95)


def env_list(name, default, conv=int):
    return [conv(x) for x in os.environ.get(name, default).split(",") if x]


def tool_len(rnd, dist):
    if dist == "appworld":
        return min(6000, int(rnd.lognormvariate(5.93, 1.0)) + 30)
    return rnd.randint(2048, 8192)


def ends_of(ids):
    return set(worker._bounds(ids, IM_START)) | {len(ids) - 1}


def probe(pc, ids, d):
    """-> (replay, lost) for a prompt `ids` diverging from the cache at d."""
    _, pos = pc.peek(ids, ends_of(ids))
    return len(ids) - pos, max(0, min(d, len(ids) - 1) - pos)


def edited(prefix, n, seed):
    """prefix + n fresh tokens (the edited rest of the message) + GEN."""
    return prefix + (prompts([n], seed)[0] if n > 0 else []) + GEN


def go(tail, user0, dist, turns, seed):
    worker.PROMPT_CACHE_CHAIN_TAIL = tail
    rnd = random.Random(seed)
    sched = scheduler(lm, max_batch=1)
    pc = sched.eng._pcache
    sys_ = [IM_START] + prompts([4700], seed)[0]
    hist = sys_ + [IM_START] + prompts([user0], seed + 1)[0]
    doc0 = (len(sys_), len(hist))                     # the first user message [start, next start)
    rows, history = [], []                           # history: the prompts so far
    for i in range(turns):
        p = hist + GEN
        n = rnd.randint(20, 100)                     # AppWorld: 50 generated tokens p50, p90 97
        r = request(p, n, f"t{i}")
        toks = run(sched, [r])[r.id][0]
        (e,) = pc.entries
        segs = {id(s): s for s, _, _ in e.chain}
        kv = sum(s.b - s.a for s in segs.values())
        states = len(e.ckpts) + (e.end is not None)
        row = dict(turn=i, L=len(e.ids), prompt=len(p), new=len(p) - r.reused, reused=r.reused,
                   ckpts=sorted(e.ckpts), states=states, mb=(kv * KV_B + states * STATE_B) / 1e6,
                   state_mb=states * STATE_B / 1e6)
        starts = [j for j, t in enumerate(p) if t == IM_START]
        # probes against the cache as the next turn finds it
        row["regen"] = probe(pc, p, len(p))
        s, g = starts[-2], starts[-1]                # the last message [s, g)
        row["last_len"] = g - s
        for f in EDIT_F:
            d = s + 1 + int(f * (g - s - 1))
            row[f"edit@{f}"] = probe(pc, edited(p[:d], g - d, 9000 + 7 * i), d)
        if len(history) >= 2:
            q = history[-2]                          # the prompt two turns back: its last message
            qs = [j for j, t in enumerate(q) if t == IM_START]
            a, b = qs[-2], qs[-1]
            d = a + 1 + (b - a - 1) // 2
            row["back2"] = probe(pc, edited(p[:d], b - d, 9100 + 7 * i), d)
            row["back2_depth"] = len(p) - d          # how far from the end it diverges
        if user0 > 1000:
            a, b = doc0
            for f in SIB_F:
                d = a + 1 + int(f * (b - a - 1))
                row[f"sib@{f}"] = probe(pc, edited(p[:d], b - d, 9200 + 7 * i), d)
        rows.append(row)
        history.append(p)
        tool = prompts([tool_len(rnd, dist)], seed + 200 + i)[0]
        eos = bool(toks) and toks[-1] == EOS
        hist = p + toks + ([] if eos else [IM_END]) + [NL, IM_START] + tool
    return rows


def stat(xs):
    xs = sorted(xs)
    if not xs:
        return "-"
    p90 = xs[min(len(xs) - 1, int(0.9 * len(xs)))]
    return f"{statistics.mean(xs):7.0f} {statistics.median(xs):6.0f} {p90:6.0f}"


def main():
    tails = env_list("SIM_TAILS", "4096,2048,0")
    user0s = env_list("SIM_USER0S", "300,8000")
    dists = env_list("SIM_DISTS", "appworld,long", str)
    seeds = env_list("SIM_SEEDS", "1,2")
    out = {}
    for dist in dists:
        turns = int(os.environ.get("SIM_TURNS", "30" if dist == "appworld" else "10"))
        for user0 in user0s:
            for tail in tails:
                rows = []
                for seed in seeds:
                    rows += [dict(r, seed=seed) for r in go(tail, user0, dist, turns, seed)]
                out[f"{dist}/{user0}/{tail}"] = rows
                rs = [r for r in rows if r["turn"] >= 3]
                print(f"\n== dist={dist} user0={user0} CHAIN_TAIL={tail}  turns={turns} "
                      f"seeds={seeds}  (turns >= 3: n={len(rs)})", flush=True)
                print(f"  prompt L mean {statistics.mean(r['L'] for r in rs):.0f}, last message "
                      f"mean/p50/p90 {stat([r['last_len'] for r in rs])}")
                print(f"  states/entry mean {statistics.mean(r['states'] for r in rs):.2f} "
                      f"(max {max(r['states'] for r in rs)}), 9B MB/entry mean "
                      f"{statistics.mean(r['mb'] for r in rs):.0f}, states MB mean "
                      f"{statistics.mean(r['state_mb'] for r in rs):.0f}")
                print("  probe           replay mean/p50/p90      lost mean/p50/p90   "
                      "(tokens)")
                keys = ["next", "regen"] + [f"edit@{f}" for f in EDIT_F] + ["back2"]
                if user0 > 1000:
                    keys += [f"sib@{f}" for f in SIB_F]
                for k in keys:
                    if k == "next":
                        rep = [r["new"] for r in rs]    # the turn's own prefill
                        lost = [0 for _ in rep]
                    else:
                        v = [r[k] for r in rs if k in r]
                        rep, lost = [x[0] for x in v], [x[1] for x in v]
                    print(f"  {k:10s} {stat(rep)}   {stat(lost)}")
    path = os.environ.get("SIM_JSON")
    if path:
        with open(path, "w") as f:
            json.dump(out, f)


if __name__ == "__main__":
    main()
