#!/usr/bin/env python3
"""Per consecutive call pair of a replay run (same episode, same task, call k -> k+1): where the
next call resumed relative to the previous prompt / reply.  reply_kv_replay_pairs.py RUN.json ..."""
import json, sys, collections, statistics as st
for f in sys.argv[1:]:
    R = json.load(open(f))
    rs = sorted(R["requests"], key=lambda r: (r["episode"], r["t_start"]))
    C = collections.Counter(); pf = []; pf_first = []; saved_vs_prev = []
    for a, b in zip(rs, rs[1:]):
        if a["episode"] != b["episode"] or a["task_seq"] != b["task_seq"] or b["call"] != a["call"] + 1:
            continue
        P, c, k = a["prompt_tokens"], a["completion_tokens"], b["cached_tokens"] or 0
        if k >= P + c:
            C["after the reply (>= prompt + completion)"] += 1
        elif k >= P - 1:
            C["at the previous prompt end (prompt-1 .. prompt+completion-1)"] += 1
        elif k > 0:
            C["before the previous prompt end (slab-end checkpoint)"] += 1
        else:
            C["miss"] += 1
        pf.append(b["prompt_tokens"] - k)
    allpf = [r["prompt_tokens"] - (r["cached_tokens"] or 0) for r in rs]
    n = sum(C.values())
    print(f"{f.rsplit('/', 1)[-1]}: {len(rs)} requests, {n} continuing pairs")
    for k, v in C.most_common():
        print(f"   {v:4d} ({100 * v / n:.1f}%) {k}")
    print(f"   prefilled tokens per continuing call: mean {st.mean(pf):.0f} median {st.median(pf):.0f}; "
          f"per request (all): mean {st.mean(allpf):.0f}; completion mean {st.mean(r['completion_tokens'] for r in rs):.1f}")

# matched calls: the same (task, call index) in the first and each other run
if len(sys.argv) > 2:
    def key(f):
        return {(r["task"], r["call"]): r for r in json.load(open(f))["requests"]}
    A = key(sys.argv[1])
    for f in sys.argv[2:]:
        B = key(f)
        ks = [k for k in A if k in B and k[1] > 0]
        hs = [k for k in ks if A[k]["cached_tokens"] and B[k]["cached_tokens"]]
        ha = sum(A[k]["prompt_tokens"] - A[k]["cached_tokens"] for k in hs)
        hb = sum(B[k]["prompt_tokens"] - B[k]["cached_tokens"] for k in hs)
        print(f"matched calls that hit in both: {len(hs)}: prefilled {ha / len(hs):.0f} vs {hb / len(hs):.0f} per call "
              f"({100 * (hb - ha) / ha:+.1f}%)")
        pa = sum(A[k]["prompt_tokens"] - (A[k]["cached_tokens"] or 0) for k in ks)
        pb = sum(B[k]["prompt_tokens"] - (B[k]["cached_tokens"] or 0) for k in ks)
        sp = sum(A[k]["prompt_tokens"] == B[k]["prompt_tokens"] for k in ks)
        print(f"matched continuing calls {sys.argv[1].rsplit('/', 1)[-1]} vs {f.rsplit('/', 1)[-1]}: {len(ks)} "
              f"(same prompt length {sp}); prefilled {pa / len(ks):.0f} vs {pb / len(ks):.0f} per call "
              f"({100 * (pb - pa) / pa:+.1f}%)")
