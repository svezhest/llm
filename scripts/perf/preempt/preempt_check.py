#!/usr/bin/env python3
"""GPU check of QWEN_PREEMPT and QWEN_DEBUG_NO_SYNC, in ONE worker process.
NEEDS THE GPU LEASE (loads its own 9B copy; peak ~12-13 GB).

Phase 0 (guard): worker.DEBUG_NO_SYNC on, 8 rows (greedy, seeded,
  penalties, logprobs; B = 8 -> mrqmm, GDN fuse, grouped sampling) and then 1
  row alone: no SyncInStep anywhere (Metal paths of _step). PASS: no error.
Phase 1 (exactness, B <= 2 = the bit-exact per-row qmv path): max_batch 2,
  two ignore_eos rows A (prompt ~1.5K, 2500 tokens, logprobs top 3) and B
  (~2.5K, 2600 tokens, logprobs). Reference: QWEN_PREEMPT off. Run:
  QWEN_PREEMPT=2 with A's growth past its 1024-token reservation
  (QWEN_BATCH_RESERVE_TOKENS) forced to fail the headroom twice (`force`: the
  memory decision isolated; the real budget path is phase 2 and the CPU
  tests): A is preempted, resumes from its entry beside B, is preempted again,
  resumes after B ends. Greedy, then seeded sampling (T 0.7, seed 5).
  PASS: 2 preemptions logged, every stream token-identical to the
  reference, the text identical, logprobs bitwise equal, finish "length" at
  max_tokens, Usage.prompt_tokens / cached_tokens = the reference's.
Phase 2 (informational): max_batch 4, four rows, and once they all run the
  REAL budget is cut to what they hold + PRE_SLACK_MB (16), prefill reserve
  0: growth evicts the prompt cache first, then preempts; with the cache
  empty a preempted row's entry may not be kept (-> a recompute). Reports
  the preemptions, completeness (every row reaches max_tokens: nothing cut)
  and how many streams stay identical (B >= 3 is not batch-invariant, and a
  recompute is prefill bits).

  cd src/worker && setopt NO_BG_NICE; mkdir -p /tmp/perf; \\
  QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 QWEN_PROMPT_CACHE=1 \\
  QWEN_PROMPT_CACHE_GB=6 QWEN_BATCH_RESERVE_TOKENS=1024 QWEN_PRESENCE_PENALTY=0 \\
  PERF_OUT=/tmp/perf PYTHONPATH=. \\
  .venv/bin/python ../../scripts/perf/preempt/preempt_check.py 2>&1 | tee /tmp/perf/preempt_check.log
"""
import json
import logging
import os
import random
import threading
import time

import worker
import worker_pb2 as pb

OUT = os.environ.get("PERF_OUT", "/tmp/perf")
SLACK = int(float(os.environ.get("PRE_SLACK_MB", "16")) * (1 << 20))
OFF = json.dumps({"enable_thinking": False})
WORDS = ("name id status amount date user playlist song album artist email phone count total "
         "created updated access token page limit offset title description value price").split()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Count(logging.Handler):
    def __init__(self):
        super().__init__()
        self.preempted, self.sync = [], []

    def emit(self, rec):
        m = rec.getMessage()
        if "preempted at" in m:
            self.preempted.append(m)
        if "SyncInStep" in m or "QWEN_DEBUG_NO_SYNC" in m:
            self.sync.append(m)


def filler(seed, n_tok, tok):
    rnd = random.Random(seed)
    lines = [f"Output {seed}:"]
    while len(tok.encode("\n".join(lines))) < n_tok:
        lines += [f"{rnd.choice(WORDS)}_{seed} = {rnd.randrange(10**6)}" for _ in range(16)]
    return "\n".join(lines)


def req(rid, text, max_tokens, **kw):
    p = dict(max_tokens=max_tokens, temperature=0.0, presence_penalty=0.0, chat_template_kwargs_json=OFF)
    p.update(kw)
    return pb.Request(id=rid, messages=[pb.Message(role="user", content=text)], params=pb.Params(**p))


def call(eng, r, out):
    toks, lps, text = [], [], []
    for ev in eng.generate(r):
        k = ev.WhichOneof("kind")
        if k == "token":
            text.append(ev.token.text)
            for t in ev.token.lp:
                toks.append(t.token_id)
                lps.append((t.logprob, [(x.token_id, x.logprob) for x in t.top]))
        elif k == "done":
            u = ev.done.usage
            out.update(finish=ev.done.finish, gen=u.gen_tokens, prompt=u.prompt_tokens, cached=u.cached_tokens)
        elif k == "error":
            out.update(error=f"{ev.error.kind}: {ev.error.msg}")
    out.update(text="".join(text), toks=toks, lps=lps)


def batch(eng, reqs):
    outs = {r.id: {} for r in reqs}
    ths = [threading.Thread(target=call, args=(eng, r, outs[r.id])) for r in reqs]
    for t in ths:
        t.start()
        time.sleep(0.2)                         # admission order = list order
    for t in ths:
        t.join()
    return outs


def squeeze(sched, rows):
    """Once `rows` rows run, cut the budget to what they hold + SLACK (on the
    scheduler thread, in _join); prefill reserve 0 (the prompts are short)."""
    orig = sched._join
    saved = (sched.budget, sched.prefill_reserve)

    def join(row):
        orig(row)
        if len(sched.rows) == rows:
            sched.prefill_reserve = 0
            sched.budget = sched.reserved + sched.eng._pcache.bytes + SLACK
            log(f"  budget cut to {sched.budget / 2**30:.3f} GiB ({len(sched.rows)} rows)")
    sched._join = join

    def undo():
        sched._join = orig
        sched.budget, sched.prefill_reserve = saved
    return undo


def force(sched, rid, times=1):
    """The growth of row `rid` fails the headroom the first `times` times it
    outgrows its reservation while another row runs (as a too-small budget
    would, without the prompt cache giving way first): the pre-step check ->
    preempt (the step's own check would cut it). Patched on the instance; the
    scheduler thread picks it up at its next call."""
    orig, n = sched._regrow, [0]

    def regrow(row, ahead=0):
        if (row.req.id.startswith(rid) and n[0] < times and row.pos + ahead >= row.tokens
                and sum(not r.done for r in sched.rows) > 1):
            n[0] += 1
            if ahead:
                return False
            sched._finish(row, "length")
            return False
        return orig(row, ahead)
    sched._regrow = regrow

    def undo():
        del sched._regrow
    return undo


def compare(ref, got):
    out = {}
    for k in ref:
        a, b = ref[k], got[k]
        out[k] = {"tokens_equal": a["toks"] == b["toks"], "text_equal": a["text"] == b["text"],
                  "lps_equal": a["lps"] == b["lps"], "n": len(b["toks"]), "finish": b.get("finish"),
                  "usage_equal": (a.get("prompt"), a.get("cached")) == (b.get("prompt"), b.get("cached")),
                  "error": b.get("error"), "first_diff": next(
                      (i for i, (x, y) in enumerate(zip(a["toks"], b["toks"])) if x != y), None)}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    cnt = Count()
    logging.getLogger("qwen.worker").addHandler(cnt)
    eng = worker.Engine(worker.DEFAULT_MODEL)
    eng.load()
    tok = eng.processor.tokenizer
    sched = eng.sched
    pc = eng._pcache
    res = {}

    # --- phase 0: the guard on Metal --------------------------------------------
    worker.DEBUG_NO_SYNC = True
    kinds = [dict(), dict(temperature=0.7, top_p=0.9, top_k=20, seed=3),
             dict(presence_penalty=0.5, frequency_penalty=0.2, repetition_penalty=1.1),
             dict(logprobs=True, top_logprobs=5)]
    rs = [req(f"g{i}", filler(10 + i, 600, tok), 64, **kinds[i % 4]) for i in range(8)]
    o8 = batch(eng, rs)
    o1 = batch(eng, [req("g-solo", filler(20, 600, tok), 64, logprobs=True)])
    worker.DEBUG_NO_SYNC = False
    errs = [o.get("error") for o in list(o8.values()) + list(o1.values()) if o.get("error")]
    res["phase0"] = {"errors": errs, "sync_logs": cnt.sync[:5], "PASS": not errs and not cnt.sync}
    log("phase 0 (guard):", res["phase0"])

    # --- phase 1: exactness at max_batch 2 ---------------------------------------
    sched.max_batch = 2
    res["phase1"] = {}
    for name, kw in (("greedy", {}), ("seeded", dict(temperature=0.7, top_p=0.95, seed=5))):
        def make():
            return [req("A", "Continue this log:\n" + filler(31, 1500, tok), 2500, ignore_eos=True,
                        logprobs=True, top_logprobs=3, **kw),
                    req("B", "Continue this log:\n" + filler(32, 2500, tok), 2600, ignore_eos=True,
                        logprobs=True, **kw)]
        pc.entries = []
        pc._recount()
        worker.PREEMPT = 0
        ref = batch(eng, make())
        pc.entries = []
        pc._recount()
        worker.PREEMPT = 2
        n0 = len(cnt.preempted)
        undo = force(sched, "A", times=2)          # twice: A resumes beside B, then alone
        try:
            got = batch(eng, make())
        finally:
            undo()
        cmp = compare(ref, got)
        pre = cnt.preempted[n0:]
        ok = (len(pre) == 2 and all(c["tokens_equal"] and c["text_equal"] and c["lps_equal"] and c["usage_equal"]
                                and c["finish"] == "length" and not c["error"] for c in cmp.values())
              and cmp["A"]["n"] == 2500 and cmp["B"]["n"] == 2600)
        res["phase1"][name] = {"preemptions": pre, "compare": cmp, "PASS": ok}
        log(f"phase 1 {name}: {len(pre)} preemptions;",
            {k: (c["tokens_equal"], c["lps_equal"], c["n"], c["finish"]) for k, c in cmp.items()},
            "PASS" if ok else "FAIL")

    # --- phase 2: informational, max_batch 4 -------------------------------------
    sched.max_batch = 4

    def make4():
        return [req(f"Q{i}", "Continue this log:\n" + filler(40 + i, 1200 + 400 * i, tok), 2000,
                    ignore_eos=True, logprobs=True) for i in range(4)]     # (logprobs: the token ids)
    pc.entries = []
    pc._recount()
    worker.PREEMPT = 0
    ref = batch(eng, make4())
    pc.entries = []
    pc._recount()
    worker.PREEMPT = 2
    n0 = len(cnt.preempted)
    undo = squeeze(sched, 4)
    try:
        got = batch(eng, make4())
    finally:
        undo()
    cmp = compare(ref, got)
    res["phase2"] = {"preemptions": cnt.preempted[n0:], "compare": cmp,
                     "complete": all(c["n"] == 2000 and c["finish"] == "length" for c in cmp.values())}
    log("phase 2:", len(cnt.preempted) - n0, "preemptions;",
        {k: (c["tokens_equal"], c["first_diff"], c["n"]) for k, c in cmp.items()})

    res["PASS"] = res["phase0"]["PASS"] and all(v["PASS"] for v in res["phase1"].values()) \
        and res["phase2"]["complete"]
    with open(os.path.join(OUT, "preempt_check.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    log("RESULT:", "PASS" if res["PASS"] else "FAIL", "->", os.path.join(OUT, "preempt_check.json"))
    os._exit(0)          # the scheduler thread never returns (see _Scheduler._run)


if __name__ == "__main__":
    main()
