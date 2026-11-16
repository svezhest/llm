"""Replay of the MTP head's draft-count policies over accept_probe.py's tables
(CPU, no model): per position the head's chained drafts and their top-1
probabilities on the real-entry cache (the worker's since 2026-11-17:
MTPDraft.verify / settle), the trajectory's tokens, the round costs.

  cd src/worker && PYTHONPATH=. .venv/bin/python ../../scripts/perf/mtp/gate_sim.py \\
      $PERF_OUT/mtp_head/accept.json [--exposed MS]

Policies (tokens per plain-step time = the speed-up over plain decode):
  adaptive   the removed MTPDraft.next_k (2026-11-16): per-depth acceptance
             EMAs (alpha 0.25, prior 0.5), k = argmax of E / T(k), one deeper
             or shallower every 8th round
  gate       the worker's rule (qwen_worker.mtp.MTPDraft.want, BURSTS): no
             draft when (1 + c1) / T(1) <= 1 / T(0) (a plain burst), else
             drafts while E / T(k) rises, a draft's confidence read once the
             next is queued (the last read predicts the rest)
  thr.X      drafts while each draft's confidence >= X (the first below is
             dropped: computed, not verified); none -> a one-step burst
  marg       drafts while E / T(k) rises with every confidence read at once
             (a sync per draft; the draft that stops it is dropped)
  fixedK     K drafts every round; best = the best K per cell (an oracle)
Sampled cells: the expected tokens of a sample-match verify (accept_tables.py).

Costs, in plain-step units, from the file's own measurements: a k-draft round
= the verify price list's T(k + 1) + (k - 1) chained head steps + the head over
the block's k + 1 rows (verify(), the next seed: "flush n"); a burst of L plain
steps = L + the head's catch-up over L rows. --exposed MS adds MS of host time
to every drafted round whose verify is built while no draft is computing (k = 1
rounds: the seed is computed with the round before; thr / marg rounds: after a
read) - the build exposure the live rounds may show.
"""
import argparse
import json
import statistics
import sys

from qwen_worker.mtp import MTPDraft

PRICES = {
    "today": {1: 1.0, 2: 1.29, 3: 1.6, 4: 1.7, 5: 1.8},
    "target": {1: 1.0, 2: 1.12, 3: 1.20, 4: 1.35, 5: 1.35},
}
CLASSES = ("code", "en", "ru")
KMAX = 4

ap = argparse.ArgumentParser()
ap.add_argument("accept", help="accept_probe.py's result (q4 head)")
ap.add_argument("--head", default="q4")
ap.add_argument("--exposed", type=float, default=0.0, help="ms of verify build exposed per round")
ap.add_argument("--bursts", default="1/1,2/1,2,4,8/1,1,2,4,8/2,4,8", help="burst schedules compared")
a = ap.parse_args()
R = json.load(open(a.accept))
C = R["costs"]
med = {k: statistics.median(v[C["skip"]:]) for k, v in C["ms"].items()}
DEC = med["plain step"]
STEP = {k: med[f"{a.head} draft k={k}"] - med[f"{a.head} draft k={k - 1}"] if k > 1 else 0.0
        for k in range(1, KMAX + 1)}                    # a chained head step
FL = {n: med[f"{a.head} flush n={n}"] for n in range(1, 6)}


def flush(n):
    return FL[n] if n in FL else FL[5] + (FL[5] - FL[1]) / 4 * (n - 5)


def T(price, k):
    """A k-draft round (k = 0: one plain step), plain-step units."""
    if k == 0:
        return 1.0
    return price[k + 1] + (sum(STEP[j] for j in range(2, k + 1)) + flush(k + 1)) / DEC


W = MTPDraft.__new__(MTPDraft)                          # the worker's want() / gain()


class Policy:
    bursts = (1,)

    def start(self, price):
        self.price, self.low = price, 0
        self.T = [T(price, k) for k in range(KMAX + 1)]
        return self

    def burst(self):
        self.low += 1
        return self.bursts[min(self.low, len(self.bursts)) - 1]

    def done(self, k, acc):
        pass


class Fixed(Policy):
    def __init__(self, k):
        self.k = k

    def decide(self, c):
        return self.k, self.k, self.k == 1


class Adaptive(Policy):
    def start(self, price):
        super().start(price)
        self.p, self.seen, self.r = [0.5] * KMAX, [0] * KMAX, 0
        return self

    def decide(self, c):
        best, bk, e, q, pk = -1.0, 1, 1.0, 1.0, 0.5
        for k in range(1, KMAX + 1):
            pk = self.p[k - 1] if self.seen[k - 1] else pk
            q *= pk
            e += q
            if e / self.T[k] > best:
                best, bk = e / self.T[k], k
        if self.r % 8 == 7:
            bk = min(KMAX, bk + 1) if (self.r // 8) % 2 == 0 else max(1, bk - 1)
        return bk, bk, bk == 1

    def done(self, k, acc):
        self.r += 1
        for i in range(min(acc + 1, k)):
            hit = 1.0 if i < acc else 0.0
            self.p[i] += 0.25 * (hit - self.p[i])
            self.seen[i] += 1


class Gate(Policy):
    def __init__(self, bursts=None):
        self.bursts = bursts or MTPDraft.BURSTS

    def decide(self, c):
        known = [c[0]]
        if not W.want(known, 0, self.T):
            return 0, 0, False
        self.low = 0
        k = 1
        while k < KMAX and W.want(known, k, self.T):
            k += 1
            if len(known) < k - 1:
                known.append(c[len(known)])
        return k, k, k == 1                             # (k = 1: nothing computing while the verify is built)


class Thr(Policy):
    def __init__(self, tau):
        self.tau = tau

    def decide(self, c):
        k = 0
        while k < KMAX and c[k] >= self.tau:
            k += 1
        if k == 0:
            return 0, 0, False
        self.low = 0
        return k, min(k + 1, KMAX), True


class Marg(Policy):
    def decide(self, c):
        k = 0
        while k < KMAX and W.gain(c, k + 1, c[k]) / self.T[k + 1] > W.gain(c, k, c[k]) / self.T[k]:
            k += 1
        if k == 0:
            return 0, 0, False
        self.low = 0
        return k, min(k + 1, KMAX), True


def run(texts, pol, price, mode):
    """-> (speed-up, rounds by k, burst tokens, dropped drafts)."""
    tok = ms = 0.0
    ks, bt, dropped = [0] * (KMAX + 1), 0, 0
    for t in texts:
        h, gen = t["heads"][a.head], t["gen"]
        st = pol.start(price)
        i, last = 0, len(gen) - 2 - KMAX
        while i <= last:
            k, comp, exposed = st.decide(h["conf"][i])
            dropped += comp - k
            ks[k] += 1
            if k == 0:
                L = min(st.burst(), last + 1 - i)
                tok += L
                ms += L + (flush(L) + comp * STEP[2]) / DEC
                bt += L
                i += L
                continue
            acc = 0
            while acc < k and h["chain"][i][acc] == gen[i + 1 + acc]:
                acc += 1
            if mode == "greedy":
                tok += acc + 1
            else:
                tok += 1 + sum(t["q_gen"][i + 1:i + 1 + acc]) + ((h["q"][i][acc] or 0.0) if acc < k else 0.0)
            ms += T(price, k) + ((comp - k) * STEP[2] + (a.exposed if exposed else 0.0)) / DEC
            st.done(k, acc)
            i += acc + 1
    return tok / ms, ks, bt, dropped


def texts(cls, mode):
    return [t for t in R["texts"] if t["mode"] == mode and t["cls"] == cls]


POL = {"adaptive": Adaptive(), "gate": Gate(), "thr.3": Thr(0.3), "thr.5": Thr(0.5), "thr.7": Thr(0.7),
       "marg": Marg(), **{f"fixed{k}": Fixed(k) for k in range(1, KMAX + 1)}}
print(f"# {a.accept.split('/')[-1]}: head {a.head}, plain step {DEC:.1f} ms, chained step "
      f"{STEP[2]:.2f} ms, flush n=1..5 {' / '.join(f'{FL[n]:.2f}' for n in range(1, 6))} ms, "
      f"exposed build {a.exposed} ms, gate bursts {MTPDraft.BURSTS}")
summary = {}
for pn, price in PRICES.items():
    for mode in ("greedy", "sampled"):
        res = {n: [run(texts(c, mode), p, price, mode) for c in CLASSES] for n, p in POL.items()}
        best = [max(range(1, KMAX + 1), key=lambda k: res[f"fixed{k}"][j][0]) for j in range(len(CLASSES))]
        print(f"\n## {pn} prices, {mode}: x plain (/ best fixed k)")
        print("| policy | " + " | ".join(CLASSES) + " | mean |")
        print("|---|" + "---|" * (len(CLASSES) + 1))
        for n in ("adaptive", "gate", "thr.3", "thr.5", "thr.7", "marg"):
            v = [r[0] for r in res[n]]
            b = [res[f"fixed{best[j]}"][j][0] for j in range(len(CLASSES))]
            print(f"| {n} | " + " | ".join(f"{x:.3f} ({x / y:.3f})" for x, y in zip(v, b))
                  + f" | {statistics.mean(v):.3f} ({statistics.mean(v) / statistics.mean(b):.3f}) |")
        b = [res[f"fixed{best[j]}"][j][0] for j in range(len(CLASSES))]
        print("| best fixed | " + " | ".join(f"{x:.3f} (k={k})" for x, k in zip(b, best))
              + f" | {statistics.mean(b):.3f} |")
        g = res["gate"]
        print("gate rounds by k: " + "; ".join(f"{c} {r[1]}" for c, r in zip(CLASSES, g))
              + " | burst tokens " + " / ".join(str(r[2]) for r in g)
              + " | dropped thr.5 / marg " + f"{sum(r[3] for r in res['thr.5'])} / {sum(r[3] for r in res['marg'])}")
        summary[(pn, mode)] = {n: statistics.mean(r[0] for r in v) for n, v in res.items()}
print("\n## gate burst schedules (mean over classes, x plain)")
print("| schedule | " + " | ".join(f"{p} {m}" for p, m in summary) + " |")
print("|---|" + "---|" * len(summary))
for sch in a.bursts.split("/"):
    s = tuple(int(x) for x in sch.split(","))
    v = [statistics.mean(run(texts(c, m), Gate(s), PRICES[p], m)[0] for c in CLASSES) for p, m in summary]
    print(f"| {sch} | " + " | ".join(f"{x:.3f}" for x in v) + " |")
sys.stdout.flush()
