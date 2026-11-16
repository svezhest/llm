"""Tables from accept_probe.py's result file(s) (stdlib only, no MLX):

  python3 scripts/perf/mtp/accept_tables.py <accept.json> [more.json ...] [--summary out.json]

Per head and class (texts pooled by counts):
  greedy     p_i = P(draft i == the target's token | drafts < i right), every
             position of the greedy trajectories, head cache of real entries
  sampling   the filtered probability of draft i (expected sample-match
             acceptance) given drafts < i equal the trajectory's tokens: at
             the greedy positions, and along the sampled trajectories (there
             an unbiased estimate of the sampling run's conditional acceptance)
  oracle     the same with the target's real hidden state at every depth
  rounds     tokens per round at fixed k: chained entries kept (the default,
             live simulation), recomputed (QWEN_MTP_REFEED=1: from the table)
  speed      tokens per round / (verify price(k + 1) + draft cost / decode
             step) = the speed-up over plain decode, for two verify price lists
"""
import argparse
import json
import math
import statistics

PRICES = {
    "today": {1: 1.0, 2: 1.29, 3: 1.6, 4: 1.7, 5: 1.8},
    "target": {1: 1.0, 2: 1.12, 3: 1.20, 4: 1.35, 5: 1.35},
}
CLASSES = ("code", "en", "ru")


def load(paths):
    res = None
    for p in paths:
        r = json.load(open(p))
        if res is None:
            res = r
        else:
            res["texts"] += r["texts"]
            res["costs"] = res["costs"] or r["costs"]
    return res


def rows_of(res, cls, mode):
    return [t for t in res["texts"] if t["mode"] == mode and (cls == "all" or t["cls"] == cls)]


def depth_counts(texts, head, kmax, oracle=False, src="chain"):
    """Pooled over the texts' positions with kmax tokens after them: per depth
    j (0-based) [positions whose drafts < j are the trajectory's, of them draft
    j right, sum of draft j's filtered probability over them, sum of the cosine]."""
    out = [[0, 0, 0.0, 0.0, 0] for _ in range(kmax)]
    n = 0
    for t in texts:
        gen, h = t["gen"], t["heads"][head]
        ch = h["wide"]["chain"] if src == "wide" else h["chain"]
        for i in range(len(gen) - kmax):
            n += 1
            for j in range(kmax):
                d = ch[i + j][0] if oracle else ch[i][j]
                q = h["q"][i + j][0] if oracle else h["q"][i][j]
                out[j][0] += 1
                out[j][2] += q or 0.0
                c = None if oracle else h["cos"][i][j]
                if c is not None:
                    out[j][3] += c
                    out[j][4] += 1
                if d != gen[i + 1 + j]:
                    break
                out[j][1] += 1
    return n, out


def sim_table(chain, gen, k, q):
    """Rounds of k drafts from a cache of real entries (accept_probe.sim_table)
    -> [[position - n0, accepted, the first wrong draft's filtered probability]]."""
    i, out, last = 0, [], len(gen) - 2
    while i + k <= last:
        a = 0
        while a < k and chain[i][a] == gen[i + 1 + a]:
            a += 1
        out.append([i, a, q[i][a] if a < k else None])
        i += a + 1
    return out


def pooled(texts, head, k, kind):
    """(tokens, rounds, accept-length histogram, expected tokens) of fixed-k
    rounds pooled over texts: kind "kept" (live simulation, the default mode;
    k = 1 has no chained entry: the table), "refeed" (table), "refeed_live".
    Tokens: 1 + the drafts that are the trajectory's, per round. Expected
    tokens (a sampled trajectory: of the sample-match verify, at these round
    starts): 1 + the filtered probabilities of the drafts up to and with the
    first that is not the trajectory's (an unbiased estimate, less noisy than
    the one realization)."""
    tok = rnd = 0
    exp = 0.0
    hist = [0] * (k + 1)
    for t in texts:
        h, n0 = t["heads"][head], t["n0"]
        if kind == "refeed" or (kind == "kept" and k == 1):
            r, base = sim_table(h["chain"], t["gen"], k, h["q"]), 0
        else:
            r, base = (h["sims"] if kind == "kept" else h["refeed"]).get(str(k)), n0
            if r is None:
                return None
        for p, x, q in r:
            i = p - base                # the anchor is gen[i]; draft j is right when == gen[i + 1 + j]
            tok, rnd = tok + x + 1, rnd + 1
            hist[x] += 1
            exp += 1 + sum(t["q_gen"][i + 1:i + 1 + x]) + (q or 0.0)
    return tok, rnd, hist, exp


def gated(texts, head, kmax, tau, price, cost, dec):
    """Confidence-gated draft count from the table (real-entry cache): draft
    j + 1 is made only while draft j's own top-1 probability >= tau; a draft
    below tau is dropped (0 drafts = a plain step). -> (tokens, ms / decode
    step, expected tokens of a sample-match verify as pooled())."""
    tok, ms, exp = 0, 0.0, 0.0
    for t in texts:
        h, gen = t["heads"][head], t["gen"]
        i, last = 0, len(gen) - 2
        while i + kmax <= last:
            k = 0
            while k < kmax and h["conf"][i][k] >= tau:
                k += 1
            a = 0
            while a < k and h["chain"][i][a] == gen[i + 1 + a]:
                a += 1
            tok += a + 1
            exp += 1 + sum(t["q_gen"][i + 1:i + 1 + a]) + ((h["q"][i][a] or 0.0) if a < k else 0.0)
            ms += price[k + 1] + cost[min(k + 1, kmax)] / dec
            i += a + 1
    return tok, ms, exp


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def se(p, n):
    return math.sqrt(max(p * (1 - p), 0.0) / n) if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+")
    ap.add_argument("--summary", default="")
    ap.add_argument("--taus", default="0.3,0.5,0.7,0.9")
    a = ap.parse_args()
    res = load(a.json)
    heads = list(res["heads"])
    kmax = res["args"]["kmax"]
    S = dict(model=res["model"], heads=res["heads"], sampling=res["sampling"], kmax=kmax, texts=[], accept={},
             rounds={}, speed={}, costs={})
    print(f"model {res['model']}, heads " + ", ".join(f"{n} = {i['dir']} ({i['mib']:.0f} MiB)"
                                                       for n, i in res["heads"].items()))
    print("\n## texts\n\n| text | class | trajectory | prompt | generated | target's mean filtered p of its token |"
          "\n|---|---|---|---|---|---|")
    for t in res["texts"]:
        S["texts"].append(dict(key=t["key"], cls=t["cls"], mode=t["mode"], prompt=t["n0"], generated=len(t["gen"])))
        print(f"| {t['key']} | {t['cls']} | {t['mode']} | {t['n0']} | {len(t['gen'])} | "
              f"{statistics.mean(t['q_gen']):.3f} |")

    # ---- per-depth conditional acceptance
    for mode, what in (("greedy", "greedy trajectories"), ("sampled", "sampled trajectories")):
        if not rows_of(res, "all", mode):
            continue
        print(f"\n## per-depth conditional acceptance, {what}\n")
        print("`match` = draft i equals the trajectory's token; `q` = mean filtered probability of draft i "
              "(expected sample-match acceptance); both given drafts < i equal the trajectory's tokens; n = "
              "positions in the condition.\n")
        print("| class | head | " + " | ".join(f"match p{j + 1} (n)" for j in range(kmax)) + " | "
              + " | ".join(f"q s{j + 1}" for j in range(kmax)) + " | E drafts match / q |")
        print("|---|---|" + "---|" * (2 * kmax + 1))
        for cls in CLASSES + ("all",):
            texts = rows_of(res, cls, mode)
            if not texts:
                continue
            for hd in heads:
                for oracle in (False, True):
                    n, c = depth_counts(texts, hd, kmax, oracle)
                    p = [x[1] / x[0] if x[0] else None for x in c]
                    q = [x[2] / x[0] if x[0] else None for x in c]
                    em, eq = sum(x[1] for x in c) / n, sum(x[2] for x in c) / n
                    name = hd + (" oracle" if oracle else "")
                    S["accept"][f"{mode}/{cls}/{name}"] = dict(
                        positions=n, n=[x[0] for x in c], match=[x[1] for x in c], q_sum=[round(x[2], 3) for x in c],
                        cos=[round(x[3] / x[4], 4) if x[4] else None for x in c])
                    print(f"| {cls} | {name} | " + " | ".join(f"{f3(p[j])} ({c[j][0]})" for j in range(kmax))
                          + " | " + " | ".join(f3(v) for v in q) + f" | {em:.3f} / {eq:.3f} |")

    # ---- heads against each other on the same positions
    texts = rows_of(res, "all", "greedy")
    if len(heads) > 1 and texts:
        print("\n## heads on the same positions (greedy trajectories, all classes)\n")
        print("| pair | depth-1: A right B wrong / B right A wrong | mean accepted of "
              f"{kmax} drafts A − B (paired, ± se) |\n|---|---|---|")
        for x in range(len(heads)):
            for y in range(x + 1, len(heads)):
                A, B = heads[x], heads[y]
                ab = ba = 0
                diffs = []
                for t in texts:
                    gen = t["gen"]
                    for i in range(len(gen) - kmax):
                        la = lb = 0
                        ca, cb = t["heads"][A]["chain"][i], t["heads"][B]["chain"][i]
                        while la < kmax and ca[la] == gen[i + 1 + la]:
                            la += 1
                        while lb < kmax and cb[lb] == gen[i + 1 + lb]:
                            lb += 1
                        ab += la > 0 and lb == 0
                        ba += lb > 0 and la == 0
                        diffs.append(la - lb)
                m = statistics.mean(diffs)
                s = statistics.stdev(diffs) / math.sqrt(len(diffs))
                S["accept"][f"pair/{A}-{B}"] = dict(a_only=ab, b_only=ba, mean_diff=m, se=s, n=len(diffs))
                print(f"| {A} vs {B} | {ab} / {ba} | {m:+.4f} ± {s:.4f} (n {len(diffs)}) |")

    # ---- the stand-in's quality
    if texts:
        print("\n## the chained stand-in: cosine of the head's output hidden with the target's hidden state it "
              "replaces (drafts so far right)\n\n| class | head | " + " | ".join(f"depth {j + 1}" for j in range(kmax))
              + " |\n|---|---|" + "---|" * kmax)
        for cls in CLASSES:
            for hd in heads:
                if rows_of(res, cls, "greedy"):
                    c = S["accept"][f"greedy/{cls}/{hd}"]["cos"]
                    print(f"| {cls} | {hd} | " + " | ".join(f3(v) for v in c) + " |")

    # ---- fp32 activations
    wide = [t for t in res["texts"] if any("wide" in h for h in t["heads"].values())]
    print("\n## fp32 activations (the table again with every activation fp32)\n")
    for hd in heads:
        bad = sum(t["heads"][hd]["nonfinite"] for t in res["texts"])
        line = f"- {hd}: non-finite head outputs in fp16: {bad}"
        if wide:
            same = tot = 0
            for t in wide:
                h = t["heads"][hd]
                for r1, r2 in zip(h["chain"], h["wide"]["chain"]):
                    same += r1 == r2
                    tot += 1
            _, c16 = depth_counts(wide, hd, kmax)
            _, c32 = depth_counts(wide, hd, kmax, src="wide")
            line += (f"; chains identical to fp32 at {same}/{tot} positions; drafts right fp16 "
                     f"{[x[1] for x in c16]} vs fp32 {[x[1] for x in c32]} (n {[x[0] for x in c16]} / "
                     f"{[x[0] for x in c32]}); fp32 non-finite {sum(t['heads'][hd]['wide']['nonfinite'] for t in wide)}")
            S["accept"][f"wide/{hd}"] = dict(same=same, positions=tot, fp16=[x[1] for x in c16],
                                             fp32=[x[1] for x in c32], nonfinite_fp16=bad)
        print(line)

    # ---- costs
    T = res.get("costs") or {}
    dec = None
    if T:
        sk = T["skip"]
        med = {k: statistics.median(v[sk:]) for k, v in T["ms"].items()}
        dec = med["plain step"]
        S["costs"] = dict(ctx=T["ctx"], median_ms={k: round(v, 3) for k, v in med.items()})
        nmax = max(int(k.split("=")[1]) for k in med if " flush n=" in k)
        print(f"\n## costs (synced stages, medians of {len(T['ms']['plain step']) - sk}, context {T['ctx']}; "
              f"plain decode step {dec:.1f} ms)\n")
        print("| head | forward 1 row | lm_head | " + " | ".join(f"draft k={k}" for k in range(1, kmax + 1)) + " | "
              + " | ".join(f"flush n={n}" for n in range(1, nmax + 1)) + " |")
        print("|---|---|---|" + "---|" * (kmax + nmax))
        for hd in heads:
            print(f"| {hd} | {med[f'{hd} forward 1 row']:.2f} | {med[f'{hd} lm_head']:.2f} | "
                  + " | ".join(f"{med[f'{hd} draft k={k}']:.2f}" for k in range(1, kmax + 1)) + " | "
                  + " | ".join(f"{med[f'{hd} flush n={n}']:.2f}" for n in range(1, nmax + 1)) + " |")

    # ---- tokens per round and the speed model
    for mode in ("greedy", "sampled"):
        if not rows_of(res, "all", mode):
            continue
        print(f"\n## tokens per round at fixed k, {mode} trajectories (kept = the default's chained entries, "
              "live; refeed = recomputed, from the table; (rounds)"
              + ("; expected of a sample-match verify [the one realization]" if mode == "sampled" else "") + ")\n")
        print("| class | head | " + " | ".join(f"kept k={k}" for k in range(1, kmax + 1)) + " | "
              + " | ".join(f"refeed k={k}" for k in range(1, kmax + 1)) + " | refeed live |")
        print("|---|---|" + "---|" * (2 * kmax + 1))
        for cls in CLASSES + ("all",):
            texts = rows_of(res, cls, mode)
            if not texts:
                continue
            for hd in heads:
                cells = []
                for kind in ("kept", "refeed"):
                    for k in range(1, kmax + 1):
                        r = pooled(texts, hd, k, kind)
                        if r is not None:
                            S["rounds"][f"{mode}/{cls}/{hd}/{kind}/{k}"] = dict(
                                tokens=r[0], rounds=r[1], hist=r[2], expected=round(r[3], 2))
                        cells.append("—" if r is None else f"{r[0] / r[1]:.3f} ({r[1]})" if mode == "greedy"
                                     else f"{r[3] / r[1]:.3f} [{r[0] / r[1]:.3f}] ({r[1]})")
                live = [f"k={k}: {r[0] / r[1]:.3f}" for k in range(1, kmax + 1)
                        for r in [pooled(texts, hd, k, "refeed_live")] if r is not None]
                print(f"| {cls} | {hd} | " + " | ".join(cells) + f" | {', '.join(live) or '—'} |")
    if dec:
        for mode in ("greedy", "sampled"):
            for sc, price in PRICES.items():
                if not rows_of(res, "all", mode):
                    continue
                print(f"\n## speed-up over plain decode, {mode}, verify prices \"{sc}\" "
                      f"({', '.join(f'T={t} x{v}' for t, v in price.items() if t > 1)}; draft cost / decode step "
                      f"{dec:.1f} ms)\n")
                print("| class | head | " + " | ".join(f"k={k}" for k in range(1, kmax + 1)) + " | best | "
                      + " | ".join(f"refeed k={k}" for k in range(1, kmax + 1)) + " | "
                      + " | ".join(f"gate {tau}" for tau in a.taus.split(",")) + " |")
                print("|---|---|" + "---|" * (2 * kmax + 1 + len(a.taus.split(","))))
                for cls in CLASSES + ("all",):
                    texts = rows_of(res, cls, mode)
                    if not texts:
                        continue
                    for hd in heads:
                        cost = {k: med[f"{hd} draft k={k}"] for k in range(1, kmax + 1)}
                        step = (cost[kmax] - cost[1]) / max(1, kmax - 1)
                        out, best = [], (0.0, 0)
                        for kind in ("kept", "refeed"):
                            for k in range(1, kmax + 1):
                                r = pooled(texts, hd, k, kind)
                                if r is None:
                                    out.append("—")
                                    continue
                                if kind == "kept":      # one pending row, or two after a full accept
                                    c = cost[k] + (med[f"{hd} flush n=2"] - med[f"{hd} flush n=1"]) * r[2][k] / r[1]
                                else:                   # the accepted positions fed again: a + 1 rows
                                    fl = sum(med[f"{hd} flush n={x + 1}"] * r[2][x] for x in range(k + 1)) / r[1]
                                    c = fl + (k - 1) * step
                                v = ((r[0] if mode == "greedy" else r[3]) / r[1]) / (price[k + 1] + c / dec)
                                S["speed"][f"{mode}/{sc}/{cls}/{hd}/{kind}/{k}"] = round(v, 4)
                                out.append(f"{v:.3f}")
                                if kind == "kept" and v > best[0]:
                                    best = (v, k)
                            if kind == "kept":
                                out.append(f"**{best[0]:.3f}** (k={best[1]})")
                        for tau in a.taus.split(","):
                            g = gated(texts, hd, kmax, float(tau), price, cost, dec)
                            v = (g[0] if mode == "greedy" else g[2]) / g[1]
                            S["speed"][f"{mode}/{sc}/{cls}/{hd}/gate/{tau}"] = round(v, 4)
                            out.append(f"{v:.3f}")
                        print(f"| {cls} | {hd} | " + " | ".join(out) + " |")
    if a.summary:
        json.dump(S, open(a.summary, "w"), indent=1)
        print("\nwrote", a.summary)


if __name__ == "__main__":
    main()
