"""Per-depth acceptance of MTP heads on the real target, several heads on the
SAME positions, one process, one target load (the MTP analogue of
scripts/perf/dflash2/draft_fast.py). No scheduler: the target runs once per
text, the heads replay its hidden states through the worker's own MTPDraft.

Per text (the bench27 prompts + the two short prose prompts of
dflash2/prompts2k.py) two trajectories of the target: its greedy continuation
and one sampled continuation (temperature / top_p / top_k below, the worker's
sampler, one seed). Kept per position: the final hidden state (what _spec
pushes to the head) and the request sampler's filtered distribution (top 20).

Per head and trajectory:
  table   at EVERY generated position p the chain of --kmax drafts from a head
          cache whose entries all come from the target's real hidden states
          (after each probe the chained entries are trimmed and the position
          is fed its real hidden state: QWEN_MTP_REFEED=1's cache at any round
          start). Per draft: token, the head's own top-1 probability, the
          sampler's filtered probability of the token (the expected
          acceptance of a sample-match verify; exact while the earlier drafts
          are the trajectory's tokens), cosine of the head's output hidden
          with the target's hidden state it stands in for.
  oracle  the depth-i draft the head would give with the target's real hidden
          state at every depth = the table's depth-1 draft i - 1 positions
          later (no extra forward): the upper bound of any stand-in.
  sim k   the rounds as _spec runs them at a fixed k (MTPDraft.draft /
          round_end / push / emit along the trajectory): the chained entries
          of accepted drafts kept (the default) or recomputed (--refeed-sim);
          per round its start, the drafts accepted (= the trajectory's) and
          the filtered probability of the first one that is not.
  wide    (--wide) the table again with fp32 activations (weights as loaded,
          scales / norms / inputs fp32): an fp16 overflow shows as a
          different chain; + a finite check of every head output.
Costs (--cost-iters): the head's draft(k), an n-row flush, a 1-row forward and
the lm_head read per head, interleaved with a plain decode step, each synced,
medians.

  cd src/worker && QWEN_MODEL=$WEIGHTS/qwen38-27b-mlx-oq4e QWEN_TEXT_ONLY=1 QWEN_MRQMM4=1 \\
    QWEN_MRQMM5=1 QWEN_MRQMM6=1 QWEN_KV_BITS=0 QWEN_MAX_BATCH=1 QWEN_WIRED_LIMIT_GB=27 \\
    QWEN_PRESENCE_PENALTY=0 QWEN_DRAFT=$WEIGHTS/qwen38-27b-mtp QWEN_MTP_K=0 QWEN_VERIFY_QMV=3 \\
    PYTHONPATH=. .venv/bin/python ../../scripts/perf/mtp/accept_probe.py \\
    --heads q4=$WEIGHTS/qwen38-27b-mtp,q8=$WEIGHTS/qwen38-27b-mtp-q8,fp16=$WEIGHTS/qwen38-27b-mtp-bf16 \\
    --traj $PERF_OUT/mtp_head/traj --json $PERF_OUT/mtp_head/accept.json

Tables: scripts/perf/mtp/accept_tables.py <json>.
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.getcwd())
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import worker as W  # noqa: E402
from qwen_worker.mtp import MTPDraft  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
TOP = 20                                # filtered probabilities kept per position


class Probe(MTPDraft):
    """MTPDraft that also keeps, per draft, the head's top-1 probability and
    its output hidden state (`tr`); `wide`: fp32 activations."""

    wide = False
    trace = True                        # False: MTPDraft's own _argmax (the cost stages)

    def start(self, at):
        super().start(at)
        self.tr = []
        self.bad = mx.array(0)          # non-finite elements of the head's outputs

    def _forward(self, toks, h):
        if not self.wide:
            return super()._forward(toks, h)
        n = toks.shape[1]
        emb = self.lm.model.embed_tokens(toks).astype(mx.float32)
        pid = (mx.arange(n, dtype=mx.int32) + self.pos)[None]
        out = self.model._forward_hidden(emb, h.astype(mx.float32), self.cache, pid)
        self.pos += n
        return out

    def _argmax(self, h):
        if not self.trace:
            return super()._argmax(h)
        lg = self.lm.lm_head(h.astype(mx.float16) if self.wide else h)
        l32 = lg.astype(mx.float32)
        conf = mx.exp(mx.max(l32, axis=-1) - mx.logsumexp(l32, axis=-1)).reshape(-1)[-1:]
        self.tr.append((conf, h[0, -1]))
        self.bad = self.bad + (~mx.isfinite(h)).sum()
        return mx.argmax(lg, axis=-1).astype(mx.int32)


def head_start(mt, seq, n0, H, slab=2048):
    """The head's cache over the prompt, as _spec's prefill: entries of the
    pairs (H[t], seq[t + 1]), t < n0 (seq[n0] = the first generated token)."""
    mt.start(0)
    for s in range(0, n0, slab):
        e = min(n0, s + slab)
        mt.push(H[:, s:e])
        mt.emit_ids(seq[s + 1:e + 1])
        mt.flush()
        mx.eval(mt.state())


def chain(mt, k, ref=None):
    """mt.draft(k), evaluated -> (tokens [k], the head's top-1 probabilities
    [k], cosines with the rows of `ref` ((<= k, H): the target's hidden
    states the draft hiddens stand in for) or None)."""
    d = mt.draft(k)
    tr, mt.tr = mt.tr[-k:], []
    conf = mx.concatenate([c for c, _ in tr])
    cos = None
    if ref is not None and ref.shape[0]:
        o = mx.stack([h for _, h in tr[:ref.shape[0]]]).astype(mx.float32)
        r = ref.astype(mx.float32)
        cos = (o * r).sum(-1) / mx.sqrt((o * o).sum(-1) * (r * r).sum(-1))
    mx.eval(d, conf, mt.bad, [] if cos is None else cos, [c.state for c in mt.cache])
    return ([int(t) for t in d[0].tolist()], [float(c) for c in conf.tolist()],
            None if cos is None else [float(c) for c in cos.tolist()])


def table(mt, seq, n0, H, kmax):
    """The chain of kmax drafts at every position p in [n0, len(H)) from a
    cache of real entries only -> {chain, conf, cos} (rows by p - n0)."""
    head_start(mt, seq, n0, H)
    out = dict(chain=[], conf=[], cos=[])
    for p in range(n0, H.shape[1]):
        toks, conf, cos = chain(mt, kmax, H[0, p:p + kmax])
        mt.round_end(0, H[:, p:p + 1])  # 0 kept: every chained entry trimmed
        mt.emit(seq[p + 1])
        out["chain"].append(toks)
        out["conf"].append(conf)
        out["cos"].append(cos + [None] * (kmax - len(cos)))
    out["nonfinite"] = int(mt.bad.item())
    return out


def sim(mt, seq, n0, H, k, qf=None):
    """Rounds of k drafts along the trajectory, the head driven as _spec
    drives it (W.MTP_REFEED as set) -> [[p, accepted, q], ...]; q = qf(position,
    token) of the first draft that is not the trajectory's token (its filtered
    probability; None when all k are)."""
    head_start(mt, seq, n0, H)
    p, last, out = n0, H.shape[1] - 1, []
    while p + k <= last:
        toks, _, _ = chain(mt, k)
        a = 0
        while a < k and toks[a] == seq[p + 1 + a]:
            a += 1
        mt.round_end(a, H[:, p:p + a + 1])
        mt.emit_ids(seq[p + 1:p + a + 2])
        mt.record(a, k)
        q = qf(p + a, toks[a]) if qf is not None and a < k else None
        out.append([p, a, None if q is None else round(q, 4)])
        p += a + 1
    return out


def sim_table(chains, seq, n0, k, last):
    """sim() with a cache of real entries only (QWEN_MTP_REFEED=1), from the
    table's chains: no forward."""
    p, out = n0, []
    while p + k <= last:
        toks, a = chains[p - n0], 0
        while a < k and toks[a] == seq[p + 1 + a]:
            a += 1
        out.append([p, a])
        p += a + 1
    return out


def trajectory(lm, ids, gen, pick, flt, stop, slab=2048):
    """The target over `ids`, then up to gen + 1 steps of its own tokens
    (`pick`: logits (1, V) -> token array) -> dict(seq, n0, H (1, len(seq) - 1,
    H) numpy fp16: the post-norm hidden state of every fed position, as _spec
    hands them to the head; top_ids / top_p (steps, TOP): `flt`'s filtered
    distribution of the token after positions n0 - 1.., pmax: the unfiltered
    top-1 probability, arg: the argmax token)."""
    from mlx_vlm.models.cache import make_prompt_cache
    cache = make_prompt_cache(lm)
    lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int32)      # as _spec (text)
    lm._position_ids = None
    n0 = len(ids)
    x = mx.array([ids], dtype=mx.int32)
    hs = []
    for s in range(0, n0 - 1, slab):
        last = lm.model(x[:, s:min(n0 - 1, s + slab)], cache=cache, capture_layer_ids=[], hidden_sink=[])
        mx.eval(last, [c.state for c in cache])
        hs.append(np.array(last))
        mx.clear_cache()
    seq, tok = list(ids), ids[-1]
    top_ids, top_p, pmax, arg = [], [], [], []
    for _ in range(gen + 1):
        out = lm(mx.array([[tok]], dtype=mx.int32), cache=cache, capture_layer_ids=[], return_hidden=True)
        lg = out.logits[:, -1, :]
        nxt = pick(lg)
        l32 = lg.astype(mx.float32)
        pm = mx.exp(mx.max(l32, axis=-1) - mx.logsumexp(l32, axis=-1))
        am = mx.argmax(lg, axis=-1)
        pr = mx.softmax(flt(lg), axis=-1)
        ti = mx.argsort(-pr, axis=-1)[..., :TOP]
        tp = mx.take_along_axis(pr, ti, axis=-1)
        h = out.hidden_states[-1]
        mx.eval(nxt, pm, am, ti, tp, h, [c.state for c in cache])
        hs.append(np.array(h))
        top_ids.append(np.array(ti[0]))
        top_p.append(np.array(tp[0]))
        pmax.append(float(pm.item()))
        arg.append(int(am.item()))
        tok = int(nxt.item())
        seq.append(tok)
        if stop(tok):
            break
    del cache
    mx.clear_cache()
    return dict(seq=np.array(seq, dtype=np.int64), n0=n0, H=np.concatenate(hs, axis=1),
                top_ids=np.stack(top_ids).astype(np.int64), top_p=np.stack(top_p).astype(np.float32),
                pmax=np.array(pmax, dtype=np.float32), arg=np.array(arg, dtype=np.int64))


def filtered_prob(tr, p, tok):
    """The filtered probability of `tok` as the token after position p
    (p >= n0 - 1), None past the trajectory's logits."""
    j = p - (tr["n0"] - 1)
    if not 0 <= j < len(tr["top_ids"]):
        return None
    hit = np.nonzero(tr["top_ids"][j] == tok)[0]
    return float(tr["top_p"][j][hit[0]]) if len(hit) else 0.0


def widen(model):
    """fp32 activations: every floating parameter (norms, quantization scales
    / biases, unquantized weights) to fp32; the quantized weights stay."""
    model.apply(lambda a: a.astype(mx.float32) if a.dtype in (mx.bfloat16, mx.float16) else a)
    mx.eval(model.parameters())
    return model


def costs(lm, heads, seq, n0, H, iters, kmax, nmax=5):
    """Synced stage times (ms, every sample) of each head beside a plain
    decode step of the target at this context, interleaved."""
    from mlx_vlm.models.cache import make_prompt_cache
    cache = make_prompt_cache(lm)
    lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int32)
    lm._position_ids = None
    x = mx.array([[int(t) for t in seq[:n0]]], dtype=mx.int32)
    for s in range(0, n0, 2048):
        mx.eval(lm.model(x[:, s:s + 2048], cache=cache), [c.state for c in cache])
    for mt in heads.values():
        mt.trace = False
        head_start(mt, seq, n0, H)
    b = mx.array([[int(seq[n0])]], dtype=mx.int32)
    hrow = H[:, n0:n0 + 1]
    T = {}

    def tm(name, f):
        t = time.perf_counter()
        mx.eval(f())
        T.setdefault(name, []).append((time.perf_counter() - t) * 1000)

    def plain():
        o = lm(b, cache=cache, capture_layer_ids=[], return_hidden=True)
        return mx.argmax(o.logits[:, -1, :], axis=-1), o.hidden_states[-1], [c.state for c in cache]

    def feed(mt, n):                    # n pending pairs (real hidden states)
        mt.round_end(0, None)
        mt.push(mx.repeat(hrow, n, axis=1))
        mt.emit_ids([int(seq[n0])] * n)

    def flush(mt):
        mt.flush()
        return mt.state()

    names = list(heads)
    for it in range(iters):
        tm("plain step", plain)
        for name in names[it % len(names):] + names[:it % len(names)]:
            mt = heads[name]
            for k in range(1, kmax + 1):
                feed(mt, 1)
                tm(f"{name} draft k={k}", lambda: (mt.draft(k), [c.state for c in mt.cache]))
            for n in range(1, nmax + 1):
                feed(mt, n)
                tm(f"{name} flush n={n}", lambda: flush(mt))
            feed(mt, 1)
            mx.eval(flush(mt))
            tok, h = mt.seed
            tm(f"{name} forward 1 row", lambda: (mt._forward(tok, h), [c.state for c in mt.cache]))
            mt.appended = 1
            tm(f"{name} lm_head", lambda: mt._argmax(h))
    for mt in heads.values():
        mt.trace = True
    del cache
    mx.clear_cache()
    return T


def r4(rows):
    return [[None if v is None else round(v, 4) for v in r] for r in rows]


def ints(s):
    return [int(x) for x in s.split(",") if x]


def probe_row(heads, key, cls, mode, tr, kmax, sims, refeeds, wide=None):
    """One trajectory through every head -> its row of the result file."""
    seq, n0 = [int(t) for t in tr["seq"]], int(tr["n0"])
    H = mx.array(tr["H"])
    mx.eval(H)
    def qf(p, t):
        return filtered_prob(tr, p, t)

    row = dict(key=key, cls=cls, mode=mode, n0=n0, gen=seq[n0:], arg=[int(t) for t in tr["arg"]],
               pmax=[round(float(v), 4) for v in tr["pmax"]],
               q_gen=[round(filtered_prob(tr, n0 - 1 + j, t), 4) for j, t in enumerate(seq[n0:])], heads={})
    for name, mt in heads.items():
        t = time.time()
        W.MTP_REFEED = False
        tb = table(mt, seq, n0, H, kmax)
        o = dict(chain=tb["chain"], conf=r4(tb["conf"]), cos=r4(tb["cos"]), nonfinite=tb["nonfinite"],
                 q=r4([[filtered_prob(tr, n0 + i + j, d) for j, d in enumerate(c)]
                       for i, c in enumerate(tb["chain"])]), sims={}, refeed={})
        for k in sims:
            o["sims"][str(k)] = sim(mt, seq, n0, H, k, qf)
        W.MTP_REFEED = True
        for k in refeeds:
            o["refeed"][str(k)] = sim(mt, seq, n0, H, k, qf)
        W.MTP_REFEED = False
        if wide is not None:
            wt = table(wide[name], seq, n0, H, kmax)
            o["wide"] = dict(chain=wt["chain"], nonfinite=wt["nonfinite"])
        row["heads"][name] = o
        acc = [sum(c[:i + 1] == seq[n0 + j + 1:n0 + j + 2 + i] for j, c in enumerate(tb["chain"]))
               for i in range(kmax)]
        print(f"  {key} {mode} {name}: {len(tb['chain'])} positions, first-i-drafts-right {acc}, "
              f"nonfinite {tb['nonfinite']}, {time.time() - t:.1f} s", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", required=True,
                    help="name=dir,... (MTP drafter dirs; dir:wide = that head with fp32 activations)")
    ap.add_argument("--prompts", default="code-2k-0,en-2k-0,ru-2k-0", help="bench27 prompt ids")
    ap.add_argument("--extra", default="", help="more bench27 ids, probed last")
    ap.add_argument("--no-short", action="store_true", help="without the two short prose prompts")
    ap.add_argument("--gen", type=int, default=320, help="generated tokens per trajectory")
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--sims", default="2,3,4", help="k of the round simulations (chained entries kept)")
    ap.add_argument("--refeed-sim", default="4", help="k of the live QWEN_MTP_REFEED=1 simulations")
    ap.add_argument("--modes", default="greedy,sampled", help="trajectories: the target's greedy and / or sampled")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--wide", default="", help="prompt ids whose greedy table is repeated in fp32")
    ap.add_argument("--cost-iters", type=int, default=30)
    ap.add_argument("--budget-min", type=float, default=20.0,
                    help="nothing new is started after this many minutes (a GPU hold): rerun with --traj")
    ap.add_argument("--traj", default="", help="dir of the trajectories (.npz; reused when present)")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    import worker_pb2 as pb
    from qwen_worker.mtp import load_mtp
    from qwen_worker.sampler import make_sampler
    sys.path.insert(0, os.path.join(HERE, "..", "dflash2"))
    bench = os.environ.get("BENCH27") or os.path.join(HERE, "..", "bench27")
    sys.path.insert(0, bench)
    import prompts as P
    from prompts2k import SHORT

    t_start = time.time()
    eng = W.Engine(os.environ["QWEN_MODEL"])
    eng.load()
    lm = eng.model.language_model
    tok = getattr(eng.processor, "tokenizer", eng.processor)
    heads, info = {}, {}
    for spec in a.heads.split(","):
        name, path = spec.split("=", 1)
        wide = path.endswith(":wide")           # this head with fp32 activations throughout
        path = path[:-5] if wide else path
        m0 = mx.get_active_memory()
        d = load_mtp(path, lm, W.FP16 and not wide)
        heads[name] = Probe(widen(d.model) if wide else d.model, lm)
        heads[name].wide = wide
        cfgp = json.load(open(os.path.join(path, "config.json")))
        info[name] = dict(dir=os.path.basename(os.path.normpath(path)), quantization=cfgp.get("quantization"),
                          mib=(mx.get_active_memory() - m0) / 2**20, path=path, fp32_activations=wide)
    print(f"loaded in {time.time() - t_start:.0f} s; active {mx.get_active_memory() / 2**30:.2f} GiB; heads "
          + ", ".join(f"{n} {i['mib']:.0f} MiB" for n, i in info.items()), flush=True)

    man = P.load_manifest(os.path.join(bench, "prompts", "manifest.json"))
    byid = {p["id"]: p for p in man["prompts"]}

    def ids_of(key):
        if key in byid:
            msgs, cls = P.messages(man, byid[key]), byid[key]["cls"]
        else:
            msgs, cls = [{"role": "user", "content": dict(SHORT)[key]}], key.split("_")[0]
        req = pb.Request(id=key, messages=[pb.Message(role=m["role"], content=m["content"]) for m in msgs],
                         params=pb.Params(chat_template_kwargs_json=json.dumps({"enable_thinking": False})))
        return tok.encode(eng._render(req, [])[0]), cls

    smp = make_sampler(SAMPLING, a.seed)
    greedy = make_sampler(dict(SAMPLING, temperature=0.0))

    def traj_of(key, mode):
        path = os.path.join(a.traj, f"{key}.{mode}.g{a.gen}.s{a.seed}.npz") if a.traj else ""
        if path and os.path.exists(path):
            z = np.load(path)
            return {k: (int(z[k]) if k == "n0" else z[k]) for k in z.files}, str(z["cls"])
        ids, cls = ids_of(key)
        t = time.time()
        if mode == "sampled":
            smp.key = mx.random.key(a.seed)     # each text: the same draws whatever ran before
        tr = trajectory(lm, ids, a.gen, greedy if mode == "greedy" else smp, smp.filtered, eng.stopping)
        print(f"trajectory {key} {mode}: prompt {tr['n0']}, generated {len(tr['seq']) - tr['n0']} "
              f"in {time.time() - t:.1f} s", flush=True)
        if path:
            os.makedirs(a.traj, exist_ok=True)
            np.savez(path, cls=cls, **tr)
        return tr, cls

    res = dict(args={k: v for k, v in vars(a).items() if k not in ("heads", "traj", "json")},
               model=os.path.basename(os.path.normpath(os.environ["QWEN_MODEL"])), sampling=SAMPLING,
               heads={n: {k: v for k, v in i.items() if k != "path"} for n, i in info.items()},
               texts=[], costs={})

    def dump():
        if a.json:
            os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
            json.dump(res, open(a.json + ".tmp", "w"))
            os.replace(a.json + ".tmp", a.json)

    wide_heads = {}

    def probe(key, mode, tr, cls):
        wide = None
        if mode == "greedy" and key in a.wide.split(","):
            for name in heads:
                if name not in wide_heads:
                    wd = load_mtp(info[name]["path"], lm, False)
                    wide_heads[name] = Probe(widen(wd.model), lm)
                    wide_heads[name].wide = True
            wide = wide_heads
        res["texts"].append(probe_row(heads, key, cls, mode, tr, a.kmax, ints(a.sims), ints(a.refeed_sim), wide))
        dump()

    keys = a.prompts.split(",") + ([] if a.no_short else [k for k, _ in SHORT])
    modes = a.modes.split(",")

    def late(what):
        if (time.time() - t_start) / 60 > a.budget_min:
            print(f"budget ({a.budget_min} min): {what} not run", flush=True)
            res.setdefault("skipped", []).append(what)
            return True
        return False

    # every trajectory first (kept in --traj: a later run goes on from them), then the
    # greedy tables, the costs, the sampled tables, the extra texts
    trajs = {(k, m): traj_of(k, m) for m in modes for k in keys if not late(f"trajectory {k} {m}")}
    for (k, m), (tr, cls) in trajs.items():
        if m == "greedy" and not late(f"probe {k} {m}"):
            probe(k, m, tr, cls)
    wide_heads.clear()
    mx.clear_cache()
    greedy_trajs = [t for (k, m), (t, _) in trajs.items() if m == "greedy"]
    if a.cost_iters and greedy_trajs and not late("costs"):
        tr = max(greedy_trajs, key=lambda t: t["n0"])
        T = costs(lm, heads, tr["seq"], tr["n0"], mx.array(tr["H"]), a.cost_iters, a.kmax)
        skip = min(3, a.cost_iters - 1)
        res["costs"] = dict(ctx=int(tr["n0"]), skip=skip, ms={k: [round(x, 3) for x in v] for k, v in T.items()})
        for k, v in T.items():
            print(f"  {k:24s} {statistics.median(v[skip:]):7.2f} ms (min {min(v[skip:]):.2f})", flush=True)
        dump()
    for (k, m), (tr, cls) in trajs.items():
        if m != "greedy" and not late(f"probe {k} {m}"):
            probe(k, m, tr, cls)
    del trajs, greedy_trajs
    for k in [k for k in a.extra.split(",") if k]:
        for m in modes:
            if not late(f"extra {k} {m}"):
                probe(k, m, *traj_of(k, m))
    res["peak_gib"] = mx.get_peak_memory() / 2**30
    print(f"peak {res['peak_gib']:.2f} GiB, {(time.time() - t_start) / 60:.1f} min", flush=True)
    dump()
    if a.json:
        print("wrote", a.json)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
