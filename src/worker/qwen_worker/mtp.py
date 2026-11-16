"""The target's native multi-token-prediction (MTP) head as a speculative drafter
(QWEN_DRAFT=<an MTP drafter dir>, or QWEN_DRAFT_MTP beside a DFlash QWEN_DRAFT:
QWEN_DRAFT_HYBRID), driven by _Scheduler._spec.

Qwen3.5 / 3.8 ship one MTP decoder layer (`mtp_num_hidden_layers: 1`) that
shares the target's embedding and lm_head: for a position t it takes the
target's final (post-norm) hidden state h_t and the embedding of the token
after t, x_{t+1}, and predicts x_{t+2} (mlx-vlm 0.7.6
`speculative/drafters/qwen3_5_mtp`: fc(concat(norm(emb), norm(h))) -> one full-
attention layer with its own KV cache -> norm; the lm_head is the target's).
Drafting k tokens chains it: the first draft is the "seed" (the head's output
at the last cache position, whose next token - the round's anchor - is known),
the next ones feed the head's own output hidden as the stand-in for the
target's hidden at the drafted position.

The head keeps one entry (K/V) per target position. Entries are made from
pairs (h_t, x_{t+1}): `push` hands it hidden states of positions that just
entered the target cache (prefill slabs, a verify block's kept part, the
plain steps of a thinking-budget close), `emit` the tokens in emission order
(each is the successor of the oldest position without one); `flush` runs the
head over every complete pair at once (lazy, no host sync). The chained
drafts add approximate entries (the head's hidden for the target's); after
the verify `round_end(accepted, h)` trims the rejected ones and, as mlx-vlm
does, keeps those of accepted positions (QWEN_MTP_REFEED=1: recompute every
accepted position from the target's real hidden states instead).

The target verifies every draft (the same verify forward as DFlash), so the
drafts only move acceptance; greedy output == the DFlash path's.
"""

import math

import mlx.core as mx

from . import config as cfg
from .config import log


class Clock:
    """Wall time of a steady round per kind ("dflash", "plain", ("mtp", k)),
    process-wide: a property of the model and the machine, shared by requests.
    The machine's speed drifts (+-30 % over minutes) and a kind that is not
    taken is not measured, so a time is kept as g * c[kind]: a sample moves
    the common factor g (weight G) more than its own c (weight C), and the
    other kinds' times follow the drift instead of going stale. A sample
    counts within [1/2, 2] of the expectation (a pause is not the kind's cost)."""

    G, C = 0.2, 0.05

    def __init__(self, ms=None):
        self.g, self.c = 1.0, dict(ms or {})
        self.sub = {}

    def at(self, tag):
        """The clock of one class of requests, `tag` = (the context's size
        class, sampling, ...): the kinds' times differ between classes (a
        plain step grows faster with the context than a verify block; the
        sampler runs per verified row). A new class starts from the times of
        the same context class when there is one (`tag[0]`), else from this
        clock's own (none, but for tools that preset them): its arms are
        probed (Chooser.PROBE)."""
        c = self.sub.get(tag)
        if c is None:
            near = next((x for t, x in self.sub.items() if t[0] == tag[0]), self)
            c = self.sub[tag] = Clock()
            c.g, c.c = near.g, dict(near.c)
        return c

    def ms(self, kind):
        c = self.c.get(kind)
        return None if c is None else self.g * c

    def add(self, kind, ms):
        if ms <= 0.0:
            return
        if kind not in self.c:
            self.c[kind] = ms / self.g
            return
        r = min(2.0, max(0.5, ms / (self.g * self.c[kind])))
        self.g *= r ** self.G
        self.c[kind] *= r ** self.C


CLOCK = Clock()


class Avg:
    """The mean of samples weighted exp(-age / tau) (age in ms) with a prior
    worth `n` samples, which ages like one. `e`: the real samples' weight
    (how much the mean knows about now)."""

    __slots__ = ("h", "w", "e", "t")

    def __init__(self, prior, n):
        self.h, self.w, self.e, self.t = prior * float(n), float(n), 0.0, 0.0

    def age(self, t, tau):
        if t > self.t:
            d = max(1e-6, math.exp(-(t - self.t) / tau))
            self.h, self.w, self.e, self.t = self.h * d, self.w * d, self.e * d, t
        return self.e

    def add(self, x, t, tau):
        self.age(t, tau)
        self.h, self.w, self.e = self.h + x, self.w + 1.0, self.e + 1.0

    @property
    def mean(self):
        return self.h / self.w


class MTPDraft:
    """A loaded MTP head (`model`: mlx-vlm's Qwen3_5MTPDraftModel) bound to
    the target language model `lm`; per-request state in start()."""

    kind = "mtp"

    def __init__(self, model, lm, k=None):
        self.model, self.lm = model, lm
        self.k = k                      # drafts per round; None: cfg.MTP_K
        self.hidden_size = int(lm.args.hidden_size)
        self.accept_lens, self.draft_lens = [], []
        self.cache = None

    # ---- per request ----

    @property
    def token_bytes(self):
        """fp16 K + V bytes per position of the head's own cache (every layer
        attends: 4 KiB on the 27B's one layer)."""
        a = self.model.config.text_config
        heads = getattr(a, "num_key_value_heads", 0)
        dim = getattr(a, "head_dim", 0) or a.hidden_size // a.num_attention_heads
        return 2 * 2 * heads * dim * len(self.model.layers)

    def start(self, at):
        """A new request whose next pushed hidden state is position `at`."""
        from mlx_vlm.models.cache import KVCache
        self.cache = [KVCache() for _ in self.model.layers]
        self.pos = at                   # position of the next head entry
        self.pend = []                  # hidden chunks of positions with no successor yet
        self.npend = 0
        self.succ = []                  # successors not paired yet
        self.skip = 0                   # successors already covered by kept draft entries
        self.appended = 0               # approximate entries of the current round
        self.seed = None                # (token (1, 1), hidden (1, 1, H)) or None
        self.fresh()

    def fresh(self):
        """A new request's acceptance statistics (no head state: tools too)."""
        self.accept_lens, self.draft_lens = [], []
        self.KMAX = cfg.MTP_KMAX        # QWEN_MTP_KMAX
        self.seen = [0] * self.KMAX     # drafts of depth i scored (rounds + shadow drafts)
        self.p = [self.P0] * self.KMAX  # QWEN_MTP_K=0: P(draft i kept | drafts < i kept)
        # the same over the last TAU ms (the Chooser's comparison: expect)
        self.st = [Avg(self.P0, self.PRIOR) for _ in range(self.KMAX)]
        self.now = 0.0                  # ms, the Chooser's time (no Chooser: plain means)
        self.clock = None               # the request's Clock (None: CLOCK)
        self.scored = 0                 # rounds + shadow drafts scored

    # QWEN_MTP_K=0: the draft count per round from the request's own acceptance
    # and the process's measured round times per k (CLOCK: shared by requests,
    # a property of the model and machine; the verify cost is not linear in T:
    # 2 tokens exact qmv, 3..16 mrqmm's flat MMA cost on the 27B)
    KMAX, P0, ALPHA, EXPLORE = 4, 0.5, 0.25, 8
    TAU, PRIOR = 2500.0, 2

    def _clock(self):
        return self.clock if self.clock is not None else CLOCK

    def _ms(self, k):
        """Expected wall time of a k-draft round: measured, else scaled from
        the nearest measured k by the linear prior (1 + QWEN_MTP_COST * k)."""
        ck = self._clock()
        ms = ck.ms(("mtp", k))
        if ms is not None:
            return ms
        c = cfg.MTP_COST
        ks = [j[1] for j in ck.c if isinstance(j, tuple) and j[0] == "mtp"]
        if ks:
            k0 = min(ks, key=lambda j: abs(j - k))
            return ck.ms(("mtp", k0)) * (1.0 + c * k) / (1.0 + c * k0)
        return 1.0 + c * k

    def _best(self, p):
        """(k, tokens, ms): the k in 1..KMAX with the most expected tokens
        per ms, (1 + sum_i prod_{j<=i} p_j) / ms(k), by the per-depth
        acceptance averages `p` and the measured round times."""
        best, out, e, q, pk = -1.0, (1, 1.0, 1.0), 1.0, 1.0, self.P0
        for k in range(1, max(self.KMAX, cfg.MTP_K) + 1):
            # a depth this request has not drafted yet takes the shallower
            # one's estimate (else a request that started at k = 1 stays
            # there: its deeper averages never move from the prior)
            pk = p[k - 1] if k <= self.KMAX and self.seen[k - 1] else pk
            q *= pk
            e += q
            ms = self._ms(k)
            if cfg.MTP_K == k or (cfg.MTP_K <= 0 and e / ms > best):
                best, out = e / ms, (k, e, ms)
        return out

    def next_k(self):
        """Drafts for the next round: QWEN_MTP_K, or (0) the k of _best by the
        fast averages; every EXPLORE-th round one more or (alternately) one
        fewer (the averages of the neighbours would go stale)."""
        if cfg.MTP_K > 0:
            return cfg.MTP_K
        bk = self._best(self.p)[0]
        r = self.scored
        if r % self.EXPLORE == self.EXPLORE - 1:
            bk = min(self.KMAX, bk + 1) if (r // self.EXPLORE) % 2 == 0 else max(1, bk - 1)
        return bk

    def expect(self):
        """(tokens, ms) of a round at the best draft count by the averages
        over the last TAU ms (mtp.Chooser's comparison with the other arms);
        ms None: no MTP round timed in this class of requests yet."""
        _, e, ms = self._best([x.mean for x in self.st])
        timed = any(isinstance(j, tuple) and j[0] == "mtp" for j in self._clock().c)
        return e, (ms if timed else None)

    def evidence(self):
        """The weight of the first draft's samples (Avg.e) now."""
        return self.st[0].age(self.now, self.TAU)

    def push(self, h):
        """Hidden states (1, n, H, the target's post-norm) of the next n
        positions that entered the target cache."""
        if h is not None and h.shape[1]:
            self.pend.append(h)
            self.npend += h.shape[1]

    def emit(self, tok):
        """The next token in emission order (= the successor of the oldest
        pushed position without one)."""
        if self.skip:
            self.skip -= 1
        else:
            self.succ.append(int(tok))

    def emit_ids(self, ids):
        for t in ids:
            self.emit(t)

    def flush(self):
        """Run the head over every (hidden, successor) pair queued; the seed
        is the head's output at the last one when nothing is left pending."""
        r = min(self.npend, len(self.succ))
        if r == 0:
            return
        h = self.pend[0] if len(self.pend) == 1 else mx.concatenate(self.pend, axis=1)
        toks = mx.array([self.succ[:r]], dtype=mx.int32)
        out = self._forward(toks, h[:, :r])
        rest = h[:, r:]
        self.pend, self.npend = ([rest] if rest.shape[1] else []), rest.shape[1]
        self.succ = self.succ[r:]
        last = out[:, -1:, :]
        self.seed = (self._argmax(last), last) if not self.npend else None

    def state(self):
        """Arrays to evaluate with a step (the head's cache and seed)."""
        out = [c.state for c in (self.cache or []) if c.keys is not None]
        if self.seed is not None:
            out += list(self.seed)
        return out

    # ---- a round ----

    def draft(self, n):
        """(1, n) draft tokens (lazy): the seed, then n - 1 chained steps
        (approximate entries, settled by round_end)."""
        self.flush()
        if self.seed is None:
            raise RuntimeError("mtp: no seed (a pushed position without its successor)")
        tok, h = self.seed
        toks = [tok]
        self.appended = 0
        for _ in range(n - 1):
            h = self._forward(tok, h)
            self.appended += 1
            tok = self._argmax(h)
            toks.append(tok)
        self.seed = None
        return mx.concatenate(toks, axis=1) if len(toks) > 1 else toks[0]

    def round_end(self, accepted, h=None):
        """After a verify that kept `accepted` drafts: the approximate entries
        past the kept ones leave the head's cache; `h` (1, >= accepted + 1, H):
        the target's hidden states of the positions the commit kept (the
        anchor + the accepted drafts), pushed but for those the kept entries
        already cover (their successors - the first emitted tokens - skipped).
        A round the head did not draft (hybrid: DFlash's) has none."""
        keep = 0 if cfg.MTP_REFEED else min(int(accepted), self.appended)
        trim = self.appended - keep
        if trim:
            for c in self.cache:
                c.trim(trim)
        self.pos -= trim
        self.appended = 0
        self.seed = None
        if h is not None:
            self.push(h[:, keep:accepted + 1])
        self.skip += keep

    def cancel(self):
        """The chained entries of the last draft() leave the head's cache: a
        shadow draft (mtp.Chooser: scored against the emitted tokens, not
        verified; the round's positions are then pushed whole by round_end)."""
        if self.appended:
            for c in self.cache:
                c.trim(self.appended)
            self.pos -= self.appended
            self.appended = 0
        self.seed = None

    def learn(self, accepted, drafted):
        """A draft of `drafted` tokens whose first `accepted` were right (a
        verified round or a scored shadow draft) -> the per-depth averages."""
        self.scored += 1
        for i in range(min(int(accepted) + 1, int(drafted), self.KMAX)):
            hit = 1.0 if i < accepted else 0.0
            self.p[i] += self.ALPHA * (hit - self.p[i])
            self.st[i].add(hit, self.now, self.TAU)
            self.seen[i] += 1

    def record(self, accepted, drafted):
        """After a round's tokens were read: its acceptance (the round's wall
        time goes to CLOCK from _spec)."""
        self.accept_lens.append(int(accepted))
        self.draft_lens.append(int(drafted))
        self.learn(accepted, drafted)

    # ---- the head ----

    def _forward(self, toks, h):
        """Head entries for the pairs (h[:, i], toks[:, i]) at positions
        self.pos.. -> the head's output hidden states (1, n, H)."""
        n = toks.shape[1]
        emb = self.lm.model.embed_tokens(toks)
        pid = (mx.arange(n, dtype=mx.int32) + self.pos)[None]
        out = self.model._forward_hidden(emb, h.astype(emb.dtype), self.cache, pid)
        self.pos += n
        return out

    def _argmax(self, h):
        return mx.argmax(self.lm.lm_head(h), axis=-1).astype(mx.int32)


class Chooser:
    """Which arm a round takes (QWEN_DRAFT=auto; QWEN_DRAFT_HYBRID /
    QWEN_SPEC_PLAIN): "dflash", "mtp" (a drafter's round) or "plain" (a burst
    of plain steps) - the one with the most expected tokens per ms.

    Wall time per round: the request class's Clock (process-wide, steady
    rounds only: _spec leaves out a round after a switch, whose drafter
    catches up on the positions it missed, and rounds with shadow drafts).
    Tokens per round (1 + accepted drafts): means over the last TAU ms (Avg) -
    DFlash's over its blocks, the MTP head's per draft depth at its best draft
    count (MTPDraft.expect), plain exactly 1.

    An arm that is not taken is still sampled: shadows() names the drafters to
    draft this round WITHOUT verifying; their drafts are scored against the
    tokens the request emits from the same anchor (Shadows) - what a verify
    would have kept, at the cost of the draft alone (27B: an MTP draft step
    ~3.5 ms, a DFlash block ~17 ms, against a 110-180 ms round). So no arm's
    estimate goes stale at the value it had when it lost (the per-round
    comparison of a noisy average with frozen ones kept the arm without
    noise: plain took 91 / 179 of 256 tokens on prose where both drafters
    beat it). A drafter is drafted every round until it has WARM drafts and
    while it looks better than the current arm without the samples to say so
    (below NEED: a trial); else as often as its drafts cost NEAR of the time
    (an arm within CLOSE of the current one's rate) or FAR.

    The request starts on the first arm of START it has (the MTP head: the
    smallest loss when it is the wrong one). The current arm is left only for
    one whose rate is MARGIN higher and, a drafter, known from >= NEED recent
    samples. An arm never timed in this class of requests is taken PROBE
    rounds first."""

    TAU = 2500.0
    PRIOR = 2
    PRIORS = {"dflash": 3.0}
    START = ("mtp", "dflash", "plain")
    MARGIN = 0.05
    WARM, NEED = 3, 2.5
    NEAR, FAR, CLOSE = 0.01, 0.004, 0.8
    COST = {"dflash": 0.10, "mtp": 0.06}        # a shadow draft, of the arm's round time
    PROBE = 3
    BURST, SHORT = 8, 2

    def __init__(self, kinds, mtp=None, clock=None, priors=None):
        self.kinds = list(kinds)
        self.mtp = mtp                          # MTPDraft: the "mtp" arm's estimate
        self.clock = clock if clock is not None else CLOCK
        pri = dict(self.PRIORS, **(priors or {}))
        self.tok = {k: Avg(pri.get(k, 2.0), self.PRIOR) for k in kinds}
        self.n = {k: 0 for k in kinds}          # rounds taken
        self.got = {k: 0 for k in kinds}        # tokens of those rounds
        self.shadowed = {k: 0 for k in kinds}   # shadow drafts
        self.pend = {k: 0 for k in kinds}       # shadow drafts not scored yet
        self.last = {k: 0.0 for k in kinds}     # ms: the arm's last draft
        self.cur, self.left, self.rounds, self.r = None, 0, 0, None
        self.t, self.hot, self.probe = 0.0, False, False
        self.switches = 0

    def expect(self, k):
        """(tokens per round, ms per round or None: never timed) of arm k."""
        if k == "plain":
            return 1.0, self.clock.ms(k)
        if k == "mtp" and self.mtp is not None:
            return self.mtp.expect()
        return self.tok[k].mean, self.clock.ms(k)

    def evidence(self, k):
        """The weight of arm k's recent samples (a plain step needs none)."""
        if k == "plain":
            return float("inf")
        if k == "mtp" and self.mtp is not None:
            return self.mtp.evidence()
        return self.tok[k].age(self.t, self.TAU)

    def pick(self, force=None, now=None):
        """The arm of the next round. `now`: ms, any origin (the samples'
        ages; None: the time stands still)."""
        self.rounds += 1
        self.r = None
        self.probe = False
        if now is not None:
            self.t = max(self.t, now)
        if self.mtp is not None:
            self.mtp.now = self.t
        was = self.cur
        if force in self.kinds:
            k = force
        elif self.left > 0:
            k, self.left, self.probe = self.cur, self.left - 1, True    # the running probe
        else:
            ex = {k: self.expect(k) for k in self.kinds}
            new = [k for k in self.kinds if ex[k][1] is None]
            if new:
                k, self.left, self.probe = new[0], self.PROBE - 1, True
            else:
                r = self.r = {k: t / ms for k, (t, ms) in ex.items()}
                k = self.cur
                if k is None:
                    k = next((x for x in self.START if x in self.kinds), self.kinds[0])
                else:
                    ok = [x for x in self.kinds if x != k and r[x] > r[k] * (1.0 + self.MARGIN)
                          and self.evidence(x) >= self.NEED]
                    if ok:
                        k = max(ok, key=r.get)
        if was is not None and k != was:
            self.switches += 1
        self.cur = k
        return k

    def shadows(self):
        """The drafters (not the arm just picked) to draft this round without
        verifying."""
        self.hot = False
        if self.r is None:                      # a forced arm, a probe
            return []
        out, r = [], self.r
        for k in self.kinds:
            if k in ("plain", self.cur):
                continue
            if self.n[k] + self.shadowed[k] < self.WARM:
                hot = True
            else:
                hot = (r[k] > r[self.cur] * (1.0 + self.MARGIN)
                       and self.evidence(k) + self.pend[k] < self.NEED)
            if hot:
                self.hot = True
                out.append(k)
                continue
            ms = self.expect(k)[1]
            share = self.NEAR if r[k] >= self.CLOSE * r[self.cur] else self.FAR
            if (self.t - self.last[k]) * share >= self.COST.get(k, 0.1) * ms:
                out.append(k)
        return out

    def sample(self, k, accepted, drafted, full=True):
        """A shadow draft of arm k (`drafted` tokens) scored: its first
        `accepted` were right."""
        self.pend[k] = max(0, self.pend[k] - 1)
        self._add(k, accepted, drafted, full)

    def _add(self, k, accepted, drafted, full=True):
        if k == "mtp" and self.mtp is not None:
            self.mtp.learn(accepted, drafted)
        elif k != "plain" and full:
            self.tok[k].add(1.0 + accepted, self.t, self.TAU)

    def steps(self):
        """Plain steps of a plain round (one burst with step-ahead; short
        while a drafter is on trial or the plain step is being timed)."""
        return self.SHORT if self.hot or self.probe else self.BURST

    def done(self, accepted, drafted=0, full=True, shadows=(), steps=1):
        """The round of the last pick() kept `accepted` of its `drafted`
        drafts (`full`: a block of the usual size); `shadows`: the arms
        drafted beside it; `steps`: a plain round's steps."""
        k = self.cur
        if k != "mtp" or self.mtp is None:      # (the head's rounds: MTPDraft.record)
            self._add(k, accepted, drafted, full)
        self.n[k] += 1
        self.got[k] += steps if k == "plain" else 1 + accepted
        self.last[k] = self.t
        for x in shadows:
            self.shadowed[x] += 1
            self.pend[x] += 1
            self.last[x] = self.t

    def summary(self):
        out = []
        for k in self.kinds:
            t, ms = self.expect(k)
            out.append(f"{k} {self.n[k]} rounds" + (f" + {self.shadowed[k]} shadows" if self.shadowed[k] else "")
                       + (f" ({t:.2f} tok / {ms:.0f} ms)" if ms else ""))
        return ", ".join(out) + f", {self.switches} switches"


class Shadows:
    """Drafts that were not verified, scored against the tokens the request
    emits after their anchor: the length of the matching prefix is what a
    verify of that draft would have kept (greedy: the emitted tokens are the
    target's argmax whichever arm drafted; sampling: the target's draws)."""

    def __init__(self):
        self.toks, self.base, self.pend = [], 0, []

    def emit(self, tok):
        self.toks.append(tok)

    def add(self, kind, drafts):
        """`drafts`: the tokens proposed after the last emitted token."""
        if drafts:
            self.pend.append((kind, self.base + len(self.toks), list(drafts)))

    def settle(self):
        """-> [(kind, accepted, drafted)] of the drafts whose outcome is known
        (a mismatch, or every draft's position emitted)."""
        out, keep = [], []
        for kind, at, d in self.pend:
            got = self.toks[at - self.base:at - self.base + len(d)]
            acc = next((i for i, (x, y) in enumerate(zip(d, got)) if x != y), len(got))
            if acc < len(got) or len(got) == len(d):
                out.append((kind, acc, len(d)))
            else:
                keep.append((kind, at, d))
        self.pend = keep
        drop = min([at for _, at, _ in keep], default=self.base + len(self.toks)) - self.base
        if drop > 256:
            self.toks, self.base = self.toks[drop:], self.base + drop
        return out


def mtp_dir(path):
    """Is `path` an MTP drafter dir (its config's kind, as mlx-vlm resolves it)?"""
    from mlx_vlm.speculative.drafters import resolve_drafter_kind
    from mlx_vlm.utils import get_model_path
    return resolve_drafter_kind(get_model_path(path), None) == "mtp"


def load_mtp(path, lm, fp16):
    """The MTP head at `path` (an mlx-vlm qwen3_5_mtp drafter dir, e.g. made
    by scripts/perf/mtp/split_mtp.py) bound to the target `lm`, its bf16
    weights cast to fp16 with the target's (QWEN_FP16)."""
    from mlx_vlm.speculative.drafters import load_drafter
    model, kind = load_drafter(path)
    if kind != "mtp":
        raise RuntimeError(f"{path}: drafter kind {kind!r}, not an MTP head")
    hs = int(model.config.text_config.hidden_size)
    if hs != int(lm.args.hidden_size):
        raise RuntimeError(f"{path}: MTP hidden size {hs} != the target's {lm.args.hidden_size}")
    if fp16:
        model.apply(lambda a: a.astype(mx.float16) if a.dtype == mx.bfloat16 else a)
    mx.eval(model.parameters())
    d = MTPDraft(model, lm)
    log.info("mtp drafter loaded: %s (%d layer(s), %.0f MiB, %d drafts per round%s)", path,
             len(model.layers), sum(v.nbytes for _, v in _flat(model)) / 2**20, cfg.MTP_K,
             ", refeed" if cfg.MTP_REFEED else "")
    return d


def _flat(model):
    from mlx.utils import tree_flatten
    return tree_flatten(model.parameters())
