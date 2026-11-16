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
        self.ps = [self.P0] * self.KMAX  # the same, slow (the hybrid's comparison: expect)
        self.scored = 0                 # rounds + shadow drafts scored

    # QWEN_MTP_K=0: the draft count per round from the request's own acceptance
    # and the process's measured round times per k (CLOCK: shared by requests,
    # a property of the model and machine; the verify cost is not linear in T:
    # 2 tokens exact qmv, 3..16 mrqmm's flat MMA cost on the 27B)
    KMAX, P0, ALPHA, EXPLORE = 4, 0.5, 0.25, 8
    SLOW, PRIOR = 1.0 / 16, 2

    def _ms(self, k):
        """Expected wall time of a k-draft round: measured, else scaled from
        the nearest measured k by the linear prior (1 + QWEN_MTP_COST * k)."""
        ms = CLOCK.ms(("mtp", k))
        if ms is not None:
            return ms
        c = cfg.MTP_COST
        ks = [j[1] for j in CLOCK.c if isinstance(j, tuple) and j[0] == "mtp"]
        if ks:
            k0 = min(ks, key=lambda j: abs(j - k))
            return CLOCK.ms(("mtp", k0)) * (1.0 + c * k) / (1.0 + c * k0)
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
        """(tokens, ms) of a round at the best draft count by the slow
        averages (mtp.Chooser's comparison with the other arms); ms None:
        no MTP round timed in this process yet."""
        _, e, ms = self._best(self.ps)
        timed = any(isinstance(j, tuple) and j[0] == "mtp" for j in CLOCK.c)
        return e, (ms if timed else None)

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
            self.ps[i] += max(self.SLOW, 1.0 / (self.seen[i] + 1 + self.PRIOR)) * (hit - self.ps[i])
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
    """Which arm a round takes (QWEN_DRAFT_HYBRID / QWEN_SPEC_PLAIN): "dflash",
    "mtp" (a drafter's round) or "plain" (one plain step) - the one with the
    most expected tokens per ms.

    Wall time per round: CLOCK (process-wide, steady rounds only: _spec leaves
    out a round after a switch, whose drafter catches up on the positions it
    missed, and rounds with shadow drafts). Tokens per round (1 + accepted
    drafts): DFlash an average over its samples (the mean of the first ones
    with a prior worth PRIOR samples, then exponential, weight ALPHA), the MTP
    head its own per-depth averages at its best draft count (MTPDraft.expect),
    plain exactly 1.

    An arm that is not taken is still sampled: shadows() names the drafters to
    draft this round WITHOUT verifying; their drafts are scored against the
    tokens the request emits from the same anchor (Shadows) - what a verify
    would have kept, at the cost of the draft alone (27B: an MTP draft step
    ~3.5 ms, a DFlash block ~17 ms, against a 110-180 ms round). So no arm's
    estimate goes stale at the value it had when it lost (the per-round
    comparison of a noisy average with frozen ones kept the arm without
    noise: plain took 91 / 179 of 256 tokens on prose where both drafters
    beat it). Every round of the first WARM, then after NEAR rounds without a
    sample for an arm within CLOSE of the current one's rate, FAR for one
    further behind. The current arm is left only for one whose rate is MARGIN higher.
    An arm never timed in this process is taken PROBE rounds first."""

    ALPHA = 1.0 / 16
    PRIOR = 2
    PRIORS = {"dflash": 3.0}
    MARGIN = 0.05
    WARM, NEAR, FAR, CLOSE = 8, 8, 32, 0.8
    PROBE = 3
    BURST = 8

    def __init__(self, kinds, mtp=None):
        self.kinds = list(kinds)
        self.mtp = mtp                          # MTPDraft: the "mtp" arm's estimate
        self.tok = {k: self.PRIORS.get(k, 2.0) for k in kinds}
        self.seen = {k: 0 for k in kinds}       # acceptance samples (rounds + shadows)
        self.n = {k: 0 for k in kinds}          # rounds taken
        self.shadowed = {k: 0 for k in kinds}   # shadow drafts
        self.since = {k: 0 for k in kinds}      # rounds since the arm's last sample
        self.cur, self.left, self.rounds, self.r = None, 0, 0, None

    def expect(self, k):
        """(tokens per round, ms per round or None: never timed) of arm k."""
        if k == "plain":
            return 1.0, CLOCK.ms(k)
        if k == "mtp" and self.mtp is not None:
            return self.mtp.expect()
        return self.tok[k], CLOCK.ms(k)

    def pick(self, force=None):
        self.rounds += 1
        self.r = None
        if force in self.kinds:
            k = force
        elif self.left > 0:
            k, self.left = self.cur, self.left - 1      # finish the running probe
        else:
            ex = {k: self.expect(k) for k in self.kinds}
            new = [k for k in self.kinds if ex[k][1] is None]
            if new:
                k, self.left = new[0], self.PROBE - 1
            else:
                r = self.r = {k: t / ms for k, (t, ms) in ex.items()}
                best = max(self.kinds, key=r.get)
                k = self.cur
                if k is None or (best != k and r[best] > r[k] * (1.0 + self.MARGIN)):
                    k = best
        self.cur = k
        return k

    def shadows(self):
        """The drafters (not the arm just picked) to draft this round without
        verifying."""
        if self.r is None:                      # a forced arm, a probe
            return []
        out = []
        for k in self.kinds:
            if k in ("plain", self.cur):
                continue
            gap = (0 if self.rounds <= self.WARM
                   else self.NEAR if self.r[k] >= self.CLOSE * self.r[self.cur] else self.FAR)
            if self.since[k] >= gap:
                out.append(k)
        return out

    def sample(self, k, accepted, drafted, full=True):
        """A draft of arm k (`drafted` tokens) whose first `accepted` were
        right: a shadow draft scored; `full`: a block of the usual size."""
        if k == "mtp" and self.mtp is not None:
            self.mtp.learn(accepted, drafted)
        elif k != "plain" and full:
            a = max(self.ALPHA, 1.0 / (self.seen[k] + 1 + self.PRIOR))
            self.tok[k] += a * (1.0 + accepted - self.tok[k])
        self.seen[k] += 1

    def steps(self):
        """Plain steps of a plain round (one burst with step-ahead)."""
        return self.BURST

    def done(self, accepted, drafted=0, full=True, shadows=(), steps=1):
        """The round of the last pick() kept `accepted` of its `drafted`
        drafts; `shadows`: the arms drafted beside it; `steps`: a plain
        round's steps."""
        k = self.cur
        if k != "mtp" or self.mtp is None:      # (the head's rounds: MTPDraft.record)
            self.sample(k, accepted, drafted, full)
        else:
            self.seen[k] += 1
        self.n[k] += 1
        for x in self.kinds:
            self.since[x] = 0 if x == k or x in shadows else self.since[x] + 1
        for x in shadows:
            self.shadowed[x] += 1

    def summary(self):
        out = []
        for k in self.kinds:
            t, ms = self.expect(k)
            out.append(f"{k} {self.n[k]} rounds" + (f" + {self.shadowed[k]} shadows" if self.shadowed[k] else "")
                       + (f" ({t:.2f} tok / {ms:.0f} ms)" if ms else ""))
        return ", ".join(out)


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
