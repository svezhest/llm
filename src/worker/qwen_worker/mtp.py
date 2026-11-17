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
entered the target cache (prefill slabs, plain steps, a verify block's kept
part), `emit` the tokens in emission order (each is the successor of the
oldest position without one); `flush` runs the head over every complete
pair at once (lazy, no host sync). The chained drafts add approximate
entries (the head's hidden for the target's); a verify of the head's own
round drops them and runs the head over the block's rows with the target's
real hidden states and tokens (`verify`, lazy, evaluated with the verify),
and after the walk `settle` keeps the rows of the kept positions: every
entry is a real one and the next round's seed is ready (2026-11-17; mlx-vlm
keeps the chained entries of accepted drafts: on the 27B fewer drafts
accepted in 8 of 10 probed cells, more in 1, and its seed needs a flush
after the round).

The target verifies every draft (the same verify forward as DFlash), so the
drafts only move acceptance; greedy output == the DFlash path's.
"""

import math

import mlx.core as mx

from . import config as cfg
from .config import log
from .kernels import _rows_linear


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
        self.rate = {}                  # MTPDraft: depth i >= 2 -> P(draft i kept | the ones before kept)
        self.seed = {}                  # MTPDraft: bin of the seed's confidence -> P(the seed kept)
        self.arm = None                 # Chooser: the head's arm as last measured (tokens, ms per round, g)

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
            c.g, c.c, c.rate, c.seed = near.g, dict(near.c), dict(near.rate), dict(near.seed)
            c.arm = near.arm
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
    the target language model `lm`; per-request state in start().

    The draft count of a round (QWEN_MTP_K=0: round()): k = argmax over
    0..KMAX of E(k) / T(k), the expected tokens per ms. E(k) = 1 + p1 +
    p1 p2 + ...: p1 is the measured rate at which seeds of this confidence
    were kept (the seed's top-1 probability, evaluated with the previous
    round - verify() runs the head over the verify block's rows on the
    target's tokens, so after the walk the seed and its confidence are read
    at no extra sync - in BINS bins, Clock.seed of the request's class: under
    greedy a seed of confidence 0.4 is kept far more often than 0.4; a bin
    starts at SEED0 or above, so it is drafted until measured, and every
    PROBE0-th no-draft decision drafts once), p_i for the chained drafts the
    measured rate at which draft i was kept when the ones before it were
    (Clock.rate of the request's class, an EMA from RATE0; the head's own
    confidence in a chained draft is not used: it does not know its input is
    an approximation, and on prose "seed above 0.8" took four drafts where
    three or more were kept in 1 round of 10). Every PROBE-th drafting round
    takes all the drafts there is room for, so the rates of depths the rule
    does not take stay measured. T(k): the measured wall time of a k-draft
    round (Clock, per class of requests), T(0) a plain step's; unmeasured
    ones scaled from the nearest measured by the prior 1 + R * k. k = 0 is a
    burst of plain steps: one step, and only from the 4th no-draft round in a
    row longer (BURSTS: 1, 1, 1, 2, 4, 8 steps with step-ahead; a drafting
    round resets them): in prose an unlikely seed is a word's first token and
    the next ones are its sure continuation, which a longer burst would skip
    (bursts of 2..8 right away cost the gated head 4-13 % against one fixed
    draft on Russian prose); a long run of them is text the head cannot
    draft, where 8-step bursts keep the plain loop's speed. The head catches
    up after a burst in one flush queued with its last step. The drafts and the verify
    are queued together: one sync per round."""

    kind = "mtp"
    R = 0.25                            # prior: a k-draft round costs 1 + R * k plain steps
    BURSTS = (1, 1, 1, 2, 4, 8)         # plain steps of the n-th k = 0 round in a row
    RATE0, A = 0.5, 0.2                 # a chained draft's keep rate: the start, the EMA's weight
    PROBE = 8                           # every PROBE-th drafting round: all the drafts
    BINS, SEED0, B = 5, 0.8, 0.1        # the seed's keep rate per confidence bin: the start (at least), the EMA's weight
    PROBE0 = 16                         # every PROBE0-th no-draft decision: one draft
    said = False                        # the lm_head's path over a block: logged once
    CHUNK = 512                         # head rows per forward (flush)

    def __init__(self, model, lm, k=None):
        self.model, self.lm = model, lm
        self.k = k                      # drafts per round; None: cfg.MTP_K
        self.hidden_size = int(lm.args.hidden_size)
        self.cache = None
        self.clock = None
        self.fresh()

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
        self.succ = []                  # successors not paired yet (ints, or lazy (1,) arrays)
        self.skip = 0                   # emitted successors already paired (verify / burst)
        self.appended = 0               # chained entries of the current round
        self.seed = None                # (token (1, 1), hidden (1, 1, H), confidence (1, 1)) or None
        self.block = None               # verify(): the head over the block's rows
        self.fresh()

    def fresh(self):
        """A new request's statistics (no head state: tools too)."""
        self.accept_lens, self.draft_lens = [], []
        self.KMAX = cfg.MTP_KMAX        # QWEN_MTP_KMAX
        self.low = 0                    # k = 0 rounds in a row
        self.nround = 0                 # drafting rounds (the gate's)
        self.nzero = 0                  # no-draft decisions
        self.bin = None                 # the confidence bin of the round's seed (the gate's rounds)
        self.ks = [0] * (max(self.KMAX, cfg.MTP_K) + 1)     # rounds by draft count (0: bursts)
        self.burst_tokens = 0           # tokens of the k = 0 rounds' plain steps
        self.dropped = 0                # drafts computed and not verified

    def stats(self):
        """The request's gate statistics (spec_stats["mtp"])."""
        acc = [0] * (max(self.draft_lens, default=0) + 1)
        for a in self.accept_lens:
            acc[a] += 1
        return dict(k=list(self.ks), accepted=acc, burst_tokens=self.burst_tokens,
                    dropped=self.dropped)

    def _clock(self):
        return self.clock if self.clock is not None else CLOCK

    @staticmethod
    def kind_of(k):
        """The Clock kind of a k-draft round. 0: a token of the head's
        no-draft rounds, their sync and the head's catch-up included (a plain
        step of the plain loop is "plain": ~15 % less on the 27B, and taking
        it for the price of a no-draft round sent a quarter of the tokens to
        bursts where one draft per round was faster)."""
        return ("mtp", k)

    def _ms(self, k):
        """Expected wall time of a k-draft round (k = 0: one plain step):
        measured, else scaled from the nearest measured k by the prior
        1 + R * k (none measured: the prior itself, in plain steps)."""
        ck = self._clock()
        ms = ck.ms(self.kind_of(k))
        if ms is not None:
            return ms
        known = [(j, ck.ms(self.kind_of(j))) for j in range(max(self.KMAX, cfg.MTP_K) + 1)]
        known = [(j, m) for j, m in known if m is not None]
        if known:
            j, m = min(known, key=lambda x: abs(x[0] - k))
            return m * (1.0 + self.R * k) / (1.0 + self.R * j)
        return 1.0 + self.R * k

    @staticmethod
    def gain(confs, k, last):
        """Expected tokens of a k-draft round: 1 + sum_i prod_{j<=i} c_j, the
        confidences `confs` known, `last` for the drafts past them."""
        e, p = 1.0, 1.0
        for i in range(k):
            p *= confs[i] if i < len(confs) else last
            e += p
        return e

    def push(self, h):
        """Hidden states (1, n, H, the target's post-norm) of the next n
        positions that entered the target cache."""
        if h is not None and h.shape[1]:
            self.pend.append(h)
            self.npend += h.shape[1]
            self.seed = None

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

    def emit_lazy(self, y):
        """The successor of the last pushed position as a lazy (1,) token not
        read yet (a burst's last step): paired at once, its emit() skipped."""
        self.succ.append(y)
        self.skip += 1

    def flush(self):
        """Run the head over every (hidden, successor) pair queued; the seed
        is the head's output at the last one when nothing is left pending."""
        r = min(self.npend, len(self.succ))
        if r == 0:
            return
        h = self.pend[0] if len(self.pend) == 1 else mx.concatenate(self.pend, axis=1)
        s = self.succ[:r]
        if all(isinstance(t, int) for t in s):
            toks = mx.array([s], dtype=mx.int32)
        else:
            toks = mx.concatenate([mx.array([t], dtype=mx.int32) if isinstance(t, int)
                                   else t.reshape(1).astype(mx.int32) for t in s])[None]
        # (a prefill slab at a long context: CHUNK rows at a time, the head's
        # attention scores are rows x keys x heads)
        for lo in range(0, r, self.CHUNK):
            out = self._forward(toks[:, lo:lo + self.CHUNK], h[:, lo:min(r, lo + self.CHUNK)])
        rest = h[:, r:]
        self.pend, self.npend = ([rest] if rest.shape[1] else []), rest.shape[1]
        self.succ = self.succ[r:]
        self.seed = None
        if not self.npend:
            last = out[:, -1:, :]
            tok, conf = self._head(last)
            self.seed = (tok, last, conf)

    def prime(self):
        """Queue the seed and its confidence (after a flush)."""
        if self.seed is not None:
            mx.async_eval(self.seed[0], self.seed[2])

    def state(self):
        """Arrays to evaluate with a step (the head's cache and seed)."""
        out = [c.state for c in (self.cache or []) if c.keys is not None]
        if self.seed is not None:
            out += list(self.seed)
        return out

    # ---- a round ----

    def round(self, room):
        """The drafts (1, k) of the next round, lazy, k in 1..min(room, KMAX)
        (QWEN_MTP_K > 0: that many, no gate), or an int: the round takes no
        draft, that many plain steps (the n-th low seed in a row: BURSTS)."""
        k0 = self.k if self.k is not None else cfg.MTP_K
        if k0 > 0:
            k = min(k0, room)
            self.ks[k] += 1
            return self.draft(k)
        self.flush()
        if self.seed is None:
            raise RuntimeError("mtp: no seed (a pushed position without its successor)")
        tok, h, conf = self.seed
        top = min(self.KMAX, room)
        k = self.depth(conf.item(), [self._ms(k) for k in range(top + 1)], top)
        if k == 0:                      # (the seed's confidence: evaluated with the last round)
            self.low += 1
            self.ks[0] += 1
            return self.BURSTS[min(self.low, len(self.BURSTS)) - 1]
        self.low = 0
        toks = [tok]
        self.appended = 0
        for _ in range(k - 1):
            h = self._forward(tok, h)
            self.appended += 1
            tok = self._head(h)[0]
            toks.append(tok)
        self.seed = None
        self.ks[k] += 1
        return mx.concatenate(toks, axis=1) if len(toks) > 1 else toks[0]

    def depth(self, c1, T, top):
        """Drafts of the next round, 0..top: the count with the most expected
        tokens per ms, gain(k) / T[k] (0: a plain step), from the measured
        keep rates of seeds of confidence `c1` and of the chained drafts
        (the request's class); every PROBE0-th no-draft decision one draft,
        every PROBE-th drafting round all `top`."""
        ck = self._clock()
        b = min(self.BINS - 1, int(c1 * self.BINS))
        ps = [ck.seed.get(b, self._seed0(b))] + [ck.rate.get(i, self.RATE0) for i in range(2, top + 1)]
        k = max(range(top + 1), key=lambda j: self.gain(ps, j, 0.0) / T[j])
        if not k:
            self.nzero += 1
            if self.nzero % self.PROBE0:
                return 0
            k = 1
        self.bin = b
        self.nround += 1
        if self.nround % self.PROBE == 0:
            k = top
        return k

    def _seed0(self, b):
        return max(self.SEED0, (b + 0.5) / self.BINS)

    def draft(self, n):
        """(1, n) draft tokens (lazy): the seed, then n - 1 chained steps
        (entries settled by verify / settle or round_end)."""
        self.flush()
        if self.seed is None:
            raise RuntimeError("mtp: no seed (a pushed position without its successor)")
        tok, h, _ = self.seed
        toks = [tok]
        self.appended = 0
        for _ in range(n - 1):
            h = self._forward(tok, h)
            self.appended += 1
            tok = self._head(h)[0]
            toks.append(tok)
        self.seed = None
        return mx.concatenate(toks, axis=1) if len(toks) > 1 else toks[0]

    def verify(self, h, targets):
        """The verify forward of the head's own round, lazy: `h` (1, T, H)
        the target's hidden states of the block's positions (the anchor and
        the drafts), `targets` (1, T) the target's tokens at them = each
        position's successor wherever the walk keeps it. The chained entries
        leave the head's cache and the head runs over the T rows (the next
        seed is the row of the last kept position): -> the arrays to evaluate
        with the verify."""
        self._trim(self.appended)
        out = self._forward(targets.reshape(1, -1).astype(mx.int32), h)
        tok, conf = self._head(out)
        self.block = (out, tok, conf, out.shape[1])
        return [tok, conf]

    def settle(self, kept, h):
        """After the walk of a verify(): the first `kept` rows had their real
        successor (the rest leave the head's cache); `h` (1, n, H): the
        positions the commit kept, those past `kept` pushed (a thinking
        budget's cut: the last one, its successor forced after it)."""
        out, tok, conf, t = self.block
        self.block = None
        self._trim(t - kept)
        self.skip += kept               # their successors: the first tokens emitted
        self.seed = ((tok[:, kept - 1:kept], out[:, kept - 1:kept], conf[:, kept - 1:kept])
                     if kept else None)
        self.push(h[:, kept:])

    def round_end(self, accepted, h=None):
        """After a verify the head did not settle (another drafter's round, a
        lookup, tools): its chained entries leave its cache, `h` (1, >=
        accepted + 1, H): the positions the commit kept (the anchor + the
        accepted drafts), pushed (their successors: the tokens emitted)."""
        self._trim(self.appended)
        self.seed = None
        if h is not None:
            self.push(h[:, :int(accepted) + 1])

    def record(self, accepted, drafted):
        """After a round's tokens were read: its acceptance (the round's wall
        time goes to the Clock from _spec)."""
        accepted, drafted = int(accepted), int(drafted)
        self.accept_lens.append(accepted)
        self.draft_lens.append(drafted)
        ck = self._clock()
        rate, b, self.bin = ck.rate, self.bin, None
        if b is not None and drafted:
            r = ck.seed.get(b, self._seed0(b))
            ck.seed[b] = r + self.B * ((accepted >= 1) - r)
        for i in range(2, min(drafted, accepted + 1) + 1):      # depths reached: the ones before kept
            r = rate.get(i, self.RATE0)
            rate[i] = r + self.A * ((accepted >= i) - r)

    # ---- the head ----

    def _trim(self, n):
        if n:
            for c in self.cache:
                c.trim(n)
            self.pos -= n
        self.appended = 0

    def _forward(self, toks, h):
        """Head entries for the pairs (h[:, i], toks[:, i]) at positions
        self.pos.. -> the head's output hidden states (1, n, H)."""
        n = toks.shape[1]
        emb = self.lm.model.embed_tokens(toks)
        pid = (mx.arange(n, dtype=mx.int32) + self.pos)[None]
        out = self.model._forward_hidden(emb, h.astype(emb.dtype), self.cache, pid)
        self.pos += n
        return out

    def _head(self, h):
        """-> (argmax (1, n) int32, top-1 probability (1, n) fp32) of the
        lm_head over h (1, n, H)."""
        if h.shape[1] > 1 and not MTPDraft.said:
            MTPDraft.said = True
            log.info("mtp: the head's lm_head over a %d-row block: input %s, mrqmv mark %s, QWEN_VERIFY_QMV %s",
                     h.shape[1], h.dtype, bool(self.lm.lm_head.__dict__.get("_qmv")), cfg.VERIFY_QMV)
        logits = _rows_linear(self.lm.lm_head, h).astype(mx.float32)
        m = mx.max(logits, axis=-1)
        return (mx.argmax(logits, axis=-1).astype(mx.int32),
                mx.exp(m - mx.logsumexp(logits, axis=-1)))


class Chooser:
    """Which arm a round takes (QWEN_DRAFT=auto; QWEN_DRAFT_HYBRID /
    QWEN_SPEC_PLAIN): "dflash", "mtp" (the head's gated rounds, its plain
    bursts included) or "plain" (a burst of plain steps; only without the
    head, whose gate takes plain steps itself) - the one with the most
    expected tokens per ms.

    The head's arm is measured as it runs: tokens over wall time of its
    steady rounds, decayed over TAU ms, scaled by the Clock's drift since
    its last sample when it is not the current arm. It is the start and the
    default arm: it needs no evidence to be taken back, is never probed and
    never shadow-drafted (its gate already reads the head's confidence every
    round). The others: wall time per round from the request class's Clock
    (process-wide, steady rounds only: _spec leaves out a round after a
    switch, whose drafter catches up on the positions it missed, and rounds
    with shadow drafts); tokens per round (1 + accepted drafts) as means over
    the last TAU ms (Avg), plain exactly 1.

    A drafter that is not taken is still sampled: shadows() names the ones
    to draft this round WITHOUT verifying; their drafts are scored against
    the tokens the request emits from the same anchor (Shadows) - what a
    verify would have kept, at the cost of the draft alone (27B: a DFlash
    block ~17 ms against a 110-180 ms round). A drafter is drafted every
    round until it has WARM drafts and while it looks better than the
    current arm without the samples to say so (below NEED: a trial); else as
    often as its drafts cost NEAR of the time (an arm within CLOSE of the
    current one's rate) or FAR.

    The head's arm is the first one until it is measured in the request's
    class (Clock.arm: its rate in the class's last request, the weight of
    PRIOR rounds in the next one); with that known a request starts on the
    first arm of START it has - DFlash: where it leads (code) no round is
    spent finding that out, where it loses its own first rounds say so and
    the head's arm, which needs no evidence, takes over. The current arm is
    left only for one whose rate is MARGIN higher and, a drafter other than
    the head, known from >= NEED recent samples. An arm other than the head
    never timed in this class of requests is taken PROBE rounds first."""

    TAU = 2500.0
    PRIOR = 2
    PRIORS = {"dflash": 3.0}
    START = ("dflash", "mtp", "plain")
    MARGIN = 0.02                               # (the gate allows 3 % below the best arm)
    WARM, NEED = 3, 2.5
    NEAR, FAR, CLOSE = 0.01, 0.004, 0.8
    COST = {"dflash": 0.10}                     # a shadow draft, of the arm's round time
    PROBE = 3
    BURST, SHORT = 8, 2

    def __init__(self, kinds, mtp=None, clock=None, priors=None):
        self.kinds = list(kinds)
        self.mtp = mtp                          # MTPDraft (the "mtp" arm: the gated head)
        self.head = mtp is not None and "mtp" in self.kinds
        self.clock = clock if clock is not None else CLOCK
        pri = dict(self.PRIORS, **(priors or {}))
        self.tok = {k: Avg(pri.get(k, 2.0), self.PRIOR) for k in kinds}
        self.m = [0.0, 0.0, 0.0, 0.0, 1.0]      # the head's arm: decayed tokens, ms, rounds; t; Clock g
        if self.head and self.clock.arm is not None:
            tok, ms, g = self.clock.arm
            self.m = [tok * self.PRIOR, ms * self.PRIOR, float(self.PRIOR), 0.0, g]
        self.n = {k: 0 for k in kinds}          # rounds taken
        self.got = {k: 0 for k in kinds}        # tokens of those rounds
        self.shadowed = {k: 0 for k in kinds}   # shadow drafts
        self.pend = {k: 0 for k in kinds}       # shadow drafts not scored yet
        self.last = {k: 0.0 for k in kinds}     # ms: the arm's last draft
        self.cur, self.left, self.rounds, self.r = None, 0, 0, None
        self.t, self.hot, self.probe = 0.0, False, False
        self.switches = 0

    def timed(self, k, tokens, ms):
        """A steady round (or burst) of the head's arm: `tokens` in `ms`."""
        if not (self.head and k == "mtp") or ms <= 0.0:
            return
        m = self.m
        d = math.exp(-max(0.0, self.t - m[3]) / self.TAU)
        m[0], m[1], m[2] = m[0] * d + tokens, m[1] * d + ms, m[2] * d + 1.0
        m[3], m[4] = max(m[3], self.t), self.clock.g
        self.clock.arm = (m[0] / m[2], m[1] / m[2], m[4])

    def expect(self, k):
        """(tokens per round, ms per round or None: never timed) of arm k."""
        if k == "plain":
            return 1.0, self.clock.ms(k)
        if k == "mtp" and self.head:
            m = self.m
            if m[2] <= 0.0:
                return 1.0, None
            return m[0] / m[2], m[1] / m[2] * self.clock.g / m[4]
        return self.tok[k].mean, self.clock.ms(k)

    def evidence(self, k):
        """The weight of arm k's recent samples (a plain step and the head's
        arm need none)."""
        if k == "plain" or (k == "mtp" and self.head):
            return float("inf")
        return self.tok[k].age(self.t, self.TAU)

    def pick(self, force=None, now=None):
        """The arm of the next round. `now`: ms, any origin (the samples'
        ages; None: the time stands still)."""
        self.rounds += 1
        self.r = None
        self.probe = False
        if now is not None:
            self.t = max(self.t, now)
        was = self.cur
        if force in self.kinds:
            k = force
        elif self.left > 0:
            k, self.left, self.probe = self.cur, self.left - 1, True    # the running probe
        else:
            ex = {k: self.expect(k) for k in self.kinds}
            new = [k for k in self.kinds if ex[k][1] is None and not (k == "mtp" and self.head)]
            if new:
                k, self.left, self.probe = new[0], self.PROBE - 1, True
            elif ex.get("mtp", (0, 0))[1] is None:
                k = "mtp"                       # the head's arm until it is measured
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
        """The drafters (not the arm just picked, nor the head) to draft this
        round without verifying."""
        self.hot = False
        if self.r is None:                      # a forced arm, a probe
            return []
        out, r = [], self.r
        for k in self.kinds:
            if k in ("plain", "mtp", self.cur):
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
        if k not in ("plain", "mtp") and full:
            self.tok[k].add(1.0 + accepted, self.t, self.TAU)

    def steps(self):
        """Plain steps of a plain round (one burst with step-ahead; short
        while a drafter is on trial or the plain step is being timed)."""
        return self.SHORT if self.hot or self.probe else self.BURST

    def done(self, accepted, drafted=0, full=True, shadows=(), steps=1):
        """The round of the last pick() kept `accepted` of its `drafted`
        drafts (`full`: a block of the usual size); `shadows`: the arms
        drafted beside it; `steps`: the tokens of a plain burst (the plain
        arm's, or the head's k = 0 round)."""
        k = self.cur
        self._add(k, accepted, drafted, full)
        self.n[k] += 1
        self.got[k] += steps if k == "plain" or (k == "mtp" and not drafted) else 1 + accepted
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
    log.info("mtp drafter loaded: %s (%d layer(s), %.0f MiB, %s)", path,
             len(model.layers), sum(v.nbytes for _, v in _flat(model)) / 2**20,
             f"{cfg.MTP_K} drafts per round" if cfg.MTP_K > 0 else f"gated 0..{cfg.MTP_KMAX} drafts per round")
    return d


def _flat(model):
    from mlx.utils import tree_flatten
    return tree_flatten(model.parameters())
