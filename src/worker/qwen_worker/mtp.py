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
    counts within [1/2, 2] of the expectation (a pause is not the kind's cost).
    A kind's time is set by the median of its first NEW samples, which do not
    move g (until 2026-11-18 by its first alone: one slow first 3-draft round
    priced the 9B's depth 3 at 117 ms against 75 for depth 4, and with depth
    3 never taken again the price stood for the whole run)."""

    G, C = 0.2, 0.05
    NEW = 3

    def __init__(self, ms=None):
        self.g, self.c = 1.0, dict(ms or {})
        self.first = {}                 # kind -> its first samples / g (until NEW)
        self.sub = {}
        self.rate = {}                  # MTPDraft: depth i -> P(draft i kept | the ones before kept), the last request's
        self.keep = None                # MTPDraft: P(a round keeps a draft) outside runs of misses
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
            c.g, c.c = near.g, dict(near.c)
            c.rate = dict(near.rate)
            c.arm, c.keep = near.arm, near.keep
        return c

    def ms(self, kind):
        c = self.c.get(kind)
        return None if c is None else self.g * c

    def add(self, kind, ms):
        if ms <= 0.0:
            return
        f = self.first.setdefault(kind, [])
        if len(f) < self.NEW:
            f.append(ms / self.g)
            self.c[kind] = sorted(f)[(len(f) - 1) // 2]
            return
        r = min(2.0, max(0.5, ms / (self.g * self.c[kind])))
        self.g *= r ** self.G
        self.c[kind] *= r ** self.C


CLOCK = Clock()


class Avg:
    """The mean of samples weighted exp(-age / tau) (age in ms) with a prior
    worth `n` samples, which ages like one from the first call on (until
    2026-11-18 from t = 0: with `now` in perf_counter ms the prior was gone at
    the first sample, and DFlash's first draft of a request was its rate).
    The prior is a guess: it counts as samples spread as widely as its mean
    (se() of the prior alone: 1 / sqrt(n)). `e`: the real samples' weight
    (how much the mean knows about now)."""

    __slots__ = ("h", "q", "w", "e", "t")

    def __init__(self, prior, n):
        self.h, self.q, self.w, self.e, self.t = prior * float(n), 2.0 * prior * prior * n, float(n), 0.0, None

    def age(self, t, tau):
        if self.t is None:
            self.t = t
        elif t > self.t:
            d = max(1e-6, math.exp(-(t - self.t) / tau))
            self.h, self.q, self.w, self.e, self.t = self.h * d, self.q * d, self.w * d, self.e * d, t
        return self.e

    def add(self, x, t, tau):
        self.age(t, tau)
        self.h, self.q, self.w, self.e = self.h + x, self.q + x * x, self.w + 1.0, self.e + 1.0

    @property
    def mean(self):
        return self.h / self.w

    def se(self):
        """The mean's standard error relative to it (the weights as counts)."""
        m = self.h / self.w
        if m <= 0.0:
            return float("inf")
        return math.sqrt(max(0.0, self.q / self.w - m * m) / self.w) / m


class MTPDraft:
    """A loaded MTP head (`model`: mlx-vlm's Qwen3_5MTPDraftModel) bound to
    the target language model `lm`; per-request state in start().

    The draft count of a round (QWEN_MTP_K=0: round()) is one depth per
    class of requests, the k in 1..KMAX with the most expected tokens per
    ms, E(k) / T(k), kept until another one's is MARGIN higher. E(k) = 1 +
    q1 + q1 q2 + ..., q_i the keep rate of draft i given the ones before it
    kept (Clock.rate of the class; a request starts from the class's last
    values with the weight of PRIOR samples, then their running mean, an
    EMA of weight 1 / N past N samples): every round measures the depths up
    to its own, deeper ones are measured by probes. T(k): the measured wall
    time of a k-draft round (Clock, per class); unmeasured ones scaled from
    the nearest measured by the prior 1 + R * k. A probe round takes another
    depth: every other one all KMAX drafts (every depth's keep rate), the
    ones between the other depths in turn (their round times); every
    PROBE-th drafting round, twice as rarely after each probe (up to
    PROBE_MAX) unless it kept two drafts more than the depth takes, every
    PROBE-th again after the depth changes. No read of the head's
    confidence: the drafts and the verify are queued together, one sync per
    round. (Until 2026-11-18 the depth was chosen per round from the seed's
    confidence bin: on the 9B 0.92-0.98 of the better of one and three
    fixed drafts, taking two drafts on prose where a 3-row block costs about
    a 4-row one and splitting code between two and four. Ideal per-round
    gating, the bins' keep rates known in hindsight, beats the best fixed
    depth by 0-2.4 % on prose and 1.6-6.5 % on code (9B, q8 head): less
    than its estimates' noise costs; replayed on recorded acceptance
    (scripts/perf/mtp/depth_sim.py) it ran code at 0.92 of three drafts,
    one depth for the class at 0.99.)

    A round with no draft is never the choice of that rule: one plain step
    between two rounds has a round's host time and the head's catch-up but
    no step-ahead, and costs about a 1-draft round (27B: priced as a plain
    step it took a third of short Russian prose and ran 4-14 % behind one
    fixed draft; in prose an unlikely seed is a word's first token, the next
    ones its sure continuation). Plain steps are for text the head cannot
    draft: after `run` rounds in a row with no draft kept, a burst of
    BURSTS[0] plain steps with step-ahead (the head catches up in one flush
    queued with its last step), then drafting rounds again; after again()
    of them in a row with no draft kept the next burst, longer (BURSTS), a
    kept draft ends them: on text where every draft fails the plain loop's
    speed within ~4 % over 300 tokens, ~2 % past 1000 (depth_sim.py
    --none). `run` (need(), set at a run's first miss) is the shortest run
    of misses less likely than EPS at the class's measured keep rate
    (Clock.keep), within LOW..LOW_MAX; again() (set with it) the shortest
    one at most as likely as EPS2, within 1..LOW (2 rounds at 0.75, 4 at
    0.5). A fixed LOW fitted the 27B (keep ~0.75: (1/4)^4 = 1/256); on the
    9B (keep 0.5-0.65) four misses came by chance every 40-70 rounds, and
    with the bursts' doubling 9-59 % of the tokens went to plain steps
    where one draft a round was 15-20 % faster. Until 2026-11-18 one missed
    round after a burst started the next one, a coin flip at the 9B's
    sampled keep rate: chains of up to six bursts (152 of 256 tokens of a
    request). Bursts are not priced by the Clock: its ("mtp", 0) reads a
    burst token at 1/1.55 of a 1-draft round on both models, end to end
    about 1/1.3."""

    kind = "mtp"
    R = 0.25                            # prior: a k-draft round costs 1 + R * k plain steps
    BURSTS = (8, 16, 32, 64, 128)       # plain steps of the n-th burst in a row
    LOW, LOW_MAX, EPS = 4, 8, 1 / 256   # rounds in a row with no draft kept before a burst; need()
    EPS2 = 1 / 16                       # ... before the next burst after one; again()
    KEEP0, B = 0.8, 0.1                 # the keep rate (Clock.keep): the start, the EMA's weight
    PRIOR, N = 2, 16                    # the keep rates per depth: the class's weight at a request's start, 1 / the EMA's
    MARGIN = 0.02                       # E / T over the current depth's to take another
    PROBE, PROBE_MAX = 8, 64            # drafting rounds between two probes: the first, the most
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
        self.seed = None                # (token (1, 1), hidden (1, 1, H), None) or None
        self.block = None               # verify(): the head over the block's rows
        self.fresh()

    def fresh(self):
        """A new request's statistics (no head state: tools too)."""
        self.accept_lens, self.draft_lens = [], []
        self.KMAX = cfg.MTP_KMAX        # QWEN_MTP_KMAX
        self.low = 0                    # bursts in a row (no draft kept between them)
        self.miss = 0                   # drafting rounds in a row with no draft kept
        self.run = self.LOW             # ... that make a burst (need() at the run's first miss)
        self.after = 1                  # ... that make the next one after a burst (again(), as `run`)
        self.cur = None                 # the depth taken (the gate's)
        self.nround = 0                 # drafting rounds (the gate's) since the last probe
        self.every = self.PROBE         # ... and how many until the next
        self.turn = 0                   # probes so far (which depth the next one takes)
        self.probe = None               # the round under way is one: the depth it stands in for
        self.seen = {}                  # depth i -> samples of its keep rate in this request
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
            self.seed = (self._head(last)[0], last, None)

    def prime(self):
        """Queue the seed (after a flush)."""
        if self.seed is not None:
            mx.async_eval(self.seed[0])

    def state(self):
        """Arrays to evaluate with a step (the head's cache and seed)."""
        out = [c.state for c in (self.cache or []) if c.keys is not None]
        if self.seed is not None:
            out += list(self.seed[:2])
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
        k, steps = self.plan(min(self.KMAX, room))
        if not k:
            return steps
        self.ks[k] += 1
        return self.draft(k)

    def plan(self, top):
        """(k, 0): the next round takes k drafts (1..top), or (0, n): a
        burst of n plain steps (host statistics only: no sync)."""
        k = self.depth(top)
        if k:
            return k, 0
        self.low += 1
        self.ks[0] += 1
        self.miss = self.run - self.after               # drafting rounds before the next one
        return 0, self.BURSTS[min(self.low, len(self.BURSTS)) - 1]

    def depth(self, top):
        """Drafts of the next round: 0 (a burst of plain steps) after `run`
        rounds in a row that kept no draft; else the class's depth (best()),
        within `top`; a probe round another one."""
        if self.miss >= self.run:
            return 0
        k = self.best(top)
        self.nround += 1
        if top > 1 and self.nround >= self.every:
            self.nround, self.probe = 0, k
            self.turn += 1
            if k < top and self.turn % 2:
                return top              # (every depth's keep rate)
            others = [j for j in range(top, 0, -1) if j != k]
            return others[self.turn // (1 if k == top else 2) % len(others)]   # (their round times)
        return k

    def best(self, top):
        """The depth in 1..top with the most measured tokens per ms, E(k) /
        T(k); the current one kept unless another is MARGIN above it (a
        change: probes every PROBE-th round again)."""
        v = [0.0] + [self.expect(j) / self._ms(j) for j in range(1, top + 1)]
        b = max(range(1, top + 1), key=v.__getitem__)
        if self.cur is None:
            self.cur = b
        k = min(self.cur, top)
        if v[b] > v[k] * (1.0 + self.MARGIN):
            self.cur, k, self.nround, self.every = b, b, 0, self.PROBE
        return k

    def expect(self, k):
        """E(k): the class's mean tokens of a k-draft round, 1 + q1 + q1 q2
        + ..., q_i = P(draft i kept | the ones before kept) (Clock.rate)."""
        rate = self._clock().rate
        e = p = 1.0
        for i in range(1, k + 1):
            p *= rate.get(i, self.KEEP0)
            e += p
        return e

    def need(self):
        """Rounds in a row with no draft kept before a burst: the fewest
        whose chance at the class's keep rate p (Clock.keep, KEEP0 until
        measured) is below EPS, (1 - p)^n < EPS, within LOW..LOW_MAX (a
        head that keeps nothing still bursts after LOW_MAX)."""
        return min(self.LOW_MAX, max(self.LOW, self._runs(self.EPS)))

    def again(self):
        """Rounds in a row with no draft kept after a burst before the next
        one: the fewest whose chance at the class's keep rate is at most
        EPS2, 1..LOW (below a keep rate of 1/2 a drafting round hardly
        beats a plain step: a burst by chance costs little)."""
        return min(self.LOW, max(1, self._runs(self.EPS2)))

    def _runs(self, eps):
        """The fewest n with (1 - p)^n <= eps, p the class's keep rate
        (Clock.keep, KEEP0 until measured)."""
        p = self._clock().keep
        p = self.KEEP0 if p is None else p
        if p >= 1.0:
            return 1
        if p <= 0.0:
            return self.LOW_MAX
        return math.ceil(math.log(eps) / math.log(1.0 - p) - 1e-9)

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
        tok = self._head(out)[0]
        self.block = (out, tok, out.shape[1])
        return [tok]

    def settle(self, kept, h):
        """After the walk of a verify(): the first `kept` rows had their real
        successor (the rest leave the head's cache); `h` (1, n, H): the
        positions the commit kept, those past `kept` pushed (a thinking
        budget's cut: the last one, its successor forced after it)."""
        out, tok, t = self.block
        self.block = None
        self._trim(t - kept)
        self.skip += kept               # their successors: the first tokens emitted
        self.seed = (tok[:, kept - 1:kept], out[:, kept - 1:kept], None) if kept else None
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
        if not accepted and not self.miss:
            self.run, self.after = self.need(), self.again()    # (at the rate before the run)
        if drafted and not self.low and self.miss < self.LOW:
            # the keep rate outside runs of misses (rounds deep in one, and
            # the ones after a burst, are the text need() tests for)
            p = self.KEEP0 if ck.keep is None else ck.keep
            ck.keep = p + self.B * ((accepted >= 1) - p)
        if accepted:
            self.miss = self.low = 0
        else:
            self.miss += 1
        if self.probe is not None:      # one that kept 2 more than the depth's drafts: soon again
            self.every = self.PROBE if accepted >= self.probe + 2 else min(2 * self.every, self.PROBE_MAX)
            self.probe = None
        for i in range(1, min(drafted, accepted + 1) + 1):     # the depths reached: the ones before kept
            n = self.seen[i] = self.seen.get(i, 0) + 1
            r = ck.rate.get(i, self.KEEP0)
            ck.rate[i] = r + ((accepted >= i) - r) / min(self.N, n + self.PRIOR)

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
        lm_head over h (1, n, H), lazy (the gate reads no confidence: only
        tools evaluate it)."""
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
    its last sample when it is not the current arm. It is the default arm:
    taken on its rate alone, never shadow-drafted (its gate already reads
    the head's confidence every round). The others: wall time per round from
    the request class's Clock (process-wide, steady rounds only: _spec leaves
    out a round after a switch, whose drafter catches up on the positions it
    missed, and rounds with shadow drafts); tokens per round (1 + accepted
    drafts) as means over the last TAU ms (Avg, from a prior worth PRIOR
    rounds), plain exactly 1.

    A drafter that is not taken is still sampled: shadows() names the ones
    to draft this round WITHOUT verifying; their drafts are scored against
    the tokens the request emits from the same anchor (Shadows) - what a
    verify would have kept, at the cost of the draft alone (27B: a DFlash
    block ~17 ms against a 110-180 ms round). A drafter is drafted every
    round until it has WARM drafts and while it looks better than the
    current arm without having shown it (a trial); else as often as its
    drafts cost NEAR of the time (an arm within CLOSE of the current one's
    rate) or FAR.

    The current arm is left for the head when the head's rate is MARGIN
    higher; for a drafter only when its rate less one standard error of its
    tokens per round is (Avg.se: DFlash's rounds spread with a CV of
    0.5-0.65; a switch on 2.5 samples, as until 2026-11-18, followed a lucky
    draft on prose and an unlucky one on code: 7-9 switches a request).

    The head's arm starts a request from the class's last rate (Clock.arm),
    a guess worth PRIOR rounds; with that known a request starts on the
    first arm of START it has, PROBE rounds (DFlash: where it leads, code,
    no round is spent finding that out; where it loses its own first rounds
    say so and the head takes over; on the 9B 43 % of code rounds keep 0-1
    drafts, and one such first round sent the request to the head). The guess stands only until the request has timed
    PROBE rounds of the head: a carried rate was 2x low on the 9B (measured
    in a slowed-down request, rescaled by the Clock's drift only in part)
    and kept two prose requests on DFlash 22-40 % behind the head. So a
    head not timed in this request is probed once, after WAIT rounds of the
    current arm: PROBE + 1 rounds (the first one after the switch is not
    timed), after which the arm left is taken back unless the head beats it
    by MARGIN (a probe is a measurement, not a switch; where DFlash leads,
    code, it costs ~3 head rounds a request). The first request of a class
    starts on the head. An arm other than the head never timed in this class
    of requests is taken PROBE rounds first."""

    TAU = 2500.0
    PRIOR = 2                                   # rounds: the weight of an arm's prior (DFlash's, the head's class rate)
    PRIORS = {"dflash": 3.0}
    START = ("dflash", "mtp", "plain")
    MARGIN = 0.02                               # (the gate allows 3 % below the best arm)
    WARM = 3
    NEAR, FAR, CLOSE = 0.01, 0.004, 0.8
    COST = {"dflash": 0.10}                     # a shadow draft, of the arm's round time
    PROBE = 3
    WAIT = 8                                    # rounds of the current arm before the head's probe
    BURST, SHORT = 8, 2

    def __init__(self, kinds, mtp=None, clock=None, priors=None):
        self.kinds = list(kinds)
        self.mtp = mtp                          # MTPDraft (the "mtp" arm: the gated head)
        self.head = mtp is not None and "mtp" in self.kinds
        self.clock = clock if clock is not None else CLOCK
        pri = dict(self.PRIORS, **(priors or {}))
        self.tok = {k: Avg(pri.get(k, 2.0), self.PRIOR) for k in kinds}
        self.m = [0.0, 0.0, 0.0, None, 1.0]     # the head's arm in this request: decayed tokens, ms, rounds; t; Clock g
        self.guess = self.clock.arm if self.head else None  # (tokens, ms per round, g): the class's last rate
        self.own = 0                            # the head's timed rounds in this request
        self.back = None                        # the arm a probe of the head left
        self.tried = False                      # the head probed in this request
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
        d = 1.0 if m[3] is None else math.exp(-max(0.0, self.t - m[3]) / self.TAU)
        m[0], m[1], m[2] = m[0] * d + tokens, m[1] * d + ms, m[2] * d + 1.0
        m[3], m[4] = self.t if m[3] is None else max(m[3], self.t), self.clock.g
        self.own += 1
        tok, per = self.expect("mtp")
        self.clock.arm = (tok, per, self.clock.g)

    def expect(self, k):
        """(tokens per round, ms per round or None: never timed) of arm k."""
        if k == "plain":
            return 1.0, self.clock.ms(k)
        if k == "mtp" and self.head:
            m, g, tok, ms, n = self.m, self.clock.g, 0.0, 0.0, 0.0
            if m[2] > 0.0:
                tok, ms, n = m[0], m[1] * g / m[4], m[2]
            if self.guess is not None and self.own < self.PROBE:
                t0, ms0, g0 = self.guess
                tok, ms, n = tok + t0 * self.PRIOR, ms + ms0 * g / g0 * self.PRIOR, n + self.PRIOR
            if n <= 0.0:
                return 1.0, None
            return tok / n, ms / n
        return self.tok[k].mean, self.clock.ms(k)

    def se(self, k):
        """The relative standard error of arm k's tokens per round (0: a plain
        step, and the head's arm, which needs no evidence)."""
        if k == "plain" or (k == "mtp" and self.head):
            return 0.0
        self.tok[k].age(self.t, self.TAU)
        return self.tok[k].se()

    def shown(self, k, cur):
        """Has arm k shown (by one standard error) that it beats `cur` by MARGIN?"""
        r = self.r
        return r[k] * (1.0 - self.se(k)) > r[cur] * (1.0 + self.MARGIN)

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
                k, back, self.back = self.cur, self.back, None
                if k is None:                   # its first PROBE rounds: a measurement
                    k = next((x for x in self.START if x in self.kinds), self.kinds[0])
                    self.left, self.probe, self.r = self.PROBE - 1, True, None
                elif back is not None:          # a probe of the head ended: a measurement
                    k = "mtp" if r["mtp"] > r[back] * (1.0 + self.MARGIN) else back
                elif self.head and k != "mtp" and not (self.own or self.tried) and self.n[k] >= self.WAIT:
                    k, self.back, self.left, self.probe, self.tried = "mtp", k, self.PROBE, True, True
                    self.r = None               # (no shadow drafts: its rounds are timed)
                else:
                    ok = [x for x in self.kinds if x != k and (
                        r[x] > r[k] * (1.0 + self.MARGIN) if self.se(x) == 0.0 else self.shown(x, k))]
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
                hot = r[k] > r[self.cur] * (1.0 + self.MARGIN) and not self.shown(k, self.cur)
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
