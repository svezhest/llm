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

import time

import mlx.core as mx

from . import config as cfg
from .config import log


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
        self.accept_lens, self.draft_lens = [], []
        self.p = [self.P0] * self.KMAX  # QWEN_MTP_K=0: P(draft i kept | drafts < i kept)

    # QWEN_MTP_K=0: the draft count per round from the request's own acceptance
    KMAX, P0, ALPHA, EXPLORE = 4, 0.5, 0.25, 8

    def next_k(self):
        """Drafts for the next round: QWEN_MTP_K, or (0) the k in 1..KMAX with
        the most expected tokens per unit of round time, (1 + sum_i prod_{j<=i}
        p_j) / (1 + QWEN_MTP_COST * k), from per-depth acceptance averages; every
        EXPLORE-th round one more (deeper averages would go stale)."""
        if cfg.MTP_K > 0:
            return cfg.MTP_K
        best, bk, e, q = -1.0, 1, 1.0, 1.0
        for k in range(1, self.KMAX + 1):
            q *= self.p[k - 1]
            e += q
            v = e / (1.0 + cfg.MTP_COST * k)
            if v > best:
                best, bk = v, k
        if len(self.accept_lens) % self.EXPLORE == self.EXPLORE - 1:
            bk = min(self.KMAX, bk + 1)
        return bk

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

    def record(self, accepted, drafted):
        self.accept_lens.append(int(accepted))
        self.draft_lens.append(int(drafted))
        for i in range(min(int(accepted) + 1, int(drafted), self.KMAX)):
            hit = 1.0 if i < accepted else 0.0
            self.p[i] += self.ALPHA * (hit - self.p[i])

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
    most expected tokens per ms, tokens per round / wall time per round.

    Tokens per round (1 + accepted drafts) start at a prior worth PRIOR rounds
    (a drafter gets the benefit of the doubt; plain is exactly 1) and follow
    an exponential average (weight max(ALPHA, 1 / (rounds + PRIOR))). Wall
    time: the arm's first round is replaced by its second (a DFlash round
    right after prefill feeds the drafter the whole prompt window), then an
    exponential average. Every arm is first taken PROBE rounds in a row; an
    arm not taken while EXPLORE tokens were emitted gets PROBE rounds again
    (re-probing: a drafter that lost on one stretch of text can win on the
    next). A plain step costs about half a drafter round on the 27B, so a
    losing drafter's re-probe costs ~PROBE plain steps per EXPLORE tokens."""

    ALPHA = 0.2
    EXPLORE = 64
    PROBE = 2
    PRIOR = 2
    PRIORS = {"dflash": 3.0, "mtp": 2.5, "plain": 1.0}

    def __init__(self, kinds):
        self.kinds = list(kinds)
        self.tok = {k: self.PRIORS.get(k, 2.0) for k in kinds}
        self.ms = {k: None for k in kinds}
        self.n = {k: 0 for k in kinds}          # rounds taken
        self.idle = {k: 0 for k in kinds}       # tokens emitted since the arm's last round
        self.left = {k: self.PROBE for k in kinds}   # probe rounds still owed
        self.t0 = self.cur = None
        self.taken = self.n

    def rate(self, k):
        return self.tok[k] / self.ms[k] if self.ms[k] else float("inf")

    def pick(self, force=None):
        if force in self.kinds:
            k = force
        elif self.cur is not None and self.left[self.cur] > 0:
            k = self.cur                        # finish the running probe
        else:
            owed = [k for k in self.kinds if self.left[k] > 0]
            stale = [k for k in self.kinds if self.idle[k] >= self.EXPLORE]
            if not owed and stale:
                self.left[stale[0]] = self.PROBE
                owed = stale[:1]
            k = owed[0] if owed else max(self.kinds, key=self.rate)
        self.cur, self.t0 = k, time.perf_counter()
        return k

    def done(self, accepted):
        """The round of the last pick() kept `accepted` drafts (call after
        its tokens were read: the wall time covers drafting + verify)."""
        k, ms = self.cur, 1000.0 * (time.perf_counter() - self.t0)
        tok = 1.0 + accepted
        if k != "plain":
            a = max(self.ALPHA, 1.0 / (self.n[k] + 1 + self.PRIOR))
            self.tok[k] = (1 - a) * self.tok[k] + a * tok
        self.ms[k] = ms if self.n[k] <= 1 else (1 - self.ALPHA) * self.ms[k] + self.ALPHA * ms
        self.n[k] += 1
        self.left[k] = max(0, self.left[k] - 1)
        for x in self.kinds:
            self.idle[x] = 0 if x == k else self.idle[x] + tok

    def summary(self):
        return ", ".join(f"{k} {self.n[k]} rounds" + (
            f" ({self.tok[k]:.2f} tok / {self.ms[k]:.0f} ms)" if self.ms[k] else "")
            for k in self.kinds)


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
