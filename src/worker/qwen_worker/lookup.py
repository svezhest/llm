"""Prompt lookup decoding (QWEN_LOOKUP): an n-gram draft source for _spec.

Agentic and coding answers often copy spans of their context: a file
re-emitted with one edit, the previous answer's code, a quoted tool output.
The draft for a round is then the continuation of the latest earlier
occurrence of the text's current suffix — free to compute (a dict lookup, no
MLX op) and, inside a copy, right for long stretches. The target verifies
every draft as it verifies the drafter's (_spec): the output is whatever the
verify path gives, never the lookup's.

_Lookup keeps the token sequence (prompt + every emitted token) and a map
n_min-gram -> the positions that FOLLOWED it (the continuation starts). An
n-gram enters the map only once its next token is known (`add`), so the
query never finds the current suffix itself. `propose` takes the latest few
occurrences of the last n_min tokens, extends each match backwards (up to
`n_max` tokens, the score) and returns the longest one's continuation (the
latest among equals: the most recent version of a re-emitted text)."""


class _Lookup:
    """Suffix-match draft source over prompt + generated tokens (pure Python)."""

    __slots__ = ("toks", "n_min", "n_max", "cand", "idx")

    def __init__(self, ids, n_min=3, n_max=8, cand=8):
        self.toks = []
        self.n_min, self.n_max, self.cand = max(1, n_min), max(n_min, n_max), max(1, cand)
        self.idx = {}
        self.extend(ids)

    def __len__(self):
        return len(self.toks)

    def add(self, t):
        """Append one token (prompt or emitted)."""
        toks, n = self.toks, self.n_min
        toks.append(int(t))
        i = len(toks) - 1                   # the new token's index = the continuation start
        if i >= n:
            key = tuple(toks[i - n:i])
            pos = self.idx.get(key)
            if pos is None:
                self.idx[key] = [i]
            else:
                pos.append(i)

    def extend(self, ids):
        for t in ids:
            self.add(t)

    def propose(self, k):
        """-> (match length, up to k continuation tokens) for the current end;
        (0, []) without a match of n_min tokens."""
        toks, n = self.toks, self.n_min
        L = len(toks)
        if L < n or k <= 0:
            return 0, []
        pos = self.idx.get(tuple(toks[L - n:]))
        if not pos:
            return 0, []
        best, at = 0, None
        for j in reversed(pos[-self.cand:]):      # latest first; ties keep the latest
            m = n
            while m < self.n_max and j - m - 1 >= 0 and toks[j - m - 1] == toks[L - m - 1]:
                m += 1
            if m > best:
                best, at = m, j
                if m >= self.n_max:
                    break
        if at + k <= L:
            return best, toks[at:at + k]
        # the copy runs into the end: it goes on with what it copies (a
        # period of L - at tokens, as an LZ77 match overlapping itself)
        per = L - at
        return best, [toks[at + i % per] for i in range(k)]



class _DraftCtx:
    """The drafter's context for its next block after rounds it did not
    draft (lookup rounds, the thinking budget's plain steps): every position
    the target cache took since its last block, in order. When every drafter
    cache is a rotating one of `win` positions (DFlash2: all layers sliding),
    only the last `win` are kept and the caches' offsets move on by the rest
    at `take` — what the drafter itself does with a longer context
    (DFlashAttention: `cache.offset += skip`); a drafter with a full-attention
    layer (the 9B's) gets them all."""

    __slots__ = ("chunks", "win", "trim", "skip")

    def __init__(self, win, draft_cache):
        self.chunks, self.win, self.skip = [], win, 0
        self.trim = bool(draft_cache) and all(hasattr(c, "max_size") for c in draft_cache)

    def __bool__(self):
        return bool(self.chunks)

    def add(self, h):
        self.chunks.append(h)
        if self.trim:
            while sum(t.shape[1] for t in self.chunks[1:]) >= self.win:
                self.skip += self.chunks.pop(0).shape[1]

    def take(self, draft_cache):
        """-> the context array for draft_block; empties the holder."""
        import mlx.core as mx
        ch = self.chunks
        h = ch[0] if len(ch) == 1 else mx.concatenate(ch, axis=1)
        skip = self.skip
        if self.trim and h.shape[1] > self.win:
            skip += h.shape[1] - self.win
            h = h[:, h.shape[1] - self.win:]
        if skip:
            for c in draft_cache or ():
                c.offset += skip
        self.chunks, self.skip = [], 0
        return h


def _lstat(s):
    """The log's lookup part: [rounds, drafts, accepted, plain steps]."""
    if not any(s):
        return ""
    return (f"; lookup {s[0]} rounds, {s[2]}/{s[1]} drafts accepted"
            + (f", {s[3]} plain steps" if s[3] else ""))
