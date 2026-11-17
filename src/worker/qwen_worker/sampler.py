"""Sampling in vLLM's order: penalties, temperature, min_p, top_k, top_p, draw."""

import os

import mlx.core as mx
import numpy as np

from .config import SAMPLER_TOPK_FAST

# The draw's randomness is a function of (the request's seed, the absolute
# position of the token drawn): `_words` mixes the two into a 64-bit key on
# the host (splitmix64, a bijection of the position for a given seed), and
# the noise of a row is a counter-based hash of (that key, the vocab column)
# computed on the GPU (`_uniform`: one compiled elementwise kernel for all
# rows of a draw, no MLX random state). So a token gets the same noise
# whatever else is drawn with it: a row alone or in a batch, a plain step or
# row i of a speculative verify block (position pos0 + i), before or after a
# prompt-cache hit or a preemption -- llama.cpp's "one draw per emitted
# position". A request without a seed gets a random one when its sampler is
# made (admission: _Scheduler._setup -> make_sampler).
# POS_KEYS False: the stream before this (the request's MLX key split per
# draw; a verify block drawn with one key): an in-process A/B switch for
# benchmarks, not a knob.
POS_KEYS = True

_M64 = (1 << 64) - 1


def _mix64(z):
    """splitmix64's output function (a bijection of 64-bit words)."""
    z = (z + 0x9E3779B97F4A7C15) & _M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return z ^ (z >> 31)


def _words(seed, pos):
    """The key of the draw at absolute position `pos` -> [low, high] uint32."""
    z = _mix64(seed ^ _mix64(pos))
    return [z & 0xFFFFFFFF, z >> 32]


def _mix32(x):
    """lowbias32 (hash-prospector): an avalanching bijection of uint32."""
    x = x ^ (x >> 16)
    x = x * 0x7FEB352D
    x = x ^ (x >> 15)
    x = x * 0x846CA68B
    return x ^ (x >> 16)


def _uniform_rows(keys, cols):
    """keys (R, 2) uint32, cols (W,) uint32 -> (R, W) float32 uniforms in
    (0, 1): (2j + 1) / 2^24 for 23 hash bits j (every value exact in fp32,
    never 0 or 1). Two rounds keyed by both words: rows whose keys differ in
    one word only are not permutations of each other's noise."""
    x = _mix32(cols[None] ^ keys[:, :1])
    x = _mix32(x + keys[:, 1:])
    return ((x >> 9).astype(mx.float32) + 0.5) * (1.0 / (1 << 23))


def _gumbel_rows(x, keys, cols):
    """x + Gumbel noise of the rows' keys (mx.random.categorical's draw is
    the argmax of this, with MLX's own noise)."""
    return x + -mx.log(-mx.log(_uniform_rows(keys, cols)))


# one fused kernel each (traced per shape; integer ops: bit-identical on any
# row count, and no captured random state, so any thread may call them)
_uniform = mx.compile(_uniform_rows)
_gumbel = mx.compile(_gumbel_rows)
_COLS = {}


def _cols(width):
    """0 .. width - 1 as uint32: a constant (numpy-backed, no graph), memoized."""
    c = _COLS.get(width)
    if c is None:
        c = _COLS[width] = mx.array(np.arange(width, dtype=np.uint32))
    return c


def _draw(x, keys):
    """Tokens (R,) of prepared rows `x` (filtered logits or a _TopK) from the
    rows' keys (R, 2): Gumbel-max over the full vocabulary, or the _TopK's
    own pick from k + 2 uniforms."""
    if isinstance(x, _TopK):
        return x.pick(_uniform(keys, _cols(x.width)))
    return mx.argmax(_gumbel(x, keys, _cols(x.shape[-1])), axis=-1)


def make_sampler(sampling, seed=None):
    """logits (rows, V) -> token ids, drawn with noise keyed by (seed, position).

    vLLM's order: temperature, then min_p, top_k and top_p, each on the
    distribution the previous one left (top_p over the top_k survivors,
    renormalized; at least one token always survives).

    The randomness is ours, not MLX's global state: mlx_lm's sampler draws
    through a function compiled with ``inputs=outputs=mx.random.state``,
    which captures the global key of the thread that first traced it (MLX
    random state is per thread: in the gRPC handler threads that key never
    advanced and temperature 0.7 returned byte-identical answers). Here every
    draw hashes (``seed``, the token's absolute position) -- see POS_KEYS --
    so nothing is carried from draw to draw; ``seed`` (request field) makes a
    request reproducible, None picks a random one.

    The result is a `_Sampler`: called on rows' logits (+ the position of
    the first row; the rows are consecutive positions) it is the per-request
    sampler; `sample_rows` draws for several requests with equal settings
    at once.
    """
    return _Sampler(sampling, seed)


class _Sampler:
    """See make_sampler. `group`: rows whose samplers have equal groups can be
    filtered together (`sample_rows`); None for nothing to share."""

    def __init__(self, sampling, seed=None):
        temp = sampling["temperature"]
        self.greedy = temp <= 0             # vLLM: 0 = greedy, (0, 0.01) is raised to 0.01
        if self.greedy:
            self.group = ("greedy",)
            return
        self.top_p, self.min_p, self.top_k = sampling["top_p"], sampling["min_p"], sampling["top_k"]
        if seed is None:
            seed = int.from_bytes(os.urandom(8), "little")
        self.seed = int(seed) & _M64
        self.key = mx.random.key(int(seed) & 0x7FFF_FFFF_FFFF_FFFF)    # POS_KEYS off only
        self.next = 0                       # the position a draw without one takes (tools, tests)
        self.temp = max(temp, 0.01)
        self.group = (self.temp, self.min_p, self.top_k, self.top_p)
        self.fast = SAMPLER_TOPK_FAST and self.top_k > 0

    def filtered(self, logits):
        """temperature, min_p, top_k, top_p on float32 logits (rows, V): row
        ops only, so a stack of rows gives each row what it alone would."""
        ninf = mx.array(-float("inf"), dtype=mx.float32)
        x = logits.astype(mx.float32) / self.temp
        if self.min_p > 0:
            p = mx.softmax(x, axis=-1)
            x = mx.where(p < p.max(axis=-1, keepdims=True) * self.min_p, ninf, x)
        if 0 < self.top_k < x.shape[-1]:
            kth = mx.topk(x, self.top_k, axis=-1).min(axis=-1, keepdims=True)
            x = mx.where(x < kth, ninf, x)
        if self.top_p < 1.0:
            idx = mx.argsort(x, axis=-1)
            xs = mx.take_along_axis(x, idx, axis=-1)
            drop = mx.cumsum(mx.softmax(xs, axis=-1), axis=-1) <= 1 - self.top_p
            drop[..., -1] = False
            x = mx.put_along_axis(x, idx, mx.where(drop, ninf, xs), axis=-1)
        return x

    def prepare(self, logits):
        """What `draw` takes for these rows: the filtered full-vocab logits,
        or (QWEN_SAMPLER_TOPK_FAST, 0 < top_k < V) the top-k survivors."""
        if self.fast and self.top_k < logits.shape[-1]:
            return self.survivors(logits)
        return self.filtered(logits)

    def survivors(self, logits):
        """filtered() over the top_k candidates only (QWEN_SAMPLER_TOPK_FAST):
        one stable argsort of the vocab instead of topk + argsort + scatter,
        and min_p / top_p / the draw over k values instead of V.

        The same distribution, in exact arithmetic: min_p and top_k both keep
        an upper set of the values (min_p: exp(x - max) >= min_p, and the max
        is always in the top k), so min_p -> top_k = their intersection, all
        inside the k largest plus the values tied with the k-th; top_p then
        drops a prefix of the survivors in ascending stable order (ties by
        vocab index, as filtered()'s argsort: MLX's sort is stable on CPU and
        Metal), so its cumulative mass needs only the survivors. The ties of
        the k-th value outside the k (filtered() keeps every one) are the
        positions [V - n, V - k) of the sorted order (n = #x >= kth): they come
        first in the ascending cumsum, top_p drops the lowest-index ones
        first (floor(thr Z / w_kth) of them), the rest is one extra candidate
        of logit kth + log(count) whose draw picks one of them uniformly.
        Only the rounding differs from filtered(): the softmax / cumsum run
        over k terms instead of V (equal nonzero terms, other summation
        order), so a token exactly at the min_p / top_p boundary (~1e-7
        relative) can land on the other side. The random stream differs (a
        draw over k + 1 values and a uniform, not V Gumbels): hence a knob."""
        x = logits.astype(mx.float32) / self.temp
        V, k = x.shape[-1], self.top_k
        ninf = mx.array(-float("inf"), dtype=mx.float32)
        order = mx.argsort(x, axis=-1)
        ids = order[..., V - k:]
        xs = mx.take_along_axis(x, ids, axis=-1)          # ascending; xs[..., -1] = max
        kth = xs[..., :1]
        w = mx.exp(xs - xs[..., -1:])                      # unnormalized probabilities
        n = (x >= kth).sum(axis=-1, keepdims=True).astype(mx.float32)
        # kth = -inf: fewer than k finite logits, the "ties" have no mass
        extra = mx.where(kth > ninf, n - k, 0.0)
        start = V - n
        keep = None
        if self.min_p > 0:
            keep = w >= self.min_p
            extra = mx.where(keep[..., :1], extra, 0.0)
        if self.top_p < 1.0:
            wk = w[..., :1]
            wkeep = w if keep is None else mx.where(keep, w, 0.0)
            base = extra * wk
            thr = (1 - self.top_p) * (base + wkeep.sum(axis=-1, keepdims=True))
            # the extras' ascending cumsum is j * wk: those with j * wk <= thr go
            # (all of them when wk underflowed to 0: no mass)
            gone = mx.where(wk > 0, mx.clip(mx.floor(thr / mx.maximum(wk, 1e-30)), 0.0, extra), extra)
            drop = base + mx.cumsum(wkeep, axis=-1) <= thr
            drop[..., -1] = False
            keep = ~drop if keep is None else keep & ~drop
            extra, start = extra - gone, start + gone
        z = xs if keep is None else mx.where(keep, xs, ninf)
        z = mx.concatenate([z, mx.where(extra > 0, kth + mx.log(mx.maximum(extra, 1.0)), ninf)], axis=-1)
        return _TopK(z, ids, order, start, extra)

    def at(self, pos, rows):
        """The keys (host words) of `rows` consecutive draws from absolute
        position `pos` (None: where the last draw ended)."""
        if pos is None:
            pos = self.next
        self.next = pos + rows
        return [_words(self.seed, p) for p in range(pos, pos + rows)]

    def draw(self, x, pos=None):
        """Tokens of prepared rows `x`, row i being the token at position
        pos + i. One host array of the rows' keys and one fused noise kernel
        per draw (_uniform / _gumbel), whatever the row count."""
        rows = (x.z if isinstance(x, _TopK) else x).shape[0]
        if not POS_KEYS:
            if isinstance(x, _TopK):
                return x.pick(self.noise(rows, x.width))
            self.key, sub = mx.random.split(self.key)
            return mx.random.categorical(x, key=sub)
        return _draw(x, mx.array(self.at(pos, rows), dtype=mx.uint32))

    def noise(self, rows, width):
        """POS_KEYS off: the fast path's randomness for `rows` rows, one
        uniform draw of (rows, width) from the next split of the key."""
        self.key, sub = mx.random.split(self.key)
        return mx.random.uniform(shape=(rows, width), key=sub)

    def __call__(self, logits, pos=None):
        """Tokens of `logits` (rows, V); row i is the token at absolute
        position pos + i (the KV length after the forward that gave its
        logits). Greedy ignores `pos`."""
        if self.greedy:
            return mx.argmax(logits - mx.logsumexp(logits, axis=-1, keepdims=True), axis=-1)
        return self.draw(self.prepare(logits), pos)


class _TopK:
    """_Sampler.survivors of some rows: z (R, k + 1) logits of the k
    candidates `ids` (R, k) + the ties-of-the-k-th bucket, `order` (R, V)
    the sorted vocab, the bucket's tokens = order[start : start + extra].
    sample_rows draws a group at once from the rows' own noise (_draw)."""
    __slots__ = ("z", "ids", "order", "start", "extra")

    def __init__(self, z, ids, order, start, extra):
        self.z, self.ids, self.order, self.start, self.extra = z, ids, order, start, extra

    @property
    def width(self):
        return self.z.shape[-1] + 1          # k + 1 Gumbels + the bucket's uniform

    def pick(self, u):
        """Tokens (R,) uint32 from uniforms u (R, width): Gumbel-max over z
        (as mx.random.categorical), the last column picks inside the bucket."""
        k = self.ids.shape[-1]
        g = -mx.log(-mx.log(mx.maximum(u[..., :k + 1], 1e-30)))
        t = mx.argmax(self.z + g, axis=-1)
        tok = mx.take_along_axis(self.ids, mx.minimum(t, k - 1)[..., None], axis=-1)[..., 0]
        u = mx.floor(u[..., k + 1:] * self.extra)
        pos = mx.clip(self.start + mx.minimum(u, self.extra - 1), 0, self.order.shape[-1] - 1)
        tie = mx.take_along_axis(self.order, pos.astype(mx.int32), axis=-1)[..., 0]
        return mx.where(t == k, tie, tok)


def sample_rows(samplers, logits, pos=None, on_error=None):
    """Next tokens of several rows -> list of (1,) arrays, each equal to what
    `samplers[i](logits[i], pos[i])` returns (`pos` None: each sampler's
    next). Rows with the same settings and logits dtype are filtered as one
    (G, V) stack: MLX's row ops (topk, argsort, softmax, cumsum, logsumexp,
    argmax) give each row of a stack bit-for-bit the result of that row alone
    (checked on Metal at the real vocabulary, ties included:
    tests/test_mrqmm.py), and each row's noise is a function of its own key
    only, so the group is drawn by one noise kernel + one argmax. At B=16 the
    card sampler (top_k 20, top_p 0.95) costs ~15 ms of GPU per step row by
    row, mostly 16 full-vocab topk + argsort launches.

    An exception while building a group's draw fails that group only: with
    `on_error`, on_error(i, e) is called for each of its rows and their
    entries stay None (the other groups are drawn); without, it is raised."""
    out = [None] * len(samplers)
    if pos is None:
        pos = [None] * len(samplers)
    groups = {}
    for i, (s, x) in enumerate(zip(samplers, logits)):
        groups.setdefault((s.group, x.dtype), []).append(i)
    for idx in groups.values():
        try:
            ys = _sample_group([samplers[i] for i in idx], [logits[i] for i in idx],
                               [pos[i] for i in idx])
        except Exception as e:  # noqa: BLE001 — this group's rows only
            if on_error is None:
                raise
            for i in idx:
                on_error(i, e)
            continue
        for i, y in zip(idx, ys):
            out[i] = y
    return out


def _sample_group(samplers, logits, pos):
    """sample_rows for rows of one group (equal settings and dtype)."""
    s0 = samplers[0]
    if len(samplers) == 1:
        return [s0(logits[0], pos[0])]
    stack = mx.concatenate(logits)
    if s0.greedy:
        toks = s0(stack)
        return [toks[j:j + 1] for j in range(len(samplers))]
    x = s0.prepare(stack)
    if POS_KEYS:                        # one draw for the stack, each row's own key
        keys = mx.array([w for s, p in zip(samplers, pos) for w in s.at(p, 1)], dtype=mx.uint32)
        toks = _draw(x, keys)
        return [toks[j:j + 1] for j in range(len(samplers))]
    if isinstance(x, _TopK):            # (POS_KEYS off: each row's own split)
        toks = x.pick(mx.concatenate([s.noise(1, x.width) for s in samplers]))
        return [toks[j:j + 1] for j in range(len(samplers))]
    return [s.draw(x[j:j + 1]) for j, s in enumerate(samplers)]


class _Penalties:
    """vLLM's (= OpenAI's) penalties on the raw logits: presence and frequency
    are additive over the tokens generated so far; repetition is the
    multiplicative HF/vLLM one (positive logits / p, negative * p) over every
    token seen in the prompt or the output. `extra` adds per-row counts (the
    drafts before each verify row in speculative decoding)."""

    def __init__(self, sampling, prompt_ids, vocab):
        self.presence = sampling["presence_penalty"]
        self.frequency = sampling["frequency_penalty"]
        self.repetition = sampling["repetition_penalty"]
        self.active = bool(self.presence or self.frequency or self.repetition != 1.0)
        self.counts = mx.zeros((vocab,), dtype=mx.float32)
        self.prompt = None
        if self.repetition != 1.0:
            self.prompt = mx.zeros((vocab,), dtype=mx.bool_)
            self.prompt[mx.array(prompt_ids)] = True

    def add(self, toks):
        self.counts = self.counts.at[toks.reshape(-1)].add(1.0)

    def __call__(self, logits, extra=None):
        x = logits.astype(mx.float32)
        c = self.counts if extra is None else self.counts + extra
        if self.repetition != 1.0:
            seen = (c > 0) | self.prompt
            x = mx.where(seen, mx.where(x > 0, x / self.repetition, x * self.repetition), x)
        if self.frequency:
            x = x - self.frequency * c
        if self.presence:
            x = x - self.presence * (c > 0)
        return x
