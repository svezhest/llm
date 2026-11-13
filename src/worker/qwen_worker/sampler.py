"""Sampling in vLLM's order: penalties, temperature, min_p, top_k, top_p, draw."""

import os

import mlx.core as mx

from .config import SAMPLER_TOPK_FAST


def make_sampler(sampling, seed=None):
    """logits (rows, V) -> token ids, with an explicit per-request RNG key.

    vLLM's order: temperature, then min_p, top_k and top_p, each on the
    distribution the previous one left (top_p over the top_k survivors,
    renormalized; at least one token always survives).

    The key is ours, not the global one: mlx_lm's sampler draws through a
    function compiled with ``inputs=outputs=mx.random.state``, which captures
    the global key of the thread that first traced it. MLX random state is per
    thread, so in the gRPC handler threads the captured key never advanced and
    temperature 0.7 returned byte-identical answers. Here the key is split per
    step; ``seed`` (request field) makes a request reproducible.

    The result is a `_Sampler`: called on one row's logits it is the per-row
    sampler; `sample_rows` draws for several rows with equal settings at once.
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
            seed = int.from_bytes(os.urandom(4), "little")
        self.key = mx.random.key(int(seed) & 0x7FFF_FFFF_FFFF_FFFF)
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

    def draw(self, x):
        if isinstance(x, _TopK):
            return x.pick(self.noise(*x.z.shape[:-1], x.width))
        self.key, sub = mx.random.split(self.key)
        return mx.random.categorical(x, key=sub)

    def noise(self, rows, width):
        """The fast path's randomness for `rows` rows: one uniform draw of
        (rows, width) from the next key (_TopK.pick turns it into Gumbels +
        the tie bucket's pick). One split + one random op per row: the rest
        of a group's draw runs once on the stack (sample_rows), since a
        per-row draw of ~25 small ops cost more host time at B = 16 than the
        filter saved on the GPU (GPU phase 2026-10-24)."""
        self.key, sub = mx.random.split(self.key)
        return mx.random.uniform(shape=(rows, width), key=sub)

    def __call__(self, logits):
        if self.greedy:
            return mx.argmax(logits - mx.logsumexp(logits, axis=-1, keepdims=True), axis=-1)
        return self.draw(self.prepare(logits))


class _TopK:
    """_Sampler.survivors of some rows: z (R, k + 1) logits of the k
    candidates `ids` (R, k) + the ties-of-the-k-th bucket, `order` (R, V)
    the sorted vocab, the bucket's tokens = order[start : start + extra].
    sample_rows draws a group at once from the rows' own noise (_Sampler.noise)."""
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


def sample_rows(samplers, logits, on_error=None):
    """Next tokens of several rows -> list of (1,) arrays, each equal to what
    `samplers[i](logits[i])` returns. Rows with the same settings and logits
    dtype are filtered as one (G, V) stack: MLX's row ops (topk, argsort,
    softmax, cumsum, logsumexp, argmax) give each row of a stack bit-for-bit
    the result of that row alone (checked on Metal at the real vocabulary,
    ties included: tests/test_mrqmm.py), and the draw stays per row with the
    row's own key. At B=16 the card sampler (top_k 20, top_p 0.95) costs ~15 ms
    of GPU per step row by row, mostly 16 full-vocab topk + argsort launches.

    An exception while building a group's draw fails that group only: with
    `on_error`, on_error(i, e) is called for each of its rows and their
    entries stay None (the other groups are drawn); without, it is raised."""
    out = [None] * len(samplers)
    groups = {}
    for i, (s, x) in enumerate(zip(samplers, logits)):
        groups.setdefault((s.group, x.dtype), []).append(i)
    for idx in groups.values():
        try:
            ys = _sample_group([samplers[i] for i in idx], [logits[i] for i in idx])
        except Exception as e:  # noqa: BLE001 — this group's rows only
            if on_error is None:
                raise
            for i in idx:
                on_error(i, e)
            continue
        for i, y in zip(idx, ys):
            out[i] = y
    return out


def _sample_group(samplers, logits):
    """sample_rows for rows of one group (equal settings and dtype)."""
    s0 = samplers[0]
    if len(samplers) == 1:
        return [s0(logits[0])]
    stack = mx.concatenate(logits)
    if s0.greedy:
        toks = s0(stack)
        return [toks[j:j + 1] for j in range(len(samplers))]
    x = s0.prepare(stack)
    if isinstance(x, _TopK):            # one draw for the stack, each row's own noise
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
