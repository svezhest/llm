"""Prompt cache: cache-array helpers, KV segments, entries, the prefix cache."""

import time
import typing

import mlx.core as mx

from .config import GIB, KV_KWARGS, log, PROMPT_CACHE_CKPTS, PROMPT_CACHE_GB, PROMPT_CACHE_STALE_S


def _copy_cache(caches, upto=None, states=None):
    """Fresh cache objects holding the same (immutable) arrays.

    `upto` cuts the KV to that many tokens and `states` (a recurrent-state
    checkpoint taken at the same position) replaces the ArraysCache contents.

    MLX `a[idx] = v` rebinds the Python array object in place, so two cache
    objects must not share array objects — slicing gives new ones over the same
    buffer (copy-on-write from there). ArraysCache entries are replaced whole
    by the model, so a new list is enough."""
    out = []
    states = iter(states) if states is not None else None
    for c in caches:
        if hasattr(c, "cache"):                       # ArraysCache (recurrent state)
            n = type(c)(size=len(c.cache))
            n.cache = list(next(states)) if states is not None else list(c.cache)
            out.append(n)
            continue
        end = c.offset if upto is None else upto
        if isinstance(c.keys, (tuple, list)):         # QuantizedKVCache (list after its tree_map resize)
            n = type(c)(group_size=c.group_size, bits=c.bits)
            n.keys = type(c.keys)(k[..., :end, :] for k in c.keys)
            n.values = type(c.values)(v[..., :end, :] for v in c.values)
        else:                                         # KVCache
            n = type(c)()
            n.keys = c.keys[..., :end, :]
            n.values = c.values[..., :end, :]
        n.offset = end
        out.append(n)
    return out


def _recurrent_states(caches):
    return [list(c.cache) for c in caches if hasattr(c, "cache")]


def _own_states(caches):
    """After a DFlash round: every recurrent layer's state in a buffer of its
    own (a gather; queued, not waited for). A block of T tokens that is not
    wholly accepted leaves each layer's state as a slice of the block's
    per-token states (mlx-vlm `_select_speculative_states`:
    `intermediate[:, keep - 1]`), which keeps all T - 1 of them alive until the
    next partly accepted block — on the 27B at block 8, 7 x 147 MiB = 1.0 GiB
    held between rounds on top of the next block's own T states. Same values."""
    idx, out = mx.array([0], dtype=mx.int32), []
    for c in caches:
        if hasattr(c, "cache"):
            for i, x in enumerate(c.cache):
                if x is not None:
                    c.cache[i] = x[idx]
                    out.append(c.cache[i])
    mx.async_eval(out)


def _with_tail(states, tail):
    """A recurrent state with the drafter's context at its position
    (QWEN_DRAFT_TAIL / _END: `tail`, (1, n, layers x hidden), the target's hidden
    states of the n positions before it) as one more item after the recurrent
    layers. It rides wherever the state goes (inherited, absorbed, thinned,
    counted in the entry's bytes, written to the disk tier); restore reads
    the recurrent layers only."""
    return states + [[tail]] if tail is not None else states


def _tail_copy(chunks, n, whole=None):
    """The last `n` positions of the (1, t, D) `chunks` (in order) as one
    array in a buffer of its own: a slice would keep its chunk's whole buffer
    alive in the cache entry. `whole`: a chunk that already is such an array
    (returned as it is when it is all there is)."""
    if n <= 0 or not chunks:
        return None
    if len(chunks) == 1 and chunks[0] is whole and whole.shape[1] <= n:
        return whole
    parts, k = [], 0
    for t in reversed(chunks):
        m = min(t.shape[1], n - k)
        parts.insert(0, t[:, t.shape[1] - m:])
        k += m
        if k >= n:
            break
    # (a concatenate of several parts writes a new buffer)
    t = mx.concatenate(parts, axis=1) if len(parts) > 1 else _own_slice(parts[0], 0, k)
    mx.eval(t)
    return t


def _state_tail(states, layout):
    """The drafter's context stored with a state (_with_tail), or None."""
    n = sum(1 for lay in layout if lay[0] == "arr")
    return states[n][0] if states is not None and len(states) > n else None


def _resume_tail(entry, at):
    """The drafter's context for a DFlash row resuming at `at` of `entry`
    (None: no entry): what the state there carries, or, when that is more,
    the part before `at` of the entry's END context (QWEN_DRAFT_TAIL_END: the
    positions [L - n, L); hidden states are causal and the entry's slabs up to
    `at` are the new prompt's, so its positions before `at` are what a cold
    run of the new prompt computes there). So the checkpoints inside the end
    state's window, the last message starts (where the next turn resumes when
    the reply is not stored: thinking on), start the drafter with nearly the
    cold run's context; a checkpoint's own shorter one (QWEN_DRAFT_TAIL) is
    for those further back. An array of its own (a slice would keep the
    entry's buffer alive in the row's next entry)."""
    if entry is None:
        return None
    L = len(entry.ids)
    tail = _state_tail(entry.end if at == L else entry.ckpts[at], entry.layout)
    if at < L:
        end = _state_tail(entry.end, entry.layout)
        k = end.shape[1] - (L - at) if end is not None else 0
        if k > (tail.shape[1] if tail is not None else 0):
            tail = _tail_copy([end[:, :k]], k)
    return tail


def _cache_arrays(caches):
    arrs = []
    for c in caches:
        if hasattr(c, "cache"):
            arrs += [a for a in c.cache if a is not None]
        else:
            for x in (c.keys, c.values):
                arrs += list(x) if isinstance(x, (tuple, list)) else [x]
    return arrs


def _lcp(a, b):
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n                                     # a[:lo] == b[:lo], a[:hi] != b[:hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid
    return lo


def _quantized(cache):
    return any(isinstance(getattr(c, "keys", None), (tuple, list)) for c in cache)


def _state_arrays(ckpts):
    return [a for st in ckpts.values() for layer in st for a in layer if a is not None]


def _thin(ckpts, cap, protect=()):
    """Drop checkpoints until `cap` are left: the left one of the closest
    pair (span to its neighbours), never the deepest, and among the ones not
    in `protect` (the end of the system block, the last message starts) while
    there are any."""
    while len(ckpts) > cap:
        ps = sorted(ckpts)
        cand = [i for i in range(len(ps) - 1) if ps[i] not in protect] or range(len(ps) - 1)
        if not cand:
            break
        i = min(cand, key=lambda i: (ps[i + 1] - (ps[i - 1] if i else 0), ps[i]))
        ckpts.pop(ps[i])
    return ckpts


def _own_slice(x, a, b):
    """x[..., a:b, :] in a buffer of its own, never a view of x's. A
    concatenate of two non-empty inputs always writes a new buffer
    (mx.contiguous of a slice spanning the whole axis hands back x's own
    buffer); a 1-token slice keeps a 2-token buffer of its own."""
    if b - a >= 2:
        m = (a + b) // 2
        return mx.concatenate([x[..., a:m, :], x[..., m:b, :]], axis=-2)
    s = x[..., a:b, :]
    return mx.concatenate([s, s], axis=-2)[..., :1, :]


class _Seg:
    """KV of tokens [a, b) of every full-attention layer, in buffers of its
    own (copied out of a row's cache, never a view of one). kv[j] = (k, v) of
    the j-th full-attention layer in cache order: (1, Hkv, b-a, D) arrays, or
    for quantized KV (packed, scales, biases) tuples. Immutable; shared by
    every entry whose chain covers it, freed when the last one goes (plain
    object references)."""
    __slots__ = ("a", "b", "kv", "arrays", "nbytes", "__weakref__")

    def __init__(self, a, b, kv):
        self.a, self.b, self.kv = a, b, kv
        self.arrays = [y for kv_ in kv for x in kv_ for y in (x if isinstance(x, tuple) else (x,))]
        self.nbytes = sum(x.nbytes for x in self.arrays)

    @classmethod
    def copy_from(cls, cache, a, b, sync=True):
        """`sync` False: the copies stay lazy (a reply store between two
        decode steps: the caller queues them with mx.async_eval, the host
        never waits)."""
        def own(x):
            return tuple(_own_slice(y, a, b) for y in x) if isinstance(x, (tuple, list)) \
                else _own_slice(x, a, b)
        seg = cls(a, b, [(own(c.keys), own(c.values)) for c in cache if not hasattr(c, "cache")])
        if sync:
            mx.eval(seg.arrays)
        return seg

    @classmethod
    def adopt(cls, cache, b, a=0):
        """[a, b) of a DYING row's caches without a copy (QWEN_PREEMPT,
        QWEN_PROMPT_CACHE_MOVE): the segment's arrays are views of the row's
        own buffers, which nothing writes again (the row is dropped right
        after; its next decode never runs). The buffers keep their capacity
        tail (< one 256-token KVCache step) that `nbytes` does not count."""
        def view(x):
            return tuple(y[..., a:b, :] for y in x) if isinstance(x, (tuple, list)) else x[..., a:b, :]
        return cls(a, b, [(view(c.keys), view(c.values)) for c in cache if not hasattr(c, "cache")])

    @classmethod
    def take(cls, cache, b, a=0):
        """[a, b) of a DYING row's caches with one copy of the KV alive at
        any time (QWEN_PROMPT_CACHE_MOVE). The row's buffers as they are
        (views, as `adopt`) when the segment is all they hold: a = 0 and less
        than a KVCache step past b. Else (an entry that ends before the
        buffers do: the prompt of a reply that is not stored; a = the part a
        sibling's entry already holds) each layer is copied out, evaluated
        and the row's buffer released at once (its cache object is emptied:
        the row is dead) — the transient is one layer's KV, and the entry
        pins nothing it does not count."""
        def own(x):
            return tuple(_own_slice(y, a, b) for y in x) if isinstance(x, (tuple, list)) \
                else _own_slice(x, a, b)

        def view(x):
            return tuple(y[..., :b, :] for y in x) if isinstance(x, (tuple, list)) else x[..., :b, :]
        kv = []
        for c in cache:
            if hasattr(c, "cache"):
                continue
            k0 = c.keys[0] if isinstance(c.keys, (tuple, list)) else c.keys
            if a == 0 and k0.shape[-2] - b < getattr(type(c), "step", 256):
                kv.append((view(c.keys), view(c.values)))
                continue
            k, v = own(c.keys), own(c.values)
            mx.eval(k, v)
            c.keys = c.values = None            # the old buffer goes before the next layer's copy
            kv.append((k, v))
        return cls(a, b, kv)

    def sub(self, a, b):
        """[a, b) of this segment as a segment of views of its buffers (no
        copy): a prompt entry and its reply over one taken buffer."""
        if (a, b) == (self.a, self.b):
            return self
        lo, hi = a - self.a, b - self.a

        def view(x):
            return tuple(y[..., lo:hi, :] for y in x) if isinstance(x, (tuple, list)) \
                else x[..., lo:hi, :]
        return _Seg(a, b, [(view(k), view(v)) for k, v in self.kv])

    @staticmethod
    def estimate(cache, a, b):
        """The bytes copy_from(cache, a, b) will hold, from the row's shapes
        alone (nothing is allocated): every buffer is uniform per token."""
        n = 0
        for c in cache:
            if hasattr(c, "cache"):
                continue
            for x in (c.keys, c.values):
                for y in (x if isinstance(x, (tuple, list)) else (x,)):
                    n += y.nbytes // y.shape[-2] * (b - a)
        return n


def _layout(cache):
    """What restore needs to rebuild each layer's cache object."""
    out = []
    for c in cache:
        if hasattr(c, "cache"):
            out.append(("arr", type(c), len(c.cache)))
        elif isinstance(c.keys, (tuple, list)):
            out.append(("q", type(c), c.group_size, c.bits, type(c.keys)))
        else:
            out.append(("kv", type(c)))
    return out


class _Inherit(typing.NamedTuple):
    """What a resumed prefill needs of its parent entry at store time (the
    snapshot closure holds this, not the entry)."""
    pieces: list                    # the parent's chain cut at the resume point
    ckpts: dict                     # the parent's checkpoints <= the resume point
    end: list | None                # its end state when it resumed at its end
    quantized: bool


class _Entry:
    """The cache state after the prefill of `ids` (= prompt[:-1]): the KV as
    a chain of pieces (seg, lo, hi) (absolute positions, [lo, hi) within
    [seg.a, seg.b), contiguous from 0 to len(ids)), recurrent-state
    checkpoints and the end state. Pieces and checkpoints are shared objects
    between entries. A state stored by a DFlash row has one more item after
    its recurrent layers: the drafter's context there (_with_tail). `reply`
    (QWEN_PROMPT_CACHE_REPLY): the entry was
    extended past its prompt by a request's decoded tokens; `reply` is where
    they start (the prompt entry's length) and the end is resumable although
    it is no slab end of the next prompt. `partial`: stored for a cancelled
    (or failed) prefill's prefix (absorbing it alone makes no chain);
    `chained`: it absorbed an older turn (or a partial entry that had)."""

    def __init__(self, ids, chain, ckpts, end, layout, quantized, low=0, reply=None,
                 partial=False):
        self.ids = ids
        self.chain = chain
        self.ckpts = ckpts                            # pos -> recurrent states, pos < len(ids)
        self.end = end                                # recurrent states at len(ids), or None
        self.layout = layout
        self.quantized = quantized
        self.low = low                                # resumable from here on (quantized KV)
        self.reply = reply                            # decoded tokens from here on (reply KV)
        self.partial = partial                        # a cancelled prefill's prefix
        self.diverge = frozenset()                    # partial: its prompt's divergence points
        self.chained = False                          # absorbed an older turn (store)
        self.used = time.time()
        self.seq = 0                                  # _PrefixCache.seq when stored / last hit
        self.index()

    def index(self):
        """(Re)build `arrays` / `nbytes` (store attaches the new segment after
        deciding the entry is kept)."""
        segs = {id(s): s for s, _, _ in self.chain}
        arrs = [a for s in segs.values() for a in s.arrays]
        arrs += [a for layer in (self.end or []) for a in layer if a is not None]
        arrs += _state_arrays(self.ckpts)
        # every array once: checkpoint states are shared objects between entries
        self.arrays = list({id(a): a for a in arrs}.values())
        self.nbytes = sum(a.nbytes for a in self.arrays)

    def state_bytes(self):
        """Bytes of its recurrent states: checkpoints + the end state (shared
        arrays counted once)."""
        arrs = _state_arrays(self.ckpts) + [a for layer in (self.end or []) for a in layer
                                            if a is not None]
        return sum(a.nbytes for a in {id(a): a for a in arrs}.values())

    def cut(self, at):
        """The pieces covering [0, at)."""
        return [(s, lo, min(hi, at)) for s, lo, hi in self.chain if lo < at]

    def best(self, ids, ends):
        """Deepest resumable position within the common prefix (one token is
        always left to feed). Only a slab end of the new prompt (`ends`) will
        do: resuming anywhere else would cut the rest of the prefill
        differently from a cold run. A quantized KV cut below the
        quantization start would not match a cold run either. The end of a
        reply entry is the exception: its KV came from decode anyway (not a
        cold run's bits whatever the cut), and the next turn's prefill runs
        from it to the new prompt's next slab end."""
        common = min(_lcp(self.ids, ids), len(ids) - 1)
        L = len(self.ids)
        tops = (*self.ckpts, L) if self.end is not None else tuple(self.ckpts)
        pos = [p for p in tops if self.low <= p <= common
               and (p in ends or (p == L and self.reply is not None))]
        return max(pos, default=0), common


class _PrefixCache:
    """The entries (evicted in victim's order). Only the scheduler thread changes it; Health reads
    the counters (plain ints, kept current by every change)."""

    def __init__(self):
        self.entries: list[_Entry] = []
        self.bytes = 0                  # live bytes: every buffer once (segments, states are shared)
        self.hits = self.misses = self.evictions = 0
        self.seq = 0                    # lookups + stores so far (the eviction order's round clock)
        self.version = 0                # bumped when entries are added (peek memos)
        self.disk = None                # pcdisk.DiskTier (QWEN_PROMPT_CACHE_DISK): evictions go there

    def bytes_of(self, entries):
        """Unique bytes of a subset of the entries."""
        seen, n = set(), 0
        for e in entries:
            for a in e.arrays:
                if id(a) not in seen:
                    seen.add(id(a))
                    n += a.nbytes
        return n

    def _recount(self, added=True):
        """Live bytes after a change of `entries`; `added`: entries may have
        come in (an eviction only removes: a peek that did not land on the
        removed entry stays right)."""
        self.bytes = self.bytes_of(self.entries)
        if added:
            self.version += 1

    def charged(self):
        """What the cache costs the batch budget: its live bytes + the RAM
        its disk tier still holds (evicted entries until written, loads until
        inserted: pcdisk.DiskTier.held_bytes). Every budget check takes this,
        the soft cap QWEN_PROMPT_CACHE_GB the live bytes alone."""
        return self.bytes + (self.disk.held_bytes() if self.disk is not None else 0)

    def _scan(self, ids, ends):
        hit, at, split = None, 0, 0
        for e in self.entries:
            pos, common = e.best(ids, ends)
            split = max(split, common)
            if pos > at:
                hit, at = e, pos
        return hit, at, split

    def peek(self, ids, ends):
        """-> (the entry lookup would resume from, its position); no side
        effects (no used / seq touch, no clock tick, no counters): for admission."""
        return self._scan(ids, ends)[:2]

    def lookup(self, ids, ends):
        """-> (entry, resume position, longest common prefix over all entries)."""
        self.seq += 1                                 # a turn = a lookup + a store: 2 ticks
        hit, at, split = self._scan(ids, ends)
        if hit is not None:
            hit.used = time.time()
            hit.seq = self.seq
            self.hits += 1
        else:
            self.misses += 1
        return hit, at, split

    def restore(self, entry, at, first=1, move=False):
        """Caches for a row resuming at `at`. Each full-attention layer: one
        concatenate of the entry's pieces over [0, at) and a zeros tail of
        round_up(first, step) tokens, `first` being the first update's token
        count (the first slab, or 1 when no slab runs). That is the buffer
        the first update_and_fetch built from a [:at] view before (same
        shape, same bits), so it writes in place, and the row never holds a
        view of a segment. Recurrent layers: the end state at len(ids), else
        the checkpoint at `at` (fresh lists, shared arrays: the model
        replaces them whole).

        `move` (QWEN_PROMPT_CACHE_MOVE): the entry's KV becomes the row's
        instead of being copied beside it. The entry leaves the cache and
        keeps its ids and states only (a husk: what the row's store inherits
        when it ends, _Scheduler._settle); each layer is built, evaluated,
        and the entry's buffers of that layer released before the next one
        (the segments no other entry's chain holds; a shared one stays with
        its other holders) — the transient is one layer's KV, never a second
        copy of it all. Same buffers, same bits as without."""
        pieces = entry.cut(at)
        states = iter(entry.end if at == len(entry.ids) else entry.ckpts[at])
        out, j = [], 0
        own = []
        if move:
            self.entries = [e for e in self.entries if e is not entry]
            held = {id(s) for e in self.entries for s, _, _ in e.chain}
            own = [s for s in {id(s): s for s, _, _ in entry.chain}.values() if id(s) not in held]
            entry.chain = []
            entry.index()                             # its states only from here on
            for s in own:                             # (kv holds the buffers until its layer is built)
                s.arrays, s.nbytes = [], 0
            self._recount(added=False)

        def build(parts, tail):
            x = parts[0]
            pad = mx.zeros((*x.shape[:-2], tail, x.shape[-1]), dtype=x.dtype)
            return mx.concatenate(parts + [pad], axis=-2)
        for lay in entry.layout:
            kind, cls = lay[0], lay[1]
            if kind == "arr":
                n = cls(size=lay[2])
                n.cache = list(next(states))
                out.append(n)
                continue
            tail = (cls.step + first - 1) // cls.step * cls.step
            sl = [(s.kv[j], lo - s.a, hi - s.a) for s, lo, hi in pieces]
            if kind == "q":
                n = cls(group_size=lay[2], bits=lay[3])
                parts = len(sl[0][0][0])                # (packed, scales, biases)
                k = [build([x[0][i][..., a:b, :] for x, a, b in sl], tail) for i in range(parts)]
                v = [build([x[1][i][..., a:b, :] for x, a, b in sl], tail) for i in range(parts)]
                n.keys, n.values = lay[4](k), lay[4](v)
            else:
                n = cls()
                n.keys = build([x[0][..., a:b, :] for x, a, b in sl], tail)
                n.values = build([x[1][..., a:b, :] for x, a, b in sl], tail)
            n.offset = at
            out.append(n)
            if move:
                mx.eval(_cache_arrays([n]))
                sl = None
                for s in own:
                    s.kv[j] = None
            j += 1
        mx.eval(_cache_arrays(out))
        return out

    def unmove(self, entry, seg, limit=None, wanted=()):
        """Put a moved entry (restore move=True: a husk) back as it was, its
        KV now `seg` = [0, len) of the row's caches (the row prefilled
        nothing past it and ends: a cancel before its first slab; taken out
        of the buffers restore pre-sized for the first slab) -> False
        when it was not kept (`insert`'s rules)."""
        entry.chain = [(seg, 0, len(entry.ids))]
        entry.index()
        if self.insert(entry, limit=limit, wanted=wanted):
            return True
        entry.chain = []
        entry.index()
        return False

    def victim(self, left, wanted=()):
        """The entry of `left` to evict first. An entry a waiting request
        would resume from (`wanted`) goes last. Then, the stale entries
        (`stale`: a finished conversation / episode, an idle cache) go first,
        least recently used first. When every entry was used within the
        round (N live conversations or agent episodes taking turns), the
        most recently used goes: its next turn is the furthest away. LRU
        there evicts exactly the one whose turn is next, and its re-prefill
        stores a new entry that evicts the next one in line (every turn a
        miss: the round-robin worst case).
        With the disk tier, the wanted entries go in reverse queue order
        (`wanted` is in queue order: _Scheduler._wanted): the one whose
        request comes last first. MRU there evicted the entry loaded back
        for a waiting request (used = its insert) before that request's turn,
        and it went cold anyway (GPU gate 2026-10-24: 20 of 57 loads)."""
        now = time.time()
        order = None
        if self.disk is not None and wanted:
            order = {}
            for i, x in enumerate(wanted):
                order.setdefault(id(x), i)

        def key(e):
            w = any(e is x for x in wanted)
            if w and order is not None:
                return (True, -order.get(id(e), 0))   # the furthest back first
            if self.stale(e, now, w):
                return (w, 0, e.used)                 # stale: LRU
            return (w, 1, -e.used)                    # live: MRU
        return min(left, key=key)

    def stale(self, e, now, wanted=False):
        """Not used within the last round: more than 2 x len(entries) clock
        ticks since it was stored or hit (a text request ticks at its lookup
        and at its store; traffic that touches no entry, vision, does not
        tick: it would age live episodes), or, unless a waiting request
        would resume from it, unused for QWEN_PROMPT_CACHE_STALE_S (an idle
        cache; a request queued that long still wants its entry)."""
        return (self.seq - e.seq > 2 * len(self.entries)
                or (not wanted and PROMPT_CACHE_STALE_S > 0
                    and now - e.used > PROMPT_CACHE_STALE_S))

    def evict(self, keep=(), wanted=(), spill=True):
        """Drop an entry but those in `keep` (in `victim`'s order; `wanted`:
        the entries waiting requests would resume from, or a callable giving
        them, called only when there is a choice) -> False: nothing to drop.
        Its segments go with the last entry holding them. A row resumed from
        it holds copies (restore); a prefill still running from it holds its
        pieces until it ends (the scheduler keeps that entry:
        _Scheduler._regrow). With the disk tier it is written there
        (`spill`) and the RAM it frees stays held, charged, until written; a
        caller that needs the memory now (it cannot wait for the writer)
        passes spill False: freed at once, as without the tier."""
        left = [e for e in self.entries if not any(e is k for k in keep)]
        if not left:
            return False
        if callable(wanted):
            try:
                wanted = wanted() if len(left) > 1 else ()
            except Exception:  # noqa: BLE001 — costs the order only, never the batch
                log.exception("prompt cache: the waiting requests' entries are unknown; "
                              "evicting without sparing them")
                wanted = ()
        e = self.victim(left, wanted)
        w = any(e is x for x in wanted)
        log.debug("prompt cache: evicting an entry of %d tokens (%s, %d ticks old, %.0f s idle)",
                  len(e.ids), "wanted" if w else
                  "stale" if self.stale(e, time.time()) else "live",
                  self.seq - e.seq, time.time() - e.used)
        self.entries.remove(e)
        self.evictions += 1
        self._recount(added=False)
        if self.disk is not None and spill:           # written at the end of the tick
            # the RAM it frees once written: its arrays no other entry holds
            live = {id(a) for x in self.entries for a in x.arrays}
            own = {id(a): a.nbytes for a in e.arrays if id(a) not in live}
            self.disk.offer(e, sum(own.values()), own)
        return True

    def insert(self, new, limit=None, wanted=(), spare=()):
        """Add an entry loaded from disk (QWEN_PROMPT_CACHE_DISK) as store
        adds a new one -> False when it was not kept: alone (beside the
        `spare` entries) over `limit`, the batch's headroom; else others are
        evicted (victim order, `wanted` last) down to QWEN_PROMPT_CACHE_GB
        and `limit`, the newest staying alone if over. It absorbs nothing
        (it was an entry of this cache: the store that made it absorbed its
        older turns) and is used now (a waiting request resumes from it)."""
        stays = [s for s in spare if any(s is e for e in self.entries)]
        held = self.charged() - self.bytes
        if limit is not None and self.bytes_of([new, *stays]) + held > limit:
            return False
        budget = PROMPT_CACHE_GB * GIB
        if callable(wanted):
            cap = budget if limit is None else min(budget, limit)
            try:
                wanted = wanted() if self.entries and self.bytes_of(self.entries + [new]) > cap \
                    else ()
            except Exception:  # noqa: BLE001 — costs the order only
                log.exception("prompt cache: the waiting requests' entries are unknown; "
                              "evicting without sparing them")
                wanted = ()
        new.used, new.seq = time.time(), self.seq
        self.entries.append(new)
        self._recount()
        # the soft cap: written to disk while the held RAM still fits the
        # headroom; the headroom: freed at once (a spill would hold as much)
        while len(self.entries) > 1 and self.bytes > budget and self.evict(
                keep=(new, *spare), wanted=wanted,
                spill=limit is None or self.charged() <= limit):
            pass
        while limit is not None and len(self.entries) > 1 and self.charged() > limit and \
                self.evict(keep=(new, *spare), wanted=wanted, spill=False):
            pass
        return True

    def store(self, ids, cache, ckpts, inherit=None, at=0, ends=(), protect=(), limit=None,
              kv_start=None, chain=None, wanted=(), reply=None, tick=True, sync=True, spare=(),
              clock=None, fresh=None, partial=False, seg=None, tail=None, moved=None):
        """Add the entry for `ids` (= prompt[:-1]) from the prefill's `cache`
        and checkpoints -> (entry or None when it was not kept, the older
        entries it absorbed). A reply store (_Scheduler._store_reply)
        passes the ids the row consumed, its cache after the last step,
        `reply` = where the decoded part starts, `tick` False (a turn stays
        two clock ticks: lookup + store), `sync` False (the copies are left
        lazy: the caller queues every finishing row's copies in one
        async_eval) and `spare` (never evicted here: the entry a running
        prefill resumed from, as _Scheduler._regrow spares it; the new entry
        must fit beside it, else it is not kept) and `clock` = the prompt
        entry's (used, seq): a reply continues the same turn, so it keeps the
        recency of its prefill (else a conversation that just finished, its
        next lookup imminent, would be the most recently used entry: evicted
        first, before the ones still decoding). `inherit`: the parent's part when the
        prefill resumed at `at`; `ends`: the prompt's slab ends; `protect`:
        the checkpoint positions thinning keeps (see _thin); `chain`: (the
        only checkpoint positions an entry that absorbs an older one keeps,
        the ones of them thinning keeps) (None: the fresh policy for it too;
        see _Scheduler._prompt_cache); absorbing only `partial` entries that
        absorbed nothing themselves makes no chain; `fresh`: the same pair
        for an entry that absorbs no chain (a long fresh prompt:
        QWEN_PROMPT_CACHE_CHAIN_FRESH_TOKENS; None: the cap policy);
        `partial`: it is a cancelled prefill's prefix (_Entry.partial);
        `tail`: the drafter's context at the entry's end (_with_tail);
        `moved` (QWEN_PROMPT_CACHE_MOVE): the husk of the entry the row
        resumed from (restore move=True: out of the cache, no KV): it is
        absorbed as if it were still an entry (its checkpoints move over, it
        makes a chain) when the new ids cover it; never returned among the
        absorbed ones (nothing to put back); `inherit` then has no pieces
        and `seg` covers [0, len);
        `limit`: the
        bytes the whole cache may hold beside the batch (entries go in
        `victim` order, `wanted` as there; when the new one alone is over, sized from the row's shapes
        before any copy, it is dropped and the cache left as it was: nothing
        absorbed, evicted or copied); `kv_start`:
        the quantization start (quantized entries resume from there on).

        The KV over [0, at) is the parent's pieces (the row's [0, at) is a
        copy of them), only [at, len) is copied out of the row; an fp16
        parent of a row its prefill quantized is not this entry's bits there,
        so then all of [0, len) is copied.

        A checkpoint at p is valid for this entry only where p is a slab end
        of this prompt: then the entry's KV over [0, p) is the cold prefill's
        (cut identically). A parent's checkpoints <= the resume point and an
        absorbed entry's checkpoints are all slab ends of a shared prefix, so
        of this prompt too; an absorbed entry's END state only if its length
        is one of `ends` (usually not: it is the token after the previous
        turn's <think>, so an exact repeat of that older prompt now resumes
        at its deepest checkpoint instead)."""
        if kv_start is None:
            kv_start = KV_KWARGS["quantized_kv_start"]
        if tick:
            self.seq += 1
        quantized = _quantized(cache)
        L = len(ids)
        ckpts = dict(ckpts)
        pieces, lo = [], 0
        if inherit is not None:                       # inherit the shared part
            if inherit.quantized == quantized and inherit.pieces:
                pieces, lo = list(inherit.pieces), at
            for p, s in inherit.ckpts.items():
                ckpts.setdefault(p, s)
            # (a resume point is always a slab end, except the end of a
            # reply entry: no checkpoint there, no prompt is ever cut there)
            if inherit.end is not None and at < L and at in ends:
                ckpts[at] = inherit.end
        # The new segment is sized before anything changes: an entry that
        # does not fit the batch's headroom alone is neither copied nor
        # allowed to absorb or evict anything (the cache stays as it was).
        # `seg` (a preempted row: _Scheduler._store_preempt): the new KV is
        # given, the dying row's own buffers [lo, L), nothing is copied
        assert seg is None or (seg.a, seg.b) == (lo, L), (seg and (seg.a, seg.b), lo, L)
        extra = seg.nbytes if seg is not None else _Seg.estimate(cache, lo, L) if L > lo else 0
        # An entry whose ids are a prefix of the new one (the previous turn of
        # this conversation, an exact repeat) is covered by it: its
        # checkpoints move over and it goes. Not an unquantized entry for a
        # quantized new one: its checkpoints below quantized_kv_start are only
        # resumable from fp16 KV, it is the one resume point there. A reply
        # entry whose prompt part (ids[:reply]) the new ids cover goes too:
        # the conversation went on without that reply (re-tokenized
        # differently, regenerated), only its checkpoints are of use.
        def prefix(e):
            return len(e.ids) <= L and ids[: len(e.ids)] == e.ids
        covered = [e for e in (self.entries if moved is None else [*self.entries, moved])
                   if (prefix(e) or (e.reply is not None and e.reply <= L
                                     and ids[: e.reply] == e.ids[: e.reply]))
                   and not (quantized and not e.quantized)
                   # a given segment (a preempted row's buffers) inherits nothing:
                   # absorbing a running prefill's parent would drop its bytes
                   # from the count while that prefill's _Inherit holds them
                   and not (seg is not None and any(e is s for s in spare))]
        for e in covered:
            for p, s in e.ckpts.items():
                if p < L:                             # (a prompt-part cover may hold deeper ones)
                    ckpts.setdefault(p, s)
            if e.end is not None and len(e.ids) in ends and len(e.ids) < L and prefix(e):
                ckpts.setdefault(len(e.ids), e.end)
        # a chain: this conversation's previous turn absorbed (a cancelled
        # prefill's prefix alone is this same turn, not a previous one)
        chained = any(not e.partial or e.chained for e in covered)
        heir = False
        if moved is not None:                         # a husk: nothing of it is put back
            covered = [e for e in covered if e is not moved]
            # a prefix of the husk itself (a cancelled prefill gives back the
            # part of it the row held): that entry cut short, a chain when
            # it was one (the retry's store absorbs it as it would have
            # absorbed the whole)
            heir = (L <= len(moved.ids) and moved.ids[:L] == ids
                    and (not moved.partial or moved.chained))
        policy = chain if chained and chain is not None else fresh
        if policy is not None:
            # its next turn resumes at the last message starts, older turns'
            # ones and strides below the tail are almost never used again
            ckpts = {p: s for p, s in ckpts.items() if p in policy[0]}
            protect = policy[1]
        if quantized:                                 # never resumable below the start
            ckpts = {p: s for p, s in ckpts.items() if p >= kv_start}
        ckpts.pop(L, None)
        _thin(ckpts, PROMPT_CACHE_CKPTS, protect)
        # the chain without its own segment yet: nbytes + extra = its size
        new = _Entry(ids, pieces, ckpts, _with_tail(_recurrent_states(cache), tail),
                     _layout(cache), quantized, kv_start if quantized else 0, reply, partial)
        new.chained = chained or heir
        new.seq = self.seq
        if clock is not None:                         # before any eviction can see it
            new.used, new.seq = clock
        # what can never be evicted for it: the spared entries it does not absorb
        stays = [s for s in spare if any(s is e for e in self.entries)
                 and not any(s is c for c in covered)]
        # (+ the RAM the disk tier still holds: evicted entries not written yet)
        if limit is not None and self.bytes_of([new, *stays]) + extra + \
                self.charged() - self.bytes > limit:
            # beside the batch (the rows' reservations + the cache within the
            # budget + the idle prefill reserve) it does not fit even alone
            # (beside the running prefill's entry, which stays)
            self.evictions += 1
            where = " beside the running prefill's entry" if stays else ""
            if reply is not None:                     # its own prefix (log parsers count it apart)
                log.info("prompt cache reply not kept: over the batch's headroom (%.2f GiB)%s: "
                         "entry of %d tokens (%.2f GiB); counted as an eviction, the cache is "
                         "unchanged", max(limit, 0) / GIB, where, L, (new.nbytes + extra) / GIB)
            else:
                log.info("prompt cache: entry of %d tokens (%.2f GiB) not kept: over the "
                         "batch's headroom (%.2f GiB)%s; counted as an eviction, the cache is "
                         "unchanged", L, (new.nbytes + extra) / GIB, max(limit, 0) / GIB, where)
            return None, []
        rest = [e for e in self.entries if not any(e is c for c in covered)]
        budget = PROMPT_CACHE_GB * GIB
        if callable(wanted):
            # asked once, before anything changes, and only when an eviction
            # is coming; a failure costs the order only, never the entry
            cap = budget if limit is None else min(budget, limit)
            try:
                wanted = wanted() if rest and self.bytes_of(rest + [new]) + extra > cap else ()
            except Exception:  # noqa: BLE001
                log.exception("prompt cache: the waiting requests' entries are unknown; "
                              "evicting without sparing them")
                wanted = ()
        self.entries = rest + [new]
        try:
            self._recount()
            # the soft cap: an evicted entry goes to disk (its RAM held until
            # written) while that still fits the headroom with the new
            # segment, which is copied right below; else it is freed at once
            while len(self.entries) > 1 and self.bytes + extra > budget and self.evict(
                    keep=(new, *spare), wanted=wanted,
                    spill=limit is None or self.charged() + extra <= limit):
                pass
            if limit is not None:
                # it fits alone: others go (victim order) until it fits beside
                # the batch, freed at once (a spill would hold what it frees)
                while len(self.entries) > 1 and self.charged() + extra > limit and \
                        self.evict(keep=(new, *spare), wanted=wanted, spill=False):
                    pass
            if L > lo:                                # copied once room is made
                if seg is None:
                    seg = (_Seg.copy_from(cache, lo, L) if sync
                           else _Seg.copy_from(cache, lo, L, sync=False))
                new.chain = pieces + [(seg, lo, L)]
                new.index()
        except BaseException:
            # no half-built entry (published without its segment): the
            # absorbed ones come back (the evicted ones are gone), the
            # caller logs and goes on
            self.entries = [e for e in self.entries if e is not new] + covered
            self._recount()
            raise
        self._recount(added=False)                    # its own segment: nothing new to peek
        if self.bytes > budget:
            log.warning("prompt cache: the newest entry %s(%d tokens, %.2f GiB) is over "
                        "QWEN_PROMPT_CACHE_GB=%.1f; kept, evicted on demand",
                        "alone " if len(self.entries) == 1 else
                        "beside the running prefill's entry ", L, self.bytes / GIB,
                        PROMPT_CACHE_GB)
        return new, covered
