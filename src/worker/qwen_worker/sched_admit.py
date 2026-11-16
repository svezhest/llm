"""_Scheduler: admission, budget, prefill lane, task queue."""

import itertools
import time
import weakref

import mlx.core as mx

from . import config as cfg
from .config import (CLEAR_CACHE, DRAFT_SAMPLING, GIB, log, LOOKUP, PREFILL_LANE, PREFILL_STEP,
                     PROMPT_CACHE, PROMPT_CACHE_KEEP_TOKENS, PROMPT_CACHE_KEEP_WAIT_S,
                     RESERVE_TOKENS)
from .sampler import _Penalties, make_sampler
from .sizing import _bounds
from .logprobs import _cache_mode, _wants
from .request import _Row


class _AdmitMixin:
    """Admission and the memory budget: reservations, _fits / _regrow, the
    prefill lane, the task queue (_next / _advance)."""

    def _lane_first(self):
        """True while the row the lane just prefilled waits for its first
        token: its first decode step computes it and the next one reads it
        (step-ahead), so with the decode share of a few rows (1 - s: ~3 % of a
        main slab at B = 1 of 16) the next main slab (seconds at long offsets)
        would run between the two and hold its first token (GPU gate
        2026-10-24: short TTFT p50 6.5 s, ~half a slab + a whole one). Two
        decode steps per lane request; the main prefill's slabs are unchanged."""
        r = self.lane_new
        if r is None:
            return False
        if r.done or r.n > 0 or not any(r is x for x in self.rows):
            self.lane_new = None
            return False
        return True

    def _bytes(self, n, kvq=False):
        """Reservation for a row of n KV tokens. Quantized KV (QWEN_KV_BITS)
        past quantized_kv_start: n quantized tokens (packed values + an fp16
        scale and bias per group: _kv_token_bytes) and, until the row's cache
        is quantized (`kvq` False), the fp16 KV it holds before: the cache is
        quantized whole (to_quantized) after the first prefill slab ending at
        or past the start (a decode step at the start), which leaves the fp16
        buffers of those c <= start + PREFILL_STEP tokens beside their
        quantized copy for that moment -- the row's peak. A row whose cache is
        quantized already (resumed from a quantized entry, or past that
        point: _kvq_mark) never holds fp16 KV again."""
        start = self.kv["quantized_kv_start"]
        if self.kv["kv_bits"] and n >= start:
            kv = n * self.kv_q + (0 if kvq else min(n, start + PREFILL_STEP) * self.kv_fp16)
        else:
            kv = n * self.kv_fp16
        # +5%: a KVCache grows by concatenating 256-token steps; the recurrent
        # state twice: the queued step's output exists beside its input
        return int(kv * 1.05) + 2 * self.gdn_row + (8 << 20)

    def _kvq_mark(self, row, known=False):
        """Quantized KV: once the row's cache is quantized (`known`: it is
        about to be, restored from a quantized entry) its reservation drops
        the fp16 copy (_bytes kvq). Plain attributes only: no MLX op, safe
        while a decode step is built."""
        if row.kvq or not self.kv["kv_bits"] or row.group is not None:
            return
        if not known:
            c = row.cache[self.fa_layers[0]] if row.cache is not None and self.fa_layers else None
            if c is None or not hasattr(c, "bits"):
                return
        row.kvq = True
        need = self._bytes(row.tokens, kvq=True)
        if not row.done:
            self.reserved += need - row.reserved
        row.reserved = need

    def _step_bytes(self, rows):
        """Bytes a decode step over `rows` must read at least: the weights
        once, every row's KV (quantized whole past quantized_kv_start), its
        recurrent state read and written."""
        start, kv = self.kv["quantized_kv_start"], 0
        for r in rows:
            n = r.pos
            kv += n * self.kv_q if self.kv["kv_bits"] and n >= start else n * self.kv_fp16
        return self.w_bytes + kv + 2 * len(rows) * self.gdn_row

    def _tokens(self, req):
        return len(req.ids) + min(req.max_tokens, RESERVE_TOKENS)

    def _reserve(self, req):
        return self._bytes(self._tokens(req))

    def _regrow(self, row, ahead=0):
        """Re-account a row that outgrew its reservation (by 1024-token steps,
        up to its own limit): admissions wait while the batch is over budget.
        Growth past the headroom too (budget + prefill reserve; the last 1 GiB
        stays slack) would end in an OOM of every row: first the prompt
        cache gives way (entries evicted in victim order while that brings the
        batch back under the headroom; never the running prefill's parent, the
        admitted-not-started request's goes last: _wanted), else the growing
        row ends with "length" — only
        when other rows run: a lone row is bounded by MAX_CONTEXT alone, as
        before batching. `ahead` (QWEN_PREEMPT, _iterate before a step): the
        growth the next step will need, -> False instead of the cut (the
        caller preempts the row); else -> False when it was cut."""
        if row.pos + ahead < row.tokens:
            return True
        tokens = min(row.pos + ahead + 1024, len(row.req.ids) + row.req.max_tokens + 1)
        g = row.group
        if g is None:
            need = self._bytes(tokens, row.kvq)
            rows = self.reserved + need - row.reserved
        else:                                   # a member grows its group's charge
            need = self._group_bytes(g.P, g.rows, {row: tokens})
            rows = self.reserved + need - g.reserved
        headroom = self.budget + self.prefill_reserve
        pc = self.eng._pcache
        if rows <= headroom < rows + pc.charged():
            # not the entry a running prefill resumed from: its pieces stay
            # allocated until that prefill stores, whatever the count says;
            # freed at once (spill False: the growth cannot wait for a write)
            keep = self._parents()
            n = 0
            while rows + pc.charged() > headroom and pc.evict(keep, self._wanted, spill=False):
                n += 1
            if n:
                log.info("prompt cache: %d entries evicted for the growth of id=%s "
                         "(%.2f GiB left)", n, row.req.id, pc.bytes / GIB)
        if rows + pc.charged() > headroom and pc.disk is not None:
            # only the disk tier's held RAM is in the way: gone within ~1 s / GB,
            # a cut answer is not (review: the cut came from a transient)
            self._disk_room(lambda: rows + pc.charged() - headroom, row.req.id)
        if rows + pc.charged() > headroom and sum(not r.done for r in self.rows) > 1:
            if ahead:
                return False
            log.warning("id=%s ended at %d tokens: the batch's KV would pass its memory "
                        "headroom (%.2f GiB reserved)", row.req.id, row.n, self.reserved / GIB)
            self._finish(row, "length")
            return False
        row.tokens = tokens
        if g is None:
            self.reserved += need - row.reserved
            row.reserved = need
        else:
            self.reserved += need - g.reserved
            g.reserved = need
        return True

    def _tasks(self):
        """The rows of the prefills in progress: the main task's, the lane's."""
        return [r for r in (self.task_row, self.lane_row) if r is not None]

    def _parents(self, but=None):
        """The entries the running prefills (but `but`) resumed from: never
        evicted while they run (their pieces stay allocated until they store,
        whatever the count says). () unless a prefill is running."""
        return tuple(r.parent for r in self._tasks() if r is not but and r.parent is not None)

    def _fits(self, req, rows, lane=False, extra=0, protect=()):
        """Would `req` (the head of the queue) be admitted next to `rows`
        running rows? The rows' reservations + the prompt cache's live bytes
        + `req` within the budget: when only the cache is in the way, its
        entries are evicted (victim order), sparing the one `req` would resume
        from when that is at least a full slab deep, and nothing when evicting
        all the others would not admit it either. When that entry alone keeps
        `req` out, `req` waits for a row to end only if the hit saves at least
        QWEN_PROMPT_CACHE_KEEP_TOKENS and for at most
        QWEN_PROMPT_CACHE_KEEP_WAIT_S (from the first time it blocked); else
        that entry is evicted first (it has to go: the others would not do)
        and `req` admitted now. The entries running prefills resumed from are
        never evicted (_parents). `lane` (_admit_lane): its entry is kept
        whatever its depth and never given up (without it the request is no
        lane request); a preempted request (_preempt) never gives its up
        either: its prompt + generated tokens are worth waiting for. `extra`:
        bytes reserved beside it (the head a lane request overtakes);
        `protect`: entries never evicted here (the hits of the requests it
        overtakes)."""
        if rows == 0:
            return True
        need = self._reserve(req) + extra
        if rows >= self.max_batch or self.reserved + need > self.budget:
            return False
        pc = self.eng._pcache
        live = mx.get_active_memory() + (0 if CLEAR_CACHE else mx.get_cache_memory())
        # held: RAM the disk tier still holds (evicted entries until written,
        # loads until inserted); 0 without it
        held = pc.charged() - pc.bytes
        if self.reserved + need + pc.bytes + held > self.budget:
            keep, depth, first = (), 0, None
            if PROMPT_CACHE and not req.images:
                e, depth = self._peek(req)
                if e is not None and (depth >= PREFILL_STEP or lane):
                    keep = (e,)
            spare = tuple(x for x in {id(x): x for x in self._parents() + tuple(protect)}.values()
                          if not any(x is k for k in keep))

            def blocked(floor, h=held):         # out even with every other entry evicted
                # (h: the tier's hold then; 0 = once the writer released it)
                return (self.reserved + need + floor + h > self.budget
                        or live - (held - h) - (pc.bytes - floor) + need + self.prefill_reserve
                        > self.cap)
            floor = pc.bytes_of(keep + spare)
            # a stuck write's hold (pcdisk TIMEOUT_S) may never go: not waited for
            stuck = pc.disk.stuck_bytes() if held else 0
            if held > stuck and blocked(floor) and not blocked(floor, stuck) and \
                    self._held_wait(req):
                # the tier's hold is what keeps it out (with every other entry
                # evicted it would not fit beside the hold), and it fits with
                # its hit once the writer released what it holds (~1 s / GB):
                # wait for that, the KEEP_WAIT clock untouched and the hit not
                # given up (review: a second's write did that). When evicting
                # the others admits it beside the hold, they go now instead
                # (spilled: it then waits for their own write only)
                return False
            if keep and blocked(floor) and not blocked(pc.bytes_of(spare)):
                # only the entry it would resume from is in the way
                if lane or req.preempts or any(keep[0] is x for x in self._parents()):
                    return False
                now = time.time()
                if req.keep_since is None:
                    req.keep_since = now
                waited = now - req.keep_since
                if depth >= PROMPT_CACHE_KEEP_TOKENS and waited <= PROMPT_CACHE_KEEP_WAIT_S:
                    return False                # worth waiting for a row to end
                log.info("prompt cache: id=%s gives up its %d-token hit to be admitted (%s)",
                         req.id, depth, f"waited {waited:.1f} s" if depth
                         >= PROMPT_CACHE_KEEP_TOKENS else "under QWEN_PROMPT_CACHE_KEEP_TOKENS")
                first, keep, floor = keep[0], (), pc.bytes_of(spare)
            if blocked(floor):
                return False                    # it waits either way: evict nothing
            n = 0
            if first is not None:               # given up: it goes before any other
                n += pc.evict(keep=[x for x in pc.entries if x is not first])
            while self.reserved + need + pc.bytes > self.budget and pc.evict(keep + spare,
                                                                              self._wanted):
                n += 1
            if n:
                log.info("prompt cache: %d entries evicted to admit id=%s (%.2f GiB left)",
                         n, req.id, pc.bytes / GIB)
            if self.reserved + need + pc.charged() > self.budget:
                # the entries just evicted went to disk: their RAM is free once
                # written (~0.3-0.8 s/GB); admitted then, nothing more evicted
                return False
            live = mx.get_active_memory() + (0 if CLEAR_CACHE else mx.get_cache_memory())
        return live + need + self.prefill_reserve <= self.cap

    def _peek(self, req):
        """pc.peek for a waiting request, memoized on it until an entry is
        added or the one it found is evicted (removing another one cannot
        change the deepest hit; a weak reference: a waiting request never
        keeps an evicted entry alive) -> (entry or None, depth)."""
        pc = self.eng._pcache
        m = req.peeked
        if m is not None and m[0] == pc.version:
            if m[1] is None:
                return None, m[2]
            e = m[1]()
            if e is not None and any(e is x for x in pc.entries):
                return e, m[2]
        look = self._lookup_ids(req)
        if look is None:
            return None, 0
        e, depth = pc.peek(look, self._ends(req))
        req.peeked = (pc.version, weakref.ref(e) if e is not None else None, depth)
        return e, depth

    @staticmethod
    def _lookup_ids(req):
        """The ids a prompt-cache lookup matches, None: no lookup
        (prompt_cache "off"). A prompt_logprobs request resumes at most at
        len - 2, so its last token is always scored (an exact repeat would
        resume at len - 1, prefilling nothing)."""
        if _cache_mode(req) == "off":
            return None
        return req.ids[:-1] if _wants(req, "prompt_logprobs") else req.ids

    def _wanted(self):
        """The entries the requests next in line would resume from at least
        a slab deep (as _fits keeps the head's): evicted last. The admitted
        request whose prefill has not started yet (its lookup is still to
        come: _regrow may run before it) and the first max_batch waiting
        ones (admitted before the next store, if at all; each cold peek is
        an _lcp per entry). The queue is copied under the lock: handlers
        append to it meanwhile."""
        with self.cv:
            reqs = list(itertools.islice(self.waiting, self.max_batch))
        for task in reversed(self._tasks()):        # the main task's first, then the lane's
            if not task.started:
                reqs.insert(0, task.req)
        out = []
        for req in reqs:
            if not req.images:
                e, depth = self._peek(req)
                if e is not None and depth >= PREFILL_STEP and not any(e is x for x in out):
                    out.append(e)
        return out

    def _bounds(self, req):
        """The prompt's slab ends (memoized on the request: _fits asks every
        tick while it waits)."""
        if req.bounds is None:
            req.bounds = [] if req.images else _bounds(req.ids, self.eng._im_start)
        return req.bounds

    def _ends(self, req):
        return set(self._bounds(req)) | {len(req.ids) - 1}

    def _admit(self):
        req = self.waiting[0]
        if not self._fits(req, len(self.rows) + (self.lane_row is not None)):
            return
        if not self.rows and self.lane_row is None:
            mx.reset_peak_memory()
        self.waiting.popleft()
        row = self._setup(req)
        if row is None:
            return
        try:
            spec = self._spec_ok(row)
        except Exception as e:  # noqa: BLE001 — this request's options, not the batch's
            log.exception("request setup failed: %s", req.id)
            self._fail(row, e)
            return
        self.task_row = row
        self.task = self._spec(row) if spec else self._prefill(row)

    def _setup(self, req):
        """The row of a request taken off the queue (its reservation counted)
        -> None when its options failed (the request got its error). A
        preempted request goes on with its own sampler (key) and penalty
        counts; its _Control masks are built already."""
        tokens = self._tokens(req)
        row = _Row(req, tokens, self._bytes(tokens))
        self.reserved += row.reserved
        if not req.preempts:
            req.t0 = time.time()                # reset when its prefill starts (_advance)
        try:
            if req.carry is not None:
                (row.sampler, row.pen), req.carry = req.carry, None
            else:
                row.sampler = make_sampler(req.sampling, seed=req.sampling.get("seed"))
                row.pen = _Penalties(req.sampling, req.ids, self.eng.vocab)
                req.ctl.build()
        except Exception as e:  # noqa: BLE001 — this request's options, not the batch's
            log.exception("request setup failed: %s", req.id)
            self._fail(row, e)
            return None
        return row

    def _lane_tokens(self, req):
        """Tokens `req` would prefill if admitted now: from where its
        prompt-cache lookup would resume (peek: the same scan) to its
        prompt's end -> None: never a lane request (vision: image work and
        positions of their own; cancelled)."""
        if req.images or req.cancel.is_set():
            return None
        e, depth = self._peek(req) if PROMPT_CACHE else (None, 0)
        return max(0, len(req.ids) - 1 - (depth if e is not None else 0))

    def _admit_lane(self):
        """QWEN_PREFILL_LANE: admit the first waiting text request that needs
        at most PREFILL_LANE prefill tokens, beside the running plain prefill
        (the main task). It overtakes the waiting requests before it only if
        the budget also holds the head's reservation (and its row): the lane
        never makes the head wait longer for memory; the queue keeps its
        order otherwise. Same admission rules as _admit, but its hit is never
        given up (_fits lane=True). Its prefill runs at prefill's next turns
        until it ends (_advance_lane: before the main prefill's next slab; up
        to PREFILL_LANE_BURST lane requests in a row), then it joins the batch
        like any row."""
        with self.cv:
            cand = list(itertools.islice(self.waiting, self.max_batch))
        if not cand:
            return
        head = cand[0]
        for req in cand:
            n = self._lane_tokens(req)
            if n is not None and n <= PREFILL_LANE:
                break
        else:
            return
        over = req is not head
        # the main prefill's row (not in self.rows yet) and, overtaken, the head's
        # the requests it overtakes keep their hits (their entries are spared)
        ahead = cand[:cand.index(req)]
        protect = tuple(e for e in (self._peek(x)[0] for x in ahead if not x.images) if e is not None)
        if not self._fits(req, len(self.rows) + 1 + over, lane=True,
                          extra=self._reserve(head) if over else 0, protect=protect):
            return
        with self.cv:
            if req.cancel.is_set() or not any(req is x for x in self.waiting):
                return                          # (cancelled meanwhile: the next tick says so)
            self.waiting.remove(req)
        row = self._setup(req)
        if row is None:
            return
        log.info("id=%s prefill lane: %d tokens beside the prefill of id=%s%s", req.id, n,
                 self.task_row.req.id, f", ahead of {cand.index(req)} waiting" if over else "")
        self.lane_row = row
        self.lane = self._prefill(row)
        self.lane_run += 1

    def _spec_ok(self, row):
        req = row.req
        greedy = req.sampling["temperature"] <= 0 and not row.pen.active
        # QWEN_LOOKUP without a drafter: lookup rounds + plain steps (_spec);
        # fp16 KV only (a round's rollback trims the cache), no thinking budget
        look = (self.eng.draft is None and LOOKUP == "1" and self.kv["kv_bits"] is None
                and req.thinking is None)
        if ((self.eng.draft is None and not look) or cfg.BATCH_INVARIANT or req.images
                or req.golden is not None or not (greedy or DRAFT_SAMPLING)):
            return False
        if row.lpk is not None or _wants(req, "prompt_logprobs"):
            # the DFlash loop computes neither (its prefill skips the logits,
            # its verify rows are drafts): plain prefill + decoding
            log.info("logprobs: speculative decoding off for this request")
            return False
        if req.guide is not None or req.ctl.active:
            # DFlash drafts are not grammar-masked: the verify rows would each
            # need a mask over the draft prefix. Plain decoding instead.
            log.info("grammar / token options: speculative decoding off for this request")
            return False
        # (a preempted request goes on as a plain row: its carried sampler
        # and penalty counts are the batch path's; a lane runs beside nothing else)
        return (not self.rows and (self.max_batch == 1 or not self.waiting)
                and req.preempts == 0 and self.lane is None)

    def _advance(self):
        self._next(lane=False)

    def _advance_lane(self):
        self._next(lane=True)

    def _next(self, lane):
        """One tick of the main task or of the lane: a slab (DFlash: a round)."""
        row = self.lane_row if lane else self.task_row
        if not row.started:
            # admitted is not started: with rows running, the first slab
            # waits for decode's share (resume) like any other
            row.started = True
            if not row.req.preempts:            # the stream started before
                row.req.t0 = time.time()
                row.req.queue_ms = int((row.req.t0 - row.req.t_arrival) * 1000)
        lm = self.lm
        if PREFILL_LANE and row.rope is not None:
            # Text slabs past the first read the model's _rope_deltas (offset
            # + delta) and a cold first slab writes both: two prefills
            # interleaved each get back the state their own last slab left.
            lm._rope_deltas, lm._position_ids = row.rope
        try:
            next(self.lane if lane else self.task)
            if PREFILL_LANE:
                row.rope = (lm._rope_deltas, lm._position_ids)
        except StopIteration:
            self._task_done(lane)
        except Exception as e:  # noqa: BLE001 — a failed prefill fails its request only
            log.exception("prefill failed: %s", row.req.id)
            self._fail(row, e)
            self._lost(row)
            self._task_done(lane)

    def _task_done(self, lane):
        if lane:
            row = self.lane_row
            if row is not None and not row.done and any(row is x for x in self.rows):
                self.lane_new = row                 # (joined: _lane_first)
            self.lane = self.lane_row = None
        else:
            self.task = self.task_row = None
