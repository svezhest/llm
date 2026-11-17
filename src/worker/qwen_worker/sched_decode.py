"""_Scheduler: the batch, the decode step, row ends, preemption."""

import mlx.core as mx

from .config import GIB, log, PREEMPT, PROMPT_CACHE_REPLY
from .sampler import sample_rows
from .pcache import _Seg
from .nosync import _no_sync
from .rows import _gdn_concat, _gdn_split, _gdn_stack, _gdn_take, RowsKV
from .logprobs import _cache_mode, _forced_logprobs, _read_logprobs, _token_logprobs, _wants


class _DecodeMixin:
    """The batch: join / drop rows, the decode step and its sampling, row ends,
    preemption (QWEN_PREEMPT)."""

    # --- the batch ----------------------------------------------------------------
    def _hidden_dtype(self):
        if self.hdtype is None:
            self.hdtype = self.lm.model.embed_tokens(mx.array([[0]])).dtype
        return self.hdtype

    def _join(self, row):
        """Add a prefilled row. The new stacked state is built first and
        swapped in only on success, so a failed join leaves the batch as it
        was. (The leading _compact never raises for a group: a failed tidy
        fails that group's rows, not the joining request.)"""
        self._compact()
        gdn = self.gdn
        if len(self.rows) == 1:
            gdn = _gdn_stack(self.lm, [self.rows[0].cache, row.cache], self._hidden_dtype())
        elif self.rows:
            gdn = _gdn_concat(self.gdn, _gdn_stack(self.lm, [row.cache], self._hidden_dtype()))
        self.gdn = gdn
        self.rows.append(row)
        self._regrow(row)
        if self.cascade and not row.done:
            try:
                self._cascade_join(row)
            except Exception:  # noqa: BLE001 — the helpers change nothing until they succeed
                log.exception("cascade: id=%s stays ungrouped", row.req.id)

    def _drop_done(self):
        keep = [i for i, r in enumerate(self.rows) if not r.done]
        if len(keep) != len(self.rows):
            # reply KV: the finished rows' caches, taken before the batch's
            # stacked state is cut (lazy; _store_reply queues the copies)
            replies = [(r, self._reply_cache(r, i)) for i, r in enumerate(self.rows)
                       if r.done and r.fed is not None and PROMPT_CACHE_REPLY and r.own is None]
            # QWEN_PROMPT_CACHE_MOVE: the finished row's buffers become its entry
            owned = [(r, self._reply_cache(r, i) if r.fed is not None and PROMPT_CACHE_REPLY
                      else None) for i, r in enumerate(self.rows) if r.done and r.own is not None]
            if self.gdn is not None:
                if len(keep) == 1:              # back to the plain library call
                    _gdn_split(self.gdn, keep[0], self.rows[keep[0]].cache)
                    self.gdn = None
                else:
                    self.gdn = _gdn_take(self.gdn, keep) if keep else None
            self.rows = [self.rows[i] for i in keep]
            for r, cache in owned:
                self._settle(r, cache)
            kept, absorbed = [], []
            for r, cache in replies:
                try:
                    e, covered = self._store_reply(r, cache)
                except Exception:  # noqa: BLE001 — the request already ended: never the batch's failure
                    log.exception("id=%s prompt cache reply store failed", r.req.id)
                    e, covered = None, []
                if e is not None:
                    kept.append(e)
                    absorbed += covered
            if kept:
                # every finishing row's copies + end states in ONE async_eval
                # (one command buffer between step k and k+1, not two per row;
                # never waited for: step-ahead)
                pc = self.eng._pcache
                try:
                    mx.async_eval([a for e in kept for a in e.arrays])
                except Exception:  # noqa: BLE001 — no entry with lazy arrays that cannot run
                    # as a failed store: the reply entries go, the prompt
                    # entries they absorbed come back (the evicted ones are gone)
                    back = [c for c in absorbed if not any(c is k for k in kept)]
                    pc.entries = [x for x in pc.entries if not any(x is e for e in kept)]
                    pc.entries += [c for c in back if not any(c is x for x in pc.entries)]
                    pc.evictions += len(kept)
                    pc._recount()
                    log.exception("prompt cache reply not kept: queueing the copies failed; %d "
                                  "reply entries dropped, %d prompt entries restored (counted "
                                  "as evictions)", len(kept), len(back))

    def _end(self, row, event):
        """The row's last event, with its stats snapshot taken here (the next
        admission may reset the peak before the handler reads it)."""
        row.done = True
        row.parent = None
        self.reserved -= row.reserved
        req = row.req
        req.peak, req.active, req.pool = (mx.get_peak_memory(), mx.get_active_memory(),
                                          mx.get_cache_memory())
        req.batch_mean = req.rows_sum / req.steps if req.steps else 0.0
        req.out.put(event)
        # give a long row's KV back to the OS; clearing after every short row
        # would only make the next step allocate fresh buffers
        if row.pos > 8192 or all(r.done for r in self.rows):
            self.clear = True

    def _finish(self, row, why):
        if not row.done:
            row.why = why
            self._end(row, ("done", why))

    def _fail(self, row, e):
        if row is None or row.done:
            return
        # str(e), not e: a log record must not keep the exception (and through
        # its traceback the failed step's frames: plan, groups) alive
        log.error("generation failed: %s: %s: %s", row.req.id, type(e).__name__, str(e))
        self._end(row, ("error", type(e).__name__, str(e)))

    def _fail_rows(self, e):
        """A failed decode step: its rows fail, a prefill in progress goes on
        (its caches are its own)."""
        for r in self.rows:
            self._fail(r, e)
            self._lost(r)
        self.rows, self.gdn = [], None
        self._drop_groups()                         # the arrays go with the rows
        self.reserved = sum(r.reserved for r in self._tasks())
        try:
            mx.clear_cache()
        except Exception:  # noqa: BLE001 — the failure is already reported
            log.exception("clear_cache after a failed step")

    def _step(self, rows):
        """Queue one decode forward for `rows` -> their next tokens (lazy (1,)
        arrays; None where the row's sampling failed)."""
        lm = self.lm
        B = len(rows)
        if B == 1:
            r = rows[0]
            if r.group is not None:
                raise RuntimeError("cascade: a grouped row decoding alone")
            out = lm(r.y[None], cache=r.cache,
                     position_ids=mx.array([[r.pos + r.delta]], dtype=mx.int32))
            logits = [out.logits[:, -1, :]]
        else:
            # Qwen3_5GatedDeltaNet silently restarts from zeros on a batch-size
            # mismatch: never let a stacking slip through
            if any(c.cache[1].shape[0] != B for c in self.gdn.values()):
                raise RuntimeError("stacked recurrent state out of step with the batch")
            plan = self._plan(rows) if self.groups else None
            cache = [self.gdn[i] if i in self.gdn else RowsKV([r.cache[i] for r in rows], i, plan)
                     for i in range(len(lm.model.layers))]
            pos = mx.array([[r.pos + r.delta] for r in rows], dtype=mx.int32)
            if plan is None:
                out = lm(mx.concatenate([r.y for r in rows])[:, None], cache=cache, position_ids=pos)
            else:
                out = self._grouped_forward(plan, mx.concatenate([r.y for r in rows])[:, None],
                                            cache, pos)
            lg = out.logits[:, -1, :]
            logits = [lg[i:i + 1] for i in range(B)]
        nxt = []
        pend = []                                   # (index in nxt, row, logits): drawn together
        want = []                                   # (index in nxt, row, raw logits): logprobs rows
        for r, x in zip(rows, logits):
            r.pos += 1
            self._regrow(r)
            if r.done:                              # ended by _regrow
                nxt.append(None)
                continue
            r.req.steps, r.req.rows_sum = r.req.steps + 1, r.req.rows_sum + B
            if r.lpk is not None:
                want.append((len(nxt), r, x))
            try:
                if B > 1 and r.req.guide is None:
                    pend.append((len(nxt), r, self._logits(r, x)))
                    nxt.append(None)
                else:
                    nxt.append(self._sample(r, x))
            except Exception as e:  # noqa: BLE001 — e.g. a grammar error: this row only
                self._fail(r, e)
                nxt.append(None)
        if pend:                                    # a drawing error fails its group's rows only
            ys = sample_rows([r.sampler for _, r, _ in pend], [x for _, _, x in pend],
                             [r.pos for _, r, _ in pend],
                             on_error=lambda i, e: self._fail(pend[i][1], e))
            for (j, _, _), y in zip(pend, ys):
                nxt[j] = y
        outs = [y for y in nxt if y is not None]
        for j, r, x in want:
            # raw logprobs of the drawn token (+ top-k): built here, queued
            # with the tokens (never an op on the step's outputs later);
            # read after the token, only for these rows
            r.lp_next = None
            if nxt[j] is not None and not r.done:
                r.lp_next = _token_logprobs(x, nxt[j], r.lpk, keep=r.req.thinking is not None
                                            or r.req.golden is not None)
                outs += [a for a in r.lp_next[:3] if a is not None]
        mx.async_eval(outs)
        self.log_rows += B
        return nxt

    def _logits(self, row, logits):
        """One row's logits before the draw: penalties, token masks (in
        vLLM's order), then the KV quantization check."""
        req = row.req
        if row.pen.active:
            if not row.first:                       # y is a generated token, not the prompt
                row.pen.add(row.y)
            logits = row.pen(logits)
        if req.ctl.active:
            if not row.first:
                req.ctl.n += 1
            logits = req.ctl.mask(logits)
        self.quantize(row.cache)
        self._kvq_mark(row)
        return logits

    def _sample(self, row, logits):
        """One row's logits -> its next token: _logits, then grammar, sampler.
        (In a batch, rows without a grammar are drawn together: sample_rows.)
        The draw's key is the token's absolute position, row.pos (the KV
        length after this step: _step counted it), so a row draws the same
        noise for a position alone, in a batch, after a cache hit or a
        preemption, and as _spec's verify rows and plain steps do."""
        logits = self._logits(row, logits)
        if row.req.guide is not None:
            return row.req.guide.sample(logits, row.sampler, row.pos)
        return row.sampler(logits, row.pos)

    def _iterate(self):
        """One decode step for the batch. Step-ahead: the next forward is
        queued before this step's tokens are read, so the host work overlaps
        the GPU; a row with a grammar or a multi-token bad word needs each
        token before its next mask, so while one is in the batch the whole
        batch reads first and steps after."""
        for r in self.rows:
            if r.req.cancel.is_set():
                self._finish(r, "cancelled")
            elif r.n == r.req.max_tokens:          # the pending token is never read
                self._finish(r, "length")
        self._compact()
        if not self.rows:
            return
        if PREEMPT:
            self._preempt_rows()
            if not self.rows:
                return
        rows = self.rows
        ahead = not any(r.sync for r in rows)
        with _no_sync():
            nxt = self._step(rows) if ahead else None
        live = [r for r in rows if not r.first and not r.done]
        # item() per row: each token was evaluated by its step's async_eval,
        # so this waits for that step only. Any new op on them (a concatenate
        # + tolist) would be queued behind the step just submitted and wait
        # for it too: the GPU then idled while the host built the next step
        # (measured −5…12% step time at B = 8/16, step_profile).
        toks = [r.y.item() for r in live]
        # logprobs rows only: evaluated with their token's step (no new op)
        lps = [_read_logprobs(r.lp) if r.lp is not None else None for r in live]
        for r, tok, lpi in zip(live, toks, lps):
            req = r.req
            if r.fed is not None:                   # in feed order (the EOS too: _store_reply)
                r.fed.append(tok)
            try:
                if req.thinking is not None:
                    req.thinking(tok)
                why = req.ctl.stop(tok) if req.ctl.active else (
                    "stop" if self.eng.stopping(tok) else None)
                if why == "stop":
                    if lpi is not None:             # the stop token's log p (no "tok" event)
                        req.stop_lp = (tok, *lpi)
                    self._finish(r, "stop")
                    continue
                if not ahead:
                    if req.guide is not None:
                        req.guide.consume(tok)
                    req.ctl.history.append(tok)
                # an ignored EOS is not text
                req.out.put(("tok", tok, why != "skip") if lpi is None
                            else ("tok", tok, why != "skip", lpi))
                r.n += 1
            except Exception as e:  # noqa: BLE001 — e.g. grammar rejected the token
                self._fail(r, e)
        if not ahead:
            self._compact()
            rows = self.rows
            with _no_sync():
                nxt = self._step(rows) if rows else []
        for r, y in zip(rows, nxt):
            if not r.done:
                # Reasoning budget: forces \n</think> as the next input (the
                # KV cache stays consistent)
                forced = r.req.thinking.pop_forced_token_id() if r.req.thinking is not None else None
                # golden (tools / tests) wins over the budget's forced token;
                # the budget's is still popped (its state moves on as if fed):
                # with both set the budget sees golden's tokens, not its own
                if r.req.golden is not None and r.n < len(r.req.golden):
                    forced = r.req.golden[r.n]          # r.n: tokens before this one
                r.y = y if forced is None else mx.array([forced], dtype=y.dtype)
                if r.lpk is not None:
                    lp, r.lp_next = r.lp_next, None
                    r.lp = (lp if forced is None or lp is None
                            else _forced_logprobs(lp, forced))
                r.first = False
        self.steps += 1
        self.failures = 0                       # consecutive failed decode steps
        if self.steps % 256 == 0:
            self.clear = True
        self._compact()

    # --- preemption (QWEN_PREEMPT) ------------------------------------------------
    def _preempt_rows(self):
        """Before a step: a row whose growth in it would pass the headroom
        (_regrow ahead=1: the cache gives way first, as in the step) is
        preempted instead of cut (_preempt); the others are re-accounted one
        step early (the same numbers _regrow would set in the step). Only
        rows _preemptable; the rest are cut in _step as before. Several rows
        preempted at once go back to the head of the queue in their batch
        (= admission) order."""
        ahead = not any(r.sync for r in self.rows)
        back = []
        for i, r in enumerate(self.rows):
            if r.pos + 1 >= r.tokens and self._preemptable(r) and not self._regrow(r, ahead=1):
                req = self._preempt(r, i, ahead)
                if req is not None:
                    # queued at once (behind the ones preempted before it), so
                    # that the next row's _regrow sees its entry as wanted
                    with self.cv:
                        self.waiting.insert(len(back), req)
                    back.append(req)
        if any(r.done for r in self.rows):
            self._compact()

    def _preemptable(self, row):
        """A row that can be resumed exactly where it stops: text, plain
        prefill (not DFlash), not in a cascade group (its KV is in the
        group's stack), its consumed tokens tracked (row.fed: the prompt cache
        on), prompt_cache not "read" / "off" (no store / no lookup), no
        prompt_logprobs (they belong to the first prefill), y a generated
        token (not `first`), preempted fewer than QWEN_PREEMPT times, and not
        alone (a lone row is never cut)."""
        req = row.req
        return (req.preempts < PREEMPT and not row.done and not row.first
                and row.fed is not None and row.group is None and not row.spec
                and not req.images and not _wants(req, "prompt_logprobs")
                and _cache_mode(req) == ""          # "off" would not look its entry up
                and all(hasattr(row.cache[j], "keys") for j in self.fa_layers)
                and sum(not r.done for r in self.rows) > 1)

    def _preempt(self, row, i, ahead):
        """Preempt row `i` before the step that would feed it its pending
        token y: read y and send it (as _iterate's read loop does: thinking
        budget, stop, grammar / bad-word history, logprobs), do with it what
        _logits does to a next input (penalty count, _Control's count: the
        resumed row's first input is `first`, skipped there), drop the row,
        hand its KV (prompt + generated[:-1] = everything but y) and its
        recurrent state after them to the prompt cache (_store_preempt: no
        copy) and -> its request, set to resume (the caller puts it back at
        the head of the queue): ids = prompt + generated, its lookup resumes
        at the entry's end, prefills nothing, and its first decode input is y
        at the same position. A row that stops on y (EOS, stop id) or reaches
        max_tokens ends normally instead (-> None). When the entry is not
        kept the request still resumes, from what the cache holds (a
        recompute)."""
        req = row.req
        tok = row.y.item()                          # waits for its own step only
        lpi = _read_logprobs(row.lp) if row.lp is not None else None
        row.fed.append(tok)
        try:
            if req.thinking is not None:
                req.thinking(tok)
            why = req.ctl.stop(tok) if req.ctl.active else (
                "stop" if self.eng.stopping(tok) else None)
            if why == "stop":
                if lpi is not None:
                    req.stop_lp = (tok, *lpi)
                self._finish(row, "stop")
                return None
            if not ahead:
                if req.guide is not None:
                    req.guide.consume(tok)
                req.ctl.history.append(tok)
            req.out.put(("tok", tok, why != "skip") if lpi is None
                        else ("tok", tok, why != "skip", lpi))
            row.n += 1
            if row.n == req.max_tokens:
                self._finish(row, "length")
                return None
            if row.pen.active:
                row.pen.add(row.y)
            if req.ctl.active:
                req.ctl.n += 1
        except Exception as e:  # noqa: BLE001 — e.g. grammar rejected the token
            self._fail(row, e)
            return None
        row.why = "preempted"
        fed, n = row.fed, row.n
        # The row ends here: its reservation goes first, its KV buffers become
        # the entry's (no copy, so nothing is allocated beside them: the
        # entry fits wherever the row did)
        row.done = True
        self.reserved -= row.reserved
        self.clear = True
        kept = None
        try:
            kept = self._store_preempt(row, self._reply_cache(row, i))
        except Exception:  # noqa: BLE001 — costs the entry only: the request resumes by recompute
            log.exception("id=%s preempt: storing the row failed", req.id)
        row.fed = row.entry = None                  # no reply store at its drop
        if req.orig is None:
            req.orig = req.ids
        req.ids = req.ids + fed
        req.max_tokens -= n
        req.carry = (row.sampler, row.pen)
        req.preempts += 1
        req.bounds = req.peeked = req.keep_since = None
        log.warning("id=%s preempted at %d tokens (%d of %d preemptions): the batch's KV would "
                    "pass its memory headroom (%.2f GiB reserved); %s, back at the head of the "
                    "queue (%d tokens, %d left to generate)", req.id, len(req.ids) - len(req.orig),
                    req.preempts, PREEMPT, self.reserved / GIB,
                    "its KV kept in the prompt cache" if kept is not None
                    else "its KV NOT kept (resumes by recompute)", len(req.ids), req.max_tokens)
        return req

    def _store_preempt(self, row, cache):
        """The preempted (already dropped) row as a prompt-cache entry: ids =
        prompt + generated[:-1] (the KV it holds), the end state its own;
        resumable at its end (`reply`, as a reply-KV entry: no slab end,
        decode bits). Its KV is the row's own buffers (_Seg.adopt: views,
        nothing copied; nothing writes them again), over [0, len): it
        inherits nothing and absorbs the row's prompt entry (its checkpoints
        move over) -- so _regrow having evicted that entry first costs
        nothing. The limit: the headroom beside the other rows (the row's
        reservation is released: its buffers are what the entry holds).
        Unlike a reply store: also with reasoning in the output (the resumed
        prompt holds it exactly); the end state's gathers are evaluated here
        (before the step; one host wait for a rare event) -> the entry or
        None (not kept)."""
        if cache is None:
            return None
        req, pc = row.req, self.eng._pcache
        ids = req.ids + row.fed[:-1]
        if len(ids) != row.pos or any(row.cache[j].offset != row.pos for j in self.fa_layers):
            log.warning("id=%s preempt: the row's cache does not hold its tokens", req.id)
            return None
        e = row.entry() if row.entry is not None else None
        clock = (e.used, e.seq) if e is not None and any(e is x for x in pc.entries) else None
        ends, protect = row.plan
        new, covered = pc.store(ids, cache, {}, None, 0, ends, protect,
                                limit=self.budget + self.prefill_reserve - self.reserved,
                                kv_start=self.kv["quantized_kv_start"], wanted=self._wanted,
                                reply=len(req.ids) - 1, tick=False, spare=self._parents(),
                                clock=clock, seg=_Seg.adopt(row.cache, len(ids)))
        if new is not None:
            try:
                mx.eval(new.arrays)                 # (the end state: gathers from the stack)
            except Exception:
                # no entry whose arrays cannot run: it goes, the absorbed ones come back
                pc.entries = [x for x in pc.entries if x is not new]
                pc.entries += [c for c in covered if not any(c is x for x in pc.entries)]
                pc._recount()
                raise
            log.info("id=%s preempt: stored %d tokens (the row's own buffers), %.1f MB (%d "
                     "entries, %.1f MB total) absorbed=%d", req.id, len(ids), new.nbytes / 1e6,
                     len(pc.entries), pc.bytes / 1e6, len(covered))
        return new
