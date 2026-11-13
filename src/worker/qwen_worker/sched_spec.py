"""_Scheduler: DFlash speculative decoding."""

import copy

import mlx.core as mx

from . import config as cfg
from .config import CLEAR_CACHE, DRAFT_TAIL, DRAFT_TAIL_END, log, PROMPT_CACHE_REPLY, VERIFY_MR_HEAD
from .pcache import _own_states, _resume_tail, _tail_copy
from .sizing import _slab
from .kernels import _verify_head


def _budget_cut(think, new, stopping):
    """DFlash with a thinking budget: the index in `new` (a round's tokens:
    accepted drafts + the target's own) of the token after whose emission
    the budget forces the next one; None = it forces nothing in this block.
    On a copy of the criteria: the request's own sees emitted tokens only."""
    sim = copy.copy(think)
    for j, tok in enumerate(new):
        if stopping(tok):
            return None
        sim(tok)
        if sim.pop_forced_token_id() is not None:
            return j
    return None


class _SpecMixin:
    """DFlash speculative decoding of a lone row (_spec) and its reply store."""

    def _spec(self, row):
        """Greedy text-only prefill + DFlash speculative decode for a lone row
        (a generator: one slab or one round per tick).

        Prefill goes through the inner Qwen3_5Model directly, capturing the
        hidden states of the drafter's target layers. Only the last
        sliding_window-1 prompt positions are kept: that is all the drafter's
        rotating cache looks at. Each round then drafts a block (greedy),
        verifies it with one target forward (mlx-vlm 0.7: speculative_verify,
        the batch-invariant rows = singleton decode, inside a cache
        transaction), keeps the longest matching prefix and commits the
        transaction to it (the rest of the block leaves every cache).

        Sampling is exact ("sample-match"): every verify row is sampled from the
        target distribution for that position - the penalties over the output
        plus the drafts before the row, then temperature/min-p/top-k/top-p -
        and drafts are kept while they equal the target's own sample. Each
        emitted token is therefore a target sample given an accepted prefix,
        i.e. the same distribution as the plain loop; temperature only lowers
        the acceptance rate.

        Thinking budget (req.thinking): the criteria sees every emitted token,
        as in the plain loop. A round's block is cut at the token after which
        it forces the next one (_budget_cut; later drafts and the target's own
        token leave the cache with the commit), the forced tokens are emitted
        and fed by plain steps (close), then rounds go on. Same tokens as the
        plain loop with that budget (tests/test_scheduler.py).

        Demotion (another request waiting, max_batch > 1): after a round the
        target cache holds every token before the last emitted one, `b` —
        exactly a batch row whose next input is b, already emitted (`first`).
        """
        from mlx_vlm.speculative.cache_state import (abort_speculative_round,
                                                     commit_speculative_round)
        from mlx_vlm.speculative.common import (_dflash_block_total, _record_speculative_round,
                                                _speculative_walk)
        from mlx_vlm.speculative.dflash import _dflash_next_block_size
        req, lm, draft = row.req, self.lm, self.eng.draft
        wide = getattr(self.eng, "draft_wide", False)    # QWEN_DRAFT_FP32
        row.spec = True                         # also after a demotion: never grouped
        input_ids, cache, pos, cuts, on_ckpt, on_prefilled = self._prompt_cache(row)
        row.cache = cache
        layers = list(draft.config.target_layer_ids)
        win = draft.config.sliding_window - 1
        argmax = lambda logits: mx.argmax(logits, axis=-1)  # noqa: E731
        sampling, sampler, pen, max_tokens = req.sampling, row.sampler, row.pen, req.max_tokens
        dtype = input_ids.dtype

        def pick(logits, fed):
            """Target tokens for every row of `logits` (1, T, V); row i follows
            the output so far plus the drafts fed[1:i+1]."""
            logits = logits[0]
            if pen.active:
                extra = None
                if fed.shape[0] > 1:
                    hot = (fed[1:, None] == mx.arange(logits.shape[-1])[None]).astype(mx.float32)
                    extra = mx.concatenate([mx.zeros((1, logits.shape[-1])),
                                            mx.cumsum(hot, axis=0)])
                logits = pen(logits, extra)
            if sampling["temperature"] <= 0:
                return mx.argmax(logits, axis=-1)[None]
            return sampler(logits)[None]

        # The first lm() call below takes the delta branch (cache offset + 0);
        # with _rope_deltas None it would rotate the last prompt token as
        # position 0 (get_rope_index of a 1-token input).
        lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int32)
        lm._position_ids = None
        ctx = input_ids[:, :-1]
        # the drafter's context so far: what the entry the row resumed from
        # holds for the resume point (QWEN_DRAFT_TAIL), then every prefilled
        # position
        tail0 = _resume_tail(row.parent or (row.own.husk if row.own is not None else None), pos)
        hidden = [tail0] if tail0 is not None else []

        def tail(n=DRAFT_TAIL):
            """The drafter's context at the cache's end, for a state stored
            there: the last n positions' hidden states (a checkpoint:
            QWEN_DRAFT_TAIL)."""
            return _tail_copy(hidden, n, tail0)

        def tail_end():                         # an entry's end: QWEN_DRAFT_TAIL_END
            return tail(min(DRAFT_TAIL_END, win))

        # A prefill that stops early stores the prefix it holds, as _prefill
        # (the partial store: prompt[:pos], pos the last completed slab's end)
        start, intact, cancelled = pos, True, False
        try:
            while ctx.shape[1] > 0:
                if req.cancel.is_set():
                    cancelled = True
                    break
                n = _slab(pos, ctx.shape[1], cuts)
                sink = []
                intact = False                  # the forward moves every cache's offset
                lm.model(ctx[:, :n], cache=cache, capture_layer_ids=layers, hidden_sink=sink)
                h = mx.concatenate(sink, axis=-1)
                hidden.append(h)
                # drop the oldest chunk while the rest still covers the window
                # (until 2026-11-11 this tested hidden[:-1], all but the NEWEST:
                # a short last slab after a full one left the drafter that
                # short slab alone, e.g. slabs 2048 + 2048 + 40 -> 40 positions)
                while sum(t.shape[1] for t in hidden[1:]) >= win:
                    hidden.pop(0)
                mx.eval(h, [c.state for c in cache])
                pos += n
                intact = True
                if pos in cuts:
                    on_ckpt(pos, cache, tail)
                ctx = ctx[:, n:]
                if CLEAR_CACHE:
                    mx.clear_cache()
                if ctx.shape[1] > 0:
                    yield
        except Exception:
            if intact and (pos > start or row.own is not None):
                self._store_prefix(row, on_prefilled, cache, pos, "failed", tail_end)
            raise
        if cancelled:
            if pos > start or row.own is not None:
                self._store_prefix(row, on_prefilled, cache, pos, "cancelled", tail_end)
            self._finish(row, "cancelled")
            return
        on_prefilled(cache, tail=tail_end())
        on_ckpt = on_prefilled = None           # nothing of the prefill outlives it here
        out = lm(input_ids[:, -1:], cache=cache, capture_layer_ids=layers)
        hidden.append(mx.concatenate(out.hidden_states, axis=-1))
        hidden = mx.concatenate(hidden, axis=1)[:, -win:]
        first = pick(out.logits[:, -1:, :], input_ids[0, -1:])
        mx.eval(first, hidden)
        first = first.item()
        req.steps, req.rows_sum = req.steps + 1, req.rows_sum + 1
        # Reply KV (QWEN_PROMPT_CACHE_REPLY): row.fed takes every generated
        # token in order (as _iterate: the stop token too), `recent` the
        # hidden states of the last DRAFT_TAIL_END positions in the target
        # cache (the reply entry's end state carries them: the next turn's
        # drafter), in chunks
        gen = row.fed
        recent, keep = None, min(DRAFT_TAIL_END, win)
        if gen is not None and PROMPT_CACHE_REPLY and keep > 0:
            recent = [_tail_copy([hidden], keep)]

        def seen(h):
            """`h`: the hidden states of the positions the cache just took."""
            if recent is None:
                return
            recent.append(h)
            while sum(t.shape[1] for t in recent[1:]) >= keep:
                recent.pop(0)
            if len(recent) > 32:
                # a chunk is a view of its verify block's buffer (the whole
                # block stays allocated): one array of their own now and then
                recent[:] = [_tail_copy(recent, keep)]

        def emit(tok):
            if gen is not None:
                gen.append(tok)
            if self.eng.stopping(tok):
                return False
            if pen.active:
                pen.add(mx.array([tok]))
            req.out.put(("tok", tok, True))
            return True

        think = req.thinking                    # ThinkingBudgetCriteria or None

        def budget(tok):
            """The token the thinking budget forces next, after `tok` was
            emitted (the plain loop: req.thinking(tok), then
            pop_forced_token_id at the next step)."""
            if think is None:
                return None
            think(tok)
            return think.pop_forced_token_id()

        def close(last, cached, f):
            """The budget forces `f` after `last` (emitted; `cached`: already
            in the target cache). Plain steps feed what is not cached yet
            (their hidden states are the drafter's next context), the forced
            tokens are emitted; again while the budget goes on forcing
            (\n, then </think>). -> (the last emitted token, not in the cache:
            the next round's anchor; the fed tokens' hidden states; "stop" or
            None)."""
            nonlocal n
            ctx = []
            while f is not None and n < max_tokens:
                if not cached:
                    o = lm(mx.array([[last]], dtype=dtype), cache=cache, capture_layer_ids=layers)
                    h = mx.concatenate(o.hidden_states, axis=-1)
                    mx.eval(h, [c.state for c in cache])
                    ctx.append(h)
                    seen(h)
                    req.steps, req.rows_sum = req.steps + 1, req.rows_sum + 1
                last, cached = f, False
                if not emit(last):
                    return last, ctx, "stop"
                n += 1
                f = budget(last)
            return last, ctx, None

        finish, n, acc, drafted = "length", 0, 0, 0
        if max_tokens > 0 and not emit(first):
            finish = "stop"
        elif max_tokens > 0:
            n = 1
            # mlx-vlm 0.7.6 `_dflash_rounds`, with `pick` as the target sampler.
            block_total = _dflash_block_total(draft, cfg.DRAFT_BLOCK or None)
            draft_cache = draft.reset(self.model)
            if VERIFY_MR_HEAD and getattr(draft, "lm_head", None) is not None:
                draft.lm_head = _verify_head(draft.lm_head)
            if wide and getattr(draft, "lm_head", None) is not None:
                # the target's head (bound at reset) takes the target's dtype
                draft.lm_head = (lambda x, head=draft.lm_head, hd=self._hidden_dtype():
                                 head(x.astype(hd)))
            b, why = first, None
            f = budget(first)
            if f is not None:                   # a budget the first token already passes
                b, ctx, why = close(first, False, f)
                if ctx:
                    hidden = mx.concatenate([hidden] + ctx, axis=1)[:, -win:]
                if why:
                    finish = why
            while n < max_tokens and not why:
                yield
                if req.cancel.is_set():
                    finish = "cancelled"
                    break
                row.pos = cache[lm.model.fa_idx].offset
                self._regrow(row)
                head = self.waiting[0] if self.waiting else None
                if (head is not None and not head.cancel.is_set() and self.max_batch > 1
                        and self._fits(head, 1)):
                    # only for a request that is admitted next to this row:
                    # otherwise it waits anyway and DFlash's speed-up is lost
                    log.info("dflash: %d tokens, accept rate %.0f%% (%d/%d drafts); "
                             "demoted to a batch row (another request is waiting)",
                             n, 100.0 * acc / max(1, drafted), acc, drafted)
                    row.y, row.first, row.n = mx.array([b]), True, n
                    self._join(row)
                    return
                bs = _dflash_next_block_size(draft, block_total, max_tokens - n + 1)
                if bs <= 1:
                    break
                drafts = draft.draft_block(b, hidden.astype(mx.float32) if wide else hidden,
                                           draft_cache, bs, argmax, dtype)
                mx.async_eval(drafts)
                fed = mx.concatenate([mx.array([[b]], dtype=dtype), drafts], axis=1)
                txn = None
                try:
                    # the verify forward opens a cache transaction (out.gdn_states)
                    # that must be committed every round, or aborted
                    out = lm(fed, cache=cache, capture_layer_ids=layers, speculative_verify=True)
                    txn = out.gdn_states
                    hidden = mx.concatenate(out.hidden_states, axis=-1)
                    targets = pick(out.logits, fed[0])
                    mx.async_eval(targets, hidden)
                    accepted, new = _speculative_walk(drafts, targets, max_tokens - n)
                    hit, cut = accepted, None       # hit: the drafter's score (statistics)
                    if think is not None:
                        cut = _budget_cut(think, new, self.eng.stopping)
                        if cut is not None and cut < accepted:
                            # the budget forces the token after the draft
                            # new[cut]: the later drafts and the target's own
                            # token leave the cache with the commit
                            accepted, new = cut + 1, new[:cut + 1]
                    commit_speculative_round(lm, cache, txn, accepted, bs)
                    if accepted + 1 < bs:       # the kept state is a view of the block's
                        _own_states(cache)
                except BaseException:
                    abort_speculative_round(txn)
                    raise
                _record_speculative_round(draft, hit, bs - 1)
                acc, drafted = acc + hit, drafted + bs - 1
                req.steps, req.rows_sum = req.steps + 1, req.rows_sum + 1
                if accepted < bs - 1:           # the positions the commit kept
                    hidden = hidden[:, : accepted + 1, :]
                seen(hidden)
                stopped, f = False, None
                for j, tok in enumerate(new):
                    if not emit(tok):
                        finish, stopped = "stop", True
                        break
                    n += 1
                    f = budget(tok)
                    if f is not None and j != len(new) - 1:
                        raise RuntimeError("dflash: the thinking budget forced a token inside "
                                           "a committed block")
                if stopped or not new:
                    break
                b = new[-1]
                if f is not None:
                    # new[-1] is a draft the commit kept (the block was cut
                    # there) or the target's own token (not in the cache yet)
                    b, ctx, why = close(b, cut is not None and cut < hit, f)
                    if ctx:
                        hidden = mx.concatenate([hidden] + ctx, axis=1)
                    if why:
                        finish = why
                        break
                if n % 256 < len(new) and CLEAR_CACHE:
                    mx.clear_cache()
        if max_tokens > 1:
            log.info("dflash: %d tokens, accept rate %.0f%% (%d/%d drafts)",
                     n, 100.0 * acc / max(1, drafted), acc, drafted)
            if drafted >= 64 and not acc:
                log.warning("dflash: none of %d drafts accepted: the drafter's output is "
                            "unusable (NaN from an fp16 overflow? try QWEN_DRAFT_FP32=1)", drafted)
        self._finish(row, finish)
        self._spec_reply(row, recent)

    def _spec_reply(self, row, recent):
        """The reply KV of a DFlash row that ended in its rounds (stop /
        length): as a batch row's (_reply_cache + _store_reply; row.fed = its
        generated tokens, of which the cache holds the first pos - len(prompt):
        a stop token among accepted drafts with more drafts after it is
        "a fed token never read", nothing stored), evaluated here (no decode
        step is in flight). `recent`: chunks of the last positions' hidden
        states, the end state's drafter context (QWEN_DRAFT_TAIL_END). Never
        raises: the request already ended."""
        if row.fed is None or not PROMPT_CACHE_REPLY or row.why not in ("stop", "length"):
            self._settle(row)                   # (MOVE: the prompt entry, the row's buffers)
            row.fed = None
            return
        pc = self.eng._pcache
        new, covered = None, []
        try:
            from mlx_vlm.models.cache import ArraysCache
            idx = mx.array([0], dtype=mx.int32)
            cache = []
            for c in row.cache:
                if hasattr(c, "cache"):         # a gather: never a view of a verify block's states
                    n = ArraysCache(size=len(c.cache))
                    n.cache = [a[idx] if a is not None else None for a in c.cache]
                    cache.append(n)
                else:
                    cache.append(c)
            row.pos = row.cache[self.lm.model.fa_idx].offset
            k = row.pos - len(row.req.ids)
            tail = (_tail_copy(recent, DRAFT_TAIL_END)
                    if recent and 1 <= k <= len(row.fed) else None)
            if row.own is not None:             # MOVE: prompt + reply, the row's buffers
                self._settle(row, cache, tail)
                return
            new, covered = self._store_reply(row, cache, tail)
            if new is not None:
                mx.eval(new.arrays)
        except Exception:  # noqa: BLE001
            if new is not None:                 # as a failed store: it goes, the absorbed come back
                pc.entries = [x for x in pc.entries if x is not new]
                pc.entries += [c for c in covered if not any(c is x for x in pc.entries)]
                pc.evictions += 1
                pc._recount()
            log.exception("id=%s prompt cache reply store failed (DFlash row)", row.req.id)
            self._settle(row)                   # (MOVE: at least the prompt entry)
