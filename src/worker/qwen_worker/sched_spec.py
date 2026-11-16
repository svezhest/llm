"""_Scheduler: DFlash speculative decoding."""

import copy
import time

import mlx.core as mx

from . import config as cfg
from .config import (CLEAR_CACHE, DRAFT_TAIL, DRAFT_TAIL_END, log, LOOKUP, LOOKUP_BACKOFF,
                     LOOKUP_BURST, LOOKUP_K, LOOKUP_K0, LOOKUP_MIN, LOOKUP_N, LOOKUP_NMAX,
                     PROMPT_CACHE_REPLY, VERIFY_MR_HEAD)
from .lookup import _DraftCtx, _Gate, _Lookup, _lstat
from .pcache import _own_states, _resume_tail, _tail_copy
from .sizing import _slab
from .kernels import _verify_head
from .mtp import CLOCK, Chooser, Shadows


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

        Quantized KV (QWEN_KV_BITS): the cache is quantized where the plain
        loop quantizes it (self.quantize after every prefill slab, after the
        first target call and every committed round / plain step: offset >=
        quantized_kv_start), and while it is still fp16 no verify block
        crosses the start (cut at it; at start - 1 a plain one-token step),
        so every position attends the same KV format as in the plain loop.
        The verify forward attends a QuantizedKVCache through the library's
        one-query quantized calls (the QWEN_VERIFY_ATT kernel is fp16-only);
        its commit / abort move the cache's offset only.

        Prompt lookup (QWEN_LOOKUP, lookup.py): a round whose draft is the
        continuation of an earlier occurrence of the text's suffix, verified
        the same way; with a drafter ("1") instead of its block when there is
        a match (the drafter's next block then gets every position since its
        last one: _DraftCtx), or ("splice") in place of its block when they
        agree on the first token. Without a drafter (eng.draft None) the
        rounds where nothing matches are plain steps (_plain_burst).
        """
        from mlx_vlm.speculative.cache_state import (abort_speculative_round,
                                                     commit_speculative_round)
        from mlx_vlm.speculative.common import (_dflash_block_total, _record_speculative_round,
                                                _speculative_walk)
        from mlx_vlm.speculative.dflash import _dflash_next_block_size
        req, lm, draft = row.req, self.lm, self.eng.draft
        # the MTP head (mtp.py): alone (draft None) or beside DFlash (hybrid);
        # the hidden-state stream below is then [DFlash's layers.., the final]
        mt = getattr(self.eng, "mtp", None)
        H = lm.args.hidden_size
        wide = getattr(self.eng, "draft_wide", False) and draft is not None  # QWEN_DRAFT_FP32
        row.spec = True                         # also after a demotion: never grouped
        input_ids, cache, pos, cuts, on_ckpt, on_prefilled = self._prompt_cache(row)
        row.cache = cache
        self._kvq_mark(row)
        # no drafter: lookup rounds + plain steps, no hidden states captured
        layers = (list(draft.config.target_layer_ids) if draft is not None
                  else [] if mt is not None else None)
        fin = mt is not None                    # return_hidden: the final hidden state too
        width = (len(layers or ()) + fin) * H
        dfl = (lambda h: h[..., :len(layers) * H]) if fin else (lambda h: h)  # noqa: E731
        kv_start = self.kv["quantized_kv_start"]
        fa = lm.model.fa_idx

        def requant():
            """As the plain loop after a slab / a step (_prefill, _logits)."""
            self.quantize(cache)
            self._kvq_mark(row)

        def fp16_left():
            """Quantized KV: positions left before quantized_kv_start while
            the cache is still fp16 (a block must stay within them), else None."""
            c = cache[fa]
            if not self.kv["kv_bits"] or hasattr(c, "bits"):
                return None
            return kv_start - c.offset
        win = (draft.config.sliding_window - 1 if draft is not None
               else max(1, DRAFT_TAIL_END) if mt is not None else 0)
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

        def one(logits):
            """A plain step's token from its (1, V) logits, as the plain loop
            draws it (_sample: penalties, then the request's sampler)."""
            return sampler(pen(logits) if pen.active else logits)

        def _wide(h):
            return h.astype(mx.float32) if wide else h

        # The first lm() call below takes the delta branch (cache offset + 0);
        # with _rope_deltas None it would rotate the last prompt token as
        # position 0 (get_rope_index of a 1-token input).
        lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int32)
        lm._position_ids = None
        ctx = input_ids[:, :-1]
        # the drafter's context so far: what the entry the row resumed from
        # holds for the resume point (QWEN_DRAFT_TAIL), then every prefilled
        # position
        tail0 = (_resume_tail(row.parent or (row.own.husk if row.own is not None else None), pos)
                 if layers is not None else None)
        if tail0 is not None and tail0.shape[-1] != width:
            tail0 = None                        # stored by another drafter setup
        hidden = [tail0] if tail0 is not None else []
        if mt is not None:                      # the head's context: the tail, then the prompt
            t0 = 0 if tail0 is None else tail0.shape[1]
            mt.start(pos - t0)
            if t0:
                mt.push(tail0[..., -H:])
                mt.emit_ids(req.ids[pos - t0 + 1:pos + 1])

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
                if layers is None:
                    lm.model(ctx[:, :n], cache=cache)
                    requant()
                    h = None
                else:
                    last = lm.model(ctx[:, :n], cache=cache, capture_layer_ids=layers, hidden_sink=sink)
                    requant()
                    h = mx.concatenate(sink + [last] if fin else sink, axis=-1)
                    if mt is not None:
                        mt.push(last)
                        mt.emit_ids(req.ids[pos + 1:pos + n + 1])
                        mt.flush()
                    hidden.append(h)
                    # drop the oldest chunk while the rest still covers the window
                    # (until 2026-11-11 this tested hidden[:-1], all but the NEWEST:
                    # a short last slab after a full one left the drafter that
                    # short slab alone, e.g. slabs 2048 + 2048 + 40 -> 40 positions)
                    while sum(t.shape[1] for t in hidden[1:]) >= win:
                        hidden.pop(0)
                mx.eval(h, [c.state for c in cache], mt.state() if mt is not None else [])
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
        if layers is not None:
            out = lm(input_ids[:, -1:], cache=cache, capture_layer_ids=layers, return_hidden=fin)
        else:                                   # as the plain loop's first step (_step)
            out = lm(input_ids[:, -1:], cache=cache, position_ids=mx.array(
                [[cache[lm.model.fa_idx].offset]], dtype=mx.int32))
        requant()
        if layers is not None:
            hidden.append(mx.concatenate(out.hidden_states, axis=-1))
            if mt is not None:
                mt.push(out.hidden_states[-1])
            hidden = mx.concatenate(hidden, axis=1)[:, -win:]
            first = pick(out.logits[:, -1:, :], input_ids[0, -1:])
        else:
            hidden = None
            first = one(out.logits[:, -1, :])
        mx.eval(first, hidden)
        first = first.item()
        on = LOOKUP in ("1", "splice") or (LOOKUP == "draft" and (draft is not None or mt is not None))
        lk = _Lookup(req.ids, LOOKUP_N, LOOKUP_NMAX) if on and max_tokens > 1 else None
        req.steps, req.rows_sum = req.steps + 1, req.rows_sum + 1
        # Reply KV (QWEN_PROMPT_CACHE_REPLY): row.fed takes every generated
        # token in order (as _iterate: the stop token too), `recent` the
        # hidden states of the last DRAFT_TAIL_END positions in the target
        # cache (the reply entry's end state carries them: the next turn's
        # drafter), in chunks
        gen = row.fed
        recent, keep = None, min(DRAFT_TAIL_END, win)
        if gen is not None and PROMPT_CACHE_REPLY and keep > 0 and hidden is not None:
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

        shadow = None                           # mtp.Shadows: with a Chooser

        def emit(tok, fed=False):
            """`fed`: a plain step on `tok` is already queued (pburst: it
            counted the token for the penalties)."""
            if mt is not None:
                mt.emit(tok)
            if shadow is not None:
                shadow.emit(tok)
            if gen is not None:
                gen.append(tok)
            if self.eng.stopping(tok):
                return False
            if lk is not None:
                lk.add(tok)
            if pen.active and not fed:
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
                    o = lm(mx.array([[last]], dtype=dtype), cache=cache, capture_layer_ids=layers,
                           return_hidden=fin)
                    requant()
                    h = mx.concatenate(o.hidden_states, axis=-1) if layers is not None else None
                    if mt is not None:
                        mt.push(o.hidden_states[-1])
                    mx.eval(h, [c.state for c in cache])
                    if h is not None:
                        ctx.append(h)
                        seen(h)
                    req.steps, req.rows_sum = req.steps + 1, req.rows_sum + 1
                last, cached = f, False
                if not emit(last):
                    return last, ctx, "stop"
                n += 1
                f = budget(last)
            return last, ctx, None

        def step1(tok):
            """One plain target step for `tok` (quantized KV: the last fp16
            position, quantized_kv_start - 1, which no verify block may
            cross) -> (its hidden states, [the target's token])."""
            o = lm(mx.array([[tok]], dtype=dtype), cache=cache, capture_layer_ids=layers,
                   return_hidden=fin)
            requant()
            h = mx.concatenate(o.hidden_states, axis=-1) if layers is not None else None
            t = pick(o.logits[:, -1:, :], mx.array([tok], dtype=dtype))
            mx.eval(t, h, [c.state for c in cache])
            return h, [t.item()]

        finish, n, acc, drafted = "length", 0, 0, 0
        lstat = [0, 0, 0, 0]                    # lookup rounds, drafts, accepted; plain steps
        drounds = 0                             # the drafter's rounds
        pick_kind = None                        # hybrid / plain arm: mtp.Chooser

        trace = [] if cfg.SPEC_TRACE else None  # QWEN_SPEC_TRACE: one tuple per round
        used = {}                               # arm -> [rounds, tokens] (spec_stats["arms"])

        def use(arm, tokens):
            u = used.setdefault(arm, [0, 0])
            u[0], u[1] = u[0] + 1, u[1] + tokens

        def stats():
            req.spec_stats = dict(dflash_rounds=drounds, dflash_drafted=drafted, dflash_accepted=acc,
                                  lookup_rounds=lstat[0], lookup_drafted=lstat[1],
                                  lookup_accepted=lstat[2], plain_steps=lstat[3],
                                  arms={k: list(v) for k, v in used.items()})
            if pick_kind is not None:
                req.spec_stats.update(shadows=dict(pick_kind.shadowed), switches=pick_kind.switches)
            if trace is not None:
                req.spec_stats["trace"] = trace
        fa = lm.model.fa_idx

        def step(x):
            """A plain decode step (no drafter, nothing to look up): the plain
            loop's call (_step at B = 1) and sampler (_sample) -> the next
            token, lazy (1,). `x`: the input token, an int or a lazy (1,)."""
            inp = mx.array([[x]], dtype=dtype) if isinstance(x, int) else x.reshape(1, 1)
            o = lm(inp, cache=cache, position_ids=mx.array([[cache[fa].offset]], dtype=mx.int32))
            return one(o.logits[:, -1, :])

        def burst(b):
            """Plain steps from `b` (emitted, not in the cache), the next one
            queued before a token is read (the plain loop's step-ahead; each
            token is read only after the step on it went out, so with
            penalties — they need it — step by step), until the lookup has a
            match, QWEN_LOOKUP_BURST steps or the end. -> (the last emitted
            token, not in the cache; stopped)."""
            nonlocal n
            ahead, k = not pen.active, 1
            y = step(b)
            mx.async_eval(y)
            while True:
                nxt = None
                if ahead and k < LOOKUP_BURST and n + 1 < max_tokens:
                    nxt, k = step(y), k + 1
                    mx.async_eval(nxt)
                tok = y.item()
                req.steps, req.rows_sum, lstat[3] = req.steps + 1, req.rows_sum + 1, lstat[3] + 1
                if not emit(tok):
                    return tok, True
                n += 1
                if n >= max_tokens:
                    return tok, False
                if nxt is None:
                    return tok, False
                if propose(max_tokens - n - 1) is not None:
                    # the queued step feeds tok: its token is the anchor
                    tok = nxt.item()
                    req.steps, req.rows_sum, lstat[3] = req.steps + 1, req.rows_sum + 1, lstat[3] + 1
                    if not emit(tok):
                        return tok, True
                    n += 1
                    return tok, False
                y = nxt

        def pstep(x):
            """A plain step with a drafter loaded: the plain loop's forward and
            draw (_step at B = 1, _logits, _sample) with the drafters' hidden
            states captured. `x`: the input token, an int (emitted) or the
            lazy (1,) token of the step before (counted for the penalties
            here, as _logits). -> (the next token, lazy (1,); the hidden
            states of x's position)."""
            lazy = not isinstance(x, int)
            o = lm(x.reshape(1, 1) if lazy else mx.array([[x]], dtype=dtype), cache=cache,
                   capture_layer_ids=layers, return_hidden=fin)
            if lazy and pen.active:
                pen.add(x)
            requant()
            return one(o.logits[:, -1, :]), mx.concatenate(o.hidden_states, axis=-1)

        def pburst(b, steps, before=None):
            """The plain arm (mtp.Chooser): up to `steps` plain steps from `b`
            (emitted, not in the cache) with the plain loop's step-ahead - the
            next step is queued before a token is read, so the GPU never waits
            for the host inside the burst. `before()`: called once the first
            step is queued (the shadow drafts are read there). A thinking
            budget ends the burst at the token after which it forces one. ->
            (the last emitted token; stopped; the steps' tokens; the token the
            budget forces next or None; is the last token in the cache: only
            with a forced one, when the step queued ahead already fed it)."""
            nonlocal n
            y, h = pstep(b)
            mx.async_eval(y, h)
            if before is not None:
                before()
            toks = []
            while True:
                nxt = None
                if len(toks) + 1 < steps and n + 1 < max_tokens:
                    nxt = pstep(y)
                    mx.async_eval(*nxt)
                tok = y.item()
                toks.append(tok)
                req.steps, req.rows_sum, lstat[3] = req.steps + 1, req.rows_sum + 1, lstat[3] + 1
                if mt is not None:              # the position this step fed: both drafters' context
                    mt.push(h[..., -H:])
                seen(h)
                dctx.add(dfl(h))
                if not emit(tok, nxt is not None):
                    return tok, True, toks, None, False
                n += 1
                f = budget(tok)
                if f is not None and nxt is not None:
                    # the step queued ahead fed tok (its position is in the
                    # cache and the drafters' context); its own token is not read
                    h = nxt[1]
                    req.steps, req.rows_sum = req.steps + 1, req.rows_sum + 1
                    if mt is not None:
                        mt.push(h[..., -H:])
                    seen(h)
                    dctx.add(dfl(h))
                    return tok, False, toks, f, True
                if nxt is None or f is not None:
                    return tok, False, toks, f, False
                y, h = nxt

        gate = _Gate(LOOKUP_K0, LOOKUP_K, LOOKUP_MIN, LOOKUP_NMAX, LOOKUP_BACKOFF)

        def propose(left):
            """The lookup's drafts after the last emitted token (<= `left`), or
            None: no match the gate takes (_Gate)."""
            if lk is None or left < 2:
                return None
            m, p = lk.propose(gate.size(left))
            return p if gate.ok(m, p) else None

        if max_tokens > 0 and not emit(first):
            finish = "stop"
        elif max_tokens > 0:
            n = 1
            # mlx-vlm 0.7.6 `_dflash_rounds`, with `pick` as the target sampler.
            draft_cache = None
            if draft is not None:
                block_total = _dflash_block_total(draft, cfg.DRAFT_BLOCK or None)
                draft_cache = draft.reset(self.model)
                if VERIFY_MR_HEAD and getattr(draft, "lm_head", None) is not None:
                    draft.lm_head = _verify_head(draft.lm_head)
                if wide and getattr(draft, "lm_head", None) is not None:
                    # the target's head (bound at reset) takes the target's dtype
                    draft.lm_head = (lambda x, head=draft.lm_head, hd=self._hidden_dtype():
                                     head(x.astype(hd)))
            # the drafter's context: every position the target cache took since its last block
            dctx = _DraftCtx(win, draft_cache)
            if hidden is not None:
                dctx.add(dfl(hidden))
            # QWEN_DRAFT_HYBRID: both drafters, one per round (mtp.Chooser);
            # QWEN_SPEC_PLAIN: plain steps are an arm too (a drafter whose
            # rounds give fewer tokens per ms than a plain step loses to it,
            # re-probed now and then). DFlash's context collects the positions
            # of the other arms' rounds (dctx).
            arms = [x for x, on in (("dflash", draft is not None), ("mtp", mt is not None),
                                    ("plain", cfg.SPEC_PLAIN and (draft is not None or mt is not None)))
                    if on]
            # the request's class for the round times: the context's size, the
            # sampler on the verified rows, a thinking budget (plain rounds are
            # then single synchronous steps)
            clock = CLOCK.at((sum(len(req.ids) >= x for x in (6144, 24576, 65536)),
                              sampling["temperature"] > 0, think is not None))
            if mt is not None:
                mt.clock = clock
            pick_kind = Chooser(arms, mt, clock) if len(arms) > 1 else None
            shadow = Shadows() if pick_kind is not None else None
            force = cfg.DRAFT_HYBRID if cfg.DRAFT_HYBRID in arms else None
            kind = arms[0] if arms else None
            # a round's wall time (start to the next round's start) goes to CLOCK
            # when it is the arm's steady cost: `timed` = (kind, start, steps),
            # None after a switch (the drafter catches up), shadow drafts, a cut
            # block, the request's first round (DFlash: the prompt window)
            timed, prev = None, None
            b, why = first, None
            f = budget(first)
            if f is not None:                   # a budget the first token already passes
                b, ctx, why = close(first, False, f)
                for h in ctx:
                    dctx.add(dfl(h))
                if why:
                    finish = why
            while n < max_tokens and not why:
                yield
                if req.cancel.is_set():
                    finish = "cancelled"
                    break
                t_round, fed_ctx = time.perf_counter(), 0
                if timed is not None:
                    clock.add(timed[0], 1000.0 * (t_round - timed[1]) / timed[2])
                    timed = None
                row.pos = cache[fa].offset
                self._regrow(row)
                head = self.waiting[0] if self.waiting else None
                if (head is not None and not head.cancel.is_set() and self.max_batch > 1
                        and self._fits(head, 1)):
                    # only for a request that is admitted next to this row:
                    # otherwise it waits anyway and DFlash's speed-up is lost
                    log.info("dflash: %d tokens, accept rate %.0f%% (%d/%d drafts)%s; "
                             "demoted to a batch row (another request is waiting)",
                             n, 100.0 * acc / max(1, drafted), acc, drafted, _lstat(lstat))
                    row.y, row.first, row.n = mx.array([b]), True, n
                    stats()
                    self._join(row)
                    return
                look = propose(max_tokens - n)
                if look is None and draft is None and mt is None:
                    b, stopped = burst(b)
                    if stopped:
                        finish = "stop"
                        break
                    if n % 256 < LOOKUP_BURST and CLEAR_CACHE:
                        mx.clear_cache()
                    continue
                also = ()                       # the arms drafted without a verify
                if look is None and pick_kind is not None:
                    kind = pick_kind.pick(force, 1000.0 * t_round)
                    also = pick_kind.shadows()
                    if cfg.SPEC_SHADOW == "all":
                        also = [x for x in arms if x not in ("plain", kind)]
                if look is None and kind == "plain":
                    bs, drafts = 1, None        # the plain arm (pburst, below)
                elif look is None:
                    if kind == "mtp":
                        bs = min(mt.next_k() + 1, max_tokens - n + 1)
                    else:
                        bs = _dflash_next_block_size(draft, block_total, max_tokens - n + 1)
                    if bs <= 1:
                        break
                    left = fp16_left()
                    if left is not None and 0 < left < bs:
                        bs = left               # the block ends at the last fp16 position
                    drafts = None
                    if bs > 1 and kind == "mtp":
                        drafts = mt.draft(bs - 1)
                        mx.async_eval(drafts)
                    elif bs > 1:
                        hctx = dctx.take(draft_cache)
                        fed_ctx = hctx.shape[1]
                        drafts = draft.draft_block(b, _wide(hctx), draft_cache, bs, argmax, dtype)
                        mx.async_eval(drafts)
                elif draft is not None and LOOKUP == "splice":
                    # the drafter's block, unless the continuation agrees on its first draft
                    bs = _dflash_next_block_size(draft, block_total, max_tokens - n + 1)
                    d = draft.draft_block(b, _wide(dctx.take(draft_cache)), draft_cache, bs,
                                          argmax, dtype)
                    mx.eval(d)
                    if d.shape[1] and d[0, 0].item() == look[0]:
                        drafts = mx.array([look], dtype=dtype)
                        bs = len(look) + 1
                    else:
                        drafts, look = d, None
                        if bs <= 1:
                            break
                else:
                    drafts = mx.array([look], dtype=dtype)
                    bs = len(look) + 1
                sdrafts = []
                for x in also:
                    # (each drafter's state moves as in a round of its own: the
                    # head's chained entries leave at once, DFlash's cache takes
                    # the context it would take at its next block)
                    if x == "mtp":
                        ks = min(mt.KMAX if cfg.SPEC_SHADOW == "all" else mt.next_k(), max_tokens - n)
                        if ks < 1:
                            continue
                        d = mt.draft(ks)
                        mt.cancel()
                    else:
                        if max_tokens - n + 1 < block_total:
                            continue
                        hctx = dctx.take(draft_cache)
                        fed_ctx = hctx.shape[1]
                        d = draft.draft_block(b, _wide(hctx), draft_cache, block_total, argmax, dtype)
                    mx.async_eval(d)
                    sdrafts.append((x, d))
                if look is None and kind == "plain" and pick_kind is not None:
                    # the plain arm: a burst of plain steps with step-ahead
                    def before():
                        for x, d in sdrafts:
                            shadow.add(x, d.reshape(-1).tolist())
                    b, stopped, new, f, cached = pburst(
                        b, 1 if cfg.SPEC_SHADOW == "all" else pick_kind.steps(), before)
                    k = len(new)
                    use("plain", k)
                    pick_kind.done(0, 0, True, [x for x, _ in sdrafts], steps=k)
                    for x, sa, sd in shadow.settle():
                        pick_kind.sample(x, sa, sd)
                    if prev == "plain" and not sdrafts:
                        timed = ("plain", t_round, k)
                    prev = "plain"
                    if trace is not None:
                        trace.append(("plain", 1, 0, round(t_round, 4),
                                      round(1000.0 * (time.perf_counter() - t_round), 2), fed_ctx,
                                      [(x, d.reshape(-1).tolist()) for x, d in sdrafts], new))
                    if stopped:
                        finish = "stop"
                        break
                    if f is not None:
                        timed = None
                        b, ctx, why = close(b, cached, f)
                        for h in ctx:
                            dctx.add(dfl(h))
                        if why:
                            finish = why
                            break
                    if n % 256 < k and CLEAR_CACHE:
                        mx.clear_cache()
                    continue
                left = fp16_left()
                if left is not None and 0 < left < bs:
                    bs = left                   # the block ends at the last fp16 position
                    if look is not None:
                        look = look[:bs - 1]
                    drafts = drafts[:, :bs - 1] if bs > 1 else None
                if bs == 1:
                    hidden, new = step1(b)
                    accepted = hit = 0
                    cut = None
                else:
                    fed = mx.concatenate([mx.array([[b]], dtype=dtype), drafts], axis=1)
                    txn = None
                    try:
                        # the verify forward opens a cache transaction (out.gdn_states)
                        # that must be committed every round, or aborted
                        out = lm(fed, cache=cache, capture_layer_ids=layers if layers is not None else [],
                                 speculative_verify=True, return_hidden=fin)
                        txn = out.gdn_states
                        hidden = mx.concatenate(out.hidden_states, axis=-1) if layers is not None else None
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
                    requant()                   # (outside the transaction: offset >= start)
                if look is None and bs > 1:
                    if kind == "mtp":
                        mt.record(hit, bs - 1)
                    else:
                        _record_speculative_round(draft, hit, bs - 1)
                    acc, drafted, drounds = acc + hit, drafted + bs - 1, drounds + 1
                elif look is not None and bs > 1:   # (not the drafter's: its adaptive block, stats)
                    lstat[0], lstat[1], lstat[2] = lstat[0] + 1, lstat[1] + bs - 1, lstat[2] + hit
                    gate.after(hit, bs - 1)
                full = kind != "dflash" or bs == block_total
                if look is None and pick_kind is not None:
                    pick_kind.done(hit, bs - 1, full, [x for x, _ in sdrafts])
                    sdrafts = [(x, d.reshape(-1).tolist()) for x, d in sdrafts]
                    for x, d in sdrafts:
                        shadow.add(x, d)
                arm = kind if look is None else None
                use(arm or "lookup", len(new))
                if arm is not None and arm == prev and not sdrafts and full and (bs > 1 or arm == "plain"):
                    timed = (("mtp", bs - 1) if arm == "mtp" else arm, t_round, 1)
                prev = arm
                if trace is not None:
                    trace.append(("lookup" if look is not None else kind, bs, hit, round(t_round, 4),
                                  round(1000.0 * (time.perf_counter() - t_round), 2), fed_ctx,
                                  sdrafts, list(new)))
                req.steps, req.rows_sum = req.steps + 1, req.rows_sum + 1
                if hidden is not None:
                    if accepted < bs - 1:       # the positions the commit kept
                        hidden = hidden[:, : accepted + 1, :]
                    if mt is not None:          # (other arms' rounds: no chained entries)
                        mt.round_end(accepted, hidden[..., -H:])
                    seen(hidden)
                    dctx.add(dfl(hidden))
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
                if shadow is not None:
                    for x, sa, sd in shadow.settle():
                        pick_kind.sample(x, sa, sd)
                if stopped or not new:
                    break
                b = new[-1]
                if f is not None:
                    # new[-1] is a draft the commit kept (the block was cut
                    # there) or the target's own token (not in the cache yet)
                    timed = None
                    b, ctx, why = close(b, cut is not None and cut < hit, f)
                    for h in ctx:
                        dctx.add(dfl(h))
                    if why:
                        finish = why
                        break
                if n % 256 < len(new) and CLEAR_CACHE:
                    mx.clear_cache()
        if max_tokens > 1:
            log.info("%s: %d tokens, accept rate %.0f%% (%d/%d drafts)%s%s",
                     "hybrid" if pick_kind is not None else ("mtp" if mt is not None else "dflash"),
                     n, 100.0 * acc / max(1, drafted), acc, drafted, _lstat(lstat),
                     f"; {pick_kind.summary()}" if pick_kind is not None else "")
            if drafted >= 64 and not acc:
                log.warning("dflash: none of %d drafts accepted: the drafter's output is "
                            "unusable (NaN from an fp16 overflow? try QWEN_DRAFT_FP32=1)", drafted)
        stats()
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
