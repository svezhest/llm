"""_Scheduler: prompt-cache lookup and the prefill slab loop."""

import bisect
import time
import weakref

import mlx.core as mx
import numpy as np

from .config import (CLEAR_CACHE, GIB, log, PREEMPT, PROMPT_CACHE, PROMPT_CACHE_CHAIN_FRESH_TOKENS,
                     PROMPT_CACHE_CHAIN_MARKS, PROMPT_CACHE_CHAIN_TAIL, PROMPT_CACHE_CKPTS,
                     PROMPT_CACHE_REPLY, PROMPT_CACHE_STRIDE)
from .pcache import _Inherit, _lcp, _recurrent_states, _thin, _with_tail
from .sizing import _slab
from .logprobs import _cache_mode, _score, _wants
from .request import _Own


def _vision_arrays(vision):
    """Scheduler side of Engine._vision_inputs: mlx-vlm prepare_inputs'
    conversion of the processor output (numpy -> mx.array, the rest as is)."""
    return {k: mx.array(x) if isinstance(x, np.ndarray) else x for k, x in vision.items()}


class _PrefillMixin:
    """Prefill: the prompt-cache lookup / checkpoint plan (_prompt_cache), the
    slab loop (_prefill) and the partial store."""

    # --- prefill ----------------------------------------------------------------
    def _prompt_cache(self, row):
        """Prompt-cache lookup / restore and the checkpoint plan (text only) ->
        (input ids to prefill, cache, reused tokens, slab cuts, on_ckpt, on_prefilled)."""
        from mlx_vlm.models.cache import make_prompt_cache
        req, eng, lm = row.req, self.eng, self.lm
        ids = req.ids
        use_cache = PROMPT_CACHE and not req.images
        bounds = self._bounds(req)
        ends = self._ends(req)
        look = self._lookup_ids(req)
        skip = use_cache and look is None
        # prompt_cache "read": a lookup as usual, nothing stored at the end
        # (no checkpoints taken, no reply KV, no partial store)
        keeps = use_cache and _cache_mode(req) != "read"
        if skip:
            # prompt_cache "off": a cold prefill (every position scored);
            # the closest entry's common prefix still places the divergence
            # checkpoint (no hit / miss counted, no clock tick)
            parent, reused, split = None, 0, eng._pcache._scan(ids, ends)[2]
        elif use_cache:
            parent, reused, split = eng._pcache.lookup(look, ends)
        else:
            parent, reused, split = None, 0, 0
        inherit = None
        # QWEN_PROMPT_CACHE_MOVE: the row holds the only copy of its KV (a hit
        # moves the entry's into it, nothing is copied at the prefill's end,
        # the row's buffers become the entry when it ends: _settle)
        move = keeps and self._move_on()
        taken = covers = False

        def room(keep=(), gone=None):
            """MOVE: an empty batch always admits, without an admission
            check, so the cache gives way to this row's reservation here
            (with one KV copy, the entry the row comes back as fits where
            the row did). `gone`: the entry about to move into the row (what
            only it holds is not counted: a segment another entry shares
            stays, beside the row's copy of it) -> False: still over."""
            pc, n = eng._pcache, 0

            def over():
                c = pc.charged()
                if gone is not None:
                    c -= pc.bytes - pc.bytes_of([e for e in pc.entries if e is not gone])
                return self.reserved + c > self.budget + self.prefill_reserve
            while over() and pc.evict(keep=keep, wanted=self._wanted, spill=False):
                n += 1
            if n:
                log.info("prompt cache: %d entries evicted for the row of id=%s (%.2f GiB left)",
                         n, req.id, pc.bytes / GIB)
            return not over()
        if parent is not None:
            # the first update's size: restore pre-sizes the buffer for it
            left = len(ids) - 1 - reused
            first = _slab(reused, left, set(bounds)) if left > 0 else 1
            # Only an entry this prompt's store absorbs anyway moves: the
            # conversation's previous turn (its ids, or a reply entry's
            # prompt part, a prefix of the prompt: _PrefixCache.store's
            # `covered`). A sibling's entry (a shared system prompt, an
            # edited branch) stays whole; the row copies its part, as
            # without the mode, and its store inherits those pieces.
            n = len(parent.ids)
            taken = move and (
                (n <= len(ids) - 1 and ids[:n] == parent.ids)
                or (parent.reply is not None and parent.reply <= len(ids) - 1
                    and ids[:parent.reply] == parent.ids[:parent.reply]))
            covers = taken
            if move and not taken and not room(keep=(parent,)):
                # ... unless the row's copy does not fit beside it even with
                # every other entry gone: then it moves too, and what it
                # held past the resume point (the other branch) is given up
                # instead of the memory
                taken = True
                log.info("id=%s prompt cache: the %d-token entry of another branch moves into "
                         "the row (no room for a copy beside it); its part past %d is dropped",
                         req.id, n, reused)
            inherit = _Inherit([] if taken else parent.cut(reused),
                               {p: s for p, s in parent.ckpts.items() if p <= reused},
                               parent.end if reused == len(parent.ids) else None,
                               parent.quantized)
            if taken:
                room(keep=(parent,), gone=parent)
                # (from here on `parent` is a husk outside the cache)
                try:
                    cache = eng._pcache.restore(parent, reused, first, move=True)
                except Exception:
                    log.warning("id=%s prompt cache: moving the %d-token entry into the row "
                                "failed; the entry is lost", req.id, n)
                    raise
            else:
                cache = eng._pcache.restore(parent, reused, first)
                row.parent = parent             # kept while this prefill runs (_regrow)
            input_ids = mx.array([ids[reused:]])
            # Text-only mRoPE continues from the cache offset with a zero
            # delta; a preceding vision request leaves its own delta here.
            lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int32)
            lm._position_ids = None
            log.info("id=%s prompt cache HIT: %d/%d tokens reused (common prefix %d), "
                     "prefilling %d%s", req.id, reused, len(ids), split, len(ids) - reused,
                     " (moved: the entry's KV is the row's now)" if taken else "")
        else:
            cache = make_prompt_cache(lm)
            input_ids = _vision_arrays(req.vision)["input_ids"] if req.images else mx.array([ids])
            if skip:
                log.info("id=%s prompt cache lookup skipped (prompt_cache=off; common prefix "
                         "%d, %d entries)", req.id, split, len(eng._pcache.entries))
            elif use_cache:
                log.info("id=%s prompt cache MISS (common prefix %d, %d entries)",
                         req.id, split, len(eng._pcache.entries))
        if not req.preempts:                    # Usage.cached_tokens: the first prefill's
            req.reused = reused
        row.cached = reused > 0
        lease = None
        if move:
            lease = row.own = _Own(parent if taken else None, reused,
                                   0 if taken else reused, len(ids) - 1)
            room(keep=(parent,) if parent is not None and not taken else ())
        keep, protect, ckpts, live = set(), set(), {}, set()
        if keeps:
            end = len(ids) - 1

            def below(c):                       # the closest slab end at or below c
                k = bisect.bisect_right(bounds, c)
                return bounds[k - 1] if k and bounds[k - 1] < end else None
            want = set(range(PROMPT_CACHE_STRIDE, end, PROMPT_CACHE_STRIDE))
            want.add(split)
            # The divergence point from the closest OTHER prompt: this
            # prompt's own cancelled attempts (partial entries that are a
            # prefix of it) do not count (their end would be the split), their
            # own divergence points carry over (`_Entry.diverge`: the
            # sibling may be gone by the retry).
            pc = eng._pcache
            # (MOVE: the entry just moved into the row is one of them still)
            ents = pc.entries if lease is None or lease.husk is None else [*pc.entries, lease.husk]
            own = [e for e in ents if e.partial and len(e.ids) < len(ids)
                   and ids[:len(e.ids)] == e.ids]
            other = split
            if own:
                other = max((min(_lcp(e.ids, ids), end) for e in ents
                             if not any(e is o for o in own)), default=0)
                want.add(other)
            diverge = {b for b in [below(other)] if b is not None}
            for o in own:
                diverge |= o.diverge
                # taken too (keep maps them through below: only this
                # prompt's slab ends) — a point past the cancel is not in
                # the partial entry, and the sibling may be gone
                want |= set(o.diverge)
            # Message boundaries are where prompts naturally diverge: the
            # end of the system/tools block (another conversation), the
            # last two messages (regenerate / edited turn) and the reply
            # (the next turn appends to it). Thinning keeps these (inherited
            # ones too); strides and older turns' boundaries go first.
            starts = [i for i, t in enumerate(ids) if t == eng._im_start]
            marks = starts[1:2] + starts[-3:]
            want.update(marks)
            protect = {b for b in map(below, marks) if b is not None}
            # A chain (the store absorbs this conversation's previous turn)
            # keeps only these: the end of the system block, the last
            # CHAIN_MARKS message starts, the strides in the last CHAIN_TAIL
            # tokens and in the first user message (see PROMPT_CACHE_CHAIN_*;
            # the end state is kept apart).
            chain = fresh = None
            # (MOVE: what the thinning during the prefill never drops)
            live = protect | diverge
            if PROMPT_CACHE_CHAIN_MARKS >= 0:
                last = starts[-PROMPT_CACHE_CHAIN_MARKS:] if PROMPT_CACHE_CHAIN_MARKS else []
                want.update(last)
                strides = [b for b in map(below, range(PROMPT_CACHE_STRIDE, end, PROMPT_CACHE_STRIDE))
                           if b is not None]
                fixed = {b for b in map(below, starts[1:2] + last) if b is not None}
                tail = {b for b in strides if b >= end - PROMPT_CACHE_CHAIN_TAIL}
                # the first user message (a shared document: another
                # conversation over it diverges anywhere inside) keeps its
                # strides too, unprotected (the cap thins them). Found by its
                # role token (without a system message starts[1] would be the
                # first reply); role unknown (tests): after the system block.
                user = getattr(eng, "_im_user", None)
                k = (next((j for j, s in enumerate(starts[:-1]) if ids[s + 1] == user), None)
                     if user is not None else 1 if len(starts) > 2 else None)
                u0, u1 = (starts[k], starts[k + 1]) if k is not None else (0, 0)
                doc = {b for b in strides if u0 <= b < u1}
                chain = (fixed | tail | doc, fixed)     # thinning drops the strides first
                live |= fixed | tail
                # a long fresh entry (absorbs no chain: a MISS, a new
                # episode) takes it too, with the divergence point from the
                # closest other entry (where a sibling would diverge again)
                # and the strides inside the system block (before the first
                # user message; unprotected like the document's: a new
                # session whose long system prompt differs near its end,
                # date / cwd / env, resumes at the last one). The length is
                # the whole prompt's, also for a cancelled prefill's prefix:
                # its retry then keeps the same set.
                if 0 < PROMPT_CACHE_CHAIN_FRESH_TOKENS <= end:
                    sysb = {b for b in strides if b < u0}
                    fresh = (chain[0] | diverge | sysb, fixed)
            # checkpoints only at slab ends
            keep = {b for b in map(below, want) if b is not None and b > reused}
            if move and chain is not None:
                # MOVE: the states stay with the row until it ends (the store
                # is deferred), so only those the store's policy can keep are
                # taken: the chain's when the entry it took makes this a
                # chain for sure, else (a long prompt) either policy's
                if taken and covers and (not parent.partial or parent.chained):
                    keep &= chain[0]
                elif fresh is not None:
                    keep &= chain[0] | fresh[0]
            start = self.kv["quantized_kv_start"]
            if self.kv["kv_bits"] and end >= start:
                # the entry will be quantized: nothing below the start is
                # resumable from it (the unquantized entry there is)
                keep = {b for b in keep if b >= start}
                protect = {b for b in protect if b >= start}
                # (chain: store drops its positions below the start itself)
            if (PROMPT_CACHE_REPLY or PREEMPT) and not req.images:
                # (QWEN_PREEMPT: the generated tokens make the resumed prompt)
                row.fed, row.plan = [], (set(bounds), protect)

        def checkpoint(pos, cache, tail=None):
            """`tail` (DFlash rows): a callable giving the drafter's context
            at `pos` (QWEN_DRAFT_TAIL), asked only for a position that is kept."""
            if pos in keep:
                ckpts[pos] = _with_tail(_recurrent_states(cache), tail() if tail else None)
                if lease is not None and len(ckpts) > PROMPT_CACHE_CKPTS:
                    # MOVE (long single-stream prompts): the states taken so
                    # far are thinned to the cap as the prefill goes, not
                    # only at the store — a 60K prefill held 29 of them (27B:
                    # 147 MiB each, 4.3 GiB beside 3.75 GiB of KV). The
                    # message starts, the chain's tail and the divergence
                    # point stay; which strides survive may differ from the
                    # store-time thinning (any subset is valid).
                    _thin(ckpts, PROMPT_CACHE_CKPTS, live)

        def snapshot(cache, upto=None, why=None, tail=None):
            """Store the entry for prompt[:-1] (the prefill's end), or with
            `upto` for the prefix prompt[:upto] a cancelled / failed prefill
            holds (`why`), when it prefilled at least a slab past `reused`.
            Either way the prefill ends here. `tail` (DFlash rows): the
            drafter's context at the stored end (QWEN_DRAFT_TAIL_END)."""
            nonlocal inherit
            if lease is not None:
                if upto is None:
                    # stored when the row ends (_settle): its buffers, no copy
                    lease.end = (_recurrent_states(cache), tail)
                    row.parent = None
                else:                           # the row ends here
                    self._settle_prefix(row, cache, upto, why, tail)
                return
            row.parent = None
            try:
                if not keeps:
                    if use_cache:
                        log.info("id=%s prompt cache: read-only (prompt_cache=read), nothing "
                                 "stored (reused=%d%s)", req.id, reused,
                                 "" if upto is None else f", {why} at {upto}/{len(ids)}")
                elif ((upto is None or upto > reused)
                        and all(getattr(c, "keys", 1) is not None for c in cache)):
                    store(cache, upto, why, tail)
            finally:
                # The caller's frame keeps this closure (a DFlash run: for its
                # whole decode): let go of the parent's pieces and of the
                # checkpoint states (the entry holds the ones it kept), so an
                # eviction frees them at once.
                inherit = None
                ckpts.clear()

        def store(cache, upto=None, why=None, tail=None, seg=None):
            """-> the entry, or None (not kept / failed). `seg`
            (QWEN_PROMPT_CACHE_MOVE): the entry's KV over [0, L), the dying
            row's own buffers; the limit then leaves the row's reservation
            out while it is still counted (the entry holds what it did)."""
            pc = eng._pcache
            L = len(ids) - 1 if upto is None else upto
            # what the entry claims: the row's KV holds exactly ids[:L] (a
            # prefix: the last completed slab's end, a _bounds cut; a failure
            # after the last slab holds the whole prompt[:-1]: no partial)
            partial = L < len(ids) - 1
            if upto is not None and (L not in ends or any(
                    c.offset != L for c in cache if not hasattr(c, "cache"))):
                log.warning("id=%s prompt cache: the %s prefill's prefix is not stored: the "
                            "cache does not hold a slab end (%d/%d)", req.id, why, L, len(ids))
                return None
            held = self.reserved
            if seg is not None and not row.done:
                held -= row.reserved
            try:
                e, covered = pc.store(ids[:L], cache, ckpts, inherit, reused, ends, protect,
                                      limit=self.budget + self.prefill_reserve - held,
                                      kv_start=self.kv["quantized_kv_start"], chain=chain,
                                      wanted=self._wanted, fresh=fresh, partial=partial,
                                      spare=self._parents(but=row), tail=tail, seg=seg,
                                      moved=lease.husk if lease is not None else None)
            except Exception:  # noqa: BLE001 — the prefill succeeded: the request goes on uncached
                log.exception("id=%s prompt cache store failed (the request goes on)", req.id)
                return None
            if e is None:
                if partial:
                    log.info("id=%s prompt cache: the %s prefill's prefix (%d/%d tokens) was "
                             "not kept", req.id, why, L, len(ids))
                return None
            row.cached = True
            if partial:                         # for its retry (see `diverge` above)
                e.diverge = frozenset(diverge)
            if row.fed is not None:
                row.entry = weakref.ref(e)
            # MB: the entry's buffers (shared ones included); "MB total": live
            # bytes, every buffer once; gap: tokens the cache held but could
            # not resume from (no checkpoint there); pieces: its KV chain;
            # states: its recurrent-state checkpoints + end state (in MB); a
            # cancelled / failed prefill's prefix: the same line (log parsers
            # match "prompt cache store: N tokens") + "(cancelled at L/len)"
            log.info("id=%s prompt cache store: %d tokens, %d checkpoints, %.1f MB "
                     "(%d entries, %.1f MB total) reused=%d gap=%d absorbed=%d pieces=%d "
                     "states=%.1f MB%s%s", req.id, len(e.ids), len(e.ckpts), e.nbytes / 1e6,
                     len(pc.entries), pc.bytes / 1e6, reused, split - reused, len(covered),
                     len(e.chain), e.state_bytes() / 1e6,
                     "" if upto is None else f" ({why} at {L}/{len(ids)})",
                     "" if seg is None else " (moved: the row's own buffers)")
            return e

        if lease is not None:
            def final(cache, upto=None, why=None, tail=None, seg=None):
                nonlocal inherit
                try:
                    return store(cache, upto, why, tail, seg)
                finally:                        # as snapshot: nothing of the prefill outlives it
                    inherit = None
                    ckpts.clear()
            lease.store = final

        return input_ids, cache, reused, set(bounds), checkpoint, snapshot

    def _slab_done(self, t):
        # Decode's time grows with the rows it serves: prefill is compute-
        # bound and B decode rows cost about one row's step, so a lone row
        # would take half the GPU from a waiting prompt for one token per
        # step, and with long prompts the batch would never form.
        s = 1 - (1 - self.share) * len(self.rows) / self.max_batch
        self.resume = time.time() + t * (1 - s) / s

    def _prefill(self, row):
        """Everything but the last prompt token goes through the cache in
        slabs (a generator: one slab per tick); the last one is the row's
        first decode input. A trimmed-down generate_step from mlx-vlm 0.6.2,
        kept local so that the prompt-cache snapshot is taken where the cache
        holds exactly prompt[:-1], and pre-tokenized vision inputs go in
        directly. When bumping mlx-vlm, diff generate/ar.py::generate_step
        against this + _iterate."""
        req, lm = row.req, self.lm
        seg = time.time()                       # start of this tick's part of the prefill
        input_ids, cache, start, cuts, on_ckpt, on_prefilled = self._prompt_cache(row)
        pixel_values = mask = None
        kwargs = {}
        if req.images:
            v = _vision_arrays(req.vision)
            req.vision = None
            pixel_values, mask = v.get("pixel_values"), v.get("attention_mask")
            kwargs = {k: x for k, x in v.items()
                      if k not in ("input_ids", "pixel_values", "attention_mask")}
        row.cache = cache
        emb = self.model.get_input_embeddings(input_ids, pixel_values, mask=mask, **kwargs)
        inputs_embeds = emb.inputs_embeds
        kwargs.update({k: v for k, v in emb.to_dict().items()
                       if k != "inputs_embeds" and v is not None})
        # mlx-vlm 0.7 returns mRoPE position_ids / rope_deltas (counted from 0
        # over the ids it was given) instead of setting them on the model.
        # Text: none of it (a prompt-cache hit's suffix would be placed at 0);
        # positions continue from the cache offset, as 0.6.2 did. Vision: the
        # positions are sliced per slab below, the delta goes to the row.
        vision_pos = kwargs.pop("position_ids", None)
        deltas = kwargs.pop("rope_deltas", None)
        if pixel_values is None:
            vision_pos = deltas = None
            lm._position_ids = None
        pos = start if pixel_values is None else int(cache[lm.model.fa_idx].offset)
        # prompt_logprobs (text only: Check refuses images): the slab's final
        # hidden states (return_hidden) through lm_head in chunks after the
        # slab's cache eval; row j's target is the token after it. Without it
        # the forward is exactly the one below (skip_logits, no hidden).
        scored = [] if _wants(req, "prompt_logprobs") and pixel_values is None else None
        # A text prefill that stops early (cancel; a failure outside the
        # forward) stores the prefix it holds, prompt[:pos] with pos the last
        # completed slab's end, once it prefilled a slab past `start`
        # (on_prefilled with upto: _prompt_cache's snapshot; vision stores
        # nothing). Like the full store it runs here, in _advance, never
        # between a decode step being queued and its tokens being read.
        intact, cancelled = True, False         # intact: the caches hold exactly prompt[:pos]
        try:
            while inputs_embeds.shape[1] > 1:
                if req.cancel.is_set():
                    cancelled = True
                    break
                t = time.time()
                n = _slab(pos, inputs_embeds.shape[1] - 1, cuts)
                # An image prompt's mRoPE positions (get_input_embeddings) are
                # sliced per slab: past the first slab the library would give
                # every token offset + delta, right only after the last image.
                if vision_pos is not None:
                    kwargs["position_ids"] = vision_pos[:, :, pos:pos + n]
                # skip_logits: no lm_head over the slab (248K vocab = ~12% of
                # the per-token FLOPs); only the cache matters here.
                intact = False                  # the forward moves every cache's offset
                if scored is None:
                    lm(input_ids[:, :n], inputs_embeds=inputs_embeds[:, :n], cache=cache,
                       skip_logits=True, **kwargs)
                else:
                    hid = lm(input_ids[:, :n], inputs_embeds=inputs_embeds[:, :n], cache=cache,
                             skip_logits=True, return_hidden=True, **kwargs).hidden_states[-1]
                self.quantize(cache)
                mx.eval([c.state for c in cache])
                pos += n
                intact = True
                if pos in cuts:
                    on_ckpt(pos, cache)
                if scored is not None:
                    scored += _score(lm, hid, input_ids[:, 1:n + 1])
                    hid = None
                inputs_embeds = inputs_embeds[:, n:]
                input_ids = input_ids[:, n:]
                if CLEAR_CACHE:
                    mx.clear_cache()
                if inputs_embeds.shape[1] > 1:      # the last slab joins at once
                    self._slab_done(time.time() - t)
                    yield
                    seg = time.time()
        except Exception:
            # (GeneratorExit is no Exception: a scheduler closing drops the
            # generator at its yield, nothing is stored then)
            if intact and (pos > start or row.own is not None) and pixel_values is None:
                self._store_prefix(row, on_prefilled, cache, pos, "failed")
            raise
        if cancelled:
            # (MOVE: also with nothing prefilled: the entry it took goes back)
            if (pos > start or row.own is not None) and pixel_values is None:
                self._store_prefix(row, on_prefilled, cache, pos, "cancelled")
            self._finish(row, "cancelled")
            return
        on_prefilled(cache)
        if scored is not None:
            # positions start + 1 .. len - 1 (the last slab scored the last token)
            assert len(scored) == len(req.ids) - 1 - start, (len(scored), len(req.ids), start)
            req.plp = (start + 1, scored)
        row.pos = pos
        if req.max_tokens == 0:
            # prefill only (prompt_logprobs): it ends here, never joins the
            # batch (no GDN restack, no cascade group, no reply KV); the
            # entry was stored above as a generation's would be
            self._finish(row, "length")
            self._settle(row)
            self._slab_done(time.time() - seg)
            return
        if deltas is not None:
            row.delta = int(deltas.reshape(-1)[0].item())
        row.y = mx.array(req.ids[-1:])
        self._join(row)
        # The last slab (a cache hit's only one) is charged too: else the
        # next waiting request's prefill would start in the next tick, and a
        # queue of short prefills would hold every running row (and the
        # first token of the rows that just joined) until it drains.
        self._slab_done(time.time() - seg)

    def _store_prefix(self, row, on_prefilled, cache, pos, why, tail=None):
        """The early-stop store (a cancelled / failed prefill's prefix): a
        failure here is logged and never changes how the request ends (a
        cancel stays a cancel, a failure keeps its own error). `tail`: a
        DFlash row's callable for the drafter's context at `pos`."""
        try:
            t = None
            try:
                t = tail() if tail else None
            except Exception:  # noqa: BLE001 — the prefix without the drafter's context
                log.exception("id=%s prompt cache: the drafter's context at %d was not taken",
                              row.req.id, pos)
            on_prefilled(cache, pos, why, **({} if t is None else {"tail": t}))
        except Exception:  # noqa: BLE001 — costs the prefix entry only
            log.exception("id=%s prompt cache: storing the %s prefill's prefix failed",
                          row.req.id, why)
