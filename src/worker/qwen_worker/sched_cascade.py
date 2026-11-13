"""_Scheduler: cascade groups."""

import mlx.core as mx
import numpy as np
from mlx_vlm.models.cache import KVCache

import cascade
import cascade_mma

from . import config as cfg
from .config import (CASCADE, CASCADE_ALIAS_SLACK, CASCADE_CHECK, CASCADE_CHUNK, CASCADE_KERNEL,
                     CASCADE_MIN_PREFIX, CASCADE_MIN_ROWS, CASCADE_SLACK, CASCADE_STEP, GIB, log,
                     PROMPT_CACHE)
from .pcache import _lcp
from .rows import _GroupRef, _PrefixGroup, _StepPlan


class _CascadeMixin:
    """Cascade groups (QWEN_CASCADE): formation, joins, tidy, the step plan."""

    def _compact(self):
        self._drop_done()
        # a group whose tidy failed ended its rows: drop them, and tidy the
        # other groups again for the smaller batch (each pass removes a group)
        while self.groups and self._cascade_tidy():
            self._drop_done()

    # --- cascade groups (QWEN_CASCADE; plan Part B) ---------------------------
    def _cascade_on(self, want):
        return bool(want and CASCADE_MIN_ROWS and CASCADE_MIN_PREFIX and PROMPT_CACHE
                    and self.max_batch >= 3 and self.kv["kv_bits"] is None
                    and not cfg.BATCH_INVARIANT)

    def cascade_state(self):
        """For the load log: on, or why not."""
        if self.cascade:
            kern = ""                   # the default (fp32 prefix) says nothing more
            if CASCADE_KERNEL:
                kern = ("; MMA kernel over the fp16 prefix (QWEN_CASCADE_KERNEL)" if self.cascade_kernel
                        else "; fp32 prefix (QWEN_CASCADE_KERNEL=1 ignored: the model is not fp16)")
            return (f"on: >= {CASCADE_MIN_ROWS} rows sharing >= "
                    f"{CASCADE_MIN_PREFIX} prompt tokens{', check' if CASCADE_CHECK else ''}{kern}")
        why = [w for w, bad in (("QWEN_CASCADE=0", not CASCADE),
                                ("min rows / prefix 0", not (CASCADE_MIN_ROWS and CASCADE_MIN_PREFIX)),
                                ("no prompt cache", not PROMPT_CACHE),
                                ("max_batch < 3", self.max_batch < 3),
                                ("quantized KV", self.kv["kv_bits"] is not None),
                                ("QWEN_BATCH_INVARIANT=1", cfg.BATCH_INVARIANT)) if bad]
        return "off (" + ", ".join(why) + ")"

    def set_cascade(self, on):
        """Tools (step_prof's A/B). Call it on the scheduler thread only: it
        builds and evaluates MLX arrays. Off: no new groups; the running ones
        are released in stages by _cascade_tidy (one member per group and
        decode step, each only when its contiguous cache fits the headroom
        beside the live group: 16 x 32K members released at once would
        materialize every fp16 cache beside the live fp32 group), so groups
        can outlive the switch while memory is short. On: the formation rule
        for every running row (as if each just joined)."""
        self.cascade = self._cascade_on(on)
        if not self.cascade:
            if self.groups:
                self._compact()
            return
        for r in list(self.rows):
            if r.group is None and not r.done:
                self._cascade_join(r)

    def _cascade_count(self):
        self.n_groups = len(self.groups)
        self.n_grouped = sum(len(g.rows) for g in self.groups)

    def _unlink(self, g):
        """Break group g's references for good: every member's `group`, its
        FA cache entries (_GroupRefs: a finished or failed row's only), and
        g.rows. Without it rows <-> group is a cycle, and the group's fp32
        arrays would live until Python's cyclic GC (generation 2: minutes in
        this process) while `reserved` already let them go."""
        for r in g.rows:
            self._unlink_row(r, g)
        g.rows = []

    def _unlink_row(self, r, g):
        if r.group is g:
            r.group = None
        if r.cache is not None:
            for i in self.fa_layers:
                if isinstance(r.cache[i], _GroupRef):
                    r.cache[i] = None

    def _drop_groups(self):
        """Every group goes (a failed step, _close): the rows are ended."""
        for g in self.groups:
            self._unlink(g)
        self.groups = []
        self._cascade_count()

    def _member_bytes(self):
        """A grouped row's own reservation: its recurrent state (twice, as
        _bytes) + the same slack; its KV is charged to the group."""
        return 2 * self.gdn_row + (8 << 20)

    def _group_suffix(self, P, rows, tokens=None):
        """Suffix tokens a group is charged for per member: the longest member
        reservation past P (every stack slot is cap wide), in stack steps.
        `tokens`: {row: tokens} overrides (_regrow's what-if, _group_wins'
        real lengths)."""
        s = max((tokens or {}).get(r, r.tokens) for r in rows) - P
        return -(-max(s, 1) // CASCADE_STEP) * CASCADE_STEP

    def _group_bytes(self, P, rows, tokens=None):
        """A group's charge (plan B.5): the fp32 prefix once (padded to whole
        chunks) + the fp32 stack at the longest member reservation (+5%, as
        _bytes) + the step's score transients (prefix + suffix scores and
        probabilities, 8 B per q head, row and key per FA layer, +
        CASCADE_SLACK) + one layer's old + new stack (take / add / grow copy
        layer by layer: review cascade_b1 finding 1). Conservative: about one
        layer's scores are alive at a time, and the copy is transient.

        Floor (review cascade_b2b3_fixes C): never less than the prefix +
        stacks + 2 contiguous fp16 rows of P + the charged suffix (+5%, as
        _bytes) -- the forced release of a group's last 2 members is
        unchecked and builds their caches while the group is alive, so the
        charge must already hold it. It also makes a group of a few rows
        over a long prefix lose to their own caches (_group_wins: 3 x 32K /
        64K / 96K do; 8 or 16 rows still win by far)."""
        n, s = len(rows), self._group_suffix(P, rows, tokens)
        kv32 = 2 * self.kv_fp16                     # bytes per token, all FA layers, fp32
        n_fa = len(self.fa_layers)
        if self.cascade_kernel:
            # QWEN_CASCADE_KERNEL: the fp16 prefix (the donor's aliased buffer,
            # <= P + slack slots, or a P + 1 copy); no prefix scores, the
            # kernel's partials instead (one bound per layer, as the scores)
            hq = self.lm.args.num_attention_heads
            prefix = (P + CASCADE_ALIAS_SLACK) * self.kv_fp16
            scores = n_fa * (8 * hq * n * s + cascade_mma.partial_bytes(
                P, n, hq, self.lm.args.head_dim)) + CASCADE_SLACK
            moves = 2 * n * s * kv32 // n_fa
            arrays = prefix + int(n * s * kv32 * 1.05)
            return arrays + max(scores + moves, 2 * int((P + s) * self.kv_fp16 * 1.05))
        ppad = -(-P // CASCADE_CHUNK) * CASCADE_CHUNK
        scores = n_fa * 8 * self.lm.args.num_attention_heads * n * (ppad + s) + CASCADE_SLACK
        moves = 2 * n * s * kv32 // n_fa
        arrays = ppad * kv32 + int(n * s * kv32 * 1.05)
        return arrays + max(scores + moves, 2 * int((P + s) * self.kv_fp16 * 1.05))

    def _group_wins(self, P, rows):
        """The memory rule (review cascade_b2b3 finding 2), at the members'
        REAL lengths (each row's KV after the next step, pos + 1): the group's
        arrays and transients (_group_bytes at those lengths) take no more
        than the members' own fp16 caches would (_bytes at those lengths, the
        recurrent state aside). At the reservations instead (the default
        4096 generation tokens per row, the fp32 stack charged at 2x), an 8K
        prefix would group only from ~6 rows and a 4K one never; the charge
        stays reservation-based, and _charge_fits bounds what grouping adds
        to it. 9B, ~150-token suffixes, with _group_bytes' floor (2
        contiguous rows): 3 rows never win (3 x 32K / 64K / 96K: 4.2 / 8.3 /
        12.4 GiB vs 3.2 / 6.3 / 9.5), 4 rows only from a 32K prefix; 5..16
        rows win at 4K / 8K / 32K. A group stops winning at a real suffix of
        257 / 1281 / 2049 tokens for 4K x 5 / 8 / 16 rows, 1025 / 3073 /
        4353 for 8K, 4865 / 13057 / 18177 for 32K (the P // 2 leave comes
        first from 8 rows at 8K). Under 3 rows never.
        QWEN_CASCADE_KERNEL (fp16 prefix, P + 1024 slots charged): 4 rows win
        at 4K / 8K / 32K, 3 rows not at 32K (3.20 vs 3.16 GiB) but barely
        from 64K (6.30 vs 6.31, 96K 9.40 vs 9.46: the floor dominates); a
        group stops winning at a real suffix of 769 / 1537 / 2049 tokens for
        4K x 5 / 8 / 16 rows, 2049 / 3585 / 4609 for 8K, 9217 / 16129 / 19713
        for 32K."""
        if len(rows) < 3:
            return False
        real = {r: r.pos + 1 for r in rows}
        own = sum(self._bytes(n) for n in real.values()) - len(rows) * self._member_bytes()
        return self._group_bytes(P, rows, real) <= own

    def _charge_fits(self, delta):
        """The budget guard of a formation / join: grouping may raise the
        reservations (the fp32 stack at the reserved length can outweigh the
        fp16 caches it replaces) only while they + the prompt cache's live
        bytes (inside the budget, as for admissions: _fits) stay within the
        budget. Refused rather than evicting: grouping is optional, and the
        shared system prompt's entry is worth more as a future hit."""
        return delta <= 0 or self.reserved + delta + self.eng._pcache.charged() <= self.budget

    def _cascadable(self, row):
        """A running row that may be grouped: a text-only plain prefill (never
        vision, never DFlash) that resumed from the prompt cache or stored an
        entry, ungrouped, with plain fp16 KVCaches at its position."""
        if (row.done or row.group is not None or not row.cached or row.spec
                or row.req.images or row.delta):
            return False
        return all(type(row.cache[i]) is KVCache and row.cache[i].offset == row.pos
                   for i in self.fa_layers)

    def _fits_prefix(self, row, P):
        """P can be `row`'s group prefix (its ids [0, P) checked by the caller):
        inside the prefilled prompt and a slab end of it -- so its KV [0, P)
        are the bits every row whose prompt shares [0, P) with P a slab end
        holds (_bounds depends only on the tokens before an end; hit == cold)
        -- and its own suffix not comparable to the prefix (the suffix part
        is the slower one)."""
        return (P <= len(row.req.ids) - 1 and row.pos - P <= P // 2
                and P in self._ends(row.req))

    def _same_prefix(self, a, b, P):
        """QWEN_CASCADE_CHECK at formation: rows a and b hold the same KV
        [0, P) bit for bit in every FA layer (evaluates)."""
        for i in self.fa_layers:
            x, y = a.cache[i], b.cache[i]
            for u, v in ((x.keys, y.keys), (x.values, y.values)):
                if u.dtype != v.dtype or not mx.array_equal(u[..., :P, :], v[..., :P, :]).item():
                    return False
        return True

    def _cascade_join(self, row):
        """`row` just joined the batch: add it to the deepest group whose
        prefix it shares, else form a new group around it when, at the
        deepest slab end P >= QWEN_CASCADE_MIN_PREFIX of its prompt, at least
        QWEN_CASCADE_MIN_ROWS running rows (it included; earlier rows are
        converted) share ids [0, P) with P a slab end of theirs. Either only
        when the memory rule holds for the members (_group_wins, real
        lengths) and the charge moved to the group keeps `reserved` + the
        prompt cache within the budget (_charge_fits). A row refused by a
        group forms no second group on that group's prefix (it would hold the
        same prefix twice); it stays ungrouped."""
        if not self.cascade or len(self.rows) < 3 or not self._cascadable(row):
            return
        ids, member = row.req.ids, self._member_bytes()
        for g in sorted(self.groups, key=lambda g: -g.P):
            if ids[:g.P] != g.ids or not self._fits_prefix(row, g.P):
                continue
            rows = g.rows + [row]
            if not self._group_wins(g.P, rows):
                continue
            delta = self._group_bytes(g.P, rows) - g.reserved - (row.reserved - member)
            if not self._charge_fits(delta):
                log.debug("cascade: id=%s not added to group P=%d: +%.2f GiB reserved would "
                          "pass the budget", row.req.id, g.P, delta / GIB)
                continue
            if CASCADE_CHECK and not g.pg.prefix_equal(row.cache):
                log.warning("cascade: id=%s shares the tokens of group P=%d but not its KV "
                            "bits: not grouped", row.req.id, g.P)
                return
            self._group_add(g, row)
            return
        need = max(3, CASCADE_MIN_ROWS)
        others = [r for r in self.rows if r is not row and self._cascadable(r)]
        if len(others) + 1 < need:
            return
        lcp = [(_lcp(ids, r.req.ids), r) for r in others]
        if sum(n >= CASCADE_MIN_PREFIX for n, _ in lcp) + 1 < need:
            return
        ends = sorted((p for p in self._ends(row.req)
                       if p >= CASCADE_MIN_PREFIX and self._fits_prefix(row, p)), reverse=True)
        for P in ends:
            if any(g.P == P and g.ids == ids[:P] for g in self.groups):
                continue                # that group refused it (or would have)
            cands = [row] + [r for n, r in lcp if n >= P and self._fits_prefix(r, P)]
            if len(cands) < need:
                continue
            if CASCADE_CHECK:
                bad = [r for r in cands[1:] if not self._same_prefix(row, r, P)]
                for r in bad:
                    log.warning("cascade: id=%s shares %d tokens with id=%s but not their KV "
                                "bits: not grouped", r.req.id, P, row.req.id)
                cands = [r for r in cands if r not in bad]
                if len(cands) < need:
                    continue
            if not self._group_wins(P, cands):
                continue
            delta = self._group_bytes(P, cands) - sum(r.reserved - member for r in cands)
            if not self._charge_fits(delta):
                log.debug("cascade: no group P=%d of %d rows: +%.2f GiB reserved would pass "
                          "the budget", P, len(cands), delta / GIB)
                continue
            self._group_build(P, cands)
            return

    def _group_build(self, P, rows):
        """A new group over `rows`: [0, P) from rows[0]'s own cache (the
        joining row's; the same bits as every member's), evaluated; the
        members in batch order (so the step plan of a wave's group is one
        slice: no gather, no output take); their FA caches replaced by
        _GroupRefs (their buffers are freed); the charges moved from the rows
        to the group."""
        source = rows[0]
        if self.cascade_kernel:
            # the prefix aliases the donor's buffer: the one with the fewest
            # slots past P. Candidates hold the same [0, P) bits only for a
            # prefill-built [0, P) (_bounds, hit == cold); a row resumed through
            # a reply entry (QWEN_PROMPT_CACHE_REPLY, the default) holds decode-
            # kernel bits up to the reply's end, which may differ. As with
            # rows[0] in the fp32 path, the group then attends the donor's bits
            # for every member; QWEN_CASCADE_CHECK=1 refuses such members.
            fa = self.fa_layers[0]
            source = min(rows, key=lambda r: r.cache[fa].keys.shape[2])
        at = {id(r): b for b, r in enumerate(self.rows)}
        rows = sorted(rows, key=lambda r: at[id(r)])
        pg = cascade.PrefixGroup.build(source.cache, [r.cache for r in rows], P, self.fa_layers,
                                       chunk=CASCADE_CHUNK, step=CASCADE_STEP,
                                       kernel=self.cascade_kernel, slack=CASCADE_ALIAS_SLACK)
        g = _PrefixGroup(pg, source.req.ids[:P], rows)
        before = self.reserved
        g.reserved = self._group_bytes(P, g.rows)
        member = self._member_bytes()
        for r in g.rows:
            for i in self.fa_layers:
                r.cache[i] = _GroupRef(r)
            r.group = g
            self.reserved += member - r.reserved
            r.reserved = member
        self.reserved += g.reserved
        self.groups.append(g)
        self._cascade_count()
        log.info("cascade: group formed: P=%d, %d rows (%s), %.1f MB (%s), "
                 "charged %.2f GiB; reserved %.2f -> %.2f GiB",
                 P, len(g.rows), ",".join(r.req.id for r in g.rows), pg.nbytes / 1e6,
                 ("prefix fp16 " + ("aliased" if pg.aliased else "copied") + " + stacks fp32")
                 if pg.kernel else "prefix + stacks, fp32",
                 g.reserved / GIB, before / GIB, self.reserved / GIB)

    def _group_add(self, g, row):
        """`row` joins group g: its suffix is copied onto the stacks (the
        S-token copies evaluated, so the row's own buffers go at once; the
        stack concat lazy: the next step's appends consume it), its FA caches
        go. The charge is computed first and pg.add is atomic: an exception
        leaves the group and the row as they were."""
        charge = self._group_bytes(g.P, g.rows + [row])
        g.pg.add(row.cache)
        g.rows.append(row)
        old, g.reserved = g.reserved, charge
        for i in self.fa_layers:
            row.cache[i] = _GroupRef(row)
        row.group = g
        member = self._member_bytes()
        self.reserved += charge - old + member - row.reserved
        row.reserved = member
        self._cascade_count()
        log.info("cascade: id=%s joined group P=%d (%d rows, charged %.2f GiB)",
                 row.req.id, g.P, len(g.rows), g.reserved / GIB)

    def _cascade_tidy(self):
        """Group membership after rows ended or grew (every _compact).
        Finished members leave (take; their FA entries cleared). Then, per
        group (_tidy_choice):
          * forced, whatever it costs: no live member left, or < 3 live
            members (which covers a batch down to <= 2 rows: B <= 2 stays
            today's bit-exact path) -- at most 2 rows are released, and their
            fp16 caches replace the group's fp32 prefix; unchecked: its peak
            (2 contiguous rows beside the live group) is inside the group's
            charge (_group_bytes' floor);
          * optional, one member per group and decode step (fewer than 3
            left go with it), checked with the group alive: reserved + the released
            rows' contiguous caches - their member charges + the prompt
            cache's live bytes <= budget + prefill reserve, crediting nothing
            for the group's shrink (release evaluates the new caches while
            the group's arrays are alive, and take is lazy) -- else they stay
            grouped (always correct) and the next tidy retries. A member
            whose own suffix passed P // 2 leaves, the longest first (a speed
            rule; a leaver costs ~(P - S) fp16 tokens more than its stack
            slot); when the memory rule fails at the members' real lengths
            (_group_wins), and with cascade switched off (set_cascade), the
            cheapest member is released (a memory dissolve is staged too: the
            rows left may win again and stay grouped).

        A released row gets contiguous KVCaches (cascade release): [0, P) are
        bit for bit the prefix it had (fp16 -> fp32 -> fp16 is exact); the
        suffix holds what it decoded while grouped -- tokens produced under
        the group's numerics, so the row's continuation equals a never-
        grouped row's given that cache, not a never-grouped run's tokens.
        Every release of a group is built (and evaluated) before any row is
        touched. `release` evaluates: from _iterate's _compact it waits for
        the step in flight, so that step loses its overlap (the GPU idles
        while the host builds the next one) -- once per leave / dissolve
        event, accepted; from _join's it is free (the slab's eval drained
        the stream). A group whose tidy raises (an OOM in a release) is
        dropped and its live members fail -- their KV is only in the group
        -- while the rest of the batch, and a request joining it, go on.
        -> True when that happened (the caller drops the ended rows)."""
        failed = False
        for g in list(self.groups):
            try:
                self._tidy_group(g)
            except Exception as e:  # noqa: BLE001 — this group's rows, not the batch's
                log.exception("cascade: group P=%d: tidy failed, its %d rows fail", g.P,
                              sum(not r.done for r in g.rows))
                for r in g.rows:
                    self._fail(r, e)
                if g in self.groups:        # not yet dropped by its own dissolve
                    self.reserved -= g.reserved
                    self.groups.remove(g)
                self._unlink(g)
                failed = True
        self._cascade_count()
        return failed

    def _tidy_choice(self, g, live):
        """-> (released rows, staying rows, why or None). `why` names the
        reason of a release (None when nobody is released); the group is
        dissolved when nobody stays."""
        # < 3 live members also covers a batch of <= 2 rows (B <= 2 stays
        # today's bit-exact path): 3 members are 3 rows of the batch
        forced = ("finished" if not live else
                  "%d members" % len(live) if len(live) < 3 else None)
        if forced:
            return live, [], forced
        if getattr(g, "released_at", -1) == self.steps:
            # one optional release per decode step: the last one's take (the
            # group's old stacks) is freed only by the step that consumes it
            return [], live, None
        if not self.cascade:                # switched off: in stages, the cheapest first
            r, why = min(live, key=lambda r: r.tokens), "switched off"
        elif not self._group_wins(g.P, live):   # memory: likewise
            r, why = min(live, key=lambda r: r.tokens), "memory"
        else:                               # the speed rule: the longest suffix first
            long = [r for r in live if r.pos - g.P > g.P // 2]
            if not long:
                return [], live, None
            r, why = max(long, key=lambda r: r.pos), "suffix past P/2"
        out, stay = [r], [x for x in live if x is not r]
        if len(stay) < 3:                   # < 3 staying members go with it
            out, stay = live, []
        # checked with the group alive (release builds the contiguous caches
        # beside it; take is lazy: nothing of the group goes before the next
        # step), so the group's shrink is not credited (review cascade_b2b3_fixes A)
        need = sum(self._bytes(x.tokens) - x.reserved for x in out)
        if self.reserved + need + self.eng._pcache.charged() > self.budget + self.prefill_reserve:
            return [], live, None
        return out, stay, why

    def _tidy_group(self, g):
        live = [r for r in g.rows if not r.done]
        out, stay, why = self._tidy_choice(g, live)
        if len(stay) == len(g.rows):
            return                      # nobody finished, nobody released
        index = {id(r): j for j, r in enumerate(g.rows)}
        caches = {id(r): g.pg.release(index[id(r)]) for r in out}
        if stay:
            g.pg.take([index[id(r)] for r in stay])
        for r in g.rows:
            if r.done:
                self._unlink_row(r, g)
        for r in out:
            for i, c in caches[id(r)].items():
                r.cache[i] = c
            r.group = None
            need = self._bytes(r.tokens)
            self.reserved += need - r.reserved
            r.reserved = need
        old, g.rows = g.reserved, stay
        if out and stay:
            g.released_at = self.steps
        if stay:
            g.reserved = self._group_bytes(g.P, stay)
            self.reserved += g.reserved - old
            if out:
                log.info("cascade: group P=%d: %s left (%s), %d rows stay", g.P,
                         ",".join(r.req.id for r in out), why, len(stay))
        else:
            self.reserved -= old
            self.groups.remove(g)
            log.info("cascade: group P=%d dissolved (%s): %d rows released, %.2f GiB "
                     "charge dropped (reserved %.2f GiB)", g.P, why, len(out),
                     old / GIB, self.reserved / GIB)

    def _plan(self, rows):
        """This step's split of `rows` into ungrouped rows and groups. Any
        disagreement between the groups and the batch (a member missing, a
        grouped row in no group, the group's lengths vs the rows' positions)
        is a bookkeeping bug: RuntimeError fails the step's rows."""
        where = {id(r): b for b, r in enumerate(rows)}
        groups, order = [], []
        for g in self.groups:
            idx = [where.get(id(r)) for r in g.rows]
            if None in idx or any(r.group is not g for r in g.rows):
                raise RuntimeError(f"cascade: group P={g.P} out of step with the batch")
            g.pg.check_lens([r.pos - g.P for r in g.rows])
            a = idx[0]
            sel = (slice(a, a + len(idx)) if idx == list(range(a, a + len(idx)))
                   else mx.array(idx, dtype=mx.int32))
            groups.append((g.pg, sel))
            order += idx
        single = [b for b, r in enumerate(rows) if r.group is None]
        order = single + order
        if sorted(order) != list(range(len(rows))):
            raise RuntimeError("cascade: a grouped row is in no group")
        inv = None if order == sorted(order) else mx.array(np.argsort(order), dtype=mx.int32)
        return _StepPlan(single, groups, inv, list(self.groups))

    def _grouped_forward(self, plan, y, cache, pos):
        """The batched forward of a step with groups: each group's step is
        opened before the graph is built and closed after (its lengths then
        include this step's token, as the rows' positions will). A failure
        while building propagates to _run's handler, whose _fail_rows drops
        every group: no abort_step (it would evaluate the stacks, re-running
        the aborted graph, for groups about to be dropped)."""
        for g in plan.owners:
            g.pg.begin_step()
        out = self.lm(y, cache=cache, position_ids=pos)
        for g in plan.owners:
            g.pg.end_step()
        return out
