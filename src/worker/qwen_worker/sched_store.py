"""_Scheduler: prompt-cache stores at a row's end, one KV copy, the disk tier."""

import itertools
import time

import mlx.core as mx
import numpy as np

import pcdisk

from .config import (CLEAR_CACHE, GIB, log, PREFILL_STEP, PROMPT_CACHE, PROMPT_CACHE_GB,
                     PROMPT_CACHE_MOVE)
from .pcache import _Inherit, _Seg
from .stream import _GenState


class _StoreMixin:
    """Prompt-cache stores at a row's end: reply KV, one KV copy
    (QWEN_PROMPT_CACHE_MOVE: _settle*), the disk tier's loads."""

    # --- reply KV (QWEN_PROMPT_CACHE_REPLY) -------------------------------------
    def _reply_cache(self, row, i):
        """The cache list a finished row `i` of the batch ends with (its KV
        objects, its recurrent state in ArraysCaches of its own) -> None when
        it is not stored: ended otherwise than stop / length, still in a
        cascade group (its KV is in the group's stack; logged), vision.
        Builds lazy arrays only. Its state is always a gather into a buffer
        of its own, never a view: at B >= 2 out of the stacked state (a
        slice would keep the batch's whole stack alive in the entry), at
        B = 1 too (right after the batch shrank to it, its arrays are still
        _gdn_split's views of the old stack)."""
        try:
            # (a demoted DFlash row: row.fed holds its rounds' tokens too)
            if row.why not in ("stop", "length", "preempted") or row.req.images:
                return None
            if row.group is not None:
                log.info("id=%s prompt cache reply skipped: in a cascade group (P=%s) when it "
                         "ended, its KV is in the group's stack; the next turn resumes at the "
                         "prompt entry", row.req.id, getattr(row.group, "P", "?"))
                return None
            if any(not hasattr(row.cache[j], "keys") for j in self.fa_layers):
                return None
            from mlx_vlm.models.cache import ArraysCache
            stacked = self.gdn is not None
            idx = mx.array([i if stacked else 0], dtype=mx.int32)
            out = []
            for j, c in enumerate(row.cache):
                if hasattr(c, "cache"):
                    n = ArraysCache(size=len(c.cache))
                    n.cache = [a[idx] if a is not None else None
                               for a in (self.gdn[j].cache if stacked else c.cache)]
                    out.append(n)
                else:
                    out.append(c)
            return out
        except Exception:  # noqa: BLE001 — the request already ended: costs the extension only
            log.exception("id=%s prompt cache reply: state not taken", row.req.id)
            return None

    # --- one KV copy (QWEN_PROMPT_CACHE_MOVE) -------------------------------------
    def _move_on(self):
        """Single stream only: with other rows around, an entry moved into a
        row would be a miss for its siblings, and a cascade group's rows hold
        no KV of their own to give back. Quantized KV (since 2026-11-15):
        only quantized entries move (their segments are (packed, scales,
        biases) tuples, taken / restored like fp16 arrays); the fp16 part
        below quantized_kv_start is copied as without the mode (an fp16 entry
        is never moved, a row still fp16 at its prefill's end stores by copy:
        _prompt_cache). Not with the disk tier (QWEN_PROMPT_CACHE_DISK: an
        evicted entry's RAM stays held until written, and its pending write
        may share a segment the move empties; not built)."""
        return bool(PROMPT_CACHE_MOVE and PROMPT_CACHE and self.max_batch == 1
                    and self.eng._pcache.disk is None)

    def move_state(self):
        if self._move_on():
            return ("on (a hit moves the entry's KV into the row; a row's buffers become its entry"
                    + (f"; {self.kv['kv_bits']}-bit KV: quantized entries move, the fp16 part "
                       f"below {self.kv['quantized_kv_start']} tokens is copied"
                       if self.kv["kv_bits"] else "") + ")")
        if not PROMPT_CACHE_MOVE:
            return "off"
        return "off: needs " + ", ".join(
            x for x, ok in (("QWEN_PROMPT_CACHE=1", PROMPT_CACHE),
                            ("QWEN_MAX_BATCH=1 (max_batch is %d)" % self.max_batch,
                             self.max_batch == 1),
                            ("no QWEN_PROMPT_CACHE_DISK", self.eng._pcache.disk is None))
            if not ok)

    def _lost(self, row):
        """MOVE: a row that failed in a forward (a prefill slab, a decode
        step, a DFlash round) stores nothing: its KV arrays may hang on the
        failed graph (evaluating them would run it again — after an
        out-of-memory, into the same wall). The copies would have left the
        entry in the cache; here the conversation's next turn is a miss."""
        own, row.own = row.own, None
        if own is None:
            return
        row.fed = row.entry = None
        log.warning("id=%s prompt cache: the row failed, its KV is not stored%s "
                    "(QWEN_PROMPT_CACHE_MOVE)", row.req.id,
                    "; the entry moved into it is lost" if own.husk is not None else "")
        own.close()

    def _settle_prefix(self, row, cache, pos, why, tail):
        """MOVE: a prefill that stops early (cancelled / failed, the caches
        holding exactly prompt[:pos], a slab end) gives the row's buffers to
        the cache: the entry it took comes back as it was when nothing was
        prefilled past its end (unmove), else the prefix prompt[:pos] is
        stored as a cancelled prefill's (the partial store, `tail`: a DFlash
        row's context there). The row ends right after (its reservation is
        still counted: left out of the limit)."""
        own, row.own = row.own, None
        if own is None:
            return
        try:
            self._settle_prefix_(row, own, cache, pos, why, tail)
        finally:
            own.close()

    def _settle_prefix_(self, row, own, cache, pos, why, tail):
        req, pc, husk = row.req, self.eng._pcache, own.husk
        if husk is None and pos <= own.at:
            return                              # a miss that prefilled nothing
        if any(c.keys is None or c.offset != pos for c in cache if not hasattr(c, "cache")):
            log.warning("id=%s prompt cache: the %s prefill's KV is not stored: the cache does "
                        "not hold prompt[:%d]%s", req.id, why, pos,
                        "; the entry moved into the row is lost" if husk is not None else "")
            return
        if husk is not None and pos == own.at == len(husk.ids):
            limit = self.budget + self.prefill_reserve - (
                self.reserved - (0 if row.done else row.reserved))
            ok = pc.unmove(husk, _Seg.take(cache, pos), limit=limit, wanted=self._wanted)
            if ok:
                mx.eval(husk.arrays)
            log.info("id=%s prompt cache: the %s prefill gives its entry back (%d tokens, "
                     "%.1f MB; %d entries, %.1f MB total)%s", req.id, why, pos, husk.nbytes / 1e6,
                     len(pc.entries), pc.bytes / 1e6, "" if ok else ": NOT kept")
            return
        e = own.store(cache, pos, why, tail, _Seg.take(cache, pos, own.lo))
        if e is not None:
            mx.eval(e.arrays)
        elif husk is not None:
            log.warning("id=%s prompt cache: the %s prefill's prefix was not kept; the entry "
                        "moved into the row is lost", req.id, why)

    def _settle(self, row, rc=None, tail=None):
        """MOVE: a row that ended after its prefill (any reason) gives its
        own KV buffers to the prompt cache — the store the prefill's end
        deferred, then the reply store, with nothing copied beside the row:
        the prompt entry prompt[:-1] (the states saved at the prefill's end),
        and for a reply that is stored (`rc`: _reply_cache's list, `tail`: a
        DFlash row's context at its end; _reply_plan) its extension over the
        same buffers. A prompt entry alone takes [0, L) out of the buffers
        when they hold a KVCache step or more past it (_Seg.take: reasoning,
        a cancel after thousands of tokens). No-op for other rows. Never
        raises: the request already ended."""
        own, row.own = row.own, None
        if own is None:
            return
        req, pc = row.req, self.eng._pcache
        try:
            cache, L = row.cache, own.L
            if own.end is None or cache is None or any(
                    getattr(cache[j], "keys", None) is None or cache[j].offset < L
                    for j in self.fa_layers):
                if own.husk is not None:
                    log.warning("id=%s prompt cache: the entry moved into the row is lost (the "
                                "row ended before its prefill did, its KV is not a prefix's)",
                                req.id)
                return
            from mlx_vlm.models.cache import ArraysCache
            states, ptail = own.end
            it, pcache = iter(states), []
            for c in cache:                     # the row's cache as it was at the prefill's end
                if hasattr(c, "cache"):
                    n = ArraysCache(size=len(c.cache))
                    n.cache = list(next(it))
                    pcache.append(n)
                else:
                    pcache.append(c)
            plan = None
            if rc is not None and row.fed is not None and all(
                    cache[j].offset == row.pos for j in self.fa_layers):
                plan = self._reply_plan(row, row.fed)
            # one buffer per layer for the prompt entry and its reply: the
            # row's own (views), or [lo, end) taken out of it (_Seg.take)
            whole = _Seg.take(cache, row.pos if plan is not None else L, own.lo)
            e = own.store(pcache, None, None, ptail, whole.sub(own.lo, L))
            if e is None:
                # (the row's buffers were the only copy: unlike a refused
                # copy, the cache does not stay as it was)
                log.warning("id=%s prompt cache: the row's KV was not kept%s", req.id,
                            "; the entry moved into it is lost" if own.husk is not None else "")
                return
            new = None
            if plan is not None:
                new, _ = self._store_reply(row, rc, tail, seg=whole.sub(L, row.pos))
            try:
                mx.eval((new or e).arrays)
            except Exception:
                pc.entries = [x for x in pc.entries if x is not e and x is not new]
                pc.evictions += 1
                pc._recount()
                raise
        except Exception:  # noqa: BLE001 — costs the entry only
            log.exception("id=%s prompt cache: storing the row's KV failed "
                          "(QWEN_PROMPT_CACHE_MOVE)", req.id)
        finally:
            row.fed = row.entry = None
            own.close()

    def _reply_plan(self, row, fed):
        """-> (the ids a reply entry of the row holds, the decoded tokens n
        among them) or None: nothing decoded, a fed token never read, or a
        reply that generated a reasoning section (logged)."""
        req = row.req
        n = row.pos - len(req.ids)
        if n < 1 or len(fed) < n:
            return None                         # nothing decoded, or a fed token never read
        # a reasoning section = what the handler's _GenState splits off as
        # reasoning: generation started inside an open <think> (thinking on;
        # it runs until </think> / <tool_call>). With thinking off a <think> /
        # </think> id inside the content is content (_GenState: plain text /
        # dropped), so it does not skip the store.
        think = getattr(self.eng, "_think", None)
        # (a resumed preempted row: its reply began at the prompt as sent;
        # the tokens before the preemption are in req.ids)
        state = _GenState.for_prompt(req.prompt_ids, think)
        if state.in_thinking:
            tags = getattr(self.eng, "_tool_tags", None)
            for t in req.ids[len(req.prompt_ids):] + fed[:n]:
                state.token(t, think[1], tags[0] if tags else -1)
            log.info("id=%s prompt cache reply not kept: reasoning in the reply (thinking on, "
                     "%d of %d generated tokens)", req.id, state.reasoning_tokens, n)
            return None
        return req.ids + fed[:n], n

    def _store_reply(self, row, cache, tail=None, seg=None):
        """Extend the entry the row's prefill stored by the tokens the model
        consumed while it decoded (prompt + generated[:k]: the step-ahead fed
        the EOS a row stops on, a sync row did not, a "length" row fed every
        generated token, the last one too: prompt + completion; row.pos says
        how many) and its state after them,
        through the same store as a prefill (budget checks, absorb, victim
        order) -> (the new entry or None, the entries it absorbed). Its
        copies are still lazy: _drop_done queues every finishing row's in one
        async_eval (between two decode steps the host must not wait for the
        step in flight) and, should that fail, puts the absorbed ones back.
        The entry keeps its prompt entry's recency (`clock`). Never evicts
        the entry the running prefill resumed from (task_row.parent: its
        pieces stay allocated until that prefill stores, as in _regrow).
        `tail`: a DFlash row's drafter context at the end (_with_tail);
        `seg` (MOVE, _settle): the reply's KV, views of the row's own buffers.
        Nothing when the entry is gone (evicted / absorbed while it decoded:
        the victim order had decided its next turn is the furthest away),
        nor for a reply that generated a reasoning section (thinking on, as
        _GenState splits it: clients rarely send it back, so the next prompt
        diverges at the reply start; the copy and the prompt-end state, a
        regenerate's full hit, would be lost). A think id the model emits
        inside the content with thinking off is content: stored."""
        fed, row.fed = row.fed, None
        req = row.req
        if cache is None:
            return None, []
        pc = self.eng._pcache
        e = row.entry() if row.entry is not None else None
        row.entry = None
        n = row.pos - len(req.ids)
        if e is None or not any(e is x for x in pc.entries):
            log.info("id=%s prompt cache reply not kept: its prompt entry is gone (evicted, "
                     "absorbed or never kept)", req.id)
            return None, []
        plan = self._reply_plan(row, fed)
        if plan is None:
            return None, []
        ids = plan[0]
        L = len(e.ids)
        if e.ids != ids[:L] or any(row.cache[j].offset != row.pos for j in self.fa_layers):
            log.warning("id=%s prompt cache reply not kept: the row's cache does not match "
                        "its entry", req.id)
            return None, []
        ends, protect = row.plan
        spare = self._parents()
        try:
            new, covered = pc.store(ids, cache, {}, _Inherit(e.chain, {}, None, e.quantized), L,
                                    ends, protect,
                                    limit=self.budget + self.prefill_reserve - self.reserved,
                                    kv_start=self.kv["quantized_kv_start"], wanted=self._wanted,
                                    reply=L, tick=False, sync=False, spare=spare,
                                    clock=(e.used, e.seq), tail=tail, seg=seg)
        except Exception:  # noqa: BLE001 — the request already ended
            log.exception("id=%s prompt cache reply store failed", req.id)
            return None, []
        if new is None:
            return None, []
        log.info("id=%s prompt cache reply kept: %d tokens (+%d generated), %d checkpoints, "
                 "%.1f MB (%d entries, %.1f MB total) absorbed=%d pieces=%d", req.id,
                 len(new.ids), n, len(new.ckpts), new.nbytes / 1e6,
                 len(pc.entries), pc.bytes / 1e6, len(covered), len(new.chain))
        return new, covered

    # --- prompt cache on disk (QWEN_PROMPT_CACHE_DISK, pcdisk.py) -----------------
    def _disk_tick(self, disk):
        """Before admission (tick): the loads that finished go into the RAM
        cache (for a request still waiting, else dropped); a load starts for
        the first of the next max_batch waiting requests whose deepest hit is
        on disk at least a slab deeper than in RAM (one load at a time: the
        SSD is the bottleneck) -> True while the head of the queue waits for
        its own load, at most pcdisk.WAIT_S (decode runs meanwhile; with
        nothing else to do the thread waits on the load 5 ms per tick rather
        than spinning). Never raises: a failure costs the load only."""
        try:
            for job in disk.done():
                self._disk_take(disk, job)
            with self.cv:
                reqs = list(itertools.islice(self.waiting, self.max_batch))
            if reqs and not disk.busy():
                # the RAM entries the requests ahead resume from (the admitted
                # one whose prefill has not started first): a load must fit
                # beside them under the soft cap, else the stores before its
                # turn evict it again (victim: the furthest back first)
                task = self.task_row
                ahead = [task.req] if task is not None and not task.started else []
                for req in reqs:
                    if self._disk_want(disk, req, ahead):
                        break
                    ahead.append(req)
            if self.task is not None or not reqs:
                return False
            w = reqs[0].disk
            job = w.job if w is not None else None
            if job is None or job.taken or time.time() - w.t0 > pcdisk.WAIT_S:
                return False
            if not self.rows:
                job.done.wait(0.005)
            return True
        except Exception:  # noqa: BLE001 — the queue goes on without the disk tier
            log.exception("prompt cache disk: load step failed")
            return False

    def _disk_want(self, disk, req, ahead=()):
        """Start loading req's deepest disk hit when it beats its RAM hit by
        at least a slab (PREFILL_STEP) and fits: beside the batch (budget +
        prefill reserve - reserved, as a store), in memory now (the live
        check of _fits) and, unless req is the first in line, under the soft
        cap beside the RAM entries the requests `ahead` of it resume from
        -> True: started. The peeks are memoized on req.disk until either
        cache changes (versions). A request loads again (at most
        pcdisk.MAX_LOADS in all) when its loaded entry left the RAM cache
        before its turn and is on disk again (never after a failed load or
        one not kept)."""
        if req.images:
            return False
        look = self._lookup_ids(req)
        if look is None:                        # prompt_cache "off"
            return False
        w = req.disk
        if w is None:
            w = req.disk = pcdisk.Want()
        if w.job is not None and (not w.job.taken or not w.kept or w.loads >= pcdisk.MAX_LOADS):
            return False                        # loading, failed / not kept, or out of loads
        pc = self.eng._pcache
        key = (disk.version, pc.version)
        if w.key != key:
            if w.ids is None:
                w.ids = np.asarray(look, dtype=np.int32)
            w.dent, w.depth = disk.peek(w.ids, self._ends(req))
            w.ram = self._peek(req)[1]
            w.key = key
        if w.dent is None or w.depth < w.ram + PREFILL_STEP:
            return False
        # within the budget beside the batch AND the cache as it is (the load
        # is extra RAM until inserted; the prefill reserve stays free for the
        # rows' growth, which a load must never cut: _regrow cancels loads
        # first anyway); from here until inserted or dropped `need` is held,
        # so every other budget check sees it
        need = disk.cost(w.dent, pc.entries)
        live = mx.get_active_memory() + (0 if CLEAR_CACHE else mx.get_cache_memory())
        if (self.reserved + pc.charged() + need > self.budget
                or live + need + self.prefill_reserve > self.cap):
            return False
        if ahead:
            mine = []
            for r in ahead:
                e = None if r.images else self._peek(r)[0]
                if e is not None and not any(e is x for x in mine):
                    mine.append(e)
            if mine and pc.bytes_of(mine) + need > PROMPT_CACHE_GB * GIB:
                return False                    # would be evicted before its turn
        job = disk.load(w.dent, pc.entries, req, need)
        if job is None:                         # gone from disk since the peek: peek again
            w.key = None
            return False
        w.job, w.t0, w.kept = job, time.time(), False
        w.loads += 1
        log.info("id=%s prompt cache disk: loading an entry of %d tokens (%.1f MB to read) for a "
                 "%d-token hit (%d in RAM)%s", req.id, w.dent.L, need / 1e6, w.depth, w.ram,
                 f", load {w.loads}" if w.loads > 1 else "")
        return True

    @staticmethod
    def _held_wait(req):
        """_fits: may `req` still wait for the disk tier's held RAM (at most
        pcdisk.HELD_WAIT_S from its first such wait)? (False without the tier.)"""
        w = req.disk
        if w is None:
            w = req.disk = pcdisk.Want()
        now = time.time()
        if w.held_since is None:
            w.held_since = now
        return now - w.held_since <= pcdisk.HELD_WAIT_S

    def _disk_room(self, over, rid):
        """`over()` (bytes past the headroom) > 0 only for the disk tier's
        held RAM (_regrow): the offers of this tick not handed to the writer
        yet go first (pcdisk.revoke: freed now, as spill False; the writer
        never sees them before the tick's flush, review 2026-10-24 #1), then
        the loads not inserted (cancelled, dropped: a waiting request goes
        without its hit), then the writer is waited for, at most
        pcdisk.HELD_WAIT_S (the scheduler blocks meanwhile: decode stalls,
        rarely, against cutting a row's answer) and only while what the tier
        can still release covers the shortfall (not a stuck write's hold);
        the caller decides after."""
        disk = self.eng._pcache.disk
        if over() <= 0 or not disk.held_bytes():
            return
        t0 = time.time()
        r, freed = disk.revoke(over())
        n = disk.cancel_loads() if over() > 0 else 0
        while over() > 0:
            for job in disk.done():
                self._disk_take(disk, job)
            disk.check_loads()                  # a stuck read's hold goes after TIMEOUT_S
            if over() <= 0 or over() > disk.held_bytes() - disk.stuck_bytes() \
                    or time.time() - t0 > pcdisk.HELD_WAIT_S:
                break
            disk.wait(0.05)
        log.info("id=%s waited %.2f s for the prompt cache disk tier's held RAM before growing "
                 "(%d unwritten evictions freed, %.1f MB; %d loads cancelled; %s)", rid,
                 time.time() - t0, r, freed / 1e6, n, "room made" if over() <= 0 else "still short")

    def _disk_take(self, disk, job):
        """_disk_insert, never raising: a failure drops that load (unpinned,
        its hold released), the batch's other loads go on (else busy()
        stayed true for good)."""
        try:
            self._disk_insert(disk, job)
        except Exception:  # noqa: BLE001 — costs this load only
            log.exception("prompt cache disk: inserting a loaded entry of %d tokens failed",
                          job.dent.L)
            disk.discard(job)

    def _disk_insert(self, disk, job):
        """A finished load into the RAM cache (pc.insert: its budget checks,
        the running prefill's parent spared) when its request still waits;
        not kept, it is simply dropped (its arrays freed here)."""
        req = job.req()
        with self.cv:
            waiting = req is not None and any(r is req for r in self.waiting)
        if not waiting:
            disk.discard(job)
            log.info("prompt cache disk: a loaded entry of %d tokens dropped: its request no "
                     "longer waits", job.dent.L)
            return
        e = disk.finish(job)
        if e is None:
            return
        pc = self.eng._pcache
        spare = self._parents()                 # the running prefills' (main, lane) parents
        if pc.insert(e, limit=self.budget + self.prefill_reserve - self.reserved,
                     wanted=self._wanted, spare=spare):
            disk.hits += 1
            if req.disk is not None:
                req.disk.kept = True            # it may load again if evicted before its turn
            log.info("id=%s prompt cache disk: entry of %d tokens loaded (%.1f MB read in %.0f "
                     "ms; %d entries, %.1f MB total)", req.id, len(e.ids), job.nbytes / 1e6,
                     job.ms, len(pc.entries), pc.bytes / 1e6)
        else:
            disk.discards += 1
            log.info("id=%s prompt cache disk: loaded entry of %d tokens not kept: over the "
                     "batch's headroom", req.id, len(e.ids))
