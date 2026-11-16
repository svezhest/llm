"""The continuous-batching scheduler (_Scheduler): threads, ticks, logging; the
rest of its methods come from the sched_* mixins."""

import collections
import functools
import threading
import time

import mlx.core as mx

from .config import (CASCADE, CASCADE_KERNEL, CLEAR_CACHE, GIB, KV_KWARGS, log, PREFILL_LANE,
                     PREFILL_LANE_BURST, PREFILL_SHARE, RESERVE_TOKENS, SCHED_LOG_S,
                     SCHED_MAX_FAILURES, SCHED_STALL_S)
from .sizing import _act_dtype, _gdn_row_bytes, _kv_token_bytes, _weight_bytes
from .rows import _PrefixGroup
from .request import _Row
from .sched_admit import _AdmitMixin
from .sched_prefill import _PrefillMixin
from .sched_spec import _SpecMixin
from .sched_store import _StoreMixin
from .sched_cascade import _CascadeMixin
from .sched_decode import _DecodeMixin


# --- continuous batching -----------------------------------------------------
# Every running request decodes in ONE forward of shape (B, 1) per step: the
# weights are read once for B tokens, and decode is bandwidth-bound (the 8-bit
# qmm costs x1.17 at 2 rows, x1.97 at 4 and a flat x2.6 from 6 to 32 rows on
# M1). Each row keeps exactly the cache objects a lone request has (a KVCache /
# QuantizedKVCache per full-attention layer, prefilled by the slab loop), so
# quantized KV, its 5000-token start and the prompt cache work unchanged: only
# the cache update + SDPA run per row (RowsKV, _rows_attention). The
# gated-delta-net state of all rows is stacked on axis 0 in one ArraysCache per
# linear layer. Positions go in explicitly per row (KV length + mRoPE delta):
# no decode step reads the model's _rope_deltas / _position_ids, which only
# prefill writes. A row decoding alone runs the plain library call on its own
# caches (explicit positions are bit-identical to the computed ones), so B=1 is
# the pre-batching output; at B>1 a row differs from its solo run only by the
# kernels' row-count-dependent rounding (B=2 is still bit-identical: qmv per
# row; 3..16 rows take mrqmm on 8-bit modules (QWEN_MRQMM), qmv/qmm on the
# others; MoE sorted gather at B*top_k >= 64) — the same class of noise as the
# DFlash verify.
# mlx-vlm's own batch path is not used: its quantized batch cache ignores left
# padding at decode (wrong logits with q8/q4 KV), its ragged SDPA kernel asks
# M1 for a 1024-thread threadgroup, and it has no hook for the prompt cache.
#


class _Scheduler(_AdmitMixin, _PrefillMixin, _SpecMixin, _StoreMixin, _CascadeMixin, _DecodeMixin):
    """Continuous batching; after load, the one thread that runs MLX.

    Handlers submit _Req (FIFO, no overtaking). The head is admitted when it
    fits: fewer than max_batch rows and its reservation (KV for prompt +
    min(max_tokens, QWEN_BATCH_RESERVE_TOKENS), recurrent state) within the
    budget, plus a live check of the active memory; an empty batch always
    admits it (the old one-at-a-time behaviour: MAX_CONTEXT validation bounds
    a lone request). Rows are re-accounted at their real length every step.
    Prefill is one
    request at a time (the slab loop, prompt cache, checkpoints), a slab per
    tick, interleaved with decode (PREFILL_SHARE); then the row joins the
    batch. A decode tick is one forward for every row, then the per-row
    post-processing.

    DFlash (QWEN_DRAFT) runs only for an eligible request admitted with
    nothing else running or waiting; when another request arrives (and
    max_batch > 1) it is demoted at the next round boundary to a plain batch
    row, and stays one.

    Usage.peak_gb: the MLX peak since the batch was last empty, read when the
    request ends; with concurrent rows it is the process peak during the
    request, not the request's own share (with QWEN_CASCADE it includes the
    groups' fp32 prefixes and the formation / step transients).

    Cascade groups (QWEN_CASCADE, on by default with fp16 KV + the prompt
    cache; _cascade_join / _cascade_tidy / _plan): rows sharing a long
    prompt prefix decode their attention as one group over one fp32 copy of
    the prefix. Formed when a row joins, joined by later rows, left by
    finished rows and by rows whose own suffix passes P // 2, dissolved (the
    members get contiguous caches back: release) when fewer than 3 members
    are left (so always at B <= 2) or, at the members' real lengths, the
    group's arrays would take more than their own fp16 caches (_group_wins).
    A group is charged once (_group_bytes, at the members' reservations,
    never below its prefix + stacks + 2 contiguous rows: what the forced
    release of its last 2 members materializes beside it) and its members
    only their recurrent state. A formation / join that raises `reserved` is
    taken only while `reserved` + the prompt cache's live bytes stay within
    the budget (one that lowers it always; growth can still pass the budget,
    as for ungrouped rows: _regrow). An optional release (a P // 2 leaver, a
    memory dissolve, set_cascade(False): one member per group and decode
    step, the last < 3 together) is checked with the group alive -- `reserved` +
    the released rows' contiguous caches + the prompt cache within the
    headroom (budget + prefill reserve), crediting nothing for the group's
    shrink -- else it waits; a forced release (< 3 live members) is not
    checked: its peak is inside the group's charge (the floor).

    Failures: a failed decode step fails the rows in it (not the prefilling
    request). SCHED_MAX_FAILURES consecutive failed steps with several rows
    (or with none: the scheduler itself) close it: everything queued or
    running gets Error(worker_fault), new requests are refused, Health
    reports not ready and on_fault() runs (Engine: exit the process so
    launchd restarts it). A step failing for a lone row is that request's
    own error and no strike. A tick running for SCHED_STALL_S only makes
    alive() false (Health not ready, waiting handlers give up) until it
    ends. stop() (Shutdown) ends the loop after the current tick and sends
    Error(shutdown) to everything left."""

    def __init__(self, eng, *, max_batch, budget, cap, prefill_reserve,
                 share=PREFILL_SHARE, kv=None, on_fault=None):
        from mlx_vlm.generate.common import maybe_quantize_kv_cache
        self.eng = eng
        self.model = eng.model
        self.lm = eng.model.language_model
        self.max_batch = max_batch
        self.budget, self.cap, self.prefill_reserve, self.share = budget, cap, prefill_reserve, share
        self.kv = kv or KV_KWARGS
        self.quantize = functools.partial(
            maybe_quantize_kv_cache, quantized_kv_start=self.kv["quantized_kv_start"],
            kv_group_size=self.kv["kv_group_size"], kv_bits=self.kv["kv_bits"])
        # + an MTP head's own KV (mtp.MTPDraft.token_bytes, fp16 whatever the
        # target's KV format): charged to every row, the rows that draft are
        # not known here
        head = getattr(getattr(eng, "mtp", None), "token_bytes", 0)
        self.kv_fp16 = _kv_token_bytes(self.lm, 0) + head
        self.kv_q = _kv_token_bytes(self.lm, self.kv["kv_bits"]) + (head if self.kv["kv_bits"] else 0)
        self.gdn_row = _gdn_row_bytes(self.lm)
        self.cv = threading.Condition()
        self.waiting = collections.deque()
        self.rows: list[_Row] = []
        self.gdn = None                 # {layer: stacked ArraysCache} while B >= 2
        self.task = None                # generator: the prefill / lone DFlash run in progress
        self.task_row = None
        # QWEN_PREFILL_LANE: the short text prefill running between the main
        # task's slabs, and how many were admitted since the main one's last slab
        self.lane = self.lane_row = None
        self.lane_run = 0
        # a lane row that joined and has not read its first token yet: decode
        # runs until it has, before the main prefill's next slab (_lane_first)
        self.lane_new = None
        self.resume = 0.0               # prefill waits until then (decode's share)
        self.reserved = 0
        self.steps = 0
        self.clear = False
        self.wired = None
        self.hdtype = None
        self.log_at, self.log_steps, self.log_rows = time.time(), 0, 0
        # per log window: wall time of decode / prefill ticks, decode bytes
        # (weights + KV + recurrent state per step: the bandwidth floor)
        self.log_dec = self.log_pf = 0.0
        self.log_bytes = 0
        self.w_bytes = _weight_bytes(self.lm)
        self.thread = None
        self.closed = None              # (kind, msg) once the scheduler stopped or died
        self.stopping = False
        self.failures = 0
        self.tick_t0 = 0.0              # start of the running tick, 0 between ticks
        self.on_fault = on_fault
        self.fa_layers = [i for i, x in enumerate(self.lm.model.layers) if not x.is_linear]
        self.groups: list[_PrefixGroup] = []
        self.cascade = self._cascade_on(CASCADE)
        # fixed for the scheduler's life: _group_bytes charges by it
        self.cascade_kernel = CASCADE_KERNEL and _act_dtype(self.lm) == mx.float16
        self.n_groups = self.n_grouped = 0  # Health: plain ints kept current here

    # --- threads -------------------------------------------------------------
    def start(self):
        self.thread = threading.Thread(target=self._run, name="scheduler", daemon=True)
        self.thread.start()

    def alive(self):
        """Running, not closed, and not stuck in one tick (Health: ready).
        Idle time never counts: only a tick that has been running too long."""
        t0 = self.tick_t0
        return (self.closed is None and self.thread is not None and self.thread.is_alive()
                and not (t0 and time.time() - t0 > SCHED_STALL_S))

    def submit(self, req):
        with self.cv:
            if self.closed is None:
                self.waiting.append(req)
                self.cv.notify()
                return
        req.out.put(("error", *self.closed))

    def wake(self):
        """Another thread freed MLX memory (pcdisk's writer, its views
        gone): an idle loop runs _idle again (the buffer pool cleared)."""
        with self.cv:
            self.cv.notify()

    def stop(self, timeout=10.0):
        """Stop after the current tick; everything left gets Error(shutdown).
        -> False when the tick did not end within `timeout`."""
        with self.cv:
            self.stopping = True
            self.cv.notify()
            return self.cv.wait_for(lambda: self.closed is not None or self.thread is None
                                    or not self.thread.is_alive(), timeout)

    def _run(self):
        from mlx_vlm.generate.common import generation_stream
        closed = ("shutdown", "the worker is shutting down")
        try:
            with mx.stream(generation_stream):
                while True:
                    with self.cv:
                        while not (self.rows or self.task or self.lane or self.waiting
                                   or self.stopping):
                            self._idle()
                            self.cv.wait()
                    if self.stopping:
                        break
                    rows = len(self.rows)
                    self.tick_t0 = time.time()
                    try:
                        self.tick()
                    except Exception as e:  # noqa: BLE001 — a failed step fails its rows, not the worker
                        log.exception("batch step failed")
                        self._fail_rows(e)
                        # a lone row's failure is its own request's error (a
                        # client retrying it must not take the worker down)
                        self.failures += rows != 1
                        if self.failures >= SCHED_MAX_FAILURES:
                            closed = ("worker_fault", f"scheduler failed {self.failures} steps "
                                                      f"in a row: {type(e).__name__}: {e}")
                            log.error("scheduler gives up: %s", closed[1])
                            break
                    finally:
                        self.tick_t0 = 0.0
        except BaseException as e:  # noqa: BLE001 — never leave handlers waiting on a dead thread
            closed = ("worker_fault", f"scheduler thread died: {type(e).__name__}: {e}")
            log.exception("scheduler thread died")
        finally:
            self._close(*closed)
            if closed[0] == "worker_fault" and self.on_fault is not None:
                self.on_fault()
            # The thread never returns: MLX tears down its thread-local
            # generation stream when the thread exits, which crashed the
            # process (segfault) while arrays from that stream were alive.
            with self.cv:
                while True:
                    self.cv.wait()

    def _close(self, kind, msg):
        """Refuse new requests and end everything queued or running."""
        with self.cv:
            self.closed = (kind, msg)
            left = list(self.waiting)
            self.waiting.clear()
            self.cv.notify_all()
        rows = self.rows + self._tasks()
        for r in rows:
            if not r.done:
                r.done = True
                r.req.out.put(("error", kind, msg))
        for req in left:
            req.out.put(("error", kind, msg))
        self.rows, self.gdn, self.task, self.task_row, self.reserved = [], None, None, None, 0
        self.lane = self.lane_row = self.lane_new = None
        self._drop_groups()
        try:
            from mlx_vlm.generate.common import generation_stream
            mx.synchronize(generation_stream)
            self._idle()
        except Exception:  # noqa: BLE001 — closing anyway
            log.exception("scheduler cleanup failed")

    def _busy(self):
        # wired_limit's exit synchronizes and restores the limit, so it wraps
        # the busy period, never a single row
        if self.wired is None and mx.default_device().type == mx.gpu:
            from mlx_vlm.generate.common import generation_stream, wired_limit
            self.wired = wired_limit(self.model, [generation_stream])
            self.wired.__enter__()

    def _idle(self):
        wired, self.wired = self.wired, None       # never retried if __exit__ raises
        if wired is not None:
            wired.__exit__(None, None, None)
        disk = self.eng._pcache.disk
        if disk is not None:                        # a load finished for no one waiting: freed
            for job in disk.done():
                self._disk_take(disk, job)
        if CLEAR_CACHE:
            mx.clear_cache()

    # --- one scheduling decision ----------------------------------------------
    def tick(self):
        self._busy()
        with self.cv:                       # handlers append meanwhile
            gone = [r for r in self.waiting if r.cancel.is_set()]
            for r in gone:
                self.waiting.remove(r)
        for r in gone:
            r.out.put(("done", "cancelled"))
        disk = self.eng._pcache.disk            # QWEN_PROMPT_CACHE_DISK (None: off)
        hold = disk is not None and self._disk_tick(disk)
        if self.task is None and self.waiting and not hold:
            self._admit()
        t = time.time()
        if ((self.task is not None or self.lane is not None)
                and (not self.rows or time.time() >= self.resume) and not self._lane_first()):
            # prefill's turn: the lane first (a short request beside a running
            # plain prefill; at most PREFILL_LANE_BURST of them in a row)
            if (PREFILL_LANE and self.lane is None and self.waiting and self.task is not None
                    and self.lane_run < PREFILL_LANE_BURST and self.task.__name__ == "_prefill"
                    and self.task_row.started):     # (its lookup ran: its hit is its parent)
                self._admit_lane()
            if self.lane is not None:
                self._advance_lane()
            else:
                self.lane_run = 0
                self._advance()
            self.log_pf += time.time() - t
        elif self.rows:
            steps = self.steps
            moved = self._step_bytes(self.rows)
            self._iterate()
            if self.steps != steps:
                self.log_dec += time.time() - t
                self.log_bytes += moved
        if self.clear:
            # a finished long row's KV and prefill buffers go back to the pool
            if CLEAR_CACHE:
                mx.clear_cache()
            self.clear = False
        if disk is not None:
            disk.flush()                        # this tick's evictions to the writer (views made here)
        self._log()

    def fit_max_batch(self):
        """max_batch, or 1 when the budget cannot hold two typical rows (a
        2K prompt + the reserved generation): then batching would only make
        admissions wait while gateways keep max_batch requests in flight."""
        typical = self._bytes(2048 + RESERVE_TOKENS)
        if self.max_batch > 1 and self.budget < 2 * typical:
            log.warning("batching off: KV budget %.2f GiB holds fewer than 2 rows of %.2f GiB "
                        "(raise QWEN_WIRED_LIMIT_GB / iogpu.wired_limit_mb or set QWEN_BATCH_MEM_GB)",
                        self.budget / GIB, typical / GIB)
            self.max_batch = 1
        self.cascade = self.cascade and self.max_batch >= 3
        return self.max_batch

    def _log(self):
        now = time.time()
        if now - self.log_at < SCHED_LOG_S:
            return
        steps = self.steps - self.log_steps
        if steps or self.rows or self.waiting or self.task or self.lane:
            # rows / waiting / prefilling: at this moment; the rest over the
            # window. step_ms = decode ticks' wall time per step (the step-ahead
            # makes it the GPU's step time), gbps = the bytes a step must read
            # (_step_bytes) over it; prefill_s / decode_s = the window's split
            log.info("batch: rows=%d waiting=%d prefilling=%s steps/s=%.1f mean_rows=%.2f "
                     "step_ms=%.1f gbps=%.0f decode_s=%.1f prefill_s=%.1f "
                     "reserved=%.2f/%.2fGiB pcache=%.2fGiB active=%.2fGiB groups=%d grouped=%d",
                     len(self.rows), len(self.waiting),
                     (self.task_row.req.id if self.task_row is not None else "-")
                     + (f"+lane:{self.lane_row.req.id}" if self.lane_row is not None else ""),
                     steps / (now - self.log_at), self.log_rows / steps if steps else 0.0,
                     1e3 * self.log_dec / steps if steps else 0.0,
                     self.log_bytes / self.log_dec / 1e9 if self.log_dec else 0.0,
                     self.log_dec, self.log_pf,
                     self.reserved / GIB, self.budget / GIB, self.eng._pcache.bytes / GIB,
                     mx.get_active_memory() / GIB, self.n_groups, self.n_grouped)
        self.log_at, self.log_steps, self.log_rows = now, self.steps, 0
        self.log_dec = self.log_pf = 0.0
        self.log_bytes = 0
