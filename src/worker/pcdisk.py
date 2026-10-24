"""Prompt cache on disk (QWEN_PROMPT_CACHE_DISK=<dir>; unset = off): a second
tier under worker._PrefixCache. Design: docs/perf/2026-10-24/design/pcdisk.md.

An entry evicted from RAM (_PrefixCache.evict: every eviction path, store /
_fits / _regrow) is written here instead of being lost; a waiting request
whose deepest hit is on disk (at least a slab deeper than in RAM) gets that
entry loaded while it waits and inserted back into the RAM cache through the
normal budget checks, so its lookup resumes from it as from any RAM entry.
Entries are immutable and fp16 / fp32 / packed-uint32 bytes come back bit for
bit, so a hit through the disk == the same hit from RAM (== cold where that
holds today).

Files (one directory, our names only; nothing else in it is ever touched):
  seg-<uid>.bin  a _Seg (KV of [a, b) of every full-attention layer)
  st-<uid>.bin   a recurrent state (one checkpoint or an end state)
  ent-<uid>.bin  an entry: its metadata (chain, checkpoint positions,
                 layout, flags) + its ids; refers to seg / st blobs by uid
  pcdisk.json    the validity key (KEEP mode)
Segments and states are shared objects between RAM entries (a system block's
segment, inherited checkpoints), so they are blobs of their own: written once,
reference-counted by the disk entries, and on a load a blob whose object is
still alive in RAM is reused instead of read (no extra RAM either).

Threads. MLX graphs only on the scheduler thread (worker invariant):
  - scheduler: offer (no MLX), flush (mx.eval of the evicted entry's arrays,
    already evaluated: a no-op wait; zero-copy numpy views of their buffers:
    no copy, no stall), peek / load start (index only), finish (builds the
    _Seg / _Entry objects; bf16 views).
  - writer: plain file I/O over the numpy views, then drops them (the MLX
    buffers are freed there: constant, evaluated, detached arrays; the
    allocator is locked, as for the image processor's to_mlx arrays the
    handler threads free).
  - loader: file I/O into numpy, then mx.array(numpy): constant arrays,
    allocation + memcpy, no graph, no stream (as the processor's to_mlx on a
    handler thread).
Disk errors never fail a request: a failed write loses the entry (as before
this tier), a failed load is a miss; both are logged.
"""
import fcntl
import hashlib
import importlib
import json
import logging
import os
import queue
import secrets
import struct
import threading
import time
import weakref

import mlx.core as mx
import numpy as np

log = logging.getLogger("qwen.worker")

FORMAT = 1
DIR = os.environ.get("QWEN_PROMPT_CACHE_DISK", "")
GB = float(os.environ.get("QWEN_PROMPT_CACHE_DISK_GB", "32") or "32")
# keep the files of a previous run whose key (model, numerics flags, code) matches;
# 0: the directory's cache files are deleted at start
KEEP = os.environ.get("QWEN_PROMPT_CACHE_DISK_KEEP", "0") == "1"
# RAM of evicted entries waiting for the writer (held until their files are
# written, charged in the batch budget beside the cache: held_bytes); an
# eviction past it is a plain deletion, as without the tier (4: one 50K-token
# entry of ~2.5 GB fits, two 1 GB AppWorld entries at once)
QUEUE_GB = float(os.environ.get("QWEN_PROMPT_CACHE_DISK_QUEUE_GB", "4") or "4")
# shorter entries are not worth a file: their hit saves less than a slab or two
MIN_TOKENS = int(os.environ.get("QWEN_PROMPT_CACHE_DISK_MIN_TOKENS", "4096") or "0")
# the head of the queue waits this long for its load before going without it
WAIT_S = float(os.environ.get("QWEN_PROMPT_CACHE_DISK_WAIT_S", "10") or "0")
# the RAM the tier holds (held_bytes) is gone within ~1 s per GB written: a
# growing row (_regrow) and an admission (_fits) that only it keeps out wait
# for it this long (the growth blocks the scheduler meanwhile, a cut answer
# would be worse) before deciding as without the wait
HELD_WAIT_S = float(os.environ.get("QWEN_PROMPT_CACHE_DISK_HELD_WAIT_S", "5") or "0")
# a write running this long means a stuck disk: the writes queued behind it
# are dropped (their RAM freed: never written), nothing more is taken until it
# returns; its own buffers stay held (the writer is reading them)
TIMEOUT_S = float(os.environ.get("QWEN_PROMPT_CACHE_DISK_TIMEOUT_S", "30") or "0")
# F_NOCACHE (macOS) on every file: the cache's GBs never sit in the unified
# buffer cache beside the model (RAM is what the tier frees; a page-cache copy
# of each written entry would compete with the batch and push other processes
# to swap); reads then measure the SSD, not RAM. 0: the OS caches as usual.
NOCACHE = os.environ.get("QWEN_PROMPT_CACHE_DISK_NOCACHE", "1") != "0"

MAGIC = b"QPCDISK1"
_ALIGN = 4096
GIB = 1024**3


def _new_uid():
    return secrets.token_hex(8)


# --- blob files --------------------------------------------------------------
def _nocache(f):
    if NOCACHE and hasattr(fcntl, "F_NOCACHE"):
        try:
            fcntl.fcntl(f.fileno(), fcntl.F_NOCACHE, 1)
        except OSError:
            pass                                      # a file system without it: cached


def _write_blob(path, header, arrays, bf16=None, sync=False):
    """MAGIC, u64 header length, header JSON, then the arrays (C order),
    each at a 4096-aligned offset. Written to path + ".tmp", then renamed:
    a file under its name is complete (its size is checked on read). `sync`
    (KEEP: files outlive the process): fsync before the rename, so a power
    loss cannot leave a full-size file of zeros under the name."""
    specs, off = [], 0
    for i, a in enumerate(arrays):
        specs.append({"dtype": a.dtype.name, "shape": list(a.shape), "off": off,
                      "bf16": bool(bf16[i]) if bf16 else False})
        off += (a.nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
    h = dict(header, arrays=specs, data=off)
    raw = json.dumps(h, separators=(",", ":")).encode()
    start = (16 + len(raw) + _ALIGN - 1) // _ALIGN * _ALIGN
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        _nocache(f)
        f.write(MAGIC + struct.pack("<Q", len(raw)) + raw)
        f.write(b"\0" * (start - 16 - len(raw)))
        for a, s in zip(arrays, specs):
            c = np.ascontiguousarray(a)               # a view of an MLX buffer: no copy
            if c.nbytes:
                f.write(memoryview(c).cast("B"))
            pad = (-c.nbytes) % _ALIGN
            if pad:
                f.write(b"\0" * pad)
            del c
        if sync:
            f.flush()
            os.fsync(f.fileno())
    os.replace(tmp, path)
    return start + off


def _size(nbytes, header=0):
    """About the size _write_blob gives arrays of these byte counts (a
    header of ~120 bytes per array + `header` more)."""
    head = (16 + 256 + 120 * len(nbytes) + header + _ALIGN - 1) // _ALIGN * _ALIGN
    return head + sum((n + _ALIGN - 1) // _ALIGN * _ALIGN for n in nbytes)


def _read_header(f, path):
    head = f.read(16)
    if len(head) != 16 or head[:8] != MAGIC:
        raise ValueError(f"{path}: not a prompt-cache file")
    n = struct.unpack("<Q", head[8:])[0]
    h = json.loads(f.read(n))
    start = (16 + n + _ALIGN - 1) // _ALIGN * _ALIGN
    size = os.fstat(f.fileno()).st_size
    if size != start + h["data"]:
        raise ValueError(f"{path}: {size} bytes, expected {start + h['data']} (torn write?)")
    return h, start


def _read_blob(path, skip=False, each=None):
    """-> (header, [numpy arrays]) (skip: the header only, size checked;
    `each`: applied to every array right after it is read, its numpy
    buffer dropped before the next one: [each(array)])."""
    with open(path, "rb") as f:
        _nocache(f)
        h, start = _read_header(f, path)
        if skip:
            return h, None
        out = []
        for s in h["arrays"]:
            a = np.empty(s["shape"], dtype=np.dtype(s["dtype"]))
            if a.nbytes:
                f.seek(start + s["off"])
                if f.readinto(memoryview(a).cast("B")) != a.nbytes:
                    raise ValueError(f"{path}: short read")
            out.append(each(a) if each is not None else a)
            del a
        return h, out


def _host(arrs):
    """Scheduler thread: zero-copy numpy views of evaluated MLX arrays (the
    view holds the array: its buffer lives until the view goes) -> (views,
    bf16 flags). bfloat16 has no buffer format: an MLX uint16 view of it."""
    mx.eval(arrs)
    views, flags = [], []
    for a in arrs:
        b = a.dtype == mx.bfloat16
        if b:
            a = a.view(mx.uint16)
            mx.eval(a)
        views.append(np.asarray(a))
        flags.append(b)
    return views, flags


# --- the index ---------------------------------------------------------------
class _Blob:
    __slots__ = ("uid", "kind", "path", "nbytes", "refs", "state")

    def __init__(self, uid, kind, path, nbytes, state="pending"):
        self.uid, self.kind, self.path, self.nbytes = uid, kind, path, nbytes
        self.refs = 0
        self.state = state                    # pending (queued) | ok (on disk)


class DiskEntry:
    """An entry on disk, as the index keeps it: what best() needs (ids as
    int32 numpy: an LCP is one vector compare), what finish() needs to
    rebuild the _Entry, its blobs."""

    def __init__(self, uid, path, meta, ids):
        self.uid, self.path = uid, path
        self.ids = ids                        # np.int32
        self.L = len(ids)
        self.chain = [tuple(x) for x in meta["chain"]]          # (seg uid, lo, hi)
        self.segs = {u: tuple(ab) for u, ab in meta["segs"].items()}   # uid -> (a, b)
        self.ckpts = {int(p): u for p, u in meta["ckpts"].items()}     # pos -> state uid
        self.end = meta["end"]                                  # state uid or None
        self.layout = meta["layout"]
        self.quantized = meta["quantized"]
        self.low = meta["low"]
        self.reply = meta["reply"]
        self.partial = meta["partial"]
        self.chained = meta["chained"]
        self.diverge = meta["diverge"]
        self.used = meta.get("used", time.time())
        self.state = "pending"
        self.pins = 0                         # loads reading it (never deleted meanwhile)
        self.meta_bytes = 0

    def blob_uids(self):
        return list(self.segs) + list({*self.ckpts.values(), *([self.end] if self.end else [])})

    def best(self, ids, ends):
        """_Entry.best over the int32 ids (the same rule, test_pcdisk checks
        both agree): the deepest resumable position within the common prefix."""
        n = min(self.L, len(ids))
        d = np.flatnonzero(self.ids[:n] != ids[:n])
        common = min(int(d[0]) if d.size else n, len(ids) - 1)
        tops = (*self.ckpts, self.L) if self.end is not None else tuple(self.ckpts)
        pos = [p for p in tops if self.low <= p <= common
               and (p in ends or (p == self.L and self.reply is not None))]
        return max(pos, default=0), common


class Want:
    """A waiting request's view of the disk tier (memoized on _Req.disk)."""
    __slots__ = ("key", "ids", "dent", "depth", "ram", "job", "t0", "held_since")

    def __init__(self):
        self.key = None                       # (disk version, RAM cache version) of the peeks
        self.ids = None                       # its lookup ids as int32
        self.dent, self.depth, self.ram = None, 0, 0
        self.job = None                       # its load (one per request)
        self.t0 = 0.0
        self.held_since = None                # _fits: first waited for the tier's held RAM


class _Cancelled(Exception):
    """A load stopped by cancel_loads (memory is short)."""


class _Load:
    __slots__ = ("dent", "req", "live", "data", "error", "done", "ms", "nbytes", "taken", "held",
                 "cancelled")

    def __init__(self, dent, req, live):
        self.dent, self.live = dent, live     # live: uid -> object still in RAM (not read)
        self.req = weakref.ref(req) if req is not None else (lambda: None)
        self.data = {}                        # uid -> (header, [mx arrays])
        self.error = None
        self.done = threading.Event()
        self.ms = 0.0
        self.nbytes = 0
        self.taken = False                    # finished / discarded by the scheduler
        self.held = 0                         # charged in DiskTier.held until then
        self.cancelled = False                # cancel_loads(): the loader stops, it is dropped


class _Write:
    __slots__ = ("dent", "new", "borrowed", "nbytes", "meta", "L", "covers", "held", "started",
                 "dropped")

    def __init__(self, dent, new, borrowed, nbytes, meta, covers=(), held=0):
        self.dent, self.new, self.borrowed, self.nbytes, self.meta = dent, new, borrowed, nbytes, meta
        self.L = dent.L
        self.covers = covers                  # disk entries it makes redundant (dropped once written)
        self.held = held                      # RAM charged for it (DiskTier.held) until written
        self.started = 0.0                    # when the writer took it (0: still queued)
        self.dropped = False                  # dropped while queued (a stuck disk): never written


class DiskTier:
    """The disk tier of one worker. `seg_cls` / `entry_cls`: worker._Seg /
    worker._Entry (built back by finish). Thread roles: module docstring.
    The index (entries, blobs, counters) is guarded by `lock`."""

    def __init__(self, root, limit, key, *, seg_cls, entry_cls, keep=False,
                 queue_bytes=None, min_tokens=None):
        self.root = os.path.abspath(root)
        self.limit = int(limit)
        self.queue_bytes = int(QUEUE_GB * GIB if queue_bytes is None else queue_bytes)
        self.min_tokens = MIN_TOKENS if min_tokens is None else min_tokens
        self.seg_cls, self.entry_cls = seg_cls, entry_cls
        self.key, self.keep = key, keep
        self.lock = threading.Lock()
        self.idle = threading.Condition(self.lock)
        self.entries: dict[str, DiskEntry] = {}
        self.blobs: dict[str, _Blob] = {}
        self.version = 0                      # bumped when an entry becomes loadable / goes
        self.used = 0                         # bytes on disk (blobs + entry files)
        self.pending = 0                      # file bytes queued for the writer (disk space)
        self.held = 0                         # RAM held for the scheduler's count (held_bytes)
        self.closed = False                   # close(): nothing new taken
        self.stalled = False                  # a write ran past TIMEOUT_S: nothing new taken
        self.timeout_s = TIMEOUT_S
        self._writes = []                     # write jobs queued or running (check_stall)
        self._loads = []                      # loads not finished / dropped (cancel_loads)
        self.queued = 0                       # writer + loader jobs not done
        self.offers = []                      # evicted entries, written at flush
        self.on_drained = None                # called by the writer when its queue empties
        # counters (Health-style: plain ints; the log prints them)
        self.writes = self.write_bytes = self.skipped = self.errors = self.deleted = 0
        self.write_s = 0.0
        self.loads = self.load_bytes = self.hits = self.discards = 0
        self.read_ms = 0.0
        # identity of live objects -> blob uid (dedup); weak: never keeps anything alive
        self._seg_uid = weakref.WeakKeyDictionary()       # _Seg -> uid
        self._segs = weakref.WeakValueDictionary()        # uid -> _Seg
        self._st = {}                                     # id(first array) -> (uid, ref, ids)
        self._classes = {}
        self._wq, self._lq = queue.SimpleQueue(), queue.SimpleQueue()
        self._done = []                                   # finished loads (scheduler polls)
        os.makedirs(self.root, exist_ok=True)
        # one worker per directory: a second one would clear / LRU the first's
        # files under it (flock, held for the process's life; released at exit)
        self._lockf = open(os.path.join(self.root, "pcdisk.lock"), "a")
        try:
            fcntl.flock(self._lockf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lockf.close()
            raise RuntimeError(f"{self.root} is in use by another worker (pcdisk.lock); give "
                               f"each worker its own QWEN_PROMPT_CACHE_DISK") from None
        self._open(keep)
        for name, q, fn in (("pcdisk-writer", self._wq, self._run_write),
                            ("pcdisk-loader", self._lq, self._run_load)):
            threading.Thread(target=self._loop, args=(q, fn), name=name, daemon=True).start()

    # --- directory ------------------------------------------------------------
    def _ours(self):
        for n in os.listdir(self.root):
            if (n.startswith(("seg-", "st-", "ent-")) and n.endswith((".bin", ".bin.tmp"))) \
                    or n in ("pcdisk.json", "pcdisk.json.tmp"):
                yield n

    def _open(self, keep):
        """KEEP and the same key: the previous run's complete entries come
        back (an entry whose blobs are missing or torn is dropped, orphan
        blobs and .tmp files deleted); else every file of ours goes."""
        kpath = os.path.join(self.root, "pcdisk.json")
        old = None
        if keep:
            try:
                with open(kpath) as f:
                    old = json.load(f)
            except (OSError, ValueError):
                old = None
        want = {"format": FORMAT, "key": self.key}
        if old != want:
            if keep and old is not None:
                log.info("prompt cache disk: %s holds another key (model / numerics flags / "
                         "code changed): cleared", self.root)
            for n in list(self._ours()):
                try:
                    os.remove(os.path.join(self.root, n))
                except OSError:
                    log.exception("prompt cache disk: cannot remove %s", n)
            tmp = kpath + ".tmp"
            with open(tmp, "w") as f:
                json.dump(want, f)
            os.replace(tmp, kpath)
            return
        sizes = {}
        for n in self._ours():
            p = os.path.join(self.root, n)
            if n.endswith(".tmp"):
                os.remove(p)
                continue
            if n.startswith(("seg-", "st-")):
                try:
                    _read_blob(p, skip=True)
                    sizes[n] = os.path.getsize(p)
                except (OSError, ValueError):
                    os.remove(p)
        for n in sorted(x for x in self._ours() if x.startswith("ent-")):
            p = os.path.join(self.root, n)
            try:
                h, (ids,) = _read_blob(p)
                d = DiskEntry(h["uid"], p, h["meta"], ids.astype(np.int32))
                files = [self._blob_name(u, "seg") for u in d.segs] + \
                    [self._blob_name(u, "st") for u in set(d.ckpts.values()) | {d.end} - {None}]
                if not all(f in sizes for f in files):
                    raise ValueError("missing blobs")
            except (OSError, ValueError, KeyError):
                log.warning("prompt cache disk: dropping %s (unreadable or incomplete)", n)
                os.remove(p)
                continue
            d.state, d.meta_bytes = "ok", os.path.getsize(p)
            self.entries[d.uid] = d
            self.used += d.meta_bytes
            for u in d.blob_uids():
                kind = "seg" if u in d.segs else "st"
                b = self.blobs.get(u)
                if b is None:
                    name = self._blob_name(u, kind)
                    b = self.blobs[u] = _Blob(u, kind, os.path.join(self.root, name),
                                              sizes[name], "ok")
                    self.used += b.nbytes
                b.refs += 1
        for n, _ in sizes.items():                    # orphans
            u = n.split("-", 1)[1][:-4]
            if u not in self.blobs:
                os.remove(os.path.join(self.root, n))
        with self.lock:
            self._unlink_locked(self._fit(0))
        log.info("prompt cache disk: %d entries (%.2f GiB) kept from the previous run in %s",
                 len(self.entries), self.used / GIB, self.root)

    def _blob_name(self, uid, kind):
        return f"{kind}-{uid}.bin"

    # --- identity of live objects (scheduler thread) ----------------------------
    def _seg_id(self, seg, assign=True):
        u = self._seg_uid.get(seg)
        if u is None and assign:
            u = _new_uid()
            self._seg_uid[seg] = u
            self._segs[u] = seg
        return u

    @staticmethod
    def _st_arrays(st):
        return [a for layer in st for a in layer if a is not None]

    def _st_id(self, st, assign=True, uid=None):
        """A state's uid: by the identity of its arrays (the first one held
        weakly, the others' ids compared: states are lists, shared by
        reference between entries, never weakref-able themselves). `uid`:
        register `st` under it (a state just loaded from that blob)."""
        arrs = self._st_arrays(st)
        if not arrs:
            return (uid or _new_uid()) if assign else None
        k, ids = id(arrs[0]), tuple(map(id, arrs))
        rec = self._st.get(k)
        if uid is None and rec is not None and rec[1]() is arrs[0] and rec[2] == ids:
            return rec[0]
        if not assign:
            return None
        u = uid or _new_uid()

        def gone(r, k=k):
            cur = self._st.get(k)
            if cur is not None and cur[1] is r:
                self._st.pop(k, None)
        self._st[k] = (u, weakref.ref(arrs[0], gone), ids)
        return u

    # --- write: eviction -> file (offer, flush: scheduler thread) ----------------
    def offer(self, e, held=0, own=None):
        """An entry the RAM cache just evicted (no MLX, no I/O) -> True: it
        is taken (written at the next flush, the end of the scheduler's
        tick), its `held` bytes (the RAM its eviction would free: the
        caller's count; `own`: the ids of those arrays) charged in
        held_bytes() until its file is written (at flush only the arrays the
        writer keeps stay charged: not those of blobs already on disk);
        False: not taken, the eviction frees its RAM at once as without the
        tier (too short, the writer QUEUE_GB behind, the disk stuck, the tier
        closed)."""
        L = len(e.ids)
        with self.lock:
            if self.closed or self.stalled:
                return False
            if L < self.min_tokens:
                self.skipped += 1
                return False
            if held and self.held + held > self.queue_bytes:
                self.skipped += 1
                why = (f"the writer holds {self.held / GIB:.2f} GiB, +{held / GIB:.2f} would pass "
                       f"QWEN_PROMPT_CACHE_DISK_QUEUE_GB ({self.queue_bytes / GIB:.1f})")
            else:
                self.held += held
                self.offers.append((e, held, own))
                return True
        log.info("prompt cache disk: evicted entry of %d tokens not written: %s", L, why)
        return False

    def held_bytes(self):
        """RAM the tier holds outside the RAM cache's count: evicted entries
        until their files are written, loads from their start until inserted
        or dropped. The scheduler charges it beside pc.bytes."""
        with self.lock:
            return self.held

    def _release(self, n):
        with self.lock:
            self.held -= n
            self.idle.notify_all()

    def wait(self, timeout):
        """Scheduler thread: until a writer / loader job ends (held may have
        gone down) or `timeout`."""
        with self.lock:
            self.idle.wait(timeout)

    def cancel_loads(self):
        """The loads not inserted yet go first when memory is short
        (_Scheduler._disk_room): the loader stops at its next array, the
        scheduler drops them (done(); their hold released there) -> how many."""
        with self.lock:
            jobs = [j for j in self._loads if not j.taken]
        for j in jobs:
            j.cancelled = True
        return len(jobs)

    def check_stall(self):
        """Scheduler thread (each flush): a write running past TIMEOUT_S is a
        stuck disk: the writes queued behind it are dropped (their views
        freed here, their RAM released, never written: as an eviction
        without the tier), nothing more is taken until it returns. Its own
        buffers stay held: the writer thread is reading them."""
        if not self.timeout_s:
            return
        now = time.time()
        with self.lock:
            stuck = any(j.started and now - j.started > self.timeout_s for j in self._writes)
            if not stuck or self.stalled:
                return
            self.stalled = True
            drop = [j for j in self._writes if not j.started]
            for j in drop:
                j.dropped = True
                self._writes.remove(j)
                self.held -= j.held
                self.pending -= j.nbytes
                j.held = j.nbytes = 0
                self._unregister(j.dent, [b for b, *_ in j.new])
                j.new = None                          # the views: their MLX buffers go
            self.errors += 1
            self.idle.notify_all()
        log.error("prompt cache disk: a write has run for over %.0f s (stuck disk?): %d queued "
                  "writes dropped, nothing more written until it returns", self.timeout_s,
                  len(drop))

    def flush(self):
        """Scheduler thread: the offered entries to the writer. Never raises."""
        try:
            self.check_stall()
        except Exception:  # noqa: BLE001
            log.exception("prompt cache disk: stall check failed")
        offers, self.offers = self.offers, []
        for e, held, own in offers:
            try:
                held = self._offer(e, held, own)
            except Exception:  # noqa: BLE001 — costs the entry's file only
                self.errors += 1
                log.exception("prompt cache disk: entry of %d tokens not written", len(e.ids))
            finally:
                if held:                              # not held by a write job: freed now
                    self._release(held)

    def _offer(self, e, held, own=None):
        """-> the held bytes it did not hand to a write job (all of them when
        no job; else those of arrays the job does not keep: blobs already on
        disk)."""
        L = len(e.ids)
        old = getattr(e, "disk_uid", None)
        with self.lock:
            d = self.entries.get(old) if old is not None else None
            if d is not None:                         # loaded from here, unchanged since
                d.used = time.time()
                return held
        segs = {}
        for s, lo, hi in e.chain:
            segs.setdefault(id(s), s)
        states = {}
        for st in [*e.ckpts.values(), *([e.end] if e.end is not None else [])]:
            states.setdefault(id(st), st)
        seg_u = {k: self._seg_id(s) for k, s in segs.items()}
        st_u = {k: self._st_id(st) for k, st in states.items()}
        need = set(seg_u.values()) | set(st_u.values())
        why = None
        with self.lock:
            fresh = [("seg", seg_u[k], s) for k, s in segs.items() if seg_u[k] not in self.blobs]
            fresh += [("st", st_u[k], st) for k, st in states.items() if st_u[k] not in self.blobs]
            # the files' sizes (each array padded to a page, the header's
            # pages; the entry file: its ids + a metadata header)
            nbytes = sum(_size([a.nbytes for a in (x.arrays if kind == "seg"
                                                   else self._st_arrays(x))])
                         for kind, _, x in fresh) + _size([4 * L], 64 * len(e.ckpts) + 4096)
            # what goes for it, decided before anything changes: the LRU
            # entries whose files (but the blobs this entry shares: it holds
            # them) make the room; None: no room even with all of them gone
            drops = self._plan(nbytes, need) if nbytes <= self.limit else None
            if nbytes > self.limit:
                why = f"over QWEN_PROMPT_CACHE_DISK_GB ({self.limit / GIB:.1f} GiB)"
            elif drops is None:
                why = "no room on disk (entries being written / read)"
            else:
                uid = _new_uid()
                meta = {
                    "chain": [[seg_u[id(s)], lo, hi] for s, lo, hi in e.chain],
                    "segs": {seg_u[k]: [s.a, s.b] for k, s in segs.items()},
                    "ckpts": {str(p): st_u[id(st)] for p, st in e.ckpts.items()},
                    "end": st_u[id(e.end)] if e.end is not None else None,
                    "layout": self._dump_layout(e.layout), "quantized": bool(e.quantized),
                    "low": int(e.low), "reply": e.reply, "partial": bool(e.partial),
                    "chained": bool(e.chained), "diverge": sorted(int(x) for x in e.diverge),
                    "used": time.time()}
                d = DiskEntry(uid, os.path.join(self.root, f"ent-{uid}.bin"), meta,
                              np.asarray(e.ids, dtype=np.int32))
                new, borrowed = [], []
                try:
                    # its references first: the drops below can never take a
                    # blob it shares (review 2026-10-24 #1: an LRU entry was
                    # the last holder of the shared system segment)
                    for kind, u, _ in fresh:
                        b = self.blobs[u] = _Blob(u, kind, os.path.join(
                            self.root, self._blob_name(u, kind)), 0)
                        new.append(b)
                    for u in d.blob_uids():
                        b = self.blobs[u]
                        b.refs += 1
                        if not any(b is x for x in new):
                            borrowed.append(b)
                    self.entries[uid] = d
                    unlink = []
                    for x in drops:
                        unlink += self._drop(x)
                        self.deleted += 1
                    self._unlink_locked(unlink)
                    # older disk entries whose ids are a prefix of this one
                    # (the turn it continued, loaded back and absorbed in
                    # RAM): as the RAM store absorbs them, the new entry
                    # stands for them once written (an unquantized one stays
                    # under a quantized entry: the only resume point below
                    # quantized_kv_start)
                    covers = [x for x in self.entries.values() if x is not d
                              and x.L <= L and not (d.quantized and not x.quantized)
                              and np.array_equal(x.ids, d.ids[:x.L])]
                    self.pending += nbytes
                    self.queued += 1
                except BaseException:
                    self._unregister(d, new)
                    raise
        if why is not None:
            self.skipped += 1
            log.info("prompt cache disk: evicted entry of %d tokens not written: %s", L, why)
            return held
        # zero-copy views (this thread: MLX); the writer only does file I/O
        jobs = []
        try:
            for (kind, u, x), b in zip(fresh, new):
                if kind == "seg":
                    arrs = x.arrays
                    header = {"kind": "seg", "uid": u, "a": x.a, "b": x.b,
                              "struct": [[len(k) if isinstance(k, tuple) else 0,
                                          len(v) if isinstance(v, tuple) else 0]
                                         for k, v in x.kv]}
                else:
                    arrs = self._st_arrays(x)
                    header = {"kind": "st", "uid": u,
                              "struct": [[a is not None for a in layer] for layer in x]}
                views, flags = _host(arrs)
                jobs.append((b, header, views, flags))
        except BaseException:
            jobs = None                               # no half-registered entry
            with self.lock:
                self._unregister(d, new)
                self.pending -= nbytes
                self.queued -= 1
            raise
        e.disk_uid = uid
        # what the job keeps alive of the RAM the eviction frees: its views'
        # arrays among the evicted entry's own (`own`; the blobs already on
        # disk and the arrays other entries hold are not the writer's to hold)
        keep = held if own is None else min(held, sum(
            a.nbytes for kind, _, x in fresh
            for a in (x.arrays if kind == "seg" else self._st_arrays(x)) if id(a) in own))
        w = _Write(d, jobs, borrowed, nbytes, meta, covers, keep)
        with self.lock:
            self._writes.append(w)
        self._wq.put(w)
        return held - keep

    def _unregister(self, d, new):
        """Under the lock: an entry registered but never handed to the writer
        out again (its own new blobs, never written, out of the index)."""
        if self.entries.get(d.uid) is d:
            self._unlink_locked(self._drop(d))
        for b in new:
            if self.blobs.get(b.uid) is b and b.state != "ok":
                self.blobs.pop(b.uid)

    def _plan(self, nbytes, keep=()):
        """Under the lock: the least recently used entries (written, not
        being read) whose deletion makes `nbytes` more fit, the blobs in
        `keep` never counted as freed (the new entry holds them) -> that
        list (empty: it fits), None: even all of them would not do. Nothing
        changes here."""
        over = self.used + self.pending + nbytes - self.limit
        if over <= 0:
            return []
        refs, out, freed = {}, [], 0
        for d in sorted((d for d in self.entries.values() if d.state == "ok" and not d.pins),
                        key=lambda d: d.used):
            out.append(d)
            freed += d.meta_bytes
            for u in d.blob_uids():
                b = self.blobs.get(u)
                if b is None or u in keep:
                    continue
                refs[u] = refs.get(u, b.refs) - 1
                if refs[u] == 0 and b.state == "ok":
                    freed += b.nbytes
            if freed >= over:
                return out
        return None

    def _fit(self, nbytes):
        """Under the lock: delete least recently used entries until `nbytes`
        more fit (as many as there are when they cannot all) -> the files to
        unlink (the caller queues them, under the lock)."""
        drops = self._plan(nbytes)
        if drops is None:
            drops = sorted((d for d in self.entries.values() if d.state == "ok" and not d.pins),
                           key=lambda d: d.used)
        unlink = []
        for d in drops:
            if self.used + self.pending + nbytes <= self.limit:
                break
            unlink += self._drop(d)
            self.deleted += 1
        return unlink

    def _drop(self, d):
        """Under the lock: an entry out of the index -> its files (and the
        blobs no other entry holds) to unlink."""
        self.entries.pop(d.uid, None)
        self.version += 1
        out = []
        if d.state == "ok":
            self.used -= d.meta_bytes
            out.append(d.path)
        for u in d.blob_uids():
            b = self.blobs.get(u)
            if b is None:
                continue
            b.refs -= 1
            if b.refs <= 0:
                self.blobs.pop(u, None)
                if b.state == "ok":
                    self.used -= b.nbytes
                out.append(b.path)
        return out

    def _unlink_locked(self, paths):
        """Under the lock: the unlink job queued while the index already says
        these files are gone, so a write of the same blob registered after it
        (a re-offer of a live segment) is queued after it (FIFO): a late
        unlink of the old file can never delete the new one (review #5)."""
        if paths:
            self.queued += 1
            self._wq.put(paths)

    def _dump_layout(self, layout):
        out = []
        for lay in layout:
            cls = lay[1]
            self._classes[(cls.__module__, cls.__qualname__)] = cls
            name = [cls.__module__, cls.__qualname__]
            if lay[0] == "arr":
                out.append(["arr", name, lay[2]])
            elif lay[0] == "q":
                out.append(["q", name, lay[2], lay[3], lay[4].__name__])
            else:
                out.append(["kv", name])
        return out

    def _load_layout(self, layout):
        def cls(name):
            c = self._classes.get(tuple(name))
            if c is None:
                c = importlib.import_module(name[0])
                for part in name[1].split("."):
                    c = getattr(c, part)
                self._classes[tuple(name)] = c
            return c
        out = []
        for lay in layout:
            if lay[0] == "arr":
                out.append(("arr", cls(lay[1]), lay[2]))
            elif lay[0] == "q":
                out.append(("q", cls(lay[1]), lay[2], lay[3], {"tuple": tuple, "list": list}[lay[4]]))
            else:
                out.append(("kv", cls(lay[1])))
        return out

    # --- background threads -------------------------------------------------------
    def _loop(self, q, fn):
        while True:
            job = q.get()
            try:
                fn(job)
            except Exception:  # noqa: BLE001 — a disk thread never dies
                log.exception("prompt cache disk: %s failed", threading.current_thread().name)
            finally:
                job = None                            # the views go here, not at the next get
                with self.lock:
                    self.queued -= 1
                    drained = self.queued == 0
                    self.idle.notify_all()
                if drained and self.on_drained is not None:
                    try:
                        # the scheduler's idle pass: clear_cache, a load
                        # nobody waits for any more freed (_Scheduler._idle)
                        self.on_drained()
                    except Exception:  # noqa: BLE001
                        log.exception("prompt cache disk: on_drained failed")

    def _run_write(self, job):
        if isinstance(job, list):                     # unlink
            for p in job:
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass
                except OSError:
                    log.exception("prompt cache disk: cannot remove %s", p)
            return
        t = time.time()
        with self.lock:                               # check_stall drops only queued jobs
            if job.dropped:
                return
            job.started = t
        d, new = job.dent, [b for b, *_ in job.new]
        try:
            for b in job.borrowed:                    # written by an earlier job (FIFO)
                with self.lock:
                    if self.blobs.get(b.uid) is not b or b.state != "ok":
                        raise RuntimeError(f"shared blob {b.uid} is not on disk")
            written = 0
            for b, header, views, flags in job.new:
                n = _write_blob(b.path, header, views, flags, sync=self.keep)
                with self.lock:
                    b.nbytes, b.state = n, "ok"
                    self.used += n
                written += n
            views = job.new = None                    # the MLX buffers go (written)
            n = _write_blob(d.path, {"kind": "ent", "uid": d.uid, "meta": job.meta}, [d.ids],
                            sync=self.keep)
            with self.lock:
                d.meta_bytes, d.state = n, "ok"
                self.used += n
                self.pending -= job.nbytes
                job.nbytes = 0
                self.version += 1
                self.writes += 1
                self.write_bytes += written + n
                self.write_s += time.time() - t
                over = []
                for x in job.covers:                  # written: the entries it stands for go
                    if self.entries.get(x.uid) is x and x.state == "ok" and not x.pins:
                        over += self._drop(x)
                over += self._fit(0)                  # the estimate was short: LRU again
                self._unlink_locked(over)             # under the lock (review #5)
            log.info("prompt cache disk: wrote an entry of %d tokens (%.1f MB new, %d blobs "
                     "shared) in %.2f s; %d entries, %.2f GiB on disk", job.L, (written + n) / 1e6,
                     len(job.borrowed), time.time() - t, len(self.entries), self.used / GIB)
        except Exception:  # noqa: BLE001 — the entry is lost, as without the disk tier
            job.new = None
            with self.lock:
                self.errors += 1
                # its blobs never written leave the index now (a later entry
                # borrowing one fails its check, a new offer writes it again);
                # the written ones stay while another entry holds them
                for b in new:
                    if b.state != "ok" and self.blobs.get(b.uid) is b:
                        self.blobs.pop(b.uid)
                paths = self._drop(d)
            for p in paths + [b.path + ".tmp" for b in new] + [d.path + ".tmp"]:
                try:
                    os.remove(p)
                except OSError:
                    pass
            log.exception("prompt cache disk: writing an entry of %d tokens failed (lost, as "
                          "an eviction without the disk tier)", job.L)
        finally:
            # the views are gone (job.new): the evicted entry's RAM is free
            with self.lock:
                self.pending -= job.nbytes
                self.held -= job.held
                job.held = 0
                if job in self._writes:
                    self._writes.remove(job)
                if self.stalled and not any(j.started for j in self._writes):
                    self.stalled = False
                    log.warning("prompt cache disk: the stuck write returned after %.0f s; "
                                "writing again", time.time() - t)

    def _run_load(self, job):
        t = time.time()
        try:
            for u in job.dent.blob_uids():
                if u in job.live:
                    continue
                with self.lock:
                    b = self.blobs.get(u)
                if b is None or b.state != "ok":
                    raise RuntimeError(f"blob {u} is not on disk")
                # each array to a constant MLX array (allocation + copy, no
                # graph: as to_mlx) as soon as it is read: the transient is one
                # array, not the blob (review #3: 2x a 1.6 GB first segment)
                def each(a):
                    if job.cancelled:                 # memory is short (cancel_loads)
                        raise _Cancelled()
                    return mx.array(a)
                if job.cancelled:
                    raise _Cancelled()
                h, arrs = _read_blob(b.path, each=each)
                if h.get("uid") != u:
                    raise ValueError(f"{b.path}: holds {h.get('uid')}")
                job.nbytes += sum(a.nbytes for a in arrs)
                job.data[u] = (h, arrs)
                del arrs
        except _Cancelled as e:
            job.error = e
            job.data = {}
            log.info("prompt cache disk: loading an entry of %d tokens cancelled (memory is short)",
                     job.dent.L)
        except Exception as e:  # noqa: BLE001 — a failed load is a miss
            job.error = e
            job.data = {}
            log.warning("prompt cache disk: loading an entry of %d tokens failed: %s: %s",
                        job.dent.L, type(e).__name__, e)
        job.ms = (time.time() - t) * 1000
        with self.lock:
            self._done.append(job)
        job.done.set()

    # --- lookup / load (scheduler thread) ------------------------------------------
    def peek(self, ids, ends):
        """-> (the written entry with the deepest resumable position for the
        int32 `ids`, that position) — DiskEntry.best over the index."""
        with self.lock:
            ents = [d for d in self.entries.values() if d.state == "ok"]
        hit, at = None, 0
        for d in ents:
            pos, _ = d.best(ids, ends)
            if pos > at:
                hit, at = d, pos
        return hit, at

    def busy(self):
        """A load is reading (one at a time: the SSD is the bottleneck)."""
        with self.lock:
            return any(d.pins for d in self.entries.values())

    def _live(self, d, ram_entries):
        """The blobs of `d` whose objects are alive in RAM (not read)."""
        live = {}
        for u in d.segs:
            s = self._segs.get(u)
            if s is not None:
                live[u] = s
        want = set(d.ckpts.values()) | ({d.end} if d.end else set())
        for e in ram_entries:
            for st in [*e.ckpts.values(), *([e.end] if e.end is not None else [])]:
                u = self._st_id(st, assign=False)
                if u in want:
                    live[u] = st
        return live

    def cost(self, d, ram_entries):
        """Bytes loading `d` would add to RAM (its blobs not alive there)."""
        live = self._live(d, ram_entries)
        with self.lock:
            return sum(b.nbytes for u in d.blob_uids() if u not in live
                       for b in [self.blobs.get(u)] if b is not None)

    def load(self, d, ram_entries, req=None, need=None):
        """Start reading `d` (pinned until finish / discard) -> the job, or
        None: `d` left the index since the peek (the writer's LRU), or the
        tier is closed. `need` (cost(); computed here when None) is charged
        in held_bytes() from now until finish / discard: the scheduler's
        budget checks see the load before its arrays exist (review #2)."""
        live = self._live(d, ram_entries)
        if need is None:
            need = self.cost(d, ram_entries)
        job = _Load(d, req, live)
        with self.lock:
            if self.closed or self.entries.get(d.uid) is not d or d.state != "ok":
                return None
            d.pins += 1
            d.used = time.time()
            job.held = need
            self.held += need
            self.queued += 1
            self._loads.append(job)
        self._lq.put(job)
        return job

    def done(self):
        """The loads that finished since the last call."""
        with self.lock:
            out, self._done = self._done, []
        return out

    def _unpin(self, job):
        if job.taken:
            return
        job.taken = True
        with self.lock:
            job.dent.pins -= 1
            self.held -= job.held                     # inserted (pc.bytes now) or dropped
            job.held = 0
            if job in self._loads:
                self._loads.remove(job)
            self.idle.notify_all()

    def discard(self, job):
        self._unpin(job)
        job.data, job.live = {}, {}
        self.discards += 1

    def finish(self, job):
        """Scheduler thread: the loaded entry as a worker._Entry (its blobs
        that were alive in RAM are those objects), or None (the load
        failed). Its arrays are evaluated (bf16: a view of the uint16 load)."""
        d = job.dent
        try:
            if job.error is not None or job.cancelled:
                if job.cancelled:
                    self.discards += 1
                else:
                    self.errors += 1
                return None
            objs = dict(job.live)
            for u, (h, arrs) in job.data.items():
                arrs = [a.view(mx.bfloat16) if s.get("bf16") else a
                        for a, s in zip(arrs, h["arrays"])]
                it = iter(arrs)
                if h["kind"] == "seg":
                    kv = []
                    for nk, nv in h["struct"]:
                        k = tuple(next(it) for _ in range(nk)) if nk else next(it)
                        v = tuple(next(it) for _ in range(nv)) if nv else next(it)
                        kv.append((k, v))
                    s = self.seg_cls(h["a"], h["b"], kv)
                    self._seg_uid[s] = u
                    self._segs[u] = s
                    objs[u] = s
                else:
                    st = [[next(it) if f else None for f in layer] for layer in h["struct"]]
                    objs[u] = st
                    self._st_id(st, uid=u)
            chain = [(objs[u], lo, hi) for u, lo, hi in d.chain]
            ckpts = {p: objs[u] for p, u in d.ckpts.items()}
            end = objs[d.end] if d.end is not None else None
            e = self.entry_cls(d.ids.tolist(), chain, ckpts, end, self._load_layout(d.layout),
                               d.quantized, d.low, d.reply, d.partial)
            e.diverge = frozenset(d.diverge)
            e.chained = d.chained
            e.disk_uid = d.uid
            mx.eval(e.arrays)
            self.loads += 1
            self.load_bytes += job.nbytes
            self.read_ms += job.ms
            return e
        finally:
            self._unpin(job)
            job.data, job.live = {}, {}

    # --- tools / tests ------------------------------------------------------------
    def drain(self, timeout=30.0):
        """Wait until the writer and the loader are idle -> False on timeout."""
        with self.lock:
            return self.idle.wait_for(lambda: self.queued == 0, timeout)

    def close(self, timeout=10.0):
        """Shutdown (worker main, after the scheduler stopped): nothing new
        is taken; wait for the writer and the loader to go idle, so neither
        is inside a file write over an MLX buffer or an mx.array when the
        interpreter tears MLX down (their threads are daemons, blocked in
        their queue's get then) -> False on timeout. The offers not flushed
        are dropped (the RAM cache is not written at shutdown)."""
        with self.lock:
            self.closed = True
            offers, self.offers = self.offers, []
            self.held -= sum(h for _, h in offers)
        ok = self.drain(timeout)
        if ok:                                        # the directory is free for another one
            self._lockf.close()
        return ok

    def stats(self):
        with self.lock:
            return {"entries": len(self.entries), "bytes": self.used, "writes": self.writes,
                    "write_bytes": self.write_bytes, "write_s": round(self.write_s, 3),
                    "skipped": self.skipped, "errors": self.errors, "deleted": self.deleted,
                    "loads": self.loads, "load_bytes": self.load_bytes, "hits": self.hits,
                    "discards": self.discards, "read_ms": round(self.read_ms, 1),
                    "pending": self.pending, "held": self.held}


def key_of(**parts):
    """A stable digest of everything that moves the cached numbers."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()
