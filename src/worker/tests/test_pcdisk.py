"""The prompt cache's disk tier (QWEN_PROMPT_CACHE_DISK, pcdisk.py) on the
tiny random model (CPU), docs/perf/2026-10-24/design/pcdisk.md: an entry
evicted from RAM is written, a waiting request whose deepest hit is on disk
gets it loaded and resumes from it exactly as from RAM (== cold), for fp32,
fp16 and quantized KV; shared segments / states are blobs written once and
reused live; the disk limit is LRU; I/O errors never fail a request; off
changes nothing."""
import json
import logging
import os

import mlx.core as mx
import numpy as np
import pytest

import pcdisk
import worker
from conftest import IM_START, Cold, check_cold, prompts, request, run, scheduler, same
from test_prompt_cache import GEN, conversation, small  # noqa: F401

BIG = 1e3                         # QWEN_PROMPT_CACHE_GB: nothing is evicted
TINY = 1e-9                       # every store keeps the newest entry alone


def tier(path, limit=1 << 30, key="k", **kw):
    kw.setdefault("min_tokens", 0)
    return pcdisk.DiskTier(str(path), limit, key, seg_cls=worker._Seg,
                           entry_cls=worker._Entry, **kw)


def files(path):
    return sorted(n for n in os.listdir(path) if n not in ("pcdisk.json", "pcdisk.lock"))


def turns(seed=60, system=40):
    """A two-turn conversation and an unrelated prompt."""
    a = conversation(3, seed, system=system, user=lambda i: 9, answer=lambda i: 10)
    b = [IM_START] + prompts([50], seed + 7)[0] + GEN
    return a, b


def play(sched, a, b, monkeypatch, gb, disk=None):
    """a[0], then b (with TINY: evicts a[0]'s entry), then a[1] -> the
    answers and a[1]'s request."""
    monkeypatch.setattr(worker, "PROMPT_CACHE_GB", gb)
    out = {}
    for rid, ids in (("a0", a[0]), ("b", b)):
        out[rid] = run(sched, [request(ids, 4, rid)])[rid]
        if disk is not None:
            assert disk.drain()
    x = request(a[1], 6, "a1")
    out["a1"] = run(sched, [x])["a1"]
    return out, x


def roundtrip(lm, tmp_path, monkeypatch, caplog, **kw):
    a, b = turns()
    ref_sched = scheduler(lm, **kw)                       # all in RAM
    ref, rx = play(ref_sched, a, b, monkeypatch, BIG)
    assert rx.reused >= worker.PREFILL_STEP
    off_sched = scheduler(lm, **kw)                       # evicted, no disk: a MISS
    off, ox = play(off_sched, a, b, monkeypatch, TINY)
    assert ox.reused < rx.reused
    caplog.set_level(logging.INFO, logger="qwen.worker")
    sched = scheduler(lm, **kw)
    pc = sched.eng._pcache
    disk = pc.disk = tier(tmp_path)
    got, x = play(sched, a, b, monkeypatch, TINY, disk)
    # a hit through the disk == the same hit from RAM == cold
    assert got == ref == off
    assert x.reused == rx.reused
    st = disk.stats()
    assert st["loads"] == st["hits"] == 1 and st["load_bytes"] > 0 and st["errors"] == 0
    assert st["writes"] >= 2                              # a[0], then b (evicted by the load)
    assert any("prompt cache disk: entry of" in m and "id=a1" in m for m in caplog.messages)
    assert check_cold(pc, Cold(lm, **kw))                 # a[1]'s entry, built on the loaded one
    assert disk.drain()
    return sched, disk


def test_disk_hit_equals_ram_hit_fp32(dense, small, tmp_path, monkeypatch, caplog):
    roundtrip(dense, tmp_path, monkeypatch, caplog)


def test_disk_hit_equals_ram_hit_fp16(dense_fused, small, tmp_path, monkeypatch, caplog):
    roundtrip(dense_fused, tmp_path, monkeypatch, caplog)


def test_disk_hit_equals_ram_hit_quantized(dense, small, tmp_path, monkeypatch, caplog):
    kw = dict(kv_bits=8, kv_start=24)
    sched, _ = roundtrip(dense, tmp_path, monkeypatch, caplog, **kw)
    assert all(e.quantized for e in sched.eng._pcache.entries)


def test_loaded_entry_bits(dense, small, tmp_path, monkeypatch):
    """The entry finish() builds restores the same bits as the evicted one at
    every resumable position, and its metadata is the same."""
    a, b = turns(70)
    sched = scheduler(dense)
    pc = sched.eng._pcache
    disk = pc.disk = tier(tmp_path)
    monkeypatch.setattr(worker, "PROMPT_CACHE_GB", BIG)
    run(sched, [request(a[0], 4, "a0")])
    (e,) = pc.entries
    pos = sorted([*e.ckpts, len(e.ids)])
    want = {p: [np.array(x) for x in _flat(pc.restore(e, p))] for p in pos}
    meta = (e.ids, sorted(e.ckpts), e.low, e.reply, e.partial, e.chained, e.quantized)
    monkeypatch.setattr(worker, "PROMPT_CACHE_GB", TINY)
    run(sched, [request(b, 4, "b")])                    # evicts it
    del e
    assert disk.drain()
    d, depth = disk.peek(np.asarray(a[0], np.int32), set(worker._bounds(a[0], IM_START))
                         | {len(a[0]) - 1})
    assert d is not None and depth > 0
    job = disk.load(d, pc.entries)
    assert job.done.wait(10) and disk.drain()
    assert disk.done() == [job]
    got = disk.finish(job)
    assert (got.ids, sorted(got.ckpts), got.low, got.reply, got.partial, got.chained,
            got.quantized) == meta
    assert job.nbytes > 0 and not d.pins
    for p in pos:
        have = [np.array(x) for x in _flat(pc.restore(got, p))]
        assert len(have) == len(want[p])
        assert all(u.dtype == v.dtype and u.shape == v.shape and u.tobytes() == v.tobytes()
                   for u, v in zip(have, want[p]))


def _flat(caches):
    from conftest import view
    return view(caches)


def test_best_matches_entry_best(dense, small, tmp_path, monkeypatch):
    """DiskEntry.best (int32 ids) == _Entry.best on the same entry, for
    prompts that share none, part or all of it."""
    a, b = turns(80)
    sched = scheduler(dense)
    pc = sched.eng._pcache
    disk = pc.disk = tier(tmp_path)
    monkeypatch.setattr(worker, "PROMPT_CACHE_GB", BIG)
    for i, ids in enumerate(a):
        run(sched, [request(ids, 4, f"a{i}")])
    run(sched, [request(b, 4, "b")])
    entries = list(pc.entries)
    for e in entries:
        disk.offer(e)
    disk.flush()
    assert disk.drain()
    by = {tuple(d.ids.tolist()): d for d in disk.entries.values()}
    probes = a + [b, a[1][:30] + prompts([20], 81)[0], a[2][:-3]]
    for e in entries:
        d = by[tuple(e.ids)]
        for ids in probes:
            ends = set(worker._bounds(ids, IM_START)) | {len(ids) - 1}
            assert d.best(np.asarray(ids, np.int32), ends) == e.best(ids, ends)


def test_shared_blobs_written_once_and_reused_live(dense, small, tmp_path, monkeypatch):
    """Two conversations over one system block: the system segment and the
    states they share are one blob each; an entry loaded from disk and
    evicted again unchanged is not rewritten; a later turn built on a loaded
    entry writes only its own new segment and states."""
    sysb = [IM_START] + prompts([40], 90)[0]
    p1 = sysb + [IM_START] + prompts([30], 91)[0] + GEN
    p2 = sysb + [IM_START] + prompts([30], 92)[0] + GEN
    sched = scheduler(dense)
    pc = sched.eng._pcache
    disk = pc.disk = tier(tmp_path)
    monkeypatch.setattr(worker, "PROMPT_CACHE_GB", BIG)
    run(sched, [request(p1, 4, "p1")])
    r2 = request(p2, 4, "p2")
    run(sched, [r2])
    assert r2.reused > 0                                    # resumed in the system block
    e1, e2 = pc.entries
    shared = {id(s) for s, _, _ in e1.chain} & {id(s) for s, _, _ in e2.chain}
    assert shared
    for e in (e1, e2):
        pc.evict(keep=[x for x in pc.entries if x is not e])
    disk.flush()
    assert disk.drain() and disk.stats()["writes"] == 2
    d1, d2 = (disk.entries[e.disk_uid] for e in (e1, e2))
    common = set(d1.blob_uids()) & set(d2.blob_uids())
    assert {u for u in common if u in d1.segs}             # the system segment: one blob
    assert len(files(tmp_path)) == len(disk.blobs) + 2      # no duplicate files
    # live reuse: e2 still alive in this test -> loading d1 reads only its own blobs
    pc.entries.append(e2)
    job = disk.load(d1, pc.entries)
    assert job.done.wait(10) and disk.done() == [job]
    assert common <= set(job.live)
    got = disk.finish(job)
    assert any(s is t for s, _, _ in got.chain for t, _, _ in e2.chain)
    # evicted again unchanged: touched, not rewritten
    pc.entries = [got]
    pc._recount()
    w = disk.stats()["writes"]
    pc.evict()
    disk.flush()
    assert disk.drain() and disk.stats()["writes"] == w
    assert len(files(tmp_path)) == len(disk.blobs) + 2


def test_next_turn_on_a_loaded_entry(dense, small, tmp_path, monkeypatch, caplog):
    """a[1] resumed from a[0] loaded back from disk and absorbed it in RAM;
    evicted in its turn, a[1]'s entry writes only what a[0]'s blobs do not
    hold, and a[0]'s disk entry goes (a[1]'s stands for it, as in RAM)."""
    a, b = turns(190)
    sched = scheduler(dense)
    pc = sched.eng._pcache
    disk = pc.disk = tier(tmp_path)
    play(sched, a, b, monkeypatch, TINY, disk)
    (e1,) = pc.entries
    assert e1.ids == a[1][:-1]
    old = [d for d in disk.entries.values() if d.ids.tolist() == a[0][:-1]]
    assert len(old) == 1
    shared = set(old[0].blob_uids())
    caplog.set_level(logging.INFO, logger="qwen.worker")
    run(sched, [request([IM_START] + prompts([50], 199)[0] + GEN, 2, "c")])
    assert disk.drain()
    (line,) = [m for m in caplog.messages if f"wrote an entry of {len(e1.ids)} tokens" in m]
    assert " 0 blobs shared" not in line
    d1 = disk.entries[e1.disk_uid]
    assert shared & set(d1.blob_uids())                     # a[0]'s segment / states reused
    assert not [d for d in disk.entries.values() if d.ids.tolist() == a[0][:-1]]
    assert len(files(tmp_path)) == len(disk.blobs) + len(disk.entries)
    x = request(a[2], 4, "a2")                              # and it serves the next turn
    assert run(sched, [x])["a2"] == run(scheduler(dense, max_batch=1), [request(a[2], 4, "a2")])["a2"]
    assert x.reused > len(a[1]) // 2 and disk.stats()["hits"] == 2


def test_lru_limit(dense, small, tmp_path, monkeypatch):
    """QWEN_PROMPT_CACHE_DISK_GB: the least recently used entries go, their
    files with them (no orphan blob), bytes == the files' sizes; an entry
    alone over the limit is not written."""
    ps = [[IM_START] + p + GEN for p in prompts([60] * 5, 100)]
    sched = scheduler(dense)
    pc = sched.eng._pcache
    probe = tier(tmp_path / "probe", min_tokens=0)
    pc.disk = probe
    monkeypatch.setattr(worker, "PROMPT_CACHE_GB", TINY)
    run(sched, [request(ps[0], 2, "x")])
    run(sched, [request(ps[1], 2, "y")])
    assert probe.drain()
    one = probe.stats()["bytes"]
    sched = scheduler(dense)                                # unrelated prompts: no load
    pc = sched.eng._pcache
    disk = pc.disk = tier(tmp_path / "d", limit=int(one * 2.5))
    for i, p in enumerate(ps):
        run(sched, [request(p, 2, f"p{i}")])
        assert disk.drain()
        assert disk.stats()["bytes"] <= disk.limit
    st = disk.stats()
    assert st["entries"] == 2 and st["deleted"] == 2 and st["writes"] == 4
    names = files(tmp_path / "d")
    assert len(names) == len(disk.blobs) + len(disk.entries)
    assert sum(os.path.getsize(tmp_path / "d" / n) for n in names) == st["bytes"]
    kept = sorted(d.ids.tolist()[:5] for d in disk.entries.values())
    assert kept == sorted(p[:5] for p in ps[2:4])         # the two most recent evictions
    small_disk = pc.disk = tier(tmp_path / "s", limit=one // 3)
    run(sched, [request(ps[0], 2, "q")])
    assert small_disk.drain() and small_disk.stats()["writes"] == 0
    assert small_disk.stats()["skipped"] >= 1 and files(tmp_path / "s") == []


def test_write_errors_never_fail_a_request(dense, small, tmp_path, monkeypatch, caplog):
    a, b = turns(110)
    ref = {}
    sched = scheduler(dense)
    ref, _ = play(sched, a, b, monkeypatch, TINY)
    sched = scheduler(dense)
    disk = sched.eng._pcache.disk = tier(tmp_path)

    def boom(*a, **k):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(pcdisk, "_write_blob", boom)
    caplog.set_level(logging.INFO, logger="qwen.worker")
    got, x = play(sched, a, b, monkeypatch, TINY, disk)
    assert got == ref and disk.stats()["errors"] >= 1 and disk.stats()["entries"] == 0
    assert files(tmp_path) == [] and not disk.blobs and disk.pending == 0
    assert any("writing an entry" in m for m in caplog.messages)


@pytest.mark.parametrize("how", ["truncate", "read"])
def test_load_errors_are_misses(dense, small, tmp_path, monkeypatch, how):
    """A torn blob file or a failing read: the load is a miss (logged),
    the request answers as cold and stores its entry as without the tier."""
    a, b = turns(120)
    sched = scheduler(dense)
    off, ox = play(sched, a, b, monkeypatch, TINY)
    sched = scheduler(dense)
    pc = sched.eng._pcache
    disk = pc.disk = tier(tmp_path)
    monkeypatch.setattr(worker, "PROMPT_CACHE_GB", TINY)
    run(sched, [request(a[0], 4, "a0")])
    run(sched, [request(b, 4, "b")])
    assert disk.drain()
    if how == "truncate":
        for n in files(tmp_path):
            if n.startswith("seg-"):
                with open(tmp_path / n, "r+b") as f:
                    f.truncate(os.path.getsize(tmp_path / n) - 100)
    else:
        def boom(*a, **k):
            raise OSError(5, "Input/output error")
        monkeypatch.setattr(pcdisk, "_read_blob", boom)
    x = request(a[1], 6, "a1")
    assert run(sched, [x])["a1"] == off["a1"]
    assert x.reused == ox.reused
    st = disk.stats()
    assert st["loads"] == 0 and st["errors"] == 1 and not any(d.pins for d in disk.entries.values())
    assert pc.entries[-1].ids == a[1][:-1]


def test_off_changes_nothing(dense, small, tmp_path, monkeypatch):
    """Without the tier (default: QWEN_PROMPT_CACHE_DISK unset) the cache has
    none; with it but nothing worth writing (min_tokens), every answer,
    resume point and entry is the same as off."""
    assert worker._PrefixCache().disk is None
    a, b = turns(130)
    res = []
    for d in (None, "min"):
        sched = scheduler(dense)
        pc = sched.eng._pcache
        if d:
            pc.disk = tier(tmp_path, min_tokens=1 << 20)
        got, x = play(sched, a, b, monkeypatch, TINY, pc.disk)
        res.append((got, x.reused, [e.ids for e in pc.entries], pc.evictions))
    assert res[0] == res[1]
    assert files(tmp_path) == []


def test_keep_across_runs(dense, small, tmp_path, monkeypatch):
    """KEEP with the same key: the entries come back (index rebuilt from the
    files) and load the same bits; another key, or KEEP off: cleared."""
    a, b = turns(140)
    sched = scheduler(dense)
    pc = sched.eng._pcache
    disk = pc.disk = tier(tmp_path, key="K1")
    monkeypatch.setattr(worker, "PROMPT_CACHE_GB", TINY)
    run(sched, [request(a[0], 4, "a0")])
    run(sched, [request(b, 4, "b")])
    assert disk.drain()
    n, nbytes = len(disk.entries), disk.stats()["bytes"]
    assert n == 1
    names = files(tmp_path)
    with pytest.raises(RuntimeError, match="in use by another worker"):
        tier(tmp_path, key="K1", keep=True)                 # one worker per directory (flock)
    assert disk.close() and not disk.offer(pc.entries[0], 1)  # closed: nothing taken
    (tmp_path / "seg-junk.bin.tmp").write_bytes(b"x")       # a torn write of a dead process
    again = tier(tmp_path, key="K1", keep=True)
    assert len(again.entries) == n and again.stats()["bytes"] == nbytes
    assert files(tmp_path) == names
    # a fresh scheduler resumes a[1] from it, as cold
    s2 = scheduler(dense)
    s2.eng._pcache.disk = again
    x = request(a[1], 6, "a1")
    assert run(s2, [x])["a1"] == run(scheduler(dense, max_batch=1), [request(a[1], 6, "a1")])["a1"]
    assert x.reused > 0 and again.stats()["loads"] == 1
    assert check_cold(s2.eng._pcache, Cold(dense))
    assert again.close()
    assert tier(tmp_path, key="K2", keep=True).close()
    assert files(tmp_path) == []
    tier(tmp_path, key="K1", keep=True)
    assert files(tmp_path) == []


def test_blob_roundtrip(tmp_path):
    """fp16 / fp32 / uint32 / bf16 / strided arrays and None layers, bit for bit."""
    mx.random.seed(3)
    base = mx.random.normal((3, 4, 10, 8))
    arrs = [base.astype(mx.float16), base, mx.arange(40, dtype=mx.uint32).reshape(5, 8),
            base.astype(mx.bfloat16), base[:, 1:3, 2:7, :], mx.zeros((0, 4))]
    views, flags = pcdisk._host(arrs)
    pcdisk._write_blob(str(tmp_path / "x.bin"), {"kind": "st"}, views, flags)
    h, got = pcdisk._read_blob(str(tmp_path / "x.bin"))
    back = [mx.array(g) for g in got]
    back = [b.view(mx.bfloat16) if s["bf16"] else b for b, s in zip(back, h["arrays"])]
    for x, y in zip(arrs, back):
        assert x.dtype == y.dtype and x.shape == y.shape and mx.array_equal(x, y).item()
    with open(tmp_path / "x.bin", "r+b") as f:
        f.truncate(os.path.getsize(tmp_path / "x.bin") - 1)
    with pytest.raises(ValueError):
        pcdisk._read_blob(str(tmp_path / "x.bin"))


def test_insert_respects_the_headroom(dense, small, monkeypatch):
    """pc.insert: an entry alone over the headroom is not kept (the cache
    unchanged); with room, others go down to QWEN_PROMPT_CACHE_GB."""
    sched = scheduler(dense)
    pc = sched.eng._pcache
    monkeypatch.setattr(worker, "PROMPT_CACHE_GB", BIG)
    ps = [[IM_START] + p + GEN for p in prompts([40, 40], 150)]
    for i, p in enumerate(ps):
        run(sched, [request(p, 2, f"p{i}")])
    e0, e1 = pc.entries
    pc.entries = [e0]
    pc._recount()
    v = pc.version
    assert not pc.insert(e1, limit=e1.nbytes - 1)
    assert pc.entries == [e0] and pc.version == v
    monkeypatch.setattr(worker, "PROMPT_CACHE_GB", TINY)
    assert pc.insert(e1, limit=1 << 40)
    assert pc.entries == [e1] and pc.version > v and pc.evictions == 1
    assert same(pc.restore(e1, len(e1.ids)), Cold(dense)(e1.ids))


def test_disk_key(tmp_path, monkeypatch):
    """Engine._disk_key: stable; moved by the weights (a safetensors header
    of the same size too) and by numerics knobs (any QWEN_* but the sizing /
    prompt-cache ones; QWEN_PROMPT_CACHE_REPLY moves it: review #4), not by
    those."""
    def st(header):
        raw = json.dumps(header).encode()
        return len(raw).to_bytes(8, "little") + raw + b"\0" * 64
    (tmp_path / "model.safetensors").write_bytes(st({"w": {"dtype": "F16", "shape": [4, 8]}}))
    (tmp_path / "config.json").write_text("{}")
    eng = worker.Engine(str(tmp_path))
    eng.max_batch = 16
    k = eng._disk_key()
    assert k == eng._disk_key() and len(k) == 64
    monkeypatch.setenv("QWEN_PROMPT_CACHE_GB", "1")
    monkeypatch.setenv("QWEN_PROMPT_CACHE_DISK_GB", "7")
    monkeypatch.setenv("QWEN_BATCH_MEM_GB", "5")
    assert eng._disk_key() == k
    monkeypatch.setenv("QWEN_KV_BITS", "8")
    assert eng._disk_key() != k
    monkeypatch.delenv("QWEN_KV_BITS")
    monkeypatch.setattr(worker, "PROMPT_CACHE_REPLY", not worker.PROMPT_CACHE_REPLY)
    assert eng._disk_key() != k
    monkeypatch.undo()
    assert eng._disk_key() == k
    (tmp_path / "model.safetensors").write_bytes(st({"w": {"dtype": "F16", "shape": [8, 4]}}))
    assert eng._disk_key() != k                             # same size, another header
    (tmp_path / "model.safetensors").write_bytes(b"x" * 11)
    assert eng._disk_key() != k


def test_reply_entry_through_disk(dense, small, tmp_path, monkeypatch):
    """QWEN_PROMPT_CACHE_REPLY: the entry extended by a reply goes to disk
    and back with its `reply` mark; the next turn resumes after the reply,
    as from RAM (the same answer and resume point)."""
    from test_reply_kv import GEN as GEN_OFF
    monkeypatch.setattr(worker, "PROMPT_CACHE_REPLY", True)
    p0 = [IM_START] + prompts([40], 160)[0] + [IM_START] + prompts([9], 161)[0] + GEN_OFF
    b = [IM_START] + prompts([50], 163)[0] + GEN_OFF

    def go(gb, with_disk):
        sched = scheduler(dense)
        pc = sched.eng._pcache
        disk = pc.disk = tier(tmp_path / str(with_disk)) if with_disk else None
        monkeypatch.setattr(worker, "PROMPT_CACHE_GB", BIG)
        t0 = run(sched, [request(p0, 6, "t0")])["t0"]
        assert pc.entries[-1].reply == len(p0) - 1
        monkeypatch.setattr(worker, "PROMPT_CACHE_GB", gb)
        run(sched, [request(b, 4, "b")])
        if disk is not None:
            assert disk.drain()
        p1 = p0 + t0[0] + [16, 12, IM_START] + prompts([9], 162)[0] + GEN_OFF
        x = request(p1, 6, "t1")
        return t0, run(sched, [x])["t1"], x.reused, disk
    *ref, _ = go(BIG, False)
    *got, disk = go(TINY, True)
    assert got == ref
    assert got[2] >= len(p0) - 1 + len(got[0][0]) - 1          # after the reply
    assert disk.stats()["loads"] == 1


def test_load_while_rows_decode(dense, small, tmp_path, monkeypatch):
    """The head waits for its load while a row decodes (decode runs
    meanwhile), then resumes from the loaded entry: answers and resume
    point as with everything in RAM."""
    a, b = turns(170)
    long = [IM_START] + prompts([30], 175)[0] + GEN

    def go(gb, with_disk):
        sched = scheduler(dense)
        pc = sched.eng._pcache
        disk = pc.disk = tier(tmp_path / str(with_disk)) if with_disk else None
        monkeypatch.setattr(worker, "PROMPT_CACHE_GB", gb)
        for rid, ids in (("a0", a[0]), ("b", b)):
            run(sched, [request(ids, 4, rid)])
            if disk is not None:
                assert disk.drain()
        y, x = request(long, 40, "long"), request(a[1], 6, "a1")
        rows = []
        out = run(sched, [y, x], arrive={4: [x]}, on_tick=lambda t, s: rows.append(len(s.rows)))
        return out, x.reused, max(rows), disk
    *ref, _ = go(BIG, False)
    *got, disk = go(TINY, True)
    assert got == ref and got[2] == 2
    assert disk.stats()["loads"] == 1


def test_wait_bound_and_abandoned_load(dense, small, tmp_path, monkeypatch):
    """pcdisk.WAIT_S: a head whose load is slower is admitted without it
    (here at once: 0); the load that finishes for no one waiting is dropped
    by the scheduler's idle pass (unpinned, its arrays freed)."""
    import time
    a, b = turns(180)
    _, ox = play(scheduler(dense), a, b, monkeypatch, TINY)
    sched = scheduler(dense)
    disk = sched.eng._pcache.disk = tier(tmp_path)
    slow = pcdisk._read_blob

    def read(*a, **k):
        time.sleep(0.3)
        return slow(*a, **k)
    monkeypatch.setattr(pcdisk, "_read_blob", read)
    monkeypatch.setattr(pcdisk, "WAIT_S", 0.0)
    got, x = play(sched, a, b, monkeypatch, TINY, disk)
    assert x.reused == ox.reused
    assert disk.drain()
    sched._idle()
    st = disk.stats()
    assert st["discards"] == 1 and st["loads"] == 0
    assert not any(d.pins for d in disk.entries.values()) and not disk.done()


# --- review 2026-10-24 (scratchpad/review_disk/): repros of the findings ------------
def _kv(n=8, seed=0):
    mx.random.seed(seed)
    k, v = mx.random.normal((1, 2, n, 4)), mx.random.normal((1, 2, n, 4))
    mx.eval(k, v)
    return k, v


def _entry(ids, chain):
    from mlx_vlm.models.cache import KVCache
    return worker._Entry(list(ids), chain, {}, None, [("kv", KVCache)], False)


def _consistent(t):
    """Index == files: every blob / entry file there, nothing else, `used` ==
    their sizes; no pending blob or entry left behind."""
    names = set(files(t.root))
    want = {os.path.basename(b.path) for b in t.blobs.values()} | \
        {os.path.basename(d.path) for d in t.entries.values()}
    assert names == want, (names, want)
    assert t.used == sum(os.path.getsize(os.path.join(t.root, n)) for n in names)
    assert all(b.state == "ok" for b in t.blobs.values())
    assert all(d.state == "ok" for d in t.entries.values()) and t.pending == 0


def test_lru_never_drops_a_blob_the_new_entry_shares(tmp_path):
    """Review #1: the disk is full and the only LRU entry is the last holder
    of the segment the new entry shares (a system block): nothing breaks.
    That entry is not dropped for nothing (its deletion would free only its
    own small file: no room either way), the new entry is skipped, the index
    stays consistent, and a later entry sharing both segments is written and
    loads them bit for bit (the old code: KeyError, the shared segment's file
    deleted, its blob left pending, every later entry with it lost)."""
    t = tier(tmp_path)
    S = worker._Seg(0, 8, [_kv(8, 1)])
    t.offer(_entry(range(8), [(S, 0, 8)]))
    t.flush()
    assert t.drain()
    t.limit = t.used + 4096
    T = worker._Seg(8, 16, [_kv(8, 2)])
    t.offer(_entry(range(100, 116), [(S, 0, 8), (T, 8, 16)]))
    t.flush()
    assert t.drain()
    st = t.stats()
    assert st["errors"] == 0 and st["deleted"] == 0 and st["entries"] == 1 and st["skipped"] == 1
    _consistent(t)
    t.limit = 1 << 30
    C = _entry(range(200, 216), [(S, 0, 8), (T, 8, 16)])
    t.offer(C)
    t.flush()
    assert t.drain() and t.stats()["errors"] == 0 and len(t.entries) == 2
    _consistent(t)
    d = t.entries[C.disk_uid]
    del C
    job = t.load(d, [])
    assert job.done.wait(10) and t.done() == [job]
    t._segs.clear()                                     # read them: S, T are alive here
    e = t.finish(job)
    for (s, _, _), want in zip(e.chain, (S, T)):
        assert all(mx.array_equal(x, y).item() for x, y in zip(s.arrays, want.arrays))
    # and when dropping the LRU entry does make room, it goes, the shared blob stays
    t.limit = t.used + 4096
    U = worker._Seg(16, 24, [_kv(8, 3)])
    t.offer(_entry(range(300, 324), [(S, 0, 8), (U, 16, 24)]))
    t.flush()
    assert t.drain() and t.stats()["errors"] == 0
    _consistent(t)


def test_late_unlink_never_deletes_a_rewritten_blob(tmp_path):
    """Review #5: the writer drops an entry (LRU / covered) whose segment is
    still alive in RAM; the unlink is queued under the lock, so a re-offer of
    that segment registered right after is written after it: its file stays,
    and it loads (was: the late unlink deleted the fresh file, the index said
    "ok", loads failed with FileNotFoundError)."""
    t = tier(tmp_path)
    S = worker._Seg(0, 8, [_kv(8, 4)])
    A = _entry(range(8), [(S, 0, 8)])
    t.offer(A)
    t.flush()
    assert t.drain()
    dA = t.entries[A.disk_uid]
    with t.lock:                                        # as _run_write does it
        t._unlink_locked(t._drop(dA))
    t.offer(_entry(range(100, 108), [(S, 0, 8)]))
    t.flush()
    assert t.drain()
    _consistent(t)
    (dB,) = t.entries.values()
    job = t.load(dB, [])
    assert job.done.wait(10) and job.error is None


def test_eviction_ram_is_charged_until_written(dense, small, tmp_path, monkeypatch):
    """Review #2: an entry evicted to disk keeps its RAM until its file is
    written; pc.charged() (what every budget check takes) counts it until
    then; past QWEN_PROMPT_CACHE_DISK_QUEUE_GB an eviction frees at once
    (not written), as without the tier."""
    import threading
    sched = scheduler(dense)
    pc = sched.eng._pcache
    monkeypatch.setattr(worker, "PROMPT_CACHE_GB", BIG)
    run(sched, [request([IM_START] + prompts([60], 210)[0] + GEN, 2, "x")])
    (e,) = pc.entries
    n = pc.bytes
    gate = threading.Event()
    slow = pcdisk._write_blob

    def held_write(*a, **k):
        gate.wait(10)
        return slow(*a, **k)
    monkeypatch.setattr(pcdisk, "_write_blob", held_write)
    disk = pc.disk = tier(tmp_path)
    pc.evict()
    assert pc.bytes == 0 and pc.charged() == n and disk.held_bytes() == n
    disk.flush()
    assert pc.charged() == n                            # the writer holds the views
    gate.set()
    assert disk.drain() and pc.charged() == 0 and disk.stats()["writes"] == 1
    # the queue cap: nothing taken past it
    pc.entries, e.disk_uid = [e], None
    pc._recount()
    small_q = pc.disk = tier(tmp_path / "q", queue_bytes=n - 1)
    pc.evict()
    assert pc.charged() == 0 and small_q.offers == [] and small_q.stats()["skipped"] == 1


def test_headroom_evictions_free_at_once(dense, small, tmp_path, monkeypatch):
    """Review #2: store / insert evict for the batch's headroom with spill
    False (freed now: the new segment is copied right after), and spill for
    the soft cap only while the held RAM fits that headroom."""
    sched = scheduler(dense)
    pc = sched.eng._pcache
    monkeypatch.setattr(worker, "PROMPT_CACHE_GB", BIG)
    ps = [[IM_START] + p + GEN for p in prompts([40, 40, 40], 220)]
    for i, p in enumerate(ps):
        run(sched, [request(p, 2, f"p{i}")])
    e0, e1, e2 = pc.entries
    disk = pc.disk = tier(tmp_path)
    # insert e2 beside e0, e1 with a headroom for e2 alone: both freed at once
    pc.entries = [e0, e1]
    pc._recount()
    assert pc.insert(e2, limit=e2.nbytes + 1)
    assert pc.entries == [e2] and disk.offers == [] and pc.charged() == pc.bytes
    # the soft cap with room: written (held), not freed
    pc.entries = [e0, e2]
    pc._recount()
    monkeypatch.setattr(worker, "PROMPT_CACHE_GB", TINY)
    assert pc.insert(e1, limit=1 << 40)
    assert pc.entries == [e1] and len(disk.offers) == 2 and pc.charged() > pc.bytes
    disk.flush()
    assert disk.drain() and pc.charged() == pc.bytes
    # store: the copy of the new segment comes right after the evictions
    cache = pc.restore(e1, len(e1.ids))
    pc.entries = [e0, e2]
    pc._recount()
    monkeypatch.setattr(worker, "PROMPT_CACHE_GB", BIG)
    extra = worker._Seg.estimate(cache, 0, len(e1.ids))
    new, _ = pc.store(e1.ids, cache, {}, limit=extra + pc.bytes_of([e2]) + 1)
    assert new is not None and disk.offers == []        # e0 / e2: whatever went, freed now
    assert not (any(x is e0 for x in pc.entries) and any(x is e2 for x in pc.entries))
    assert pc.charged() == pc.bytes


def test_admission_waits_for_the_writer(dense, small, tmp_path, monkeypatch):
    """Review #2: _fits evicts to disk what admission needs, then admits
    only once the writer released their RAM (reserved + need + charged within
    the budget), never on the count alone; the answers are as without it."""
    import time
    a_ids, b_ids = prompts([10, 9], 105)
    from test_prompt_cache import cold, fill
    ref = {"a": cold(dense, a_ids, 30, "a"), "b": cold(dense, b_ids, 6, "b")}
    sched = scheduler(dense)
    pc = sched.eng._pcache
    fill(sched)
    disk = pc.disk = tier(tmp_path)
    slow = pcdisk._write_blob

    def write(*a, **k):
        time.sleep(0.05)
        return slow(*a, **k)
    monkeypatch.setattr(pcdisk, "_write_blob", write)
    a, b = request(a_ids, 30, "a"), request(b_ids, 6, "b")
    seen = []
    fits = sched._fits

    def spy(req, rows):
        ok = fits(req, rows)
        if req is b and rows:
            seen.append((ok, sched.reserved + sched._reserve(b) + pc.charged() <= sched.budget,
                         disk.held_bytes()))
        return ok

    def squeeze(t, s):
        if t == 3:
            s.budget = s.reserved + s._reserve(b) + pc.bytes - 1
    sched._fits = spy
    got = run(sched, [a, b], arrive={3: [b]}, on_tick=squeeze)
    assert got == ref
    assert any(not ok and held for ok, _, held in seen)    # it waited for the writer
    assert all(fit for ok, fit, _ in seen if ok)            # admitted only within the budget
    assert disk.drain() and disk.stats()["writes"] >= 1 and pc.charged() == pc.bytes


def test_load_is_charged_until_inserted(dense, small, tmp_path, monkeypatch):
    """Review #2: a load's bytes are held from its start (before its arrays
    exist) until it is inserted (then they are pc.bytes) or dropped."""
    import threading
    a, b = turns(230)
    sched = scheduler(dense)
    pc = sched.eng._pcache
    disk = pc.disk = tier(tmp_path)
    monkeypatch.setattr(worker, "PROMPT_CACHE_GB", TINY)
    for rid, ids in (("a0", a[0]), ("b", b)):
        run(sched, [request(ids, 4, rid)])
        assert disk.drain()
    gate = threading.Event()
    slow = pcdisk._read_blob

    def read(*a, **k):
        gate.wait(10)
        return slow(*a, **k)
    monkeypatch.setattr(pcdisk, "_read_blob", read)
    x = request(a[1], 6, "a1")
    held = []

    def watch(t, s):
        held.append(disk.held_bytes())
        if t == 2:
            gate.set()
    out = run(sched, [x], on_tick=watch)["a1"]
    assert out == run(scheduler(dense, max_batch=1), [request(a[1], 6, "a1")])["a1"]
    assert x.reused > 0 and disk.stats()["loads"] == 1
    assert held[1] > 0 and held[-1] == disk.held_bytes() == 0


def test_load_reads_one_array_at_a_time(tmp_path):
    """Review #3: the loader turns each array into an MLX array as it is
    read: the numpy transient is one array, not the blob."""
    import gc
    import tracemalloc
    t = tier(tmp_path)
    kv = [(mx.random.normal((1, 2, 4096, 64)), mx.random.normal((1, 2, 4096, 64)))
          for _ in range(8)]
    mx.eval(kv)
    S = worker._Seg(0, 4096, kv)
    t.offer(_entry(range(4096), [(S, 0, 4096)]))
    t.flush()
    assert t.drain()
    one, blob = kv[0][0].nbytes, S.nbytes
    del kv, S
    gc.collect()
    (d,) = t.entries.values()
    tracemalloc.start()
    job = pcdisk._Load(d, None, {})
    t._run_load(job)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert job.error is None and job.nbytes == blob
    assert peak < 2 * one < blob / 4


def test_insert_failure_drops_the_load(dense, small, tmp_path, monkeypatch):
    """Review #6: an exception while a loaded entry goes in drops that load
    (unpinned, hold released: busy() false again); the request goes on as
    without the tier."""
    a, b = turns(240)
    _, ox = play(scheduler(dense), a, b, monkeypatch, TINY)
    sched = scheduler(dense)
    pc = sched.eng._pcache
    disk = pc.disk = tier(tmp_path)

    def boom(*a, **k):
        raise MemoryError("no room")
    monkeypatch.setattr(pc, "insert", boom)
    got, x = play(sched, a, b, monkeypatch, TINY, disk)
    assert x.reused == ox.reused
    assert not disk.busy() and disk.held_bytes() == 0 and disk.stats()["discards"] == 1


# --- re-review of ed3b242 ----------------------------------------------------------
def _release_later(disk, n, after=0.3):
    import threading
    t = threading.Timer(after, disk._release, args=(n,))
    t.start()
    return t


def test_growth_waits_for_the_tier_instead_of_cutting(dense, small, tmp_path, monkeypatch):
    """Re-review #1: rows growing past the headroom only because of the
    tier's held RAM (another eviction's write, a load) are not cut: _regrow
    waits for it (bounded); with the wait bound passed it decides as before
    (the row is cut)."""
    from test_prompt_cache import fill
    monkeypatch.setattr(worker, "RESERVE_TOKENS", 2)
    ps = prompts([8, 8], 54)
    probe = scheduler(dense)
    short, grown = probe._bytes(8 + 2), probe._bytes(8 + 40 + 1)
    for released in (True, False):
        sched = scheduler(dense, budget=1 << 40)
        pc = sched.eng._pcache
        fill(sched)
        pc.entries = []
        pc._recount()
        disk = pc.disk = tier(tmp_path / str(released))
        if not released:
            monkeypatch.setattr(pcdisk, "HELD_WAIT_S", 0.2)
        a, b = request(ps[0], 40, "a"), request(ps[1], 40, "b")
        H = 4 * (grown - short)

        def squeeze(t, s):
            if len(s.rows) == 2 and s.budget > 2 * short:    # both in: headroom for the rows
                s.budget, s.prefill_reserve = 2 * short, 2 * (grown - short) + 4096
                with disk.lock:
                    disk.held += H                          # a write in flight elsewhere
                if released:
                    _release_later(disk, H)
        got = run(sched, [a, b], on_tick=squeeze)
        cut = [k for k in "ab" if len(got[k][0]) < 40]
        assert (cut == []) == released, cut
        assert disk.held_bytes() in (0, H)


def test_cancel_loads_frees_their_hold(dense, small, tmp_path, monkeypatch):
    """Re-review #1: cancel_loads stops a running load at its next array;
    finish gives nothing (dropped), the hold is released."""
    import threading
    a, b = turns(250)
    sched = scheduler(dense)
    pc = sched.eng._pcache
    disk = pc.disk = tier(tmp_path)
    monkeypatch.setattr(worker, "PROMPT_CACHE_GB", TINY)
    for rid, ids in (("a0", a[0]), ("b", b)):
        run(sched, [request(ids, 4, rid)])
        assert disk.drain()
    gate = threading.Event()
    slow = pcdisk._read_blob

    def read(*x, **k):
        gate.wait(10)
        return slow(*x, **k)
    monkeypatch.setattr(pcdisk, "_read_blob", read)
    d, _ = disk.peek(np.asarray(a[1], np.int32), set(worker._bounds(a[1], IM_START)))
    job = disk.load(d, pc.entries)
    assert disk.held_bytes() > 0 and disk.cancel_loads() == 1
    gate.set()
    assert job.done.wait(10) and disk.done() == [job]
    assert disk.finish(job) is None and disk.held_bytes() == 0 and not disk.busy()


def test_admission_keeps_its_hit_while_the_writer_holds_ram(dense, small, tmp_path, monkeypatch,
                                                            caplog):
    """Re-review #2: the head fits with its hit once the tier's held RAM is
    released: it waits for that (bounded), its KEEP_WAIT clock untouched,
    and is admitted with the hit (was: it gave the hit up at once)."""
    from test_prompt_cache import cold, fill
    monkeypatch.setattr(worker, "PROMPT_CACHE_KEEP_TOKENS", 1 << 20)   # no keep-wait: gives up
    sched = scheduler(dense)
    pc = sched.eng._pcache
    ws = fill(sched)
    disk = pc.disk = tier(tmp_path)
    a_ids = prompts([10], 106)[0]
    b_ids = ws[0][:30] + prompts([8], 107)[0]
    target = pc.entries[0]
    a, b = request(a_ids, 30, "a"), request(b_ids, 6, "b")
    caplog.set_level(logging.INFO, logger="qwen.worker")

    H = target.nbytes // 2
    rows = {}

    def squeeze(t, s):
        if t == 3:
            # (+ 64 KiB: a's own growth meanwhile; the hold is ~10x that)
            s.budget = s.reserved + s._reserve(b) + target.nbytes + (64 << 10)
            with disk.lock:
                disk.held += H                          # a write in flight
        if t == 10:
            disk._release(H)                            # ... done
        rows[t] = len(s.rows)
    got = run(sched, [a, b], arrive={3: [b]}, on_tick=squeeze)
    assert got == {"a": cold(dense, a_ids, 30, "a"), "b": cold(dense, b_ids, 6, "b")}
    assert b.reused > 0 and b.keep_since is None
    assert not any("gives up" in m for m in caplog.messages)
    assert max(rows[t] for t in range(4, 11)) == 1 and max(rows.values()) == 2   # joined a after
    assert disk.drain() and disk.held_bytes() == 0


def test_stuck_write_drops_the_queue(tmp_path, monkeypatch):
    """Re-review #3: a write running past TIMEOUT_S: the writes queued behind
    it are dropped (their RAM released, never written), nothing is taken
    meanwhile; once it returns the tier writes again and the index matches
    the files."""
    import threading
    import time
    gate = threading.Event()
    slow = pcdisk._write_blob
    first = []

    def write(*a, **k):
        if not first:
            first.append(1)
            gate.wait(10)
        return slow(*a, **k)
    monkeypatch.setattr(pcdisk, "_write_blob", write)
    t = tier(tmp_path)
    t.timeout_s = 0.2
    e1 = _entry(range(8), [(worker._Seg(0, 8, [_kv(8, 11)]), 0, 8)])
    e2 = _entry(range(100, 108), [(worker._Seg(0, 8, [_kv(8, 12)]), 0, 8)])
    for e in (e1, e2):
        assert t.offer(e, 1000)
        t.flush()
    time.sleep(0.4)
    t.flush()                                           # the stall check
    assert t.stalled and t.held_bytes() <= 1000 and len(t.entries) == 1
    e3 = _entry(range(200, 208), [(worker._Seg(0, 8, [_kv(8, 13)]), 0, 8)])
    assert not t.offer(e3, 1000)
    gate.set()
    assert t.drain() and not t.stalled and t.held_bytes() == 0
    assert t.stats()["writes"] == 1 and e1.disk_uid in t.entries
    _consistent(t)
    assert t.offer(e3, 1000)
    t.flush()
    assert t.drain() and t.stats()["writes"] == 2
    _consistent(t)


def test_eviction_holds_only_what_the_writer_keeps(tmp_path):
    """Re-review #4: an evicted entry whose segment is already on disk (a
    loaded entry's, shared) holds only its new arrays once flushed."""
    t = tier(tmp_path)
    S = worker._Seg(0, 8, [_kv(8, 21)])
    t.offer(_entry(range(8), [(S, 0, 8)]))
    t.flush()
    assert t.drain()
    T = worker._Seg(8, 16, [_kv(8, 22)])
    e = _entry(range(16), [(S, 0, 8), (T, 8, 16)])
    own = {id(a): a.nbytes for a in e.arrays}
    import threading
    gate = threading.Event()
    slow = pcdisk._write_blob
    pcdisk_write = pcdisk._write_blob

    def write(*a, **k):
        gate.wait(10)
        return slow(*a, **k)
    pcdisk._write_blob = write
    try:
        assert t.offer(e, sum(own.values()), own)
        t.flush()
        assert t.held_bytes() == T.nbytes                   # not S's: on disk already
        gate.set()
        assert t.drain() and t.held_bytes() == 0
    finally:
        pcdisk._write_blob = pcdisk_write
