"""The prompt cache's disk tier (QWEN_PROMPT_CACHE_DISK, pcdisk.py) on the
tiny random model (CPU), docs/perf/2026-10-24/design/pcdisk.md: an entry
evicted from RAM is written, a waiting request whose deepest hit is on disk
gets it loaded and resumes from it exactly as from RAM (== cold), for fp32,
fp16 and quantized KV; shared segments / states are blobs written once and
reused live; the disk limit is LRU; I/O errors never fail a request; off
changes nothing."""
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
    return sorted(n for n in os.listdir(path) if n != "pcdisk.json")


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
    tier(tmp_path, key="K2", keep=True)
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
    """Engine._disk_key: stable; moved by the weights and by numerics knobs
    (any QWEN_* but the sizing / prompt-cache ones), not by those."""
    (tmp_path / "model.safetensors").write_bytes(b"x" * 10)
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
