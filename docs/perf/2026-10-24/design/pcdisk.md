# Prompt cache on disk (`QWEN_PROMPT_CACHE_DISK`), 2026-10-24

Item 2 of `../deepinfer_ideas.md`. Motivation: `../../2026-10-17/REPORT-appworld-run-1017.md`.

- **8 processes.** The 6 GB cache holds 7 histories. The 8th is evicted every round, and each of
  its MISSes re-prefills 15–36K tokens (52–82 s).
- **The tail.** Histories of 27–58K tokens, 3–4 entries in RAM. MISSes on 54.6K / 58.4K tokens
  took 181 / 203 s.
- **Swap.** It stays at 5–6.7 GB with the cache in RAM, so a bigger RAM cache is not the answer.

An eviction used to be a deletion. Now an evicted entry is written to the SSD. A waiting request
whose deepest hit is on disk gets that entry loaded back while it waits. Its lookup then resumes
from it exactly as from a RAM entry.

GPU gate done on 2026-10-24 (see "GPU results" at the end): AppWorld replay at
`QWEN_PROMPT_CACHE_GB=3`, 8 episodes, req/min ×2.3 (10.3 / 10.4 → 23.4 / 24.5 in two interleaved
pairs), after one fix found by the gate (a loaded entry was evicted again before its request's
turn).

## What was done

- **`src/worker/pcdisk.py`** (new): `DiskTier`, the index, the blob files, and the writer and
  loader threads.
- **`src/worker/worker.py`**, small hooks:
  - `_PrefixCache.disk`, and `evict()` → `disk.offer(e)`;
  - `_PrefixCache.insert()`, which puts a loaded entry back through the store's budget rules;
  - `_Req.disk`, the per-request memo;
  - `_Scheduler.tick`: `_disk_tick` before admission, and `disk.flush()` at the end of the tick;
  - `_Scheduler._disk_tick` / `_disk_want` / `_disk_insert`, a new section next to the reply KV;
  - `_Scheduler._idle`, which drops finished loads that nobody waits for;
  - `_Scheduler.wake`;
  - `Engine.load`, which creates the tier, and `Engine._disk_key`.
- **`src/worker/tests/test_pcdisk.py`**: 19 CPU tests.
- **`scripts/perf/pcdisk/`**: `io_bench.py`, `exact_check.py`, `run_exact.sh`,
  `run_replay_disk.sh`, `disk_log.py`, `run_all.sh`.

## Knobs

| var | default | |
|---|---|---|
| `QWEN_PROMPT_CACHE_DISK` | unset = off | Directory for the cache files. Only our file names in it are ever touched. Needs `QWEN_PROMPT_CACHE=1`. |
| `QWEN_PROMPT_CACHE_DISK_GB` | `32` | Disk limit. The least recently used entries go first (used = written or loaded). |
| `QWEN_PROMPT_CACHE_DISK_KEEP` | `0` | `0`: our files are deleted at start. `1`: a previous run's files are kept when the validity key matches, else cleared. |
| `QWEN_PROMPT_CACHE_DISK_QUEUE_GB` | `4` | RAM of evicted entries waiting for the writer. Their MLX buffers stay alive until written and are charged in every budget check (`held_bytes`). An eviction past this cap is a plain deletion, as without the tier. There is no "one entry always" exception. `4` fits one 50K-token entry (~2.5 GB) or two 1 GB AppWorld entries. |
| `QWEN_PROMPT_CACHE_DISK_MIN_TOKENS` | `4096` | Shorter entries are not written. Their hit saves under ~2 slabs. |
| `QWEN_PROMPT_CACHE_DISK_WAIT_S` | `10` | How long the head of the queue waits for its own load. After that it is admitted without it. |
| `QWEN_PROMPT_CACHE_DISK_HELD_WAIT_S` | `5` | A growing row (`_regrow`) or the head of the queue (`_fits`) that only the tier's held RAM keeps out waits this long for it, then decides as without the wait. |
| `QWEN_PROMPT_CACHE_DISK_TIMEOUT_S` | `30` | A write running this long is a stuck disk. The writes queued behind it are dropped (RAM released, never written) and nothing is taken until it returns. Its own buffers stay held, since the writer is reading them, and no budget check waits for that hold (`stuck_bytes`). A load running this long is abandoned: its hold and pin go, its request goes without it, and no new load starts until the loader returns. |
| `QWEN_PROMPT_CACHE_DISK_NOCACHE` | `1` | `F_NOCACHE` on every file, so the cache's GBs never sit in the macOS unified buffer cache next to the model. `0`: the OS caches as usual. |

The numeric knobs are parsed only when `QWEN_PROMPT_CACHE_DISK` is set: a bad value never stops a
worker that does not use the tier. With the tier on, an unparsable value turns it off, and
`Engine.load` logs which one (`pcdisk.ENV_ERRORS`; logging is not set up at import).

## Design

### When an entry is written

- **On eviction, not on store.** Every eviction goes through `_PrefixCache.evict`: the store's
  soft cap and headroom, `_fits` (including "gives up its hit"), `_regrow`, `pc.insert`, and the
  reply store. Each of these calls `disk.offer(e)`.
- **Absorbed entries are not written.** An entry absorbed by a newer turn is removed from
  `entries`, not evicted, so it is not offered. The same holds for checkpoints thinned away
  before an eviction. A file therefore holds only what the RAM cache still had: one chain per
  conversation.
- **"Not kept" entries are not written.** An entry that does not fit the batch's headroom even
  alone would need its `[at, len)` copied out of the row. The batch has no room for that copy,
  and the row's own KV buffers are in use (decode writes into them).
- **Partial (cancelled-prefill) and reply entries are written like any other.**
- **Shutdown does not flush the RAM cache.** It is optional, and the `KEEP` use case only.
  `_Scheduler.stop()` returns as soon as `_close` sets `closed`, so a flush there would race the
  exit. This needs `stop()` to wait for a drain. Not done.
- **Shutdown drains the tier.** `main()` calls `disk.close(10)` after `sched.stop()`:
  1. nothing new is taken, and the offers not flushed are dropped;
  2. it waits until the writer and the loader are idle, so neither is inside a file write over
     an MLX buffer or an `mx.array` when the interpreter tears MLX down;
  3. if they are still busy after 10 s, `os._exit(1)`, as for a stuck scheduler tick.
- **One worker per directory.** `flock` on `pcdisk.lock`. A second worker on the same directory
  gets a `RuntimeError` at start, and its tier is off (logged).

### How an entry is written without blocking the scheduler

- **`offer`** appends the entry to a list. It runs inside `evict`, without MLX and without I/O.
- **`flush`**, at the end of the tick, on the scheduler thread:
  1. It takes the entry's segments (`_Seg`) and recurrent states, and finds the ones not already
     on disk (dedup, below).
  2. It sizes the files, then plans the disk LRU (`_plan`) before anything changes. The plan
     simulates the refcounts and never counts the blobs this entry shares as freed. When
     deleting every LRU entry would still leave no room, nothing is deleted and the entry is
     skipped.
  3. It registers the entry as `pending` and takes its blob references first. Only then are the
     planned entries dropped, so a blob it shares (a system segment whose last other holder was
     the LRU entry) can never go. The old order raised a KeyError there and left `pending`
     blobs behind (review #1).
  4. It takes **zero-copy numpy views** of the arrays: `mx.eval` (a no-op wait; they are
     evaluated), then `np.asarray(a)`. Checked on the GPU device: the numpy array shares the MLX
     buffer, with no memcpy and no stall. bfloat16 has no buffer format, so it goes as an MLX
     `uint16` view.
- **The writer thread** does plain file I/O over the views:
  - each file is written to `*.tmp` and then renamed;
  - then it drops the views. The MLX buffers are freed there: constant, evaluated arrays,
    through the allocator's lock. Handler threads already free the image processor's `to_mlx`
    arrays the same way.
- **Unlinks are queued under the index lock.** The writer's own drops (covered entries, LRU)
  queue their unlink while the index already says the files are gone. A re-offer of a live
  segment registered after that is therefore written after the unlink (FIFO), and a late unlink
  can never delete the fresh file (review #5).
- **Memory: an eviction to disk does not free RAM until the write ends** (~0.3–0.8 s/GB, not
  measured). This was review #2: the class of OOM on the 27B / 35B.
  - **What is held.** `DiskTier.held_bytes()` is the RAM the tier holds outside `pc.bytes`:
    - an evicted entry's freed bytes (`evict` passes `bytes before − after`), from the offer
      until its views are dropped after the write;
    - a load's `need`, from its start (before its arrays exist) until it is inserted (then it
      is `pc.bytes`) or dropped.
  - **What is charged.** `pc.charged() = pc.bytes + held` in every budget check: `_fits`,
    `_regrow`, the `limit` checks of `store` / `insert`, `_charge_fits`, the cascade tidy and
    `_disk_want`. The soft cap `QWEN_PROMPT_CACHE_GB` stays on `pc.bytes` alone.
  - **`evict(spill=…)`.** A spill (a write to disk) does not change `charged`, so a caller that
    needs the memory now passes `spill=False` and the entry is freed at once, as without the
    tier:
    - the `limit` loops of `store` / `insert` (the new segment is copied right after);
    - `_regrow`.
  - **The soft-cap loops spill only while it fits.** They spill while `charged + extra ≤ limit`
    (the held RAM still fits the headroom); else the entry is freed.
  - **`_fits` waits.** It evicts (with spill) until `reserved + need + pc.bytes ≤ budget`, as
    before. It then admits only when `reserved + need + charged ≤ budget`: a request whose
    room went to disk waits for the writer (≤ ~1–2 s), and nothing more is evicted meanwhile.
    Its live check sees the held buffers in `mx.get_active_memory()`.
  - **The queue cap is honest.** `held + freed ≤ QUEUE_GB`, else the eviction is a plain
    deletion.
  - **Precise hold.** At the offer, held = the bytes of the entry's arrays no other RAM entry
    holds. At the flush it becomes what the write job keeps alive, i.e. **every** array of its
    fresh blobs: a blob already on disk is freed with the entry, so it is not held. A fresh
    segment it shares with an entry still in RAM is held too (final review #2). Before, it was
    left out of the hold because the other entry held it. Once that entry was evicted (freed at
    once, or its own write borrowing the pending blob), the segment stayed alive in this job's
    views but was charged nowhere. It is now counted twice while the other entry is in RAM
    (`pc.bytes` + held), which errs on the safe side. The `QUEUE_GB` check at the offer sees
    only the entry's own bytes, so the hold can pass the cap by such shared arrays.
  - **Unflushed offers are taken back when the RAM is needed within the tick** (final review
    #1). An offer reaches the writer only at the tick's flush, so waiting for its hold inside
    the same tick waits for nothing. Such an offer comes from `_drop_done` → reply store,
    `_fits`, or `pc.insert` in `_disk_tick`, earlier in the tick. `_disk_room` therefore first
    calls `revoke(over)`: the latest offers go until the shortfall is covered, their hold is
    released and their entries are freed at once, as `spill=False` (logged "not written: its
    RAM is needed now"). Before, the row waited `HELD_WAIT_S` and was then cut anyway, where
    without the tier the entries would have been freed at once.
  - **A stuck write's hold is never waited for** (final review #4). `stuck_bytes()` is the hold
    of a write running past `TIMEOUT_S`. `_disk_room` waits only while `held − stuck` covers
    the shortfall. `_fits` waits only when the hold beyond `stuck` would admit the head. Before,
    each growing row and each head waited `HELD_WAIT_S` for a hold that might never go.
  - **A transient hold never cuts an answer** (re-review #1). When only the hold keeps a growing
    row out of the headroom, `_regrow` → `_disk_room`:
    1. it takes back this tick's unflushed offers (`revoke`, above);
    2. it cancels the loads not inserted yet (`cancel_loads`: the loader stops at its next
       array; the scheduler drops them; their memory goes first);
    3. it then waits for the writer for at most `HELD_WAIT_S`, blocking the scheduler (decode
       stalls, rarely), and only while what the tier can still release (not a stuck write's
       hold) covers the shortfall;
    4. only after that does it decide as before.

    A load also starts only when `reserved + charged + need ≤ budget`. That leaves the prefill
    reserve free for growth.
  - **A transient hold never takes a hit** (re-review #2). When the head fits with its hit once
    the hold is released, `_fits` waits for at most `HELD_WAIT_S` (`Want.held_since`). The
    `KEEP_WAIT` clock stays untouched and the hit is not given up. Checked in the final review
    (#5): it waits only when the hold is what keeps it out. That means two conditions. First,
    with every other entry evicted it would still not fit beside the current hold
    (`blocked(floor)`). Second, it would fit once the hold beyond a stuck write's is gone
    (`not blocked(floor, stuck)`). When evicting the others admits it beside the hold, they go
    at once (spilled), and the head waits only for their own write, not for another eviction's.
  - **A stuck disk** (re-review #3): see `TIMEOUT_S`. `check_stall` runs at every flush.
  - **With the tier off**, `charged == bytes` and `spill` is unused, so every check is the old
    one bit for bit. The new `_fits` return can never trigger then: after its loop the bytes are
    within the budget, or only `keep` is left, which `blocked()` already ruled out.
- **Wake-up.** When the writer's and loader's queues drain, `on_drained` → `_Scheduler.wake()`
  wakes an idle loop. Its `_idle()` then clears the buffer pool and frees any load nobody waits
  for.

### Format and dedup

```
seg-<uid>.bin   one _Seg: KV of [a, b) of every full-attention layer
st-<uid>.bin    one recurrent state (a checkpoint or an end state)
ent-<uid>.bin   one entry: metadata header + its ids (int32)
pcdisk.json     {format, key}
```

- **Blob layout.** Each blob file is `MAGIC`, a u64 header length, then a JSON header (dtype,
  shape, offset and bf16 flag per array, plus the structure). The arrays follow in C order, at
  4096-aligned offsets. Its size is checked on every read, which catches torn files.
- **Entry metadata.** The chain as `(seg uid, lo, hi)`, the segments' `(a, b)`, `ckpts`
  (pos → state uid), `end`, the layout (class module and qualname, group size, bits, container),
  `quantized`, `low`, `reply`, `partial`, `chained`, `diverge`, `used`.
- **Why per-blob files and not one file per entry.** Segments and states are shared objects
  between RAM entries:
  - a system block's segment, shared by all episodes;
  - inherited checkpoints;
  - the pieces of a loaded entry, which the next turn absorbs.

  Blobs are therefore deduplicated by object identity:
  - `_Seg` → uid in a `WeakKeyDictionary`;
  - a state → uid keyed by the ids of all its arrays, every one held weakly. A record matches
    only while all of them are alive and identical, so a reused `id()` can never alias another
    state (final review #6).

  Blobs are reference-counted by the disk entries. A blob already on disk is not written again,
  and a blob whose object is still alive in RAM is reused at load, neither read nor duplicated.
  CPU tests show it:
  - two conversations over one system block share the segment file;
  - a turn built on a loaded entry writes only its new segment and states ("5 blobs shared");
  - a loaded entry evicted again unchanged is only touched, not rewritten;
  - an exact repeat writes 0.0 MB new.
- **Covered disk entries are dropped.** Once a written entry is complete, disk entries whose ids
  are a prefix of its ids are dropped, as the RAM store absorbs them. The same quantization
  exception applies: an unquantized entry stays under a quantized one.

### Index and lookup

- **The index lives in RAM.** `DiskEntry` keeps the ids as int32 numpy (4 B/token), the
  checkpoint positions, the flags and the blob uids.
- **`DiskEntry.best` is `_Entry.best` over numpy.** The LCP is one vector compare
  (`flatnonzero(a != b)`), so a scan is ~µs per entry. A test checks that the two agree on every
  entry and probe.
- **No prefix hashes.** At 32 GB and ~1 GB per entry there are about 30–60 entries, and a linear
  numpy scan is cheap. Hash chains at `<|im_start|>` / cut boundaries would only pay at thousands
  of entries.

### Load

- **The decision.** It is taken on the scheduler thread, in `_disk_tick`, every tick, for the
  first `max_batch` waiting requests:
  - the disk peek is memoized on `req.disk` by `(disk.version, pc.version)`;
  - the RAM depth comes from the existing `_peek` memo.
  - A load starts when the disk hit is ≥ RAM hit + `PREFILL_STEP`.
  - It also has to fit: the bytes not alive in RAM must be ≤ budget + prefill reserve −
    reserved (what a store may take), and live + that + prefill reserve ≤ cap (`_fits`'s live
    check).
  - One load at a time (`busy()`): the SSD is the bottleneck.
  - **Room beside the requests ahead** (GPU gate fix): a request that is not first in line
    (the admitted request whose prefill has not started counts as ahead) loads only when its
    `need` fits under `QWEN_PROMPT_CACHE_GB` beside the RAM entries the requests ahead of it
    resume from. Otherwise their stores before its turn evict it again.
  - **Reload**: a request whose loaded entry was kept and then left the RAM cache before its
    turn loads it again when the peek finds it on disk (written back: its blobs are shared, so
    the write costs ~0.1 MB). At most `pcdisk.MAX_LOADS` (3) loads per request, and never after
    a failed load or one that was not kept. The log says `…, load N` from the 2nd on.
- **The loader thread** reads each array of a blob into numpy (`readinto`, GIL released) and
  turns it into `mx.array(numpy)` at once, before reading the next one (`_read_blob(each=…)`).
  The result is constant arrays: allocation + memcpy, no graph, no stream, the same as the
  processor's `to_mlx` on a handler thread. The numpy transient is one array, ≤ a layer's K or
  V (~100 MB at 50K tokens). Before review #3 it was the whole blob, i.e. 2× a 1.6 GB first
  segment; a test bounds it under 2 arrays.
- **Insert**, on the scheduler thread, at the next tick (`_disk_insert`). If the request still
  waits:
  1. `finish()` builds the `_Seg` / `_Entry` objects, reusing the live ones, and applies the
     bf16 views and `mx.eval`;
  2. then `pc.insert()`: it is not kept when it alone exceeds the headroom (beside the running
     prefill's parent). Otherwise other entries are evicted in `victim` order down to
     `QWEN_PROMPT_CACHE_GB` and the headroom, with `wanted` last and the running prefill's
     parent spared.
  3. `pc.version` bumps, so every peek memo (`req.peeked`) is recomputed. The head's `_fits`
     keeps the entry (≥ a slab deep), and `_wanted` protects it.
  4. **Wanted order with the tier** (GPU gate fix): `victim` evicts the wanted entries in
     reverse queue order (`_wanted` lists them in queue order), the one whose request comes
     last first. Before, it was MRU among wanted ones. A just-inserted entry has `used` = now,
     so it was the first wanted victim at the next store ahead of its request, and 21 of 57
     loads in the first "on" run went for nothing. The order without the tier is unchanged.

  The lookup in `_prompt_cache` is unchanged: it finds the entry as any RAM entry. A load whose
  request was admitted or cancelled meanwhile is dropped, at the next tick or at the idle pass.
- **Hold.** While the head's load runs (≤ `WAIT_S`), `tick` does not admit, so FIFO order is
  kept. Decode rows keep stepping. With nothing to do, the scheduler thread waits 5 ms on the
  load per tick instead of spinning.
- **Failures.** If the entry left the disk between the peek and the load (the writer's LRU),
  `load()` returns None and the scheduler peeks again. A failed read (a torn file, EIO) is a
  miss: logged, the request is prefilled as without the tier.
- **A stuck read** (final review #4). `check_loads()` runs at every flush and in `_disk_room`'s
  wait. It abandons a load running past `TIMEOUT_S`: the load is cancelled and marked taken,
  its hold and the entry's pin go at once, and its request goes without it. Only the accounting
  and the pin are released. The numpy and MLX buffers the loader thread is filling stay its
  own, since freeing them under it would be a use-after-free. `busy()` stays true while the
  abandoned load is still inside the loader (`_hung`), because a new load would only queue
  behind it on the single loader thread. When the read returns, the loader stops at its next
  array, and the scheduler drops the late result without releasing anything twice.

### Validity

`Engine._disk_key()` is a sha256 over:

- the model id; for each weights file its name, size and a sha256: of the header of a
  `*.safetensors` (the 8-byte length + the JSON of names, dtypes, shapes and offsets; the 10 GB
  of data are not read), of the whole of a `*.json`;
- the resolved numerics settings: `KV_KWARGS`, `FP16`, the slab policy (`PREFILL_STEP`,
  `PREFILL_MIN_STEP`, `PREFILL_BUDGET`, `_FIXED_SLAB`), `SDPA_FUSED`, `GDN_FAST`, `GDN_FUSE`,
  `max_batch` (mrqmm install, which matters for reply KV), and `PROMPT_CACHE_REPLY`. Reply
  entries hold decode-kernel KV, and the `QWEN_PROMPT_CACHE*` env prefix is otherwise excluded
  (review #4);
- every `QWEN_*` env var except the sizing and prompt-cache ones;
- the mlx and mlx-vlm versions;
- the sha256 of `worker.py`, `pfqmm.py`, `mrqmm.py`, `cascade.py` and `pcdisk.py`;
- the format version.

Clearing at start is the default. `KEEP=1` reuses files only under the same key. Being that
conservative is deliberate: any code change invalidates.

With `KEEP=1` every file is also `fsync`ed before its rename. Otherwise a power loss could leave
a full-size file of zeros under its name, which the size check would not catch. Without `KEEP`
the directory is cleared at start anyway, so the writes skip the `fsync`.

### Errors

Disk errors are never a request's failure:

| failure | effect |
|---|---|
| write failed | the entry is lost, as before the tier; its `.tmp` files are removed |
| read or validation failed | a miss |
| flush / insert / tick exception | logged; the queue goes on; a load whose insert raised is dropped (`_disk_take`: unpinned, hold released, so `busy()` cannot stay true) |

The writer's and loader's loops never die. A later entry that borrows a blob whose write failed
fails its own check and drops itself.

### Metrics

There are no proto changes: Health stays as is. Adding `prompt_cache_disk_*` fields 13–17 would
mean a gateway change as well; that is left to the integrator.

- **Log lines**, parsed by `scripts/perf/pcdisk/disk_log.py`:
  - `prompt cache disk: wrote an entry of N tokens (X MB new, k blobs shared) in S s; n entries, G GiB on disk`
  - `id=… prompt cache disk: loading an entry of N tokens (X MB to read) for a D-token hit (R in RAM)`
  - `id=… prompt cache disk: entry of N tokens loaded (X MB read in T ms; …)`
  - `… not written: <why>`, `… failed`, `… dropped: its request no longer waits`, `… not kept`
- **Counters.** `DiskTier.stats()` returns entries, bytes, writes, write_bytes, write_s, skipped,
  errors, deleted, loads, load_bytes, hits, discards, read_ms, pending and held.

## Invariants (self-review)

- **MLX on the scheduler thread only**, apart from constant-array allocation and free, as the
  image processor already does on handler threads:
  - views, `eval`, bf16 views, `_Entry` / `_Seg` construction and insert run on the scheduler;
  - the writer reads raw memory;
  - the loader calls `mx.array(numpy)`.
- **Memory accounting.**
  - Evicted entries not yet written and loads not yet inserted are `held` and charged with the
    cache in every budget check (`pc.charged()`). Evictions that need the RAM now free it at
    once (`spill=False`).
  - **Loaded entries** are inserted through the same headroom as a store. A load starts only when
    its non-live bytes fit beside what is already held, by budget and by live memory.
  - **Pending writes** are bounded by `QUEUE_GB` with no exception, counted on the evicted
    entries' own bytes. A fresh segment shared with a RAM entry is held on top of that (final
    review #2). Their MLX buffers count in active memory until written.
- **Bits.**
  - **The bytes are exact.** fp16, fp32, packed uint32 and bf16 (as uint16) are written and read
    back as raw bytes. The tests compare every restore position of the loaded entry with the
    evicted one, byte by byte.
  - **A disk hit equals a RAM hit.** The tests check fp32, fp16 (8-bit model) and quantized KV
    (kv8, `kv_start` 24) against an all-RAM run: the same answers and the same resume point.
    `check_cold` on the resulting entries confirms that the restored bits equal a cold prefill.
- **Off is unchanged.** `QWEN_PROMPT_CACHE_DISK` unset means `pc.disk` is None. `tick` then
  makes one attribute read and one `is not None` test, `evict` / `_idle` the same, and nothing
  else changes. A test also runs with the tier on but `min_tokens` huge: the answers, resume
  points, entries and eviction counts equal those with it off.

## Tests (`tests/test_pcdisk.py`, CPU, ~35 s)

| test | what it checks |
|---|---|
| `test_disk_hit_equals_ram_hit_fp32` / `_fp16` / `_quantized` | a disk hit gives the same answers and resume point as an all-RAM run, `check_cold` passes, and an off run MISSes |
| `test_loaded_entry_bits` | metadata and every restore position byte-equal after write → load |
| `test_best_matches_entry_best` | `DiskEntry.best` agrees with `_Entry.best` |
| `test_shared_blobs_written_once_and_reused_live` | dedup, live reuse, no rewrite of an unchanged entry |
| `test_next_turn_on_a_loaded_entry` | the chain through the disk: borrowed blobs, the covered entry dropped, the next turn served |
| `test_lru_limit` | the limit, LRU order, no orphan files, bytes equal file sizes, an over-limit entry not written |
| `test_write_errors_never_fail_a_request` | ENOSPC: nothing on disk, nothing pending |
| `test_load_errors_are_misses` | a torn file and EIO both fall back to a miss |
| `test_off_changes_nothing` | off and on-without-writes give identical results |
| `test_keep_across_runs` | the index is rebuilt and loads; another key or KEEP off clears it; a second worker on the directory is refused (flock); `close()` |
| `test_blob_roundtrip` | dtypes, strided arrays, empty arrays, a truncated file |
| `test_insert_respects_the_headroom` | `pc.insert` respects the headroom |
| `test_disk_key` | the key is stable and moves with weights / numerics knobs, not with sizing knobs |
| `test_reply_entry_through_disk` | a reply entry goes through the disk and the next turn resumes after the reply, as with RAM |
| `test_load_while_rows_decode` | the head waits for its load while a row decodes |
| `test_wait_bound_and_abandoned_load` | the `WAIT_S` bound, and the idle pass frees an abandoned load |
| `test_lru_never_drops_a_blob_the_new_entry_shares` | review #1: an LRU entry that alone holds the shared segment is not dropped for nothing; the index equals the files; later entries are written and load |
| `test_late_unlink_never_deletes_a_rewritten_blob` | review #5: a re-offer after the writer's drop keeps its file and loads |
| `test_eviction_ram_is_charged_until_written` | review #2: `charged()` counts an evicted entry until its write ends; past the queue cap it is freed at once |
| `test_headroom_evictions_free_at_once` | review #2: the `limit` loops of `insert` / `store` free at once, the soft cap spills only within the headroom |
| `test_admission_waits_for_the_writer` | review #2: `_fits` admits only within the budget, the writer's hold included |
| `test_load_is_charged_until_inserted` | review #2: a load is held from its start until inserted |
| `test_load_reads_one_array_at_a_time` | review #3: the numpy peak is under 2 arrays, not the blob |
| `test_insert_failure_drops_the_load` | review #6: a raising insert drops the load; `busy()` false, nothing held |
| `test_growth_waits_for_the_tier_instead_of_cutting` | re-review #1: no cut while only the hold is in the way; past `HELD_WAIT_S`, cut as before |
| `test_cancel_loads_frees_their_hold` | re-review #1: a cancelled load yields nothing; hold released, `busy()` false |
| `test_admission_keeps_its_hit_while_the_writer_holds_ram` | re-review #2: the head waits for the hold, keeps its hit, `keep_since` untouched |
| `test_stuck_write_drops_the_queue` | re-review #3: queued writes dropped, nothing taken while stalled, resumes after |
| `test_eviction_holds_only_what_the_writer_keeps` | re-review #4: a blob already on disk is not held |
| `test_growth_frees_this_ticks_unwritten_evictions` | final review #1: a spill earlier in the same tick is taken back; the growth neither waits nor cuts, the same as with the tier off |
| `test_revoke_takes_back_unflushed_offers` | final review #1: `revoke(need)` releases the latest offers until `need` is covered; a revoked entry is never written |
| `test_shared_fresh_segment_stays_held_while_written` | final review #2: a fresh segment shared with another entry stays held until written |
| `test_close_with_unflushed_offer` | final review #3: `close()` with an unflushed offer releases its hold (was a ValueError) |
| `test_growth_does_not_wait_for_a_stuck_write` | final review #4: `_regrow` does not wait `HELD_WAIT_S` for a stuck write's hold |
| `test_admission_does_not_wait_for_a_stuck_write` | final review #4/#5: `_fits` does not wait for a stuck write's hold |
| `test_stuck_load_is_abandoned` | final review #4: a load past `TIMEOUT_S` loses its hold and pin; `busy()` holds until the loader returns; nothing is released twice |
| `test_state_identity_holds_every_array_weakly` | final review #6: a state's uid record goes when any of its arrays dies |
| `test_env_parsed_only_with_the_tier_on` | final review #6: a bad knob value never breaks the import; with the tier on it is recorded |

The test file takes ~25 s (41 tests).

Full suite (`pytest -q --ignore tests/test_mrqmm.py`, 2026-10-24): 359 passed, 49 skipped,
9.5 min before the review fixes; after them (ed3b242) 367 passed; after the re-review fixes 372 passed, 49 skipped, 8.4 min; after the final-review fixes 381 passed, 49 skipped, 7.2 min. A light Metal correctness check, `io_bench.py --sizes-gb 0.4 --rounds 1` on the GPU
device, came back bit-exact (zero-copy views and loader-thread `mx.array` on Metal); it was not
a speed measurement.

## GPU plan

One command, run with the GPU lease held and nothing else on the GPU:

```zsh
setopt NO_BG_NICE
PERF_OUT=/tmp/perf scripts/perf/pcdisk/run_all.sh 10 8 3     # ~65 min
```

It writes `$PERF_OUT/pcdisk/summary.md` and copies the raw files to
`docs/perf/2026-10-24/raw/pcdisk/`. The worker comes from this worktree's `src/worker`, run with
the main tree's venv python. Never `uv`.

### 1. I/O microbench (`io_bench.py`, ~3 min, no model)

- **Setup.** Entries shaped like the 9B's: 8 FA layers × fp16 K/V (1, 4, T, 256) in two segments,
  plus 5 states of 24 GDN layers (~50 MiB each). Sizes 1 and 2.5 GB, 3 interleaved rounds,
  medians.
- **Measured.**
  - `flush_ms`: the scheduler's stall per write;
  - writer GB/s;
  - loader `read_ms` and GB/s;
  - `finish_ms`: the scheduler's stall per load;
  - the MLX peak;
  - a bit-exact check against arrays regenerated from their seeds.
- **Peak memory** ≈ the entry + one array: ≤ 3 GB.
- **Expected:**
  - flush < 5 ms;
  - write 1.5–3 GB/s, i.e. 0.4–0.7 s / 1 GB and 1–1.7 s / 2.5 GB;
  - read with `F_NOCACHE` 2–5 GB/s including the memcpy into MLX, i.e. 0.3–0.6 s / 1 GB and
    0.6–1.3 s / 2.5 GB;
  - finish < 20 ms.
- **Pass:** bit-exact in every round; flush_ms < 20 and finish_ms < 50 (the scheduler stalls
  less than one decode step); a 2.5 GB read < 2 s.

### 2. Exactness on the real 9B (`run_exact.sh 3`, ~8 min)

- **Worker.** `QWEN_MAX_BATCH=1`, `QWEN_PROMPT_CACHE_REPLY=0`, `QWEN_PROMPT_CACHE_GB=0.3` (each
  store keeps only the newest), disk on.
- **Pairs.** 3 pairs of AppWorld last calls (6–30K tokens). Per pair:
  1. prime A, evict it with B;
  2. teacher-forced `prompt_logprobs` of A with `prompt_cache: "read"` (a disk hit);
  3. the same with `prompt_cache: false` (cold), and compare: the scored values after the resume
     point must be `==`;
  4. evict again, then 64 greedy tokens of A + "Continue." as a hit and as cold: the content must
     be identical.

  The worker log must show exactly one disk load per hit.
- **Pass:** 3/3 pairs. Peak ~10–12 GB.

### 3. AppWorld replay, disk off vs on (`run_replay_disk.sh` × 4, ~50 min)

- **Load.** `replay.py --own-replies --ignore-eos off --episodes 8 --duration-min 10`, through
  :8090. The gateway is started if it is not running (thinking off).
- **Worker env.** The 17.10 knobs with a small cache, so evictions are frequent:

  ```
  QWEN_KV_BITS=0 QWEN_PROMPT_CACHE=1 QWEN_PROMPT_CACHE_GB=3 QWEN_BATCH_RESERVE_TOKENS=1024
  QWEN_PRESENCE_PENALTY=0 QWEN_PROMPT_CACHE_REPLY=1 [QWEN_PROMPT_CACHE_DISK=$PERF_OUT/pcdisk/disk_<run>]
  ```

- **Order.** `off_a → on_a → off_b → on_b`. Interleaving controls for drift: each pair is compared
  on its own, and both must agree.
- **Per run:**
  - memsampler (swap, CMPRS, pageins);
  - `sched_timer` (prefill / decode split);
  - the warm 8K probe after the replay (drift control);
  - `disk_log.py`: MISS / RAM-hit / disk-hit shares, TTFT and arrival → first token by class,
    MISS ≥ 10K, and the writer's and loader's MB, seconds and failures.
  - `compare.py` puts the four runs side by side.
- **Expected:**
  - **off:** many MISSes (8 histories vs ~3–4 entries of 0.65–1 GB).
  - **on:** MISSes drop to cold starts of new tasks. Disk-hit TTFT ≈ RAM-hit TTFT + 0.3–1 s
    (load). req/min +15–40 %. Peak and swap within ±1 GB of off: the tier holds nothing in RAM
    beyond the write queue.
- **Pass:**
  1. req/min(on) > req/min(off) in both pairs, with a mean gain ≥ +10 %;
  2. MISS share (on) < (off) in both pairs;
  3. no write or load failures, no request errors;
  4. worker peak (on) ≤ peak (off) + 1 GB, swap max (on) ≤ (off) + 1 GB;
  5. load p50 < 1.5 s;
  6. warm probes within ±15 % of each other (else repeat the pair that disagrees).
- **On fail:**
  - load latency too high → raise the prefetch depth (loads start for all `max_batch` waiting
    requests already), or check `NOCACHE=0`;
  - memory up → lower `QUEUE_GB`;
  - few disk hits → check `MIN_TOKENS` and the log's "not written" reasons.

### Risks to watch

- Write bandwidth competes with nothing on the GPU, but the SSD is shared with swap. Watch
  pageins and swapins against off.
- `_fits` charges an evicted entry until its write ends (`held`), so a request whose room went to
  disk waits for the writer, ≤ ~1–2 s per GB. Look for `queue_ms` growth in "on" at equal
  req/min. With `QUEUE_GB` at 4 the budget can shrink by up to 4 GB for that time.
- The `limit` loops of `store` / `insert` and `_regrow` free at once (`spill=False`), so under
  memory pressure entries are deleted, not written, exactly as without the tier. Count the "not
  written" log lines against the writes.
- Disk space: ≤ `QWEN_PROMPT_CACHE_DISK_GB` (32), deleted at the end of each run. 179 GB free on
  2026-10-24.

## GPU results (2026-10-24)

M1 Max 32 GB, Ornith-1.5 9B 8-bit, the worker of this worktree, the main tree's venv. Raw files:
`../raw/pcdisk/`. Summary: `../raw/pcdisk/summary.md`. The cache files went to the session's
scratchpad (`PCDISK_DIR`, new in the scripts) and were deleted after each run. The gateway on
:8090 was already running with the scripts' settings and was reused.

**Verdict: PASS on 5 of 6 replay criteria plus io_bench and exactness. Criterion 4 (peak) fails
as written, but the extra peak comes from the load, not the tier.** That took one worker fix
(below). The first "on" run (`on_a_v1`) used the code before the fix: it got +25 % req/min, and
21 of its 57 loads went for nothing.

### 1. io_bench: PASS

Medians of 3 interleaved rounds. `F_NOCACHE` on.

| entry | flush ms | write | read | finish ms | bit-exact |
|---|---|---|---|---|---|
| 1.0 GB | 0.9 | 0.16 s (6.1 GB/s) | 411 ms (2.4 GB/s) | 0.4 | 3/3 |
| 2.5 GB | 1.7 | 0.41 s (6.1 GB/s) | 683 ms (3.7 GB/s) | 0.7 | 3/3 |

- Every number beats the bounds: flush < 20 ms, finish < 50 ms, a 2.5 GB read < 2 s.
- The SSD writes about twice as fast as expected.
- `peak_gb` (1.95 / 5.79) is the benchmark's own entry build: the fp32 `normal` → fp16 cast. A
  separate probe shows the load adds exactly the entry (0.998 GB for 1 GB) and the tier holds
  nothing after the write.
- Script fix: `io_bench` called `_unlink`, which the review had renamed to `_unlink_locked`.

### 2. Exactness on the real 9B: PASS (3/3, before and after the fix)

| pair | prompt | resumed at | tf `==` cold | 64 greedy tokens `==` | disk loads | tf hit / cold s | gen hit / cold s |
|---|---|---|---|---|---|---|---|
| 0 | 15622 | 15401 | yes | yes | 1 + 1 | 1.3 / 35.9 | 2.6 / 31.2 |
| 1 | 67947 | 67940 | yes | yes | 1 + 1 | 1.3 / 198.7 | 2.8 / 169.0 |
| 2 | 25668 | 25661 | yes | yes | 1 + 1 | 0.7 / 58.8 | 2.0 / 51.5 |

- Writes took 0.15–0.45 s. Loads took 258–754 ms for 0.69–2.5 GB.
- Peak was 18.9 GB, from the cold 68K prefill. The plan expected 6–30K prompts, but pair 1's
  last call is 68K.
- The scored teacher-forced positions start right after the resume point, so the `==` covers
  only the last 7–221 tokens of each prompt. The 64-token generation covers the rest.
- Rerun on the fixed code (commit `c09ef8d`): again 3/3, the same resume points, one disk load per hit, peak 18.9 GB (`exact.txt`; the first run is `exact_v1.txt`).

### 3. AppWorld replay: req/min ×2.3

Settings: `QWEN_PROMPT_CACHE_GB=3`, 8 episodes × 10 min, order
`off_a, on_a_v1 | off_b, on_b, off_c, on_c`.

- `on_a_v1` ran before the fix. The worker was then fixed (commit `c09ef8d`), and the pairs
  `(off_b, on_b)` and `(off_c, on_c)` ran interleaved on the fixed code.
- The off path is unchanged by the fix: the new victim order applies only with the tier.

| | off_a | on_a_v1 | off_b | **on_b** | off_c | **on_c** |
|---|---|---|---|---|---|---|
| requests/min | 10.3 | 12.9 | 10.3 | **23.4** | 10.4 | **24.5** |
| MISS share (`disk_log`) | 38.1 % | 23.0 % | 34.5 % | **7.0 %** | 36.8 % | **7.1 %** |
| disk hits (share) | – | 36 (25.9 %) | – | 137 (56.1 %) | – | 148 (58.0 %) |
| loaded, then resumed from nothing | – | 21 | – | 3 | – | 3 |
| reloads (`load 2/3`) | – | – | – | 20 | – | 19 |
| MISS TTFT p50 / p90 s | 14.5 / 22.5 | 21.9 / 26.0 | 15.8 / 26.3 | 9.1 / 36.4 | 16.6 / 22.3 | 10.8 / 34.3 |
| disk-hit TTFT p50 / p90 s | – | 1.5 / 6.5 | – | 1.5 / 4.7 | – | 1.5 / 5.1 |
| RAM-hit TTFT p50 / p90 s | 1.2 / 4.5 | 0.7 / 4.5 | 1.2 / 4.6 | 0.8 / 5.2 | 1.1 / 4.5 | 1.1 / 4.6 |
| queue mean / p90 s | 24.9 / 55.6 | 22.1 / 85.1 | 25.2 / 60.5 | 7.3 / 20.0 | 26.5 / 62.9 | 6.9 / 20.2 |
| client latency p50 / p90 s | 38.4 / 85.4 | 15.5 / 127.1 | 35.8 / 100.4 | 15.0 / 41.0 | 36.8 / 92.9 | 14.1 / 34.9 |
| mean reused tokens | 3927 | 5285 | 4145 | 9668 | 4120 | 9580 |
| sched prefill / decode % | 87 / 9 | 86 / 10 | 87 / 9 | 78 / 16 | 87 / 9 | 79 / 16 |
| mean rows | 2.52 | 2.38 | 2.48 | 4.16 | 2.60 | 4.32 |
| writes: n, GB, writer s (p50 per entry) | – | 63, 26.6, 6.8 (0.09) | – | 148, 25.5, 7.3 (0.04) | – | 161, 27.4, 7.9 (0.04) |
| loads: n, GB, p50 / max ms | – | 57, 34.2, 281 / 684 | – | 140, 92.2, 322 / 749 | – | 151, 97.6, 319 / 593 |
| on disk max | – | 10 entries, 6.7 GiB | – | 13, 7.8 GiB | – | 12, 8.1 GiB |
| worker peak GB | 16.75 | 17.37 | 16.10 | 19.43 | 16.31 | 19.74 |
| swap used GB max | 3.7 | 3.7 | 3.7 | 3.6 | 3.6 | 3.6 |
| worker CMPRS GB max | 0 | 0 | 0 | 0.34 | 0 | 0.39 |
| warm 8K probe TTFT s | 14.4 | 15.4 | 15.3 | 15.6 | 15.4 | 15.6 |

Criteria, pairs b and c:

1. **req/min, on > off in both pairs, mean ≥ +10 %: PASS.** +127 % and +136 %. With few MISSes,
   prefill drops from 87 % to 78 % of the scheduler's time, twice as many rows decode together,
   and the queue wait falls from ~25 s to ~7 s.
2. **MISS share, on < off in both pairs: PASS.** 34.5 → 7.0 % and 36.8 → 7.1 %. The MISSes left
   are mostly the first calls of new tasks: of 15–16 replay MISSes per run, 8 are turn 1, plus
   the probe's 2 requests. The rest include the 3 wasted loads (below).
3. **No write or load failures, no request errors: PASS.** 0 failed writes, 0 failed loads,
   0 "not written", 0 dropped, 0 "not kept", 0 request errors.
4. **Peak ≤ off + 1 GB: FAIL as written (+3.3 / +3.4 GB). Swap ≤ off + 1 GB: PASS** (3.6 vs
   3.7 GB, flat).
   - The extra peak is the rows' KV. Over every `batch:` line, active − reserved − pcache (the
     weights plus whatever the tier holds plus transients) peaks at 9.52–9.54 GiB with the tier
     and 9.17 GiB without it, i.e. the tier adds ≤ 0.4 GB.
   - `reserved` peaks at 5.1–5.4 GiB with the tier and 2.2–2.6 GiB without it. At twice the
     throughput the episodes get twice as far in 10 minutes: disk hits average 12.4–12.8K
     tokens, the off prompts ~7K. Mean rows rise too (4.2–4.3 vs 2.5).
   - The cost is real for a 32 GB machine all the same. The compressor reached 1.9 GB (off:
     1.4), and the worker had 0.34–0.39 GB compressed (off: 0).
5. **Load p50 < 1.5 s: PASS.** 322 / 319 ms. The max is 749 ms.
6. **Warm probes within ±15 %: PASS.** 8K TTFT 14.4–15.6 s across all six runs (±4 %).

**The fix the gate found** (commit `c09ef8d`; tests `test_victim_wanted_in_queue_order_with_the_tier`,
`test_no_load_that_the_requests_ahead_would_evict`,
`test_reload_when_the_loaded_entry_left_before_its_turn`):

- **The bug.** In `on_a_v1`, 21 of 57 loads were wasted. The entry was loaded and inserted for
  a request 3rd to 6th in line, then evicted at the next store ahead of it, so the request
  prefilled cold anyway.
- **Why.** `pc.insert` sets `used` = now. When every entry is wanted (8 episodes vs ~4 entries
  in 3 GB), `victim` evicted the MRU wanted entry, which was the one just loaded.
- **The fix** has three parts:
  1. with the tier, wanted entries go in reverse queue order;
  2. a request that is not first loads only when it fits under the soft cap beside the
     entries of the requests ahead;
  3. a request can reload, up to 3 loads, when its kept entry left RAM before its turn.
- **After the fix**, 3 wasted loads per run remain. Each is a request that used all 3 loads,
  with loads 2 and 3 0.25–0.5 s apart. Two neighbours in the queue (e.g. `e1-28` / `e5-28`)
  loaded and evicted each other in turn. An open refinement: don't let a reload for a request
  further back evict the entry of a request ahead of it.
- The loads went from 57 to ~145 per run (92–98 GB read in 10 min, ~150 MB/s on average). The
  SSD has room for that.

Also fixed in the scripts:

- `disk_log.py` counted a request that had an entry loaded but resumed from nothing as a disk
  hit. Such a request is now a MISS and is counted in "loaded for but resumed from nothing". The
  plan's run_all summary for `on_a` (before both fixes) would have read MISS 7.9 %; the true
  value was 23 %.
- Script bugs fixed on the way: `io_bench` (`_unlink_locked`), the `PCDISK_DIR` override for the
  cache files.

**Recommendation.** Turn the tier on for AppWorld-type agentic load with
`QWEN_PROMPT_CACHE_GB=3`, as measured (`QWEN_PROMPT_CACHE_DISK=<dir on the internal SSD>`, other
disk knobs at their defaults).

- **Why 3 GB.** A small RAM cache plus the disk beats a big RAM cache. The 3 GB leave room for
  the extra rows that the higher throughput brings: peak 19.7 GB against a 28 GB cap.
- **6 GB** (the 17.10 setting) is not measured with the tier. It would add ~3 GB of RAM to a
  peak that has already grown with the rows, for at most a few % fewer loads.
- **Before 8+ processes or long-tail histories (27–58K)**, rerun the replay at that load and
  watch CMPRS and swap. Each such entry is 1.5–2.5 GB, so the write queue cap (`QUEUE_GB`=4)
  and the head's wait for the writer matter there. This gate did not exercise them: entries
  were 0.4–1.1 GB.

## Left open

- Shutdown / SIGTERM flush of the RAM cache (only for `KEEP=1`; needs `stop()` to wait for a
  drain).
- Health proto fields (`prompt_cache_disk_entries`, `_bytes`, `_writes`, `_loads`, `_read_ms`)
  plus the gateway metrics.
- Writing "not kept" entries: it would need the row's KV copied while the batch has no room. A
  zero-copy view of the row's live KV would keep decode from writing in place.
- A disk TTL (DeepInfer: 3 days). Today only the LRU and the size limit apply.
- `KEEP=1` across code changes: the key hashes the sources, so any edit clears the directory. A
  finer key (numerics-relevant code only) is possible, but riskier.
